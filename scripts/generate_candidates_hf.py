"""
Step 1: Generate Candidate Responses (HuggingFace Version for low VRAM)

Uses HuggingFace transformers with bitsandbytes 4-bit quantization.
Designed to fit in 4GB VRAM by splitting large candidate requests.

Usage:
    python scripts/generate_candidates_hf.py \
        --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct" \
        --dataset math500 \
        --n-candidates 512 \
        --max-examples 0
"""

import json
import sys
import time
import math
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logger import ExperimentLogger
from src.core.data import (
    load_dataset_examples,
    dataset_system_message as _dataset_system_message,
)

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", help="Generator model."),
    dataset: str = typer.Option("math500", help="Dataset: gsm8k, math500, or alpaca_eval."),
    n_candidates: int = typer.Option(512, help="Number of candidates per example."),
    max_examples: int = typer.Option(0, help="Max examples. 0 = full dataset."),
    temperature: float = typer.Option(0.7, help="Generation temperature."),
    top_p: float = typer.Option(0.95, help="Generation top_p."),
    max_new_tokens: int = typer.Option(2048, help="Max tokens per generation."),
    batch_size: int = typer.Option(16, help="How many candidates to generate at once to fit in VRAM."),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path."),
):
    """Generate N candidates using HuggingFace with 4-bit quantization."""
    
    dataset = dataset.lower()
    resolved_max = None if max_examples <= 0 else max_examples

    if output_path is None:
        scope = "full" if resolved_max is None else f"{resolved_max}ex"
        output_path = f"data/candidates/{dataset}_n{n_candidates}_{scope}_hf.jsonl"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_indices = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                completed_indices.add(row["example_index"])
        print(f"Resuming: {len(completed_indices)} examples already completed.")

    # Logger
    exp_logger = ExperimentLogger(log_dir="logs", experiment_name=f"hf_gen_{dataset}_n{n_candidates}")
    exp_logger.log_environment()

    # Load Dataset
    t0 = time.time()
    exp_logger.log_phase_start("dataset_loading", dataset)
    all_examples = load_dataset_examples(dataset, "test", resolved_max)
    exp_logger.log_phase_end("dataset_loading", time.time() - t0)

    # Filter examples we still need to process
    examples_to_run = []
    for i, ex in enumerate(all_examples):
        if i not in completed_indices:
            ex["_global_index"] = i
            examples_to_run.append(ex)

    total_to_do = len(examples_to_run)
    if total_to_do == 0:
        print(f"All {len(all_examples)} examples already completed. Nothing to do.")
        return

    print(f"\n{'=' * 70}")
    print(f"  HuggingFace 4-Bit CANDIDATE GENERATION (Step 1 of 3)")
    print(f"  Model:       {model_name}")
    print(f"  Dataset:     {dataset} ({len(all_examples)} examples)")
    print(f"  Candidates:  {n_candidates} per example")
    print(f"  Remaining:   {total_to_do}")
    print(f"  Batch Size:  {batch_size} (candidates per forward pass)")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 70}\n")

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model with 4-bit quantization to fit in 4GB VRAM
    print(f"Loading model {model_name} in 4-bit...")
    t0 = time.time()
    exp_logger.log_phase_start("model_loading", f"HF 4-bit: {model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    exp_logger.log_phase_end("model_loading", time.time() - t0)

    system_msg = _dataset_system_message(dataset)
    
    print("\nStarting generation...")
    exp_logger.log_inference_start(total_to_do, n_candidates)
    
    t_gen_start = time.perf_counter()
    
    for ex in tqdm(examples_to_run, desc="Generating Candidates"):
        idx = ex["_global_index"]
        prompt = ex["question"]
        gold_answer = ex.get("answer") or ex.get("reference", "")
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = f"Question: {prompt}\nAnswer: Let me solve this step by step.\n"
        
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        
        candidates = []
        
        # Split n_candidates into batches to prevent OOM
        num_batches = math.ceil(n_candidates / batch_size)
        
        for _ in range(num_batches):
            current_batch_size = min(batch_size, n_candidates - len(candidates))
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=current_batch_size,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Slice off the input tokens
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            
            batch_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            candidates.extend([text.strip() for text in batch_texts])
        
        # Calculate approximate time per example
        avg_gen_time = (time.perf_counter() - t_gen_start) / (idx + 1)
        
        # Write to JSONL
        with out_path.open("a", encoding="utf-8") as f_out:
            row = {
                "example_index": idx,
                "prompt": prompt,
                "ground_truth": gold_answer,
                "candidates": candidates,
                "n_candidates": n_candidates,
                "gen_time_sec": avg_gen_time,
                "model_name": model_name,
                "dataset": dataset,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            exp_logger.log_example_done(idx, [{"method": "hf_generation", "is_correct": None}])

    total_gen_time = time.perf_counter() - t_gen_start
    exp_logger.log_inference_end(total_to_do)
    
    print(f"\n✓ Generated {total_to_do} examples with {n_candidates} candidates each in {total_gen_time:.1f}s")
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    app()
