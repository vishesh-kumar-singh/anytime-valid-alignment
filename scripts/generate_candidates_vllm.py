"""
Step 1: Generate Candidate Responses (Fast vLLM Version)

Uses vLLM for continuous batching and PagedAttention.
This is 10x-30x faster than standard HuggingFace generation.

Usage:
    python scripts/generate_candidates_vllm.py \
        --model-name "Qwen/Qwen2.5-3B-Instruct" \
        --dataset gsm8k \
        --n-candidates 24 \
        --max-examples 200
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logger import ExperimentLogger
from src.core.data import (
    load_dataset_examples,
    dataset_system_message as _dataset_system_message,
)

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", help="Generator model."),
    dataset: str = typer.Option("gsm8k", help="Dataset: gsm8k, math500, or alpaca_eval."),
    n_candidates: int = typer.Option(24, help="Number of candidates per example."),
    max_examples: int = typer.Option(200, help="Max examples. 0 = full dataset."),
    temperature: float = typer.Option(0.7, help="Generation temperature."),
    top_p: float = typer.Option(0.95, help="Generation top_p."),
    max_new_tokens: int = typer.Option(512, help="Max tokens per generation."),
    gpu_memory_utilization: float = typer.Option(0.85, help="vLLM memory utilization."),
    num_gpus: int = typer.Option(1, help="Number of GPUs to use (tensor_parallel_size)."),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path."),
):
    """Generate N candidates using ultra-fast vLLM."""
    
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed. Please install it first:")
        print("pip install vllm")
        raise typer.Exit(1)

    dataset = dataset.lower()
    resolved_max = None if max_examples <= 0 else max_examples

    if output_path is None:
        scope = "full" if resolved_max is None else f"{resolved_max}ex"
        output_path = f"data/candidates/{dataset}_n{n_candidates}_{scope}_vllm.jsonl"

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
    exp_logger = ExperimentLogger(log_dir="logs", experiment_name=f"vllm_gen_{dataset}_n{n_candidates}")
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
    print(f"  vLLM CANDIDATE GENERATION (Step 1 of 3)")
    print(f"  Model:       {model_name}")
    print(f"  Dataset:     {dataset} ({len(all_examples)} examples)")
    print(f"  Candidates:  {n_candidates} per example")
    print(f"  Remaining:   {total_to_do}")
    print(f"  GPUs:        {num_gpus}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 70}\n")

    # Load Tokenizer & Format Prompts
    print("Formatting prompts with chat template...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    system_msg = _dataset_system_message(dataset)
    
    formatted_prompts = []
    for ex in examples_to_run:
        prompt = ex["question"]
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = f"Question: {prompt}\nAnswer: Let me solve this step by step.\n"
        formatted_prompts.append(formatted)

    # Load vLLM
    t0 = time.time()
    exp_logger.log_phase_start("model_loading", f"vLLM: {model_name}")
    print(f"\nLoading vLLM engine: {model_name}...")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True, enforce_eager=True,
    )
    exp_logger.log_phase_end("model_loading", time.time() - t0)

    # Configure Sampling
    sampling_params = SamplingParams(
        n=n_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=42,
    )

    # Generate in chunks so we save progress and don't lose everything on a crash!
    CHUNK_SIZE = 500
    print(f"\nStarting batched generation in chunks of {CHUNK_SIZE} examples...")
    exp_logger.log_inference_start(total_to_do, n_candidates)
    
    t_gen_start = time.perf_counter()
    
    for chunk_start in range(0, total_to_do, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_to_do)
        chunk_examples = examples_to_run[chunk_start:chunk_end]
        chunk_prompts = formatted_prompts[chunk_start:chunk_end]
        
        print(f"\nProcessing chunk {chunk_start//CHUNK_SIZE + 1} ({chunk_start} to {chunk_end} of {total_to_do})...")
        
        chunk_t0 = time.perf_counter()
        outputs = llm.generate(chunk_prompts, sampling_params)
        chunk_gen_time = time.perf_counter() - chunk_t0
        avg_gen_time = chunk_gen_time / len(chunk_examples) if len(chunk_examples) > 0 else 0
        
        # Save chunk to disk immediately
        with out_path.open("a", encoding="utf-8") as f_out:
            for ex, vllm_out in zip(chunk_examples, outputs):
                idx = ex["_global_index"]
                prompt = ex["question"]
                gold_answer = ex.get("answer") or ex.get("reference", "")
                
                candidates = [out.text.strip() for out in vllm_out.outputs]
                
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
                exp_logger.log_example_done(idx, [{"method": "vllm_generation", "is_correct": None}])
        
        print(f"✓ Saved chunk to {output_path}")

    total_gen_time = time.perf_counter() - t_gen_start
    exp_logger.log_inference_end(total_to_do)
    print(f"\n✓ Generated {total_to_do} examples with {n_candidates} candidates each in {total_gen_time:.1f}s")
    print(f"✓ Saved to: {output_path}")
    print(f"  Next step: python scripts/score_candidates.py --candidates-path {output_path} --reward-model <model_name>")


if __name__ == "__main__":
    app()
