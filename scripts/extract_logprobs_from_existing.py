import json
import sys
import time
import math
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils.math500 import extract_math_answer
from src.data_utils.gsm8k import extract_gsm8k_answer
from src.data_utils.mmlu import extract_mmlu_answer
from src.pipelines.base import _extract_answer

def extract_cruxeval_answer(text: str) -> Optional[str]:
    return _extract_answer(text, "cruxeval")


app = typer.Typer()

def get_extracted_answer(dataset: str, text: str) -> Optional[str]:
    if dataset == "math500":
        return extract_math_answer(text)
    elif dataset == "gsm8k":
        return extract_gsm8k_answer(text)
    elif dataset == "mmlu":
        return extract_mmlu_answer(text)
    elif dataset == "cruxeval":
        return extract_cruxeval_answer(text)
    return None

def compute_wcite_weight(full_text: str, token_ids: list, prompt_logprobs: list, tokenizer, dataset: str) -> float:
    extracted_answer = get_extracted_answer(dataset, full_text)
    if not extracted_answer or extracted_answer not in full_text:
        return 0.0

    start_char_idx = full_text.rfind(extracted_answer)
    end_char_idx = start_char_idx + len(extracted_answer)
    
    current_char_idx = 0
    answer_logprobs = []
    
    for i, token_id in enumerate(token_ids):
        # Decode this specific token to see its string representation
        token_str = tokenizer.decode([token_id])
        
        # prompt_logprobs[i] contains the logprob for token_ids[i] given tokens 0 to i-1
        logprob_dict = prompt_logprobs[i]
        
        token_logprob = 0.0
        if logprob_dict and token_id in logprob_dict:
            token_logprob = logprob_dict[token_id].logprob
            
        token_end_idx = current_char_idx + len(token_str)
        
        if current_char_idx < end_char_idx and token_end_idx > start_char_idx:
            answer_logprobs.append(token_logprob)
            
        current_char_idx = token_end_idx
        
    if not answer_logprobs:
        return 0.0
        
    mean_logprob = sum(answer_logprobs) / len(answer_logprobs)
    return math.exp(mean_logprob)

@app.command()
def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset: str = "gsm8k",
    candidates_path: str = "",
    gpu_memory_utilization: float = 0.7,
):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed.")
        raise typer.Exit(1)

    in_path = Path(candidates_path)
    model_shortname = model_name.split("/")[-1].lower()
    out_path = Path(f"data/scores/{model_shortname}/{in_path.name.replace('.jsonl', '_scores_wcite.jsonl')}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_indices = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                completed_indices.add(row["example_index"])

    examples = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex["example_index"] not in completed_indices:
                examples.append(ex)

    if not examples:
        print("All examples already scored.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from src.core.data import dataset_system_message
    system_msg = dataset_system_message(dataset)

    print(f"\nLoading vLLM engine: {model_name}...")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
    )

    CHUNK_SIZE = 100
    
    for chunk_start in range(0, len(examples), CHUNK_SIZE):
        chunk = examples[chunk_start:chunk_start+CHUNK_SIZE]
        
        flat_prompts = []
        for ex in chunk:
            user_prompt = ex["prompt"]
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ]
            try:
                formatted_prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prefix = f"Question: {user_prompt}\nAnswer: Let me solve this step by step.\n"
                
            for cand in ex["candidates"][:24]:
                flat_prompts.append(formatted_prefix + cand)
                
        print(f"Running vLLM forward pass for {len(flat_prompts)} sequences...")
        outputs = llm.generate(flat_prompts, sampling_params)
        
        # Reconstruct and compute weights
        output_idx = 0
        with out_path.open("a", encoding="utf-8") as f_out:
            for ex in chunk:
                wcite_weights = []
                for cand in ex["candidates"][:24]:
                    out = outputs[output_idx]
                    token_ids = out.prompt_token_ids
                    prompt_logprobs = out.prompt_logprobs
                    
                    # We need to map token_ids back to the *candidate* part of the string.
                    # Since we passed formatted_prefix + cand, the candidate is at the end.
                    # But it's easier to just pass the candidate string to compute_wcite_weight
                    # wait, compute_wcite_weight expects the full text!
                    full_text = flat_prompts[output_idx]
                    weight = compute_wcite_weight(full_text, token_ids, prompt_logprobs, tokenizer, dataset)
                    wcite_weights.append(weight)
                    
                    output_idx += 1
                    
                out_row = {
                    "example_index": ex["example_index"],
                    "prompt": ex["prompt"],
                    "reward_model": "wcite",
                    "scores": wcite_weights,
                }
                f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                
    print(f"Finished extracting W-CITE scores to {out_path}")

if __name__ == "__main__":
    app()
