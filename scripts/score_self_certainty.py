import json
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from pathlib import Path

def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1/V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    batch_confidence_list = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()
    return batch_confidence_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    in_path = Path(args.candidates_path)
    model_dir = args.model
    
    out_name = in_path.stem + "_scores_self_certainty.jsonl"
    out_path = in_path.parent.parent.parent / "scores" / in_path.parent.name / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    completed_indices = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                completed_indices.add(json.loads(line)["example_index"])
        print(f"Resuming: {len(completed_indices)} already scored.")

    examples = []
    with open(in_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            
    to_score = [e for e in examples if e["example_index"] not in completed_indices]
    if not to_score:
        print("All done!")
        return

    print(f"Loading {model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        llm.config.pad_token_id = tokenizer.pad_token_id

    llm.eval()
    V = llm.config.vocab_size

    with open(out_path, "a") as f_out:
        for ex in tqdm.tqdm(to_score, desc="Scoring"):
            prompt = ex["prompt"] if "prompt" in ex else ex["question"]
            candidates = ex["candidates"]
            
            # Combine prompt and candidates
            # Wait, for causal LM, we need to feed the prompt + candidate and extract logprobs of the candidate.
            # To match confidence_list.py, we encode prompt and prompt+candidate.
            
            scores = []
            
            # Process in batches
            for i in range(0, len(candidates), args.batch_size):
                batch_cands = candidates[i:i+args.batch_size]
                
                # To get exact candidate logprobs, we tokenize prompt, and prompt+cand.
                # Actually, confidence_list.py formats them with Chat Template!
                # Let's apply chat template.
                formatted_texts = []
                for cand in batch_cands:
                    messages = [
                        {"role": "system", "content": "Please answer the following math problem step by step."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cand}
                    ]
                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    except:
                        text = f"Question: {prompt}\nAnswer: Let me solve this step by step.\n{cand}"
                    formatted_texts.append(text)
                    
                # We need to find where the candidate starts.
                # A simpler approximation that confidence_list.py uses: it just tokenizes the prompt, gets length, and ignores those tokens!
                prompt_messages = [
                    {"role": "system", "content": "Please answer the following math problem step by step."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                except:
                    prompt_text = f"Question: {prompt}\nAnswer: Let me solve this step by step.\n"
                    
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                
                enc = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(device)
                
                with torch.no_grad():
                    outputs = llm(**enc)
                    
                logits = outputs.logits # (batch, seq_len, vocab)
                
                # We want logprobs of the actual tokens.
                # Shift logits and labels
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = enc["input_ids"][:, 1:].contiguous()
                shift_mask = enc["attention_mask"][:, 1:].contiguous()
                
                # Zero out the mask for prompt tokens
                shift_mask[:, :prompt_len-1] = 0
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                selected_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1) # (batch, seq_len-1)
                
                batch_scores = confidence_logprob_sum(selected_log_probs, shift_mask, V)
                scores.extend(batch_scores)
                
            out_row = {
                "example_index": ex["example_index"] if "example_index" in ex else ex["_global_index"],
                "prompt": prompt,
                "reward_model": "self_certainty",
                "scores": scores,
                "rm_time_sec": 0.0
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()
