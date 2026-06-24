"""
Step 2: Score Candidates with a Reward Model
Reads candidates from Step 1, scores them, and saves the scores.

Usage:
    python scripts/score_candidates.py \
        --candidates-path data/candidates/math500_n24_full_vllm.jsonl \
        --reward-model "OpenAssistant/reward-model-deberta-v3-large-v2" \
        --load-in-4bit
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logger import ExperimentLogger
from src.core.config import BaseExperimentConfig as ExperimentConfig
from src.core.scorer import build_scorer

app = typer.Typer()

@app.command()
def main(
    candidates_path: str = typer.Option(..., help="Path to generated candidates JSONL"),
    reward_model: str = typer.Option(..., help="Reward model name"),
    load_in_4bit: bool = typer.Option(False, help="Load RM in 4-bit"),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path"),
    num_chunks: int = typer.Option(1, help="Total number of chunks to split the data into"),
    chunk_idx: int = typer.Option(0, help="Index of the chunk to process (0 to num_chunks-1)"),
):
    in_path = Path(candidates_path)
    if not in_path.exists():
        print(f"ERROR: Cannot find {in_path}")
        raise typer.Exit(1)
        
    _name = in_path.name.lower()
    dataset = "gsm8k" if "gsm8k" in _name else ("alpaca_eval" if "alpaca" in _name else "math500")
    rm_short = reward_model.split("/")[-1]

    if output_path is None:
        chunk_suffix = f"_chunk{chunk_idx}of{num_chunks}" if num_chunks > 1 else ""
        out_name = in_path.stem + f"_scores_{rm_short}{chunk_suffix}.jsonl"
        out_path_obj = in_path.parent.parent / "scores" / out_name
    else:
        out_path_obj = Path(output_path)

    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_indices = set()
    if out_path_obj.exists():
        with out_path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                completed_indices.add(row["example_index"])
        print(f"Resuming: {len(completed_indices)} examples already scored in this chunk.")

    # Load candidates
    examples = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    # Slice examples for the current chunk
    if num_chunks > 1:
        chunk_size = len(examples) // num_chunks
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if chunk_idx < num_chunks - 1 else len(examples)
        examples = examples[start_idx:end_idx]
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: {len(examples)} examples (from {start_idx} to {end_idx-1})")

    examples_to_score = [ex for ex in examples if ex["example_index"] not in completed_indices]
    
    if not examples_to_score:
        print(f"All {len(examples)} examples in chunk already scored!")
        return

    # Logger
    exp_logger = ExperimentLogger(log_dir="logs", experiment_name=f"score_{dataset}_{rm_short}")
    exp_logger.log_environment()

    # Load RM
    print(f"\nLoading Reward Model: {reward_model}...")
    t0 = time.time()
    exp_logger.log_phase_start("rm_loading", reward_model)
    
    cfg = ExperimentConfig(
        dataset=dataset,
        reward_model_names=[reward_model],
        rm_mode="trivial4",
        reward_load_in_4bit=load_in_4bit,
    )
    scorer = build_scorer(cfg)
    exp_logger.log_phase_end("rm_loading", time.time() - t0)

    print(f"\n{'=' * 70}")
    print(f"  SCORING CANDIDATES (Step 2 of 3)")
    print(f"  Dataset:     {dataset}")
    print(f"  Input:       {candidates_path}")
    print(f"  RM:          {reward_model}")
    print(f"  Remaining:   {len(examples_to_score)}")
    print(f"  Output:      {out_path_obj}")
    print(f"{'=' * 70}\n")

    exp_logger.log_inference_start(len(examples_to_score), examples_to_score[0].get("n_candidates", 24))

    with out_path_obj.open("a", encoding="utf-8") as f_out:
        for ex in tqdm(examples_to_score, desc="Scoring"):
            idx = ex["example_index"]
            prompt = ex["prompt"]
            candidates = ex["candidates"]
            
            t_rm = time.perf_counter()
            raw_means, raw_variances = scorer.score(prompt, candidates)
            rm_time = time.perf_counter() - t_rm
            
            # Use raw member_scores if available (since there's only 1 RM, means = member_scores[0])
            member_scores = getattr(scorer, "last_member_scores", None)
            scores_to_save = member_scores[0] if member_scores else raw_means
            
            out_row = {
                "example_index": idx,
                "prompt": prompt,
                "reward_model": reward_model,
                "scores": scores_to_save,
                "rm_time_sec": rm_time,
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()
            
            exp_logger.log_example_done(idx, [{"method": "scoring", "is_correct": None}])

    exp_logger.log_inference_end(len(examples_to_score))
    print(f"\n✓ Scored {len(examples_to_score)} examples!")
    print(f"✓ Saved to: {out_path_obj}")
    print("  Next step: run scripts/evaluate_offline.py")

if __name__ == "__main__":
    app()
