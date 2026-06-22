"""
Generate Candidate Responses via Ollama (for low-VRAM GPUs)

Uses the Ollama API for fast, quantized inference. Designed for
RTX 2050 / similar cards where vLLM is too slow and HF 4-bit is finicky.

Features:
  - Live-save: writes to JSONL after each question (crash-safe)
  - Resumable: skips already-completed example indices on restart
  - tqdm progress with ETA and speed metrics
  - Full responses: num_predict=2048 by default (no truncation)

Usage:
    conda run -n langchain python scripts/generate_candidates_ollama.py \
        --model-name qwen2.5:3b \
        --dataset math500 \
        --n-candidates 64 \
        --max-examples 0

    # Resume an interrupted run (just re-run the same command):
    conda run -n langchain python scripts/generate_candidates_ollama.py \
        --model-name qwen2.5:3b \
        --dataset math500 \
        --n-candidates 64 \
        --max-examples 0
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data import (
    load_dataset_examples,
    dataset_system_message as _dataset_system_message,
)

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option("qwen2.5:3b", help="Ollama model name."),
    dataset: str = typer.Option("math500", help="Dataset: gsm8k, math500, alpaca_eval, or alpaca."),
    n_candidates: int = typer.Option(64, help="Number of candidates per example."),
    max_examples: int = typer.Option(0, help="Max examples. 0 = full dataset."),
    temperature: float = typer.Option(0.7, help="Generation temperature."),
    top_p: float = typer.Option(0.95, help="Generation top_p."),
    num_predict: int = typer.Option(2048, help="Max tokens per generation (Ollama num_predict)."),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path."),
):
    """Generate N candidates per example using Ollama."""

    try:
        import ollama
    except ImportError:
        print("ERROR: ollama python client not installed.")
        print("  pip install ollama")
        raise typer.Exit(1)

    # Verify Ollama is running
    try:
        ollama.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama server: {e}")
        print("  Make sure Ollama is running: ollama serve")
        raise typer.Exit(1)

    dataset = dataset.lower()
    resolved_max = None if max_examples <= 0 else max_examples

    if output_path is None:
        scope = "full" if resolved_max is None else f"{resolved_max}ex"
        model_short = model_name.replace(":", "-").replace("/", "-")
        output_path = f"data/candidates/{dataset}_n{n_candidates}_{scope}_ollama.jsonl"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume support ──────────────────────────────────────────────────
    completed_indices = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed_indices.add(row["example_index"])
                except json.JSONDecodeError:
                    continue  # skip corrupted lines from crash
        print(f"📂 Resuming: {len(completed_indices)} examples already completed.")

    # ── Load Dataset ────────────────────────────────────────────────────
    print(f"Loading {dataset} dataset...")
    all_examples = load_dataset_examples(dataset, "test", resolved_max)

    # Filter to remaining
    examples_to_run = []
    for i, ex in enumerate(all_examples):
        if i not in completed_indices:
            ex["_global_index"] = i
            examples_to_run.append(ex)

    total_to_do = len(examples_to_run)
    total_overall = len(all_examples)
    already_done = len(completed_indices)

    if total_to_do == 0:
        print(f"✅ All {total_overall} examples already completed. Nothing to do.")
        return

    print(f"\n{'=' * 70}")
    print(f"  OLLAMA CANDIDATE GENERATION")
    print(f"  Model:       {model_name}")
    print(f"  Dataset:     {dataset} ({total_overall} examples)")
    print(f"  Candidates:  {n_candidates} per example")
    print(f"  Already done: {already_done}/{total_overall}")
    print(f"  Remaining:   {total_to_do}")
    print(f"  Tokens/resp: up to {num_predict}")
    print(f"  Temperature: {temperature}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 70}\n")

    system_msg = _dataset_system_message(dataset)

    # ── Main generation loop ────────────────────────────────────────────
    t_total_start = time.perf_counter()
    total_candidates_generated = 0

    pbar = tqdm(
        examples_to_run,
        desc="Questions",
        unit="q",
        dynamic_ncols=True,
    )

    for ex in pbar:
        idx = ex["_global_index"]
        prompt = ex["question"]
        gold_answer = ex.get("answer") or ex.get("reference", "")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        candidates = []
        t_q_start = time.perf_counter()

        # Inner loop: generate n_candidates one at a time
        for c_idx in range(n_candidates):
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": num_predict,
                        "seed": -1,  # random seed each time for diversity
                    },
                )
                candidate_text = response["message"]["content"].strip()
                candidates.append(candidate_text)
            except Exception as e:
                # Log error but continue — partial candidates still useful
                tqdm.write(f"  ⚠ Error on example {idx}, candidate {c_idx}: {e}")
                candidates.append("")

            # Update progress bar postfix with live stats
            elapsed = time.perf_counter() - t_total_start
            total_c = total_candidates_generated + len(candidates)
            cps = total_c / elapsed if elapsed > 0 else 0
            remaining_c = (total_to_do * n_candidates) - total_c
            eta_sec = remaining_c / cps if cps > 0 else 0
            eta_h = eta_sec / 3600

            pbar.set_postfix_str(
                f"c={c_idx+1}/{n_candidates} | "
                f"{cps:.1f} c/s | "
                f"ETA {eta_h:.1f}h",
                refresh=True,
            )

        q_time = time.perf_counter() - t_q_start
        total_candidates_generated += len(candidates)

        # ── Live save: append immediately ───────────────────────────────
        row = {
            "example_index": idx,
            "prompt": prompt,
            "ground_truth": gold_answer,
            "candidates": candidates,
            "n_candidates": n_candidates,
            "gen_time_sec": round(q_time, 2),
            "model_name": model_name,
            "dataset": dataset,
        }
        with out_path.open("a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Log progress
        completed_total = already_done + pbar.n
        tqdm.write(
            f"  ✓ [{completed_total}/{total_overall}] "
            f"Example {idx}: {len(candidates)} candidates in {q_time:.1f}s "
            f"(avg {q_time/max(len(candidates),1):.1f}s/candidate)"
        )

    total_time = time.perf_counter() - t_total_start
    total_hours = total_time / 3600

    print(f"\n{'=' * 70}")
    print(f"  ✅ GENERATION COMPLETE")
    print(f"  Total examples:   {total_to_do}")
    print(f"  Total candidates: {total_candidates_generated}")
    print(f"  Total time:       {total_hours:.2f} hours ({total_time:.0f}s)")
    print(f"  Throughput:       {total_candidates_generated/total_time:.2f} candidates/sec")
    print(f"  Saved to:         {output_path}")
    print(f"{'=' * 70}")
    print(f"\n  Next step: score with reward models:")
    print(f"  python scripts/score_candidates.py --candidates-path {output_path} --reward-model <model_name>")


if __name__ == "__main__":
    app()
