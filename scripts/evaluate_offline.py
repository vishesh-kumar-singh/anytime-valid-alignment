"""
Step 3: Offline Evaluation
Evaluates all risk-aware ensembling algorithms using the pre-computed candidates and RM scores.

Usage:
    python scripts/evaluate_offline.py \
        --candidates-path data/candidates/math500_n24_full_vllm.jsonl \
        --score-paths "data/scores/math500_n24_full_vllm_scores_*.jsonl" \
        --eval-n 1,4,8,16,24
"""

import glob
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils.gsm8k import check_answer_correct as check_gsm8k
from src.data_utils.math500 import check_answer_correct as check_math500
from src.experiments.run_risk_aware_inference import (
    run_vanilla_bon, run_bon_sharpe, run_bon_linear, run_soft_bon, 
    run_softbon_sharpe, run_softbon_linear, run_mob_bon, run_wbon, run_wbon_sharpe,
    _normalize_member_scores
)

app = typer.Typer()
console = Console()

def check_correctness(prediction: str, ground_truth: str, dataset: str) -> bool:
    if "gsm8k" in dataset.lower():
        return check_gsm8k(prediction, ground_truth)
    return check_math500(prediction, ground_truth)

@app.command()
def main(
    candidates_path: str = typer.Option(..., help="Path to generated candidates JSONL"),
    score_paths: str = typer.Option(..., help="Glob pattern for RM score JSONL files"),
    eval_n: str = typer.Option("1,4,8,16,24", help="Comma-separated N values to evaluate"),
    variance_norm_mode: str = typer.Option("robust_zscore", help="Normalization for RM ensemble"),
):
    """Run offline evaluation on precomputed candidates and scores."""
    
    cand_path = Path(candidates_path)
    if not cand_path.exists():
        console.print(f"[red]Error: Cannot find candidates file: {candidates_path}[/red]")
        raise typer.Exit(1)
        
    dataset = "gsm8k" if "gsm8k" in cand_path.name.lower() else "math500"
    n_values = [int(x.strip()) for x in eval_n.split(",")]
    
    # Load candidates
    examples = []
    with cand_path.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            
    # Resolve and load score files
    score_files = glob.glob(score_paths)
    if not score_files:
        console.print(f"[red]Error: No score files found matching pattern: {score_paths}[/red]")
        raise typer.Exit(1)
        
    rm_names = []
    all_rm_scores = {} # idx -> list of scores per RM
    for sf in score_files:
        rm_name = Path(sf).stem.split("_scores_")[-1]
        rm_names.append(rm_name)
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                idx = row["example_index"]
                if idx not in all_rm_scores:
                    all_rm_scores[idx] = []
                all_rm_scores[idx].append(row["scores"])
                
    console.print(f"[bold green]Loaded {len(examples)} examples and {len(rm_names)} RM score files.[/bold green]")
    
    # Evaluate for each N
    results_by_n = {}
    
    for n in sorted(n_values):
        console.print(f"\nEvaluating N={n}...")
        correct_counts = {}
        total_counts = {}
        
        for ex in examples:
            idx = ex["example_index"]
            if idx not in all_rm_scores or len(all_rm_scores[idx]) != len(rm_names):
                continue
                
            candidates = ex["candidates"][:n]
            gold = ex["ground_truth"]
            
            # Slice scores to N
            member_scores = [s[:n] for s in all_rm_scores[idx]]
            
            # Ensemble
            normalized_ms = _normalize_member_scores(member_scores, variance_norm_mode, 1e-6, False)
            norm_arr = np.asarray(normalized_ms, dtype=float)
            means = np.mean(norm_arr, axis=0).tolist()
            variances = np.var(norm_arr, axis=0, ddof=0).tolist()
            
            # Methods
            methods = []
            
            # Individual RMs
            for j, rm in enumerate(rm_names):
                m = run_vanilla_bon(candidates, member_scores[j], variances)
                m.method = f"vanilla_bon ({rm})"
                methods.append(m)
                
            # Ensemble methods
            m_ens = run_vanilla_bon(candidates, means, variances)
            m_ens.method = "vanilla_bon (ensemble)"
            methods.append(m_ens)
            
            rng = np.random.default_rng(42)
            methods.append(run_bon_sharpe(candidates, means, variances, eps=1e-8))
            methods.append(run_wbon(candidates, member_scores, eps=1e-8))
            methods.append(run_wbon_sharpe(candidates, member_scores, eps=1e-8))
            methods.append(run_mob_bon(candidates, means, variances, n_bootstrap=50, subsample_frac=0.6, use_variance_weighting=False, rng=rng))
            methods.append(run_mob_bon(candidates, means, variances, n_bootstrap=50, subsample_frac=0.6, use_variance_weighting=True, eps=1e-8, rng=rng))
            
            # Grade
            for m in methods:
                method_name = m.method
                is_correct = check_correctness(m.selected_answer, gold, dataset)
                
                if method_name not in total_counts:
                    total_counts[method_name] = 0
                    correct_counts[method_name] = 0
                    
                total_counts[method_name] += 1
                if is_correct:
                    correct_counts[method_name] += 1
                    
        results_by_n[n] = {m: correct_counts[m] / total_counts[m] for m in total_counts}
        
    # Print Table
    table = Table(title=f"Offline Evaluation ({dataset.upper()})")
    table.add_column("Method", style="cyan")
    for n in sorted(n_values):
        table.add_column(f"N={n}", justify="right")
        
    method_names = list(results_by_n[sorted(n_values)[-1]].keys())
    for method in method_names:
        row = [method]
        for n in sorted(n_values):
            acc = results_by_n[n].get(method, 0.0)
            row.append(f"{acc * 100:.1f}%")
        table.add_row(*row)
        
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    app()
