#!/usr/bin/env python3
"""Run a single evaluation pipeline.

Usage::

    # List available pipelines
    python experiments/run_pipeline.py --list

    # Run single_best_rm_rwsc pipeline
    python experiments/run_pipeline.py \
        --pipeline single_best_rm_rwsc \
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
        --eval-n 1,4,8,16,24

    # Run sprt_rwsc pipeline
    python experiments/run_pipeline.py \
        --pipeline sprt_rwsc \
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
        --variance-norm-mode robust_zscore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines import get_pipeline, list_pipelines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pipeline experiment from the modular framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available pipelines and exit.",
    )
    parser.add_argument(
        "--pipeline", "-P", type=str, default="",
        help="Pipeline ID to run (e.g. sprt_rwsc, top_k_mbr_rwsc).",
    )
    parser.add_argument(
        "--candidates-path", type=str, default="",
        help="Path to precomputed candidates JSONL file.",
    )
    parser.add_argument(
        "--score-paths", type=str, default="",
        help="Glob pattern or comma-separated paths to RM score JSONL files.",
    )
    parser.add_argument(
        "--eval-n", type=str, default="1,4,8,16,24",
        help="Comma-separated N values to evaluate (default: 1,4,8,16,24).",
    )
    parser.add_argument(
        "--variance-norm-mode", type=str, default="robust_zscore",
        help="Normalization mode: none, zscore, robust_zscore (default: robust_zscore).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="",
        help="Custom experiment name (default: auto-generated with timestamp).",
    )

    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print("\n  Available Pipelines")
        print("  " + "=" * 60)
        for pid, desc in list_pipelines().items():
            print(f"  {pid:>20s} │ {desc}")
        print("  " + "=" * 60)
        print(f"\n  Usage: python experiments/run_pipeline.py --pipeline sprt_rwsc --candidates-path <path> --score-paths <glob>\n")
        return

    # ── Validation ──
    if not args.pipeline:
        parser.error("--pipeline is required (e.g. --pipeline sprt_rwsc). Use --list to see options.")
    if not args.candidates_path:
        parser.error("--candidates-path is required.")
    if not args.score_paths:
        parser.error("--score-paths is required.")

    # ── Run ──
    p = get_pipeline(args.pipeline)
    print(f"\n{'=' * 60}")
    print(f"  Running: {p.name}")
    print(f"  {p.description}")
    print(f"{'=' * 60}\n")

    results = p.run(
        candidates_path=args.candidates_path,
        score_paths=args.score_paths,
        eval_n=args.eval_n,
        variance_norm_mode=args.variance_norm_mode,
    )

    dataset = results.get("dataset", "unknown")
    p.save(
        results,
        dataset=dataset,
        experiment_name=args.experiment_name or None,
    )


if __name__ == "__main__":
    main()
