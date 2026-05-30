#!/usr/bin/env python3
"""Run a single hypothesis experiment.

Usage::

    # List available hypotheses
    python experiments/run_hypothesis.py --list

    # Run hypothesis h5 (offline parallel methods comparison)
    python experiments/run_hypothesis.py \\
        --hypothesis h5 \\
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \\
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \\
        --eval-n 1,4,8,16,24

    # Run hypothesis h1 with custom normalization
    python experiments/run_hypothesis.py \\
        --hypothesis h1 \\
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \\
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \\
        --variance-norm-mode robust_zscore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hypotheses import get_hypothesis, list_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hypothesis experiment from the modular framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available hypotheses and exit.",
    )
    parser.add_argument(
        "--hypothesis", "-H", type=str, default="",
        help="Hypothesis ID to run (e.g. h1, h2, ..., h10).",
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
        print("\n  Available Hypotheses")
        print("  " + "=" * 60)
        for hid, desc in list_hypotheses().items():
            print(f"  {hid:>4s} │ {desc}")
        print("  " + "=" * 60)
        print(f"\n  Usage: python experiments/run_hypothesis.py --hypothesis h5 --candidates-path <path> --score-paths <glob>\n")
        return

    # ── Validation ──
    if not args.hypothesis:
        parser.error("--hypothesis is required (e.g. --hypothesis h5). Use --list to see options.")
    if not args.candidates_path:
        parser.error("--candidates-path is required.")
    if not args.score_paths:
        parser.error("--score-paths is required.")

    # ── Run ──
    h = get_hypothesis(args.hypothesis)
    print(f"\n{'=' * 60}")
    print(f"  Running: {h.name}")
    print(f"  {h.description}")
    print(f"{'=' * 60}\n")

    results = h.run(
        candidates_path=args.candidates_path,
        score_paths=args.score_paths,
        eval_n=args.eval_n,
        variance_norm_mode=args.variance_norm_mode,
    )

    dataset = results.get("dataset", "unknown")
    h.save(
        results,
        dataset=dataset,
        experiment_name=args.experiment_name or None,
    )


if __name__ == "__main__":
    main()
