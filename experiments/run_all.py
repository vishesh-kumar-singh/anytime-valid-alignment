#!/usr/bin/env python3
"""Run all (or a subset of) hypothesis experiments.

Usage::

    # Run all offline hypotheses
    python experiments/run_all.py \\
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \\
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

    # Run specific hypotheses
    python experiments/run_all.py \\
        --hypotheses h1,h3,h5 \\
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \\
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hypotheses import get_hypothesis, list_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple hypothesis experiments in sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hypotheses", type=str, default="",
        help="Comma-separated hypothesis IDs. Default: all registered hypotheses.",
    )
    parser.add_argument(
        "--candidates-path", type=str, required=True,
        help="Path to precomputed candidates JSONL file.",
    )
    parser.add_argument(
        "--score-paths", type=str, required=True,
        help="Glob pattern or comma-separated paths to RM score JSONL files.",
    )
    parser.add_argument(
        "--eval-n", type=str, default="1,4,8,16,24",
        help="Comma-separated N values (default: 1,4,8,16,24).",
    )
    parser.add_argument(
        "--variance-norm-mode", type=str, default="robust_zscore",
        help="Normalization mode (default: robust_zscore).",
    )

    args = parser.parse_args()

    # Determine which hypotheses to run
    if args.hypotheses:
        hypothesis_ids = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    else:
        hypothesis_ids = list(list_hypotheses().keys())

    print(f"\n{'=' * 60}")
    print(f"  Running {len(hypothesis_ids)} hypotheses: {hypothesis_ids}")
    print(f"  Candidates: {args.candidates_path}")
    print(f"  Scores:     {args.score_paths}")
    print(f"  N values:   {args.eval_n}")
    print(f"{'=' * 60}\n")

    results_summary = {}
    total_start = time.time()

    for i, hid in enumerate(hypothesis_ids, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(hypothesis_ids)}] {hid}")
        print(f"{'─' * 60}")

        t0 = time.time()
        try:
            h = get_hypothesis(hid)
            results = h.run(
                candidates_path=args.candidates_path,
                score_paths=args.score_paths,
                eval_n=args.eval_n,
                variance_norm_mode=args.variance_norm_mode,
            )
            dataset = results.get("dataset", "unknown")
            manager = h.save(results, dataset=dataset)
            elapsed = time.time() - t0
            results_summary[hid] = {
                "status": "success",
                "time_sec": elapsed,
                "output_dir": str(manager.output_dir),
            }
            print(f"  ✓ {hid} completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results_summary[hid] = {
                "status": "failed",
                "time_sec": elapsed,
                "error": str(e),
            }
            print(f"  ✗ {hid} failed in {elapsed:.1f}s: {e}")
            traceback.print_exc()

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {len(hypothesis_ids)} hypotheses in {total_time:.1f}s")
    print(f"{'=' * 60}")
    for hid, info in results_summary.items():
        status = "✓" if info["status"] == "success" else "✗"
        line = f"  {status} {hid:<6s} ({info['time_sec']:.1f}s)"
        if info["status"] == "success":
            line += f"  → {info['output_dir']}"
        else:
            line += f"  ERROR: {info.get('error', 'unknown')}"
        print(line)
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
