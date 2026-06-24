#!/usr/bin/env python3
"""Run all (or a subset of) evaluation pipelines.

Usage::

    # Run all offline pipelines
    python experiments/run_all_pipelines.py \
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

    # Run specific pipelines
    python experiments/run_all_pipelines.py \
        --pipelines sprt_rwsc,top_k_mbr_rwsc \
        --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
        --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines import get_pipeline, list_pipelines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple evaluation pipelines in sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipelines", "-P", type=str, default="",
        help="Comma-separated pipeline IDs. Default: all registered pipelines.",
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
        "--eval-n", type=str, default="24",
        help="Comma-separated N values (default: 24).",
    )
    parser.add_argument(
        "--variance-norm-mode", type=str, default="robust_zscore",
        help="Normalization mode (default: robust_zscore).",
    )
    parser.add_argument(
        "--experiment-suffix", type=str, default="",
        help="Optional suffix to append to the experiment output directory name.",
    )

    args = parser.parse_args()

    # Determine which pipelines to run
    if args.pipelines:
        pipeline_ids = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    else:
        pipeline_ids = list(list_pipelines().keys())

    print(f"\n{'=' * 60}")
    print(f"  Running {len(pipeline_ids)} pipelines: {pipeline_ids}")
    print(f"  Candidates: {args.candidates_path}")
    print(f"  Scores:     {args.score_paths}")
    print(f"  N values:   {args.eval_n}")
    print(f"{'=' * 60}\n")

    results_summary = {}
    total_start = time.time()

    for i, pid in enumerate(pipeline_ids, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(pipeline_ids)}] {pid}")
        print(f"{'─' * 60}")

        t0 = time.time()
        try:
            p = get_pipeline(pid)
            results = p.run(
                candidates_path=args.candidates_path,
                score_paths=args.score_paths,
                eval_n=args.eval_n,
                variance_norm_mode=args.variance_norm_mode,
            )
            dataset = results.get("dataset", "unknown")
            exp_name = p.make_experiment_name(dataset)
            if args.experiment_suffix:
                exp_name += args.experiment_suffix
            manager = p.save(results, dataset=dataset, experiment_name=exp_name)
            elapsed = time.time() - t0
            results_summary[pid] = {
                "status": "success",
                "time_sec": elapsed,
                "output_dir": str(manager.output_dir),
            }
            print(f"  ✓ {pid} completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results_summary[pid] = {
                "status": "failed",
                "time_sec": elapsed,
                "error": str(e),
            }
            print(f"  ✗ {pid} failed in {elapsed:.1f}s: {e}")
            traceback.print_exc()

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {len(pipeline_ids)} pipelines in {total_time:.1f}s")
    print(f"{'=' * 60}")
    for pid, info in results_summary.items():
        status = "✓" if info["status"] == "success" else "✗"
        line = f"  {status} {pid:<20s} ({info['time_sec']:.1f}s)"
        if info["status"] == "success":
            line += f"  → {info['output_dir']}"
        else:
            line += f"  ERROR: {info.get('error', 'unknown')}"
        print(line)
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
