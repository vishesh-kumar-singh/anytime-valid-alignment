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
        "--model", type=str, required=True,
        help="Base model short name (e.g. qwen2.5-3b-instruct).",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (e.g. gsm8k, math500, mmlu).",
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
        "--candidates-n", type=str, default="24",
        help="The N value used in the candidate filename (e.g., 24 or 512).",
    )
    parser.add_argument(
        "--experiment-suffix", type=str, default="",
        help="Optional suffix to append to the experiment output directory name.",
    )

    parser.add_argument(
        "--output-model", type=str, default="",
        help="Optional output model name (e.g. strong-qwen2.5-3b-instruct).",
    )
    parser.add_argument(
        "--reward-models", type=str, default="*",
        help="Comma-separated list of reward model shortnames to evaluate, or * for all.",
    )


    parser.add_argument(
        "--alpha-values", type=str, default="",
        help="Comma-separated alpha values for sweeps.",
    )
    parser.add_argument(
        "--p-dagger-values", type=str, default="",
        help="Comma-separated p_dagger values for sweeps.",
    )
    parser.add_argument(
        "--c-values", type=str, default="",
        help="Comma-separated c values for sweeps.",
    )
    args = parser.parse_args()
    args.output_model = args.output_model or args.model

    # Determine which pipelines to run
    if args.pipelines:
        pipeline_ids = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    else:
        pipeline_ids = list(list_pipelines().keys())

    cand_n = args.candidates_n
    candidates_path = f"data/candidates/{args.model}/{args.dataset}_n{cand_n}_full_vllm.jsonl"
    if args.reward_models == "*":
        score_paths = f"data/scores/{args.model}/{args.dataset}_n{cand_n}_full_vllm_scores_*.jsonl"
    else:
        rms = [rm.strip() for rm in args.reward_models.split(",")]
        score_paths = ",".join([f"data/scores/{args.model}/{args.dataset}_n{cand_n}_full_vllm_scores_{rm}.jsonl" for rm in rms])

    print(f"\n{'=' * 60}")
    print(f"  Running {len(pipeline_ids)} pipelines: {pipeline_ids}")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Candidates: {candidates_path}")
    print(f"  Scores:     {score_paths}")
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

            kwargs = {}
            if args.alpha_values:
                kwargs["alpha_values"] = [float(x.strip()) for x in args.alpha_values.split(",")]
            if args.p_dagger_values:
                kwargs["p_dagger_values"] = [float(x.strip()) for x in args.p_dagger_values.split(",")]
            if args.c_values:
                kwargs["c_values"] = [float(x.strip()) for x in args.c_values.split(",")]
            p = get_pipeline(pid, **kwargs)

            results = p.run(
                candidates_path=candidates_path,
                score_paths=score_paths,
                eval_n=args.eval_n,
                variance_norm_mode=args.variance_norm_mode,
            )
            dataset = results.get("dataset", "unknown")
            exp_name = p.make_experiment_name(dataset)
            if args.experiment_suffix:
                exp_name += args.experiment_suffix
            manager = p.save(results, dataset=dataset, experiment_name=exp_name, model=args.output_model)
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
