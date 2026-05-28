#!/usr/bin/env python3
"""
Quick demo: run all 6 inference strategies on a few examples.
Useful for sanity-checking everything works before a full experiment.

Supports two modes:
  --mode trivial  (default) Pre-trained scalar RMs, no training needed
  --mode robust              Uncertainty-aware RMs (train first with uncertainty_rm.py)
"""

import sys


from src.experiments.run_risk_aware_inference import ExperimentConfig, run_inference


def make_config(
    mode: str = "trivial",
    dataset: str = "gsm8k",
    controller: str = "threshold",
    max_examples: int = 3,
) -> ExperimentConfig:
    """Build a small demo config.

    Args:
        mode: "trivial" (pre-trained RMs, no training) or "robust" (trained URM ensemble).
        dataset: "gsm8k" (math), "math500" (harder math), or "alpaca_eval" (open-ended).
        controller: "threshold" (default Sharpe), "thompson", "lcb", or "cvar".
        max_examples: Number of examples to run (keep small for demo).
    """
    demo_max_new_tokens = 256 if dataset.lower() == "math500" else 512

    base = ExperimentConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=demo_max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        torch_dtype="bfloat16",
        bon_n=4,
        seq_n=4,
        lam=0.05,
        gamma_vanilla=0.95,
        gamma_sharpe=0.95,
        gamma_lin=0.95,
        lin_c=1.0,
        history_budget=2,
        reward_batch_size=8,
        dataset=dataset,
        dataset_split="test",
        max_examples=max_examples,
        output_path=f"runs/demo_{mode}_{dataset}.jsonl",
        rm_mode=mode,
        controller_type=controller,
    )
    if mode == "robust":
        base.urm_ensemble_dir = "models/urm_ensemble"
        base.urm_base_model = "microsoft/deberta-v3-base"
    return base


if __name__ == "__main__":
    # Parse simple args: python run_demo.py [--mode trivial|robust] [--dataset gsm8k|math500|alpaca_eval]
    mode = "trivial"
    dataset = "gsm8k"
    controller = "threshold"
    max_ex = 3

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--controller" and i + 1 < len(args):
            controller = args[i + 1]
            i += 2
        elif args[i] == "--max-examples" and i + 1 < len(args):
            max_ex = int(args[i + 1])
            i += 2
        else:
            i += 1

    print(f"Demo: mode={mode}, dataset={dataset}, controller={controller}, max_examples={max_ex}")
    config = make_config(mode=mode, dataset=dataset, controller=controller, max_examples=max_ex)
    rows = run_inference(config)
    print(f"\nDone! {len(rows)} results saved to {config.output_path}")
