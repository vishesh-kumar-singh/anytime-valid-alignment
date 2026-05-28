#!/usr/bin/env python3
"""
Direction 1 — Plot N-sweep results for reward hacking mitigation at scale.

Reads the per-N analysis JSONs produced by analyze_uncertainty_reward_hacking.py
(or falls back to reading the raw JSONL files directly), then produces:

  1. accuracy_vs_n.png          — Accuracy vs N for each selection method
  2. reward_hacking_vs_n.png    — Top-10% Reward Wrong Rate vs N
  3. variance_gap_vs_n.png      — (Var_wrong − Var_correct) vs N
  4. combined_n_sweep.png       — All three in a single figure

One set of plots per dataset (e.g. accuracy_vs_n_gsm8k.png).

Usage:
    python scripts/plot_n_sweep_results.py --sweep-dir runs/direction1_n_sweep
    python scripts/plot_n_sweep_results.py --sweep-dir runs/direction1_n_sweep --out-dir runs/direction1_n_sweep
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---- Publication-quality matplotlib defaults ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

# ---- Method display configuration ----
# We focus on the key methods that illustrate the reward hacking story.
# Map internal method names → display label, color, marker, linestyle
METHOD_CONFIG = {
    "vanilla_bon": {
        "label": "Vanilla BoN (argmax μ)",
        "color": "#e74c3c",
        "marker": "o",
        "linestyle": "-",
    },
    "bon_sharpe": {
        "label": "BoN-Sharpe (argmax μ/σ)",
        "color": "#2ecc71",
        "marker": "s",
        "linestyle": "-",
    },
    "bon_linear": {
        "label": "BoN-Linear (argmax μ−cσ)",
        "color": "#3498db",
        "marker": "D",
        "linestyle": "-",
    },
    "risk_adjusted_bon": {
        "label": "Risk-Adj BoN (μ−λσ²)",
        "color": "#9b59b6",
        "marker": "^",
        "linestyle": "--",
    },
    "soft_bon": {
        "label": "SoftBoN",
        "color": "#f39c12",
        "marker": "v",
        "linestyle": "--",
    },
    "softbon_sharpe": {
        "label": "SoftBoN-Sharpe",
        "color": "#1abc9c",
        "marker": "P",
        "linestyle": "--",
    },
    "softbon_linear": {
        "label": "SoftBoN-Linear",
        "color": "#2980b9",
        "marker": "X",
        "linestyle": "--",
    },
    "sharpe_ratio_bon": {
        "label": "Sharpe-Ratio BoN",
        "color": "#27ae60",
        "marker": "s",
        "linestyle": "-",
    },
}

# Primary methods to highlight (deterministic selection — cleaner story)
PRIMARY_METHODS = ["vanilla_bon", "bon_sharpe", "bon_linear"]

# All methods (including stochastic soft variants)
ALL_METHODS = list(METHOD_CONFIG.keys())


def _parse_sweep_dir(sweep_dir: str) -> Dict[str, Dict[int, Dict[str, dict]]]:
    """Parse the sweep directory to extract metrics per dataset, per N, per method.

    Returns:
        {dataset: {N: {method: {accuracy, top10_reward_wrong_rate, var_gap, ...}}}}
    """
    results: Dict[str, Dict[int, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))

    sweep_path = Path(sweep_dir)

    # Strategy 1: Read from analysis JSON files (preferred)
    for analysis_dir in sorted(sweep_path.glob("analysis_*_n*")):
        # Parse dataset and N from directory name: analysis_gsm8k_n8
        match = re.match(r"analysis_(\w+)_n(\d+)", analysis_dir.name)
        if not match:
            continue
        dataset = match.group(1)
        n_val = int(match.group(2))

        metrics_json = analysis_dir / "uncertainty_rewardhacking_metrics.json"
        if not metrics_json.exists():
            continue

        with open(metrics_json, "r", encoding="utf-8") as f:
            method_metrics = json.load(f)

        for m in method_metrics:
            method_name = m["method"]
            results[dataset][n_val][method_name] = {
                "accuracy": m.get("accuracy", float("nan")),
                "top10_reward_wrong_rate": m.get("top10_reward_wrong_rate", float("nan")),
                "var_correct_mean": m.get("var_correct_mean", float("nan")),
                "var_wrong_mean": m.get("var_wrong_mean", float("nan")),
                "var_gap": m.get("var_wrong_minus_correct", float("nan")),
                "auc_confidence": m.get("auc_confidence_negvar", float("nan")),
            }

    # Strategy 2: If no analysis dirs found, fall back to raw JSONL files
    if not results:
        for jsonl_file in sorted(sweep_path.glob("nsweep_*_n*.jsonl")):
            match = re.match(r"nsweep_(\w+)_n(\d+)\.jsonl", jsonl_file.name)
            if not match:
                continue
            dataset = match.group(1)
            n_val = int(match.group(2))

            rows = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))

            # Group by method
            by_method: Dict[str, List[dict]] = defaultdict(list)
            for row in rows:
                by_method[row["method"]].append(row)

            for method_name, method_rows in by_method.items():
                y = np.array([1 if r["is_correct"] else 0 for r in method_rows])
                mu = np.array([float(r["reward_mean"]) for r in method_rows])
                var = np.array([float(r["reward_variance"]) for r in method_rows])

                accuracy = float(y.mean())

                # Top-10% reward wrong rate
                q90 = float(np.quantile(mu, 0.9))
                top_mask = mu >= q90
                top_n = int(top_mask.sum())
                wrong_top = int(((y == 0) & top_mask).sum())
                top10_wrong_rate = float(wrong_top / top_n) if top_n > 0 else float("nan")

                var_correct = float(var[y == 1].mean()) if (y == 1).sum() > 0 else float("nan")
                var_wrong = float(var[y == 0].mean()) if (y == 0).sum() > 0 else float("nan")

                results[dataset][n_val][method_name] = {
                    "accuracy": accuracy,
                    "top10_reward_wrong_rate": top10_wrong_rate,
                    "var_correct_mean": var_correct,
                    "var_wrong_mean": var_wrong,
                    "var_gap": float(var_wrong - var_correct),
                    "auc_confidence": float("nan"),
                }

    return dict(results)


def _plot_metric_vs_n(
    results: Dict[int, Dict[str, dict]],
    metric_key: str,
    ylabel: str,
    title: str,
    ax: plt.Axes,
    methods: Optional[List[str]] = None,
    invert_better: bool = False,
) -> None:
    """Plot a single metric vs N on the given axes."""
    if methods is None:
        methods = PRIMARY_METHODS

    n_values = sorted(results.keys())

    for method_name in methods:
        cfg = METHOD_CONFIG.get(method_name)
        if cfg is None:
            # Unknown method — use defaults
            cfg = {
                "label": method_name,
                "color": "#7f8c8d",
                "marker": "o",
                "linestyle": "-",
            }

        xs = []
        ys = []
        for n in n_values:
            if method_name in results[n]:
                val = results[n][method_name].get(metric_key, float("nan"))
                if not np.isnan(val):
                    xs.append(n)
                    ys.append(val)

        if xs:
            ax.plot(
                xs, ys,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            )

    ax.set_xlabel("Number of Candidates (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_dataset(
    dataset: str,
    results: Dict[int, Dict[str, dict]],
    out_dir: str,
    methods: Optional[List[str]] = None,
) -> None:
    """Generate all plots for a single dataset."""
    os.makedirs(out_dir, exist_ok=True)
    ds_label = dataset.upper()

    if methods is None:
        # Auto-detect: use primary methods if available, otherwise all found
        all_found = set()
        for n_data in results.values():
            all_found.update(n_data.keys())
        methods = [m for m in PRIMARY_METHODS if m in all_found]
        # Add any extra methods found
        extra = [m for m in ALL_METHODS if m in all_found and m not in methods]
        methods.extend(extra)

    if not methods:
        print(f"  ⚠ No methods found for {dataset}, skipping.")
        return

    # ---- Plot 1: Accuracy vs N ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_metric_vs_n(
        results, "accuracy", "Accuracy",
        f"Accuracy vs N — {ds_label}",
        ax, methods,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"accuracy_vs_n_{dataset}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

    # ---- Plot 2: Top-10% Reward Wrong Rate vs N ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_metric_vs_n(
        results, "top10_reward_wrong_rate",
        "Top-10% Reward Wrong Rate",
        f"Reward Hacking Proxy vs N — {ds_label}",
        ax, methods,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"reward_hacking_vs_n_{dataset}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

    # ---- Plot 3: Variance Gap vs N ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_metric_vs_n(
        results, "var_gap",
        "Var(wrong) − Var(correct)",
        f"Variance Gap vs N — {ds_label}",
        ax, methods,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"variance_gap_vs_n_{dataset}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

    # ---- Plot 4: Combined 3-panel figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_metric_vs_n(
        results, "accuracy", "Accuracy",
        f"Accuracy vs N — {ds_label}",
        axes[0], methods,
    )
    _plot_metric_vs_n(
        results, "top10_reward_wrong_rate",
        "Top-10% Reward Wrong Rate",
        f"Reward Hacking Proxy — {ds_label}",
        axes[1], methods,
    )
    _plot_metric_vs_n(
        results, "var_gap",
        "Var(wrong) − Var(correct)",
        f"Variance Gap — {ds_label}",
        axes[2], methods,
    )
    fig.suptitle(f"Direction 1: Reward Hacking at Scale — {ds_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"combined_n_sweep_{dataset}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def export_summary_csv(
    all_results: Dict[str, Dict[int, Dict[str, dict]]],
    out_dir: str,
) -> None:
    """Export a single CSV with all metrics across datasets/N/methods."""
    import csv

    csv_path = os.path.join(out_dir, "n_sweep_summary.csv")
    fieldnames = [
        "dataset", "n", "method",
        "accuracy", "top10_reward_wrong_rate",
        "var_correct_mean", "var_wrong_mean", "var_gap",
        "auc_confidence",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(all_results):
            for n in sorted(all_results[dataset]):
                for method, metrics in sorted(all_results[dataset][n].items()):
                    writer.writerow({
                        "dataset": dataset,
                        "n": n,
                        "method": method,
                        **{k: metrics.get(k, "") for k in fieldnames[3:]},
                    })

    print(f"  ✓ {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot N-sweep results for Direction 1 (Reward Hacking at Scale)."
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Directory containing nsweep_*_n*.jsonl and/or analysis_*_n*/ subdirs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots. Defaults to sweep-dir.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names to include. Default: auto-detect.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.sweep_dir
    os.makedirs(out_dir, exist_ok=True)

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"Parsing sweep directory: {args.sweep_dir}")
    all_results = _parse_sweep_dir(args.sweep_dir)

    if not all_results:
        print("⚠ No results found. Check that the sweep directory contains:")
        print("  - analysis_<dataset>_n<N>/uncertainty_rewardhacking_metrics.json")
        print("  - OR nsweep_<dataset>_n<N>.jsonl files")
        return

    for dataset in sorted(all_results):
        print(f"\n{'='*50}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  N values: {sorted(all_results[dataset].keys())}")
        print(f"{'='*50}")
        plot_dataset(dataset, all_results[dataset], out_dir, methods)

    print(f"\nExporting summary CSV...")
    export_summary_csv(all_results, out_dir)

    print(f"\n✓ All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
