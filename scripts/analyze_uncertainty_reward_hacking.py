#!/usr/bin/env python3
""" uncertainty-calibration and reward-hacking proxy diagnostics from a BoN JSONL run."""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC for binary labels using pairwise wins (no sklearn dependency)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty and reward-hacking proxies from BoN JSONL.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--out-dir", required=True, help="Directory to save analysis plots/reports.")
    args = parser.parse_args()

    input_jsonl = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = [json.loads(line) for line in open(input_jsonl, encoding="utf-8") if line.strip()]
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    report_rows = []

    for method in sorted(by_method):
        method_rows = by_method[method]
        mu = np.array([float(r["reward_mean"]) for r in method_rows], dtype=float)
        var = np.array([float(r["reward_variance"]) for r in method_rows], dtype=float)
        y = np.array([1 if r["is_correct"] else 0 for r in method_rows], dtype=int)

        # Plot 1: Uncertainty calibration (accuracy by variance bins).
        q = np.quantile(var, np.linspace(0.0, 1.0, 11))
        xs = []
        ys = []
        for i in range(10):
            lo = q[i]
            hi = q[i + 1]
            mask = (var >= lo) & ((var <= hi) if i == 9 else (var < hi))
            if int(mask.sum()) == 0:
                continue
            xs.append(float(var[mask].mean()))
            ys.append(float(y[mask].mean()))

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Mean selected variance per bin")
        plt.ylabel("Accuracy")
        plt.title(f"Uncertainty Calibration - {method}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/uncertainty_calibration_{method}.png", dpi=150)
        plt.close()

        # Plot 2: Reward-hacking proxy scatter.
        plt.figure(figsize=(6, 4))
        plt.scatter(mu[y == 1], var[y == 1], s=12, alpha=0.6, label="correct")
        plt.scatter(mu[y == 0], var[y == 0], s=12, alpha=0.6, label="wrong")
        plt.xlabel("Selected reward mean")
        plt.ylabel("Selected variance")
        plt.title(f"Reward-vs-Error - {method}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/reward_hacking_proxy_scatter_{method}.png", dpi=150)
        plt.close()

        conf_auc = auc_binary(-var, y)  # Lower variance => higher confidence
        q90 = float(np.quantile(mu, 0.9))
        top = mu >= q90
        top_n = int(top.sum())
        wrong_top = int(((y == 0) & top).sum())

        var_correct = float(var[y == 1].mean()) if int((y == 1).sum()) else float("nan")
        var_wrong = float(var[y == 0].mean()) if int((y == 0).sum()) else float("nan")

        report_rows.append(
            {
                "method": method,
                "accuracy": float(y.mean()),
                "auc_confidence_negvar": float(conf_auc),
                "var_correct_mean": var_correct,
                "var_wrong_mean": var_wrong,
                "var_wrong_minus_correct": float(var_wrong - var_correct),
                "top10_reward_wrong_rate": float(wrong_top / top_n) if top_n else float("nan"),
                "top10_reward_count": top_n,
                "top10_reward_wrong_count": wrong_top,
            }
        )

    report_json = f"{out_dir}/uncertainty_rewardhacking_metrics.json"
    report_csv = f"{out_dir}/uncertainty_rewardhacking_metrics.csv"
    report_txt = f"{out_dir}/uncertainty_rewardhacking_metrics.txt"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2)

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    lines = [
        "Method | Acc | AUC(-var->correct) | Var_wrong-Var_correct | Top10RewardWrongRate",
        "-" * 90,
    ]
    for r in report_rows:
        lines.append(
            f"{r['method']:16s} | {r['accuracy']:.3f} | {r['auc_confidence_negvar']:.3f} | "
            f"{r['var_wrong_minus_correct']:+.3f} | {r['top10_reward_wrong_rate']:.3f}"
        )

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {report_json}")
    print(f"Wrote: {report_csv}")
    print(f"Wrote: {report_txt}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
