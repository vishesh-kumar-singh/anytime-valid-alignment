#!/usr/bin/env python3
""" evaluation of reward-variance normalization strategies.

This script reads completed JSONL run files (with candidate_member_scores),
reconstructs per-method selected model-score vectors, and compares uncertainty
quality under different normalization choices:
  - raw
  - zscore
  - robust_zscore
  - sigmoid_zscore
  - sigmoid_robust_zscore

Outputs per dataset:
  - normalization_metrics.csv
  - normalization_metrics.txt
  - normalization_rankings.txt
  - disagreement_calibration_<method>.png
  - disagreement_variance_boxplot.png
  - model_score_boxplot_raw_vs_norm.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-8)
    z = np.clip(x / t, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via pairwise wins, no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


@dataclass
class ModelStats:
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    iqr: np.ndarray


def fit_model_stats(all_model_scores: np.ndarray) -> ModelStats:
    # all_model_scores shape: (num_points, num_models)
    mean = np.mean(all_model_scores, axis=0)
    std = np.std(all_model_scores, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    median = np.median(all_model_scores, axis=0)
    q1 = np.percentile(all_model_scores, 25.0, axis=0)
    q3 = np.percentile(all_model_scores, 75.0, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr < 1e-8, 1.0, iqr)

    return ModelStats(mean=mean, std=std, median=median, iqr=iqr)


def normalize_vector(vec: np.ndarray, stats: ModelStats, method: str, temperature: float) -> np.ndarray:
    if method == "raw":
        return vec
    if method == "zscore":
        return (vec - stats.mean) / stats.std
    if method == "robust_zscore":
        return (vec - stats.median) / stats.iqr
    if method == "sigmoid_zscore":
        z = (vec - stats.mean) / stats.std
        return sigmoid(z, temperature)
    if method == "sigmoid_robust_zscore":
        z = (vec - stats.median) / stats.iqr
        return sigmoid(z, temperature)
    raise ValueError(f"Unknown normalization method: {method}")


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def get_unique_example_rows(rows: List[dict]) -> Dict[int, dict]:
    by_example: Dict[int, dict] = {}
    for r in rows:
        idx = int(r["example_index"])
        if idx not in by_example:
            by_example[idx] = r
    return by_example


def collect_all_candidate_model_scores(rows: List[dict]) -> np.ndarray:
    by_example = get_unique_example_rows(rows)
    vectors = []
    for ex in by_example.values():
        member_scores = ex.get("candidate_member_scores")
        if not member_scores:
            continue
        arr = np.asarray(member_scores, dtype=float)  # shape: (n_models, n_candidates)
        # Transpose to (n_candidates, n_models) and append.
        vectors.append(arr.T)

    if not vectors:
        raise ValueError("No candidate_member_scores found. Cannot run normalization eval.")
    return np.vstack(vectors)


def infer_selected_index(row: dict) -> int:
    means = np.asarray(row.get("candidate_means", []), dtype=float)
    variances = np.asarray(row.get("candidate_variances", []), dtype=float)
    target_mu = float(row.get("reward_mean", 0.0))
    target_var = float(row.get("reward_variance", 0.0))

    if means.size == 0:
        return 0

    # Match the selected candidate by nearest (mu, var) pair.
    dist = (means - target_mu) ** 2 + (variances - target_var) ** 2
    return int(np.argmin(dist))


def reconstruct_selected_model_vectors(rows: List[dict]) -> List[dict]:
    reconstructed = []
    for r in rows:
        member_scores = r.get("candidate_member_scores")
        if not member_scores:
            continue

        arr = np.asarray(member_scores, dtype=float)  # (n_models, n_candidates)
        idx = infer_selected_index(r)
        idx = int(np.clip(idx, 0, arr.shape[1] - 1))

        selected_vec = arr[:, idx]  # (n_models,)
        out = dict(r)
        out["selected_member_scores"] = selected_vec.tolist()
        reconstructed.append(out)

    if not reconstructed:
        raise ValueError("Could not reconstruct selected model-score vectors.")
    return reconstructed


def summarize_metrics(rows: List[dict], stats: ModelStats, temperature: float) -> Tuple[List[dict], Dict[Tuple[str, str], dict]]:
    methods = sorted({r["method"] for r in rows})
    norms = ["raw", "zscore", "robust_zscore", "sigmoid_zscore", "sigmoid_robust_zscore"]

    out_rows = []
    curve_data: Dict[Tuple[str, str], dict] = {}

    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        y = np.asarray([1 if r.get("is_correct") else 0 for r in method_rows], dtype=int)
        acc = float(np.mean(y))

        selected_matrix = np.asarray([r["selected_member_scores"] for r in method_rows], dtype=float)

        for norm in norms:
            norm_matrix = np.asarray(
                [normalize_vector(v, stats, norm, temperature) for v in selected_matrix],
                dtype=float,
            )
            var = np.var(norm_matrix, axis=1, ddof=0)

            # Confidence quality: low var should imply higher correctness.
            auc = auc_binary(-var, y)

            var_correct = float(np.mean(var[y == 1])) if np.any(y == 1) else float("nan")
            var_wrong = float(np.mean(var[y == 0])) if np.any(y == 0) else float("nan")
            delta = float(var_wrong - var_correct)

            q90 = float(np.quantile(var, 0.9))
            high_unc = var >= q90
            high_unc_acc = float(np.mean(y[high_unc])) if np.any(high_unc) else float("nan")

            # Reward-hacking proxy from model-agnostic selected reward_mean.
            mu = np.asarray([float(r["reward_mean"]) for r in method_rows], dtype=float)
            mu_q90 = float(np.quantile(mu, 0.9))
            top_reward = mu >= mu_q90
            top_wrong_rate = float(np.mean(1 - y[top_reward])) if np.any(top_reward) else float("nan")

            out_rows.append(
                {
                    "method": method,
                    "normalization": norm,
                    "accuracy": acc,
                    "auc_confidence_negvar": float(auc),
                    "var_correct_mean": var_correct,
                    "var_wrong_mean": var_wrong,
                    "var_wrong_minus_correct": delta,
                    "high_uncertainty_accuracy": high_unc_acc,
                    "top10_reward_wrong_rate": top_wrong_rate,
                }
            )

            # Calibration curve data
            q = np.quantile(var, np.linspace(0.0, 1.0, 11))
            xs = []
            ys = []
            for i in range(10):
                lo = q[i]
                hi = q[i + 1]
                mask = (var >= lo) & ((var <= hi) if i == 9 else (var < hi))
                if np.sum(mask) == 0:
                    continue
                xs.append(float(np.mean(var[mask])))
                ys.append(float(np.mean(y[mask])))
            curve_data[(method, norm)] = {"x": xs, "y": ys, "var": var}

    return out_rows, curve_data


def save_tables(dataset_name: str, rows: List[dict], out_dir: Path) -> None:
    csv_path = out_dir / f"{dataset_name}_normalization_metrics.csv"
    txt_path = out_dir / f"{dataset_name}_normalization_metrics.txt"
    rank_path = out_dir / f"{dataset_name}_normalization_rankings.txt"

    fields = [
        "method",
        "normalization",
        "accuracy",
        "auc_confidence_negvar",
        "var_correct_mean",
        "var_wrong_mean",
        "var_wrong_minus_correct",
        "high_uncertainty_accuracy",
        "top10_reward_wrong_rate",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append("Method | Norm | Acc | AUC(-var->correct) | Var_wrong-Var_correct | HighUncAcc | Top10RewardWrong")
    lines.append("-" * 120)
    for r in rows:
        lines.append(
            f"{r['method']:16s} | {r['normalization']:21s} | {r['accuracy']:.3f} | "
            f"{r['auc_confidence_negvar']:.3f} | {r['var_wrong_minus_correct']:+.3f} | "
            f"{r['high_uncertainty_accuracy']:.3f} | {r['top10_reward_wrong_rate']:.3f}"
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Ranking: best normalization per method by AUC(-var->correct)
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    rank_lines = ["Best normalization per method by AUC(-var->correct)", "-" * 70]
    for method in sorted(by_method):
        best = max(by_method[method], key=lambda x: x["auc_confidence_negvar"])
        rank_lines.append(
            f"{method:16s} -> {best['normalization']:21s} | "
            f"AUC={best['auc_confidence_negvar']:.3f} | "
            f"delta(var_wrong-var_correct)={best['var_wrong_minus_correct']:+.3f}"
        )

    with rank_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(rank_lines) + "\n")


def plot_for_dataset(dataset_name: str, rows: List[dict], curve_data: Dict[Tuple[str, str], dict], stats: ModelStats, all_scores: np.ndarray, out_dir: Path) -> None:
    methods = sorted({r["method"] for r in rows})
    norms = ["raw", "zscore", "robust_zscore", "sigmoid_zscore", "sigmoid_robust_zscore"]

    # 1) Per-method calibration curves across normalization choices.
    for method in methods:
        plt.figure(figsize=(8, 5))
        for norm in norms:
            c = curve_data.get((method, norm), {"x": [], "y": []})
            if not c["x"]:
                continue
            plt.plot(c["x"], c["y"], marker="o", linewidth=1.2, label=norm)
        plt.xlabel("Mean disagreement variance per bin")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name}: calibration by normalization - {method}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset_name}_calibration_{method}.png", dpi=160)
        plt.close()

    # 2) Variance distributions by normalization (aggregate over all methods/examples).
    var_by_norm = {n: [] for n in norms}
    for method in methods:
        for norm in norms:
            c = curve_data.get((method, norm))
            if c:
                var_by_norm[norm].extend(c["var"])

    plt.figure(figsize=(8, 5))
    plt.boxplot([var_by_norm[n] for n in norms], tick_labels=norms, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Disagreement variance")
    plt.title(f"{dataset_name}: disagreement variance distribution by normalization")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_disagreement_variance_boxplot.png", dpi=160)
    plt.close()

    # 3) Per-model score distributions raw vs normalized views using all candidates.
    z_all = (all_scores - stats.mean) / stats.std
    rz_all = (all_scores - stats.median) / stats.iqr

    n_models = all_scores.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    axes[0].boxplot([all_scores[:, j] for j in range(n_models)], showfliers=False)
    axes[0].set_title("Raw scores")
    axes[0].set_xlabel("Model index")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].boxplot([z_all[:, j] for j in range(n_models)], showfliers=False)
    axes[1].set_title("Z-score normalized")
    axes[1].set_xlabel("Model index")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].boxplot([rz_all[:, j] for j in range(n_models)], showfliers=False)
    axes[2].set_title("Robust z-score normalized")
    axes[2].set_xlabel("Model index")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"{dataset_name}: model-score scale alignment")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset_name}_model_score_boxplot_raw_vs_norm.png", dpi=160)
    plt.close(fig)


def run_one(input_path: Path, out_dir: Path, temperature: float) -> None:
    dataset_name = input_path.stem.replace(".jsonl", "")
    rows = load_rows(input_path)
    reconstructed = reconstruct_selected_model_vectors(rows)

    all_scores = collect_all_candidate_model_scores(rows)
    stats = fit_model_stats(all_scores)

    metric_rows, curve_data = summarize_metrics(reconstructed, stats, temperature)
    save_tables(dataset_name, metric_rows, out_dir)
    plot_for_dataset(dataset_name, metric_rows, curve_data, stats, all_scores, out_dir)

    print(f"[done] {dataset_name}")
    print(f"  metrics: {(out_dir / f'{dataset_name}_normalization_metrics.txt')}")
    print(f"  ranking: {(out_dir / f'{dataset_name}_normalization_rankings.txt')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for variance normalization strategies.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/normalization_eval",
        help="Output directory for plots and reports.",
    )
    parser.add_argument(
        "--sigmoid-temperature",
        type=float,
        default=1.0,
        help="Temperature T used for sigmoid(z/T).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in args.inputs:
        run_one(Path(p), out_dir, float(args.sigmoid_temperature))


if __name__ == "__main__":
    main()
