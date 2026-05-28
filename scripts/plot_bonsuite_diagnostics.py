import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(jsonl_path: Path):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {jsonl_path}")
    return rows


def build_method_series(rows):
    per_method = defaultdict(list)
    for r in rows:
        per_method[r["method"]].append(r)

    for method in per_method:
        per_method[method] = sorted(per_method[method], key=lambda x: x["example_index"])
    return per_method


def build_disagreement_series(rows):
    by_example = {}
    for r in rows:
        idx = int(r["example_index"])
        if idx not in by_example:
            by_example[idx] = r.get("candidate_variances", [])

    xs = sorted(by_example.keys())
    mean_var = []
    max_var = []
    for x in xs:
        vals = by_example[x]
        if not vals:
            mean_var.append(0.0)
            max_var.append(0.0)
            continue
        arr = np.asarray(vals, dtype=float)
        mean_var.append(float(np.mean(arr)))
        max_var.append(float(np.max(arr)))
    return xs, mean_var, max_var


def extract_member_score_table(rows) -> Tuple[List[str], np.ndarray]:
    """Extract per-example, per-model mean scores over candidates.

    Returns:
      model_names, table with shape (n_examples, n_models)
    """
    by_example: Dict[int, dict] = {}
    for r in rows:
        idx = int(r["example_index"])
        if idx not in by_example:
            by_example[idx] = r

    ordered_examples = [by_example[k] for k in sorted(by_example.keys())]
    model_names: List[str] = []
    vectors: List[List[float]] = []

    for ex in ordered_examples:
        member_scores = ex.get("candidate_member_scores")
        if not member_scores:
            continue
        if not model_names:
            model_names = ex.get("reward_model_names") or [f"model_{i}" for i in range(len(member_scores))]

        per_model_means = []
        for score_list in member_scores:
            arr = np.asarray(score_list, dtype=float)
            per_model_means.append(float(np.mean(arr)))
        vectors.append(per_model_means)

    if not vectors:
        return [], np.zeros((0, 0), dtype=float)

    return model_names, np.asarray(vectors, dtype=float)


def plot_member_means_over_examples(model_names: List[str], table: np.ndarray, out_dir: Path):
    if table.size == 0:
        return
    plt.figure(figsize=(11, 6))
    xs = np.arange(table.shape[0])
    for j, name in enumerate(model_names):
        plt.plot(xs, table[:, j], linewidth=1.2, alpha=0.85, label=name)
    plt.xlabel("Example index")
    plt.ylabel("Per-example model mean score")
    plt.title("Reward model score profiles across examples")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "rm_model_means_by_example.png", dpi=180)
    plt.close()


def plot_member_boxplot(model_names: List[str], table: np.ndarray, out_dir: Path):
    if table.size == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot([table[:, j] for j in range(table.shape[1])], tick_labels=model_names, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Per-example model mean score")
    plt.title("Distribution of reward-model scores")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rm_model_score_boxplot.png", dpi=180)
    plt.close()


def plot_member_correlation_heatmap(model_names: List[str], table: np.ndarray, out_dir: Path):
    if table.size == 0 or table.shape[0] < 2:
        return
    corr = np.corrcoef(table, rowvar=False)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(model_names)), model_names, rotation=20, ha="right")
    plt.yticks(range(len(model_names)), model_names)
    plt.title("Correlation between reward models")
    plt.tight_layout()
    plt.savefig(out_dir / "rm_model_correlation_heatmap.png", dpi=180)
    plt.close()


def save_summary(rows, out_dir: Path):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    out_path = out_dir / "method_summary_selected_stats.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("method,n_rows,avg_selected_mu,avg_selected_var\n")
        for method, vals in sorted(by_method.items()):
            mus = [float(v["reward_mean"]) for v in vals]
            vars_ = [float(v["reward_variance"]) for v in vals]
            f.write(
                f"{method},{len(vals)},{np.mean(mus):.6f},{np.mean(vars_):.6f}\n"
            )


def plot_selected_variance(per_method, out_dir: Path):
    plt.figure(figsize=(10, 6))
    for method, vals in sorted(per_method.items()):
        x = [int(v["example_index"]) for v in vals]
        y = [float(v["reward_variance"]) for v in vals]
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=method)

    plt.xlabel("Example index")
    plt.ylabel("Selected candidate variance")
    plt.title("Selected variance by example and method")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "selected_variance_by_example.png", dpi=180)
    plt.close()


def plot_selected_mu(per_method, out_dir: Path):
    plt.figure(figsize=(10, 6))
    for method, vals in sorted(per_method.items()):
        x = [int(v["example_index"]) for v in vals]
        y = [float(v["reward_mean"]) for v in vals]
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=method)

    plt.xlabel("Example index")
    plt.ylabel("Selected candidate mean reward (mu)")
    plt.title("Selected mu by example and method")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "selected_mu_by_example.png", dpi=180)
    plt.close()


def plot_disagreement(xs, mean_var, max_var, out_dir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(xs, mean_var, marker="o", linewidth=1.5, markersize=3, label="Mean candidate variance")
    plt.plot(xs, max_var, marker="o", linewidth=1.5, markersize=3, label="Max candidate variance")
    plt.xlabel("Example index")
    plt.ylabel("Variance")
    plt.title("RM disagreement profile per example")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rm_disagreement_profile.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot BoN-suite diagnostics from JSONL output.")
    parser.add_argument("--input", required=True, help="Path to JSONL results file")
    parser.add_argument("--out-dir", default="runs/plots", help="Output directory for plots")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(in_path)
    per_method = build_method_series(rows)
    xs, mean_var, max_var = build_disagreement_series(rows)
    model_names, member_table = extract_member_score_table(rows)

    save_summary(rows, out_dir)
    plot_selected_variance(per_method, out_dir)
    plot_selected_mu(per_method, out_dir)
    plot_disagreement(xs, mean_var, max_var, out_dir)
    plot_member_means_over_examples(model_names, member_table, out_dir)
    plot_member_boxplot(model_names, member_table, out_dir)
    plot_member_correlation_heatmap(model_names, member_table, out_dir)

    print(f"Saved diagnostics to {out_dir}")
    print(" - method_summary_selected_stats.csv")
    print(" - selected_variance_by_example.png")
    print(" - selected_mu_by_example.png")
    print(" - rm_disagreement_profile.png")
    if member_table.size > 0:
        print(" - rm_model_means_by_example.png")
        print(" - rm_model_score_boxplot.png")
        print(" - rm_model_correlation_heatmap.png")


if __name__ == "__main__":
    main()
