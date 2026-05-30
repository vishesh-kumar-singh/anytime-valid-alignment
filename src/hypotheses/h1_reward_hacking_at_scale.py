"""H1: Reward Hacking Mitigation at Scale.

**Hypothesis**: As N grows, vanilla BoN's accuracy plateaus or drops due to
reward hacking, while uncertainty-aware methods (Sharpe, Linear, MoB)
maintain gains.  The reward hacking proxy (high-μ, high-σ² selections)
should increase with N for vanilla BoN but stay flat for Sharpe.

**Experiments**:
  - Sweep N ∈ {1, 4, 8, 16, 24} using offline precomputed candidates + scores
  - Track accuracy, reward hacking proxy, and variance gap vs N

**Plots**:
  - accuracy_vs_n.png
  - reward_hacking_vs_n.png
  - variance_gap_vs_n.png
  - combined_n_sweep.png
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..core.data import check_correctness
from ..core.method_runners import (
    run_bon_linear,
    run_bon_sharpe,
    run_mob_bon,
    run_vanilla_bon,
    run_wbon,
    run_wbon_sharpe,
    run_soft_bon,
    run_softbon_sharpe,
    run_softbon_linear,
)
from ..core.results import MethodResult, ResultsManager
from .base import Hypothesis



plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

METHOD_STYLE = {
    "vanilla_bon (ensemble)": {"color": "#e74c3c", "marker": "o", "ls": "-"},
    "bon_sharpe":             {"color": "#2ecc71", "marker": "s", "ls": "-"},
    "bon_linear":             {"color": "#3498db", "marker": "D", "ls": "-"},
    "soft_bon":               {"color": "#e67e22", "marker": "^", "ls": "--"},
    "softbon_sharpe":         {"color": "#9b59b6", "marker": "v", "ls": "--"},
    "softbon_linear":         {"color": "#1abc9c", "marker": "X", "ls": "--"},
    "majority_vote":          {"color": "#f1c40f", "marker": "*", "ls": "-"},
    "rm_within_majority":     {"color": "#34495e", "marker": "p", "ls": "-"},
    "mob_vanilla":            {"color": "#f39c12", "marker": "h", "ls": "--"},
    "mob_sharpe":             {"color": "#1abc9c", "marker": "H", "ls": "--"},
    "wbon_inv_var":           {"color": "#9b59b6", "marker": "d", "ls": "--"},
    "wbon_sharpe":            {"color": "#e67e22", "marker": "p", "ls": "--"},
}

def _extract_answer(text: str, dataset: str) -> Optional[str]:
    """Extract final answer from a response for comparison."""
    if dataset == "gsm8k":
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "").strip()
    elif dataset == "math500":
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].strip()
    return None


class H1RewardHackingAtScale(Hypothesis):
    name = "h1_reward_hacking_at_scale"
    category = "parallel"
    description = (
        "As N grows, vanilla BoN plateaus/drops due to reward hacking; "
        "uncertainty-aware methods maintain gains."
    )

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "1,4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        lin_c: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run N-sweep evaluation."""
        examples = self.load_candidates(candidates_path)
        rm_names, all_rm_scores = self.load_scores(score_paths)
        dataset = self.detect_dataset(candidates_path)
        n_values = [int(x.strip()) for x in eval_n.split(",")]

        print(f"[H1] N-sweep on {dataset}: N = {n_values}")
        print(f"[H1] {len(examples)} examples, {len(rm_names)} RMs: {rm_names}")

        results_by_n: Dict[str, Dict[str, float]] = {}
        hacking_by_n: Dict[str, Dict[str, float]] = {}
        vargap_by_n: Dict[str, Dict[str, float]] = {}

        for n in sorted(n_values):
            print(f"  Evaluating N={n}...")
            correct: Dict[str, int] = defaultdict(int)
            total: Dict[str, int] = defaultdict(int)
            reward_means_by_method: Dict[str, List[float]] = defaultdict(list)
            variances_by_method: Dict[str, List[float]] = defaultdict(list)
            correct_flags_by_method: Dict[str, List[bool]] = defaultdict(list)

            rng = np.random.default_rng(42)

            for ex in examples:
                idx = ex["example_index"]
                if idx not in all_rm_scores or len(all_rm_scores[idx]) != len(rm_names):
                    continue

                candidates = ex["candidates"][:n]
                gold = ex["ground_truth"]
                member_scores = [s[:n] for s in all_rm_scores[idx]]

                means, variances, _norm = self.ensemble_stats(
                    member_scores, norm_mode=variance_norm_mode
                )

                # Run methods
                methods = []

                # Individual RM baselines
                for j, rm in enumerate(rm_names):
                    m = run_vanilla_bon(candidates, member_scores[j], variances)
                    m.method = f"vanilla_bon ({rm})"
                    methods.append(m)

                # Ensemble methods
                m_ens = run_vanilla_bon(candidates, means, variances)
                m_ens.method = "vanilla_bon (ensemble)"
                methods.append(m_ens)
                methods.append(run_bon_sharpe(candidates, means, variances))
                methods.append(run_bon_linear(candidates, means, variances, c=lin_c))
                methods.append(run_mob_bon(candidates, means, variances, use_variance_weighting=False, rng=rng))
                methods.append(run_mob_bon(candidates, means, variances, use_variance_weighting=True, rng=rng))
                methods.append(run_wbon(candidates, member_scores))
                methods.append(run_wbon_sharpe(candidates, member_scores))
                
                # SoftBoN methods
                methods.append(run_soft_bon(candidates, means, variances, temperature=1.0, rng=rng))
                methods.append(run_softbon_sharpe(candidates, means, variances, eps=1e-8, temperature=1.0, rng=rng))
                methods.append(run_softbon_linear(candidates, means, variances, c=lin_c, temperature=1.0, rng=rng))
                
                # Self-Consistency methods
                answers = [_extract_answer(c, dataset) for c in candidates]
                answer_counts = Counter(a for a in answers if a is not None)
                
                total["majority_vote"] += 1
                total["rm_within_majority"] += 1
                total["rm_consistency_agree"] += 1
                
                if answer_counts:
                    majority_answer = answer_counts.most_common(1)[0][0]
                    
                    # 1. Majority Vote
                    majority_idx = next((i for i, a in enumerate(answers) if a == majority_answer), 0)
                    is_correct_maj = check_correctness(candidates[majority_idx], gold, dataset)
                    if is_correct_maj:
                        correct["majority_vote"] += 1
                    reward_means_by_method["majority_vote"].append(means[majority_idx])
                    variances_by_method["majority_vote"].append(variances[majority_idx])
                    correct_flags_by_method["majority_vote"].append(is_correct_maj)
                    
                    # 2. RM-within-majority
                    majority_indices = [i for i, a in enumerate(answers) if a == majority_answer]
                    majority_means = [means[i] for i in majority_indices]
                    best_in_majority = majority_indices[int(np.argmax(majority_means))]
                    is_correct_rm_maj = check_correctness(candidates[best_in_majority], gold, dataset)
                    if is_correct_rm_maj:
                        correct["rm_within_majority"] += 1
                    reward_means_by_method["rm_within_majority"].append(means[best_in_majority])
                    variances_by_method["rm_within_majority"].append(variances[best_in_majority])
                    correct_flags_by_method["rm_within_majority"].append(is_correct_rm_maj)
                    
                    # 3. RM-consistency agree
                    rm_idx = int(np.argmax(means))
                    rm_answer = answers[rm_idx]
                    selected = rm_idx if rm_answer == majority_answer else majority_idx
                    is_correct_agree = check_correctness(candidates[selected], gold, dataset)
                    if is_correct_agree:
                        correct["rm_consistency_agree"] += 1
                    reward_means_by_method["rm_consistency_agree"].append(means[selected])
                    variances_by_method["rm_consistency_agree"].append(variances[selected])
                    correct_flags_by_method["rm_consistency_agree"].append(is_correct_agree)
                else:
                    reward_means_by_method["majority_vote"].append(means[0])
                    variances_by_method["majority_vote"].append(variances[0])
                    correct_flags_by_method["majority_vote"].append(False)
                    reward_means_by_method["rm_within_majority"].append(means[0])
                    variances_by_method["rm_within_majority"].append(variances[0])
                    correct_flags_by_method["rm_within_majority"].append(False)
                    reward_means_by_method["rm_consistency_agree"].append(means[0])
                    variances_by_method["rm_consistency_agree"].append(variances[0])
                    correct_flags_by_method["rm_consistency_agree"].append(False)

                for m in methods:
                    is_correct = check_correctness(m.selected_answer, gold, dataset)
                    total[m.method] += 1
                    if is_correct:
                        correct[m.method] += 1
                    reward_means_by_method[m.method].append(m.reward_mean)
                    variances_by_method[m.method].append(m.reward_variance)
                    correct_flags_by_method[m.method].append(is_correct)

            # Compute metrics
            n_key = str(n)
            results_by_n[n_key] = {m: correct[m] / total[m] for m in total}

            hacking: Dict[str, float] = {}
            vargap: Dict[str, float] = {}
            for method in total:
                mu = np.array(reward_means_by_method[method])
                var = np.array(variances_by_method[method])
                y = np.array(correct_flags_by_method[method], dtype=int)

                # Top-10% reward wrong rate
                q90 = float(np.quantile(mu, 0.9)) if len(mu) > 0 else 0
                top = mu >= q90
                top_n = int(top.sum())
                wrong_top = int(((y == 0) & top).sum())
                hacking[method] = wrong_top / top_n if top_n > 0 else 0.0

                # Variance gap
                var_correct = float(var[y == 1].mean()) if (y == 1).sum() > 0 else 0
                var_wrong = float(var[y == 0].mean()) if (y == 0).sum() > 0 else 0
                vargap[method] = var_wrong - var_correct

            hacking_by_n[n_key] = hacking
            vargap_by_n[n_key] = vargap

        return {
            "hypothesis": self.name,
            "dataset": dataset,
            "n_values": n_values,
            "n_examples": len(examples),
            "rm_names": rm_names,
            "variance_norm_mode": variance_norm_mode,
            "results_by_n": results_by_n,
            "hacking_by_n": hacking_by_n,
            "vargap_by_n": vargap_by_n,
        }

    def plot(self, results: Dict[str, Any], manager: ResultsManager) -> List[Path]:
        dataset = results["dataset"].upper()
        n_values = results["n_values"]
        results_by_n = results["results_by_n"]
        hacking_by_n = results["hacking_by_n"]
        vargap_by_n = results["vargap_by_n"]

        all_methods = list(results_by_n[str(n_values[-1])].keys())
        styled_methods = [m for m in all_methods if m in METHOD_STYLE]
        plot_methods = styled_methods or all_methods[:7]

        paths: List[Path] = []

        def _plot_metric(data_by_n, ylabel, title, filename):
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in plot_methods:
                style = METHOD_STYLE.get(method, {"color": "#7f8c8d", "marker": "o", "ls": "-"})
                xs, ys = [], []
                for n in sorted(n_values):
                    val = data_by_n.get(str(n), {}).get(method)
                    if val is not None:
                        xs.append(n)
                        ys.append(val)
                if xs:
                    ax.plot(xs, ys, label=method, color=style["color"],
                            marker=style["marker"], linestyle=style["ls"],
                            markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xlabel("Number of Candidates (N)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {dataset}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values])
            ax.legend(loc="best", framealpha=0.9, fontsize=8)
            fig.tight_layout()
            p = manager.save_plot(fig, filename)
            plt.close(fig)
            return p

        paths.append(_plot_metric(results_by_n, "Accuracy", "Accuracy vs N", "accuracy_vs_n"))
        paths.append(_plot_metric(hacking_by_n, "Top-10% Reward Wrong Rate",
                                  "Reward Hacking Proxy vs N", "reward_hacking_vs_n"))
        paths.append(_plot_metric(vargap_by_n, "Var(wrong) − Var(correct)",
                                  "Variance Gap vs N", "variance_gap_vs_n"))

        # Combined 3-panel
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        for method in plot_methods:
            style = METHOD_STYLE.get(method, {"color": "#7f8c8d", "marker": "o", "ls": "-"})
            for ax_idx, (data, ylabel) in enumerate([
                (results_by_n, "Accuracy"),
                (hacking_by_n, "Top-10% Wrong Rate"),
                (vargap_by_n, "Var Gap"),
            ]):
                xs, ys = [], []
                for n in sorted(n_values):
                    val = data.get(str(n), {}).get(method)
                    if val is not None:
                        xs.append(n)
                        ys.append(val)
                if xs:
                    axes[ax_idx].plot(xs, ys, label=method, color=style["color"],
                                     marker=style["marker"], linestyle=style["ls"],
                                     markeredgecolor="white", markeredgewidth=0.8)
        for ax in axes:
            ax.set_xlabel("N")
            ax.set_xscale("log", base=2)
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values])
            ax.legend(fontsize=7, framealpha=0.8)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"Accuracy vs N — {dataset}")
        axes[1].set_ylabel("Top-10% Wrong Rate")
        axes[1].set_title(f"Reward Hacking — {dataset}")
        axes[2].set_ylabel("Var(wrong) − Var(correct)")
        axes[2].set_title(f"Variance Gap — {dataset}")
        fig.suptitle(f"H1: Reward Hacking at Scale — {dataset}", fontsize=14, y=1.02)
        fig.tight_layout()
        paths.append(manager.save_plot(fig, "combined_n_sweep"))
        plt.close(fig)

        return paths
