"""H6: Self-Consistency as a Free Reward Model.

**Hypothesis**: For tasks with verifiable answers (math), majority vote
over final answers provides a signal orthogonal to RM scoring.  Combining
self-consistency with RM scoring yields better selection than either alone.

**Experiments**:
  - Pure majority vote baseline
  - RM-within-majority: only let RM pick among majority-cluster candidates
  - Weighted hybrid: α·RM_score + (1−α)·consistency_score
  - RM-consistency agreement signal

**Plots**:
  - method_comparison.png
  - alpha_sweep.png
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..core.data import check_correctness
from ..core.results import ResultsManager
from .base import Hypothesis


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


class H6SelfConsistency(Hypothesis):
    name = "h6_self_consistency"
    category = "parallel"
    description = (
        "Self-consistency (majority vote) provides a free signal orthogonal "
        "to RM scoring; combining them yields better selection."
    )

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "24",
        variance_norm_mode: str = "robust_zscore",
        alpha_values: str = "0.0,0.2,0.4,0.6,0.8,1.0",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        examples = self.load_candidates(candidates_path)
        rm_names, all_rm_scores = self.load_scores(score_paths)
        dataset = self.detect_dataset(candidates_path)
        # Self-consistency needs all candidates; use the largest N available
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        n = max(n_values)
        alphas = [float(a) for a in alpha_values.split(",")]

        print(f"[H6] Self-consistency: N={n}, alphas={alphas}, {len(rm_names)} RMs")

        method_correct: Dict[str, int] = defaultdict(int)
        method_total: Dict[str, int] = defaultdict(int)
        alpha_correct: Dict[float, int] = defaultdict(int)
        alpha_total: Dict[float, int] = defaultdict(int)

        for ex in examples:
            idx = ex["example_index"]
            if idx not in all_rm_scores or len(all_rm_scores[idx]) != len(rm_names):
                continue
            candidates = ex["candidates"][:n]
            gold = ex["ground_truth"]
            member_scores = [s[:n] for s in all_rm_scores[idx]]
            means, variances, _ = self.ensemble_stats(
                member_scores, norm_mode=variance_norm_mode
            )

            # Extract final answers for self-consistency
            answers = [_extract_answer(c, dataset) for c in candidates]
            answer_counts = Counter(a for a in answers if a is not None)

            # ── 1. Pure majority vote ──
            method_total["majority_vote"] += 1
            if answer_counts:
                majority_answer = answer_counts.most_common(1)[0][0]
                majority_idx = next((i for i, a in enumerate(answers) if a == majority_answer), 0)
                is_correct = check_correctness(candidates[majority_idx], gold, dataset)
                method_correct["majority_vote"] += int(is_correct)

            # ── 2. Vanilla BoN (ensemble mean) ──
            rm_idx = int(np.argmax(means))
            is_correct = check_correctness(candidates[rm_idx], gold, dataset)
            method_correct["vanilla_bon (ensemble)"] += int(is_correct)
            method_total["vanilla_bon (ensemble)"] += 1

            # ── 2b. Individual RM baselines ──
            for j, rm in enumerate(rm_names):
                raw_scores = member_scores[j]
                rm_best_idx = int(np.argmax(raw_scores))
                is_correct = check_correctness(candidates[rm_best_idx], gold, dataset)
                method_correct[f"vanilla_bon ({rm})"] += int(is_correct)
                method_total[f"vanilla_bon ({rm})"] += 1

            # ── 3. RM-within-majority ──
            method_total["rm_within_majority"] += 1
            if answer_counts:
                majority_answer = answer_counts.most_common(1)[0][0]
                majority_indices = [i for i, a in enumerate(answers) if a == majority_answer]
                if majority_indices:
                    majority_means = [means[i] for i in majority_indices]
                    best_in_majority = majority_indices[int(np.argmax(majority_means))]
                    is_correct = check_correctness(candidates[best_in_majority], gold, dataset)
                    method_correct["rm_within_majority"] += int(is_correct)

            # ── 4. RM-consistency agreement ──
            method_total["rm_consistency_agree"] += 1
            if answer_counts:
                rm_answer = _extract_answer(candidates[rm_idx], dataset)
                majority_answer = answer_counts.most_common(1)[0][0]
                if rm_answer == majority_answer:
                    selected = rm_idx
                else:
                    selected = majority_idx
                is_correct = check_correctness(candidates[selected], gold, dataset)
                method_correct["rm_consistency_agree"] += int(is_correct)

            # ── 5. Weighted hybrid sweep ──
            consistency_scores = []
            for i, ans in enumerate(answers):
                if ans is not None and ans in answer_counts:
                    consistency_scores.append(answer_counts[ans] / n)
                else:
                    consistency_scores.append(0.0)

            for alpha in alphas:
                hybrid_scores = [
                    alpha * means[i] + (1 - alpha) * consistency_scores[i]
                    for i in range(len(candidates))
                ]
                best_hybrid = int(np.argmax(hybrid_scores))
                is_correct = check_correctness(candidates[best_hybrid], gold, dataset)
                alpha_correct[alpha] += int(is_correct)
                alpha_total[alpha] += 1

        accuracy_by_method = {m: method_correct[m] / method_total[m] for m in method_total}
        accuracy_by_alpha = {a: alpha_correct[a] / alpha_total[a] for a in alphas if alpha_total[a] > 0}

        return {
            "hypothesis": self.name,
            "dataset": dataset,
            "n": n,
            "n_examples": len(examples),
            "accuracy_by_method": accuracy_by_method,
            "accuracy_by_alpha": accuracy_by_alpha,
            "alphas": alphas,
        }

    def plot(self, results: Dict[str, Any], manager: ResultsManager) -> List[Path]:
        dataset = results["dataset"].upper()
        paths: List[Path] = []

        # 1. Method comparison bar chart
        acc = results["accuracy_by_method"]
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = sorted(acc.keys())
        vals = [acc[m] for m in methods]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        ax.barh(methods, vals, color=colors[:len(methods)], alpha=0.85,
                edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=10)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Self-Consistency Methods — {dataset}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        paths.append(manager.save_plot(fig, "method_comparison"))
        plt.close(fig)

        # 2. Alpha sweep
        alpha_acc = results["accuracy_by_alpha"]
        if alpha_acc:
            fig, ax = plt.subplots(figsize=(7, 5))
            xs = sorted(alpha_acc.keys())
            ys = [alpha_acc[a] for a in xs]
            ax.plot(xs, ys, marker="o", color="#2ecc71", linewidth=2)
            ax.set_xlabel("α (RM weight)")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Hybrid Score: α·RM + (1−α)·Consistency — {dataset}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            paths.append(manager.save_plot(fig, "alpha_sweep"))
            plt.close(fig)

        return paths
