"""H9: Difficulty-Stratified Analysis.

**Hypothesis**: Uncertainty-aware methods help most on medium-difficulty
problems where the RM is uncertain but not random.  Averaging across all
difficulties masks this heterogeneous effect.

**Experiments**:
  - Estimate per-prompt difficulty via candidate answer agreement and RM
    score spread
  - Stratify results into easy / medium / hard buckets
  - Report per-stratum accuracy for each method

**Plots**:
  - stratified_accuracy.png
  - difficulty_distribution.png
  - method_advantage_by_difficulty.png
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..core.data import check_correctness
from ..core.method_runners import (
    run_bon_sharpe, 
    run_vanilla_bon, 
    run_bon_linear,
    run_soft_bon,
    run_softbon_sharpe,
    run_softbon_linear,
    run_mob_bon,
    run_wbon,
    run_wbon_sharpe,
)
from ..core.results import ResultsManager
from .base import Hypothesis


def _extract_answer(text: str, dataset: str) -> Optional[str]:
    """Extract final answer from response text."""
    if dataset == "gsm8k":
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        return match.group(1).replace(",", "").strip() if match else None
    elif dataset == "math500":
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        return matches[-1].strip() if matches else None
    return None


def _difficulty_score(
    candidates: List[str],
    means: List[float],
    dataset: str,
) -> float:
    """Estimate prompt difficulty from candidate statistics.

    Uses two signals:
      1. Answer diversity: fraction of distinct answers among candidates
      2. Score spread: std of ensemble mean scores

    High diversity + high spread → hard.  Low diversity + low spread → easy.
    Returns a score in [0, 1] where 1 = hardest.
    """
    # Answer agreement
    answers = [_extract_answer(c, dataset) for c in candidates]
    valid = [a for a in answers if a is not None]
    if valid:
        counts = Counter(valid)
        majority_frac = counts.most_common(1)[0][1] / len(valid)
        diversity = 1.0 - majority_frac
    else:
        diversity = 1.0

    # Score spread
    spread = float(np.std(means)) if len(means) > 1 else 0.0
    max_spread = 2.0  # normalise roughly
    norm_spread = min(spread / max_spread, 1.0)

    return 0.5 * diversity + 0.5 * norm_spread


class H9DifficultyStratified(Hypothesis):
    name = "h9_difficulty_stratified"
    category = "parallel"
    description = (
        "Uncertainty-aware methods help most on medium-difficulty problems; "
        "stratified analysis reveals heterogeneous effects masked by averaging."
    )

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "8",
        variance_norm_mode: str = "robust_zscore",
        n_strata: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        examples = self.load_candidates(candidates_path)
        rm_names, all_rm_scores = self.load_scores(score_paths)
        dataset = self.detect_dataset(candidates_path)
        n = max(int(x) for x in eval_n.split(","))

        print(f"[H9] Difficulty-stratified: N={n}, {n_strata} strata")

        # First pass: compute difficulty scores
        per_example: List[Dict[str, Any]] = []
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

            diff = _difficulty_score(candidates, means, dataset)

            # Run methods
            m_van = run_vanilla_bon(candidates, means, variances)
            m_sh = run_bon_sharpe(candidates, means, variances)
            m_lin = run_bon_linear(candidates, means, variances, c=1.0)
            
            rng = np.random.default_rng(42)
            m_soft = run_soft_bon(candidates, means, variances, temperature=1.0, rng=rng)
            m_soft_sh = run_softbon_sharpe(candidates, means, variances, eps=1e-8, temperature=1.0, rng=rng)
            m_soft_lin = run_softbon_linear(candidates, means, variances, c=1.0, temperature=1.0, rng=rng)
            
            m_mob_van = run_mob_bon(candidates, means, variances, use_variance_weighting=False, rng=rng)
            m_mob_sh = run_mob_bon(candidates, means, variances, use_variance_weighting=True, rng=rng)
            m_wbon = run_wbon(candidates, member_scores)
            m_wbon_sh = run_wbon_sharpe(candidates, member_scores)

            # Self-consistency
            answers = [_extract_answer(c, dataset) for c in candidates]
            answer_counts = Counter(a for a in answers if a is not None)
            
            maj_correct = False
            rm_maj_correct = False
            if answer_counts:
                majority_answer = answer_counts.most_common(1)[0][0]
                majority_idx = next((i for i, a in enumerate(answers) if a == majority_answer), 0)
                maj_correct = check_correctness(candidates[majority_idx], gold, dataset)
                
                majority_indices = [i for i, a in enumerate(answers) if a == majority_answer]
                majority_means = [means[i] for i in majority_indices]
                best_in_majority = majority_indices[int(np.argmax(majority_means))]
                rm_maj_correct = check_correctness(candidates[best_in_majority], gold, dataset)

            rm_corrects = {}
            for j, rm in enumerate(rm_names):
                m_rm = run_vanilla_bon(candidates, member_scores[j], variances)
                rm_corrects[f"vanilla_bon ({rm})"] = check_correctness(m_rm.selected_answer, gold, dataset)

            ex_dict = {
                "difficulty": diff,
                "vanilla_correct": check_correctness(m_van.selected_answer, gold, dataset),
                "sharpe_correct": check_correctness(m_sh.selected_answer, gold, dataset),
                "linear_correct": check_correctness(m_lin.selected_answer, gold, dataset),
                "soft_correct": check_correctness(m_soft.selected_answer, gold, dataset),
                "soft_sharpe_correct": check_correctness(m_soft_sh.selected_answer, gold, dataset),
                "soft_linear_correct": check_correctness(m_soft_lin.selected_answer, gold, dataset),
                "mob_vanilla_correct": check_correctness(m_mob_van.selected_answer, gold, dataset),
                "mob_sharpe_correct": check_correctness(m_mob_sh.selected_answer, gold, dataset),
                "wbon_correct": check_correctness(m_wbon.selected_answer, gold, dataset),
                "wbon_sharpe_correct": check_correctness(m_wbon_sh.selected_answer, gold, dataset),
                "majority_correct": maj_correct,
                "rm_majority_correct": rm_maj_correct,
            }
            ex_dict.update(rm_corrects)
            per_example.append(ex_dict)

        # Stratify by difficulty
        difficulties = np.array([e["difficulty"] for e in per_example])
        thresholds = np.quantile(difficulties, np.linspace(0, 1, n_strata + 1))
        stratum_labels = ["easy", "medium", "hard"] if n_strata == 3 else \
            [f"stratum_{i}" for i in range(n_strata)]

        stratified: Dict[str, Dict[str, Any]] = {}
        for s_idx in range(n_strata):
            lo, hi = thresholds[s_idx], thresholds[s_idx + 1]
            if s_idx == n_strata - 1:
                mask = (difficulties >= lo) & (difficulties <= hi)
            else:
                mask = (difficulties >= lo) & (difficulties < hi)

            stratum_examples = [e for e, m in zip(per_example, mask) if m]
            label = stratum_labels[s_idx] if s_idx < len(stratum_labels) else f"stratum_{s_idx}"

            if stratum_examples:
                s_dict = {
                    "n_examples": len(stratum_examples),
                    "difficulty_range": [float(lo), float(hi)],
                    "vanilla_bon": float(np.mean([e["vanilla_correct"] for e in stratum_examples])),
                    "bon_sharpe": float(np.mean([e["sharpe_correct"] for e in stratum_examples])),
                    "bon_linear": float(np.mean([e["linear_correct"] for e in stratum_examples])),
                    "soft_bon": float(np.mean([e["soft_correct"] for e in stratum_examples])),
                    "softbon_sharpe": float(np.mean([e["soft_sharpe_correct"] for e in stratum_examples])),
                    "softbon_linear": float(np.mean([e["soft_linear_correct"] for e in stratum_examples])),
                    "mob_vanilla": float(np.mean([e["mob_vanilla_correct"] for e in stratum_examples])),
                    "mob_sharpe": float(np.mean([e["mob_sharpe_correct"] for e in stratum_examples])),
                    "wbon": float(np.mean([e["wbon_correct"] for e in stratum_examples])),
                    "wbon_sharpe": float(np.mean([e["wbon_sharpe_correct"] for e in stratum_examples])),
                    "majority_vote": float(np.mean([e["majority_correct"] for e in stratum_examples])),
                    "rm_within_majority": float(np.mean([e["rm_majority_correct"] for e in stratum_examples])),
                }
                for rm in rm_names:
                    rm_key = f"vanilla_bon ({rm})"
                    s_dict[rm_key] = float(np.mean([e[rm_key] for e in stratum_examples]))
                stratified[label] = s_dict

        # Overall
        overall = {
            "vanilla_bon": float(np.mean([e["vanilla_correct"] for e in per_example])),
            "bon_sharpe": float(np.mean([e["sharpe_correct"] for e in per_example])),
            "bon_linear": float(np.mean([e["linear_correct"] for e in per_example])),
            "soft_bon": float(np.mean([e["soft_correct"] for e in per_example])),
            "softbon_sharpe": float(np.mean([e["soft_sharpe_correct"] for e in per_example])),
            "softbon_linear": float(np.mean([e["soft_linear_correct"] for e in per_example])),
            "mob_vanilla": float(np.mean([e["mob_vanilla_correct"] for e in per_example])),
            "mob_sharpe": float(np.mean([e["mob_sharpe_correct"] for e in per_example])),
            "wbon": float(np.mean([e["wbon_correct"] for e in per_example])),
            "wbon_sharpe": float(np.mean([e["wbon_sharpe_correct"] for e in per_example])),
            "majority_vote": float(np.mean([e["majority_correct"] for e in per_example])),
            "rm_within_majority": float(np.mean([e["rm_majority_correct"] for e in per_example])),
        }
        for rm in rm_names:
            rm_key = f"vanilla_bon ({rm})"
            overall[rm_key] = float(np.mean([e[rm_key] for e in per_example]))

        return {
            "hypothesis": self.name,
            "dataset": dataset,
            "n": n,
            "n_examples": len(per_example),
            "n_strata": n_strata,
            "stratified": stratified,
            "overall": overall,
            "accuracy_by_method": overall,
            "difficulty_stats": {
                "mean": float(np.mean(difficulties)),
                "std": float(np.std(difficulties)),
                "min": float(np.min(difficulties)),
                "max": float(np.max(difficulties)),
            },
        }

    def plot(self, results: Dict[str, Any], manager: ResultsManager) -> List[Path]:
        dataset = results["dataset"].upper()
        stratified = results["stratified"]
        paths: List[Path] = []

        methods = ["vanilla_bon", "bon_sharpe", "majority_vote", "rm_within_majority"]
        strata = list(stratified.keys())
        method_colors = {"vanilla_bon": "#e74c3c", "bon_sharpe": "#2ecc71", "majority_vote": "#f1c40f", "rm_within_majority": "#34495e"}

        # 1. Stratified accuracy grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(strata))
        width = 0.25
        for i, method in enumerate(methods):
            vals = [stratified[s].get(method, 0) for s in strata]
            ax.bar(x + i * width, vals, width, label=method,
                   color=method_colors.get(method, "#7f8c8d"), alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width)
        stratum_labels = [f"{s}\n(n={stratified[s]['n_examples']})" for s in strata]
        ax.set_xticklabels(stratum_labels)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy by Difficulty Stratum — {dataset}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        paths.append(manager.save_plot(fig, "stratified_accuracy"))
        plt.close(fig)

        # 2. Sharpe advantage over vanilla by stratum
        fig, ax = plt.subplots(figsize=(8, 5))
        advantages = []
        for s in strata:
            adv = stratified[s].get("bon_sharpe", 0) - stratified[s].get("vanilla_bon", 0)
            advantages.append(adv)
        colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in advantages]
        ax.bar(strata, advantages, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Sharpe − Vanilla Accuracy")
        ax.set_title(f"Sharpe BoN Advantage by Difficulty — {dataset}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        paths.append(manager.save_plot(fig, "method_advantage_by_difficulty"))
        plt.close(fig)

        return paths
