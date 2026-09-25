"""H9: SPRT-RWSC — Sequential Probability Ratio Test for Reward-Weighted Self-Consistency.

Applies Wald's Sequential Probability Ratio Test (SPRT) to the early stopping
problem in test-time alignment.  Instead of an ad-hoc confidence threshold τ
(as in H7), we accumulate a *log-likelihood ratio* that measures the statistical
evidence for the leading answer cluster being dominant.  The SPRT stopping
boundaries are derived from user-specified Type-I (α) and Type-II (β) error
rates, yielding *provably optimal* expected stopping times.

Three variants:
  1. **Vanilla SPRT-RWSC**: Binary evidence (does new candidate agree with leader?)
  2. **E-SPRT-RWSC** (Ensemble-weighted): Weights each evidence increment by
     inverse ensemble variance — high-agreement (low-variance) candidates
     contribute more evidence, naturally guarding against reward-hacked outliers.
  3. **AB-SPRT-RWSC** (Adaptive Burn-in): Estimates p₁ from a short burn-in
     phase, making the test prompt-adaptive.

Mathematical Foundation:
  Under H₁ (leader is correct): new candidates agree with probability p₁ > 0.5.
  Under H₀ (no convergence):  new candidates agree with probability p₀ ≤ 0.5.

  Log-likelihood ratio after step n:
    Λₙ = Σᵢ wᵢ · [Xᵢ log(p₁/p₀) + (1-Xᵢ) log((1-p₁)/(1-p₀))]

  Stopping boundaries (Wald):
    A = log((1-β)/α)   → stop and accept leader
    B = log(β/(1-α))   → reset (leader likely wrong)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .base import EvaluationPipeline, _extract_answer


# ────────────────────────────────────────────────────────────────────
# Pipeline class
# ────────────────────────────────────────────────────────────────────

# Preference order of "strongest" general-purpose RM (reused from H7)
_STRONGEST_RM_PREFERENCE = (
    "FsfairX",
    "Llama-3-OffsetBias",
    "reward-model-deberta",
)

def _pick_best_rm(rm_names: List[str]) -> int:
    for pref in _STRONGEST_RM_PREFERENCE:
        for j, name in enumerate(rm_names):
            if pref.lower() in name.lower():
                return j
    return 0


class SPRTPipeline(EvaluationPipeline):
    name = "sprt_rwsc"
    category = "parallel"  # runs on cached parallel data
    description = (
        "SPRT-RWSC: Sequential Probability Ratio Test for early stopping "
        "with provable error control, ensemble-weighted evidence, and "
        "adaptive burn-in prior estimation."
    )

    TEMPERATURE = 1.0

    def __init__(self, use_ensemble=False):
        super().__init__()
        self.steps_used = defaultdict(list)
        self.use_ensemble = use_ensemble

    def evaluate_methods(
        self,
        candidates: List[str],
        means: List[float],
        variances: List[float],
        member_scores: List[List[float]],
        rm_names: List[str],
        dataset: str,
        rng: np.random.Generator,
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        N = len(candidates)
        if N < 4:
            return results

        means_arr = np.array(means, dtype=float)
        vars_arr = np.array(variances, dtype=float)

        best_j = _pick_best_rm(rm_names)
        single_rm_scores = np.asarray(member_scores[best_j], dtype=float)

        from .base import get_semantic_cluster_ids
        from ..methods.sequential.sprt import _sprt_stop

        global_cluster_ids = get_semantic_cluster_ids(candidates, dataset)

        ab_configs = [(0.10, 0.20)]
        burnin_sizes = [2, 4, 8, 16, 32]

        for alpha, beta in ab_configs:
            for burnin in burnin_sizes:
                # Vanilla SPRT-RWSC
                tag = f"sprt_rwsc_a{alpha}_b{beta}_n0{burnin}"
                res = _sprt_stop(
                    candidates=candidates,
                    raw_scores=single_rm_scores,
                    variances=None,
                    global_cluster_ids=global_cluster_ids,
                    alpha=alpha,
                    beta=beta,
                    burnin=burnin,
                    p0_mode="uniform",
                    use_ensemble_weighting=False,
                    temperature=self.TEMPERATURE,
                    agg_mode="rwsc",
                )
                results[tag] = candidates[res["best_idx"]]
                self.steps_used[tag].append(res["stopped_at"])

                # Vanilla SPRT-Borda
                tag_borda = f"sprt_borda_a{alpha}_b{beta}_n0{burnin}"
                res_borda = _sprt_stop(
                    candidates=candidates,
                    raw_scores=single_rm_scores,
                    variances=None,
                    global_cluster_ids=global_cluster_ids,
                    alpha=alpha,
                    beta=beta,
                    burnin=burnin,
                    p0_mode="uniform",
                    use_ensemble_weighting=False,
                    temperature=self.TEMPERATURE,
                    agg_mode="borda",
                )
                results[tag_borda] = candidates[res_borda["best_idx"]]
                self.steps_used[tag_borda].append(res_borda["stopped_at"])

                # Ensemble variants
                if self.use_ensemble:
                    tag_ens = f"ens_sprt_rwsc_a{alpha}_b{beta}_n0{burnin}"
                    res_ens = _sprt_stop(
                        candidates=candidates,
                        raw_scores=means_arr,
                        variances=vars_arr,
                        global_cluster_ids=global_cluster_ids,
                        alpha=alpha,
                        beta=beta,
                        burnin=burnin,
                        p0_mode="uniform",
                        use_ensemble_weighting=True,
                        temperature=self.TEMPERATURE,
                        agg_mode="rwsc",
                    )
                    results[tag_ens] = candidates[res_ens["best_idx"]]
                    self.steps_used[tag_ens].append(res_ens["stopped_at"])

                    tag_ens_borda = f"ens_sprt_borda_a{alpha}_b{beta}_n0{burnin}"
                    res_ens_borda = _sprt_stop(
                        candidates=candidates,
                        raw_scores=means_arr,
                        variances=vars_arr,
                        global_cluster_ids=global_cluster_ids,
                        alpha=alpha,
                        beta=beta,
                        burnin=burnin,
                        p0_mode="uniform",
                        use_ensemble_weighting=True,
                        temperature=self.TEMPERATURE,
                        agg_mode="borda",
                    )
                    results[tag_ens_borda] = candidates[res_ens_borda["best_idx"]]
                    self.steps_used[tag_ens_borda].append(res_ens_borda["stopped_at"])

        return results

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Override run to only use the maximum N (since early stopping evaluates up to max budget)."""
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)

        print(f"\n[{self.name}] Overriding eval_n to {max_n} (early stopping evaluates up to max budget).")

        # Reset steps tracking
        self.steps_used.clear()

        # Run base evaluation
        results = super().run(
            candidates_path=candidates_path,
            score_paths=score_paths,
            eval_n=str(max_n),
            variance_norm_mode=variance_norm_mode,
            **kwargs
        )

        # Print average steps (cost analysis)
        print(f"\n{'=' * 70}")
        print(f"  SPRT-RWSC: Average Candidates Generated (Cost Analysis)")
        print(f"{'=' * 70}")

        avg_budget = {}

        # Group methods
        sprt_methods = sorted([m for m in self.steps_used if m.startswith("sprt_") and "borda" not in m])
        ens_sprt_methods = sorted([m for m in self.steps_used if m.startswith("ens_sprt_") and "borda" not in m])
        sprt_borda = sorted([m for m in self.steps_used if m.startswith("sprt_borda")])
        ens_sprt_borda = sorted([m for m in self.steps_used if m.startswith("ens_sprt_borda")])

        for group_name, methods in [
            ("Vanilla SPRT-RWSC (Single RM)", sprt_methods),
            ("Vanilla SPRT-Borda (Single RM)", sprt_borda),
            ("Ensemble SPRT-RWSC (Variance-Weighted)", ens_sprt_methods),
            ("Ensemble SPRT-Borda (Variance-Weighted)", ens_sprt_borda),
        ]:
            if methods:
                print(f"\n  {group_name}:")
                print(f"  {'-' * 60}")
                for method_name in methods:
                    steps = self.steps_used[method_name]
                    avg_steps = np.mean(steps)
                    std_steps = np.std(steps)
                    avg_budget[method_name] = {"mean": float(avg_steps), "std": float(std_steps)}
                    print(f"    {method_name:<40s}: {avg_steps:6.2f} ± {std_steps:5.2f}")

        print(f"\n{'=' * 70}\n")
        
        results["average_budget"] = avg_budget
        return results


