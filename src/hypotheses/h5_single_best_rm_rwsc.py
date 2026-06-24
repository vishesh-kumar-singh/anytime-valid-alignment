"""H6: Single-Best-RM Reward-Weighted Self-Consistency.

The central negative result of the parallel-ensembling study is that
*averaging* a heterogeneous bag of general-purpose reward models DILUTES the
one model that actually tracks mathematical correctness.  On MATH500 the
5-RM ensemble z-mean Best-of-N (67.4%) scores *below* the single strongest RM
(Skywork, 69.2%).

H6 follows that diagnosis to its conclusion: drop the ensemble average, keep
the single best reward model, and combine it with self-consistency at the
*answer* level (Reward-Weighted Self-Consistency) rather than the candidate
level.  Each unique final answer is scored by the sum of softmax-weighted RM
scores of the candidates that produced it; the highest-mass answer wins.

This is the strongest parallel method we found (~70.8% vs 69.4% for the
ensemble RWSC of H4).  The gain is small and well short of the 79.4% oracle
ceiling — which is precisely the point: with general-purpose RMs and a fixed
candidate set, parallel selection is saturated, motivating the move to
sequential / generative-verification methods (see FUTURE_WORK.md).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import Hypothesis, _extract_answer

# Preference order of "strongest" general-purpose RM (independent RewardBench
# ranking, NOT tuned on MATH500). The first match present in the ensemble wins.
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


class H5SingleBestRMRWSC(Hypothesis):
    name = "h5_single_best_rm_rwsc"
    category = "parallel"
    description = (
        "Single-best-RM Reward-Weighted Self-Consistency: drop the diluting "
        "ensemble average, weight answer clusters by the single strongest RM."
    )

    TEMPERATURE = 1.0

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
        n = len(candidates)
        if n == 0:
            return results

        best_j = _pick_best_rm(rm_names)
        raw = np.asarray(member_scores[best_j], dtype=float)
        # z-normalise this RM's scores across the candidate set, then softmax.
        z = (raw - raw.mean()) / (raw.std() + 1e-6)
        w = np.exp((z - z.max()) / self.TEMPERATURE)

        # --- Baseline: single-best-RM Best-of-N (candidate-level argmax) ---
        results["single_best_rm_bon"] = candidates[int(np.argmax(raw))]

        # --- Single-best-RM RWSC (answer-level aggregation) ---
        cluster_mass: Dict[str, float] = {}
        cluster_members: Dict[str, list] = {}
        for i in range(n):
            ans = _extract_answer(candidates[i], dataset)
            if ans is None:
                ans = f"__noans__{i}"  # unclustered singleton
            cluster_mass[ans] = cluster_mass.get(ans, 0.0) + float(w[i])
            cluster_members.setdefault(ans, []).append((i, raw[i]))

        best_ans = max(cluster_mass, key=cluster_mass.get)
        # Representative = highest-scoring candidate within the winning cluster.
        best_i = max(cluster_members[best_ans], key=lambda t: t[1])[0]
        results["single_best_rm_rwsc"] = candidates[best_i]

        return results
