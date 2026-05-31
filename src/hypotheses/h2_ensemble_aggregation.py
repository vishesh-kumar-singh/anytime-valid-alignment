"""H2: Ensemble Aggregation Methods.

Evaluates Mixture-of-Bests (MoB) and Weighted BoN (WBoN) variants
that aggregate across the RM ensemble in different ways.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.method_runners import (
    run_mob_bon,
    run_wbon,
    run_wbon_sharpe,
)
from .base import Hypothesis


class H2EnsembleAggregation(Hypothesis):
    name = "h2_ensemble_aggregation"
    category = "parallel"
    description = (
        "MoB and WBoN provide alternative ensemble aggregation strategies "
        "beyond simple mean/variance."
    )

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

        # MoB variants
        results["mob_vanilla"] = run_mob_bon(candidates, means, variances, use_variance_weighting=False, rng=rng).selected_answer
        results["mob_sharpe"] = run_mob_bon(candidates, means, variances, use_variance_weighting=True, rng=rng).selected_answer

        # WBoN variants
        results["wbon_inv_var"] = run_wbon(candidates, member_scores).selected_answer
        results["wbon_sharpe"] = run_wbon_sharpe(candidates, member_scores).selected_answer

        return results
