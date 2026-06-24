"""H1: Uncertainty-Aware BoN Methods.

Evaluates Vanilla BoN (ensemble), Sharpe BoN, Linear BoN, and
SoftBoN variants (Vanilla, Sharpe, Linear).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.method_runners import (
    run_bon_linear,
    run_bon_sharpe,
    run_soft_bon,
    run_softbon_linear,
    run_softbon_sharpe,
    run_vanilla_bon,
)
from .base import Hypothesis


class H1UncertaintyAware(Hypothesis):
    name = "h1_uncertainty_aware"
    category = "parallel"
    description = (
        "Uncertainty-aware BoN methods (Sharpe, Linear, SoftBoN) resist "
        "reward hacking better than vanilla argmax."
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

        # Deterministic methods
        results["bon_sharpe"] = run_bon_sharpe(candidates, means, variances).selected_answer
        results["bon_linear"] = run_bon_linear(candidates, means, variances, c=1.0).selected_answer

        # Probabilistic methods
        results["soft_bon"] = run_soft_bon(candidates, means, variances, temperature=1.0, rng=rng).selected_answer
        results["softbon_sharpe"] = run_softbon_sharpe(candidates, means, variances, eps=1e-8, temperature=1.0, rng=rng).selected_answer
        results["softbon_linear"] = run_softbon_linear(candidates, means, variances, c=1.0, temperature=1.0, rng=rng).selected_answer

        return results
