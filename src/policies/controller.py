"""
Adaptive inference controller.

Decides which inference strategy to use based on reward model uncertainty:
  - Low uncertainty  → Parallel BoN (fast, reliable)
  - Medium uncertainty → Sequential RF-SeqBoN (slower, filters bad answers)
  - High uncertainty → Early stop (RM is unreliable, use safest candidate)

Also provides advanced candidate selection strategies:
  - Thompson Sampling: sample from N(μ, σ²), pick max
  - Lower Confidence Bound (LCB): pick argmax(μ - z·σ)
  - CVaR (Conditional Value at Risk): minimize tail risk
"""

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import numpy as np


Decision = Literal["bon", "sequential", "early_stop"]


@dataclass
class BudgetConfig:
    """Uncertainty thresholds for routing decisions."""
    u_low: float = 0.05    # Below this → BoN is sufficient
    u_high: float = 0.2    # Above this → early stop (RM too uncertain)
    # Between u_low and u_high → sequential filtering


def summarize_uncertainty(variances: Iterable[float], top_k: int = 3) -> float:
    """Summarize candidate-level variances into a single uncertainty score.

    Uses mean of top-k smallest variances (most confident candidates).
    If even the most confident candidates have high variance, the RM
    is unreliable on this prompt.
    """
    vals = np.asarray(list(variances), dtype=float)
    if vals.size == 0:
        return 0.0
    if top_k > 0:
        vals = np.sort(vals)[: min(top_k, vals.size)]
    return float(np.mean(vals))


def decide_path(
    variances: Iterable[float],
    budget_cfg: BudgetConfig,
    top_k: int = 3,
) -> Decision:
    """Decide inference strategy based on ensemble RM uncertainty.

    Args:
        variances: Per-candidate σ² from the ensemble RM.
        budget_cfg: Threshold configuration.
        top_k: Number of most-confident candidates to consider.

    Returns:
        "bon" | "sequential" | "early_stop"
    """
    u = summarize_uncertainty(variances, top_k=top_k)
    if u <= budget_cfg.u_low:
        return "bon"
    if u <= budget_cfg.u_high:
        return "sequential"
    return "early_stop"


# =========================================================================
# Advanced Selection Strategies
# =========================================================================

def thompson_sampling_select(
    means: Iterable[float],
    variances: Iterable[float],
) -> int:
    """Thompson Sampling: sample from each candidate's posterior, pick max.

    For each candidate i, draw r_i ~ N(μ_i, σ²_i) and select argmax(r_i).
    This naturally balances exploration (high σ → wider samples) and
    exploitation (high μ → higher expected sample).
    """
    means_arr = np.asarray(list(means), dtype=float)
    vars_arr = np.asarray(list(variances), dtype=float)
    sigmas = np.sqrt(np.maximum(vars_arr, 1e-10))
    samples = np.random.normal(means_arr, sigmas)
    return int(np.argmax(samples))


def lcb_select(
    means: Iterable[float],
    variances: Iterable[float],
    z: float = 1.96,
) -> int:
    """Lower Confidence Bound: pick candidate with highest lower bound.

    Selects argmax(μ_i - z · σ_i). With z=1.96, this is the 95% confidence
    lower bound. Ensures the selected candidate is reliably good.

    This is similar to risk-adjusted BoN but uses standard deviation (σ)
    instead of variance (σ²), which is more principled statistically.
    """
    means_arr = np.asarray(list(means), dtype=float)
    vars_arr = np.asarray(list(variances), dtype=float)
    sigmas = np.sqrt(np.maximum(vars_arr, 1e-10))
    lcbs = means_arr - z * sigmas
    return int(np.argmax(lcbs))


def cvar_select(
    means: Iterable[float],
    variances: Iterable[float],
    alpha: float = 0.05,
) -> int:
    """CVaR (Conditional Value at Risk): minimize worst-case tail risk.

    CVaR_α = E[R | R ≤ VaR_α] for Gaussian: CVaR_α = μ - σ · φ(Φ⁻¹(α)) / α
    where φ is the PDF and Φ⁻¹ is the inverse CDF.

    Picks the candidate with the highest CVaR — the one whose worst-case
    outcomes are still acceptable. Useful for safety-critical applications.
    """
    from scipy.stats import norm

    means_arr = np.asarray(list(means), dtype=float)
    vars_arr = np.asarray(list(variances), dtype=float)
    sigmas = np.sqrt(np.maximum(vars_arr, 1e-10))

    # CVaR for Gaussian: μ - σ * φ(Φ⁻¹(α)) / α
    z_alpha = norm.ppf(alpha)
    pdf_z = norm.pdf(z_alpha)
    cvars = means_arr - sigmas * pdf_z / alpha

    return int(np.argmax(cvars))


