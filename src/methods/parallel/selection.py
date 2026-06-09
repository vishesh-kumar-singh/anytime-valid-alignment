"""
Selection policies for Best-of-N (BoN) candidate selection.

BoN strategies:
  1. Vanilla BoN (greedy_argmax): Select candidate with highest scalar reward.
  2. Risk-adjusted BoN (risk_adjusted_argmax): Select max(μ - λσ²).
    3. Sharpe-ratio BoN (sharpe_ratio_argmax): Select max(μ / σ).
    4. SoftBoN utilities for probabilistic selection via softmax over utilities.

The risk-aware methods require a reward model that outputs (μ, σ²) per candidate.
"""

import math
from typing import Iterable, List, Tuple

import numpy as np


def _to_list(x: Iterable[float]) -> List[float]:
    return [float(v) for v in x]


# ---------------------------------------------------------------------------
# 1. Vanilla BoN: argmax of scalar rewards
# ---------------------------------------------------------------------------

def greedy_argmax(rewards: Iterable[float]) -> int:
    """Select the candidate with the highest reward score."""
    rewards_list = _to_list(rewards)
    if not rewards_list:
        raise ValueError("greedy_argmax received an empty rewards list.")
    return max(range(len(rewards_list)), key=lambda i: rewards_list[i])


# ---------------------------------------------------------------------------
# 2. Risk-adjusted BoN: argmax of μ - λ * σ²
# ---------------------------------------------------------------------------

def risk_adjusted_argmax(
    means: Iterable[float],
    variances: Iterable[float],
    lam: float,
) -> Tuple[int, List[float]]:
    """Select the candidate maximising μ - λσ².

    Args:
        means: Per-candidate mean rewards (μ).
        variances: Per-candidate reward variances (σ²).
        lam: Risk-aversion coefficient λ ≥ 0.

    Returns:
        (best_index, adjusted_scores) where adjusted_scores[i] = μ_i - λ·σ²_i.
    """
    means_list = _to_list(means)
    vars_list = _to_list(variances)

    if len(means_list) != len(vars_list):
        raise ValueError(
            f"Shape mismatch: len(means)={len(means_list)}, len(vars)={len(vars_list)}"
        )
    if not means_list:
        raise ValueError("risk_adjusted_argmax received an empty means list.")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}.")

    adjusted = [m - lam * v for m, v in zip(means_list, vars_list)]
    best_idx = max(range(len(adjusted)), key=lambda i: adjusted[i])
    return best_idx, adjusted


# ---------------------------------------------------------------------------
# 3. Sharpe-ratio BoN: argmax of μ / σ
# ---------------------------------------------------------------------------

def sharpe_ratio_argmax(
    means: Iterable[float],
    variances: Iterable[float],
    eps: float = 1e-8,
) -> Tuple[int, List[float]]:
    """Select the candidate maximising μ / σ (Sharpe-ratio style).

    This is our proposed uncertainty-aware selection: high reward AND low
    variance are both rewarded. A candidate with a moderate reward but very
    low uncertainty can beat one with high reward but high uncertainty.

    Args:
        means: Per-candidate mean rewards (μ).
        variances: Per-candidate reward variances (σ²).
        eps: Small constant to avoid division by zero.

    Returns:
        (best_index, sharpe_scores) where sharpe_scores[i] = μ_i / (σ_i + ε).
    """
    means_list = _to_list(means)
    vars_list = _to_list(variances)

    if len(means_list) != len(vars_list):
        raise ValueError(
            f"Shape mismatch: len(means)={len(means_list)}, len(vars)={len(vars_list)}"
        )
    if not means_list:
        raise ValueError("sharpe_ratio_argmax received an empty means list.")

    sharpe = [m / (math.sqrt(max(v, 0.0)) + eps) for m, v in zip(means_list, vars_list)]
    best_idx = max(range(len(sharpe)), key=lambda i: sharpe[i])
    return best_idx, sharpe


def softmax_probabilities(values: Iterable[float], temperature: float = 1.0) -> List[float]:
    """Convert utilities to a categorical distribution using softmax."""
    vals = _to_list(values)
    if not vals:
        raise ValueError("softmax_probabilities received an empty values list.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    arr = np.asarray(vals, dtype=float) / temperature
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / np.sum(probs)
    return probs.tolist()


def softbon_select(
    values: Iterable[float],
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tuple[int, List[float]]:
    """Sample an index from softmax(values / temperature)."""
    probs = softmax_probabilities(values, temperature=temperature)
    gen = rng or np.random.default_rng()
    idx = int(gen.choice(len(probs), p=np.asarray(probs, dtype=float)))
    return idx, probs


# ---------------------------------------------------------------------------
# 5. Majority-of-Bests (MoB) — Bootstrap resampling + majority vote
# ---------------------------------------------------------------------------

def majority_of_bests(
    candidates: list,
    means: Iterable[float],
    variances: Iterable[float],
    n_bootstrap: int = 50,
    subsample_frac: float = 0.6,
    use_variance_weighting: bool = True,
    eps: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> Tuple[int, dict]:
    """Majority-of-Bests (MoB) with optional uncertainty weighting.

    Instead of picking the single argmax candidate (fragile extreme-value
    selection), we:
      1. Bootstrap B subsets of the N candidates.
      2. Pick the best from each subset (by mean reward or Sharpe).
      3. Majority-vote among those "bests".

    When use_variance_weighting=True, the "best" in each subset is chosen
    by Sharpe ratio (mu / (sigma + eps)) rather than raw mu. This naturally
    prefers confident candidates.

    Reference: "Is Best-of-N the Best of Them?" (arXiv)

    Args:
        candidates: List of candidate answer strings.
        means: Per-candidate mean rewards.
        variances: Per-candidate reward variances.
        n_bootstrap: Number of bootstrap subsets (B).
        subsample_frac: Fraction of N to include in each subset.
        use_variance_weighting: Use Sharpe ratio for per-subset selection.
        eps: Stability constant for Sharpe denominator.
        rng: Random generator for reproducibility.

    Returns:
        (best_index, info_dict) where info_dict has vote counts and details.
    """
    means_list = _to_list(means)
    vars_list = _to_list(variances)
    n = len(means_list)
    gen = rng or np.random.default_rng()

    if n == 0:
        raise ValueError("majority_of_bests received empty candidates.")

    subset_size = max(2, int(n * subsample_frac))
    vote_counts = np.zeros(n, dtype=int)

    for _ in range(n_bootstrap):
        # Sample subset indices with replacement
        subset_idx = gen.choice(n, size=subset_size, replace=True)

        if use_variance_weighting:
            # Select by Sharpe ratio within subset
            best_in_subset = max(
                subset_idx,
                key=lambda i: means_list[i] / (math.sqrt(max(vars_list[i], 0.0)) + eps),
            )
        else:
            # Select by raw mean within subset
            best_in_subset = max(subset_idx, key=lambda i: means_list[i])

        vote_counts[best_in_subset] += 1

    # Majority vote — pick the candidate with the most wins
    # Break ties by mean reward
    winner = int(max(range(n), key=lambda i: (vote_counts[i], means_list[i])))

    info = {
        "vote_counts": vote_counts.tolist(),
        "n_bootstrap": n_bootstrap,
        "subset_size": subset_size,
        "use_variance_weighting": use_variance_weighting,
    }
    return winner, info


# ---------------------------------------------------------------------------
# 6. Inverse-Variance Weighted BoN (WBoN)
# ---------------------------------------------------------------------------

def inverse_variance_weighted_aggregation(
    member_scores: list,
    eps: float = 1e-8,
) -> Tuple[List[float], List[float]]:
    """Re-aggregate member scores using inverse-variance weighting.

    Instead of treating all K reward models equally (simple mean), weight
    each model's contribution by the inverse of its deviation from the
    ensemble mean. Models that are outliers for a specific candidate get
    downweighted.

    Reference: WARM (Weight Averaged Reward Models), Ramé et al. 2024.

    Args:
        member_scores: Shape (K, N) — K models, N candidates.
        eps: Stability constant.

    Returns:
        (weighted_means, weighted_variances) for each candidate.
    """
    arr = np.asarray(member_scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"member_scores must be (K, N), got shape={arr.shape}")

    K, N = arr.shape
    if K < 2:
        # Fallback to simple mean/var
        means = arr.mean(axis=0).tolist()
        variances = [0.0] * N
        return means, variances

    # Simple ensemble mean for reference
    simple_mean = arr.mean(axis=0)  # (N,)

    weighted_means = []
    weighted_variances = []

    for j in range(N):
        scores_j = arr[:, j]  # (K,) — all models' scores for candidate j
        deviations = np.abs(scores_j - simple_mean[j])
        # Inverse-deviation weights (more deviation = less weight)
        weights = 1.0 / (deviations + eps)
        weights = weights / weights.sum()  # normalise to sum to 1

        w_mean = float(np.dot(weights, scores_j))
        # Weighted variance
        w_var = float(np.dot(weights, (scores_j - w_mean) ** 2))

        weighted_means.append(w_mean)
        weighted_variances.append(w_var)

    return weighted_means, weighted_variances


def minimax_argmax(member_scores: list) -> int:
    """Worst-Case Optimization (Minimax): select candidate with highest minimum RM score.

    For each candidate, computes the minimum score assigned by any reward model.
    Selects the candidate that maximizes this minimum score.

    Args:
        member_scores: Shape (K, N) — K models, N candidates.

    Returns:
        Index of the selected candidate.
    """
    arr = np.asarray(member_scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"member_scores must be (K, N), got shape={arr.shape}")
    
    # Compute minimum score across all K models for each candidate N
    # min_scores shape: (N,)
    min_scores = np.min(arr, axis=0)
    
    # Return index of candidate with the maximum minimum score
    return int(np.argmax(min_scores))


__all__ = [
    "greedy_argmax",
    "risk_adjusted_argmax",
    "sharpe_ratio_argmax",
    "softmax_probabilities",
    "softbon_select",
    "majority_of_bests",
    "inverse_variance_weighted_aggregation",
    "minimax_argmax",
]
