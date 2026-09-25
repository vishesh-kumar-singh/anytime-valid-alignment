"""BoN method wrapper functions.

Thin wrappers that call the underlying selection primitives from
``src.methods.parallel.selection`` and return a ``MethodResult`` with
timing information.  Used by hypothesis modules and the legacy runner.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ...core.results import MethodResult
import math


# ═════════════════════════════════════════════════════════════════════════
# Parallel (BoN) Methods
# ═════════════════════════════════════════════════════════════════════════

def run_borda_voting(
    candidates: List[str],
    scores: List[float],
    p: float = 1.0,
) -> MethodResult:
    """Borda voting based on ranking of scores (e.g., self-certainty)."""
    t0 = time.perf_counter()
    idx, vote_counts = borda_voting(candidates, scores, p=p)
    return MethodResult(
        method=f"borda_voting_p{p}",
        selected_answer=candidates[idx],
        reward_mean=scores[idx],
        reward_variance=0.0,
        score=vote_counts[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )

def run_vanilla_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
) -> MethodResult:
    """Vanilla BoN: select candidate with highest scalar reward."""
    t0 = time.perf_counter()
    idx = greedy_argmax(means)
    return MethodResult(
        method="vanilla_bon",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=means[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_risk_adjusted_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    lam: float,
) -> MethodResult:
    """Risk-adjusted BoN: argmax of μ − λσ²."""
    t0 = time.perf_counter()
    idx, adjusted = risk_adjusted_argmax(means, variances, lam=lam)
    return MethodResult(
        method="risk_adjusted_bon",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=adjusted[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_bon_sharpe(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    eps: float = 1e-8,
) -> MethodResult:
    """BoN-Sharpe: deterministic argmax of μ / σ."""
    t0 = time.perf_counter()
    idx, sharpe_scores = sharpe_ratio_argmax(means, variances, eps=eps)
    return MethodResult(
        method="bon_sharpe",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=sharpe_scores[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_bon_linear(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    c: float,
) -> MethodResult:
    """BoN-Linear: deterministic argmax of μ − c·σ."""
    t0 = time.perf_counter()
    linear_scores = [m - c * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx = int(np.argmax(np.asarray(linear_scores, dtype=float)))
    return MethodResult(
        method="bon_linear",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=linear_scores[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_bon_pessimism(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    beta: float,
) -> MethodResult:
    """Inference-time pessimism: deterministic argmax of μ − β·σ."""
    t0 = time.perf_counter()
    scores = [m - beta * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx = int(np.argmax(np.asarray(scores, dtype=float)))
    return MethodResult(
        method="bon_pessimism",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=scores[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_soft_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    temperature: float,
    rng: np.random.Generator,
) -> MethodResult:
    """SoftBoN: sample from softmax(μ / T)."""
    t0 = time.perf_counter()
    idx, probs = softbon_select(means, temperature=temperature, rng=rng)
    return MethodResult(
        method="soft_bon",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=probs[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_softbon_sharpe(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    temperature: float,
    eps: float,
    rng: np.random.Generator,
) -> MethodResult:
    """SoftBoN-Sharpe: sample from softmax((μ/σ) / T)."""
    t0 = time.perf_counter()
    _, sharpe_scores = sharpe_ratio_argmax(means, variances, eps=eps)
    idx, probs = softbon_select(sharpe_scores, temperature=temperature, rng=rng)
    return MethodResult(
        method="softbon_sharpe",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=probs[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_softbon_linear(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    c: float,
    temperature: float,
    rng: np.random.Generator,
) -> MethodResult:
    """SoftBoN-Linear: sample from softmax((μ − c·σ) / T)."""
    t0 = time.perf_counter()
    linear_scores = [m - c * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx, probs = softbon_select(linear_scores, temperature=temperature, rng=rng)
    return MethodResult(
        method="softbon_linear",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=probs[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_softbon_pessimism(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    beta: float,
    temperature: float,
    rng: np.random.Generator,
) -> MethodResult:
    """Soft pessimism: sample from softmax((μ − β·σ) / T)."""
    t0 = time.perf_counter()
    scores = [m - beta * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx, probs = softbon_select(scores, temperature=temperature, rng=rng)
    return MethodResult(
        method="softbon_pessimism",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=probs[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════
# Ensemble-Aware Methods
# ═════════════════════════════════════════════════════════════════════════

def run_mob_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    n_bootstrap: int = 50,
    subsample_frac: float = 0.6,
    use_variance_weighting: bool = True,
    eps: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
) -> MethodResult:
    """Majority-of-Bests: bootstrap resampling + majority vote."""
    t0 = time.perf_counter()
    idx, _info = majority_of_bests(
        candidates, means, variances,
        n_bootstrap=n_bootstrap,
        subsample_frac=subsample_frac,
        use_variance_weighting=use_variance_weighting,
        eps=eps,
        rng=rng,
    )
    method_name = "mob_sharpe" if use_variance_weighting else "mob_vanilla"
    return MethodResult(
        method=method_name,
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=means[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_wbon(
    candidates: List[str],
    member_scores: List[List[float]],
    eps: float = 1e-8,
) -> MethodResult:
    """Weighted BoN: inverse-variance weighted aggregation then argmax."""
    t0 = time.perf_counter()
    if member_scores is None:
        raise ValueError("WBoN requires member_scores (per-model raw scores).")
    w_means, w_variances = inverse_variance_weighted_aggregation(member_scores, eps=eps)
    idx = greedy_argmax(w_means)
    return MethodResult(
        method="wbon_inv_var",
        selected_answer=candidates[idx],
        reward_mean=w_means[idx],
        reward_variance=w_variances[idx],
        score=w_means[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_wbon_sharpe(
    candidates: List[str],
    member_scores: List[List[float]],
    eps: float = 1e-8,
) -> MethodResult:
    """Weighted BoN + Sharpe: inverse-variance aggregation then Sharpe."""
    t0 = time.perf_counter()
    if member_scores is None:
        raise ValueError("WBoN-Sharpe requires member_scores.")
    w_means, w_variances = inverse_variance_weighted_aggregation(member_scores, eps=eps)
    idx, sharpe_scores = sharpe_ratio_argmax(w_means, w_variances, eps=eps)
    return MethodResult(
        method="wbon_sharpe",
        selected_answer=candidates[idx],
        reward_mean=w_means[idx],
        reward_variance=w_variances[idx],
        score=sharpe_scores[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def run_minimax(
    candidates: List[str],
    member_scores: List[List[float]],
) -> MethodResult:
    """Minimax BoN: worst-case optimization across RM ensemble."""
    t0 = time.perf_counter()
    if member_scores is None:
        raise ValueError("Minimax requires member_scores.")
    idx = minimax_argmax(member_scores)
    
    # Calculate means and variances manually for the returned result
    arr = np.asarray(member_scores, dtype=float)
    means = arr.mean(axis=0).tolist()
    variances = arr.var(axis=0, ddof=0).tolist()
    min_scores = arr.min(axis=0).tolist()
    
    return MethodResult(
        method="minimax",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=min_scores[idx],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


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

def borda_voting(
    candidates: List[str],
    scores: Iterable[float],
    p: float = 1.0,
) -> Tuple[int, List[float]]:
    """Borda Voting based on score ranking.
    Formula: v(r) = (N - r + 1)^p
    """
    scores_list = _to_list(scores)
    n = len(scores_list)
    if n == 0:
        raise ValueError("borda_voting received empty lists.")
        
    # Rank candidates by score (highest score -> rank 1)
    # argsort returns indices that would sort the array, so we sort descending
    sorted_indices = sorted(range(n), key=lambda i: scores_list[i], reverse=True)
    
    # Calculate ranks (1-indexed)
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank
        
    # Calculate votes for each candidate
    votes = [(n - ranks[i] + 1) ** p for i in range(n)]
    
    # We must aggregate votes for candidates with the EXACT same extracted answer!
    # Wait, the paper says: "Each valid response contributes votes to its final answer proportional to its rank. 
    # The answer accumulating the highest vote total becomes the consensus selection."
    # Since standard.py works at candidate level and doesn't know how to extract answers...
    # Wait, the RWSC methods extract answers!
    # In standard.py, we only have candidate strings. How do we extract answers?
    # I should implement this in src/methods/parallel/rwsc.py instead!

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


