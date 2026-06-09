"""BoN method wrapper functions.

Thin wrappers that call the underlying selection primitives from
``src.methods.parallel.selection`` and return a ``MethodResult`` with
timing information.  Used by hypothesis modules and the legacy runner.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .results import MethodResult
from ..methods.parallel.selection import (
    greedy_argmax,
    inverse_variance_weighted_aggregation,
    majority_of_bests,
    risk_adjusted_argmax,
    sharpe_ratio_argmax,
    softbon_select,
    minimax_argmax,
)


# ═════════════════════════════════════════════════════════════════════════
# Parallel (BoN) Methods
# ═════════════════════════════════════════════════════════════════════════

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
