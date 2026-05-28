"""
Main experiment runner for Uncertainty-Guided Test-Time Alignment.

Supports two RM modes:
  - trivial:  Pre-trained scalar RM ensemble (fast, no training needed)
  - robust:   Uncertainty-aware RM ensemble with learned (mu, sigma) per model

Supports three datasets:
  - gsm8k:       Grade-school math (numeric answer extraction)
  - math500:     Competition math from MATH-500 (boxed answer extraction)
  - alpaca_eval: Open-ended instruction following (RM-only, no correctness)

Evaluates multiple inference strategies:
  BoN methods:
    1. Vanilla BoN         — argmax of scalar reward
    2. Risk-adjusted BoN   — argmax of (mu - lambda * sigma^2)
        3. Sharpe-ratio BoN    — argmax of (mu / sigma)
        4. SoftBoN             — sample from softmax(mu)
        5. SoftBoN-Sharpe      — sample from softmax(mu / sigma)
        6. SoftBoN-Linear      — sample from softmax(mu - c*sigma)
  Sequential methods:
    4. Vanilla RF-SeqBoN   — accept if r(a) >= gamma
                5. Sharpe SeqBoN       — accept if mu/sigma^2 >= gamma
        6. Lin SeqBoN          — accept if mu - c*sigma >= gamma
  Adaptive:
        7. Controller           — routes based on uncertainty (supports thompson/lcb/cvar)

Usage:
    # Trivial RM on GSM8K (default):
    python -m src.experiments.run_risk_aware_inference --max-examples 50

    # Robust URM on MATH500:
    python -m src.experiments.run_risk_aware_inference \\
        --rm-mode robust --dataset math500 \\
        --urm-ensemble-dir models/urm_ensemble --max-examples 50

    # With advanced controller:
    python -m src.experiments.run_risk_aware_inference \\
        --controller-type thompson --max-examples 50
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import typer
from tqdm import tqdm

from src.generation.generator import GenerationConfig, Generator
from src.policies.controller import (
    BudgetConfig, decide_path, summarize_uncertainty,
    thompson_sampling_select, lcb_select, cvar_select,
)
from src.policies.selection import greedy_argmax, risk_adjusted_argmax, sharpe_ratio_argmax
from src.policies.selection import softbon_select
from src.policies.selection import majority_of_bests, inverse_variance_weighted_aggregation
from src.reward_modeling.inference import RewardEnsemble
from src.sequential.search import (
    rf_seqbon_vanilla,
    rf_seqbon_sharpe_ratio,
    rf_seqbon_linear_uncertainty,
    rf_seqbon_adaptive_threshold,
)


# =========================================================================
# Unified RM Scorer Interface
# =========================================================================

class RMScorer(Protocol):
    """Unified interface for both trivial and robust RM ensembles."""

    def score(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        """Return (means, variances) for each candidate."""
        ...

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        """Return scalar rewards for each candidate."""
        ...


class TrivialEnsembleScorer:
    """Wraps the pre-trained scalar RM ensemble."""

    def __init__(self, model_names: List[str], batch_size: int = 8, load_in_4bit: bool = False):
        self.ensemble = RewardEnsemble(model_names, batch_size=batch_size, load_in_4bit=load_in_4bit)
        self.last_member_scores: Optional[List[List[float]]] = None

    def score(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        stats = self.ensemble.score(prompt, candidates)
        self.last_member_scores = stats.member_scores
        return stats.means, stats.variances

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        return self.ensemble.score_scalar(prompt, candidates)


class RobustURMScorer:
    """Wraps the uncertainty-aware RM ensemble (trained with MLE loss)."""

    def __init__(self, ensemble_dir: str, base_model_name: str = "microsoft/deberta-v3-base"):
        from src.reward_modeling.uncertainty_rm import URMEnsemble
        # Current trainer saves member_* directories; keep model_* for backward compatibility.
        model_paths = sorted(Path(ensemble_dir).glob("member_*"))
        if not model_paths:
            model_paths = sorted(Path(ensemble_dir).glob("model_*"))
        if not model_paths:
            raise FileNotFoundError(
                f"No URM model directories found in {ensemble_dir} (expected member_* or model_*). "
                "Train with: python -m src.reward_modeling.uncertainty_rm ..."
            )
        self.ensemble = URMEnsemble(
            model_paths=[str(p) for p in model_paths],
            base_model_name=base_model_name,
        )
        self.last_member_scores: Optional[List[List[float]]] = None

    def score(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        stats = self.ensemble.score(prompt, candidates)
        self.last_member_scores = getattr(stats, "member_scores", None)
        return list(stats.means), list(stats.variances)

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        return self.ensemble.score_scalar(prompt, candidates)


TRIVIAL_4_RM_PRESET = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
    "NCSOFT/Llama-3-OffsetBias-RM-8B",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
]


def build_scorer(cfg: "ExperimentConfig") -> RMScorer:
    """Factory: build the appropriate RM scorer based on config."""
    if cfg.rm_mode == "robust":
        if not cfg.urm_ensemble_dir:
            raise ValueError("--urm-ensemble-dir required when --rm-mode=robust")
        print(f"Loading robust URM ensemble from: {cfg.urm_ensemble_dir}")
        return RobustURMScorer(cfg.urm_ensemble_dir, cfg.urm_base_model)
    if cfg.rm_mode == "trivial4":
        model_names = cfg.reward_model_names
        if len(model_names) == 0:
            model_names = TRIVIAL_4_RM_PRESET
        print(f"Loading trivial4-mode ensemble ({len(model_names)} models): {model_names}")
        return TrivialEnsembleScorer(model_names, cfg.reward_batch_size, cfg.reward_load_in_4bit)
    else:
        print(f"Loading trivial RM ensemble: {cfg.reward_model_names}")
        return TrivialEnsembleScorer(cfg.reward_model_names, cfg.reward_batch_size, cfg.reward_load_in_4bit)


# =========================================================================
# Dataset Loading
# =========================================================================

def load_dataset_examples(
    dataset: str, split: str = "test", max_examples: Optional[int] = None,
) -> List[dict]:
    """Load examples from the specified dataset."""
    dataset = dataset.lower()
    if dataset == "gsm8k":
        from src.data_utils.gsm8k import load_gsm8k
        return load_gsm8k(split=split, max_examples=max_examples)
    elif dataset == "math500":
        from src.data_utils.math500 import load_math500
        return load_math500(max_examples=max_examples)
    elif dataset == "alpaca_eval":
        from src.data_utils.alpaca_eval import load_alpaca_eval
        return load_alpaca_eval(max_examples=max_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: gsm8k, math500, alpaca_eval")


def check_correctness(prediction: str, ground_truth: str, dataset: str) -> bool:
    """Check answer correctness based on dataset type."""
    dataset = dataset.lower()
    if dataset == "gsm8k":
        from src.data_utils.gsm8k import check_answer_correct
        return check_answer_correct(prediction, ground_truth)
    elif dataset == "math500":
        from src.data_utils.math500 import check_math_correct
        return check_math_correct(prediction, ground_truth)
    elif dataset == "alpaca_eval":
        # Open-ended: no ground truth correctness, always False
        return False
    return False


# =========================================================================
# Config & Result
# =========================================================================

@dataclass
class ExperimentConfig:
    # Generation
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    gen_device: int = 0

    # BoN
    bon_n: int = 5
    lam: float = 0.05
    softbon_temperature: float = 1.0
    softbon_sharpe_temperature: Optional[float] = None
    softbon_linear_temperature: Optional[float] = None
    softbon_pessimism_temperature: Optional[float] = None
    sharpe_eps: float = 1e-8
    variance_norm_mode: str = "none"  # none | zscore | robust_zscore
    variance_norm_eps: float = 1e-6
    variance_bound_tanh: bool = False
    pessimism_beta: float = 1.0
    enable_pessimism: bool = False
    naive_bon_baseline_only: bool = False
    bon_only: bool = False
    print_variances: bool = False
    seed: int = 42

    # Sequential
    seq_n: int = 5
    gamma_vanilla: float = 0.95
    gamma_sharpe: float = 0.95
    gamma_lin: float = 0.95
    lin_c: float = 1.0
    history_budget: int = 2

    # Controller
    controller_type: str = "threshold"  # threshold | thompson | lcb | cvar
    uncertainty_top_k: int = 3
    u_low: float = 0.05
    u_high: float = 0.2

    # RM mode
    rm_mode: str = "trivial"  # trivial | trivial4 | robust
    reward_model_names: List[str] = field(default_factory=lambda: [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-deberta-v3-large-v2",
    ])
    reward_batch_size: int = 8
    reward_load_in_4bit: bool = False
    urm_ensemble_dir: Optional[str] = None
    urm_base_model: str = "microsoft/deberta-v3-base"

    # Dataset
    dataset: str = "gsm8k"  # gsm8k | math500 | alpaca_eval
    dataset_split: str = "test"
    max_examples: Optional[int] = None

    # Output
    output_path: str = "runs/results.jsonl"
    save_jsonl: bool = True


@dataclass
class MethodResult:
    method: str
    selected_answer: str
    reward_mean: float
    reward_variance: float
    score: float
    n_candidates: int
    n_rm_calls: int
    latency_sec: float


def _normalize_member_scores(
    member_scores: List[List[float]],
    mode: str,
    eps: float,
    bound_tanh: bool,
) -> List[List[float]]:
    """Normalize each RM member's candidate scores independently.

    Args:
        member_scores: Shape (K, N) where K=models, N=candidates.
        mode: none | zscore | robust_zscore | minmax.
        eps: Stability term for scale denominators.
        bound_tanh: If True, map normalized values to [-1, 1] via tanh.
    """
    norm_mode = mode.lower().strip()
    if norm_mode == "none":
        arr = np.asarray(member_scores, dtype=float)
        return arr.tolist()

    arr = np.asarray(member_scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"member_scores must be rank-2 (K,N), got shape={arr.shape}")

    normalized = np.zeros_like(arr, dtype=float)
    for k in range(arr.shape[0]):
        row = arr[k]
        if norm_mode == "zscore":
            center = float(np.mean(row))
            scale = float(np.std(row, ddof=0))
            if scale < eps:
                normalized[k] = np.zeros_like(row)
            else:
                normalized[k] = (row - center) / (scale + eps)
        elif norm_mode == "robust_zscore":
            center = float(np.median(row))
            mad = float(np.median(np.abs(row - center)))
            robust_scale = 1.4826 * mad
            if robust_scale < eps:
                std = float(np.std(row, ddof=0))
                if std < eps:
                    normalized[k] = np.zeros_like(row)
                else:
                    normalized[k] = (row - center) / (std + eps)
            else:
                normalized[k] = (row - center) / (robust_scale + eps)
        elif norm_mode == "minmax":
            lo = float(np.min(row))
            hi = float(np.max(row))
            span = hi - lo
            if span < eps:
                normalized[k] = np.zeros_like(row)
            else:
                normalized[k] = (row - lo) / (span + eps)
        else:
            raise ValueError(f"Unknown variance_norm_mode: {mode}. Use none, zscore, robust_zscore, or minmax")

    if bound_tanh:
        normalized = np.tanh(normalized)

    return normalized.tolist()


def _dataset_system_message(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "math500":
        return (
            "You are a careful math problem solver. Keep reasoning concise and correct. "
            "If needed, use short step-by-step derivations. "
            "Your final line must be exactly: \\boxed{answer}. "
            "Do not add any text after that final boxed line."
        )
    if dataset == "gsm8k":
        return (
            "You are a helpful assistant for arithmetic word problems. Solve briefly and clearly. "
            "Put your final numerical answer after ####."
        )
    return "You are a helpful assistant."


# =========================================================================
# BoN Methods
# =========================================================================

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
    """Risk-adjusted BoN: select candidate with highest mu - lambda * sigma^2."""
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


def run_sharpe_ratio_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    eps: float = 1e-8,
) -> MethodResult:
    """Sharpe-ratio BoN: select candidate with highest mu / sigma."""
    t0 = time.perf_counter()
    idx, sharpe_scores = sharpe_ratio_argmax(means, variances, eps=eps)
    return MethodResult(
        method="sharpe_ratio_bon",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=sharpe_scores[idx],
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
    """BoN-Sharpe: deterministic argmax of mu / sigma."""
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
    """BoN-Linear: deterministic argmax of mu - c*sigma."""
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


def run_soft_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    temperature: float,
    rng: np.random.Generator,
) -> MethodResult:
    """SoftBoN baseline: sample from softmax(mu / T)."""
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
    """SoftBoN-Sharpe: sample from softmax((mu/sigma) / T)."""
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
    """SoftBoN-Linear: sample from softmax((mu - c*sigma) / T)."""
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


def run_bon_pessimism(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    beta: float,
) -> MethodResult:
    """Inference-time pessimism: deterministic argmax of mu - beta*sigma."""
    t0 = time.perf_counter()
    pessimistic_scores = [m - beta * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx = int(np.argmax(np.asarray(pessimistic_scores, dtype=float)))
    return MethodResult(
        method="bon_pessimism",
        selected_answer=candidates[idx],
        reward_mean=means[idx],
        reward_variance=variances[idx],
        score=pessimistic_scores[idx],
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
    """Soft inference-time pessimism: sample from softmax((mu - beta*sigma)/T)."""
    t0 = time.perf_counter()
    pessimistic_scores = [m - beta * (max(v, 0.0) ** 0.5) for m, v in zip(means, variances)]
    idx, probs = softbon_select(pessimistic_scores, temperature=temperature, rng=rng)
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


# =========================================================================
# Sequential Methods (RF-SeqBoN)
# =========================================================================

def run_vanilla_seqbon(
    prompt: str,
    generator: Generator,
    scorer: RMScorer,
    cfg: ExperimentConfig,
) -> MethodResult:
    t0 = time.perf_counter()
    rm_calls = [0]
    system_msg = _dataset_system_message(cfg.dataset)

    def generate_fn(hist: str, k: int) -> List[str]:
        return generator.generate(hist, n=k, system_msg=system_msg)
    def reward_fn(p: str, answers: List[str]) -> List[float]:
        rm_calls[0] += 1
        return scorer.score_scalar(p, answers)

    result = rf_seqbon_vanilla(
        prompt=prompt,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n=cfg.seq_n,
        gamma=cfg.gamma_vanilla,
        history_budget=cfg.history_budget,
        dataset=cfg.dataset,
    )
    # best among all collected
    best_idx = result.all_answers.index(result.final_answer) if result.final_answer in result.all_answers else -1
    return MethodResult(
        method="vanilla_seqbon",
        selected_answer=result.final_answer,
        reward_mean=result.all_rewards[best_idx] if best_idx != -1 else 0.0,
        reward_variance=0.0,
        score=result.all_rewards[best_idx] if best_idx != -1 else 0.0,
        n_candidates=result.n_total,
        n_rm_calls=rm_calls[0],
        latency_sec=time.perf_counter() - t0,
    )

def run_sharpe_seqbon(
    prompt: str,
    generator: Generator,
    scorer: RMScorer,
    cfg: ExperimentConfig,
) -> MethodResult:
    t0 = time.perf_counter()
    rm_calls = [0]
    system_msg = _dataset_system_message(cfg.dataset)

    def generate_fn(hist: str, k: int) -> List[str]:
        return generator.generate(hist, n=k, system_msg=system_msg)
    def reward_fn(p: str, answers: List[str]) -> Tuple[List[float], List[float]]:
        rm_calls[0] += 1
        return scorer.score(p, answers)

    result = rf_seqbon_sharpe_ratio(
        prompt=prompt,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n=cfg.seq_n,
        gamma=cfg.gamma_sharpe,
        history_budget=cfg.history_budget,
        dataset=cfg.dataset,
    )
    best_idx = result.all_answers.index(result.final_answer) if result.final_answer in result.all_answers else -1
    return MethodResult(
        method="Sharpe_seqbon",
        selected_answer=result.final_answer,
        reward_mean=result.all_rewards[best_idx] if best_idx != -1 else 0.0,
        reward_variance=result.all_variances[best_idx] if best_idx != -1 else 0.0,
        score=(result.all_rewards[best_idx] / ((max(result.all_variances[best_idx], 0.0) ** 0.5) + 1e-8)) if best_idx != -1 else 0.0,
        n_candidates=result.n_total,
        n_rm_calls=rm_calls[0],
        latency_sec=time.perf_counter() - t0,
    )

def run_lin_seqbon(
    prompt: str,
    generator: Generator,
    scorer: RMScorer,
    cfg: ExperimentConfig,
) -> MethodResult:
    t0 = time.perf_counter()
    rm_calls = [0]
    system_msg = _dataset_system_message(cfg.dataset)

    def generate_fn(hist: str, k: int) -> List[str]:
        return generator.generate(hist, n=k, system_msg=system_msg)
    def reward_fn(p: str, answers: List[str]) -> Tuple[List[float], List[float]]:
        rm_calls[0] += 1
        return scorer.score(p, answers)

    result = rf_seqbon_linear_uncertainty(
        prompt=prompt,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n=cfg.seq_n,
        gamma=cfg.gamma_lin,
        c=cfg.lin_c,
        history_budget=cfg.history_budget,
        dataset=cfg.dataset,
    )
    best_idx = result.all_answers.index(result.final_answer) if result.final_answer in result.all_answers else -1
    score = (result.all_rewards[best_idx] - cfg.lin_c * (max(0, result.all_variances[best_idx])**0.5)) if best_idx != -1 else 0.0
    return MethodResult(
        method="Lin_seqBon",
        selected_answer=result.final_answer,
        reward_mean=result.all_rewards[best_idx] if best_idx != -1 else 0.0,
        reward_variance=result.all_variances[best_idx] if best_idx != -1 else 0.0,
        score=score,
        n_candidates=result.n_total,
        n_rm_calls=rm_calls[0],
        latency_sec=time.perf_counter() - t0,
    )


# =========================================================================
# New Research Ideas: MoB, WBoN, Adaptive SeqBoN
# =========================================================================

def run_mob_bon(
    candidates: List[str],
    means: List[float],
    variances: List[float],
    n_bootstrap: int = 50,
    subsample_frac: float = 0.6,
    use_variance_weighting: bool = True,
    eps: float = 1e-8,
    rng: np.random.Generator = None,
) -> MethodResult:
    """Majority-of-Bests: bootstrap resampling + majority vote."""
    t0 = time.perf_counter()
    idx, info = majority_of_bests(
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
    """Weighted BoN + Sharpe: inverse-variance aggregation then Sharpe selection."""
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


def run_adaptive_seqbon(
    prompt: str,
    generator: Generator,
    scorer: RMScorer,
    cfg: ExperimentConfig,
    n_burnin: int = 3,
    gamma_percentile: float = 50.0,
) -> MethodResult:
    """Adaptive-threshold SeqBoN with burn-in calibration."""
    t0 = time.perf_counter()
    rm_calls = [0]
    system_msg = _dataset_system_message(cfg.dataset)

    def generate_fn(hist: str, k: int) -> List[str]:
        return generator.generate(hist, n=k, system_msg=system_msg)
    def reward_fn(p: str, answers: List[str]) -> Tuple[List[float], List[float]]:
        rm_calls[0] += 1
        return scorer.score(p, answers)

    result = rf_seqbon_adaptive_threshold(
        prompt=prompt,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n=cfg.seq_n,
        n_burnin=n_burnin,
        gamma_percentile=gamma_percentile,
        use_sharpe=True,
        history_budget=cfg.history_budget,
        dataset=cfg.dataset,
    )
    best_idx = result.all_answers.index(result.final_answer) if result.final_answer in result.all_answers else -1
    return MethodResult(
        method=f"adaptive_seqbon_p{int(gamma_percentile)}",
        selected_answer=result.final_answer,
        reward_mean=result.all_rewards[best_idx] if best_idx != -1 else 0.0,
        reward_variance=result.all_variances[best_idx] if best_idx != -1 else 0.0,
        score=(result.all_rewards[best_idx] / ((max(result.all_variances[best_idx], 0.0) ** 0.5) + 1e-8)) if best_idx != -1 else 0.0,
        n_candidates=result.n_total,
        n_rm_calls=rm_calls[0],
        latency_sec=time.perf_counter() - t0,
    )


# =========================================================================
# Controller (Adaptive Strategy Selection)
# =========================================================================

def _select_by_controller_type(
    controller_type: str,
    means: List[float],
    variances: List[float],
) -> int:
    """Pick best candidate using the specified controller selection strategy."""
    if controller_type == "thompson":
        return thompson_sampling_select(means, variances)
    elif controller_type == "lcb":
        return lcb_select(means, variances)
    elif controller_type == "cvar":
        return cvar_select(means, variances)
    else:
        # Default: sharpe ratio
        idx, _ = sharpe_ratio_argmax(means, variances)
        return idx


def run_controller(
    prompt: str,
    candidates: List[str],
    means: List[float],
    variances: List[float],
    generator: Generator,
    scorer: RMScorer,
    cfg: ExperimentConfig,
) -> MethodResult:
    """Adaptive controller: routes to BoN / SeqBoN / early-stop."""
    t0 = time.perf_counter()

    decision = decide_path(
        variances,
        budget_cfg=BudgetConfig(u_low=cfg.u_low, u_high=cfg.u_high),
        top_k=cfg.uncertainty_top_k,
    )

    ctrl_label = cfg.controller_type if cfg.controller_type != "threshold" else "sharpe"

    if decision == "bon":
        idx = _select_by_controller_type(cfg.controller_type, means, variances)
        return MethodResult(
            method=f"controller({ctrl_label})->bon",
            selected_answer=candidates[idx],
            reward_mean=means[idx],
            reward_variance=variances[idx],
            score=means[idx],
            n_candidates=len(candidates),
            n_rm_calls=1,
            latency_sec=time.perf_counter() - t0,
        )

    if decision == "sequential":
        result = run_sharpe_seqbon(
            prompt, generator, scorer, cfg
        )
        result.method = f"controller({ctrl_label})->seqbon"
        result.latency_sec = time.perf_counter() - t0
        return result

    # High uncertainty -> Early stop (pick safest candidate: lowest variance)
    vars_arr = np.asarray(variances, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    safest = int(np.lexsort((-means_arr, vars_arr))[0])

    return MethodResult(
        method=f"controller({ctrl_label})->early_stop",
        selected_answer=candidates[safest],
        reward_mean=means[safest],
        reward_variance=variances[safest],
        score=means[safest],
        n_candidates=len(candidates),
        n_rm_calls=1,
        latency_sec=time.perf_counter() - t0,
    )


def _evaluate_examples(
    cfg: ExperimentConfig,
    generator: Generator,
    scorer: RMScorer,
    examples: List[dict],
    desc: str = "Evaluating",
) -> List[dict]:
    """Evaluate configured methods on a preloaded example set."""
    has_correctness = cfg.dataset in ("gsm8k", "math500")
    rows: List[dict] = []
    system_msg = _dataset_system_message(cfg.dataset)
    rng = np.random.default_rng(cfg.seed)

    for i, example in enumerate(tqdm(examples, desc=desc)):
        prompt = example["question"]
        gold_answer = example.get("answer", "")

        # Step 1: Generate N candidates (shared across BoN methods + controller path)
        # Sequential methods generate their own candidates independently.
        t_gen = time.perf_counter()
        candidates = generator.generate(prompt, n=cfg.bon_n, system_msg=system_msg)
        gen_time = time.perf_counter() - t_gen

        # Step 2: Score all candidates with the RM ensemble → (μ_i, σ²_i)
        t_rm = time.perf_counter()
        raw_means, raw_variances = scorer.score(prompt, candidates)
        member_scores = getattr(scorer, "last_member_scores", None)
        means, variances = raw_means, raw_variances
        normalized_member_scores = None

        if cfg.variance_norm_mode.lower() != "none":
            if member_scores is None:
                print(
                    "[warn] variance normalization requested but candidate_member_scores are unavailable; "
                    "falling back to raw means/variances for this example."
                )
            else:
                normalized_member_scores = _normalize_member_scores(
                    member_scores,
                    cfg.variance_norm_mode,
                    cfg.variance_norm_eps,
                    cfg.variance_bound_tanh,
                )
                norm_arr = np.asarray(normalized_member_scores, dtype=float)
                means = np.mean(norm_arr, axis=0).tolist()
                variances = np.var(norm_arr, axis=0, ddof=0).tolist()
        rm_time = time.perf_counter() - t_rm

        if cfg.print_variances:
            var_str = ", ".join(f"{v:.6f}" for v in variances)
            print(f"[example {i}] candidate variances: [{var_str}]")

        methods: List[MethodResult] = []
        methods.append(run_vanilla_bon(candidates, means, variances))

        if cfg.bon_only:
            methods.append(run_soft_bon(candidates, means, variances, cfg.softbon_temperature, rng))
            if not cfg.naive_bon_baseline_only:
                sharpe_temp = cfg.softbon_sharpe_temperature if cfg.softbon_sharpe_temperature is not None else cfg.softbon_temperature
                linear_temp = cfg.softbon_linear_temperature if cfg.softbon_linear_temperature is not None else cfg.softbon_temperature
                pessimism_temp = cfg.softbon_pessimism_temperature if cfg.softbon_pessimism_temperature is not None else cfg.softbon_temperature
                methods.append(run_bon_sharpe(candidates, means, variances, eps=cfg.sharpe_eps))
                methods.append(run_bon_linear(candidates, means, variances, c=cfg.lin_c))
                if cfg.enable_pessimism:
                    methods.append(run_bon_pessimism(candidates, means, variances, beta=cfg.pessimism_beta))
                methods.append(run_softbon_sharpe(candidates, means, variances, sharpe_temp, cfg.sharpe_eps, rng))
                methods.append(run_softbon_linear(candidates, means, variances, cfg.lin_c, linear_temp, rng))
                if cfg.enable_pessimism:
                    methods.append(run_softbon_pessimism(candidates, means, variances, cfg.pessimism_beta, pessimism_temp, rng))
        else:
            methods.append(run_risk_adjusted_bon(candidates, means, variances, lam=cfg.lam))
            methods.append(run_sharpe_ratio_bon(candidates, means, variances, eps=cfg.sharpe_eps))
            methods.append(run_vanilla_seqbon(prompt, generator, scorer, cfg))
            methods.append(run_sharpe_seqbon(prompt, generator, scorer, cfg))
            methods.append(run_lin_seqbon(prompt, generator, scorer, cfg))

            if cfg.controller_type.lower() != "none":
                methods.append(run_controller(
                    prompt, candidates, means, variances,
                    generator, scorer, cfg,
                ))

        for m in methods:
            is_correct = check_correctness(m.selected_answer, gold_answer, cfg.dataset) if has_correctness else None
            row = {
                "example_index": i,
                "prompt": prompt,
                "ground_truth": gold_answer,
                "method": m.method,
                "selected_answer": m.selected_answer,
                "is_correct": is_correct,
                "reward_mean": m.reward_mean,
                "reward_variance": m.reward_variance,
                "score": m.score,
                "n_candidates": m.n_candidates,
                "n_rm_calls": m.n_rm_calls,
                "latency_sec": m.latency_sec + gen_time + rm_time,
                "gen_time_sec": gen_time,
                "rm_time_sec": rm_time,
                "rm_mode": cfg.rm_mode,
                "dataset": cfg.dataset,
                "controller_type": cfg.controller_type,
                "candidate_means": means,
                "candidate_variances": variances,
                "candidate_member_scores": normalized_member_scores if normalized_member_scores is not None else member_scores,
                "raw_candidate_means": raw_means,
                "raw_candidate_variances": raw_variances,
                "raw_candidate_member_scores": member_scores,
                "variance_norm_mode": cfg.variance_norm_mode,
                "variance_norm_eps": cfg.variance_norm_eps,
                "variance_bound_tanh": cfg.variance_bound_tanh,
                "reward_model_names": cfg.reward_model_names,
            }
            rows.append(row)

    return rows


# =========================================================================
# Main Experiment Loop
# =========================================================================

def run_inference(cfg: ExperimentConfig) -> List[dict]:
    """Run all inference strategies and collect results."""

    # Allow dataset names like "GSM8k" while preserving existing defaults.
    cfg.dataset = cfg.dataset.lower()

    print(f"Loading generation model: {cfg.model_name}")
    generator = Generator(
        GenerationConfig(
            model_name=cfg.model_name,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            torch_dtype=cfg.torch_dtype,
            device=cfg.gen_device,
            load_in_4bit=cfg.load_in_4bit,
        )
    )

    # Build RM scorer (trivial or robust)
    scorer = build_scorer(cfg)

    # Load dataset
    print(f"Loading dataset: {cfg.dataset} ({cfg.dataset_split}) ...")
    examples = load_dataset_examples(cfg.dataset, cfg.dataset_split, cfg.max_examples)
    print(f"Running on {len(examples)} examples with N={cfg.bon_n} candidates")
    print(f"RM mode: {cfg.rm_mode} | Controller: {cfg.controller_type} | Dataset: {cfg.dataset}\n")

    has_correctness = cfg.dataset in ("gsm8k", "math500")
    rows = _evaluate_examples(cfg, generator, scorer, examples, desc="Evaluating")

    # Save results
    if cfg.output_path and cfg.save_jsonl:
        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {cfg.output_path}")

    summary = _print_summary(rows, has_correctness)

    # Persist summary metrics next to the configured output stem.
    if cfg.output_path:
        out_path = Path(cfg.output_path)
        stem = out_path.stem
        summary_json_path = out_path.parent / f"{stem}_summary.json"
        summary_txt_path = out_path.parent / f"{stem}_summary.txt"

        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with summary_txt_path.open("w", encoding="utf-8") as f:
            f.write(_summary_to_text(summary, has_correctness))

        print(f"Summary saved to {summary_json_path}")
        print(f"Summary saved to {summary_txt_path}")

    return rows


def _summary_to_text(summary: Dict[str, object], has_correctness: bool = True) -> str:
    """Render summary dict into the same table text shown in console."""
    lines: List[str] = []
    lines.append(f"\n{'=' * 100}")
    lines.append(
        f"  RM Mode: {summary['rm_mode']} | Dataset: {summary['dataset']} | Examples: {summary['n_examples']}"
    )
    lines.append(f"{'=' * 100}")

    header = f"{'Method':<32}"
    if has_correctness:
        header += f" {'Accuracy':>10}"
    header += f" {'Avg mu':>10} {'Avg var':>10} {'RM Calls':>10} {'Latency':>10}"
    lines.append(header)
    lines.append("-" * 100)

    for m in summary["methods"]:  # type: ignore[index]
        line = f"{m['method']:<32}"
        if has_correctness:
            line += f" {m['accuracy']:>9.1%}"
        line += (
            f" {m['avg_mean']:>10.4f} {m['avg_var']:>10.6f} "
            f"{m['avg_rm_calls']:>10.1f} {m['avg_latency_sec']:>9.2f}s"
        )
        lines.append(line)

    lines.append("=" * 100)

    if has_correctness and summary.get("risk_coverage"):
        lines.append("\nRisk-Coverage Analysis")
        for rc in summary["risk_coverage"]:  # type: ignore[index]
            lines.append(
                f"  Top {rc['percentile']}% lowest-variance (var <= {rc['threshold']:.6f}): "
                f"accuracy = {rc['accuracy']:.1%} ({rc['samples']} samples)"
            )

    lines.append("")
    return "\n".join(lines)


def _print_summary(rows: List[dict], has_correctness: bool = True) -> Dict[str, object]:
    """Print summary metrics table with risk-coverage analysis."""
    by_method: Dict[str, List[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    rm_mode = rows[0]["rm_mode"] if rows else "?"
    dataset = rows[0]["dataset"] if rows else "?"

    # Count unique examples (not method variants) using example_index
    n_examples = len(set(r["example_index"] for r in rows)) if rows else 0

    print(f"\n{'=' * 100}")
    print(f"  RM Mode: {rm_mode} | Dataset: {dataset} | Examples: {n_examples}")
    print(f"{'=' * 100}")

    header = f"{'Method':<32}"
    if has_correctness:
        header += f" {'Accuracy':>10}"
    header += f" {'Avg mu':>10} {'Avg var':>10} {'RM Calls':>10} {'Latency':>10}"
    print(header)
    print("-" * 100)

    method_summaries: List[Dict[str, object]] = []

    for method, method_rows in by_method.items():
        avg_mean = np.mean([r["reward_mean"] for r in method_rows])
        avg_var = np.mean([r["reward_variance"] for r in method_rows])
        avg_rm = np.mean([r["n_rm_calls"] for r in method_rows])
        avg_lat = np.mean([r["latency_sec"] for r in method_rows])

        line = f"{method:<32}"
        if has_correctness:
            n_correct = sum(1 for r in method_rows if r["is_correct"])
            accuracy = n_correct / len(method_rows) if method_rows else 0.0
            line += f" {accuracy:>9.1%}"
        line += (
            f" {avg_mean:>10.4f} {avg_var:>10.6f} "
            f"{avg_rm:>10.1f} {avg_lat:>9.2f}s"
        )
        print(line)

        method_summaries.append(
            {
                "method": method,
                "accuracy": accuracy if has_correctness else None,
                "avg_mean": float(avg_mean),
                "avg_var": float(avg_var),
                "avg_rm_calls": float(avg_rm),
                "avg_latency_sec": float(avg_lat),
            }
        )

    print("=" * 100)

    risk_coverage: List[Dict[str, object]] = []

    # Risk-coverage analysis: what % of examples have variance below thresholds
    if has_correctness:
        print(f"\n{'Risk-Coverage Analysis':}")
        all_vars = [r["reward_variance"] for r in rows]
        all_correct = [r["is_correct"] for r in rows if r["is_correct"] is not None]
        if all_vars:
            for pct in [25, 50, 75, 90]:
                threshold = np.percentile(all_vars, pct)
                low_var = [r for r in rows if r["reward_variance"] <= threshold and r["is_correct"] is not None]
                if low_var:
                    acc = np.mean([r["is_correct"] for r in low_var])
                    print(f"  Top {pct}% lowest-variance (var <= {threshold:.6f}): accuracy = {acc:.1%} ({len(low_var)} samples)")
                    risk_coverage.append(
                        {
                            "percentile": pct,
                            "threshold": float(threshold),
                            "accuracy": float(acc),
                            "samples": len(low_var),
                        }
                    )

    print()

    return {
        "rm_mode": rm_mode,
        "dataset": dataset,
        "n_examples": n_examples,
        "methods": method_summaries,
        "risk_coverage": risk_coverage,
    }


# =========================================================================
# CLI
# =========================================================================

app = typer.Typer(help="Run uncertainty-guided test-time inference experiments.")


@app.command()
def main(
    model_name: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace generative model name.",
    ),
    reward_model_names: str = typer.Option(
        "OpenAssistant/reward-model-deberta-v3-base,OpenAssistant/reward-model-deberta-v3-large-v2",
        help="Comma-separated reward model names for trivial ensemble.",
    ),
    rm_mode: str = typer.Option("trivial", help="RM mode: trivial, trivial4, or robust."),
    urm_ensemble_dir: Optional[str] = typer.Option(None, help="Path to trained URM ensemble (for --rm-mode=robust)."),
    urm_base_model: str = typer.Option("microsoft/deberta-v3-base", help="Base model for URM."),
    dataset: str = typer.Option("gsm8k", help="Dataset: gsm8k, math500, or alpaca_eval."),
    controller_type: str = typer.Option("threshold", help="Controller: threshold, thompson, lcb, cvar, or none."),
    max_examples: int = typer.Option(
        50,
        help="Max examples to evaluate. Use 0 (or negative) to run the full split.",
    ),
    output_path: Optional[str] = typer.Option(
        None,
        help=(
            "Output JSONL path. If omitted, uses runs/inf_<rm_mode>_<dataset>.jsonl "
            "for sampled runs and runs/full_<rm_mode>_<dataset>.jsonl when --max-examples <= 0."
        ),
    ),
    save_jsonl: bool = typer.Option(
        True,
        "--save-jsonl/--no-save-jsonl",
        help="Whether to save per-example JSONL rows.",
    ),
    bon_n: int = typer.Option(5, help="Number of BoN candidates (N)."),
    bon_only: bool = typer.Option(False, help="Run only BoN-family methods (no sequential/controller)."),
    naive_bon_baseline_only: bool = typer.Option(False, "--naive-bon-baseline-only/--no-naive-bon-baseline-only", help="In bon-only mode, run only vanilla_bon and soft_bon (single-RM naive baseline)."),
    softbon_temperature: float = typer.Option(1.0, help="SoftBoN sampling temperature."),
    softbon_sharpe_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Sharpe."),
    softbon_linear_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Linear."),
    softbon_pessimism_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Pessimism."),
    sharpe_eps: float = typer.Option(1e-8, help="Stability epsilon in Sharpe denominator (mu / (sigma + eps))."),
    variance_norm_mode: str = typer.Option("none", help="Cross-model variance pre-normalization: none, zscore, or robust_zscore."),
    variance_norm_eps: float = typer.Option(1e-6, help="Stability epsilon for per-model normalization scales."),
    variance_bound_tanh: bool = typer.Option(False, "--variance-bound-tanh/--no-variance-bound-tanh", help="Bound normalized per-model rewards to [-1,1] with tanh before cross-model aggregation."),
    pessimism_beta: float = typer.Option(1.0, help="Inference-time pessimism coefficient beta in (mu - beta*sigma)."),
    enable_pessimism: bool = typer.Option(False, "--enable-pessimism/--no-enable-pessimism", help="Enable BoN-Pessimism and SoftBoN-Pessimism methods in bon-only runs."),
    seq_n: int = typer.Option(5, help="Number of sequential samples."),
    lam: float = typer.Option(0.05, help="Risk penalty lambda for risk-adjusted BoN."),
    gamma_vanilla: float = typer.Option(0.95, help="Threshold gamma for vanilla RF-SeqBoN."),
    gamma_sharpe: float = typer.Option(0.95, help="Threshold gamma for Sharpe RF-SeqBoN."),
    gamma_lin: float = typer.Option(0.95, help="Threshold gamma for Lin_seqBon."),
    lin_c: float = typer.Option(1.0, help="Multiplier c for Lin_seqBon (mu - c*sigma)."),
    history_budget: int = typer.Option(2, help="Number of previous attempts to keep in prompt history."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    max_new_tokens: int = typer.Option(512, help="Max tokens per generation."),
    torch_dtype: str = typer.Option("bfloat16", help="Model dtype."),
    gen_device: int = typer.Option(0, help="GPU index for generation model."),
    load_in_4bit: bool = typer.Option(False, help="4-bit quantization."),
    reward_load_in_4bit: bool = typer.Option(False, help="4-bit quantization for reward models."),
    print_variances: bool = typer.Option(False, help="Print per-example candidate variances."),
    seed: int = typer.Option(42, help="Random seed for SoftBoN stochastic selection."),
    u_low: float = typer.Option(0.05, help="Controller low uncertainty threshold."),
    u_high: float = typer.Option(0.2, help="Controller high uncertainty threshold."),
):
    """Run the full experiment pipeline."""
    normalized_mode = rm_mode.lower()
    normalized_dataset = dataset.lower()
    resolved_max_examples: Optional[int] = None if max_examples <= 0 else max_examples
    run_scope = "full" if resolved_max_examples is None else "inf"
    resolved_output_path = output_path or f"runs/{run_scope}_{normalized_mode}_{normalized_dataset}.jsonl"
    parsed_reward_models = [s.strip() for s in reward_model_names.split(",") if s.strip()]
    default_trivial_models = [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-deberta-v3-large-v2",
    ]

    # In trivial4 mode, default to the paper 4-RM preset unless the user explicitly provides names.
    if normalized_mode == "trivial4" and parsed_reward_models == default_trivial_models:
        parsed_reward_models = list(TRIVIAL_4_RM_PRESET)

    cfg = ExperimentConfig(
        model_name=model_name,
        reward_model_names=parsed_reward_models,
        reward_load_in_4bit=reward_load_in_4bit,
        rm_mode=rm_mode,
        urm_ensemble_dir=urm_ensemble_dir,
        urm_base_model=urm_base_model,
        dataset=dataset,
        controller_type=controller_type,
        max_examples=resolved_max_examples,
        output_path=resolved_output_path,
        save_jsonl=save_jsonl,
        bon_n=bon_n,
        bon_only=bon_only,
        naive_bon_baseline_only=naive_bon_baseline_only,
        softbon_temperature=softbon_temperature,
        softbon_sharpe_temperature=softbon_sharpe_temperature,
        softbon_linear_temperature=softbon_linear_temperature,
        softbon_pessimism_temperature=softbon_pessimism_temperature,
        sharpe_eps=sharpe_eps,
        variance_norm_mode=variance_norm_mode,
        variance_norm_eps=variance_norm_eps,
        variance_bound_tanh=variance_bound_tanh,
        pessimism_beta=pessimism_beta,
        enable_pessimism=enable_pessimism,
        seq_n=seq_n,
        lam=lam,
        gamma_vanilla=gamma_vanilla,
        gamma_sharpe=gamma_sharpe,
        gamma_lin=gamma_lin,
        lin_c=lin_c,
        history_budget=history_budget,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        gen_device=gen_device,
        load_in_4bit=load_in_4bit,
        print_variances=print_variances,
        seed=seed,
        u_low=u_low,
        u_high=u_high,
    )

    run_inference(cfg)


@app.command("gamma-ablation")
def gamma_ablation(
    model_name: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HuggingFace generative model name."),
    rm_mode: str = typer.Option("robust", help="Use robust mode for ablations."),
    urm_ensemble_dir: str = typer.Option("models/urm_ensemble", help="Path to trained URM ensemble."),
    urm_base_model: str = typer.Option("microsoft/deberta-v3-base", help="Base model for URM."),
    dataset: str = typer.Option("gsm8k", help="Dataset for ablation."),
    max_examples: int = typer.Option(100, help="Max examples for each gamma point."),
    bon_n: int = typer.Option(5, help="Number of BoN candidates (N)."),
    seq_n: int = typer.Option(5, help="Number of sequential samples."),
    lam: float = typer.Option(0.05, help="Risk penalty lambda for risk-adjusted BoN."),
    lin_c: float = typer.Option(1.0, help="Coefficient c for Lin_seqBon."),
    history_budget: int = typer.Option(2, help="History budget for sequential prompts."),
    gammas: str = typer.Option("0.5,0.6,0.7,0.8,0.85,0.9,0.95", help="Comma-separated gamma list."),
    output_dir: str = typer.Option("runs/Ablation", help="Directory for ablation summaries."),
    include_controller: bool = typer.Option(
        False,
        "--include-controller/--no-include-controller",
        help="Include controller method in ablation runs.",
    ),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    max_new_tokens: int = typer.Option(512, help="Max tokens per generation."),
    torch_dtype: str = typer.Option("bfloat16", help="Model dtype."),
    gen_device: int = typer.Option(0, help="GPU index for generation model."),
    load_in_4bit: bool = typer.Option(False, help="4-bit quantization."),
):
    """Run gamma sweep in a single process (loads model/RM/dataset once)."""
    parsed_gammas = [float(g.strip()) for g in gammas.split(",") if g.strip()]
    dataset = dataset.lower()
    resolved_max_examples: Optional[int] = None if max_examples <= 0 else max_examples
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if rm_mode.lower() != "robust":
        raise ValueError("gamma-ablation is intended for robust RM mode; use --rm-mode robust")

    base_cfg = ExperimentConfig(
        model_name=model_name,
        rm_mode=rm_mode,
        urm_ensemble_dir=urm_ensemble_dir,
        urm_base_model=urm_base_model,
        dataset=dataset,
        dataset_split="test",
        max_examples=resolved_max_examples,
        bon_n=bon_n,
        seq_n=seq_n,
        lam=lam,
        lin_c=lin_c,
        history_budget=history_budget,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        gen_device=gen_device,
        load_in_4bit=load_in_4bit,
        controller_type="threshold" if include_controller else "none",
        save_jsonl=False,
    )

    print(f"Loading generation model once: {base_cfg.model_name}")
    generator = Generator(
        GenerationConfig(
            model_name=base_cfg.model_name,
            max_new_tokens=base_cfg.max_new_tokens,
            temperature=base_cfg.temperature,
            top_p=base_cfg.top_p,
            torch_dtype=base_cfg.torch_dtype,
            device=base_cfg.gen_device,
            load_in_4bit=base_cfg.load_in_4bit,
        )
    )
    scorer = build_scorer(base_cfg)
    print(f"Loading dataset once: {base_cfg.dataset} ({base_cfg.dataset_split})")
    examples = load_dataset_examples(base_cfg.dataset, base_cfg.dataset_split, base_cfg.max_examples)
    print(f"Ablation examples: {len(examples)} | gammas: {parsed_gammas}\n")

    for gamma in parsed_gammas:
        cfg = ExperimentConfig(**{**base_cfg.__dict__})
        cfg.gamma_vanilla = gamma
        cfg.gamma_sharpe = gamma
        cfg.gamma_lin = gamma
        cfg.output_path = str(out_dir / f"Gamma_ablation_{dataset.upper()}_g{gamma}.jsonl")

        print(f"Running gamma={gamma}...")
        rows = _evaluate_examples(cfg, generator, scorer, examples, desc=f"Gamma={gamma}")
        summary = _print_summary(rows, cfg.dataset in ("gsm8k", "math500"))

        out_path = Path(cfg.output_path)
        stem = out_path.stem
        summary_json_path = out_path.parent / f"{stem}_summary.json"
        summary_txt_path = out_path.parent / f"{stem}_summary.txt"

        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with summary_txt_path.open("w", encoding="utf-8") as f:
            f.write(_summary_to_text(summary, cfg.dataset in ("gsm8k", "math500")))

        print(f"Summary saved to {summary_json_path}")
        print(f"Summary saved to {summary_txt_path}\n")


@app.command("bon-suite")
def bon_suite(
    model_name: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HuggingFace generative model name."),
    reward_model_names: str = typer.Option(
        ",".join(TRIVIAL_4_RM_PRESET),
        help="Comma-separated reward model names. Defaults to OASST/Gemma-RM/Llama-RM/Armo-RM preset.",
    ),
    max_examples: int = typer.Option(
        0,
        help="Max GSM8K test examples to evaluate. Use 0 (or negative) for full split.",
    ),
    bon_n: int = typer.Option(8, help="Number of BoN candidates (N)."),
    naive_bon_baseline_only: bool = typer.Option(False, "--naive-bon-baseline-only/--no-naive-bon-baseline-only", help="Run only vanilla_bon and soft_bon."),
    lin_c: float = typer.Option(1.0, help="Coefficient c for SoftBoN-Linear utility (mu - c*sigma)."),
    softbon_temperature: float = typer.Option(1.0, help="SoftBoN sampling temperature."),
    softbon_sharpe_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Sharpe."),
    softbon_linear_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Linear."),
    softbon_pessimism_temperature: Optional[float] = typer.Option(None, help="Optional temperature override for SoftBoN-Pessimism."),
    sharpe_eps: float = typer.Option(1e-8, help="Stability epsilon in Sharpe denominator (mu / (sigma + eps))."),
    variance_norm_mode: str = typer.Option("none", help="Cross-model variance pre-normalization: none, zscore, or robust_zscore."),
    variance_norm_eps: float = typer.Option(1e-6, help="Stability epsilon for per-model normalization scales."),
    variance_bound_tanh: bool = typer.Option(False, "--variance-bound-tanh/--no-variance-bound-tanh", help="Bound normalized per-model rewards to [-1,1] with tanh before cross-model aggregation."),
    pessimism_beta: float = typer.Option(1.0, help="Inference-time pessimism coefficient beta in (mu - beta*sigma)."),
    enable_pessimism: bool = typer.Option(False, "--enable-pessimism/--no-enable-pessimism", help="Enable BoN-Pessimism and SoftBoN-Pessimism methods."),
    output_path: Optional[str] = typer.Option(
        None,
        help=(
            "Output JSONL path. If omitted, uses runs/inf_trivial4_gsm8k_bonsuite.jsonl "
            "or runs/full_trivial4_gsm8k_bonsuite.jsonl for full split."
        ),
    ),
    save_jsonl: bool = typer.Option(True, "--save-jsonl/--no-save-jsonl", help="Whether to save per-example JSONL rows."),
    reward_batch_size: int = typer.Option(4, help="Batch size used when scoring candidates with reward models."),
    reward_load_in_4bit: bool = typer.Option(True, help="Enable 4-bit quantization for reward models."),
    temperature: float = typer.Option(0.7, help="Generation temperature."),
    max_new_tokens: int = typer.Option(512, help="Max tokens per generation."),
    torch_dtype: str = typer.Option("bfloat16", help="Generation model dtype."),
    gen_device: int = typer.Option(0, help="GPU index for generation model."),
    load_in_4bit: bool = typer.Option(False, help="4-bit quantization for generation model."),
    print_variances: bool = typer.Option(True, help="Print per-example candidate variances."),
    seed: int = typer.Option(42, help="Random seed for SoftBoN stochastic selection."),
):
    """Run BoN + SoftBoN baselines on GSM8K test split using the 4-RM trivial ensemble."""
    resolved_max_examples: Optional[int] = None if max_examples <= 0 else max_examples
    run_scope = "full" if resolved_max_examples is None else "inf"
    resolved_output_path = output_path or f"runs/{run_scope}_trivial4_gsm8k_bonsuite.jsonl"

    cfg = ExperimentConfig(
        model_name=model_name,
        rm_mode="trivial4",
        reward_model_names=[s.strip() for s in reward_model_names.split(",") if s.strip()],
        reward_batch_size=reward_batch_size,
        reward_load_in_4bit=reward_load_in_4bit,
        dataset="gsm8k",
        dataset_split="test",
        max_examples=resolved_max_examples,
        bon_n=bon_n,
        bon_only=True,
        naive_bon_baseline_only=naive_bon_baseline_only,
        lin_c=lin_c,
        softbon_temperature=softbon_temperature,
        softbon_sharpe_temperature=softbon_sharpe_temperature,
        softbon_linear_temperature=softbon_linear_temperature,
        softbon_pessimism_temperature=softbon_pessimism_temperature,
        sharpe_eps=sharpe_eps,
        variance_norm_mode=variance_norm_mode,
        variance_norm_eps=variance_norm_eps,
        variance_bound_tanh=variance_bound_tanh,
        pessimism_beta=pessimism_beta,
        enable_pessimism=enable_pessimism,
        output_path=resolved_output_path,
        save_jsonl=save_jsonl,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        gen_device=gen_device,
        load_in_4bit=load_in_4bit,
        print_variances=print_variances,
        seed=seed,
        controller_type="none",
    )

    run_inference(cfg)


@app.command("ideas-suite")
def ideas_suite(
    model_name: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HuggingFace generative model name."),
    reward_model_names: str = typer.Option(
        "OpenAssistant/reward-model-deberta-v3-large-v2,NCSOFT/Llama-3-OffsetBias-RM-8B,RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="Comma-separated reward model names for 3-RM trivial ensemble.",
    ),
    dataset: str = typer.Option("gsm8k", help="Dataset: gsm8k or math500."),
    max_examples: int = typer.Option(50, help="Max examples. Use 0 for full split."),
    bon_n: int = typer.Option(8, help="Number of BoN candidates (N)."),
    seq_n: int = typer.Option(5, help="Number of sequential rounds."),
    mob_n_bootstrap: int = typer.Option(50, help="Number of bootstrap subsamples for MoB."),
    mob_subsample_frac: float = typer.Option(0.6, help="Fraction of N in each MoB subset."),
    adaptive_burnin: int = typer.Option(3, help="Number of burn-in samples for adaptive SeqBoN."),
    adaptive_percentile: float = typer.Option(50.0, help="Percentile for adaptive gamma calibration."),
    history_budget: int = typer.Option(2, help="History budget for sequential prompts."),
    variance_norm_mode: str = typer.Option("robust_zscore", help="Normalization: none, zscore, robust_zscore."),
    variance_bound_tanh: bool = typer.Option(True, "--variance-bound-tanh/--no-variance-bound-tanh", help="Apply tanh bounding."),
    variance_norm_eps: float = typer.Option(1e-6, help="Normalization epsilon."),
    sharpe_eps: float = typer.Option(1e-6, help="Sharpe denominator epsilon."),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path."),
    temperature: float = typer.Option(0.7, help="Generation temperature."),
    max_new_tokens: int = typer.Option(512, help="Max tokens per generation."),
    gen_device: int = typer.Option(0, help="GPU index."),
    load_in_4bit: bool = typer.Option(False, help="4-bit quantization for gen model."),
    reward_load_in_4bit: bool = typer.Option(True, help="4-bit quantization for RMs."),
    seed: int = typer.Option(42, help="Random seed."),
    include_seq: bool = typer.Option(False, "--include-seq/--no-include-seq", help="Include sequential methods (slower)."),
):
    """Run new research ideas: MoB, WBoN, and Adaptive SeqBoN alongside baselines."""
    dataset = dataset.lower()
    resolved_max = None if max_examples <= 0 else max_examples
    run_scope = "full" if resolved_max is None else "inf"
    resolved_output = output_path or f"runs/{run_scope}_ideas_suite_{dataset}.jsonl"
    parsed_rms = [s.strip() for s in reward_model_names.split(",") if s.strip()]

    cfg = ExperimentConfig(
        model_name=model_name,
        rm_mode="trivial4",
        reward_model_names=parsed_rms,
        reward_batch_size=1,
        reward_load_in_4bit=reward_load_in_4bit,
        dataset=dataset,
        dataset_split="test",
        max_examples=resolved_max,
        bon_n=bon_n,
        seq_n=seq_n,
        history_budget=history_budget,
        variance_norm_mode=variance_norm_mode,
        variance_norm_eps=variance_norm_eps,
        variance_bound_tanh=variance_bound_tanh,
        sharpe_eps=sharpe_eps,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        gen_device=gen_device,
        load_in_4bit=load_in_4bit,
        output_path=resolved_output,
        save_jsonl=True,
        controller_type="none",
        seed=seed,
    )

    print(f"Loading generation model: {cfg.model_name}")
    generator = Generator(
        GenerationConfig(
            model_name=cfg.model_name,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            torch_dtype=cfg.torch_dtype,
            device=cfg.gen_device,
            load_in_4bit=cfg.load_in_4bit,
        )
    )
    scorer = build_scorer(cfg)
    print(f"Loading dataset: {cfg.dataset}")
    examples = load_dataset_examples(cfg.dataset, cfg.dataset_split, cfg.max_examples)
    print(f"Ideas Suite: {len(examples)} examples | N={cfg.bon_n}")
    print(f"Methods: vanilla_bon, bon_sharpe, mob_vanilla, mob_sharpe, wbon_inv_var, wbon_sharpe"
          + (", adaptive_seqbon" if include_seq else ""))

    has_correctness = cfg.dataset in ("gsm8k", "math500")
    system_msg = _dataset_system_message(cfg.dataset)
    rng = np.random.default_rng(cfg.seed)
    rows: List[dict] = []

    for i, example in enumerate(tqdm(examples, desc="Ideas Suite")):
        prompt = example["question"]
        gold_answer = example.get("answer", "")

        t_gen = time.perf_counter()
        candidates = generator.generate(prompt, n=cfg.bon_n, system_msg=system_msg)
        gen_time = time.perf_counter() - t_gen

        t_rm = time.perf_counter()
        raw_means, raw_variances = scorer.score(prompt, candidates)
        member_scores = getattr(scorer, "last_member_scores", None)
        means, variances = raw_means, raw_variances

        # Apply normalization
        if cfg.variance_norm_mode.lower() != "none" and member_scores is not None:
            normalized_ms = _normalize_member_scores(
                member_scores, cfg.variance_norm_mode, cfg.variance_norm_eps, cfg.variance_bound_tanh,
            )
            norm_arr = np.asarray(normalized_ms, dtype=float)
            means = np.mean(norm_arr, axis=0).tolist()
            variances = np.var(norm_arr, axis=0, ddof=0).tolist()
        rm_time = time.perf_counter() - t_rm

        # Run all methods
        methods: List[MethodResult] = []
        methods.append(run_vanilla_bon(candidates, means, variances))
        methods.append(run_bon_sharpe(candidates, means, variances, eps=cfg.sharpe_eps))
        methods.append(run_mob_bon(candidates, means, variances, n_bootstrap=mob_n_bootstrap,
                                   subsample_frac=mob_subsample_frac, use_variance_weighting=False, rng=rng))
        methods.append(run_mob_bon(candidates, means, variances, n_bootstrap=mob_n_bootstrap,
                                   subsample_frac=mob_subsample_frac, use_variance_weighting=True,
                                   eps=cfg.sharpe_eps, rng=rng))

        # WBoN needs raw member_scores (before normalization)
        if member_scores is not None:
            methods.append(run_wbon(candidates, member_scores, eps=cfg.sharpe_eps))
            methods.append(run_wbon_sharpe(candidates, member_scores, eps=cfg.sharpe_eps))

        # Adaptive SeqBoN (optional — slow)
        if include_seq:
            methods.append(run_adaptive_seqbon(
                prompt, generator, scorer, cfg,
                n_burnin=adaptive_burnin, gamma_percentile=adaptive_percentile,
            ))

        for m in methods:
            is_correct = check_correctness(m.selected_answer, gold_answer, cfg.dataset) if has_correctness else None
            row = {
                "example_index": i,
                "prompt": prompt,
                "ground_truth": gold_answer,
                "method": m.method,
                "selected_answer": m.selected_answer,
                "is_correct": is_correct,
                "reward_mean": m.reward_mean,
                "reward_variance": m.reward_variance,
                "score": m.score,
                "n_candidates": m.n_candidates,
                "n_rm_calls": m.n_rm_calls,
                "latency_sec": m.latency_sec + gen_time + rm_time,
                "rm_mode": cfg.rm_mode,
                "dataset": cfg.dataset,
            }
            rows.append(row)

    # Save results
    if cfg.output_path and cfg.save_jsonl:
        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {cfg.output_path}")

    summary = _print_summary(rows, has_correctness)

    if cfg.output_path:
        out_path = Path(cfg.output_path)
        stem = out_path.stem
        summary_json_path = out_path.parent / f"{stem}_summary.json"
        summary_txt_path = out_path.parent / f"{stem}_summary.txt"
        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with summary_txt_path.open("w", encoding="utf-8") as f:
            f.write(_summary_to_text(summary, has_correctness))
        print(f"Summary saved to {summary_json_path}")
        print(f"Summary saved to {summary_txt_path}")


if __name__ == "__main__":
    # Backward compatibility: allow legacy invocation without explicit subcommand.
    # Examples that remain valid:
    #   python -m src.experiments.run_risk_aware_inference --rm-mode robust ...
    #   python -m src.experiments.run_risk_aware_inference
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "main")
    app()

