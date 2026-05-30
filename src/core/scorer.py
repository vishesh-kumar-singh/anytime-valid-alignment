"""Reward-model scoring interface and implementations.

Supports two backends:
  - TrivialEnsembleScorer : wraps pre-trained scalar RM ensemble
  - RobustURMScorer       : wraps uncertainty-aware RM ensemble (trained)

Both conform to the ``RMScorer`` protocol, so downstream code never needs
to know which backend is active.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from .config import BaseExperimentConfig


# ── Preset RM Ensembles ──────────────────────────────────────────────────

TRIVIAL_4_RM_PRESET: List[str] = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
    "NCSOFT/Llama-3-OffsetBias-RM-8B",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
]

TRIVIAL_5_RM_PRESET: List[str] = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "NCSOFT/Llama-3-OffsetBias-RM-8B",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    "sfairXC/FsfairX-LLaMA3-RM-v0.1",
]


# ── Protocol ─────────────────────────────────────────────────────────────

class RMScorer(Protocol):
    """Unified interface for both trivial and robust RM ensembles."""

    def score(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        """Return ``(means, variances)`` for each candidate."""
        ...

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        """Return scalar rewards for each candidate."""
        ...


# ── Trivial Ensemble ─────────────────────────────────────────────────────

class TrivialEnsembleScorer:
    """Wraps the pre-trained scalar RM ensemble."""

    def __init__(
        self,
        model_names: List[str],
        batch_size: int = 8,
        load_in_4bit: bool = False,
    ) -> None:
        from ..reward_modeling.inference import RewardEnsemble

        self.ensemble = RewardEnsemble(model_names, batch_size=batch_size, load_in_4bit=load_in_4bit)
        self.last_member_scores: Optional[List[List[float]]] = None

    def score(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        stats = self.ensemble.score(prompt, candidates)
        self.last_member_scores = stats.member_scores
        return stats.means, stats.variances

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        return self.ensemble.score_scalar(prompt, candidates)


# ── Robust URM Ensemble ──────────────────────────────────────────────────

class RobustURMScorer:
    """Wraps the uncertainty-aware RM ensemble (trained with MLE loss)."""

    def __init__(
        self,
        ensemble_dir: str,
        base_model_name: str = "microsoft/deberta-v3-base",
    ) -> None:
        from ..reward_modeling.uncertainty_rm import URMEnsemble

        model_paths = sorted(Path(ensemble_dir).glob("member_*"))
        if not model_paths:
            model_paths = sorted(Path(ensemble_dir).glob("model_*"))
        if not model_paths:
            raise FileNotFoundError(
                f"No URM model directories found in {ensemble_dir} "
                "(expected member_* or model_*). "
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


# ── Factory ──────────────────────────────────────────────────────────────

def build_scorer(cfg: BaseExperimentConfig) -> RMScorer:
    """Construct the appropriate RM scorer from experiment config."""
    if cfg.rm_mode == "robust":
        if not cfg.urm_ensemble_dir:
            raise ValueError("--urm-ensemble-dir required when --rm-mode=robust")
        print(f"Loading robust URM ensemble from: {cfg.urm_ensemble_dir}")
        return RobustURMScorer(cfg.urm_ensemble_dir, cfg.urm_base_model)

    if cfg.rm_mode == "trivial4":
        model_names = cfg.reward_model_names or list(TRIVIAL_4_RM_PRESET)
        print(f"Loading trivial4-mode ensemble ({len(model_names)} models)")
        return TrivialEnsembleScorer(model_names, cfg.reward_batch_size, cfg.reward_load_in_4bit)

    # Default: trivial
    print(f"Loading trivial RM ensemble: {cfg.reward_model_names}")
    return TrivialEnsembleScorer(cfg.reward_model_names, cfg.reward_batch_size, cfg.reward_load_in_4bit)
