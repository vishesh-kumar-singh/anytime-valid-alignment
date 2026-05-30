"""Core infrastructure for Test-Time Alignment experiments.

Provides shared configuration, scoring, data loading, normalization,
result management, and logging utilities used by all hypothesis modules.
"""

from .config import BaseExperimentConfig
from .data import check_correctness, dataset_system_message, load_dataset_examples
from .logger import ExperimentLogger
from .normalize import normalize_member_scores
from .results import MethodResult, ResultsManager
from .scorer import (
    TRIVIAL_4_RM_PRESET,
    TRIVIAL_5_RM_PRESET,
    RMScorer,
    TrivialEnsembleScorer,
    build_scorer,
)

__all__ = [
    "BaseExperimentConfig",
    "RMScorer",
    "TrivialEnsembleScorer",
    "build_scorer",
    "TRIVIAL_4_RM_PRESET",
    "TRIVIAL_5_RM_PRESET",
    "load_dataset_examples",
    "check_correctness",
    "dataset_system_message",
    "normalize_member_scores",
    "MethodResult",
    "ResultsManager",
    "ExperimentLogger",
]
