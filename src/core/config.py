"""Experiment configuration for Test-Time Alignment.

Centralises all hyperparameters so that every hypothesis module, CLI entry
point, and utility script refers to a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseExperimentConfig:
    """Configuration shared across all experiment types.

    Attributes are grouped by subsystem:  generation, reward modelling,
    selection, dataset, and output.  Each hypothesis may use only a subset.
    """

    # ── Generation ────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    gen_device: int = 0

    # ── BoN (Best-of-N) ──────────────────────────────────────────────
    bon_n: int = 8
    lam: float = 0.05
    softbon_temperature: float = 1.0
    softbon_sharpe_temperature: Optional[float] = None
    softbon_linear_temperature: Optional[float] = None
    softbon_pessimism_temperature: Optional[float] = None
    sharpe_eps: float = 1e-8
    variance_norm_mode: str = "none"
    variance_norm_eps: float = 1e-6
    variance_bound_tanh: bool = False
    pessimism_beta: float = 1.0
    enable_pessimism: bool = False
    naive_bon_baseline_only: bool = False
    bon_only: bool = False
    print_variances: bool = False
    seed: int = 42
    lin_c: float = 1.0

    # ── Sequential ────────────────────────────────────────────────────
    seq_n: int = 5
    gamma_vanilla: float = 0.95
    gamma_sharpe: float = 0.95
    gamma_lin: float = 0.95
    history_budget: int = 2

    # ── Controller ────────────────────────────────────────────────────
    controller_type: str = "threshold"
    uncertainty_top_k: int = 3
    u_low: float = 0.05
    u_high: float = 0.2

    # ── Reward Modelling ──────────────────────────────────────────────
    rm_mode: str = "trivial"
    reward_model_names: List[str] = field(default_factory=lambda: [
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        "NCSOFT/Llama-3-OffsetBias-RM-8B",
    ])
    reward_batch_size: int = 8
    reward_load_in_4bit: bool = False
    urm_ensemble_dir: Optional[str] = None
    urm_base_model: str = "microsoft/deberta-v3-base"

    # ── Dataset ───────────────────────────────────────────────────────
    dataset: str = "gsm8k"
    dataset_split: str = "test"
    max_examples: Optional[int] = None

    # ── Output ────────────────────────────────────────────────────────
    experiment_name: str = ""
    category: str = "parallel"  # "parallel" | "sequential"
    hypothesis_id: str = ""
    output_dir: str = ""
    save_jsonl: bool = True

    @property
    def resolved_output_dir(self) -> str:
        """Compute ``runs/{category}/{experiment_name}/`` if not set."""
        if self.output_dir:
            return self.output_dir
        if self.experiment_name:
            return f"runs/{self.category}/{self.experiment_name}"
        return "runs"
