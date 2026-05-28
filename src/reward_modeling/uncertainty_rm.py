"""
Uncertainty-aware Reward Model (URM) — based on:
"Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown"
(arXiv:2410.00847, Lou et al. 2025)

Key idea from the paper:
  Each individual URM has a **probabilistic value head** that outputs parameters
  of a Normal distribution N(μ_i, exp(2σ_i)) for each of n attributes (e.g.
  helpfulness, correctness, coherence, complexity, verbosity).

  - Aleatoric uncertainty (inherent noise in human preferences) is captured by
    each single URM's variance.
  - Epistemic uncertainty (model ignorance) is captured by ensemble disagreement.

Our implementation:
  - We use a smaller base (DeBERTa-v3-base/large) instead of Llama3-8B to be
    practical on a single 48GB GPU.
  - Probabilistic value head replaces the deterministic scalar head.
  - Trained with MLE loss on HelpSteer2 multi-attribute annotations.
  - At inference: forward pass → (μ_1..μ_n, σ_1..σ_n) → weighted sum → scalar
    reward R with aleatoric uncertainty u = Σ exp(2σ_i).
  - Ensemble of K URMs → epistemic uncertainty from disagreement.

Two RM modes available in the project:
  1. "trivial" (inference.py) — Pre-trained scalar RMs, ensemble disagreement
     for uncertainty. No training needed. Quick baseline.
  2. "robust" (this file) — Proper URM with probabilistic head, trained on
     HelpSteer2. Better uncertainty estimates. Requires training.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _resolve_hf_source(model_name_or_path: str) -> Tuple[str, bool]:
    """Resolve a local model source if available in path or HF cache.

    Returns:
        (source, local_only)
    """
    local_path = Path(model_name_or_path).expanduser()
    if local_path.exists():
        return str(local_path), True

    if "/" in model_name_or_path:
        repo_dir = f"models--{model_name_or_path.replace('/', '--')}"
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        snapshots_dir = hf_home / "hub" / repo_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                return str(latest_snapshot), True

    return model_name_or_path, False


# =========================================================================
# Probabilistic Value Head (from the paper)
# =========================================================================

class ProbabilisticValueHead(nn.Module):
    """Value head that outputs (μ, log_σ) for each of n attributes.

    Architecture (from paper):
        hidden_state -> Linear(hidden, n_attributes) -> μ
        hidden_state -> Linear(hidden, n_attributes) -> log_σ

    The distribution for attribute i is N(μ_i, exp(2·log_σ_i)).
    """

    def __init__(self, hidden_size: int, n_attributes: int = 5):
        super().__init__()
        self.n_attributes = n_attributes
        # Separate heads for μ and log_σ
        self.mu_head = nn.Linear(hidden_size, n_attributes)
        self.log_sigma_head = nn.Linear(hidden_size, n_attributes)

        # Initialize log_sigma to output small positive values (low initial uncertainty)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)
        nn.init.xavier_uniform_(self.log_sigma_head.weight, gain=0.01)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (batch, hidden_size) — last token's hidden state.

        Returns:
            mu: (batch, n_attributes)
            log_sigma: (batch, n_attributes)
        """
        # full precision for the value head's linear layers.
        hidden_state = hidden_state.float()
        mu = self.mu_head(hidden_state)
        log_sigma = self.log_sigma_head(hidden_state)
        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=5.0)
        return mu, log_sigma


# =========================================================================
# Gating Layer for combining multi-attribute scores 
# =========================================================================

class GatingLayer(nn.Module):
    """Learned gating layer to combine multi-attribute scores into scalar reward.

    Architecture (from paper): 2 hidden layers, SELU activation.
    Alternatively, use pre-determined weights (simpler, no extra training).
    """

    def __init__(self, n_attributes: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_attributes, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, attribute_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attribute_scores: (batch, n_attributes)
        Returns:
            scalar_reward: (batch,)
        """
        return self.net(attribute_scores).squeeze(-1)


# =========================================================================
# Uncertainty-aware Reward Model
# =========================================================================

# HelpSteer2 attribute names and their default weights (from Nemotron-4 paper)
HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
# These weights are from the Nemotron-4 340B paper (Adler et al. 2024)
# Used when gating layer is not trained
DEFAULT_ATTRIBUTE_WEIGHTS = {
    "helpfulness": 0.65,
    "correctness": 0.8,
    "coherence": 0.45,
    "complexity": 0.55,
    "verbosity": -0.4,
}


class UncertaintyRewardModel(nn.Module):
    """Single URM with probabilistic value head.

    Forward pass for (prompt, response):
        1. Encode with base model → last hidden state h
        2. Probabilistic head → (μ_1..μ_n, log_σ_1..log_σ_n) per attribute
        3. Weighted sum → scalar reward R with uncertainty u

    The model can output:
        - Per-attribute (μ, σ²) distributions
        - Scalar reward R (weighted combination)
        - Aleatoric uncertainty u = Σ exp(2·log_σ_i)
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-base",
        n_attributes: int = 5,
        attribute_weights: Optional[Dict[str, float]] = None,
        use_gating: bool = False,
        gating_hidden: int = 256,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.n_attributes = n_attributes
        self.use_gating = use_gating

        base_source, local_only = _resolve_hf_source(base_model_name)

        # Load base encoder — force float32 to avoid NaN from float16
        # disentangled attention on long sequences
        config = AutoConfig.from_pretrained(base_source, local_files_only=local_only)
        self.encoder = AutoModel.from_pretrained(
            base_source,
            config=config,
            torch_dtype=torch.float32,
            local_files_only=local_only,
        )
        hidden_size = config.hidden_size

        # Probabilistic value head
        self.value_head = ProbabilisticValueHead(hidden_size, n_attributes)

        # Reward combination: gating layer or fixed weights
        if use_gating:
            self.gating = GatingLayer(n_attributes, gating_hidden)
        else:
            self.gating = None
            # Set fixed attribute weights
            if attribute_weights is None:
                attribute_weights = DEFAULT_ATTRIBUTE_WEIGHTS
            weight_list = [
                attribute_weights.get(attr, 0.5)
                for attr in HELPSTEER_ATTRIBUTES[:n_attributes]
            ]
            self.register_buffer(
                "attr_weights",
                torch.tensor(weight_list, dtype=torch.float32),
            )

    def _get_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract last non-padding token's hidden state."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        # Get the last non-padding token for each sequence
        # attention_mask: 1 for real tokens, 0 for padding
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, seq_lengths]  # (batch, hidden)
        # Ensure float32 regardless of encoder precision
        return last_hidden.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_per_attribute: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with:
            "reward": (batch,) — scalar reward (weighted sum of attribute means)
            "mu": (batch, n_attr) — per-attribute means
            "log_sigma": (batch, n_attr) — per-attribute log standard deviations
            "variance": (batch,) — aleatoric uncertainty Σ exp(2·log_σ_i)
        """
        last_hidden = self._get_last_hidden(input_ids, attention_mask)
        mu, log_sigma = self.value_head(last_hidden)

        # Aleatoric uncertainty: sum of variances across attributes
        per_attr_var = torch.exp(2.0 * log_sigma)  # (batch, n_attr)
        aleatoric_var = per_attr_var.sum(dim=-1)    # (batch,)

        # Scalar reward
        if self.use_gating and self.gating is not None:
            reward = self.gating(mu)
        else:
            reward = (mu * self.attr_weights.unsqueeze(0)).sum(dim=-1)

        result = {
            "reward": reward,
            "mu": mu,
            "log_sigma": log_sigma,
            "variance": aleatoric_var,
        }
        return result

class MLELoss(nn.Module):
    """Maximum Likelihood Estimation loss for probabilistic value head.

    L = -E[ Σ_i log P_θ(R_i | x, y) ]

    where P_θ(R_i | x, y) = N(R_i; μ_i, exp(2·log_σ_i))

    This is the negative log-likelihood of the true attribute scores
    under the predicted Normal distributions.
    """

    def forward(
        self,
        mu: torch.Tensor,           # (batch, n_attributes) predicted means
        log_sigma: torch.Tensor,     # (batch, n_attributes) predicted log std
        targets: torch.Tensor,       # (batch, n_attributes) ground-truth scores
    ) -> torch.Tensor:
        log_var = 2.0 * log_sigma
        inv_var = torch.exp(-log_var)          # 1 / σ², stays finite because log_sigma is clamped
        sq_diff = (targets - mu).pow(2)
        nll = 0.5 * (1.8378770664093453 + log_var + sq_diff * inv_var)  # 1.8378... = log(2π)
        return nll.mean()


# =========================================================================
# Ensemble of URMs for epistemic uncertainty
# =========================================================================

@dataclass
class URMStats:
    """Reward statistics from Uncertainty-aware RM ensemble."""
    means: List[float]           # Per-candidate mean reward μ
    variances: List[float]       # Per-candidate total variance (aleatoric + epistemic)
    aleatoric_vars: List[float]  # Per-candidate aleatoric variance (from single model)
    epistemic_vars: List[float]  # Per-candidate epistemic variance (from ensemble disagreement)


class URMEnsemble:
    """Ensemble of K Uncertainty-aware Reward Models.

    For each candidate response:
        1. Each URM_k produces: reward_k, aleatoric_var_k
        2. Epistemic uncertainty = variance of rewards across K models
        3. Total variance = mean(aleatoric) + epistemic

    This gives us two types of uncertainty:
        - Aleatoric: "this response is inherently ambiguous"
        - Epistemic: "our model(s) disagree on this response"
    """

    def __init__(
        self,
        model_paths: List[str],
        base_model_name: str = "microsoft/deberta-v3-base",
        n_attributes: int = 5,
        batch_size: int = 8,
        max_length: int = 1024,
        device: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_models = len(model_paths)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer from first trained member directory (fully local).
        self.tokenizer = AutoTokenizer.from_pretrained(model_paths[0], local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load ensemble members
        self.models: List[UncertaintyRewardModel] = []
        for path in model_paths:
            model = UncertaintyRewardModel(
                base_model_name=base_model_name,
                n_attributes=n_attributes,
            )
            # Load trained weights
            state_dict = torch.load(
                os.path.join(path, "urm_weights.pt"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    @torch.inference_mode()
    def _score_single_model(
        self,
        model: UncertaintyRewardModel,
        samples: List[str],
    ) -> Tuple[List[float], List[float]]:
        """Score all samples with a single URM, returning rewards and aleatoric vars."""
        all_rewards = []
        all_vars = []

        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            all_rewards.extend(out["reward"].cpu().tolist())
            all_vars.extend(out["variance"].cpu().tolist())

        return all_rewards, all_vars

    def score(self, prompt: str, candidates: List[str]) -> URMStats:
        """Score candidates with all ensemble members.

        Returns:
            URMStats with per-candidate μ, total_var, aleatoric_var, epistemic_var.
        """
        samples = [f"{prompt}\n{c}" for c in candidates]

        # Collect (reward, aleatoric_var) from each model
        all_rewards = []   # (K, N)
        all_ale_vars = []  # (K, N)

        for model in self.models:
            rewards, ale_vars = self._score_single_model(model, samples)
            all_rewards.append(rewards)
            all_ale_vars.append(ale_vars)

        all_rewards = np.array(all_rewards)   # (K, N)
        all_ale_vars = np.array(all_ale_vars) # (K, N)

        # Mean reward across ensemble
        means = np.mean(all_rewards, axis=0)  # (N,)

        # Mean aleatoric variance across ensemble
        aleatoric = np.mean(all_ale_vars, axis=0)  # (N,)

        # Epistemic variance = variance of rewards across ensemble
        if self.n_models > 1:
            epistemic = np.var(all_rewards, axis=0, ddof=1)  # (N,)
        else:
            epistemic = np.zeros_like(means)

        # Total variance = aleatoric + epistemic
        total_var = aleatoric + epistemic

        return URMStats(
            means=means.tolist(),
            variances=total_var.tolist(),
            aleatoric_vars=aleatoric.tolist(),
            epistemic_vars=epistemic.tolist(),
        )

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        """Return only scalar rewards (for vanilla BoN)."""
        stats = self.score(prompt, candidates)
        return stats.means


# =========================================================================
# Training utilities
# =========================================================================

class HelpSteerDataset(torch.utils.data.Dataset):
    """HelpSteer2 dataset for training the probabilistic value head.

    HelpSteer2 has multi-attribute annotations:
        helpfulness, correctness, coherence, complexity, verbosity
    Each scored 0-4 (integer scale).

    We train the URM to predict these attribute scores with calibrated uncertainty.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        attributes: Optional[List[str]] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.attributes = attributes or HELPSTEER_ATTRIBUTES

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['prompt']}\n{item['response']}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract attribute scores as targets
        targets = torch.tensor(
            [float(item.get(attr, 0.0)) for attr in self.attributes],
            dtype=torch.float32,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": targets,
        }


@dataclass
class URMTrainingConfig:
    base_model_name: str = "microsoft/deberta-v3-base"
    n_attributes: int = 5
    n_ensemble: int = 3
    learning_rate: float = 2e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 16  # Effective batch = 64
    max_length: int = 1024
    weight_decay: float = 1e-3
    warmup_ratio: float = 0.1
    output_dir: str = "models/urm_ensemble"
    seed_base: int = 42
    use_gating: bool = False


def load_helpsteer2(max_examples: Optional[int] = None) -> List[Dict]:
    """Load the HelpSteer2 dataset from HuggingFace.

    Returns list of dicts with keys:
        prompt, response, helpfulness, correctness, coherence, complexity, verbosity
    """
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="train")

    examples = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        examples.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "helpfulness": item["helpfulness"],
            "correctness": item["correctness"],
            "coherence": item["coherence"],
            "complexity": item["complexity"],
            "verbosity": item["verbosity"],
        })

    return examples


def train_single_urm(
    config: URMTrainingConfig,
    train_data: List[Dict],
    eval_data: Optional[List[Dict]],
    member_idx: int,
    seed: int,
) -> str:
    """Train a single URM ensemble member.

    Returns path to saved model weights.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shuffle data with this seed
    shuffled = train_data.copy()
    random.shuffle(shuffled)

    # Build model
    model = UncertaintyRewardModel(
        base_model_name=config.base_model_name,
        n_attributes=config.n_attributes,
        use_gating=config.use_gating,
    ).to(device)

    # Build datasets
    base_source, local_only = _resolve_hf_source(config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_source, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = HelpSteerDataset(shuffled, tokenizer, config.max_length)
    eval_dataset = HelpSteerDataset(eval_data, tokenizer, config.max_length) if eval_data else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )

    loss_fn = MLELoss()

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            out = model(input_ids, attention_mask)
            loss = loss_fn(out["mu"], out["log_sigma"], targets)

            # Guard against NaN loss (e.g. from encoder precision issues)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Member {member_idx}] WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = epoch_loss / n_batches
                    print(f"  [Member {member_idx}] Step {global_step}, Loss: {avg:.4f}")

        # Final gradient step for remaining batches
        if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  [Member {member_idx}] Epoch {epoch+1}/{config.num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Eval
        if eval_dataset:
            model.eval()
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size)
            eval_loss = 0.0
            n_eval = 0
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["targets"].to(device)
                    out = model(input_ids, attention_mask)
                    eval_loss += loss_fn(out["mu"], out["log_sigma"], targets).item()
                    n_eval += 1
            print(f"  [Member {member_idx}] Eval Loss: {eval_loss / max(n_eval, 1):.4f}")
            model.train()

    # Save
    save_dir = Path(config.output_dir) / f"member_{member_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "urm_weights.pt")
    tokenizer.save_pretrained(str(save_dir))

    # Save config
    save_config = {
        "base_model_name": config.base_model_name,
        "n_attributes": config.n_attributes,
        "use_gating": config.use_gating,
        "member_idx": member_idx,
        "seed": seed,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(save_config, f, indent=2)

    print(f"  [Member {member_idx}] Saved to {save_dir}")
    return str(save_dir)


def train_urm_ensemble(
    config: URMTrainingConfig,
    train_data: List[Dict],
    eval_data: Optional[List[Dict]] = None,
):
    """Train K uncertainty-aware reward models with different seeds.

    Each model sees the same data in different order (different shuffles).
    At inference, ensemble disagreement → epistemic uncertainty.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    member_paths = []

    for k in range(config.n_ensemble):
        seed = config.seed_base + k * 1000
        print(f"\n{'='*60}")
        print(f"Training URM member {k+1}/{config.n_ensemble} (seed={seed})")
        print(f"{'='*60}")

        path = train_single_urm(config, train_data, eval_data, k, seed)
        member_paths.append(path)

    # Save ensemble config
    ensemble_config = {
        "base_model_name": config.base_model_name,
        "n_attributes": config.n_attributes,
        "n_ensemble": config.n_ensemble,
        "use_gating": config.use_gating,
        "member_paths": member_paths,
    }
    with open(output_dir / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\nEnsemble config saved to {output_dir / 'ensemble_config.json'}")
    print(f"Use: --rm-mode robust --urm-ensemble-dir {config.output_dir}")


# =========================================================================
# CLI for training
# =========================================================================

def train_from_cli():
    """CLI entry point for training the URM ensemble."""
    import typer

    app = typer.Typer(help="Train Uncertainty-aware Reward Model ensemble on HelpSteer2")

    @app.command()
    def train(
        base_model: str = typer.Option(
            "microsoft/deberta-v3-base",
            help="Base encoder model.",
        ),
        n_attributes: int = typer.Option(5, help="Number of attributes."),
        n_ensemble: int = typer.Option(3, help="Number of ensemble members."),
        num_epochs: int = typer.Option(1, help="Training epochs per member."),
        batch_size: int = typer.Option(4, help="Micro-batch size per GPU."),
        grad_accum: int = typer.Option(16, help="Gradient accumulation steps."),
        learning_rate: float = typer.Option(2e-6, help="Learning rate."),
        max_length: int = typer.Option(1024, help="Max sequence length."),
        max_examples: Optional[int] = typer.Option(None, help="Max training examples (None=all)."),
        output_dir: str = typer.Option("models/urm_ensemble", help="Output directory."),
        use_gating: bool = typer.Option(False, help="Use learned gating layer."),
    ):
        """Train URM ensemble on HelpSteer2 dataset."""
        print("Loading HelpSteer2 dataset...")
        all_data = load_helpsteer2(max_examples=max_examples)
        print(f"Loaded {len(all_data)} examples")

        # 90/10 split
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]
        print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

        config = URMTrainingConfig(
            base_model_name=base_model,
            n_attributes=n_attributes,
            n_ensemble=n_ensemble,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            max_length=max_length,
            output_dir=output_dir,
            use_gating=use_gating,
        )

        train_urm_ensemble(config, train_data, eval_data)

    app()


if __name__ == "__main__":
    train_from_cli()
