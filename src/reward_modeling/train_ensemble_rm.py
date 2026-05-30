

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


# =========================================================================
# Preference Dataset for RM Training
# =========================================================================

class PreferencePairDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) preference pairs.

    Each item returns tokenized inputs for the chosen and rejected responses.
    The RM training loss ensures score(chosen) > score(rejected).
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            data: List of dicts with keys: prompt, chosen, rejected
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"

        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# =========================================================================
# Bradley-Terry Ranking Loss
# =========================================================================

class RankingLoss(nn.Module):
    """Bradley-Terry pairwise ranking loss for reward model training.

    loss = -log(sigmoid(score_chosen - score_rejected))

    This is the standard loss used in RLHF reward model training.
    """

    def forward(self, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.sigmoid(chosen_scores - rejected_scores) + 1e-8).mean()


# =========================================================================
# Reward Model Trainer
# =========================================================================

@dataclass
class RMTrainingConfig:
    base_model_name: str = "microsoft/deberta-v3-base"
    n_ensemble: int = 3             # Number of ensemble members (K)
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 8
    max_length: int = 512
    output_dir: str = "models/ensemble_rm"
    seed_base: int = 42


def generate_preference_data_from_gsm8k(
    train_data: List[Dict],
    generator=None,
    max_pairs: int = 1000,
) -> List[Dict]:
    """Generate preference pairs from GSM8K training data.

    Strategy:
      - For each question, generate multiple candidate answers
      - Check against gold answer
      - Correct answers = chosen, incorrect = rejected
      - Create all (correct, incorrect) pairs

    If no generator is provided, creates simple synthetic pairs
    using the gold answer as chosen and a corrupted version as rejected.
    """
    pairs = []

    for item in train_data[:max_pairs]:
        question = item["question"]
        gold = item["answer"]

        if generator is not None:
            # Generate candidates and classify as correct/incorrect
            candidates = generator.generate(question, n=4)
            from ..data_utils.gsm8k import check_answer_correct

            correct = [c for c in candidates if check_answer_correct(c, gold)]
            incorrect = [c for c in candidates if not check_answer_correct(c, gold)]

            # Create pairs
            for c in correct:
                for w in incorrect:
                    pairs.append({
                        "prompt": question,
                        "chosen": c,
                        "rejected": w,
                    })
        else:
            # Synthetic: gold answer is chosen, truncated/corrupted is rejected
            pairs.append({
                "prompt": question,
                "chosen": gold,
                "rejected": gold[:len(gold) // 3] + "\n#### 999999",  # Wrong answer
            })

        if len(pairs) >= max_pairs:
            break

    return pairs[:max_pairs]


def train_ensemble_rm(
    config: RMTrainingConfig,
    preference_data: List[Dict],
    eval_data: Optional[List[Dict]] = None,
):
    """Train K reward models with different random seeds.

    Each model is a copy of the base model, trained on the same preference
    data but with different random seeds (different data ordering, different
    weight initialization randomness). This creates an ensemble where
    disagreement = uncertainty.

    Args:
        config: Training configuration.
        preference_data: List of {prompt, chosen, rejected} dicts.
        eval_data: Optional validation preference data.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in range(config.n_ensemble):
        seed = config.seed_base + k * 1000
        print(f"\n{'='*60}")
        print(f"Training ensemble member {k+1}/{config.n_ensemble} (seed={seed})")
        print(f"{'='*60}")

        # Set seed for this ensemble member
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Shuffle data differently for each member
        shuffled_data = preference_data.copy()
        random.shuffle(shuffled_data)

        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name,
            num_labels=1,
        )

        # Create dataset
        train_dataset = PreferencePairDataset(
            shuffled_data, tokenizer, max_length=config.max_length
        )

        eval_dataset = None
        if eval_data:
            eval_dataset = PreferencePairDataset(
                eval_data, tokenizer, max_length=config.max_length
            )

        # Training arguments
        member_dir = output_dir / f"member_{k}"
        training_args = TrainingArguments(
            output_dir=str(member_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            seed=seed,
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )

        # Custom trainer with ranking loss
        trainer = RMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Save the model
        final_path = member_dir / "final"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"Saved ensemble member {k+1} to {final_path}")

    # Save config
    config_path = output_dir / "ensemble_config.json"
    config_dict = {
        "base_model_name": config.base_model_name,
        "n_ensemble": config.n_ensemble,
        "member_paths": [
            str(output_dir / f"member_{k}" / "final")
            for k in range(config.n_ensemble)
        ],
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nEnsemble config saved to {config_path}")
    print(f"Use member paths in --reward-model-names to run inference")


class RMTrainer(Trainer):
    """Custom trainer for reward model with Bradley-Terry ranking loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ranking_loss = RankingLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Score chosen responses
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        chosen_scores = chosen_outputs.logits[:, 0]

        # Score rejected responses
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        rejected_scores = rejected_outputs.logits[:, 0]

        # Ranking loss
        loss = self.ranking_loss(chosen_scores, rejected_scores)

        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss


# =========================================================================
# CLI for training
# =========================================================================

def train_from_cli():
    """CLI entry point for training the ensemble RM."""
    import typer

    app = typer.Typer(help="Train ensemble reward model")

    @app.command()
    def train(
        base_model: str = typer.Option(
            "microsoft/deberta-v3-base",
            help="Base model for RM (e.g., deberta-v3-base).",
        ),
        n_ensemble: int = typer.Option(3, help="Number of ensemble members."),
        output_dir: str = typer.Option("models/ensemble_rm", help="Output directory."),
        max_pairs: int = typer.Option(1000, help="Max preference pairs to generate."),
        num_epochs: int = typer.Option(2, help="Training epochs per member."),
        batch_size: int = typer.Option(8, help="Batch size."),
        learning_rate: float = typer.Option(1e-5, help="Learning rate."),
        use_generator: bool = typer.Option(False, help="Use LLM to generate candidates (slow)."),
        generator_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="Model for generating candidates."),
    ):
        """Train an ensemble of K reward models on GSM8K preference data."""
        from ..data_utils.gsm8k import load_gsm8k

        print("Loading GSM8K training data...")
        train_examples = load_gsm8k(split="train", max_examples=max_pairs * 2)

        generator = None
        if use_generator:
            from ..generation.generator import GenerationConfig, Generator
            print(f"Loading generator: {generator_model}")
            generator = Generator(GenerationConfig(model_name=generator_model))

        print("Generating preference pairs...")
        preference_data = generate_preference_data_from_gsm8k(
            train_examples, generator=generator, max_pairs=max_pairs,
        )

        # Split 90/10 for train/eval
        split_idx = int(len(preference_data) * 0.9)
        train_pairs = preference_data[:split_idx]
        eval_pairs = preference_data[split_idx:]

        print(f"Training pairs: {len(train_pairs)}, Eval pairs: {len(eval_pairs)}")

        config = RMTrainingConfig(
            base_model_name=base_model,
            n_ensemble=n_ensemble,
            output_dir=output_dir,
            max_length=512,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        train_ensemble_rm(config, train_pairs, eval_pairs)

    app()


if __name__ == "__main__":
    train_from_cli()
