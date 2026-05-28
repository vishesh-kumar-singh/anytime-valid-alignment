"""
Reward model ensemble for uncertainty-aware scoring.

Loads K pre-trained reward models (e.g., DeBERTa-based) and scores each
candidate with all K models. Returns:
  - μ (mean): average reward across ensemble members
  - σ² (variance): disagreement across ensemble members

This is inspired by the Uncertainty-aware Reward Model paper
(arXiv:2410.00847) which uses ensembles to estimate epistemic uncertainty.

Two modes:
  1. Scalar mode: Uses a single RM, returns (reward, 0). For vanilla BoN.
  2. Ensemble mode: Uses K RMs, returns (mean, variance). For risk-adjusted
     and Sharpe-ratio methods.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RewardStats:
    """Reward statistics for a set of candidates."""
    means: List[float]      # Per-candidate mean reward μ
    variances: List[float]  # Per-candidate reward variance σ²
    member_scores: Optional[List[List[float]]] = None  # Shape (K, N)


class RewardEnsemble:
    """Ensemble of K reward models for uncertainty-aware scoring.

    For each candidate, computes:
        μ = (1/K) Σ_k r_k(x, y)
        σ² = (1/K) Σ_k (r_k - μ)²  (population variance)

    When K=1, σ²=0 (no uncertainty estimate available → vanilla mode).
    """

    @staticmethod
    def _resolve_model_source(model_name: str) -> tuple[str, bool]:
        """Resolve a local model path when available.

        Returns:
            (model_source, local_only)
            - model_source: local snapshot path or the original model name
            - local_only: whether to force local file loading
        """
        local_path = Path(model_name).expanduser()
        if local_path.exists():
            return str(local_path), True

        if "/" in model_name:
            repo_dir = f"models--{model_name.replace('/', '--')}"
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            snapshots_dir = hf_home / "hub" / repo_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
                complete = []
                for snap in snapshots:
                    names = {f.name for f in snap.iterdir()}
                    has_config = "config.json" in names
                    has_weights = (
                        "model.safetensors" in names
                        or "pytorch_model.bin" in names
                        or "model.safetensors.index.json" in names
                        or any(name.startswith("model-") and name.endswith(".safetensors") for name in names)
                    )
                    if has_config and has_weights:
                        complete.append(snap)

                if complete:
                    best_snapshot = max(complete, key=lambda p: p.stat().st_mtime)
                    return str(best_snapshot), True

        return model_name, False

    @staticmethod
    def _resolve_base_tokenizer_source() -> Optional[str]:
        """Best-effort local tokenizer source for DeBERTa-family reward models."""
        base_model = "microsoft/deberta-v3-base"
        repo_dir = f"models--{base_model.replace('/', '--')}"
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        snapshots_dir = hf_home / "hub" / repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        tokenized = []
        for snap in snapshots:
            names = {f.name for f in snap.iterdir()}
            has_tokenizer = (
                "tokenizer.json" in names
                or "spm.model" in names
                or "vocab.json" in names
            )
            if has_tokenizer:
                tokenized.append(snap)

        if not tokenized:
            return None
        return str(max(tokenized, key=lambda p: p.stat().st_mtime))

    def __init__(
        self,
        model_names: List[str],
        batch_size: int = 8,
        max_length: int = 1024,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        if not model_names:
            raise ValueError("model_names must be non-empty.")

        self.batch_size = batch_size
        self.max_length = max_length
        self.n_models = len(model_names)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use reduced precision on GPU to keep multi-RM inference memory-feasible.
        if self.device.type == "cuda":
            self.model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.model_dtype = torch.float32
        self.load_in_4bit = load_in_4bit and self.device.type == "cuda"

        resolved_sources = [self._resolve_model_source(name) for name in model_names]
        self.members = []

        for source, local_only in resolved_sources:
            tokenizer = self._load_tokenizer(source, local_only)
            model = self._load_model(source, local_only)
            self.members.append((tokenizer, model))

    def _load_tokenizer(self, source: str, local_only: bool):
        """Load tokenizer for a specific reward model source."""
        source_lower = source.lower()
        tokenizer_sources = [source]
        # Only DeBERTa-family models should fallback to the DeBERTa base tokenizer.
        # For Llama/Armo-style reward models, a fallback can silently corrupt token IDs.
        should_allow_deberta_fallback = ("deberta" in source_lower) or ("openassistant" in source_lower)
        if should_allow_deberta_fallback:
            base_tok = self._resolve_base_tokenizer_source()
            if base_tok is not None:
                tokenizer_sources.append(base_tok)

        last_err = None
        for tok_src in tokenizer_sources:
            try:
                tokenizer_kwargs = {
                    "pretrained_model_name_or_path": tok_src,
                    "trust_remote_code": True,
                }
                if local_only:
                    tokenizer_kwargs["local_files_only"] = True
                # Some reward models fail fast-tokenizer conversion when optional
                # deps (e.g. tiktoken/sentencepiece bridges) are unavailable.
                for use_fast in (None, False, True):
                    try:
                        kwargs = dict(tokenizer_kwargs)
                        if use_fast is not None:
                            kwargs["use_fast"] = use_fast
                        tokenizer = AutoTokenizer.from_pretrained(**kwargs)
                        # Keep suffix tokens for chat-style reward models (assistant header + answer).
                        # ArmoRM gating expects the assistant-header token pattern near the end.
                        if "llama" in source_lower or "armorm" in source_lower or "rlhflow" in source_lower:
                            tokenizer.truncation_side = "left"
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        return tokenizer
                    except Exception as inner:
                        last_err = inner
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Failed to load tokenizer for reward model source: {source}"
        ) from last_err

    def _load_model(self, source: str, local_only: bool):
        """Load a sequence-classification reward model for a specific source."""
        if "ArmoRM" in source or "armorm" in source.lower():
            # RLHFlow/ArmoRM custom code expects this symbol in some transformers versions.
            try:
                from transformers.models.llama import modeling_llama

                if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
                    modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
            except Exception:
                pass

        model_kwargs = {
            "pretrained_model_name_or_path": source,
            "trust_remote_code": True,
        }
        if local_only:
            model_kwargs["local_files_only"] = True
        model_kwargs["low_cpu_mem_usage"] = True

        def _dtype_load_kwargs() -> dict:
            kwargs = dict(model_kwargs)
            kwargs["dtype"] = self.model_dtype
            return kwargs

        tried_quantized = False
        quant_err = None
        load_kwargs = _dtype_load_kwargs()
        if self.load_in_4bit:
            tried_quantized = True
            load_kwargs = dict(model_kwargs)
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}

        try:
            model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
            if not (tried_quantized and "load_in_4bit" in load_kwargs):
                model = model.to(self.device)
        except TypeError as exc:
            # Some model classes reject `load_in_4bit` in __init__. Retry in regular dtype mode.
            if "load_in_4bit" in str(exc) and tried_quantized:
                quant_err = exc
                model = AutoModelForSequenceClassification.from_pretrained(**_dtype_load_kwargs()).to(self.device)
            else:
                raise
        except ImportError as exc:
            if self.load_in_4bit:
                raise RuntimeError(
                    "Reward-model 4-bit loading requested but required dependencies are missing. "
                    "Install/upgrade bitsandbytes and accelerate, or disable --reward-load-in-4bit."
                ) from exc
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Out of GPU memory while loading reward model. "
                "Try a smaller RM ensemble (e.g., remove Gemma-27B), "
                "set --load-in-4bit for generation, set --reward-load-in-4bit, "
                "and lower --reward-batch-size to 1."
            ) from exc

        if quant_err is not None:
            print(f"Warning: 4-bit RM loading unsupported for {source}; using dtype={self.model_dtype} for this model.")
        model.eval()
        return model

    @staticmethod
    def _extract_scores(output) -> torch.Tensor:
        """Extract scalar reward scores from model output."""
        logits = getattr(output, "logits", None)
        if logits is None:
            if isinstance(output, (tuple, list)) and len(output) > 0:
                logits = output[0]
            else:
                raise RuntimeError("Reward model output does not contain logits.")

        if logits.ndim == 1:
            return logits
        return logits[:, 0]

    @torch.inference_mode()
    def _score_single_model(self, tokenizer, model, samples: List[str]) -> List[float]:
        """Score all samples with a single reward model, in batches."""
        all_scores = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            try:
                output = model(**inputs)
            except TypeError as exc:
                # Some custom Llama-based reward heads do not accept token_type_ids.
                if "token_type_ids" in str(exc) and "token_type_ids" in inputs:
                    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
                    output = model(**inputs)
                else:
                    raise
            scores = self._extract_scores(output)
            all_scores.extend(scores.cpu().tolist())
        return all_scores

    @staticmethod
    def _format_samples_for_member(tokenizer, model, prompt: str, candidates: List[str]) -> List[str]:
        """Build model-specific RM input strings.

        ArmoRM expects a chat-template token pattern before assistant content.
        """
        model_path = str(getattr(getattr(model, "config", None), "_name_or_path", "")).lower()
        if "armorm" in model_path or "rlhflow" in model_path:
            formatted = []
            for candidate in candidates:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = (
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{candidate}<|eot_id|>"
                    )
                formatted.append(text)
            return formatted

        return [f"{prompt}\n{candidate}" for candidate in candidates]

    def score(self, prompt: str, candidates: List[str]) -> RewardStats:
        """Score candidates with all ensemble members.

        Args:
            prompt: The input question/instruction.
            candidates: List of candidate answers.

        Returns:
            RewardStats with per-candidate μ and σ².
        """
        if self.n_models == 1:
            # Single model: no variance estimation possible
            tokenizer, model = self.members[0]
            samples = self._format_samples_for_member(tokenizer, model, prompt, candidates)
            scores = self._score_single_model(tokenizer, model, samples)
            return RewardStats(
                means=scores,
                variances=[0.0] * len(scores),
                member_scores=[scores],
            )

        # Ensemble: collect scores from all K models
        # Shape: (K, N) where K = n_models, N = n_candidates
        all_scores = []
        for tokenizer, model in self.members:
            samples = self._format_samples_for_member(tokenizer, model, prompt, candidates)
            scores = self._score_single_model(tokenizer, model, samples)
            all_scores.append(scores)

        all_scores = np.array(all_scores)  # (K, N)

        # μ = mean across ensemble members (axis=0)
        means = np.mean(all_scores, axis=0).tolist()

        # σ² = variance across ensemble members (population variance, ddof=0)
        if self.n_models > 1:
            variances = np.var(all_scores, axis=0, ddof=0).tolist()
        else:
            variances = [0.0] * len(means)

        return RewardStats(means=means, variances=variances, member_scores=all_scores.tolist())

    def score_scalar(self, prompt: str, candidates: List[str]) -> List[float]:
        """Score candidates and return only scalar rewards (for vanilla BoN)."""
        stats = self.score(prompt, candidates)
        return stats.means

