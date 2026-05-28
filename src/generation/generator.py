"""
LLM candidate generation module.

Supports loading local HuggingFace causal LMs (default: Qwen2.5-7B-Instruct).
Generates N candidate responses for a given prompt with configurable sampling.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50            # Limits vocab to top-k tokens before sampling (prevents NaN from extreme logits)
    do_sample: bool = True
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device: int = 0             # GPU index — do NOT use "auto" with multi-GPU (causes NaN)
    load_in_4bit: bool = False


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Generator:
    """Loads a local causal LM and generates N candidate completions."""
#
    @staticmethod
    def _resolve_model_source(model_name: str) -> tuple[str, bool]:
        """Resolve a local model path when available.

        Returns:
            (model_source, local_only)
            - model_source: local snapshot path or the original model name
            - local_only: whether to force local file loading
        """
        # If user already passed a local path, use it directly.
        local_path = Path(model_name).expanduser()
        if local_path.exists():
            return str(local_path), True

        # Otherwise, try Hugging Face cache snapshots to avoid network calls.
        if "/" in model_name:
            repo_dir = f"models--{model_name.replace('/', '--')}"
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            snapshots_dir = hf_home / "hub" / repo_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
                if snapshots:
                    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                    return str(latest_snapshot), True

        return model_name, False
#
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        dtype = _DTYPE_MAP.get(cfg.torch_dtype, torch.bfloat16)
#
        model_source, local_only = self._resolve_model_source(cfg.model_name)
#
        # IMPORTANT: Use device_map={"":device_id} to force single-GPU loading.
        # device_map="auto" splits model across GPUs which causes NaN logits
        # during cross-device communication (known issue with PyTorch 2.10 + multi-GPU).
        load_kwargs = dict(
            #pretrained_model_name_or_path=cfg.model_name,
            #
            pretrained_model_name_or_path=model_source,
            #
            dtype=dtype,
            device_map={"": cfg.device},
        )
        if local_only:
            load_kwargs["local_files_only"] = True

        if cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                raise ImportError("bitsandbytes required for 4-bit loading: pip install bitsandbytes")

        tokenizer_kwargs = {"pretrained_model_name_or_path": model_source}
        if local_only:
            tokenizer_kwargs["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()

        # All parameters are on the same device thanks to device_map={"": device_id}
        self.device = next(self.model.parameters()).device

    def _build_prompt(self, question: str, system_msg: Optional[str] = None) -> str:
        """Wrap the question in a chat template if the model supports it."""
        if system_msg is None:
            system_msg = (
                "You are a helpful assistant. Solve the problem step by step. "
                "Put your final answer after ####."
            )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"Question: {question}\nAnswer: Let me solve this step by step.\n"

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_new_tokens: Optional[int] = None,
        system_msg: Optional[str] = None,
        use_chat_template: bool = True,
    ) -> List[str]:
        """Generate n candidate responses for a given prompt.

        To avoid CUDA errors with num_return_sequences > 1 (NaN/Inf in
        expanded logits), we generate one sequence at a time when n > 1.
        This is more robust and avoids the device-side assert that occurs
        when the model's probability tensor becomes invalid during batch
        expansion.
        """
        formatted = self._build_prompt(prompt, system_msg=system_msg) if use_chat_template else prompt
        tokens = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        results = []
        for _ in range(n):
            outputs = self.model.generate(
                **inputs,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                max_new_tokens=tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_tokens = outputs[0][input_len:]
            text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            results.append(text)

        return results

