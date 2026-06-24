"""Dataset loading and correctness checking.

Centralises all dataset-specific logic so that hypothesis modules and
scripts never need to import from individual data_utils files directly.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import pickle
import threading
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────
# Rigorous math grading (math_verify) with a persistent disk cache.
#
# The legacy string grader in ``data_utils/math500.py`` is BOTH too lenient
# (its ``gold in pred`` substring fallback marks "2" correct inside "12") and
# too strict (exact match fails on equivalent LaTeX like ``11\sqrt2`` vs
# ``11\sqrt{2}``).  Measured against math_verify it produced ~14% false
# positives, inflating MATH500 accuracy by ~12 points.  We therefore grade
# MATH500 with ``math_verify`` (the HF/Qwen/DeepSeek standard).  Results are
# cached to disk keyed by (dataset, gold, prediction) so repeated hypothesis
# runs over the same candidate set are effectively free after the first pass.
# ─────────────────────────────────────────────────────────────────────────

_GRADE_CACHE_PATH = os.path.join("data", "cache", "math_grade_cache.pkl")
_GRADE_CACHE: Optional[dict] = None
_GRADE_CACHE_LOCK = threading.Lock()
_GRADE_CACHE_DIRTY = False
_MV = None  # lazily-imported (parse, verify, CFG)


def _load_grade_cache() -> dict:
    global _GRADE_CACHE
    if _GRADE_CACHE is None:
        try:
            with open(_GRADE_CACHE_PATH, "rb") as f:
                _GRADE_CACHE = pickle.load(f)
        except Exception:
            _GRADE_CACHE = {}
        atexit.register(_save_grade_cache)
    return _GRADE_CACHE


def _save_grade_cache() -> None:
    global _GRADE_CACHE_DIRTY
    if _GRADE_CACHE is None or not _GRADE_CACHE_DIRTY:
        return
    os.makedirs(os.path.dirname(_GRADE_CACHE_PATH), exist_ok=True)
    tmp = _GRADE_CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(_GRADE_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _GRADE_CACHE_PATH)
    _GRADE_CACHE_DIRTY = False


def _grade_key(dataset: str, gold: str, pred: str) -> str:
    h = hashlib.sha1()
    h.update(dataset.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(gold.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(pred.encode("utf-8", "ignore"))
    return h.hexdigest()


def _get_mv():
    global _MV
    if _MV is None:
        from math_verify import parse, verify  # type: ignore
        from math_verify.parser import (  # type: ignore
            ExprExtractionConfig,
            LatexExtractionConfig,
        )
        _MV = (parse, verify, [LatexExtractionConfig(), ExprExtractionConfig()])
    return _MV


def math_verify_correct(prediction: str, ground_truth: str) -> bool:
    """Rigorous correctness for MATH-style answers via math_verify.

    Parses the *full* prediction text (extracts the last boxed/LaTeX/expression
    automatically — this recovers answers even when ``\\boxed{}`` formatting is
    imperfect) and checks symbolic/numeric equivalence against the gold answer.
    """
    if not prediction or not ground_truth:
        return False
    parse, verify, cfg = _get_mv()
    # Gold answers are bare expressions; wrapping in $...$ parses most reliably.
    gold_p = None
    for g in (f"${ground_truth}$", ground_truth):
        try:
            gold_p = parse(g, extraction_config=cfg)
        except Exception:
            gold_p = None
        if gold_p:
            break
    if not gold_p:
        return False
    try:
        pred_p = parse(prediction, extraction_config=cfg)
    except Exception:
        return False
    if not pred_p:
        return False
    try:
        return bool(verify(gold_p, pred_p))
    except Exception:
        return False


def check_math_correct_cached(prediction: str, ground_truth: str) -> bool:
    """Disk-cached wrapper around :func:`math_verify_correct`."""
    global _GRADE_CACHE_DIRTY
    cache = _load_grade_cache()
    key = _grade_key("math500", ground_truth, prediction)
    with _GRADE_CACHE_LOCK:
        if key in cache:
            return cache[key]
    val = math_verify_correct(prediction, ground_truth)
    with _GRADE_CACHE_LOCK:
        cache[key] = val
        _GRADE_CACHE_DIRTY = True
    return val


def load_dataset_examples(
    dataset: str,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load examples from the specified dataset.

    Args:
        dataset: One of ``gsm8k``, ``math500``, ``alpaca_eval``.
        split: Dataset split (e.g. ``test``).
        max_examples: Cap the number of examples. ``None`` = full split.

    Returns:
        List of dicts, each with at least ``question`` and ``answer`` keys.
    """
    dataset = dataset.lower()
    if dataset == "gsm8k":
        from ..data_utils.gsm8k import load_gsm8k
        return load_gsm8k(split=split, max_examples=max_examples)
    elif dataset == "math500":
        from ..data_utils.math500 import load_math500
        return load_math500(max_examples=max_examples)
    elif dataset == "alpaca_eval":
        from ..data_utils.alpaca_eval import load_alpaca_eval
        return load_alpaca_eval(max_examples=max_examples)
    elif dataset == "alpaca":
        from ..data_utils.alpaca import load_alpaca
        return load_alpaca(max_examples=max_examples)
    elif dataset == "mmlu":
        from ..data_utils.mmlu import load_mmlu
        return load_mmlu(split=split, max_examples=max_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: gsm8k, math500, alpaca_eval, alpaca, mmlu")


def check_correctness(prediction: str, ground_truth: str, dataset: str) -> bool:
    """Check whether *prediction* is correct given *ground_truth*.

    Args:
        prediction: The model's selected response text.
        ground_truth: The gold-standard answer string.
        dataset: Dataset identifier (determines comparison logic).

    Returns:
        ``True`` if the answer is correct, ``False`` otherwise.
    """
    dataset = dataset.lower()
    if dataset == "gsm8k":
        from ..data_utils.gsm8k import check_answer_correct
        return check_answer_correct(prediction, ground_truth)
    elif dataset == "math500":
        # Rigorous, cached math_verify grading (see module docstring above).
        # Falls back to the legacy string grader only if math_verify is missing.
        try:
            return check_math_correct_cached(prediction, ground_truth)
        except ImportError:
            from ..data_utils.math500 import check_math_correct
            return check_math_correct(prediction, ground_truth)
    elif dataset == "mmlu":
        from ..data_utils.mmlu import check_answer_correct as check_mmlu_correct
        return check_mmlu_correct(prediction, ground_truth)
    elif dataset in ("alpaca_eval", "alpaca"):
        return False  # open-ended: no ground-truth correctness
    return False


def dataset_system_message(dataset: str) -> str:
    """Return the system prompt used for generation on *dataset*."""
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
            "You are a helpful assistant for arithmetic word problems. "
            "Solve briefly and clearly. Put your final numerical answer after ####."
        )
    if dataset == "alpaca":
        return "You are a helpful, harmless, and honest assistant. Follow the instruction carefully."
    if dataset == "mmlu":
        return "You are a knowledgeable assistant. Answer the multiple choice question by selecting the correct option (A, B, C, or D). Provide a brief explanation followed by your final answer in the format: 'Answer: (X)' where X is the correct letter."
    return "You are a helpful assistant."
