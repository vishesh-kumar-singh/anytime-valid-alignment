"""Dataset loading and correctness checking.

Centralises all dataset-specific logic so that hypothesis modules and
scripts never need to import from individual data_utils files directly.
"""

from __future__ import annotations

from typing import List, Optional


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
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: gsm8k, math500, alpaca_eval")


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
        from ..data_utils.math500 import check_math_correct
        return check_math_correct(prediction, ground_truth)
    elif dataset == "alpaca_eval":
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
    return "You are a helpful assistant."
