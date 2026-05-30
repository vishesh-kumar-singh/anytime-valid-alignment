"""
GSM8K dataset loading and answer extraction utilities.

GSM8K (Grade School Math 8K) is the primary evaluation benchmark.
Each example has a 'question' and 'answer' field, where the final
numeric answer appears after '####'.
"""

import re
from typing import List, Optional, Tuple

from datasets import DownloadConfig, load_dataset


def load_gsm8k(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load GSM8K dataset examples.

    Args:
        split: "train" or "test".
        max_examples: If set, truncate to this many examples.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    try:
        # Try fully offline load first to avoid unnecessary Hub SSL checks.
        ds = load_dataset(
            "gsm8k",
            "main",
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception:
        ds = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
        })

    return examples


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract the final numeric answer from a GSM8K answer string.

    GSM8K ground-truth answers end with '#### <number>'.
    Model-generated answers should also be trained to follow this format.

    Returns:
        The extracted number as a string, or None if not found.
    """
    if not answer_text:
        return None

    # Try the standard #### marker first
    if "####" in answer_text:
        after_marker = answer_text.split("####")[-1].strip()
        # Clean up: remove commas, dollar signs, etc.
        cleaned = re.sub(r"[,$%]", "", after_marker).strip()
        # Extract the first number (int or float)
        match = re.search(r"-?[\d,]+\.?\d*", cleaned)
        if match:
            return match.group(0).replace(",", "")

    # Fallback: try to find the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer_correct(prediction: str, ground_truth: str) -> bool:
    """Check if a predicted answer matches the ground truth.

    Extracts numeric answers from both strings and compares.
    Handles minor formatting differences (commas, whitespace, etc.).
    """
    pred_num = extract_gsm8k_answer(prediction)
    gold_num = extract_gsm8k_answer(ground_truth)

    if pred_num is None or gold_num is None:
        return False

    # Try numeric comparison to handle float vs int (e.g., "5.0" vs "5")
    try:
        return abs(float(pred_num) - float(gold_num)) < 1e-6
    except ValueError:
        return pred_num.strip() == gold_num.strip()
