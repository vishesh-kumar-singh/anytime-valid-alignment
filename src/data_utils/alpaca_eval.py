"""
AlpacaEval dataset loading.

AlpacaEval is a benchmark for evaluating LLM generation quality on
open-ended instruction-following tasks. Used in the URM paper for
evaluating RM quality via Best-of-N and DPO.

This provides descriptive, non-numeric evaluation where the quality
of explanation matters — complementing GSM8K/MATH500 which focus on
verifiable numeric answers.
"""

from typing import List, Optional

from datasets import DownloadConfig, load_dataset


def load_alpaca_eval(
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load the AlpacaEval dataset.

    Returns:
        List of dicts with keys: question (instruction), reference
    """
    try:
        ds = load_dataset(
            "tatsu-lab/alpaca_eval",
            "alpaca_eval",
            split="eval",
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception:
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)

    examples = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        examples.append({
            "question": item["instruction"],
            "reference": item.get("output", ""),
        })

    return examples
