"""H3: Self-Consistency Baselines.

Evaluates Majority Vote, RM-within-Majority, and RM-Consistency-Agree —
methods that leverage answer-level consensus rather than (or combined with)
reward model scores.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Hypothesis, _extract_answer


class H3SelfConsistency(Hypothesis):
    name = "h3_self_consistency"
    category = "parallel"
    description = (
        "Self-consistency (majority vote) provides a free signal orthogonal "
        "to RM scoring; combining them yields better selection."
    )

    def evaluate_methods(
        self,
        candidates: List[str],
        means: List[float],
        variances: List[float],
        member_scores: List[List[float]],
        rm_names: List[str],
        dataset: str,
        rng: np.random.Generator,
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}

        answers = [_extract_answer(c, dataset) for c in candidates]
        answer_counts = Counter(a for a in answers if a is not None)

        # Fallback: if no answers can be extracted, default to first candidate
        fallback = candidates[0]

        if answer_counts:
            majority_answer = answer_counts.most_common(1)[0][0]

            # 1. Majority Vote
            majority_idx = next((i for i, a in enumerate(answers) if a == majority_answer), 0)
            results["majority_vote"] = candidates[majority_idx]

            # 2. RM-within-Majority
            majority_indices = [i for i, a in enumerate(answers) if a == majority_answer]
            majority_means = [means[i] for i in majority_indices]
            best_in_majority = majority_indices[int(np.argmax(majority_means))]
            results["rm_within_majority"] = candidates[best_in_majority]

            # 3. RM-Consistency Agree
            rm_idx = int(np.argmax(means))
            rm_answer = answers[rm_idx]
            selected = rm_idx if rm_answer == majority_answer else majority_idx
            results["rm_consistency_agree"] = candidates[selected]
        else:
            # No extractable answers — all methods get the fallback
            results["majority_vote"] = fallback
            results["rm_within_majority"] = fallback
            results["rm_consistency_agree"] = fallback

        return results
