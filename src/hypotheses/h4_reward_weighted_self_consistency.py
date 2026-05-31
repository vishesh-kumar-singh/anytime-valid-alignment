"""H8: Reward-Weighted Self-Consistency (Answer-Level BoN).

Instead of picking the candidate with the highest RM score (Candidate-level),
or simply voting by frequency (Majority Vote), this method extracts the final
answer for each candidate and scores each unique answer by the sum of the
exponentiated RM scores of all candidates that produced it.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from .base import Hypothesis, _extract_answer

class H4RewardWeightedSelfConsistency(Hypothesis):
    name = "h4_reward_weighted_self_consistency"
    category = "parallel"
    description = "Answer-level BoN (Reward-Weighted Self-Consistency) aggregates RM scores across candidates that yield the same answer."

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
        
        N = len(candidates)
        if N == 0:
            return results
            
        # Group candidates by extracted answer
        ans_to_scores = {}
        ans_to_candidates = {}
        
        for i in range(N):
            ans = _extract_answer(candidates[i], dataset)
            if ans is None:
                ans = candidates[i]
                
            if ans not in ans_to_scores:
                ans_to_scores[ans] = []
                ans_to_candidates[ans] = []
                
            ans_to_scores[ans].append(means[i])
            ans_to_candidates[ans].append((i, candidates[i], means[i]))
            
        # Score each answer using the sum of exponentiated RM scores
        best_ans = None
        best_ans_score = -float('inf')
        
        for ans, scores in ans_to_scores.items():
            ans_score = np.sum(np.exp(scores))
            if ans_score > best_ans_score:
                best_ans_score = ans_score
                best_ans = ans
                
        # Pick the candidate for the best answer that had the highest individual RM score
        if best_ans is not None:
            best_c = max(ans_to_candidates[best_ans], key=lambda x: x[2])[1]
            results["reward_weighted_self_consistency"] = best_c
            
        return results
