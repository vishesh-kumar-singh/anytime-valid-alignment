"""H8: Reward-Weighted Self-Consistency (Answer-Level BoN).

Instead of picking the candidate with the highest RM score (Candidate-level),
or simply voting by frequency (Majority Vote), this method extracts the final
answer for each candidate and scores each unique answer by the sum of the
exponentiated RM scores of all candidates that produced it.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from .base import EvaluationPipeline, _extract_answer

class RWSCPipeline(EvaluationPipeline):
    name = "rwsc"
    category = "parallel"
    description = "Answer-level BoN (Reward-Weighted Self-Consistency) aggregates RM scores across candidates that yield the same answer. Computes for both ensemble and individual RMs."

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
            
        # Helper function to compute RWSC for a given set of scores
        def compute_rwsc(scores: List[float]) -> str:
            ans_to_scores = {}
            ans_to_candidates = {}
            
            for i in range(N):
                ans = _extract_answer(candidates[i], dataset)
                if ans is None:
                    ans = f"__noans__{i}"
                    
                if ans not in ans_to_scores:
                    ans_to_scores[ans] = []
                    ans_to_candidates[ans] = []
                    
                ans_to_scores[ans].append(scores[i])
                ans_to_candidates[ans].append((i, candidates[i], scores[i]))
                
            best_ans = None
            best_ans_score = -float('inf')
            
            for ans, s_list in ans_to_scores.items():
                # Apply robust z-scoring to prevent overflow in softmax
                raw = np.asarray(scores, dtype=float)
                z = (raw - raw.mean()) / (raw.std() + 1e-6)
                
                # We calculate the sum of softmax probabilities for the cluster
                # Note: to accurately match the original rwsc logic, we exponentiate the z-scores
                w = np.exp(z - z.max())
                
                ans_score = 0.0
                for cand_idx, _, _ in ans_to_candidates[ans]:
                    ans_score += float(w[cand_idx])
                    
                if ans_score > best_ans_score:
                    best_ans_score = ans_score
                    best_ans = ans
                    
            if best_ans is not None:
                # Pick the candidate for the best answer that had the highest individual RM score
                best_c = max(ans_to_candidates[best_ans], key=lambda x: x[2])[1]
                return best_c
            return candidates[0]

        # 1. Ensemble RWSC
        results["ensemble_rwsc"] = compute_rwsc(means)
        
        # 2. Individual RM RWSC
        for idx, rm_name in enumerate(rm_names):
            clean_rm_name = rm_name.split('/')[-1]
            rm_scores = member_scores[idx]
            results[f"single_rm_rwsc ({clean_rm_name})"] = compute_rwsc(rm_scores)
            
        return results
