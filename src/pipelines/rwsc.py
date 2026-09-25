"""H8: Reward-Weighted Self-Consistency (Answer-Level BoN).

Instead of picking the candidate with the highest RM score (Candidate-level),
or simply voting by frequency (Majority Vote), this method extracts the final
answer for each candidate and scores each unique answer by the sum of the
exponentiated RM scores of all candidates that produced it.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from collections import Counter

from .base import EvaluationPipeline, _extract_answer, get_semantic_cluster_ids
from ..methods.parallel.rwsc import compute_rwsc, compute_borda_voting

class RWSCPipeline(EvaluationPipeline):
    name = "rwsc"
    category = "parallel"
    description = "Answer-level BoN (Reward-Weighted Self-Consistency) aggregates RM scores across candidates that yield the same answer. Computes for both ensemble and individual RMs."

    def __init__(self, temperatures=None, p_borda=1.2):
        super().__init__()
        self.temperatures = temperatures if temperatures is not None else [1.0]
        self.P_BORDA = p_borda

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
        # Pre-compute semantic clusters for this question
        cluster_ids = get_semantic_cluster_ids(candidates, dataset)
        
        # 1. Ensemble Aggregations
        for t in self.temperatures:
            tag = f"ensemble_rwsc_t{t}" if len(self.temperatures) > 1 else "ensemble_rwsc"
            results[tag] = compute_rwsc(means, candidates, cluster_ids, t)
            
        results["ensemble_borda"] = compute_borda_voting(means, candidates, cluster_ids, self.P_BORDA)
        
        # 2. Individual RM Aggregations
        for idx, rm_name in enumerate(rm_names):
            clean_rm_name = rm_name.split('/')[-1]
            rm_scores = member_scores[idx]
            for t in self.temperatures:
                tag = f"single_rm_rwsc ({clean_rm_name})_t{t}" if len(self.temperatures) > 1 else f"single_rm_rwsc ({clean_rm_name})"
                results[tag] = compute_rwsc(rm_scores, candidates, cluster_ids, t)
                
            results[f"single_rm_borda ({clean_rm_name})"] = compute_borda_voting(rm_scores, candidates, cluster_ids, self.P_BORDA)
                
        # 3. Pure SC (Majority Vote without scores)
        ans_counts = Counter(cluster_ids)
        if ans_counts:
            majority_c_id = ans_counts.most_common(1)[0][0]
            majority_idx = cluster_ids.index(majority_c_id)
            results["pure_sc"] = candidates[majority_idx]
            
        return results
