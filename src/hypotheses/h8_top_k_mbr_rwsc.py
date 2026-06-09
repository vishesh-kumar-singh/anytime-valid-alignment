"""H8: Top-K MBR-Filtered RWSC.

This hypothesis explores pushing the theoretical accuracy ceiling by pre-filtering
the candidate pool before running Reward-Weighted Self-Consistency.
It scores candidates using both Minimum Bayes Risk (topological consensus) and
Variance Penalization (Sharpe Ratio), and then performs RWSC exclusively on the
Top-K highest-scoring candidates to filter out reward-hacked anomalies.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from collections import defaultdict

from .base import Hypothesis, _extract_answer


class H8TopKMBRRWSC(Hypothesis):
    name = "h8_top_k_mbr_rwsc"
    category = "parallel"
    description = "Pre-filters candidates by MBR+Sharpe and runs RWSC only on the top K candidates."

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
            
        means_arr = np.array(means)
        variances_arr = np.array(variances)
        
        # 1. Compute Variance Penalization (Sharpe Ratio)
        sharpe = means_arr / (np.sqrt(variances_arr) + 1e-6)
        
        # 2. Extract Answers
        answers = [_extract_answer(c, dataset) or c for c in candidates]
        
        # 3. Compute MBR overlap score (fraction of pool with exact same answer)
        adj = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if answers[i] == answers[j]:
                    adj[i, j] = 1.0
        mbr_scores = np.mean(adj, axis=1)
        
        # 4. Combined Filtering Score
        combined_scores = sharpe + mbr_scores
        sorted_idx = np.argsort(combined_scores)[::-1]
        
        # 5. Sweep Top-K RWSC
        for k in [5, 10, 11, 13, 15]:
            if k > N:
                continue
                
            top_k_idx = sorted_idx[:k]
            
            ans_rwsc = defaultdict(float)
            ans_cands = defaultdict(list)
            
            for i in top_k_idx:
                ans_rwsc[answers[i]] += np.exp(means_arr[i])
                ans_cands[answers[i]].append((i, candidates[i], means_arr[i]))
                
            best_ans = max(ans_rwsc, key=ans_rwsc.get)
            best_c = max(ans_cands[best_ans], key=lambda x: x[2])[1]
            
            results[f"top_{k}_mbr_rwsc"] = best_c
            
        return results
