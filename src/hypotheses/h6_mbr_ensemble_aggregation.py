"""H6: MBR Ensemble Aggregation Methods.

Evaluates multiple ensemble strategies including MBR regularization,
robust normalization, Best-of-N (BoN), Sharpe, Linear, Soft variants,
Minimax (worst-case), Mixture-of-Bests (MoB), and Weighted BoN (WBoN).
"""

from __future__ import annotations

from typing import Any, Dict, List
from collections import Counter

import numpy as np

from ..core.method_runners import (
    run_vanilla_bon,
    run_bon_sharpe,
    run_bon_linear,
    run_soft_bon,
    run_softbon_sharpe,
    run_softbon_linear,
    run_minimax,
    run_mob_bon,
    run_wbon,
    run_wbon_sharpe,
)
from ..core.normalize import normalize_member_scores
from ..data_utils.math500 import extract_math_answer
from ..data_utils.gsm8k import extract_gsm8k_answer
from .base import Hypothesis


class H6MBREnsembleAggregation(Hypothesis):
    name = "h6_mbr_ensemble_aggregation"
    category = "parallel"
    description = (
        "Evaluates advanced ensemble aggregation strategies including MBR "
        "regularization, robust normalization, and various selection policies."
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
        N = len(candidates)
        
        # 1. Extract answers to calculate exact match overlap (MBR)
        extracted_answers = []
        is_math = "math" in dataset.lower()
        for cand in candidates:
            if is_math:
                ans = extract_math_answer(cand)
            else:
                ans = extract_gsm8k_answer(cand)
            # Default to raw string if extractor fails to at least group identical outputs
            extracted_answers.append(ans if ans is not None else cand.strip())
            
        # 2. Compute MBR overlap score U(y, y') = 1 if match else 0
        answer_counts = Counter(extracted_answers)
        mbr_scores = [answer_counts[ans] / N for ans in extracted_answers]
        
        # 3. Regularize RM scores: R(x, y) + beta * U(y) 
        beta = 1.0
        regularized_scores = []
        for rm_scores in member_scores:
            reg = [score + beta * mbr for score, mbr in zip(rm_scores, mbr_scores)]
            regularized_scores.append(reg)
            
        # 4. Robust Normalization
        norm_scores = normalize_member_scores(regularized_scores, mode="robust_zscore")
        
        # 5. Recalculate means and variances for the ensemble based on normalized scores
        arr = np.asarray(norm_scores, dtype=float)
        new_means = arr.mean(axis=0).tolist()
        new_variances = arr.var(axis=0, ddof=0).tolist()
        
        # BoN variants
        results["mbr_vanilla_bon"] = run_vanilla_bon(candidates, new_means, new_variances).selected_answer
        results["mbr_bon_sharpe"] = run_bon_sharpe(candidates, new_means, new_variances).selected_answer
        results["mbr_bon_linear"] = run_bon_linear(candidates, new_means, new_variances, c=1.0).selected_answer
        
        # Soft variants
        results["mbr_soft_bon"] = run_soft_bon(candidates, new_means, new_variances, temperature=1.0, rng=rng).selected_answer
        results["mbr_softbon_sharpe"] = run_softbon_sharpe(candidates, new_means, new_variances, temperature=1.0, eps=1e-8, rng=rng).selected_answer
        results["mbr_softbon_linear"] = run_softbon_linear(candidates, new_means, new_variances, c=1.0, temperature=1.0, rng=rng).selected_answer
        
        # Minimax (worst-case)
        results["mbr_minimax"] = run_minimax(candidates, norm_scores).selected_answer

        # MoB variants
        results["mbr_mob_vanilla"] = run_mob_bon(candidates, new_means, new_variances, use_variance_weighting=False, rng=rng).selected_answer
        results["mbr_mob_sharpe"] = run_mob_bon(candidates, new_means, new_variances, use_variance_weighting=True, rng=rng).selected_answer

        # WBoN variants
        results["mbr_wbon_inv_var"] = run_wbon(candidates, norm_scores).selected_answer
        results["mbr_wbon_sharpe"] = run_wbon_sharpe(candidates, norm_scores).selected_answer

        # MBR RWSC
        ans_to_scores = {}
        ans_to_candidates = {}
        for i in range(N):
            ans = extracted_answers[i]
            if ans not in ans_to_scores:
                ans_to_scores[ans] = []
                ans_to_candidates[ans] = []
            ans_to_scores[ans].append(new_means[i])
            ans_to_candidates[ans].append((i, candidates[i], new_means[i]))
            
        best_ans = None
        best_ans_score = -float('inf')
        for ans, scores in ans_to_scores.items():
            ans_score = np.sum(np.exp(scores))
            if ans_score > best_ans_score:
                best_ans_score = ans_score
                best_ans = ans
                
        if best_ans is not None:
            best_c = max(ans_to_candidates[best_ans], key=lambda x: x[2])[1]
            results["mbr_rwsc"] = best_c

        return results
