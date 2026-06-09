"""Literature Review Baselines.

Implements the formal baselines exactly as described in the literature review,
including Single RM MBR-BoN, Single RM Soft BoN, Worst-Case Optimization (Minimax),
and Uncertainty-Weighted Optimization (UWO / Sharpe / Linear).
"""

from __future__ import annotations

from typing import Dict, List
from collections import Counter

import numpy as np

from ..core.method_runners import (
    run_vanilla_bon,
    run_minimax,
    run_bon_sharpe,
    run_bon_linear,
    run_soft_bon
)
from ..core.normalize import normalize_member_scores
from ..data_utils.math500 import extract_math_answer
from ..data_utils.gsm8k import extract_gsm8k_answer
from .base import Hypothesis


class LitReviewBaselines(Hypothesis):
    name = "lit_review_baselines"
    category = "parallel"
    description = "Formal baselines from the literature review (Single RM MBR, Soft BoN, WCO, UWO)."

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
        
        # 1. Extract answers to calculate exact match overlap (for MBR)
        extracted_answers = []
        is_math = "math" in dataset.lower()
        for cand in candidates:
            if is_math:
                ans = extract_math_answer(cand)
            else:
                ans = extract_gsm8k_answer(cand)
            extracted_answers.append(ans if ans is not None else cand.strip())
            
        # Compute MBR overlap score U(y, y') = 1 if match else 0
        answer_counts = Counter(extracted_answers)
        mbr_scores = [answer_counts[ans] / N for ans in extracted_answers]

        # Ensure scores are normalized for threshold/temperature consistency
        norm_scores = normalize_member_scores(member_scores, mode="robust_zscore")
        arr = np.asarray(norm_scores, dtype=float)

        # Baseline 1: Single RM Methods (MBR-BoN and Soft BoN)
        beta = 1.0  # MBR penalty strength
        temperature = 1.0 # Soft BoN temperature
        
        for idx, rm_name in enumerate(rm_names):
            clean_rm_name = rm_name.split('/')[-1]
            rm_scores = arr[idx]
            
            # --- Single RM MBR-BoN ---
            # R(x, y) + beta * U(y)
            mbr_regularized = [s + beta * mbr for s, mbr in zip(rm_scores, mbr_scores)]
            best_idx_mbr = int(np.argmax(mbr_regularized))
            results[f"single_rm_mbr_bon ({clean_rm_name})"] = candidates[best_idx_mbr]
            
            # --- Single RM Soft BoN ---
            # Sample based on exp(r_i / T)
            # We use robust z-scores to avoid overflow and maintain scale
            z = (rm_scores - rm_scores.mean()) / (rm_scores.std() + 1e-6)
            w = np.exp((z - z.max()) / temperature)
            p = w / w.sum()
            sampled_idx = rng.choice(N, p=p)
            results[f"single_rm_soft_bon ({clean_rm_name})"] = candidates[sampled_idx]

        # Baseline 2: Worst-Case Optimization (WCO)
        # Taking the minimum across the ensemble
        results["wco_minimax"] = run_minimax(candidates, norm_scores).selected_answer
        
        # Baseline 3: Uncertainty-Weighted Optimization (UWO)
        # Using variance to penalize the mean
        ensemble_means = arr.mean(axis=0).tolist()
        ensemble_vars = arr.var(axis=0, ddof=0).tolist()
        
        results["uwo_sharpe"] = run_bon_sharpe(candidates, ensemble_means, ensemble_vars).selected_answer
        results["uwo_linear"] = run_bon_linear(candidates, ensemble_means, ensemble_vars, c=1.0).selected_answer

        return results
