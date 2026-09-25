"""H10: Sequential Literature Baselines.

Implements exact mathematical sequential early stopping baselines from literature:
1. AdaBoN (Adaptive global variance-proportional budget allocation)
2. BEACON (Bayesian posterior variance stopping with adaptive target)
3. Pandora's Box (Weitzman exact Gaussian expected marginal gain)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .base import EvaluationPipeline, _extract_answer, get_semantic_cluster_ids

def _pick_best_rm(rm_names: List[str]) -> int:
    prefs = ["FsfairX", "Llama-3-OffsetBias", "reward-model-deberta", "ArmoRM", "Skywork"]
    for pref in prefs:
        for j, name in enumerate(rm_names):
            if pref.lower() in name.lower():
                return j
    return 0

from ..methods.sequential.baselines import compute_adabon, compute_beacon, compute_pandora, compute_consol

class SeqBaselinesPipeline(EvaluationPipeline):
    name = "seq_baselines"
    category = "parallel"
    description = "Exact sequential baselines from literature: AdaBoN, BEACON, and Pandora's Box."

    def __init__(self):
        super().__init__()
        self.steps_used = defaultdict(list)
        self.adabon_budgets = {}
        self._candidate_to_idx = {}

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
        if N < 4:
            return results

        best_j = _pick_best_rm(rm_names)
        scores = np.asarray(member_scores[best_j], dtype=float)

        first_cand = candidates[0]
        idx = self._candidate_to_idx.get(first_cand, -1)

        # ---------------------------------------------------------
        # 1. Exact AdaBoN (Global variance-proportional allocation)
        # ---------------------------------------------------------
        n_budget = self.adabon_budgets.get(N, {}).get(idx, N)
        adabon_ans, adabon_steps = compute_adabon(candidates, scores, n_budget)
        results["baseline_adabon"] = adabon_ans
        self.steps_used["baseline_adabon"].append(adabon_steps)

        # ---------------------------------------------------------
        # 2. Adaptive BEACON (Bayesian sequential stopping)
        # ---------------------------------------------------------
        beacon_ans, beacon_steps = compute_beacon(candidates, scores, gamma=0.85)
        results["baseline_beacon"] = beacon_ans
        self.steps_used["baseline_beacon"].append(beacon_steps)

        # ---------------------------------------------------------
        # 3. Exact Pandora's Box (Weitzman Expected Marginal Gain)
        # ---------------------------------------------------------
        pandora_ans, pandora_steps = compute_pandora(candidates, scores, c=0.05)
        results["baseline_pandora"] = pandora_ans
        self.steps_used["baseline_pandora"].append(pandora_steps)

        # ---------------------------------------------------------
        # 4. ConSol (SPRT on top 2 frequent answers)
        # ---------------------------------------------------------
        global_cluster_ids = get_semantic_cluster_ids(candidates, dataset)
        consol_ans, consol_steps = compute_consol(candidates, global_cluster_ids)
        results["baseline_consol"] = consol_ans
        self.steps_used["baseline_consol"].append(consol_steps)

        return results

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        examples = self.load_candidates(candidates_path)
        rm_names, all_rm_scores = self.load_scores(score_paths)
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        
        self._candidate_to_idx.clear()
        for ex in examples:
            self._candidate_to_idx[ex["candidates"][0]] = ex["example_index"]

        self.adabon_budgets.clear()
        K = 4
        best_j = _pick_best_rm(rm_names)
        
        for n in sorted(n_values):
            self.adabon_budgets[n] = {}
            if n <= K:
                for ex in examples:
                    self.adabon_budgets[n][ex["example_index"]] = n
                continue
                
            stds = {}
            for ex in examples:
                idx = ex["example_index"]
                if idx not in all_rm_scores or len(all_rm_scores[idx]) != len(rm_names):
                    continue
                scores = np.asarray(all_rm_scores[idx][best_j][:n], dtype=float)
                if len(scores) >= K:
                    stds[idx] = np.std(scores[:K])
                    
            total_stds = sum(stds.values())
            if total_stds > 1e-6:
                M = len(stds)
                total_remaining_budget = M * (n - K)
                for idx, std in stds.items():
                    prop = std / total_stds
                    budget = K + int(round(total_remaining_budget * prop))
                    self.adabon_budgets[n][idx] = min(budget, n)
            else:
                for idx in stds:
                    self.adabon_budgets[n][idx] = n
                    
        self.steps_used.clear()

        results = super().run(
            candidates_path=candidates_path,
            score_paths=score_paths,
            eval_n=eval_n,
            variance_norm_mode=variance_norm_mode,
            **kwargs
        )

        print(f"\n{'=' * 70}")
        print(f"  Sequential Baselines: Average Candidates Generated")
        print(f"{'=' * 70}")
        avg_budget = {}
        for method_name in sorted(self.steps_used.keys()):
            steps = self.steps_used[method_name]
            avg_steps = np.mean(steps)
            std_steps = np.std(steps)
            avg_budget[method_name] = {"mean": float(avg_steps), "std": float(std_steps)}
            print(f"    {method_name:<30s}: {avg_steps:6.2f} ± {std_steps:5.2f}")
        print(f"{'=' * 70}\n")

        results["average_budget"] = avg_budget
        return results
