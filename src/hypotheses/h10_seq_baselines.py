"""H10: Sequential Literature Baselines.

Implements sequential early stopping baselines from literature:
1. AdaBoN (Adaptive burn-in budget allocation)
2. BEACON (Bayesian posterior variance stopping)
3. Pandora's Box (Weitzman UCB reservation price)

Note: Speculative Rejection is omitted because it requires token-level 
intermediate proxy rewards, which our cached dataset does not contain.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .base import Hypothesis, _extract_answer

def _pick_best_rm(rm_names: List[str]) -> int:
    """Select the best RM for single-RM baselines."""
    prefs = ["FsfairX", "Llama-3-OffsetBias", "reward-model-deberta"]
    for pref in prefs:
        for j, name in enumerate(rm_names):
            if pref.lower() in name.lower():
                return j
    return 0

class H10SeqBaselines(Hypothesis):
    name = "h10_seq_baselines"
    category = "parallel"
    description = "Sequential baselines from literature: AdaBoN, BEACON, and Pandora's Box."

    def __init__(self):
        super().__init__()
        self.steps_used = defaultdict(list)

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
        # Use a single RM's scores for these baselines as they don't natively support ensembles
        scores = np.asarray(member_scores[best_j], dtype=float)

        # ---------------------------------------------------------
        # 1. Simulated AdaBoN (Burn-in and adaptive budget)
        # ---------------------------------------------------------
        K = 4 # Burn-in
        burn_in_scores = scores[:K]
        var_burn_in = np.var(burn_in_scores)
        
        # Heuristic allocation based on variance
        var_threshold = 1.0  # z-scores typically have variance ~1.0
        if var_burn_in > var_threshold:
            # High variance = hard prompt, allocate more budget
            n_budget = min(N, K + 12)
        else:
            # Low variance = easy prompt, allocate less budget
            n_budget = min(N, K + 4)
            
        best_idx_adabon = int(np.argmax(scores[:n_budget]))
        results["baseline_adabon"] = candidates[best_idx_adabon]
        self.steps_used["baseline_adabon"].append(n_budget)

        # ---------------------------------------------------------
        # 2. Simulated BEACON (Bayesian sequential stopping)
        # ---------------------------------------------------------
        # We model the reward as drawn from N(mu, sigma^2). 
        # We maintain a posterior over mu. Stop when posterior confidence P(mu > theta_target) > gamma.
        prior_mu = 0.0
        prior_var = 1.0
        obs_var = 1.0 # Assumed observation noise
        
        gamma = 0.85 # Stopping threshold for posterior confidence
        theta_target = 0.5 # Target acceptable reward score (normalized z-score)
        stopped_beacon = False
        
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
        for i in range(1, N + 1):
            current_scores = scores[:i]
            post_var = 1.0 / ((1.0 / prior_var) + (i / obs_var))
            post_mu = post_var * ((prior_mu / prior_var) + (np.sum(current_scores) / obs_var))
            
            prob_good = norm_cdf((post_mu - theta_target) / math.sqrt(post_var))
            
            if prob_good > gamma and i >= 4:
                best_idx = int(np.argmax(current_scores))
                results["baseline_beacon"] = candidates[best_idx]
                self.steps_used["baseline_beacon"].append(i)
                stopped_beacon = True
                break
                
        if not stopped_beacon:
            best_idx = int(np.argmax(scores))
            results["baseline_beacon"] = candidates[best_idx]
            self.steps_used["baseline_beacon"].append(N)

        # ---------------------------------------------------------
        # 3. Simulated Pandora's Box (Weitzman UCB)
        # ---------------------------------------------------------
        # Stop when r* >= E[r_next] - c
        # We approximate E[r_next] via UCB of next observation: mu_hat + alpha * sigma_hat
        c = 0.05 # Cost of generating one candidate
        alpha = 1.0 # UCB exploration parameter
        
        stopped_pandora = False
        for i in range(4, N):
            current_scores = scores[:i]
            r_star = np.max(current_scores)
            mu_hat = np.mean(current_scores)
            sigma_hat = np.std(current_scores)
            
            # Fix: UCB of NEXT OBSERVATION uses sigma, not standard error of the mean (sigma/sqrt(i))
            ucb_next = mu_hat + alpha * sigma_hat
            
            if r_star >= ucb_next - c:
                best_idx = int(np.argmax(current_scores))
                results["baseline_pandora"] = candidates[best_idx]
                self.steps_used["baseline_pandora"].append(i)
                stopped_pandora = True
                break
                
        if not stopped_pandora:
            best_idx = int(np.argmax(scores))
            results["baseline_pandora"] = candidates[best_idx]
            self.steps_used["baseline_pandora"].append(N)

        return results

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)
        self.steps_used.clear()

        results = super().run(
            candidates_path=candidates_path,
            score_paths=score_paths,
            eval_n=str(max_n),
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

    def _build_summary(self, results: Dict[str, Any], plot_paths: list) -> str:
        summary = super()._build_summary(results, plot_paths)
        if "average_budget" in results:
            summary += "Average Candidates Generated (Budget):\n"
            summary += "-" * 70 + "\n"
            for m, stats in sorted(results["average_budget"].items()):
                summary += f"  {m:<40s}: {stats['mean']:6.2f} ± {stats['std']:5.2f}\n"
            summary += "\n"
        return summary
