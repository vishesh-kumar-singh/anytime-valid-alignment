"""H7: Sequential Early Stopping based on Reward-Weighted Self-Consistency (RWSC).

Evaluates a strategy simulating sequential generation: we observe candidates
one by one (starting from an initial pool of 4). At each step, we cluster the
candidates by their extracted answer and score them using the single best RM
(Skywork) as in H5. If the maximum cluster mass (confidence) exceeds a threshold, 
we stop and return that candidate. Otherwise, we continue generating up to N.
"""

from __future__ import annotations

from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

from .base import Hypothesis, _extract_answer

# Preference order of "strongest" general-purpose RM
_STRONGEST_RM_PREFERENCE = (
    "Skywork-Reward",
    "ArmoRM",
    "FsfairX",
    "Llama-3-OffsetBias",
    "reward-model-deberta",
)

def _pick_best_rm(rm_names: List[str]) -> int:
    for pref in _STRONGEST_RM_PREFERENCE:
        for j, name in enumerate(rm_names):
            if pref.lower() in name.lower():
                return j
    return 0


class H7ThresholdSC(Hypothesis):
    name = "h7_sequential_early_stopping"
    category = "parallel"
    description = (
        "RWSC Sequential Early Stopping: Stops generation once a candidate's "
        "answer cluster mass (using single strongest RM) exceeds a threshold."
    )
    
    TEMPERATURE = 1.0

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
        
        best_j = _pick_best_rm(rm_names)
        raw_scores = np.asarray(member_scores[best_j], dtype=float)
        
        # --- RWSC Early Stopping ---
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        initial_pool_size = 4
        
        for threshold in thresholds:
            method_name = f"seq_rwsc_thresh_{threshold}"
            
            stopped = False
            for i in range(initial_pool_size, N + 1):
                # We consider the first i candidates
                current_raw = raw_scores[:i]
                
                # z-normalise this RM's scores across the CURRENT candidate set, then softmax.
                # If std is 0, we add epsilon.
                std = current_raw.std()
                z = (current_raw - current_raw.mean()) / (std + 1e-6)
                w = np.exp((z - z.max()) / self.TEMPERATURE)
                w = w / w.sum() # Normalize so mass sums to 1.0 (confidence probability)
                
                # --- Single-best-RM RWSC (answer-level aggregation) ---
                cluster_mass: Dict[str, float] = {}
                cluster_members: Dict[str, list] = {}
                for k in range(i):
                    ans = _extract_answer(candidates[k], dataset)
                    if ans is None:
                        ans = f"__noans__{k}"  # unclustered singleton
                    cluster_mass[ans] = cluster_mass.get(ans, 0.0) + float(w[k])
                    cluster_members.setdefault(ans, []).append((k, current_raw[k]))

                best_ans = max(cluster_mass, key=cluster_mass.get)
                max_mass = cluster_mass[best_ans]
                
                if max_mass >= threshold:
                    # Representative = highest-scoring candidate within the winning cluster.
                    best_k = max(cluster_members[best_ans], key=lambda t: t[1])[0]
                    results[method_name] = candidates[best_k]
                    self.steps_used[method_name].append(i)
                    stopped = True
                    break
            
            # Fallback if we never hit the threshold: return the best seen among all N
            if not stopped:
                # Representative = highest-scoring candidate within the winning cluster at N.
                best_k = max(cluster_members[best_ans], key=lambda t: t[1])[0]
                results[method_name] = candidates[best_k]
                self.steps_used[method_name].append(N)

        # --- Pure Best RM Early Stopping ---
        best_rm_thresholds = [5.0, 10.0, 15.0, 20.0, 25.0]
        for threshold in best_rm_thresholds:
            method_name = f"seq_pure_best_rm_thresh_{threshold}"
            
            stopped = False
            for i in range(initial_pool_size, N + 1):
                current_raw = raw_scores[:i]
                max_score = np.max(current_raw)
                
                if max_score >= threshold:
                    best_k = int(np.argmax(current_raw))
                    results[method_name] = candidates[best_k]
                    self.steps_used[method_name].append(i)
                    stopped = True
                    break
            
            if not stopped:
                best_k = int(np.argmax(raw_scores))
                results[method_name] = candidates[best_k]
                self.steps_used[method_name].append(N)

        # Add baselines
        results["baseline_best_rm_bon_N"] = candidates[int(np.argmax(raw_scores))]
        
        # Add baseline H5 RWSC at N
        current_raw = raw_scores[:N]
        std = current_raw.std()
        z = (current_raw - current_raw.mean()) / (std + 1e-6)
        w = np.exp((z - z.max()) / self.TEMPERATURE)
        cluster_mass = {}
        cluster_members = {}
        for k in range(N):
            ans = _extract_answer(candidates[k], dataset)
            if ans is None:
                ans = f"__noans__{k}"
            cluster_mass[ans] = cluster_mass.get(ans, 0.0) + float(w[k])
            cluster_members.setdefault(ans, []).append((k, current_raw[k]))
        best_ans = max(cluster_mass, key=cluster_mass.get)
        best_k = max(cluster_members[best_ans], key=lambda t: t[1])[0]
        results["baseline_rwsc_h5_N"] = candidates[best_k]

        for base_method in ["baseline_skywork_bon_N", "baseline_rwsc_h5_N"]:
            self.steps_used[base_method].append(N)

        return results

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Override run to only use the maximum N (budget) since early stopping is free to stop at any N."""
        # Find the maximum N requested
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)
        
        print(f"\n[{self.name}] Overriding eval_n to only {max_n} since early stopping evaluates up to the max budget.")
        
        # Reset steps tracking
        self.steps_used.clear()
        
        # Run base evaluation
        results = super().run(
            candidates_path=candidates_path,
            score_paths=score_paths,
            eval_n=str(max_n),
            variance_norm_mode=variance_norm_mode,
            **kwargs
        )
        
        print(f"\n{'-'*60}")
        print(f"Average Candidates Generated (Cost):")
        print(f"{'-'*60}")
        avg_budget = {}
        for method_name, steps in sorted(self.steps_used.items()):
            avg_steps = np.mean(steps)
            std_steps = np.std(steps)
            avg_budget[method_name] = {"mean": float(avg_steps), "std": float(std_steps)}
            print(f"  {method_name:<30s}: {avg_steps:.2f} candidates on average")
        print(f"{'-'*60}\n")
        
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
