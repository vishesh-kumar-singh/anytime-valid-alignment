"""H9: Anytime-Valid Early Stopping via E-Processes (e-RWSC)

Applies anytime-valid stopping based on betting e-processes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .base import EvaluationPipeline, _extract_answer, get_semantic_cluster_ids
from ..methods.sequential.e_process import _e_rwsc_stop, _e_borda_stop, _cite_stop
from ..methods.sequential.sprt import _pick_best_rm


class EProcessPipeline(EvaluationPipeline):
    name = "e_rwsc"
    category = "parallel"
    description = (
        "e-RWSC: Anytime-valid early stopping using e-processes, "
        "supporting variance-gated robust certification."
    )

    def __init__(self, alpha_values=None, p_dagger_values=None, m_values=None, c_values=None, temperature=1.0, use_ensemble=False):
        super().__init__()
        self.steps_used = defaultdict(list)
        self.prompt_stats = {"theta_star": [], "steps": defaultdict(list)}
        self.alpha_values = alpha_values if alpha_values else [0.05, 0.1, 0.2, 0.3, 0.4]
        self.p_dagger_values = p_dagger_values if p_dagger_values else [0.5]
        self.m_values = m_values if m_values else [4.0]
        self.c_values = c_values if c_values else [0.8]
        self.burnin_sizes = [4]
        self.TEMPERATURE = temperature
        self.use_ensemble = use_ensemble


    def evaluate_methods(
        self,
        candidates,
        means,
        variances,
        member_scores,
        rm_names,
        dataset,
        rng,
    ):
        results = {}
        N = len(candidates)
        if N < 4:
            return results

        means_arr = np.array(means, dtype=float)
        vars_arr = np.array(variances, dtype=float)

        best_j = _pick_best_rm(rm_names)
        single_rm_scores = np.asarray(member_scores[best_j], dtype=float)

        global_cluster_ids = get_semantic_cluster_ids(candidates, dataset)
        
        from collections import Counter
        counts = Counter(global_cluster_ids)
        theta_star = max(counts.values()) / N if counts else 0.0
        self.prompt_stats["theta_star"].append(theta_star)

        
        m = self.m_values[0]
        

        for alpha in getattr(self, "alpha_values", [0.05, 0.1, 0.2, 0.3, 0.4]):
            for pdag in getattr(self, "p_dagger_values", [0.5]):
                for c_val in getattr(self, "c_values", [0.8]):
                    for burnin in getattr(self, "burnin_sizes", [4]):
                        b_str = f"_n0{burnin}" if len(getattr(self, "burnin_sizes", [4])) > 1 else ""
                        tag_suffix = f"_a{alpha}_p{pdag}_m{m}_c{c_val}{b_str}"

                        # Vanilla e-RWSC
                        tag_erwsc = f"e_rwsc{tag_suffix}"
                        res_erwsc = _e_rwsc_stop(
                            candidates=candidates,
                            raw_scores=single_rm_scores,
                            variances=None,
                            global_cluster_ids=global_cluster_ids,
                            alpha=alpha,
                            p_dagger=pdag,
                            c=c_val,
                            m=m,
                            use_ensemble_weighting=False,
                            temperature=self.TEMPERATURE,
                            burnin=burnin,
                        )
                        results[tag_erwsc] = candidates[res_erwsc["best_idx"]]
                        self.steps_used[tag_erwsc].append(res_erwsc["stopped_at"])

                        # Vanilla e-Borda
                        tag_eborda = f"e_borda{tag_suffix}"
                        res_eborda = _e_borda_stop(
                            candidates=candidates,
                            raw_scores=single_rm_scores,
                            variances=None,
                            global_cluster_ids=global_cluster_ids,
                            alpha=alpha,
                            p_dagger=pdag,
                            c=c_val,
                            m=m,
                            burnin=burnin,
                        )
                        results[tag_eborda] = candidates[res_eborda["best_idx"]]
                        self.steps_used[tag_eborda].append(res_eborda["stopped_at"])
                        
                        # CITE (Vanilla Unweighted)
                        tag_cite = f"cite{tag_suffix}"
                        res_cite = _cite_stop(
                            candidates=candidates,
                            raw_scores=None,
                            variances=None,
                            global_cluster_ids=global_cluster_ids,
                            alpha=alpha,
                            burnin=burnin,
                            use_wcite=False,
                        )
                        results[tag_cite] = candidates[res_cite["best_idx"]]
                        self.steps_used[tag_cite].append(res_cite["stopped_at"])
                        
                        # W-CITE (Reward-Weighted)
                        tag_wcite = f"w_cite{tag_suffix}"
                        res_wcite = _cite_stop(
                            candidates=candidates,
                            raw_scores=single_rm_scores,
                            variances=None,
                            global_cluster_ids=global_cluster_ids,
                            alpha=alpha,
                            burnin=burnin,
                            use_wcite=True,
                        )
                        results[tag_wcite] = candidates[res_wcite["best_idx"]]
                        self.steps_used[tag_wcite].append(res_wcite["stopped_at"])
                
                        # e-RWSC-V (Variance Gated)
                        if self.use_ensemble:
                            tag_erwsc_v = f"e_rwsc_v{tag_suffix}"
                            res_erwsc_v = _e_rwsc_stop(
                                candidates=candidates,
                                raw_scores=means_arr,
                                variances=vars_arr,
                                global_cluster_ids=global_cluster_ids,
                                alpha=alpha,
                                p_dagger=pdag,
                                c=c_val,
                                m=m,
                                use_ensemble_weighting=False,
                                temperature=self.TEMPERATURE,
                                burnin=burnin,
                            )
                            results[tag_erwsc_v] = candidates[res_erwsc_v["best_idx"]]
                            self.steps_used[tag_erwsc_v].append(res_erwsc_v["stopped_at"])

                            # e-SC-V (Variance Gated SC)
                            tag_esc_v = f"e_sc_v{tag_suffix}"
                            res_esc_v = _e_rwsc_stop(
                                candidates=candidates,
                                raw_scores=None,
                                variances=vars_arr,
                                global_cluster_ids=global_cluster_ids,
                                alpha=alpha,
                                p_dagger=pdag,
                                c=c_val,
                                m=m,
                                use_ensemble_weighting=False,
                                temperature=self.TEMPERATURE,
                                burnin=burnin,
                            )
                            results[tag_esc_v] = candidates[res_esc_v["best_idx"]]
                            self.steps_used[tag_esc_v].append(res_esc_v["stopped_at"])

                            tag_eborda_v = f"e_borda_v{tag_suffix}"
                            res_eborda_v = _e_borda_stop(
                                candidates=candidates,
                                raw_scores=means_arr,
                                variances=vars_arr,
                                global_cluster_ids=global_cluster_ids,
                                alpha=alpha,
                                p_dagger=pdag,
                                c=c_val,
                                m=m,
                                burnin=burnin,
                            )
                            results[tag_eborda_v] = candidates[res_eborda_v["best_idx"]]
                            self.steps_used[tag_eborda_v].append(res_eborda_v["stopped_at"])

        return results

    def run(self, candidates_path: str, score_paths: list[str], eval_n: str = "24", variance_norm_mode: str = "none", **kwargs):
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)

        print(f"\n[{self.name}] Overriding eval_n to {max_n} (early stopping evaluates up to max budget).")

        self.steps_used.clear()

        results = super().run(
            candidates_path=candidates_path,
            score_paths=score_paths,
            eval_n=str(max_n),
            variance_norm_mode=variance_norm_mode,
            **kwargs
        )

        print(f"\n{'=' * 70}")
        print(f"  E-Process: Average Candidates Generated (Cost Analysis)")
        print(f"{'=' * 70}")

        avg_budget = {}

        vanilla_erwsc = sorted([m for m in self.steps_used if m.startswith("e_rwsc_a")])
        vanilla_borda = sorted([m for m in self.steps_used if m.startswith("e_borda_a")])
        vanilla_cite = sorted([m for m in self.steps_used if m.startswith("cite_a")])
        vanilla_wcite = sorted([m for m in self.steps_used if m.startswith("w_cite_a")])
        ens_erwsc = sorted([m for m in self.steps_used if m.startswith("e_rwsc_v")])
        ens_borda = sorted([m for m in self.steps_used if m.startswith("e_borda_v")])

        for group_name, methods in [
            ("Vanilla e-RWSC (Anytime-Valid)", vanilla_erwsc),
            ("Vanilla e-Borda", vanilla_borda),
            ("Vanilla CITE", vanilla_cite),
            ("W-CITE", vanilla_wcite),
            ("Ensemble e-RWSC (Variance-Gated)", ens_erwsc),
            ("Ensemble e-Borda (Variance-Gated)", ens_borda),
        ]:
            if methods:
                print(f"\n  {group_name}:")
                print(f"  {'-' * 60}")
                for method_name in methods:
                    steps = self.steps_used[method_name]
                    avg_steps = sum(steps) / len(steps)
                    
                    import math
                    mean = sum(steps) / len(steps)
                    variance = sum([((x - mean) ** 2) for x in steps]) / len(steps)
                    std_steps = math.sqrt(variance)
                    
                    avg_budget[method_name] = {"mean": float(avg_steps), "std": float(std_steps)}
                    print(f"    {method_name:<40s}: {avg_steps:6.2f} ± {std_steps:5.2f}")

        print(f"\n{'=' * 70}\n")
        
        results["average_budget"] = avg_budget
        return results
