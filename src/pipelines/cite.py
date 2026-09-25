"""H9: Original W-CITE Pipeline

This isolates the CITE and original W-CITE algorithms from the E-Process pipeline.
It loads candidates that already have W-CITE generative probabilities calculated.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .base import EvaluationPipeline, _extract_answer, get_semantic_cluster_ids
from ..methods.sequential.e_process import _cite_stop
from ..methods.sequential.sprt import _pick_best_rm


class CITEPipeline(EvaluationPipeline):
    name = "cite_isolated"
    category = "parallel"
    description = (
        "Isolated CITE pipeline for Vanilla CITE and generative W-CITE."
    )

    def __init__(self, alpha_values=None, burnin_sizes=None):
        super().__init__()
        self.steps_used = defaultdict(list)
        self.prompt_stats = {"theta_star": [], "steps": defaultdict(list)}
        
        self.alpha_values = alpha_values if alpha_values else [0.05, 0.1, 0.2, 0.3, 0.4]
        self.burnin_sizes = burnin_sizes if burnin_sizes else [4]

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

        global_cluster_ids = get_semantic_cluster_ids(candidates, dataset)
        
        from collections import Counter
        counts = Counter(global_cluster_ids)
        theta_star = max(counts.values()) / N if counts else 0.0
        self.prompt_stats["theta_star"].append(theta_star)

        # W-CITE original probabilities are passed via `means` by the caller in this special pipeline!
        # Wait! How do we pass wcite_weights? 
        # In `scripts/score_candidates.py` or similar, usually `scores` is passed.
        # But this pipeline will just take wcite_weights instead of RM scores.
        # So we expect `means` to actually be the generative wcite_weights!
        wcite_weights = np.array(means, dtype=float)

        for alpha in self.alpha_values:
            for burnin in self.burnin_sizes:
                b_str = f"_n0{burnin}" if len(self.burnin_sizes) > 1 else ""
                tag_suffix = f"_a{alpha}{b_str}"

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
                
                # W-CITE (Generative Probabilities)
                tag_wcite = f"w_cite_original{tag_suffix}"
                res_wcite = _cite_stop(
                    candidates=candidates,
                    raw_scores=wcite_weights,
                    variances=None,
                    global_cluster_ids=global_cluster_ids,
                    alpha=alpha,
                    burnin=burnin,
                    use_wcite=True,
                )
                results[tag_wcite] = candidates[res_wcite["best_idx"]]
                self.steps_used[tag_wcite].append(res_wcite["stopped_at"])

        return results
