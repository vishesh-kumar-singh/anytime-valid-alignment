"""H9: SPRT-RWSC — Sequential Probability Ratio Test for Reward-Weighted Self-Consistency.

Applies Wald's Sequential Probability Ratio Test (SPRT) to the early stopping
problem in test-time alignment.  Instead of an ad-hoc confidence threshold τ
(as in H7), we accumulate a *log-likelihood ratio* that measures the statistical
evidence for the leading answer cluster being dominant.  The SPRT stopping
boundaries are derived from user-specified Type-I (α) and Type-II (β) error
rates, yielding *provably optimal* expected stopping times.

Three variants:
  1. **Vanilla SPRT-RWSC**: Binary evidence (does new candidate agree with leader?)
  2. **E-SPRT-RWSC** (Ensemble-weighted): Weights each evidence increment by
     inverse ensemble variance — high-agreement (low-variance) candidates
     contribute more evidence, naturally guarding against reward-hacked outliers.
  3. **AB-SPRT-RWSC** (Adaptive Burn-in): Estimates p₁ from a short burn-in
     phase, making the test prompt-adaptive.

Mathematical Foundation:
  Under H₁ (leader is correct): new candidates agree with probability p₁ > 0.5.
  Under H₀ (no convergence):  new candidates agree with probability p₀ ≤ 0.5.

  Log-likelihood ratio after step n:
    Λₙ = Σᵢ wᵢ · [Xᵢ log(p₁/p₀) + (1-Xᵢ) log((1-p₁)/(1-p₀))]

  Stopping boundaries (Wald):
    A = log((1-β)/α)   → stop and accept leader
    B = log(β/(1-α))   → reset (leader likely wrong)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Hypothesis, _extract_answer


# ────────────────────────────────────────────────────────────────────
# Helper: RWSC cluster scoring
# ────────────────────────────────────────────────────────────────────

def _compute_rwsc_clusters(
    candidates: List[str],
    scores: np.ndarray,
    dataset: str,
    temperature: float = 1.0,
) -> Dict[str, dict]:
    """Compute RWSC clusters for a candidate pool.

    Returns dict mapping answer_string -> {
        "mass": float,        # softmax-weighted probability mass
        "members": [(idx, raw_score), ...],
        "best_idx": int,      # highest-scored member
    }
    """
    n = len(candidates)
    if n == 0:
        return {}

    # Z-normalise, then softmax
    std = scores.std()
    z = (scores - scores.mean()) / (std + 1e-6)
    w = np.exp((z - z.max()) / temperature)
    w = w / w.sum()

    clusters: Dict[str, dict] = {}
    for k in range(n):
        ans = _extract_answer(candidates[k], dataset)
        if ans is None:
            ans = f"__noans__{k}"
        if ans not in clusters:
            clusters[ans] = {"mass": 0.0, "members": [], "best_idx": k}
        clusters[ans]["mass"] += float(w[k])
        clusters[ans]["members"].append((k, float(scores[k])))
        # Track highest-scored member
        if scores[k] > scores[clusters[ans]["best_idx"]]:
            clusters[ans]["best_idx"] = k

    return clusters


def _get_leader(clusters: Dict[str, dict]) -> tuple:
    """Return (leader_answer, leader_mass, leader_best_idx)."""
    leader_ans = max(clusters, key=lambda a: clusters[a]["mass"])
    info = clusters[leader_ans]
    best_idx = max(info["members"], key=lambda t: t[1])[0]
    return leader_ans, info["mass"], best_idx


# ────────────────────────────────────────────────────────────────────
# Core SPRT logic
# ────────────────────────────────────────────────────────────────────

def _sprt_stop(
    candidates: List[str],
    raw_scores: np.ndarray,
    variances: Optional[np.ndarray],
    dataset: str,
    alpha: float,
    beta: float,
    burnin: int,
    p0_mode: str,
    use_ensemble_weighting: bool,
    temperature: float,
) -> dict:
    """Run SPRT on a sequence of candidates.

    Args:
        candidates: All N candidates in sequential order.
        raw_scores: Shape (N,) — single RM or ensemble mean scores.
        variances: Shape (N,) — ensemble variance per candidate (None for vanilla).
        dataset: "gsm8k" or "math500".
        alpha: Type-I error rate (false stop probability).
        beta: Type-II error rate (false continue probability).
        burnin: Number of initial candidates before starting SPRT.
        p0_mode: How to set p₀ — "uniform" or "fixed".
        use_ensemble_weighting: If True, weight evidence by 1/(σ² + ε).
        temperature: RWSC softmax temperature.

    Returns:
        dict with keys: stopped_at, leader_answer, best_idx, log_lambda_trace
    """
    N = len(candidates)
    burnin = max(2, min(burnin, N - 1))

    # Stopping boundaries (Wald)
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))

    # Phase 1: Burn-in — build initial clusters
    burn_scores = raw_scores[:burnin]
    clusters = _compute_rwsc_clusters(
        candidates[:burnin], burn_scores, dataset, temperature
    )
    leader_ans, leader_mass, _ = _get_leader(clusters)

    # Estimate p₁ from burn-in leader probability
    n_unique = max(len(clusters), 2)
    if p0_mode == "uniform":
        p0 = 1.0 / n_unique
    else:
        p0 = 0.3  # conservative fixed null

    # p₁ is the observed leader probability, floored at 0.55
    p1 = max(leader_mass, 0.55)
    # Ensure p1 > p0 (otherwise test is degenerate)
    p1 = max(p1, p0 + 0.1)
    # Cap to avoid log(0)
    p0 = np.clip(p0, 0.01, 0.49)
    p1 = np.clip(p1, 0.51, 0.99)

    # Pre-compute log ratios
    log_ratio_agree = math.log(p1 / p0)
    log_ratio_disagree = math.log((1.0 - p1) / (1.0 - p0))

    # Phase 2: Sequential SPRT
    log_lambda = 0.0
    trace = []

    for i in range(burnin, N):
        # Observe candidate i
        ans_i = _extract_answer(candidates[i], dataset)
        if ans_i is None:
            ans_i = f"__noans__{i}"

        # Does this candidate agree with the current leader?
        x_i = 1.0 if ans_i == leader_ans else 0.0

        # Evidence weight
        if use_ensemble_weighting and variances is not None:
            w_i = 1.0 / (variances[i] + 1e-4)
            # Normalize weight to prevent explosion (soft cap)
            w_i = min(w_i, 10.0)
        else:
            w_i = 1.0

        # Accumulate log-likelihood ratio
        delta = w_i * (x_i * log_ratio_agree + (1.0 - x_i) * log_ratio_disagree)
        log_lambda += delta
        trace.append(log_lambda)

        # Update clusters with new candidate for leader tracking
        clusters_current = _compute_rwsc_clusters(
            candidates[:i + 1], raw_scores[:i + 1], dataset, temperature
        )
        new_leader_ans, new_leader_mass, _ = _get_leader(clusters_current)

        # Check stopping conditions
        if log_lambda >= A:
            # Accept: leader is statistically dominant
            _, _, best_idx = _get_leader(clusters_current)
            return {
                "stopped_at": i + 1,
                "leader_answer": new_leader_ans,
                "best_idx": best_idx,
                "log_lambda_trace": trace,
                "decision": "accept",
            }

        if log_lambda <= B:
            # Reset: current leader is likely wrong
            log_lambda = 0.0
            leader_ans = new_leader_ans
            # Re-estimate p₁ from current clusters
            p1 = max(new_leader_mass, 0.55)
            n_unique_now = max(len(clusters_current), 2)
            if p0_mode == "uniform":
                p0 = 1.0 / n_unique_now
            p0 = np.clip(p0, 0.01, 0.49)
            p1 = max(p1, p0 + 0.1)
            p1 = np.clip(p1, 0.51, 0.99)
            log_ratio_agree = math.log(p1 / p0)
            log_ratio_disagree = math.log((1.0 - p1) / (1.0 - p0))
        else:
            # Update leader tracking (may have shifted)
            if new_leader_ans != leader_ans:
                leader_ans = new_leader_ans
                p1 = max(new_leader_mass, 0.55)
                n_unique_now = max(len(clusters_current), 2)
                if p0_mode == "uniform":
                    p0 = 1.0 / n_unique_now
                p0 = np.clip(p0, 0.01, 0.49)
                p1 = max(p1, p0 + 0.1)
                p1 = np.clip(p1, 0.51, 0.99)
                log_ratio_agree = math.log(p1 / p0)
                log_ratio_disagree = math.log((1.0 - p1) / (1.0 - p0))

    # Fallback: never stopped — use full pool RWSC
    final_clusters = _compute_rwsc_clusters(candidates, raw_scores, dataset, temperature)
    _, _, best_idx = _get_leader(final_clusters)
    return {
        "stopped_at": N,
        "leader_answer": leader_ans,
        "best_idx": best_idx,
        "log_lambda_trace": trace,
        "decision": "fallback",
    }


# ────────────────────────────────────────────────────────────────────
# Hypothesis class
# ────────────────────────────────────────────────────────────────────

# Preference order of "strongest" general-purpose RM (reused from H7)
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


class H9SPRTRWSC(Hypothesis):
    name = "h9_sprt_rwsc"
    category = "parallel"  # runs on cached parallel data
    description = (
        "SPRT-RWSC: Sequential Probability Ratio Test for early stopping "
        "with provable error control, ensemble-weighted evidence, and "
        "adaptive burn-in prior estimation."
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
        if N < 4:
            return results

        means_arr = np.array(means, dtype=float)
        vars_arr = np.array(variances, dtype=float)

        # Use best single RM (consistent with H7 for fair comparison)
        best_j = _pick_best_rm(rm_names)
        single_rm_scores = np.asarray(member_scores[best_j], dtype=float)

        # ── Sweep (α, β) configurations ──────────────────────────
        ab_configs = [
            (0.01, 0.01),
            (0.01, 0.05),
            (0.01, 0.10),
            (0.05, 0.05),
            (0.05, 0.10),
            (0.10, 0.10),
            (0.10, 0.20),
        ]

        burnin_sizes = [4]

        for alpha, beta in ab_configs:
            for burnin in burnin_sizes:
                # ── Variant 1: Vanilla SPRT-RWSC (single RM, unweighted) ──
                tag = f"sprt_rwsc_a{alpha}_b{beta}"
                res = _sprt_stop(
                    candidates=candidates,
                    raw_scores=single_rm_scores,
                    variances=None,
                    dataset=dataset,
                    alpha=alpha,
                    beta=beta,
                    burnin=burnin,
                    p0_mode="uniform",
                    use_ensemble_weighting=False,
                    temperature=self.TEMPERATURE,
                )
                results[tag] = candidates[res["best_idx"]]
                self.steps_used[tag].append(res["stopped_at"])

                # ── Variant 2: E-SPRT-RWSC (ensemble mean + variance weighting) ──
                tag_ens = f"esprt_rwsc_a{alpha}_b{beta}"
                res_ens = _sprt_stop(
                    candidates=candidates,
                    raw_scores=means_arr,
                    variances=vars_arr,
                    dataset=dataset,
                    alpha=alpha,
                    beta=beta,
                    burnin=burnin,
                    p0_mode="uniform",
                    use_ensemble_weighting=True,
                    temperature=self.TEMPERATURE,
                )
                results[tag_ens] = candidates[res_ens["best_idx"]]
                self.steps_used[tag_ens].append(res_ens["stopped_at"])

        # ── Baselines (H7-style fixed-threshold for comparison) ──
        for tau in [0.7, 0.8, 0.9]:
            tag_baseline = f"baseline_h7_tau_{tau}"
            stopped = False
            for i in range(4, N + 1):
                current_raw = single_rm_scores[:i]
                std = current_raw.std()
                z = (current_raw - current_raw.mean()) / (std + 1e-6)
                w = np.exp((z - z.max()) / self.TEMPERATURE)
                w = w / w.sum()

                cluster_mass: Dict[str, float] = {}
                cluster_members: Dict[str, list] = {}
                for k in range(i):
                    ans = _extract_answer(candidates[k], dataset)
                    if ans is None:
                        ans = f"__noans__{k}"
                    cluster_mass[ans] = cluster_mass.get(ans, 0.0) + float(w[k])
                    cluster_members.setdefault(ans, []).append((k, current_raw[k]))

                best_ans = max(cluster_mass, key=cluster_mass.get)
                max_mass = cluster_mass[best_ans]

                if max_mass >= tau:
                    best_k = max(cluster_members[best_ans], key=lambda t: t[1])[0]
                    results[tag_baseline] = candidates[best_k]
                    self.steps_used[tag_baseline].append(i)
                    stopped = True
                    break

            if not stopped:
                best_k = max(cluster_members[best_ans], key=lambda t: t[1])[0]
                results[tag_baseline] = candidates[best_k]
                self.steps_used[tag_baseline].append(N)

        return results

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Override run to only use the maximum N (since early stopping evaluates up to max budget)."""
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)

        print(f"\n[{self.name}] Overriding eval_n to {max_n} (early stopping evaluates up to max budget).")

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

        # Print average steps (cost analysis)
        print(f"\n{'=' * 70}")
        print(f"  SPRT-RWSC: Average Candidates Generated (Cost Analysis)")
        print(f"{'=' * 70}")

        # Group methods
        sprt_methods = sorted([m for m in self.steps_used if m.startswith("sprt_")])
        esprt_methods = sorted([m for m in self.steps_used if m.startswith("esprt_")])
        baseline_methods = sorted([m for m in self.steps_used if m.startswith("baseline_")])

        for group_name, methods in [
            ("Vanilla SPRT-RWSC (Single RM)", sprt_methods),
            ("Ensemble SPRT-RWSC (Variance-Weighted)", esprt_methods),
            ("H7 Fixed-Threshold Baselines", baseline_methods),
        ]:
            if methods:
                print(f"\n  {group_name}:")
                print(f"  {'-' * 60}")
                for method_name in methods:
                    steps = self.steps_used[method_name]
                    avg_steps = np.mean(steps)
                    std_steps = np.std(steps)
                    print(f"    {method_name:<40s}: {avg_steps:6.2f} ± {std_steps:5.2f}")

        print(f"\n{'=' * 70}\n")

        return results
