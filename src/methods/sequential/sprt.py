import math
from typing import Dict, List, Optional
import numpy as np

def _compute_rwsc_clusters(
    cluster_ids: List[int],
    scores: np.ndarray,
    temperature: float = 1.0,
) -> Dict[int, dict]:
    """Compute RWSC clusters for a candidate pool using precomputed semantic cluster IDs.

    Returns dict mapping cluster_id -> {
        "mass": float,        # softmax-weighted probability mass
        "members": [(idx, raw_score), ...],
        "best_idx": int,      # highest-scored member
    }
    """
    n = len(cluster_ids)
    if n == 0:
        return {}

    # Z-normalise, then softmax
    std = scores.std()
    z = (scores - scores.mean()) / (std + 1e-6)
    w = np.exp((z - z.max()) / temperature)
    w = w / w.sum()

    clusters: Dict[int, dict] = {}
    for k in range(n):
        c_id = cluster_ids[k]
        if c_id not in clusters:
            clusters[c_id] = {"mass": 0.0, "members": [], "best_idx": k}
        
        clusters[c_id]["mass"] += float(w[k])
        clusters[c_id]["members"].append((k, float(scores[k])))
        if scores[k] > scores[clusters[c_id]["best_idx"]]:
            clusters[c_id]["best_idx"] = k

    return clusters
def _compute_borda_clusters(
    cluster_ids: List[int],
    scores: np.ndarray,
    p: float = 1.0,
) -> Dict[int, dict]:
    """Compute Borda clusters and normalize them into a pseudo-probability mass for SPRT.

    Returns dict mapping cluster_id -> {
        "mass": float,        # pseudo-probability mass (votes / total_votes)
        "members": [(idx, raw_score), ...],
        "best_idx": int,      # highest-scored member
    }
    """
    n = len(cluster_ids)
    if n == 0:
        return {}

    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank
        
    clusters: Dict[int, dict] = {}
    total_votes = 0.0
    
    for k in range(n):
        c_id = cluster_ids[k]
        if c_id not in clusters:
            clusters[c_id] = {"votes": 0.0, "members": [], "best_idx": k}
            
        vote = (n - ranks[k] + 1) ** p
        clusters[c_id]["votes"] += vote
        total_votes += vote
        clusters[c_id]["members"].append((k, float(scores[k])))
        if scores[k] > scores[clusters[c_id]["best_idx"]]:
            clusters[c_id]["best_idx"] = k

    # Normalize to probabilities for SPRT
    for c_id in clusters:
        clusters[c_id]["mass"] = clusters[c_id]["votes"] / (total_votes + 1e-9)

    return clusters

def _get_leader(clusters: Dict[int, dict]) -> tuple:
    """Return (leader_cluster_id, leader_mass, leader_best_idx)."""
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
    global_cluster_ids: List[int],
    alpha: float,
    beta: float,
    burnin: int,
    p0_mode: str,
    use_ensemble_weighting: bool,
    temperature: float,
    reset_strategy: str = "immediate",
    agg_mode: str = "rwsc",
    p_borda: float = 1.0,
) -> dict:
    """Run SPRT on a sequence of candidates.

    Args:
        candidates: All N candidates in sequential order.
        raw_scores: Shape (N,) — single RM or ensemble mean scores.
        variances: Shape (N,) — ensemble variance per candidate (None for vanilla).
        global_cluster_ids: List of semantic cluster labels per candidate.
        alpha: Type-I error rate (false stop probability).
        beta: Type-II error rate (false continue probability).
        burnin: Number of initial candidates before starting SPRT.
        p0_mode: How to set p₀ — "uniform" or "fixed".
        use_ensemble_weighting: If True, weight evidence by 1/(σ² + ε).
        temperature: RWSC softmax temperature.
        reset_strategy: "immediate", "adaptive", "never", or "parallel".
        agg_mode: "rwsc" or "borda".
        p_borda: Borda weight power for agg_mode="borda".

    Returns:
        dict with keys: stopped_at, leader_answer, best_idx, log_lambda_trace
    """
    N = len(candidates)
    burnin = max(0, min(burnin, N - 1))

    # Stopping boundaries (Wald)
    B = math.log((1.0 - beta) / alpha) # Upper boundary
    A = math.log(beta / (1.0 - alpha)) # Lower boundary
    
    # Phase 1: Burn-in — build initial clusters (only if burnin > 0)
    if burnin > 0:
        burn_scores = raw_scores[:burnin]
        if agg_mode == "borda":
            clusters = _compute_borda_clusters(
                global_cluster_ids[:burnin], burn_scores, p_borda
            )
        else:
            clusters = _compute_rwsc_clusters(
                global_cluster_ids[:burnin], burn_scores, temperature
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
    else:
        leader_ans = None

    # Phase 2: Sequential SPRT
    log_lambda = 0.0
    trace = []
    
    if reset_strategy == "parallel":
        active_hypotheses = {}
        if burnin > 0 and leader_ans is not None:
            active_hypotheses[leader_ans] = {
                "log_lambda": 0.0,
                "log_ratio_agree": log_ratio_agree,
                "log_ratio_disagree": log_ratio_disagree,
            }
        rejected_hypotheses = set()

    cumulative_w = 0.0

    for i in range(burnin, N):
        # Observe candidate i
        ans_i = global_cluster_ids[i]

        # Evidence weight
        if use_ensemble_weighting and variances is not None:
            w_i = 1.0 / (variances[i] + 1e-4)
            w_i = min(w_i, 10.0)
        else:
            w_i = 1.0

        # Update clusters with new candidate for leader tracking / base rates
        if agg_mode == "borda":
            clusters_current = _compute_borda_clusters(
                global_cluster_ids[:i + 1], raw_scores[:i + 1], p_borda
            )
        else:
            clusters_current = _compute_rwsc_clusters(
                global_cluster_ids[:i + 1], raw_scores[:i + 1], temperature
            )
            
        new_leader_ans, new_leader_mass, _ = _get_leader(clusters_current)

        if reset_strategy == "parallel":
            # CoSol strategy: Test all active hypotheses against the NEW observation FIRST
            to_remove = []
            accepted_h = None
            for h_ans, h_data in active_hypotheses.items():
                x_h = 1.0 if ans_i == h_ans else 0.0
                delta = w_i * (x_h * h_data["log_ratio_agree"] + (1.0 - x_h) * h_data["log_ratio_disagree"])
                h_data["log_lambda"] += delta
                
                if h_data["log_lambda"] >= B:
                    accepted_h = h_ans
                    break
                if h_data["log_lambda"] <= A:
                    to_remove.append(h_ans)
                    
            if accepted_h is not None:
                idx_for_h = global_cluster_ids[:i+1].index(accepted_h)
                return {
                    "stopped_at": i + 1,
                    "leader_answer": accepted_h,
                    "best_idx": idx_for_h,
                    "log_lambda_trace": trace,
                    "decision": "accept",
                }
                
            for h in to_remove:
                del active_hypotheses[h]
                rejected_hypotheses.add(h)

            # Lazy Instantiation: If we just saw a brand new answer, instantiate it NOW so it can be tested on FUTURE data
            if ans_i not in active_hypotheses and ans_i not in rejected_hypotheses:
                ans_i_mass = clusters_current[ans_i]["mass"]
                p1_new = max(ans_i_mass, 0.55)
                n_unique_now = max(len(clusters_current), 2)
                p0_new = 1.0 / n_unique_now if p0_mode == "uniform" else 0.3
                p0_new = np.clip(p0_new, 0.01, 0.49)
                p1_new = max(p1_new, p0_new + 0.1)
                p1_new = np.clip(p1_new, 0.51, 0.99)
                
                log_ratio_disagree = math.log((1.0 - p1_new) / (1.0 - p0_new))
                retroactive_penalty = cumulative_w * log_ratio_disagree
                
                active_hypotheses[ans_i] = {
                    "log_lambda": retroactive_penalty,
                    "log_ratio_agree": math.log(p1_new / p0_new),
                    "log_ratio_disagree": log_ratio_disagree
                }
            
            # Trace tracking for parallel mode
            if active_hypotheses:
                current_max_log_lambda = max(h["log_lambda"] for h in active_hypotheses.values())
                trace.append(current_max_log_lambda)
            else:
                trace.append(0.0)
            leader_ans = new_leader_ans
        else:
            if new_leader_ans != leader_ans:
                if reset_strategy == "never":
                    # Fallback to old behavior if never resetting, just track normally
                    x_i = 1.0 if ans_i == leader_ans else 0.0
                    delta = w_i * (x_i * log_ratio_agree + (1.0 - x_i) * log_ratio_disagree)
                    log_lambda += delta
                    trace.append(log_lambda)
                else:
                    if reset_strategy == "adaptive":
                        # Calculate expected step increments
                        mu1 = p1 * log_ratio_agree + (1.0 - p1) * log_ratio_disagree
                        mu0 = -(p0 * log_ratio_agree + (1.0 - p0) * log_ratio_disagree)
                        
                        # Compute theoretical probability threshold gamma*
                        num = A * mu1 - log_lambda * mu0
                        den = 2 * A * mu1 - log_lambda * (mu0 + mu1)
                        gamma_star = num / den if den != 0 else 1.0
                        
                        # If empirical mass of new leader is below threshold, it's likely noise
                        if new_leader_mass <= gamma_star:
                            x_i = 1.0 if ans_i == leader_ans else 0.0
                            delta = w_i * (x_i * log_ratio_agree + (1.0 - x_i) * log_ratio_disagree)
                            log_lambda += delta
                            trace.append(log_lambda)
                            # (Omitted A/B checks here as adaptive mode usually defers stopping when threshold is missed)
                            continue
                    
                    leader_ans = new_leader_ans
                    # Mathematically, the Alternative Hypothesis H1 has changed.
                    # We MUST reset the accumulated log-likelihood to preserve SPRT validity.
                    log_lambda = 0.0
                    
                    p1 = max(new_leader_mass, 0.55)
                    n_unique_now = max(len(clusters_current), 2)
                    if p0_mode == "uniform":
                        p0 = 1.0 / n_unique_now
                    p0 = np.clip(p0, 0.01, 0.49)
                    p1 = max(p1, p0 + 0.1)
                    p1 = np.clip(p1, 0.51, 0.99)
                    log_ratio_agree = math.log(p1 / p0)
                    log_ratio_disagree = math.log((1.0 - p1) / (1.0 - p0))
                    trace.append(log_lambda)
            else:
                # Accumulate log-likelihood ratio for standard modes
                x_i = 1.0 if ans_i == leader_ans else 0.0
                delta = w_i * (x_i * log_ratio_agree + (1.0 - x_i) * log_ratio_disagree)
                log_lambda += delta
                trace.append(log_lambda)
                
                # Check stopping conditions
                if log_lambda >= B:
                    # Accept: leader is statistically dominant
                    _, _, best_idx = _get_leader(clusters_current)
                    return {
                        "stopped_at": i + 1,
                        "leader_answer": new_leader_ans,
                        "best_idx": best_idx,
                        "log_lambda_trace": trace,
                        "decision": "accept",
                    }
        
                if log_lambda <= A:
                    # Reset: current leader is likely wrong
                    log_lambda = 0.0
                    # Note: we explicitly don't shift leader here since the leader hasn't changed.
                    # We just restart the SPRT process to evaluate the SAME leader under fresh evidence.
                    p1 = max(new_leader_mass, 0.55)
                    n_unique_now = max(len(clusters_current), 2)
                    if p0_mode == "uniform":
                        p0 = 1.0 / n_unique_now
                    p0 = np.clip(p0, 0.01, 0.49)
                    p1 = max(p1, p0 + 0.1)
                    p1 = np.clip(p1, 0.51, 0.99)
                    log_ratio_agree = math.log(p1 / p0)
                    log_ratio_disagree = math.log((1.0 - p1) / (1.0 - p0))

        # Accumulate weight for retroactive penalties
        cumulative_w += w_i

    # Fallback: never stopped — use full pool RWSC
    if agg_mode == "borda":
        final_clusters = _compute_borda_clusters(global_cluster_ids, raw_scores, p_borda)
    else:
        final_clusters = _compute_rwsc_clusters(global_cluster_ids, raw_scores, temperature)
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
    "ArmoRM",
    "Skywork",
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



