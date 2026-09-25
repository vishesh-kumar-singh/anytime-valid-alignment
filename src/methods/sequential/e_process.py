import math
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from .sprt import _compute_rwsc_clusters, _get_leader

def _e_rwsc_stop(
    candidates: List[str],
    raw_scores: np.ndarray,
    variances: Optional[np.ndarray],
    global_cluster_ids: List[int],
    alpha: float,
    p_dagger: float = 0.4,
    c: float = 0.8,
    m: float = 4.0,
    use_ensemble_weighting: bool = False,
    temperature: float = 1.0,
    burnin: int = 2,
) -> dict:
    """Run e-RWSC anytime-valid sequential stopping (Algorithm 1).
    
    Args:
        candidates: All N candidates in sequential order.
        raw_scores: Shape (N,) — single RM or ensemble mean scores.
        variances: Shape (N,) — ensemble variance per candidate (None for vanilla).
        global_cluster_ids: List of semantic cluster labels per candidate.
        alpha: Type-I error rate (certification level).
        p_dagger: Dominance threshold (default 0.5 for strict majority).
        c: Clip parameter for Kelly bets.
        m: Prior pseudo-observation strength.
        use_ensemble_weighting: If True, uses variance-gated observations (e-RWSC-V).
        temperature: RWSC softmax temperature.
        
    Returns:
        dict with keys: stopped_at, leader_answer, best_idx, log_lambda_trace, decision
    """
    N = len(candidates)
    
    active_answers = set()
    J_count = 0
    J = {}
    pi = {}
    sigma_a = {}
    E = {}
    S = {}
    
    trace = []
    
    for i in range(N):
        # 1. Update running median of variances for gating
        if use_ensemble_weighting and variances is not None:
            if i == 0:
                sigma2_0 = 1.0
            else:
                sigma2_0 = float(np.median(variances[:i]))
            sigma2_i = variances[i]
            # Add epsilon to numerator to handle sigma2_0 == 0 and sigma2_i == 0 (perfect agreement)
            h_i = (sigma2_0 + 1e-6) / (sigma2_0 + sigma2_i + 1e-6)
        else:
            h_i = 1.0
            
        # 2. Recompute RWSC masses over y_1:n-1 (which is index 0 to i-1)
        if i > 0:
            clusters_prev = _compute_rwsc_clusters(
                global_cluster_ids[:i], raw_scores[:i], temperature
            )
        else:
            clusters_prev = {}
            
        # Compute Kelly bets for all active answers
        lambdas = {}
        for a in active_answers:
            s_hat = clusters_prev.get(a, {}).get("mass", 0.0)
            n_trials = i - sigma_a[a]
            theta_hat = (S[a] + m * s_hat) / (n_trials + m)
            
            # Kelly bet
            kelly = (theta_hat - p_dagger) / (p_dagger * (1.0 - p_dagger))
            
            # Clip
            lambdas[a] = np.clip(kelly, 0.0, c / p_dagger)
            
        # 3. Settle active bets against the new observation
        ans_i = global_cluster_ids[i]
        for a in active_answers:
            X = 1.0 if ans_i == a else 0.0
            Z = X * h_i
            E[a] *= (1.0 + lambdas[a] * (Z - p_dagger))
            S[a] += Z
            
        # 4. Lazy instantiation of new answer
        if ans_i not in active_answers:
            active_answers.add(ans_i)
            J_count += 1
            J[ans_i] = J_count
            pi[ans_i] = 1.0 / (J_count * (J_count + 1.0))
            sigma_a[ans_i] = i
            E[ans_i] = 1.0
            S[ans_i] = 0.0
            
        # 5. Check stopping condition
        certified_a = None
        max_weighted_E = -1.0
        
        if (i + 1) >= burnin:
            for a in active_answers:
                weighted_E = pi[a] * E[a]
                if weighted_E >= 1.0 / alpha:
                    if weighted_E > max_weighted_E:
                        max_weighted_E = weighted_E
                        certified_a = a
                    
        # Track max E for logging (using log scale to match standard SPRT trace)
        if active_answers:
            current_max_E = max(pi[a] * E[a] for a in active_answers)
            trace.append(math.log(current_max_E) if current_max_E > 0 else 0.0)
        else:
            trace.append(0.0)
            
        if certified_a is not None:
            clusters_current = _compute_rwsc_clusters(
                global_cluster_ids[:i+1], raw_scores[:i+1], temperature
            )
            best_idx = clusters_current[certified_a]["best_idx"]
            return {
                "stopped_at": i + 1,
                "leader_answer": certified_a,
                "best_idx": best_idx,
                "log_lambda_trace": trace,
                "decision": "accept",
            }
            
    # Budget exhausted: uncertified fallback
    clusters_final = _compute_rwsc_clusters(global_cluster_ids, raw_scores, temperature)
    leader_ans, _, best_idx = _get_leader(clusters_final)
    return {
        "stopped_at": N,
        "leader_answer": leader_ans,
        "best_idx": best_idx,
        "log_lambda_trace": trace,
        "decision": "fallback",
    }

from .sprt import _compute_borda_clusters

def _e_borda_stop(
    candidates: List[str],
    raw_scores: np.ndarray,
    variances: Optional[np.ndarray],
    global_cluster_ids: List[int],
    alpha: float,
    p_dagger: float = 0.4,
    c: float = 0.8,
    m: float = 4.0,
    use_ensemble_weighting: bool = False,
    k_penalty: float = 2.0,
    burnin: int = 2,
) -> dict:
    """Run e-Borda anytime-valid sequential stopping.
    """
    N = len(candidates)
    
    active_answers = set()
    J_count = 0
    J = {}
    pi = {}
    sigma_a = {}
    E = {}
    S = {}
    
    trace = []
    
    for i in range(N):
        if use_ensemble_weighting and variances is not None:
            if i == 0:
                sigma2_0 = 1.0
            else:
                sigma2_0 = float(np.median(variances[:i]))
            sigma2_i = variances[i]
            h_i = (sigma2_0 + 1e-6) / (sigma2_0 + sigma2_i + 1e-6)
        else:
            h_i = 1.0
            
        if i > 0:
            clusters_prev = _compute_borda_clusters(
                global_cluster_ids[:i], raw_scores[:i], k_penalty
            )
        else:
            clusters_prev = {}
            
        lambdas = {}
        for a in active_answers:
            s_hat = clusters_prev.get(a, {}).get("mass", 0.0)
            n_trials = i - sigma_a[a]
            theta_hat = (S[a] + m * s_hat) / (n_trials + m)
            kelly = (theta_hat - p_dagger) / (p_dagger * (1.0 - p_dagger))
            lambdas[a] = np.clip(kelly, 0.0, c / p_dagger)
            
        ans_i = global_cluster_ids[i]
        for a in active_answers:
            X = 1.0 if ans_i == a else 0.0
            Z = X * h_i
            E[a] *= (1.0 + lambdas[a] * (Z - p_dagger))
            S[a] += Z
            
        if ans_i not in active_answers:
            active_answers.add(ans_i)
            J_count += 1
            J[ans_i] = J_count
            pi[ans_i] = 1.0 / (J_count * (J_count + 1.0))
            sigma_a[ans_i] = i
            E[ans_i] = 1.0
            S[ans_i] = 0.0
            
        certified_a = None
        max_weighted_E = -1.0
        
        if (i + 1) >= burnin:
            for a in active_answers:
                weighted_E = pi[a] * E[a]
                if weighted_E >= 1.0 / alpha:
                    if weighted_E > max_weighted_E:
                        max_weighted_E = weighted_E
                        certified_a = a
                    
        if active_answers:
            current_max_E = max(pi[a] * E[a] for a in active_answers)
            trace.append(math.log(current_max_E) if current_max_E > 0 else 0.0)
        else:
            trace.append(0.0)
            
        if certified_a is not None:
            clusters_current = _compute_borda_clusters(
                global_cluster_ids[:i+1], raw_scores[:i+1], k_penalty
            )
            best_idx = clusters_current[certified_a]["best_idx"]
            return {
                "stopped_at": i + 1,
                "leader_answer": certified_a,
                "best_idx": best_idx,
                "log_lambda_trace": trace,
                "decision": "accept",
            }
            
    clusters_final = _compute_borda_clusters(global_cluster_ids, raw_scores, k_penalty)
    leader_ans, _, best_idx = _get_leader(clusters_final)
    return {
        "stopped_at": N,
        "leader_answer": leader_ans,
        "best_idx": best_idx,
        "log_lambda_trace": trace,
        "decision": "fallback",
    }


def _cite_stop(
    candidates: List[str],
    raw_scores: Optional[np.ndarray],
    variances: Optional[np.ndarray],
    global_cluster_ids: List[int],
    alpha: float,
    burnin: int = 4,
    use_wcite: bool = False,
) -> dict:
    """Run CITE (and W-CITE) anytime-valid sequential stopping (Ota et al. 2026)."""
    N = len(candidates)
    
    # Oracle target r
    counts = defaultdict(float)
    for i, cid in enumerate(global_cluster_ids):
        # Weight by raw_scores if W-CITE, else 1.0
        w = np.clip(float(raw_scores[i]), 0.0, 1.0) if use_wcite and raw_scores is not None else 1.0
        counts[cid] += w
    target_r = max(counts, key=counts.get)
    
    alpha_pw = alpha / 3.0
    alpha_r = alpha / 3.0
    alpha_u = alpha / 3.0
    
    lambda_pw = [2.0**(-k) for k in range(1, 11)]
    w_lambda = 1.0 / len(lambda_pw)
    
    lambda_r = [2.0**(-k) for k in range(1, 11)]
    v_lambda = 1.0 / len(lambda_r)
    
    q_grid = np.linspace(0.01, 0.99, 99)
    
    # Precompute Unseen Upper Bound U_t
    U_t_arr = np.ones(N + 1)
    u_grid = np.linspace(1e-6, 1.0, 1000)
    for t_idx in range(1, N + 1):
        # u^{-1} (1-u)^t <= alpha_u
        valid_u = u_grid[(1.0 / u_grid) * ((1.0 - u_grid)**t_idx) <= alpha_u]
        if len(valid_u) > 0:
            U_t_arr[t_idx] = valid_u[0]
            
    E_pw_l = defaultdict(lambda: {l: 1.0 for l in lambda_pw})
    M_q_l = {q: {l: 1.0 for l in lambda_r if l < 1.0/q} for q in q_grid}
    
    active_answers = set()
    N_t = defaultdict(int)
    trace = []
    
    for t in range(1, N + 1):
        ans_t = global_cluster_ids[t - 1]
        active_answers.add(ans_t)
        N_t[ans_t] += 1
        
        W_t = 1.0
        if use_wcite and raw_scores is not None:
            W_t = np.clip(float(raw_scores[t - 1]), 0.0, 1.0)
            
        # Update pairwise e-processes for all a != target_r
        for a in active_answers:
            if a == target_r:
                continue
            Z = W_t * (1.0 if ans_t == target_r else 0.0) - W_t * (1.0 if ans_t == a else 0.0)
            for l in lambda_pw:
                E_pw_l[a][l] *= (1.0 + l * Z)
                
        # Update LCB e-processes
        for q in q_grid:
            Z_L = W_t * (1.0 if ans_t == target_r else 0.0) - q
            for l in M_q_l[q]:
                M_q_l[q][l] *= (1.0 + l * Z_L)
                
        # Check conditions
        if t >= burnin:
            # Pairwise check
            is_pw_passed = True
            min_pw_E = float('inf')
            
            if len(active_answers - {target_r}) == 0:
                is_pw_passed = True
            else:
                if use_wcite:
                    # W-CITE: check ALL competitors
                    for a in active_answers - {target_r}:
                        E_a = sum(w_lambda * E_pw_l[a][l] for l in lambda_pw)
                        min_pw_E = min(min_pw_E, E_a)
                        if E_a < 1.0 / alpha_pw:
                            is_pw_passed = False
                            break
                else:
                    # CITE: check ONLY empirical runner-up
                    a_t = max(active_answers - {target_r}, key=lambda a: N_t[a])
                    E_a = sum(w_lambda * E_pw_l[a_t][l] for l in lambda_pw)
                    min_pw_E = E_a
                    if E_a < 1.0 / alpha_pw:
                        is_pw_passed = False
                        
            # LCB check
            L_t = 0.0
            for q in q_grid:
                M_q = sum(v_lambda * M_q_l[q][l] for l in M_q_l[q])
                if M_q >= 1.0 / alpha_r:
                    L_t = max(L_t, q)
                    
            U_t = U_t_arr[t]
            
            trace.append(math.log(min_pw_E) if min_pw_E != float('inf') and min_pw_E > 0 else 0.0)
            
            if is_pw_passed and L_t > U_t:
                best_idx = t - 1
                for i in range(t):
                    if global_cluster_ids[i] == target_r:
                        best_idx = i
                        break
                return {
                    "stopped_at": t,
                    "leader_answer": target_r,
                    "best_idx": best_idx,
                    "log_lambda_trace": trace,
                    "decision": "accept"
                }
        else:
            trace.append(0.0)
            
    # Fallback
    best_idx = N - 1
    for i in range(N):
        if global_cluster_ids[i] == target_r:
            best_idx = i
            break
    return {
        "stopped_at": N,
        "leader_answer": target_r,
        "best_idx": best_idx,
        "log_lambda_trace": trace,
        "decision": "fallback"
    }
