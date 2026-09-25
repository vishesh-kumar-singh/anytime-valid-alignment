import math
from typing import List, Tuple
import numpy as np

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def exact_expected_gain(r_star: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-6:
        return max(0.0, mu - r_star)
    z = (r_star - mu) / sigma
    return sigma * (norm_pdf(z) - z * (1.0 - norm_cdf(z)))

def compute_adabon(candidates: List[str], scores: np.ndarray, global_n_budget: int) -> Tuple[str, int]:
    N = len(candidates)
    n_budget = max(4, min(N, global_n_budget))
    best_idx = int(np.argmax(scores[:n_budget]))
    return candidates[best_idx], n_budget

def compute_beacon(candidates: List[str], scores: np.ndarray, gamma: float = 0.85) -> Tuple[str, int]:
    N = len(candidates)
    if N < 4:
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], N

    prior_mu = np.mean(scores[:4])
    prior_var = np.var(scores[:4]) + 1e-6
    obs_var = prior_var 
    
    theta_target = np.percentile(scores[:4], 75)
    
    for i in range(4, N + 1):
        current_scores = scores[:i]
        post_var = 1.0 / ((1.0 / prior_var) + (i / obs_var))
        post_mu = post_var * ((prior_mu / prior_var) + (np.sum(current_scores) / obs_var))
        
        prob_good = norm_cdf((post_mu - theta_target) / math.sqrt(post_var))
        
        if prob_good > gamma:
            best_idx = int(np.argmax(current_scores))
            return candidates[best_idx], i
            
    best_idx = int(np.argmax(scores))
    return candidates[best_idx], N

def compute_pandora(candidates: List[str], scores: np.ndarray, c: float = 0.05) -> Tuple[str, int]:
    N = len(candidates)
    if N < 4:
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], N

    for i in range(4, N):
        current_scores = scores[:i]
        r_star = np.max(current_scores)
        mu_hat = np.mean(current_scores)
        sigma_hat = np.std(current_scores)
        
        gain = exact_expected_gain(r_star, mu_hat, sigma_hat)
        
        if gain < c:
            best_idx = int(np.argmax(current_scores))
            return candidates[best_idx], i
            
    best_idx = int(np.argmax(scores))
    return candidates[best_idx], N

import sys
import os
consol_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'scratch', 'consol')
if consol_path not in sys.path:
    sys.path.append(consol_path)

try:
    from consol.confidence_models import SprtConfidenceModel
except ImportError:
    SprtConfidenceModel = None

def compute_consol(candidates: List[str], global_cluster_ids: List[int], p1: float = 0.5001, alpha: float = 0.05, beta: float = 0.05) -> Tuple[str, int]:
    N = len(candidates)
    if N < 2 or SprtConfidenceModel is None:
        if SprtConfidenceModel is None:
            print("Warning: consol not found in scratch/consol")
        best_idx = 0
        return candidates[best_idx], N

    # ConSol defaults to max_trials=256, but we only have N candidates
    # beta in their default is 1-.05-.000024 = 0.949976
    model = SprtConfidenceModel(p1=p1, alpha=alpha, beta=1-0.05-0.000024)
    
    for i in range(2, N + 1):
        current_ids = global_cluster_ids[:i]
        
        counts = {}
        for cid in current_ids:
            counts[cid] = counts.get(cid, 0) + 1
            
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        first_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        
        if model.test(first_count, second_count):
            best_cid = sorted_counts[0][0]
            best_idx = current_ids.index(best_cid)
            return candidates[best_idx], i
            
    # Fallback to most frequent
    counts = {}
    for cid in global_cluster_ids:
        counts[cid] = counts.get(cid, 0) + 1
    best_cid = max(counts.items(), key=lambda x: x[1])[0]
    best_idx = global_cluster_ids.index(best_cid)
    return candidates[best_idx], N
