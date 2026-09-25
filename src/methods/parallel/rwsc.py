from typing import List
import numpy as np

def compute_rwsc(scores: List[float], candidates: List[str], cluster_ids: List[int], temperature: float) -> str:
    ans_to_scores = {}
    ans_to_candidates = {}
    
    N = len(candidates)
    for i in range(N):
        c_id = cluster_ids[i]
            
        if c_id not in ans_to_scores:
            ans_to_scores[c_id] = []
            ans_to_candidates[c_id] = []
            
        ans_to_scores[c_id].append(scores[i])
        ans_to_candidates[c_id].append((i, candidates[i], scores[i]))
        
    best_c_id = None
    best_c_score = -float('inf')
    
    for c_id, s_list in ans_to_scores.items():
        # Apply robust z-scoring to prevent overflow in softmax
        raw = np.asarray(scores, dtype=float)
        z = (raw - raw.mean()) / (raw.std() + 1e-6)
        
        # We calculate the sum of softmax probabilities for the cluster
        w = np.exp((z - z.max()) / temperature)
        w = w / w.sum()
        
        ans_score = 0.0
        for cand_idx, _, _ in ans_to_candidates[c_id]:
            ans_score += float(w[cand_idx])
            
        if ans_score > best_c_score:
            best_c_score = ans_score
            best_c_id = c_id
            
    if best_c_id is not None:
        # Pick the candidate for the best answer that had the highest individual RM score
        best_c = max(ans_to_candidates[best_c_id], key=lambda x: x[2])[1]
        return best_c
    return candidates[0]

def compute_borda_voting(scores: List[float], candidates: List[str], cluster_ids: List[int], p: float) -> str:
    n = len(scores)
    if n == 0:
        return candidates[0] if candidates else ""
        
    # Rank candidates by score (highest score -> rank 1)
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank
        
    # Aggregate votes by cluster_id
    ans_to_votes = {}
    ans_to_candidates = {}
    
    for i in range(n):
        c_id = cluster_ids[i]
        vote = (n - ranks[i] + 1) ** p
        
        if c_id not in ans_to_votes:
            ans_to_votes[c_id] = 0.0
            ans_to_candidates[c_id] = []
            
        ans_to_votes[c_id] += vote
        ans_to_candidates[c_id].append((i, candidates[i], scores[i]))
        
    # Find the cluster with the most votes
    best_c_id = max(ans_to_votes.keys(), key=lambda c: ans_to_votes[c])
    
    # Pick the highest-scoring candidate from the winning cluster
    best_c = max(ans_to_candidates[best_c_id], key=lambda x: x[2])[1]
    return best_c
