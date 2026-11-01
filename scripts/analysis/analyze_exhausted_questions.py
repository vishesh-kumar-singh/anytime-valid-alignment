import json
import os
import sys
import numpy as np
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.methods.sequential.e_process import _e_borda_stop
from src.pipelines.base import _extract_answer
from math_verify import parse, verify

dataset = "math500"
candidates_path = "data/candidates/qwen2.5-3b-instruct/math500_n512_full_vllm.jsonl"
scores_path = "data/scores/qwen2.5-3b-instruct/math500_n512_full_vllm_scores_ArmoRM-Llama3-8B-v0.1.jsonl"

cands_data = {}
with open(candidates_path, "r") as f:
    for line in f:
        d = json.loads(line)
        cands_data[d["example_index"]] = d

scores_data = {}
with open(scores_path, "r") as f:
    for line in f:
        d = json.loads(line)
        scores_data[d["example_index"]] = d["scores"]

def fast_semantic_clustering(candidates, dataset):
    ans_list = [_extract_answer(c, dataset) for c in candidates]
    unique_ans = list(set(ans_list))
    
    # Pre-parse all unique answers
    parsed_ans = {}
    for a in unique_ans:
        try:
            parsed_ans[a] = parse(a)
        except:
            parsed_ans[a] = None

    # Group into clusters
    clusters = []
    ans_to_cid = {}
    
    for a in unique_ans:
        found_cluster = -1
        pa = parsed_ans[a]
        for cid, rep_ans in enumerate(clusters):
            prep = parsed_ans[rep_ans]
            if pa is not None and prep is not None:
                try:
                    if verify(pa, prep):
                        found_cluster = cid
                        break
                except:
                    pass
            else:
                if a == rep_ans:
                    found_cluster = cid
                    break
                    
        if found_cluster == -1:
            clusters.append(a)
            ans_to_cid[a] = len(clusters) - 1
        else:
            ans_to_cid[a] = found_cluster
            
    return [ans_to_cid[a] for a in ans_list]

exhausted_indices = []
vanilla_bon_correct = 0
self_cons_correct = 0

print("Finding exhausted questions (E-process reaching 512 budget)...")
for idx in sorted(cands_data.keys()):
    if idx not in scores_data: continue
    
    ex = cands_data[idx]
    candidates = ex["candidates"]
    raw_scores = scores_data[idx]
    gt = ex["ground_truth"]
    
    global_cluster_ids = fast_semantic_clustering(candidates, dataset)
    
    res = _e_borda_stop(
        candidates=candidates,
        raw_scores=np.array(raw_scores),
        variances=None,
        global_cluster_ids=global_cluster_ids,
        alpha=0.1,
        p_dagger=0.5,
        m=4.0,
        c=0.8,
        burnin=4,
        use_ensemble_weighting=False,
    )
    
    if res["stopped_at"] == 512:
        exhausted_indices.append(idx)
        
        # Vanilla BoN
        best_bon_idx = int(np.argmax(raw_scores))
        bon_ans = _extract_answer(candidates[best_bon_idx], dataset)
        bon_corr = False
        try:
            bon_corr = verify(parse(bon_ans), parse(gt))
        except: pass
        if bon_corr: vanilla_bon_correct += 1
            
        # Self Consistency
        counts = Counter(global_cluster_ids)
        best_sc_cid = counts.most_common(1)[0][0]
        best_sc_idx = global_cluster_ids.index(best_sc_cid)
        sc_ans = _extract_answer(candidates[best_sc_idx], dataset)
        sc_corr = False
        try:
            sc_corr = verify(parse(sc_ans), parse(gt))
        except: pass
        if sc_corr: self_cons_correct += 1

n_exhausted = len(exhausted_indices)
print(f"\nFound {n_exhausted} questions that hit the 512 limit.")
if n_exhausted > 0:
    print(f"Vanilla BoN Accuracy:      {vanilla_bon_correct / n_exhausted:.1%}  ({vanilla_bon_correct}/{n_exhausted})")
    print(f"Self-Consistency Accuracy: {self_cons_correct / n_exhausted:.1%}  ({self_cons_correct}/{n_exhausted})")
    print(f"Exhausted Question Indices (First 20): {exhausted_indices[:20]}")
