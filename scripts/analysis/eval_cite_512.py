import json
import os
import sys
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.pipelines.base import _extract_answer
from math_verify import parse, verify
from src.methods.sequential.e_process import _cite_stop

dataset = "math500"
candidates_path = "data/candidates/qwen2.5-3b-instruct/math500_n512_full_vllm.jsonl"
scores_path = "data/scores/qwen2.5-3b-instruct/math500_n512_full_vllm_scores_ArmoRM-Llama3-8B-v0.1.jsonl"

def fast_semantic_clustering(candidates, dataset):
    ans_list = [_extract_answer(c, dataset) for c in candidates]
    unique_ans = list(set(ans_list))
    
    parsed_ans = {}
    for a in unique_ans:
        try:
            parsed_ans[a] = parse(a)
        except:
            parsed_ans[a] = None

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
                except: pass
            else:
                if str(a).strip() == str(rep_ans).strip():
                    found_cluster = cid
                    break
                    
        if found_cluster == -1:
            clusters.append(a)
            ans_to_cid[a] = len(clusters) - 1
        else:
            ans_to_cid[a] = found_cluster
            
    return [ans_to_cid[a] for a in ans_list], clusters

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

idx_list = sorted(cands_data.keys())

print(f"Precomputing clusters for {len(idx_list)} examples...")
precomputed_clusters = {}
for idx in tqdm(idx_list, desc="Clustering"):
    if idx not in scores_data: continue
    ex = cands_data[idx]
    candidates = ex["candidates"]
    global_cluster_ids, _ = fast_semantic_clustering(candidates, dataset)
    precomputed_clusters[idx] = global_cluster_ids

budgets = [24, 32, 64, 128, 256, 512]
target_alpha = 0.1

print(f"\n--- Vanilla CITE Budget Sweep Results (Alpha={target_alpha}) ---")
latex_row = "CITE"

for budget in budgets:
    stopping_steps = []
    correct_count = 0
    
    for idx in idx_list:
        if idx not in scores_data: continue
        
        # Slice candidates and clusters up to the budget
        global_cluster_ids = precomputed_clusters[idx][:budget]
        candidates = cands_data[idx]["candidates"][:budget]
        
        res = _cite_stop(
            candidates=candidates,
            raw_scores=None,
            variances=None,
            global_cluster_ids=global_cluster_ids,
            alpha=target_alpha,
            burnin=4,
            use_wcite=False
        )
        
        if res["decision"] == "accept":
            stopping_steps.append(res["stopped_at"])
        else:
            stopping_steps.append(budget)
            
        c_ans = candidates[res["best_idx"]]
        gt = cands_data[idx]["ground_truth"]
        
        # check correctness
        from src.core.data import check_correctness
        is_correct = check_correctness(c_ans, gt, dataset)
                
        if is_correct:
            correct_count += 1
            
    acc = correct_count / len(idx_list) * 100
    mean_budget = np.mean(stopping_steps)
    std_budget = np.std(stopping_steps)
    
    # Format according to whether std is usually provided (the table seems to show std for CITE-like methods)
    if mean_budget == budget:
        latex_row += f" & {acc:.1f}\\% & {mean_budget:.1f}"
    else:
        latex_row += f" & {acc:.1f}\\% & {mean_budget:.1f}$\\pm${std_budget:.1f}"
        
    print(f"Budget {budget}: Acc {acc:.1f}%, Mean Budget {mean_budget:.1f} ± {std_budget:.1f}")

latex_row += " \\\\"
print("\nHere is your LaTeX row:")
print(latex_row)

