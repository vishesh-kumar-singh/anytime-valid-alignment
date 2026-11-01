import json
import os
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

groups = {
    "Exhausted (Budget=512)": {"bon": 0, "sc": 0, "borda": 0, "rwsc": 0, "total": 0},
    "Stopped Early (Budget<512)": {"bon": 0, "sc": 0, "borda": 0, "rwsc": 0, "total": 0}
}

stopping_steps = []

print("Evaluating questions using math_verify (this may take a minute or two)...")
processed = 0

for idx in sorted(cands_data.keys()):
    if idx not in scores_data: continue
    
    ex = cands_data[idx]
    candidates = ex["candidates"]
    raw_scores = scores_data[idx]
    gt_ans = _extract_answer(ex["ground_truth"], dataset)
    
    global_cluster_ids, cluster_reps = fast_semantic_clustering(candidates, dataset)
    
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
    
    group_key = "Exhausted (Budget=512)" if res["stopped_at"] == 512 else "Stopped Early (Budget<512)"
    groups[group_key]["total"] += 1
    stopping_steps.append(res["stopped_at"])
    
    # 1. Vanilla BoN
    best_bon_idx = int(np.argmax(raw_scores))
    bon_ans = _extract_answer(candidates[best_bon_idx], dataset)
    bon_corr = False
    try:
        if bon_ans and gt_ans:
            bon_corr = verify(parse(bon_ans), parse(gt_ans))
    except: pass
    if bon_corr:
        groups[group_key]["bon"] += 1
        
    # 2. Self Consistency
    counts = Counter(global_cluster_ids)
    best_sc_cid = counts.most_common(1)[0][0]
    sc_ans = cluster_reps[best_sc_cid]
    sc_corr = False
    try:
        if sc_ans and gt_ans:
            sc_corr = verify(parse(sc_ans), parse(gt_ans))
    except: pass
    if sc_corr:
        groups[group_key]["sc"] += 1
        
    # 3. Borda
    N_cands = len(candidates)
    borda_scores = {cid: 0.0 for cid in set(global_cluster_ids)}
    sorted_indices = np.argsort(raw_scores)[::-1]
    for rank, i in enumerate(sorted_indices):
        cid = global_cluster_ids[i]
        borda_scores[cid] += (N_cands - 1 - rank)
    best_borda_cid = max(borda_scores.items(), key=lambda x: x[1])[0]
    borda_ans = cluster_reps[best_borda_cid]
    borda_corr = False
    try:
        if borda_ans and gt_ans:
            borda_corr = verify(parse(borda_ans), parse(gt_ans))
    except: pass
    if borda_corr:
        groups[group_key]["borda"] += 1
        
    # 4. RWSC
    rwsc_scores = {cid: 0.0 for cid in set(global_cluster_ids)}
    for cid, r in zip(global_cluster_ids, raw_scores):
        rwsc_scores[cid] += np.exp(r)
    best_rwsc_cid = max(rwsc_scores.items(), key=lambda x: x[1])[0]
    rwsc_ans = cluster_reps[best_rwsc_cid]
    rwsc_corr = False
    try:
        if rwsc_ans and gt_ans:
            rwsc_corr = verify(parse(rwsc_ans), parse(gt_ans))
    except: pass
    if rwsc_corr:
        groups[group_key]["rwsc"] += 1
        
    processed += 1
    if processed % 50 == 0:
        print(f"Processed {processed} / 500 questions...")

labels = list(groups.keys())
bon_accs = [groups[l]["bon"] / groups[l]["total"] * 100 for l in labels]
sc_accs = [groups[l]["sc"] / groups[l]["total"] * 100 for l in labels]
rwsc_accs = [groups[l]["rwsc"] / groups[l]["total"] * 100 for l in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, bon_accs, width, label='Vanilla BoN', color='indianred')
rects2 = ax.bar(x, sc_accs, width, label='Majority Voting (SC)', color='lightsteelblue')
rects4 = ax.bar(x + width, rwsc_accs, width, label='RWSC', color='darkblue')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy on Confident vs. Highly Uncertain Questions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={groups[l]['total']})" for l in labels], fontsize=11)
ax.legend(loc='upper left')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects4)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/hard_vs_easy_accuracy.png", dpi=300)
print("\nSaved accurate plot to figures/hard_vs_easy_accuracy.png")


