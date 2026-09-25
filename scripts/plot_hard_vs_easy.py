import json
import os
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from src.methods.sequential.e_process import _e_borda_stop
from src.pipelines.base import _extract_answer

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

# Metrics storage
groups = {
    "Exhausted (Budget=512)": {"bon": 0, "sc": 0, "total": 0},
    "Stopped Early (Budget<512)": {"bon": 0, "sc": 0, "total": 0}
}

print("Evaluating questions...")
for idx in sorted(cands_data.keys()):
    if idx not in scores_data: continue
    
    ex = cands_data[idx]
    candidates = ex["candidates"]
    raw_scores = scores_data[idx]
    gt_ans = str(_extract_answer(ex["ground_truth"], dataset)).strip()
    
    ans_list = [str(_extract_answer(c, dataset)).strip() for c in candidates]
    unique_ans = list(set(ans_list))
    ans_to_cid = {a: i for i, a in enumerate(unique_ans)}
    global_cluster_ids = [ans_to_cid[a] for a in ans_list]
    
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
    
    # Vanilla BoN
    best_bon_idx = int(np.argmax(raw_scores))
    if ans_list[best_bon_idx] == gt_ans:
        groups[group_key]["bon"] += 1
        
    # Self Consistency
    counts = Counter(global_cluster_ids)
    best_sc_cid = counts.most_common(1)[0][0]
    if unique_ans[best_sc_cid] == gt_ans:
        groups[group_key]["sc"] += 1

# Calculate percentages
labels = list(groups.keys())
bon_accs = [groups[l]["bon"] / groups[l]["total"] * 100 for l in labels]
sc_accs = [groups[l]["sc"] / groups[l]["total"] * 100 for l in labels]

print("\nResults:")
for l in labels:
    print(f"{l} (N={groups[l]['total']}): BoN={bon_accs[labels.index(l)]:.1f}%, SC={sc_accs[labels.index(l)]:.1f}%")

# Plotting
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, bon_accs, width, label='Vanilla BoN', color='indianred')
rects2 = ax.bar(x + width/2, sc_accs, width, label='Majority Voting (SC)', color='steelblue')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy on Confident vs. Highly Uncertain Questions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={groups[l]['total']})" for l in labels], fontsize=11)
ax.legend()

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/hard_vs_easy_accuracy.png", dpi=300)
print("\nSaved plot to figures/hard_vs_easy_accuracy.png")
