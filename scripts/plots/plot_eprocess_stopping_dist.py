import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.pipelines.base import _extract_answer
from math_verify import parse, verify
from src.methods.sequential.e_process import _e_rwsc_stop

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

precomputed_clusters = {}
for idx in tqdm(idx_list, desc="Precomputing N=512 Clusters"):
    if idx not in scores_data: continue
    ex = cands_data[idx]
    candidates = ex["candidates"]
    global_cluster_ids, _ = fast_semantic_clustering(candidates, dataset)
    precomputed_clusters[idx] = global_cluster_ids

target_alpha = 0.1
p_dagger = 0.5
stopping_steps = []
exhausted = 0

for idx in tqdm(idx_list, desc="Evaluating E-Process"):
    if idx not in scores_data: continue
    global_cluster_ids = precomputed_clusters[idx]
    
    res = _e_rwsc_stop(
        candidates=cands_data[idx]["candidates"],
        raw_scores=np.array(scores_data[idx]),
        variances=None,
        global_cluster_ids=global_cluster_ids,
        alpha=target_alpha,
        p_dagger=p_dagger,
        c=0.8,
        m=4.0,
        burnin=4,
        use_ensemble_weighting=False,
        temperature=1.0
    )
    
    if res["decision"] == "accept":
        stopping_steps.append(res["stopped_at"])
    else:
        exhausted += 1
        stopping_steps.append(512)

print(f"Mean stopping step: {np.mean(stopping_steps):.2f}")
print(f"Exhausted total budget (512): {exhausted}")

# PLOTTING
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(5, 4))

# Plot histogram of stopping steps on log bins
bins = np.logspace(np.log10(4), np.log10(512), 20)
ax.hist(stopping_steps, bins=bins, color='steelblue', edgecolor='black', alpha=0.8)

ax.set_xscale('log')
from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
ax.tick_params(axis='both', labelsize=14)

ax.set_xlabel("Stopping Step ($N$)", fontsize=16)
ax.set_ylabel("Questions", fontsize=16)
ax.set_title(r"Stopping Dist ($\alpha=0.1$)", fontsize=18, fontweight='bold')

# Text box for mean and exhausted
textstr = f'Mean N: {np.mean(stopping_steps):.1f}\nExhausted: {exhausted}'
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/eprocess_stopping_distribution.png", dpi=300)
print("Saved plot to figures/eprocess_stopping_distribution.png")
