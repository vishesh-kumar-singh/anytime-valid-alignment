import json
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.pipelines.base import _extract_answer
from math_verify import parse, verify
from src.methods.sequential.sprt import _sprt_stop, _compute_rwsc_clusters, _get_leader

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

def load_data():
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
            
    return cands_data, scores_data

def main():
    print("Loading datasets...")
    cands_data, scores_data = load_data()
    
    # We use the full MATH-500 dataset
    idx_list = sorted(cands_data.keys())
    
    print(f"Evaluating {len(idx_list)} questions using math_verify (this will take a couple minutes)...")
    precomputed_clusters = {}
    full_budget_ans = {}
    
    for idx in tqdm(idx_list, desc="Precomputing N=512 Clusters"):
        if idx not in scores_data: continue
        ex = cands_data[idx]
        candidates = ex["candidates"]
        raw_scores = np.array(scores_data[idx])
        global_cluster_ids, _ = fast_semantic_clustering(candidates, dataset)
        precomputed_clusters[idx] = global_cluster_ids
        
        # Calculate what the model WOULD have chosen with the full 512 budget
        clusters_final = _compute_rwsc_clusters(global_cluster_ids, raw_scores, temperature=1.0)
        leader_ans, _, best_idx = _get_leader(clusters_final)
        full_budget_ans[idx] = str(_extract_answer(candidates[best_idx], dataset)).strip()

    # Part 1: Realized Error vs Alpha
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    realized_errors = []
    
    print("\nSweeping nominal error rates (alpha)...")
    for alpha in tqdm(alphas, desc="Alphas"):
        errors = 0
        total = 0
        for idx in idx_list:
            if idx not in scores_data: continue
            ex = cands_data[idx]
            raw_scores = scores_data[idx]
            global_cluster_ids = precomputed_clusters[idx]
            
            res = _sprt_stop(
                candidates=ex["candidates"],
                raw_scores=np.array(raw_scores),
                variances=None,
                global_cluster_ids=global_cluster_ids,
                alpha=alpha,
                beta=0.1,
                burnin=4,
                p0_mode="uniform",
                use_ensemble_weighting=False,
                temperature=1.0,
                reset_strategy="immediate",
                agg_mode="rwsc"
            )
            
            if res["decision"] == "accept":
                c_ans = str(_extract_answer(ex["candidates"][res["best_idx"]], dataset)).strip()
                fb_ans = full_budget_ans[idx]
                if c_ans != fb_ans:
                    errors += 1
                total += 1
                
        if total > 0:
            realized_errors.append(errors / total)
        else:
            realized_errors.append(0)

    # Part 2: D_KL Scaling (Fix Alpha = 0.05)
    print("\nCalculating sample count vs 1/D_KL...")
    target_alpha = 0.05
    dkl_inv_list = []
    stopping_steps = []
    
    for idx in idx_list:
        if idx not in scores_data: continue
        ex = cands_data[idx]
        raw_scores = np.array(scores_data[idx])
        global_cluster_ids = precomputed_clusters[idx]
        
        res = _sprt_stop(
            candidates=ex["candidates"],
            raw_scores=raw_scores,
            variances=None,
            global_cluster_ids=global_cluster_ids,
            alpha=target_alpha,
            beta=0.1,
            burnin=4,
            p0_mode="uniform",
            use_ensemble_weighting=False,
            temperature=1.0,
            reset_strategy="immediate",
            agg_mode="rwsc"
        )
        
        if res["decision"] == "accept":
            # Estimate D_KL using the burnin parameters
            burn_scores = raw_scores[:4]
            clusters = _compute_rwsc_clusters(global_cluster_ids[:4], burn_scores, 1.0)
            if len(clusters) == 0:
                continue
            leader_ans, leader_mass, _ = _get_leader(clusters)
            
            p0 = np.clip(1.0 / max(len(clusters), 2), 0.01, 0.49)
            p1 = np.clip(max(max(leader_mass, 0.55), p0 + 0.1), 0.51, 0.99)
            
            # Bernoulli KL Divergence
            dkl = p1 * math.log(p1 / p0) + (1.0 - p1) * math.log((1.0 - p1) / (1.0 - p0))
            if dkl > 0:
                dkl_inv_list.append(1.0 / dkl)
                stopping_steps.append(res["stopped_at"])


    # PLOTTING
    print("\nGenerating Figures...")
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Realized Error
    axs[0].plot(alphas, realized_errors, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    axs[0].plot([0, 0.5], [0, 0.5], linestyle='--', color='gray', label='y = x (Perfect Calibration)')
    axs[0].set_xlabel("Nominal Error Bound ($\\alpha$)", fontsize=12)
    axs[0].set_ylabel("Realized Error Rate (Mismatch vs N=512)", fontsize=12)
    axs[0].set_title("Realized Error vs. Nominal Level", fontsize=14, fontweight='bold')
    axs[0].legend(fontsize=11)
    
    # Plot 2: 1/D_KL Scaling
    axs[1].scatter(dkl_inv_list, stopping_steps, alpha=0.5, color='mediumseagreen')
    
    # Linear trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(dkl_inv_list, stopping_steps)
    x_trend = np.linspace(min(dkl_inv_list), max(dkl_inv_list), 100)
    y_trend = slope * x_trend + intercept
    axs[1].plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2, label=f"Trend ($R^2 = {r_value**2:.2f}$)")
    
    axs[1].set_xlabel("Inverse Signal Strength ($1 / D_{KL}$)", fontsize=12)
    axs[1].set_ylabel("Stopping Step ($N$)", fontsize=12)
    axs[1].set_title("Per-Prompt Sample Counts follow $1/D_{KL}$", fontsize=14, fontweight='bold')
    axs[1].legend(fontsize=11)
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/wald_bounds.png", dpi=300)
    print("Saved plot to figures/wald_bounds.png")

if __name__ == "__main__":
    main()
