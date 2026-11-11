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
from src.methods.sequential.e_process import _e_rwsc_stop
from src.methods.sequential.sprt import _compute_rwsc_clusters, _get_leader

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
    
    idx_list = sorted(cands_data.keys())
    
    print(f"Evaluating {len(idx_list)} questions...")
    cache_path = "data/cache_clusters_math500_n512.pkl"
    if os.path.exists(cache_path):
        print(f"Loading precomputed clusters from {cache_path}...")
        import pickle
        with open(cache_path, "rb") as f:
            precomputed_clusters = pickle.load(f)
    else:
        for idx in tqdm(idx_list, desc="Precomputing N=512 Clusters"):
            if idx not in scores_data: continue
            ex = cands_data[idx]
            candidates = ex["candidates"]
            global_cluster_ids, _ = fast_semantic_clustering(candidates, dataset)
            precomputed_clusters[idx] = global_cluster_ids

    # Empirical generation probabilities theta_hat(a) over full N=512 pool
    from collections import Counter
    empirical_thetas = {}
    for idx in idx_list:
        if idx not in precomputed_clusters: continue
        c_list = precomputed_clusters[idx]
        counts = Counter(c_list)
        empirical_thetas[idx] = {cid: cnt / len(c_list) for cid, cnt in counts.items()}

    # 1. Sweep Alphas for Realized Error (Theorem 1: P(theta_a <= p_dagger) <= alpha)
    print("\nSweeping alphas for Realized Error (Theorem 1)...")
    alpha_sweep = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    realized_errors = []
    p_dagger = 0.5
    
    for alpha in tqdm(alpha_sweep, desc="Alpha Sweep"):
        errors = 0
        total_prompts = 0
        
        for idx in idx_list:
            if idx not in scores_data or idx not in precomputed_clusters: continue
            total_prompts += 1
            res = _e_rwsc_stop(
                candidates=[''] * len(scores_data[idx]),
                raw_scores=np.array(scores_data[idx]),
                variances=None,
                global_cluster_ids=precomputed_clusters[idx],
                alpha=alpha,
                p_dagger=p_dagger,
                c=0.8,
                m=4.0,
                burnin=4,
                use_ensemble_weighting=False,
                temperature=1.0
            )
            
            if res["decision"] == "accept":
                leader_cid = res["leader_answer"]
                theta_hat = empirical_thetas[idx].get(leader_cid, 0.0)
                # Theorem 1: False certification occurs if theta_a <= p_dagger
                if theta_hat <= p_dagger:
                    errors += 1
                    
        emp_err = errors / total_prompts if total_prompts > 0 else 0
        realized_errors.append(emp_err)

    # 2. D_KL Scaling (Fix Alpha = 0.05)
    print("\nCalculating sample count vs 1/d_KL...")
    target_alpha = 0.05
    dkl_inv_list = []
    stopping_steps = []
    
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
            N_stop = res["stopped_at"]
            leader = res["leader_answer"]
            agreements = sum(1 for cid in global_cluster_ids[:N_stop] if cid == leader)
            theta_hat = agreements / N_stop
            theta_hat = max(min(theta_hat, 0.99), p_dagger + 0.01)
            dkl = theta_hat * math.log(theta_hat / p_dagger) + (1.0 - theta_hat) * math.log((1.0 - theta_hat) / (1.0 - p_dagger))
            if dkl > 0:
                dkl_inv_list.append(1.0 / dkl)
                stopping_steps.append(N_stop)

    # PLOTTING
    print("\nGenerating Figures...")
    plt.style.use('ggplot')
    os.makedirs("figures", exist_ok=True)
    
    # Cache computed results
    cache_results_path = "data/eprocess_bounds_results.npz"
    np.savez(
        cache_results_path,
        alpha_sweep=alpha_sweep,
        realized_errors=realized_errors,
        dkl_inv_list=dkl_inv_list,
        stopping_steps=stopping_steps
    )
    print(f"Cached results to {cache_results_path}")

    # Figure 1: Realized Error vs Alpha
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(alpha_sweep, realized_errors, marker='o', color='royalblue', linewidth=2.5, markersize=8, label="E-RWSC Realized Error", zorder=4)
    ax1.plot([0, 0.52], [0, 0.52], color='black', linestyle='--', linewidth=2, alpha=0.7, label="Theoretical Bound ($y=x$)", zorder=3)
    ax1.fill_between([0, 0.52], [0, 0.52], 0, color='gray', alpha=0.15, zorder=2)
    ax1.set_xlim(0, 0.52)
    ax1.set_ylim(-0.015, 0.52)
    ax1.set_yticks(np.arange(0.0, 0.55, 0.05))
    ax1.set_xlabel("Nominal Type-I Error ($\\alpha$)", fontsize=12)
    ax1.set_ylabel("Realized Error Rate", fontsize=12)
    ax1.set_title("Anytime-Valid Certification Bounds", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc="upper left")
    
    fig1.tight_layout()
    fig1.savefig("figures/eprocess_realized_error.png", dpi=300)
    print("Saved plot to figures/eprocess_realized_error.png")
    
    # Figure 2: 1/D_KL Scaling
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(dkl_inv_list, stopping_steps, alpha=0.5, color='mediumseagreen')
    slope, intercept, r_value, p_value, std_err = stats.linregress(dkl_inv_list, stopping_steps)
    x_trend = np.linspace(min(dkl_inv_list), max(dkl_inv_list), 100)
    y_trend = slope * x_trend + intercept
    ax2.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2, label=f"Trend ($R^2 = {r_value**2:.2f}$)")
    ax2.set_xlabel("Inverse Signal Strength ($1 / d_{KL}(\\hat{\\theta} \\| p^\\dagger)$)", fontsize=12)
    ax2.set_ylabel("Stopping Step ($N$)", fontsize=12)
    ax2.set_title("Sample Complexity vs Signal Strength", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    
    fig2.tight_layout()
    fig2.savefig("figures/eprocess_dkl_scaling.png", dpi=300)
    print("Saved plot to figures/eprocess_dkl_scaling.png")

if __name__ == "__main__":
    main()
