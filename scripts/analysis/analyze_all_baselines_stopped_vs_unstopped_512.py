import json
import os
import sys
import time
import pickle
import math
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

consol_path = os.path.join(ROOT_DIR, "scratch", "consol")
if consol_path not in sys.path:
    sys.path.append(consol_path)

from src.pipelines.base import _extract_answer
from math_verify import parse, verify
from src.methods.sequential.e_process import _e_rwsc_stop, _cite_stop
from src.methods.sequential.sprt import _sprt_stop
from src.methods.sequential.mmc import _mmc_stop
from src.methods.sequential.baselines import (
    compute_adabon,
    compute_beacon,
    compute_pandora,
    compute_consol,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET = "math500"
N_MAX = 512
CANDIDATES_PATH = os.path.join(
    ROOT_DIR, "data/candidates/qwen2.5-3b-instruct/math500_n512_full_vllm.jsonl"
)
SCORES_PATH = os.path.join(
    ROOT_DIR, "data/scores/qwen2.5-3b-instruct/math500_n512_full_vllm_scores_ArmoRM-Llama3-8B-v0.1.jsonl"
)
CLUSTER_CACHE_PATH = os.path.join(ROOT_DIR, "data/cache_clusters_math500_n512.pkl")
TOKEN_CACHE_PATH = os.path.join(ROOT_DIR, "data/cache_tokens_math500_n512.pkl")
RESULTS_JSON_PATH = os.path.join(ROOT_DIR, "runs/math500_n512_stopped_vs_unstopped_all_baselines.json")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)

# -----------------------------------------------------------------------------
# Helper: Fast Semantic Clustering
# -----------------------------------------------------------------------------
def fast_semantic_clustering(candidates, dataset):
    ans_list = [_extract_answer(c, dataset) for c in candidates]
    unique_ans = list(set(ans_list))

    parsed_ans = {}
    for a in unique_ans:
        try:
            parsed_ans[a] = parse(a)
        except Exception:
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
                except Exception:
                    pass
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

from src.core.data import check_correctness

# -----------------------------------------------------------------------------
# Helper: Check Math Correctness (Official Rigorous Evaluation)
# -----------------------------------------------------------------------------
def check_math_correct(cand_text, gt_text, dataset="math500"):
    return check_correctness(cand_text, gt_text, dataset)

def main():
    print("=" * 80)
    print(f"  Stopped vs. Unstopped Performance & Token Footprint (N_max = {N_MAX})")
    print(f"  Dataset: {DATASET} | GPU / CUDA: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Default')}")
    print("=" * 80)

    # 1. Load Data
    print(f"\n[1/5] Loading Candidates and ArmoRM Scores...")
    cands_data = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cands_data[d["example_index"]] = d

    scores_data = {}
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            scores_data[d["example_index"]] = d["scores"]

    idx_list = sorted([idx for idx in cands_data if idx in scores_data])
    n_examples = len(idx_list)
    print(f"  Loaded {n_examples} examples with N={N_MAX} candidates and scores.")

    # 2. Token Counts Precomputation / Cache
    print(f"\n[2/5] Computing / Loading Token Counts...")
    if os.path.exists(TOKEN_CACHE_PATH):
        print(f"  Loading cached tokens from {TOKEN_CACHE_PATH}...")
        with open(TOKEN_CACHE_PATH, "rb") as f:
            token_counts = pickle.load(f)
    else:
        print(f"  Tokenizing candidates using Qwen/Qwen2.5-3B-Instruct tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
        
        token_counts = {}
        for idx in tqdm(idx_list, desc="Tokenizing"):
            candidates = cands_data[idx]["candidates"][:N_MAX]
            # Fast batch token length calculation
            encodings = tokenizer(candidates, add_special_tokens=False, padding=False, return_attention_mask=False)
            token_counts[idx] = [len(ids) for ids in encodings["input_ids"]]
            
        with open(TOKEN_CACHE_PATH, "wb") as f:
            pickle.dump(token_counts, f)
        print(f"  Cached token lengths to {TOKEN_CACHE_PATH}.")

    # 3. Semantic Clusters Precomputation / Cache
    print(f"\n[3/5] Computing / Loading Semantic Clusters...")
    if os.path.exists(CLUSTER_CACHE_PATH):
        print(f"  Loading cached semantic clusters from {CLUSTER_CACHE_PATH}...")
        with open(CLUSTER_CACHE_PATH, "rb") as f:
            precomputed_clusters = pickle.load(f)
    else:
        print(f"  Clustering {n_examples} examples (N={N_MAX})...")
        precomputed_clusters = {}
        for idx in tqdm(idx_list, desc="Clustering"):
            candidates = cands_data[idx]["candidates"][:N_MAX]
            cids, _ = fast_semantic_clustering(candidates, DATASET)
            precomputed_clusters[idx] = cids
            
        with open(CLUSTER_CACHE_PATH, "wb") as f:
            pickle.dump(precomputed_clusters, f)
        print(f"  Cached semantic clusters to {CLUSTER_CACHE_PATH}.")

    # Precalculate AdaBoN budget allocation
    print("\n  Precalculating AdaBoN global variance budgets...")
    K = 4
    stds = {}
    for idx in idx_list:
        scores = np.asarray(scores_data[idx][:N_MAX], dtype=float)
        stds[idx] = float(np.std(scores[:K]))
    total_stds = sum(stds.values())
    total_remaining_budget = n_examples * (N_MAX - K)
    adabon_budgets = {}
    if total_stds > 1e-6:
        for idx, std in stds.items():
            prop = std / total_stds
            budget = K + int(round(total_remaining_budget * prop))
            adabon_budgets[idx] = max(K, min(budget, N_MAX))
    else:
        for idx in stds:
            adabon_budgets[idx] = N_MAX

    # 4. Run All Methods
    print(f"\n[4/5] Evaluating Adaptive Stopping Methods on N_max = {N_MAX}...")
    
    methods = [
        "Vanilla BoN",
        "Self-Consistency",
        "AdaBoN",
        "Pandora's Box",
        "BEACON",
        "ConSol",
        "CITE",
        "MMC",
        "SPRT-RWSC (Ours)",
        "E-RWSC (Ours)"
    ]

    method_results = {m: {
        "stopped_at": [],
        "is_correct": [],
        "stopped_early": [],
        "tokens": []
    } for m in methods}

    for idx in tqdm(idx_list, desc="Evaluating All Methods"):
        candidates = cands_data[idx]["candidates"][:N_MAX]
        scores = np.asarray(scores_data[idx][:N_MAX], dtype=float)
        global_cluster_ids = precomputed_clusters[idx][:N_MAX]
        tok_lens = token_counts[idx][:N_MAX]
        gt = cands_data[idx]["ground_truth"]

        # ---------------------------------------------------------------------
        # 1. Vanilla BoN (N=512)
        # ---------------------------------------------------------------------
        bon_idx = int(np.argmax(scores))
        bon_corr = check_math_correct(candidates[bon_idx], gt, DATASET)
        method_results["Vanilla BoN"]["stopped_at"].append(N_MAX)
        method_results["Vanilla BoN"]["is_correct"].append(bon_corr)
        method_results["Vanilla BoN"]["stopped_early"].append(False)
        method_results["Vanilla BoN"]["tokens"].append(sum(tok_lens))

        # ---------------------------------------------------------------------
        # 2. Self-Consistency (N=512)
        # ---------------------------------------------------------------------
        counts = Counter(global_cluster_ids)
        best_sc_cid = counts.most_common(1)[0][0]
        best_sc_idx = global_cluster_ids.index(best_sc_cid)
        sc_corr = check_math_correct(candidates[best_sc_idx], gt, DATASET)
        method_results["Self-Consistency"]["stopped_at"].append(N_MAX)
        method_results["Self-Consistency"]["is_correct"].append(sc_corr)
        method_results["Self-Consistency"]["stopped_early"].append(False)
        method_results["Self-Consistency"]["tokens"].append(sum(tok_lens))

        # ---------------------------------------------------------------------
        # 3. AdaBoN
        # ---------------------------------------------------------------------
        ad_budget = adabon_budgets[idx]
        ad_cand, ad_steps = compute_adabon(candidates, scores, ad_budget)
        ad_corr = check_math_correct(ad_cand, gt, DATASET)
        method_results["AdaBoN"]["stopped_at"].append(ad_steps)
        method_results["AdaBoN"]["is_correct"].append(ad_corr)
        method_results["AdaBoN"]["stopped_early"].append(ad_steps < N_MAX)
        method_results["AdaBoN"]["tokens"].append(sum(tok_lens[:ad_steps]))

        # ---------------------------------------------------------------------
        # 4. Pandora's Box
        # ---------------------------------------------------------------------
        pan_cand, pan_steps = compute_pandora(candidates, scores, c=0.05)
        pan_corr = check_math_correct(pan_cand, gt, DATASET)
        method_results["Pandora's Box"]["stopped_at"].append(pan_steps)
        method_results["Pandora's Box"]["is_correct"].append(pan_corr)
        method_results["Pandora's Box"]["stopped_early"].append(pan_steps < N_MAX)
        method_results["Pandora's Box"]["tokens"].append(sum(tok_lens[:pan_steps]))

        # ---------------------------------------------------------------------
        # 5. BEACON
        # ---------------------------------------------------------------------
        bea_cand, bea_steps = compute_beacon(candidates, scores, gamma=0.85)
        bea_corr = check_math_correct(bea_cand, gt, DATASET)
        method_results["BEACON"]["stopped_at"].append(bea_steps)
        method_results["BEACON"]["is_correct"].append(bea_corr)
        method_results["BEACON"]["stopped_early"].append(bea_steps < N_MAX)
        method_results["BEACON"]["tokens"].append(sum(tok_lens[:bea_steps]))

        # ---------------------------------------------------------------------
        # 6. ConSol
        # ---------------------------------------------------------------------
        con_cand, con_steps = compute_consol(candidates, global_cluster_ids)
        con_corr = check_math_correct(con_cand, gt, DATASET)
        method_results["ConSol"]["stopped_at"].append(con_steps)
        method_results["ConSol"]["is_correct"].append(con_corr)
        method_results["ConSol"]["stopped_early"].append(con_steps < N_MAX)
        method_results["ConSol"]["tokens"].append(sum(tok_lens[:con_steps]))

        # ---------------------------------------------------------------------
        # 7. CITE (Alpha = 0.1)
        # ---------------------------------------------------------------------
        cite_res = _cite_stop(
            candidates=candidates,
            raw_scores=None,
            variances=None,
            global_cluster_ids=global_cluster_ids,
            alpha=0.1,
            burnin=4,
            use_wcite=False,
        )
        cite_steps = cite_res["stopped_at"]
        cite_cand = candidates[cite_res["best_idx"]]
        cite_corr = check_math_correct(cite_cand, gt, DATASET)
        cite_stopped = (cite_res["decision"] == "accept" and cite_steps < N_MAX)
        method_results["CITE"]["stopped_at"].append(cite_steps)
        method_results["CITE"]["is_correct"].append(cite_corr)
        method_results["CITE"]["stopped_early"].append(cite_stopped)
        method_results["CITE"]["tokens"].append(sum(tok_lens[:cite_steps]))

        # ---------------------------------------------------------------------
        # 7b. MMC (Martingale Majority Certificate)
        # ---------------------------------------------------------------------
        mmc_res = _mmc_stop(candidates, global_cluster_ids, alpha=0.1, N_max=N_MAX)
        mmc_steps = mmc_res["stopped_at"]
        mmc_cand = candidates[mmc_res["best_idx"]]
        mmc_corr = check_math_correct(mmc_cand, gt, DATASET)
        mmc_stopped = (mmc_res["is_certified"] and mmc_steps < N_MAX)
        method_results["MMC"]["stopped_at"].append(mmc_steps)
        method_results["MMC"]["is_correct"].append(mmc_corr)
        method_results["MMC"]["stopped_early"].append(mmc_stopped)
        method_results["MMC"]["tokens"].append(sum(tok_lens[:mmc_steps]))

        # ---------------------------------------------------------------------
        # 8. SPRT-RWSC (Alpha=0.1, Beta=0.1)
        # ---------------------------------------------------------------------
        sprt_res = _sprt_stop(
            candidates=candidates,
            raw_scores=scores,
            variances=None,
            global_cluster_ids=global_cluster_ids,
            alpha=0.1,
            beta=0.1,
            burnin=4,
            p0_mode="uniform",
            use_ensemble_weighting=False,
            temperature=1.0,
            agg_mode="rwsc"
        )
        sprt_steps = sprt_res["stopped_at"]
        sprt_cand = candidates[sprt_res["best_idx"]]
        sprt_corr = check_math_correct(sprt_cand, gt, DATASET)
        sprt_stopped = (sprt_res["decision"] == "accept" and sprt_steps < N_MAX)
        method_results["SPRT-RWSC (Ours)"]["stopped_at"].append(sprt_steps)
        method_results["SPRT-RWSC (Ours)"]["is_correct"].append(sprt_corr)
        method_results["SPRT-RWSC (Ours)"]["stopped_early"].append(sprt_stopped)
        method_results["SPRT-RWSC (Ours)"]["tokens"].append(sum(tok_lens[:sprt_steps]))

        # ---------------------------------------------------------------------
        # 9. E-RWSC (Alpha=0.1, p_dagger=0.5, c=0.8)
        # ---------------------------------------------------------------------
        erwsc_res = _e_rwsc_stop(
            candidates=candidates,
            raw_scores=scores,
            variances=None,
            global_cluster_ids=global_cluster_ids,
            alpha=0.1,
            p_dagger=0.5,
            c=0.8,
            m=4.0,
            burnin=4,
            use_ensemble_weighting=False,
            temperature=1.0
        )
        erwsc_steps = erwsc_res["stopped_at"]
        erwsc_cand = candidates[erwsc_res["best_idx"]]
        erwsc_corr = check_math_correct(erwsc_cand, gt, DATASET)
        erwsc_stopped = (erwsc_res["decision"] == "accept" and erwsc_steps < N_MAX)
        method_results["E-RWSC (Ours)"]["stopped_at"].append(erwsc_steps)
        method_results["E-RWSC (Ours)"]["is_correct"].append(erwsc_corr)
        method_results["E-RWSC (Ours)"]["stopped_early"].append(erwsc_stopped)
        method_results["E-RWSC (Ours)"]["tokens"].append(sum(tok_lens[:erwsc_steps]))

    # 5. Summarize & Print Results Table
    print(f"\n[5/5] Compiling Statistics & Generating Publication Outputs...")
    
    summary_data = {}
    full_budget_tokens = np.mean(method_results["Vanilla BoN"]["tokens"])

    for m in methods:
        steps = np.array(method_results[m]["stopped_at"])
        corrs = np.array(method_results[m]["is_correct"], dtype=bool)
        stopped_flags = np.array(method_results[m]["stopped_early"], dtype=bool)
        tokens = np.array(method_results[m]["tokens"])

        overall_acc = np.mean(corrs) * 100
        overall_steps = np.mean(steps)
        overall_tokens_k = np.mean(tokens) / 1000.0

        n_stopped = np.sum(stopped_flags)
        stop_rate = (n_stopped / n_examples) * 100.0

        if n_stopped > 0:
            stopped_acc = np.mean(corrs[stopped_flags]) * 100.0
            stopped_steps = np.mean(steps[stopped_flags])
            stopped_tokens_k = np.mean(tokens[stopped_flags]) / 1000.0
        else:
            stopped_acc = float('nan')
            stopped_steps = float('nan')
            stopped_tokens_k = float('nan')

        n_exhausted = n_examples - n_stopped
        exhaust_rate = 100.0 - stop_rate
        if n_exhausted > 0:
            exhausted_acc = np.mean(corrs[~stopped_flags]) * 100.0
            exhausted_tokens_k = np.mean(tokens[~stopped_flags]) / 1000.0
        else:
            exhausted_acc = float('nan')
            exhausted_tokens_k = float('nan')

        token_savings = ((full_budget_tokens - np.mean(tokens)) / full_budget_tokens) * 100.0
        reliability_gap = stopped_acc - exhausted_acc if (not np.isnan(stopped_acc) and not np.isnan(exhausted_acc)) else float('nan')

        summary_data[m] = {
            "overall_acc": float(overall_acc),
            "overall_steps": float(overall_steps),
            "overall_tokens_k": float(overall_tokens_k),
            "stop_rate": float(stop_rate),
            "stopped_acc": float(stopped_acc) if not np.isnan(stopped_acc) else None,
            "stopped_steps": float(stopped_steps) if not np.isnan(stopped_steps) else None,
            "stopped_tokens_k": float(stopped_tokens_k) if not np.isnan(stopped_tokens_k) else None,
            "exhaust_rate": float(exhaust_rate),
            "exhausted_acc": float(exhausted_acc) if not np.isnan(exhausted_acc) else None,
            "exhausted_tokens_k": float(exhausted_tokens_k) if not np.isnan(exhausted_tokens_k) else None,
            "token_savings": float(token_savings),
            "reliability_gap": float(reliability_gap) if not np.isnan(reliability_gap) else None,
        }

    # Save to JSON
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Saved full results JSON to {RESULTS_JSON_PATH}")

    # Print ASCII Table
    print("\n" + "=" * 125)
    print(f"{'Method':<18} | {'Overall':<20} | {'Stopped Early (τ < 512)':<32} | {'Budget Exhausted (τ = 512)':<25} | {'Token':<6} | {'Gap'}")
    print(f"{'':<18} | {'Acc%':<6} {'Avg N':<6} {'Tok(k)':<6} | {'Stop%':<6} {'Acc%':<6} {'Avg τ':<6} {'Tok(k)':<7} | {'Exh%':<6} {'Acc%':<6} {'Tok(k)':<7} | {'Sav%':<6} | {'Δ_acc'}")
    print("-" * 125)

    for m in methods:
        d = summary_data[m]
        st_acc_str = f"{d['stopped_acc']:5.1f}%" if d['stopped_acc'] is not None else "  --- "
        st_tau_str = f"{d['stopped_steps']:5.1f}" if d['stopped_steps'] is not None else "  --- "
        st_tok_str = f"{d['stopped_tokens_k']:5.1f}k" if d['stopped_tokens_k'] is not None else "  --- "
        
        ex_acc_str = f"{d['exhausted_acc']:5.1f}%" if d['exhausted_acc'] is not None else "  --- "
        ex_tok_str = f"{d['exhausted_tokens_k']:5.1f}k" if d['exhausted_tokens_k'] is not None else "  --- "
        gap_str = f"{d['reliability_gap']:+5.1f}%" if d['reliability_gap'] is not None else "  --- "

        print(
            f"{m:<18} | "
            f"{d['overall_acc']:5.1f}% {d['overall_steps']:6.1f} {d['overall_tokens_k']:5.1f}k | "
            f"{d['stop_rate']:5.1f}% {st_acc_str} {st_tau_str} {st_tok_str} | "
            f"{d['exhaust_rate']:5.1f}% {ex_acc_str} {ex_tok_str} | "
            f"{d['token_savings']:5.1f}% | "
            f"{gap_str}"
        )
    print("=" * 125 + "\n")

    # Generate LaTeX Table
    latex_table = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{l ccc cccc ccc c c}
\toprule
& \multicolumn{3}{c}{\textbf{Overall}} 
& \multicolumn{4}{c}{\textbf{Stopped Early} ($\tau < N_{\max}$)} 
& \multicolumn{3}{c}{\textbf{Budget Exhausted} ($\tau = N_{\max}$)} 
& \textbf{Token} 
& \textbf{Gap} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-8} \cmidrule(lr){9-11} \cmidrule(lr){12-12} \cmidrule(lr){13-13}
\textbf{Method} 
& Acc & $\bar{N}$ & Tokens 
& Stop Rate & Acc & $\bar{\tau}$ & Tokens 
& Exhaust & Acc & Tokens 
& Savings & $\Delta_{\text{acc}}$ \\
& (\%) & & (k) 
& (\%) & (\%) & & (k) 
& (\%) & (\%) & (k) 
& (\%) & (\%) \\
\midrule
"""
    for m in methods:
        d = summary_data[m]
        st_acc_tex = f"{d['stopped_acc']:.1f}" if d['stopped_acc'] is not None else "---"
        st_tau_tex = f"{d['stopped_steps']:.1f}" if d['stopped_steps'] is not None else "---"
        st_tok_tex = f"{d['stopped_tokens_k']:.1f}" if d['stopped_tokens_k'] is not None else "---"
        
        ex_acc_tex = f"{d['exhausted_acc']:.1f}" if d['exhausted_acc'] is not None else "---"
        ex_tok_tex = f"{d['exhausted_tokens_k']:.1f}" if d['exhausted_tokens_k'] is not None else "---"
        gap_tex = f"{d['reliability_gap']:+.1f}" if d['reliability_gap'] is not None else "---"

        latex_table += (
            f"{m:<18} & {d['overall_acc']:.1f} & {d['overall_steps']:.1f} & {d['overall_tokens_k']:.1f} & "
            f"{d['stop_rate']:.1f} & {st_acc_tex} & {st_tau_tex} & {st_tok_tex} & "
            f"{d['exhaust_rate']:.1f} & {ex_acc_tex} & {ex_tok_tex} & "
            f"{d['token_savings']:.1f}\\% & {gap_tex} \\\\\n"
        )
    latex_table += r"""\bottomrule
\end{tabular}
\caption{Stopped vs. exhausted performance and token expenditure on MATH500 ($N_{\max}=512$). E-RWSC attains peak accuracy when stopping while requiring minimal tokens, and gracefully falls back to consensus on exhausted queries.}
\label{tab:math500_n512_stopped_vs_unstopped}
\end{table*}
"""
    latex_path = os.path.join(ROOT_DIR, "scratch/table_math500_n512_stopped_vs_unstopped.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"  Saved LaTeX table to {latex_path}")

    # Generate Publication Plots
    plot_results(summary_data, methods)

def plot_results(summary_data, methods):
    plt.style.use('ggplot')
    
    # Filter methods with both stopped and exhausted (or baselines)
    plot_methods = [m for m in methods if m not in ["Vanilla BoN", "Self-Consistency", "Pandora's Box"]]
    
    stopped_accs = [summary_data[m]["stopped_acc"] if summary_data[m]["stopped_acc"] is not None else np.nan for m in plot_methods]
    exhaust_accs = [summary_data[m]["exhausted_acc"] if summary_data[m]["exhausted_acc"] is not None else np.nan for m in plot_methods]
    overall_accs = [summary_data[m]["overall_acc"] for m in plot_methods]

    # Figure 1: Stopped vs. Exhausted Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(plot_methods))
    width = 0.28

    r1 = ax.bar(x - width, stopped_accs, width, label="Stopped Early (τ < 512)", color="#2ecc71", edgecolor="black", alpha=0.9)
    r2 = ax.bar(x, overall_accs, width, label="Overall Accuracy", color="#3498db", edgecolor="black", alpha=0.9)
    r3 = ax.bar(x + width, exhaust_accs, width, label="Exhausted Budget (τ = 512)", color="#e74c3c", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("MATH500 (N_max=512): Accuracy on Stopped vs. Exhausted Queries", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_methods, fontsize=12, fontweight="bold", rotation=15)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.tight_layout()
    fig_path1 = os.path.join(FIGURES_DIR, "math500_n512_stopped_vs_exhausted_acc.png")
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    print(f"  Saved figure to {fig_path1}")

    # Figure 2: Token Consumption vs. Stopping Reliability
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    stop_rates = [summary_data[m]["stop_rate"] for m in plot_methods]
    tokens_k = [summary_data[m]["overall_tokens_k"] for m in plot_methods]
    
    colors = ['#f39c12', '#9b59b6', '#34495e', '#16a085', '#2980b9', '#27ae60']
    offsets = {
        "AdaBoN": (8, -2),
        "BEACON": (8, -2),
        "ConSol": (10, -13),
        "CITE": (8, 4),
        "SPRT-RWSC (Ours)": (10, 8),
        "E-RWSC (Ours)": (8, 4),
    }
    for i, m in enumerate(plot_methods):
        ax2.scatter(tokens_k[i], stopped_accs[i], s=120, color=colors[i % len(colors)], edgecolors="black", linewidth=1.5, zorder=5)
        offset = offsets.get(m, (5, 5))
        ax2.annotate(m, (tokens_k[i], stopped_accs[i]), textcoords="offset points", xytext=offset, fontsize=11, fontweight="bold")

    ax2.set_xlim(-5, 240)
    ax2.set_ylim(69, 97)
    ax2.set_xlabel("Overall Average Tokens Generated (k-tokens)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Accuracy on Stopped Queries (%)", fontsize=13, fontweight="bold")
    ax2.set_title("Decision Reliability vs. Token Footprint (MATH500 N_max=512)", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "math500_n512_tokens_vs_stopped_acc.png")
    fig2.savefig(fig_path2, dpi=300)
    plt.close(fig2)
    print(f"  Saved figure to {fig_path2}")

if __name__ == "__main__":
    main()
