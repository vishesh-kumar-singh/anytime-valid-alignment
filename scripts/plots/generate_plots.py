import json
import math
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

def kl_div(p, q):
    if p <= 0 or p >= 1:
        return np.inf
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def load_results(model, dataset, pipeline, suffix="_n512"):
    path = Path(f"runs/{model}/{pipeline}_{dataset}{suffix}/results.json")
    if not path.exists():
        print(f"Warning: {path} not found.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def plot_p1(e_rwsc_res, dataset, output_dir):
    """Plot P1: Mean N vs log(1/alpha)"""
    if not e_rwsc_res or "average_budget" not in e_rwsc_res:
        return
    
    alpha_values = [0.3, 0.2, 0.1, 0.05]
    pdag = 0.2
    m = 2.0
    c_val = 0.5
    
    xs = []
    ys = []
    for alpha in alpha_values:
        tag = f"e_rwsc_a{alpha}_p{pdag}_m{m}_c{c_val}"
        if tag in e_rwsc_res["average_budget"]:
            xs.append(math.log(1/alpha))
            ys.append(e_rwsc_res["average_budget"][tag]["mean"])
            
    if not xs:
        return

    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, marker='o', linestyle='-', color='blue')
    plt.xlabel(r"$\log(1/\alpha)$")
    plt.ylabel("Mean Candidates Generated ($N$)")
    plt.title(f"Stopping Time vs $\\alpha$ ({dataset.upper()})")
    
    plt.savefig(output_dir / f"P1_stopping_time_{dataset}.png")
    plt.close()

def plot_p2(e_rwsc_res, dataset, output_dir):
    """Plot P2: Per-prompt N vs theta_a* at alpha=0.05"""
    if not e_rwsc_res or "prompt_stats" not in e_rwsc_res:
        return

    alpha = 0.05
    pdag = 0.2
    m = 2.0
    c_val = 0.5
    tag = f"e_rwsc_a{alpha}_p{pdag}_m{m}_c{c_val}"
    
    if tag not in e_rwsc_res["prompt_stats"]["steps"]:
        return

    steps = e_rwsc_res["prompt_stats"]["steps"][tag]
    theta_star = e_rwsc_res["prompt_stats"]["theta_star"]

    if not steps or not theta_star or len(steps) != len(theta_star):
        return

    plt.figure(figsize=(7, 5))
    
    # Add small jitter to steps to see density
    steps_jittered = np.array(steps) + np.random.uniform(-1, 1, len(steps))
    
    plt.scatter(theta_star, steps_jittered, alpha=0.3, color='purple', s=20, label='Prompts')
    
    # Theoretical Bound: log(1/alpha) / D_KL(theta || p_dagger)
    x_theory = np.linspace(pdag + 0.01, 1.0, 100)
    y_theory = [math.log(1/alpha) / kl_div(th, pdag) for th in x_theory]
    plt.plot(x_theory, y_theory, 'r--', linewidth=2, label='Theoretical Bound')
    
    plt.xlabel(r"Empirical Frequency $\bar{\theta}_{a^*}$")
    plt.ylabel("Candidates Generated ($N$)")
    plt.title(f"Adaptivity to Difficulty ({dataset.upper()})")
    plt.ylim(0, 512)
    plt.xlim(0, 1.05)
    plt.legend()
    
    plt.savefig(output_dir / f"P2_adaptivity_{dataset}.png")
    plt.close()

def plot_p3(model, dataset, suffix, output_dir):
    """Plot P3: Accuracy-Cost Frontier"""
    # Load all relevant pipeline results
    baselines = load_results(model, dataset, "baselines", suffix)
    e_rwsc = load_results(model, dataset, "e_rwsc", suffix)
    sprt = load_results(model, dataset, "sprt_rwsc", suffix)
    seq = load_results(model, dataset, "seq_baselines", suffix)
    
    plt.figure(figsize=(8, 6))
    
    # 1. Plot Vanilla BoN Frontier
    if baselines and "results_by_n" in baselines:
        K_vals = []
        acc_vals = []
        for n_str, res in baselines["results_by_n"].items():
            if "vanilla_bon (ensemble)" in res:
                K_vals.append(int(n_str))
                acc_vals.append(res["vanilla_bon (ensemble)"])
        
        # Sort by K
        sorted_pairs = sorted(zip(K_vals, acc_vals))
        if sorted_pairs:
            K_vals, acc_vals = zip(*sorted_pairs)
            plt.plot(K_vals, acc_vals, marker='o', color='gray', linestyle='--', label='Fixed-budget BoN', markersize=6)
            
    # 2. Plot e-RWSC Points (sweep alpha)
    if e_rwsc and "results_by_n" in e_rwsc:
        max_n = str(max([int(n) for n in e_rwsc["results_by_n"].keys()]))
        res_max = e_rwsc["results_by_n"][max_n]
        budgets = e_rwsc.get("average_budget", {})
        
        pdag = 0.2
        m = 2.0
        c_val = 0.5
        xs, ys = [], []
        for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
            tag = f"e_rwsc_a{alpha}_p{pdag}_m{m}_c{c_val}"
            if tag in res_max and tag in budgets:
                ys.append(res_max[tag])
                xs.append(budgets[tag]["mean"])
        if xs:
            # Sort by budget
            sorted_pairs = sorted(zip(xs, ys))
            xs, ys = zip(*sorted_pairs)
            plt.plot(xs, ys, marker='s', color='blue', linestyle='-', label='e-RWSC', markersize=8)

    # 3. Plot SPRT Points
    if sprt and "results_by_n" in sprt:
        max_n = str(max([int(n) for n in sprt["results_by_n"].keys()]))
        res_max = sprt["results_by_n"][max_n]
        budgets = sprt.get("average_budget", {})
        
        sprt_variants = {
            "sprt_rwsc_a0.05_b0.05": ("SPRT-RWSC", "^", "orange"),
            "sprt_sc_a0.05_b0.05": ("SPRT-SC", "v", "green"),
            "sprt_borda_a0.05_b0.05": ("SPRT-Borda", "D", "purple")
        }
        
        for tag, (label, marker, color) in sprt_variants.items():
            if tag in res_max and tag in budgets:
                plt.plot(budgets[tag]["mean"], res_max[tag], marker=marker, color=color, 
                         linestyle='None', label=label, markersize=9, zorder=5)

    plt.xlabel("Mean Candidates Generated ($N$)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Compute Pareto Frontier ({dataset.upper()})")
    plt.xscale('log', base=2)
    plt.legend()
    
    plt.savefig(output_dir / f"P3_pareto_{dataset}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-3b-instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--suffix", default="_n512")
    args = parser.parse_args()

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    e_rwsc_res = load_results(args.model, args.dataset, "e_rwsc", args.suffix)
    if e_rwsc_res:
        plot_p1(e_rwsc_res, args.dataset, out_dir)
        plot_p2(e_rwsc_res, args.dataset, out_dir)
    
    plot_p3(args.model, args.dataset, args.suffix, out_dir)
    print(f"Saved plots for {args.dataset} to {out_dir}")
