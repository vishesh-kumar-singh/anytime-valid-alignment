import json
import os
import sys
import time
import pickle
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.methods.sequential.mmc import _mmc_stop
from src.core.data import check_correctness
from src.pipelines.base import _extract_answer

def eval_mmc_512():
    print("=" * 80)
    print("  EVALUATING MMC ON MATH500 (N_max = 512)")
    print("=" * 80)

    cands_path = os.path.join(ROOT_DIR, "data/candidates/qwen2.5-3b-instruct/math500_n512_full_vllm.jsonl")
    cluster_cache_path = os.path.join(ROOT_DIR, "data/cache_clusters_math500_n512.pkl")
    token_cache_path = os.path.join(ROOT_DIR, "data/cache_tokens_math500_n512.pkl")

    with open(cluster_cache_path, "rb") as f:
        precomputed_clusters = pickle.load(f)
    with open(token_cache_path, "rb") as f:
        token_counts = pickle.load(f)

    examples = []
    with open(cands_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    N_MAX = 512
    stopped_at_list = []
    is_correct_list = []
    is_certified_list = []
    tokens_list = []

    for ex in examples:
        idx = ex["example_index"]
        candidates = ex["candidates"][:N_MAX]
        gt = ex["ground_truth"]
        cids = precomputed_clusters[idx][:N_MAX]
        tok_lens = token_counts[idx][:N_MAX]

        res = _mmc_stop(candidates, cids, alpha=0.1, N_max=N_MAX)
        tau = res["stopped_at"]
        cand = candidates[res["best_idx"]]
        corr = check_correctness(cand, gt, "math500")
        is_cert = res["is_certified"] and (tau < N_MAX)

        stopped_at_list.append(tau)
        is_correct_list.append(corr)
        is_certified_list.append(is_cert)
        tokens_list.append(sum(tok_lens[:tau]))

    stopped_at = np.array(stopped_at_list)
    is_correct = np.array(is_correct_list)
    is_certified = np.array(is_certified_list)
    tokens = np.array(tokens_list)

    overall_acc = np.mean(is_correct) * 100.0
    overall_steps = np.mean(stopped_at)
    overall_tokens_k = np.mean(tokens) / 1000.0

    n_stopped = np.sum(is_certified)
    n_total = len(examples)
    stop_rate = (n_stopped / n_total) * 100.0

    if n_stopped > 0:
        certified_acc = np.mean(is_correct[is_certified]) * 100.0
        stopped_steps = np.mean(stopped_at[is_certified])
        stopped_tokens_k = np.mean(tokens[is_certified]) / 1000.0
    else:
        certified_acc = float('nan')
        stopped_steps = float('nan')
        stopped_tokens_k = float('nan')

    n_exhausted = n_total - n_stopped
    exhaust_rate = 100.0 - stop_rate
    if n_exhausted > 0:
        exhausted_acc = np.mean(is_correct[~is_certified]) * 100.0
        exhausted_steps = np.mean(stopped_at[~is_certified])
        exhausted_tokens_k = np.mean(tokens[~is_certified]) / 1000.0
    else:
        exhausted_acc = float('nan')
        exhausted_steps = float('nan')
        exhausted_tokens_k = float('nan')

    full_budget_tokens = 256285.8
    token_savings = ((full_budget_tokens - np.mean(tokens)) / full_budget_tokens) * 100.0
    gap = certified_acc - exhausted_acc

    print("\n--- RESULTS FOR MATH500 (N=512) ---")
    print(f"Overall Accuracy:         {overall_acc:.1f}%")
    print(f"Overall Average Budget N: {overall_steps:.1f}")
    print(f"Overall Tokens:           {overall_tokens_k:.1f}k")
    print(f"Certification (Stop) Rate:{stop_rate:.1f}% ({n_stopped}/{n_total})")
    print(f"CERTIFIED ACCURACY:       {certified_acc:.1f}%")
    print(f"Certified Budget tau:     {stopped_steps:.1f}")
    print(f"Certified Tokens:         {stopped_tokens_k:.1f}k")
    print(f"Exhaustion Rate:          {exhaust_rate:.1f}% ({n_exhausted}/{n_total})")
    print(f"Exhausted Accuracy:       {exhausted_acc:.1f}%")
    print(f"Token Savings:            {token_savings:.1f}%")
    print(f"Reliability Gap:          {gap:+.1f}%\n")

    return {
        "overall_acc": overall_acc,
        "overall_steps": overall_steps,
        "overall_tokens_k": overall_tokens_k,
        "stop_rate": stop_rate,
        "certified_acc": certified_acc,
        "stopped_steps": stopped_steps,
        "stopped_tokens_k": stopped_tokens_k,
        "exhaust_rate": exhaust_rate,
        "exhausted_acc": exhausted_acc,
        "token_savings": token_savings,
        "gap": gap,
    }


def eval_mmc_n24():
    print("=" * 80)
    print("  EVALUATING MMC ACROSS 4 BENCHMARKS (N_max = 24)")
    print("=" * 80)

    from src.pipelines.seq_baselines import SeqBaselinesPipeline

    datasets = ["gsm8k", "math500", "mmlu", "cruxeval"]
    results = {}

    for ds in datasets:
        cand_path = os.path.join(ROOT_DIR, f"data/candidates/qwen2.5-3b-instruct/{ds}_n24_full_vllm.jsonl")
        score_path = os.path.join(ROOT_DIR, f"data/scores/qwen2.5-3b-instruct/{ds}_n24_full_vllm_scores_ArmoRM-Llama3-8B-v0.1.jsonl")
        print(f"\nProcessing {ds.upper()}...")
        p = SeqBaselinesPipeline()
        res = p.run(candidates_path=cand_path, score_paths=score_path, eval_n="24")
        acc = res["results_by_n"]["24"]["baseline_mmc"] * 100.0
        budget = res["average_budget"].get("baseline_mmc", "N/A")
        results[ds] = {"acc": acc, "budget": budget}
        print(f"  {ds.upper()}: MMC Acc = {acc:.1f}%, Budget = {budget}")

    return results

if __name__ == "__main__":
    t0 = time.time()
    res_512 = eval_mmc_512()
    res_24 = eval_mmc_n24()
    print(f"\nAll evaluations finished in {time.time() - t0:.1f}s.")

