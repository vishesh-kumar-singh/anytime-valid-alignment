import os
import sys
import glob
import json
from pathlib import Path

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipelines.sprt import SPRTPipeline

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base model short name.")
    parser.add_argument("--reward-models", type=str, default="*", help="Comma-separated shortnames of reward models, or * for all.")
    parser.add_argument("--output-model", type=str, default=None, help="Output folder name for runs (defaults to --model).")
    parser.add_argument("--dataset", type=str, default="math500")
    args = parser.parse_args()
    args.output_model = args.output_model or args.model
    
    print("=============================================")
    print(f" Running SPRT Burn-in Ablation on {args.dataset}")
    print("=============================================")
    
    candidates_path = f"data/candidates/{args.model}/{args.dataset}_n24_full_vllm.jsonl"
    if args.reward_models == "*":
        score_paths = f"data/scores/{args.model}/{args.dataset}_n24_full_vllm_scores_*.jsonl"
    else:
        rms = [rm.strip() for rm in args.reward_models.split(",")]
        score_paths = ",".join([f"data/scores/{args.model}/{args.dataset}_n24_full_vllm_scores_{rm}.jsonl" for rm in rms])
    
    # Isolate Burn-in: We fix alpha=beta=0.05, and sweep over burnin sizes
    ab_configs = [(0.05, 0.05)]
    burnin_sizes = [2, 4, 6, 8]
    
    pipeline = SPRTPipeline(ab_configs=ab_configs, burnin_sizes=burnin_sizes)
    pipeline.name = f"ablations/ablation_burnin_{args.dataset}"
    results = pipeline.run(candidates_path, score_paths, eval_n="24")
    
    from src.pipelines.e_process import EProcessPipeline
    epipe = EProcessPipeline(alpha_values=[0.05], p_dagger_values=[0.2], m_values=[4.0], c_values=[0.8])
    epipe.burnin_sizes = burnin_sizes
    e_results = epipe.run(candidates_path, score_paths, eval_n="24")
    
    # Merge EProcess results
    for n in e_results.get("results_by_n", {}):
        if n not in results["results_by_n"]:
            results["results_by_n"][n] = {}
        results["results_by_n"][n].update(e_results["results_by_n"][n])
        
    if "average_budget" in e_results:
        if "average_budget" not in results:
            results["average_budget"] = {}
        results["average_budget"].update(e_results["average_budget"])
    
    # Filter out SC results so we only report RWSC and Borda
    for n in results.get("results_by_n", {}):
        filtered_methods = {}
        for method, acc in results["results_by_n"][n].items():
            if "_sc_" not in method:
                filtered_methods[method] = acc
        results["results_by_n"][n] = filtered_methods
        
    if "average_budget" in results:
        filtered_budgets = {}
        for method, budget in results["average_budget"].items():
            if "_sc_" not in method:
                filtered_budgets[method] = budget
        results["average_budget"] = filtered_budgets

    pipeline.save(results, args.dataset, experiment_name=pipeline.name, model=args.output_model)
    
    # We no longer run SPRT SC via a separate ablation pipeline, as it's now cleanly integrated into reward_free
    
    print(f"\n✓ Burn-in ablation completed. Results saved to runs/{args.output_model}/{pipeline.name}")

if __name__ == "__main__":
    main()
