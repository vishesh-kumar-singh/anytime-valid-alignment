# Test-Time Alignment with Uncertainty-Aware Ensemble Methods

Research on improving LLM inference-time alignment using reward model ensembles, uncertainty estimation, and mathematically rigorous sequential early stopping. 

This repository contains the complete supplementary codebase for our ICLR submission, providing reference implementations of all proposed algorithms (RWSC, SPRT-RWSC, E-RWSC), alongside automated generation pipelines, rigorous theoretical bound validation, and all manuscript visualizations.

## Background & Problem Statement

Aligning Large Language Models (LLMs) via test-time scaling typically relies on the Best-of-$N$ (BoN) approach: generating $N$ candidate responses and using a proxy Reward Model (RM) to select the best one. However, proxy RMs are imperfect heuristics. As $N$ scales, standard BoN aggressively optimizes against the RM, inevitably forcing the policy into out-of-distribution (OOD) spaces. In these high-divergence regions, maximizing the proxy reward often causes a collapse in the actual, true utility—a phenomenon known as **reward over-optimization** or **reward hacking**.

Furthermore, generating $N$ independent candidates in parallel incurs massive computational overhead, often wasting resources on trivial prompts that do not require high generation budgets.

## Key Contributions

This project establishes a new Pareto frontier on complex reasoning benchmarks by addressing both reward hacking and computational inefficiency:

1. **Reward-Weighted Self-Consistency (RWSC)**
   We shift the selection paradigm from candidate-level evaluation to **answer-level topological consensus**. By leveraging the ensemble's mathematical consensus, we filter out low-quality "tail" candidates and prevent consensus dilution, mitigating isolated RM vulnerabilities and achieving state-of-the-art accuracy.

2. **Sequential Early Stopping (SPRT-RWSC)**
   We introduce a mathematically rigorous sequential halting mechanism based on Wald's Sequential Probability Ratio Test (SPRT). By strictly bounding Type-I/Type-II errors, it guarantees near-optimal accuracy while dynamically halting generation the exact moment statistical convergence is achieved.

3. **E-Process Early Stopping (E-RWSC)**
   Leveraging betting martingales and the E-Process, we introduce an anytime-valid sequential stopping mechanism that provides non-asymptotic bounds. E-RWSC effectively bridges the gap between efficiency and inference scaling laws, adaptively consuming compute only on highly uncertain prompts and reducing computational waste by up to 60%.

## Evaluated Matrix

The codebase is fully equipped to run automated pipelines across the following extensive experimental grid:
* **Generators (Base LLMs)**: `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Meta-Llama-3-8B-Instruct`, `Phi-3-Mini-4K-Instruct`
* **Reward Verifiers**: `ArmoRM`, `Skywork V1/V2`, `FsfairX`, `OffsetBias`, `DeBERTa`, and native logprob-based `W-CITE`
* **Benchmarks**: `GSM8K`, `MATH500`, `MMLU`, `CruxEval`

## Repository Architecture

The execution structure is cleanly decoupled into high-level shell orchestration and modular Python core logic:

```text
├── bash_scripts/               # Automated high-level orchestration
│   ├── 01_setup_and_download.sh        # Setup virtual environment and dependencies
│   ├── 02_generate_and_score.sh        # End-to-end vLLM generation and automated scoring
│   └── 03_run_experiments.sh           # Batch execution of all core baselines and pipelines
│
├── scripts/                    # Direct Python execution entrypoints & analysis
│   ├── generate_candidates_*.py        # High-throughput generation using vLLM
│   ├── score_candidates.py             # RM verification scoring
│   ├── extract_logprobs_from_existing.py # Retrospective VLLM logprob extraction for W-CITE
│   ├── run_all_pipelines.py            # Master evaluation pipeline runner
│   ├── generate_plots.py               # Pareto frontiers and adaptivity visualizations
│   ├── plot_eprocess_bounds.py         # Empirical validation of theoretical E-Process bounds
│   └── analyze_exhausted_*.py          # Hard vs. Easy prompt failure-mode analysis
│
├── src/                        # Core algorithmic framework
│   ├── experiments/ablations/          # Specialized isolated ablation studies
│   ├── pipelines/                      # Orchestrator definitions (RWSC, E-Process, SPRT)
│   ├── methods/                        # Pure algorithmic logic
│   │   ├── parallel/                   # RWSC, Vanilla BoN, Soft BoN
│   │   └── sequential/                 # SPRT, E-Process, AdaBoN, BEACON
│   └── core/                           # Data handling and JSONL I/O
│
├── data/                       # [Auto-Generated] Candidates and parsed scores
└── runs/                       # [Auto-Generated] Final evaluation metrics and summaries
```

## Quick Start Guide

Our workflow is completely streamlined using the `bash_scripts/` directory. Output directories like `data/` and `runs/` are dynamically generated on the fly.

### 1. Environment Setup
Initialize the virtual environment and install all core dependencies, including vLLM and math verification suites:
```bash
bash bash_scripts/01_setup_and_download.sh
```

### 2. Generate and Score Candidates
Run the automated script to generate candidates via vLLM, extract retrospective W-CITE token probabilities, and score candidates across the entire Reward Model ensemble. 
*(Note: It loops over all 4 base models and all datasets automatically).*
```bash
bash bash_scripts/02_generate_and_score.sh
```
*To run on a specific benchmark, simply pass the flag: `--dataset math500`*

### 3. Run Experiments and Baselines
Once candidates and scores are cached, execute the master pipeline to evaluate them across all parallel and sequential alignment algorithms (including RWSC, SPRT, and E-Process).
```bash
bash bash_scripts/03_run_experiments.sh
```
*Results will automatically populate inside the `runs/` directory.*

### 4. Reproduce Figures and Bounds
All plotting and theoretical bounding scripts are centrally located in the `scripts/` directory. For example, to generate the Pareto frontier and stopping-time bounds for MATH500:
```bash
python scripts/generate_plots.py --dataset math500
python scripts/plot_eprocess_bounds.py
```
