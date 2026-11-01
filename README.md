# Sequential Anytime Valid Tests for Efficient Inference-Time Alignment

Official codebase for **"Sequential Anytime Valid Tests for Efficient Inference-Time Alignment"**.

This repository provides reference implementations of our anytime-valid sequential early-stopping frameworks (**E-RWSC**, **SPRT-RWSC**) and topological consensus selection (**RWSC**), alongside automated candidate generation pipelines (via vLLM), reward model scoring ensembles, theoretical bound verification, and reproduction scripts for all paper figures.

---

## Overview & Key Contributions

Aligning Large Language Models (LLMs) via test-time compute scaling typically relies on Best-of-$N$ (BoN) sampling: generating $N$ candidates and selecting the highest-scoring candidate under a proxy Reward Model (RM). However, proxy RMs are imperfect heuristics susceptible to **reward hacking** when over-optimized. Moreover, allocating a static maximum budget $N_{\max}$ across all prompts is computationally wasteful, spending identical compute on confident/trivial queries and highly uncertain ones.

Our framework introduces anytime-valid statistical testing and rank-weighted topological consensus to achieve an optimal Pareto frontier between accuracy and inference compute:

1. **Rank-Weighted Self-Consistency (RWSC)**:
   Shifts the selection objective from individual candidate scores to **answer-level topological consensus**. By integrating rank-based reward confidence with semantic clustering, RWSC outperforms Vanilla BoN and standard Self-Consistency across mathematical and multi-turn reasoning benchmarks while guarding against reward over-optimization.

2. **Anytime-Valid E-Process Early Stopping (E-RWSC)**:
   Constructs a non-negative supermartingale (test martingale / $e$-process) over sequential candidate draws. Guarantees non-asymptotic Type-I error control under arbitrary stopping times (anytime validity), dynamically halting compute as soon as evidence is statistically decisive and cutting average inference budget by up to **60%**.

3. **Sequential Probability Ratio Test (SPRT-RWSC)**:
   Sequential early stopping under Wald's SPRT framework with bounded Type-I ($\alpha$) and Type-II ($\beta$) error rates, achieving near-optimal budget reduction.

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
│   ├── 03_run_experiments.sh           # Batch execution of all core baselines and pipelines
│   ├── 05_run_burnin_ablation.sh       # Burn-in ablation sweep
│   ├── 06_run_budget_sweep.sh          # Full N_max budget sweep experiments
│   ├── 07_run_pdagger_sweep.sh         # Prior alignment (p_dagger) sweep
│   └── 08_run_c_sweep.sh               # Stopping boundary parameter sweep
│
├── scripts/                    # Core reproducible pipeline entrypoints
│   ├── generate_candidates_*.py        # High-throughput candidate generation (vLLM / HF / Ollama)
│   ├── score_candidates.py             # RM verification scoring across ensemble
│   ├── extract_logprobs_from_existing.py # Retrospective VLLM logprob extraction for W-CITE
│   ├── score_self_certainty.py         # Self-certainty confidence scoring
│   ├── run_all_pipelines.py            # Master evaluation pipeline runner
│   ├── run_pipeline.py                 # Single pipeline execution entrypoint
│   │
│   ├── plots/                          # Paper figures & Pareto visualizers
│   │   ├── plot_table1.py              # Generates full-budget bar charts & Pareto frontier subplots
│   │   ├── plot_eprocess_stopping_dist.py # Stopping distributions under N_max=512
│   │   ├── plot_hard_vs_easy_accurate.py  # Confident vs. highly uncertain subset failure modes
│   │   ├── plot_eprocess_bounds.py     # Empirical validation of theoretical E-Process bounds
│   │   ├── plot_wald_bounds.py         # Empirical validation of SPRT bounds
│   │   ├── plot_ablations.py           # Sensitivity ablations (alpha, p_dagger, c, burn-in)
│   │   └── generate_plots.py           # Multi-benchmark Pareto frontier curves
│   │
│   └── analysis/                       # Deep-dive analytical scripts
│       ├── analyze_exhausted_questions.py # Failure-mode breakdown on exhausted questions
│       └── eval_cite_512.py            # Vanilla CITE budget sweep evaluations for N=512
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
All plotting and theoretical bounding scripts are centrally located in the `scripts/plots/` directory:
```bash
# Table 1: Full-budget bar charts and Early-Stopping Pareto frontiers
python scripts/plots/plot_table1.py

# Stopping step distribution under N=512
python scripts/plots/plot_eprocess_stopping_dist.py

# Failure modes on Confident vs. Highly Uncertain queries
python scripts/plots/plot_hard_vs_easy_accurate.py

# Empirical validation of theoretical stopping-time bounds
python scripts/plots/plot_eprocess_bounds.py
```

