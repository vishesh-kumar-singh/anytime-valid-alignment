# Test-Time Alignment with Uncertainty-Aware Ensemble Methods

Research on improving LLM inference-time alignment using reward model ensembles and uncertainty estimation.

**Advisor:** Prof. Sayak Ray Chowdhury, IIT Kanpur

## Background

This project continues from prior UGP work ([noisy-reward-alignment.pdf](noisy-reward-alignment.pdf)) on uncertainty-guided policies for test-time alignment. The prior work showed that:

- Ensemble RM disagreement is a useful signal for candidate quality
- Robust normalization (z-score + tanh) is critical for heterogeneous RM ensembles  
- Uncertainty-aware selection (Sharpe BoN) selects 30–35% lower variance candidates
- Sequential filtering (RF-SeqBoN) provides +4.5% accuracy over parallel BoN on GSM8K

However, the accuracy improvements were marginal (~0.1% on full GSM8K), limiting publishability. This project explores stronger directions.

## Repository Structure

```
├── src/
│   ├── core/                   # Shared infrastructure
│   │   ├── config.py           #   BaseExperimentConfig
│   │   ├── scorer.py           #   RM scoring protocol + implementations
│   │   ├── data.py             #   Dataset loading + correctness checking
│   │   ├── normalize.py        #   Variance normalization strategies
│   │   ├── results.py          #   MethodResult + ResultsManager
│   │   ├── logger.py           #   ExperimentLogger
│   │   └── method_runners.py   #   BoN wrapper functions
│   ├── hypotheses/             # One file per hypothesis
│   │   ├── base.py             #   Abstract Hypothesis class
│   │   ├── h1_reward_hacking_at_scale.py
│   │   ├── h2_ensemble_diversity.py
│   │   ├── h3_variance_normalization.py
│   │   ├── h4_disagreement_hybrid.py
│   │   ├── h5_offline_parallel_methods.py
│   │   ├── h6_self_consistency.py
│   │   ├── h7_rm_debiasing.py
│   │   ├── h8_rank_based_selection.py
│   │   ├── h9_difficulty_stratified.py
│   │   └── h10_pairwise_tournament.py
│   ├── methods/                # Selection strategy implementations
│   │   ├── parallel/           #   BoN, Sharpe, SoftBoN, MoB, WBoN
│   │   ├── sequential/         #   SeqBoN search
│   │   └── adaptive/           #   Controller policies
│   ├── generation/             # LLM candidate generation
│   ├── reward_modeling/        # Ensemble RM scoring
│   └── data_utils/             # Dataset loaders
├── experiments/                # CLI entry points
│   ├── run_hypothesis.py       #   Run a single hypothesis
│   └── run_all.py              #   Run all hypotheses
├── scripts/                    # Data pipeline (generate → score)
│   ├── generate_candidates_vllm.py
│   └── score_candidates.py
├── runs/                       # Experiment outputs
│   ├── parallel/               #   Parallel hypothesis results
│   │   └── <experiment_name>/
│   │       ├── config.json
│   │       ├── results.json
│   │       ├── summary.txt
│   │       └── plots/
│   └── sequential/             #   Sequential hypothesis results
├── configs/experiment.yaml
└── tests/test_all.py
```

## Quick Start

```bash
# Setup
pip install -e .

# Step 1: Generate candidates (needs GPU)
python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset gsm8k --n-candidates 24

# Step 2: Score candidates with RMs
python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "OpenAssistant/reward-model-deberta-v3-large-v2"

# Step 3: Run hypotheses (offline, no GPU needed)
python experiments/run_hypothesis.py --list                    # see options
python experiments/run_hypothesis.py -H h5 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

# Run all hypotheses at once
python experiments/run_all.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
```

## Hypotheses

| ID  | Name | Description |
|-----|------|-------------|
| h1  | Reward Hacking at Scale | Vanilla BoN plateaus/drops at large N; uncertainty-aware methods don't |
| h2  | Ensemble Diversity | Pairwise RM correlation + effective ensemble size analysis |
| h3  | Variance Normalization | Robust normalization is critical for heterogeneous RM ensembles |
| h4  | Disagreement Hybrid | Disagreement-gated policy switches methods based on RM agreement |
| h5  | Offline Methods | Comprehensive comparison of all parallel selection methods |
| h6  | Self-Consistency | Majority vote + RM hybrid selection |
| h7  | RM Debiasing | Length-conditional scoring + per-prompt normalization |
| h8  | Rank-Based Selection | Rank aggregation and pessimistic BoN |
| h9  | Difficulty-Stratified | Stratified analysis by prompt difficulty |
| h10 | Pairwise Tournament | Bradley-Terry + tournament-style selection |

## Methods

| Category | Method | Selection Rule |
|----------|--------|---------------|
| Parallel | Vanilla BoN | argmax μ |
| Parallel | Sharpe BoN | argmax μ/σ |
| Parallel | Linear BoN | argmax (μ − c·σ) |
| Parallel | SoftBoN variants | softmax over utilities |
| Parallel | MoB | Bootstrap majority-of-bests |
| Parallel | WBoN | Inverse-variance weighted aggregation |
| Sequential | Vanilla SeqBoN | accept if r ≥ γ |
| Sequential | Sharpe SeqBoN | accept if μ/σ ≥ γ |
| Routing | Hybrid | disagreement-gated switching |
