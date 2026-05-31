# Test-Time Alignment with Uncertainty-Aware Ensemble Methods

Research on improving LLM inference-time alignment using reward model ensembles and uncertainty estimation.

**Advisor:** Prof. Sayak Ray Chowdhury, IIT Kanpur

## Background

This project continues from prior UGP work on uncertainty-guided policies for test-time alignment. The prior work showed that:

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
│   ├── hypotheses/             # One file per experiment
│   │   ├── base.py             #   Abstract Hypothesis class
│   │   ├── h1_uncertainty_aware.py
│   │   ├── h2_ensemble_aggregation.py
│   │   ├── h3_self_consistency.py
│   │   ├── h4_reward_weighted_self_consistency.py
│   │   └── h5_single_best_rm_rwsc.py
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

# Step 3: Run experiments (offline, no GPU needed)
python experiments/run_hypothesis.py --list                    # see options
python experiments/run_hypothesis.py -H h1 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

# Run all experiments at once
python experiments/run_all.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
```

## Experiments

Correctness is graded with **`math_verify`** (the HF/Qwen/DeepSeek standard);
the legacy substring grader inflated MATH500 accuracy by ~12 points and is no
longer used. See `FUTURE_WORK.md` for why parallel selection saturates here.

| ID  | Name | Description |
|-----|------|-------------|
| h1  | Uncertainty-Aware BoN | Sharpe, Linear, SoftBoN variants vs vanilla argmax |
| h2  | Ensemble Aggregation | Mixture-of-Bests (MoB) and Weighted BoN (WBoN) |
| h3  | Self-Consistency | Majority vote, RM-within-majority, RM-consistency-agree |
| h4  | Reward-Weighted Self-Consistency | Answer-level BoN: aggregate RM mass per answer |
| h5  | Single-Best-RM RWSC | Drop the diluting ensemble; weight by the single strongest RM (best parallel method, ~70.6%) |

## Methods

| Category | Method | Selection Rule |
|----------|--------|----------------|
| Parallel | Vanilla BoN | argmax μ |
| Parallel | Sharpe BoN | argmax μ/σ |
| Parallel | Linear BoN | argmax (μ − c·σ) |
| Parallel | SoftBoN variants | softmax over utilities |
| Parallel | MoB | Bootstrap majority-of-bests |
| Parallel | WBoN | Inverse-variance weighted aggregation |
| Consensus | Majority Vote | Pure answer-level consistency |
| Consensus | RM-within-Majority | RM-guided selection within majority cluster |
| Sequential | Vanilla SeqBoN | accept if r ≥ γ |
| Sequential | Sharpe SeqBoN | accept if μ/σ ≥ γ |
