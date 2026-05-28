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
├── src/                        # Core source code
│   ├── generation/             # LLM candidate generation (Qwen2.5-7B)
│   ├── reward_modeling/        # Ensemble RM scoring (trivial + robust URM)
│   ├── policies/               # Selection strategies (BoN, Sharpe, SoftBoN, etc.)
│   ├── sequential/             # RF-SeqBoN sequential filtering
│   ├── data_utils/             # Dataset loaders (GSM8K, MATH500, AlpacaEval)
│   └── experiments/            # Main experiment pipeline
├── configs/experiment.yaml     # Hyperparameters
├── scripts/                    # Analysis & plotting
├── runs/                       # Experiment outputs (organized by experiment)
│   └── <experiment_name>/      # Each experiment gets its own subfolder
├── tests/                      # Unit tests
├── models/                     # Trained models
├── noisy-reward-alignment.pdf  # Prior UGP report (reference)
└── run_demo.py                 # Quick demo
```

## Setup

```bash
conda activate ugp
pip install -e .
python -m pytest tests/test_all.py -v  # verify installation
python run_demo.py                      # quick test
```

## Current Methods (Inherited)

| Category | Method | Selection Rule |
|----------|--------|---------------|
| Parallel | Vanilla BoN | argmax μ |
| Parallel | Sharpe BoN | argmax μ/σ |
| Parallel | Linear BoN | argmax (μ − c·σ) |
| Parallel | SoftBoN variants | softmax over utilities |
| Sequential | Vanilla SeqBoN | accept if r ≥ γ |
| Sequential | Sharpe SeqBoN | accept if μ/σ ≥ γ |
| Routing | Hybrid | disagreement-gated switching |
