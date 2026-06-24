# Test-Time Alignment with Uncertainty-Aware Ensemble Methods

Research on improving LLM inference-time alignment using reward model ensembles, uncertainty estimation, and mathematically rigorous sequential early stopping.

**Advisor:** Prof. Sayak Ray Chowdhury, IIT Kanpur

## Background & Key Findings

This project explores scaling test-time compute for Large Language Models (LLMs) by utilizing Reward Model (RM) ensembles. By moving beyond naive Best-of-N (BoN), we developed novel parallel and sequential algorithms that establish a new Pareto frontier on complex reasoning benchmarks (MATH500, GSM8K, MMLU).

**Key Contributions:**
1. **Top-K MBR-RWSC:** A parallel algorithm that merges Minimum Bayes Risk (MBR) decoding with Reward-Weighted Self-Consistency (RWSC). By filtering out low-quality "tail" candidates before applying topological consensus, we prevent consensus dilution and achieve state-of-the-art accuracy.
2. **SPRT-RWSC:** A sequential early stopping framework based on Wald's Sequential Probability Ratio Test (SPRT). By mathematically bounding Type-I/Type-II errors, it guarantees near-optimal accuracy while dynamically halting generation the moment statistical convergence is achieved.
3. **E-SPRT (Ensemble SPRT):** By scaling the SPRT evidence thresholds dynamically using the disagreement (variance) of the RM ensemble, E-SPRT dramatically slashes test-time compute requirements while maintaining strict error bounds.

### Pareto Frontier on MATH500 (N=24)
- **Top-13 MBR-RWSC:** 68.9% Accuracy (100% compute budget, $N=24$)
- **Vanilla BoN:** ~61.8% Accuracy (100% compute budget, $N=24$)
- **E-SPRT (α=0.1, β=0.2):** 68.9% Accuracy (**36% compute budget, Avg N=8.70**)

*E-SPRT achieves the exact same state-of-the-art accuracy as the best parallel method, but uses almost 3x less compute!*

## Repository Structure

```text
├── src/
│   ├── core/                   # Shared infrastructure (scoring, datasets)
│   ├── hypotheses/             # One file per experiment
│   │   ├── h6_mbr_ensemble_aggregation.py # MBR Consensus
│   │   ├── h8_top_k_mbr_rwsc.py           # Top-K MBR-RWSC (Parallel)
│   │   ├── h9_sprt_rwsc.py                # SPRT / E-SPRT (Sequential)
│   │   ├── h10_seq_baselines.py           # Literature Baselines (Pandora/BEACON)
│   ├── data_utils/             # Dataset loaders (GSM8K, MATH500, MMLU, Alpaca)
├── experiments/                # CLI entry points
│   ├── run_hypothesis.py       # Run a single hypothesis
│   └── run_all.py              # Run all hypotheses
├── bash_scripts/               # Orchestration scripts for vLLM generation
├── scripts/                    # Data pipeline (generate → score)
│   ├── generate_candidates_vllm.py
│   └── score_candidates.py
└── rough.tex                   # Academic manuscript draft (ICML/ICLR formatting)
```

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install the project and core dependencies
pip install -e .

# (Optional) For generating candidates, install vLLM:
pip install vllm
```

### 2. Generate and Score Candidates
We use vLLM for high-throughput generation and HuggingFace for RM scoring. See `bash_scripts/` for automated orchestration.
```bash
python scripts/generate_candidates_vllm.py --model-name "Qwen/Qwen2.5-3B-Instruct" --dataset math500 --n-candidates 24
python scripts/score_candidates.py --candidates-path data/candidates/math500_n24_full_vllm.jsonl --reward-model "sfairXC/FsfairX-LLaMA3-RM-v0.1"
```

### 3. Run Experiments
Evaluate the pre-scored candidates using our proposed alignment algorithms:
```bash
# See all available hypotheses
python experiments/run_hypothesis.py --list

# Run E-SPRT Sequential Early Stopping on MATH500
python experiments/run_all.py \
    --hypotheses h9 \
    --candidates-path data/candidates/math500_n24_full_vllm.jsonl \
    --score-paths data/scores/math500_n24_full_vllm_scores_*.jsonl
```
