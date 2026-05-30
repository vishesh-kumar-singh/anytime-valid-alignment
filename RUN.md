# Test-Time Alignment: Run Guide

All experiment outputs are saved to `runs/parallel/<experiment_name>/` with `results.json`, `summary.txt`, and `plots/`.

---

## 🚀 Pipeline Overview

```
Step 1: Generate candidates (GPU, run once)  →  data/candidates/*.jsonl
Step 2: Score with each RM   (GPU, run once)  →  data/scores/*.jsonl
Step 3: Run hypotheses       (CPU, instant)   →  runs/parallel/<name>/
```

---

## Step 1: Generate Candidates (GPU, run once)

Generates 24 responses per question using vLLM (10–30x faster than HuggingFace).

```bash
# GSM8K (1,319 examples)
conda run -n langchain python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset gsm8k \
    --n-candidates 24 \
    --max-examples 0

# MATH500 (500 examples)
conda run -n langchain python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset math500 \
    --n-candidates 24 \
    --max-examples 0
```

**Output:** `data/candidates/gsm8k_n24_full_vllm.jsonl`, `data/candidates/math500_n24_full_vllm.jsonl`

---

## Step 2: Score with Each RM (GPU, run once per RM)

Score all candidates with one RM at a time.

```bash
# RM 1: DeBERTa (small, fast baseline)
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "OpenAssistant/reward-model-deberta-v3-large-v2" \
    --load-in-4bit

# RM 2: OffsetBias (Robust Llama-3-8B)
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "NCSOFT/Llama-3-OffsetBias-RM-8B" \
    --load-in-4bit

# RM 3: ArmoRM (Mixture of Experts Llama-3-8B)
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --load-in-4bit

# RM 4: Skywork Reward (Llama-3.1-8B)
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2" \
    --load-in-4bit

# RM 5: FsfairX (Llama-3-8B)
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --load-in-4bit
```

**Output:** `data/scores/gsm8k_n24_full_vllm_scores_<rm_name>.jsonl` (one per RM)

---

## Step 3: Run Hypotheses (NO GPU, instant)

### List all available hypotheses

```bash
conda run -n langchain python experiments/run_hypothesis.py --list
```

### Run a single hypothesis

```bash
# H1: Reward hacking at scale (N-sweep analysis)
# (Automatically generates Accuracy, Reward Hacking Proxy, and Variance Gap plots!)
conda run -n langchain python experiments/run_hypothesis.py -H h1 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 1,2,4,6,8,12,16,20,24

# H2: Ensemble diversity (correlation + effective size)
conda run -n langchain python experiments/run_hypothesis.py -H h2 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

# H3: Variance normalization comparison
conda run -n langchain python experiments/run_hypothesis.py -H h3 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"

# H5: Full offline method comparison
conda run -n langchain python experiments/run_hypothesis.py -H h5 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 1,4,8,16,24

# H6: Self-consistency + RM hybrid
conda run -n langchain python experiments/run_hypothesis.py -H h6 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
```

### Run all hypotheses at once

```bash
conda run -n langchain python experiments/run_all.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 1,4,8,16,24
```

### Run a subset

```bash
conda run -n langchain python experiments/run_all.py \
    --hypotheses h1,h3,h5 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl"
```

### With custom normalization

```bash
conda run -n langchain python experiments/run_hypothesis.py -H h5 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --variance-norm-mode zscore
```

---

## Output Structure

Each hypothesis run creates:

```
runs/parallel/h5_offline_parallel_methods_gsm8k_20260530_120000/
├── results.json       # Structured metrics (machine-readable)
├── summary.txt        # Human-readable summary table
└── plots/
    ├── accuracy_vs_n.png
    └── method_comparison_bar.png
```

---

## Hypothesis Reference

| ID  | Name | What it tests |
|-----|------|---------------|
| h1  | Reward Hacking at Scale | Vanilla BoN plateaus at large N; Sharpe/Linear don't |
| h2  | Ensemble Diversity | RM correlation heatmap, effective ensemble size |
| h3  | Variance Normalization | zscore vs robust_zscore calibration |
| h4  | Disagreement Hybrid | Switching policies based on RM disagreement |
| h5  | Offline Methods | All parallel selection methods comparison |
| h6  | Self-Consistency | Majority vote + RM hybrid scoring |
| h7  | RM Debiasing | Length-conditional scoring, per-prompt normalization |
| h8  | Rank-Based Selection | Rank aggregation + pessimistic BoN (β sweep) |
| h9  | Difficulty-Stratified | Easy/medium/hard stratified analysis |
| h10 | Pairwise Tournament | Bradley-Terry + tournament-style selection |

---

## Troubleshooting

```bash
# Clear corrupted model cache
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct

# Verify installation
conda run -n langchain python -c "from src.hypotheses import list_hypotheses; print(list_hypotheses())"
```
