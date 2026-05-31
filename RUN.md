# Test-Time Alignment: Run Guide

All experiment outputs are saved to `runs/parallel/<experiment_name>/` with `results.json`, `summary.txt`, and `plots/`. Note that directories are not timestamped; re-running an experiment will cleanly overwrite the previous run.

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

Every hypothesis automatically generates N-scaling results (N=4, 8, 16, 24).

### List all available hypotheses

```bash
conda run -n langchain python experiments/run_hypothesis.py --list
```

### Run a single hypothesis

```bash
# H1: Uncertainty-Aware BoN
# (Automatically generates Accuracy vs N plots!)
conda run -n langchain python experiments/run_hypothesis.py -H h1 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 4,8,16,24

# H2: Ensemble Aggregation
conda run -n langchain python experiments/run_hypothesis.py -H h2 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 4,8,16,24

# H3: Self-Consistency
conda run -n langchain python experiments/run_hypothesis.py -H h3 \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 4,8,16,24
```

### Run all hypotheses at once

```bash
conda run -n langchain python experiments/run_all.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 4,8,16,24
```

---

## Output Structure

Each hypothesis run creates a non-timestamped directory so re-runs stay clean:

```
runs/parallel/h1_uncertainty_aware_gsm8k/
├── results.json       # Structured metrics (machine-readable)
├── summary.txt        # Human-readable summary table
└── plots/
    └── accuracy_vs_n.png
```

---

## Hypothesis Reference

| ID  | Name | What it tests |
|-----|------|---------------|
| h1  | Uncertainty-Aware BoN | Sharpe, Linear, and SoftBoN variants vs Vanilla BoN |
| h2  | Ensemble Aggregation | MoB and WBoN variants |
| h3  | Self-Consistency | Majority Vote, RM-within-Majority, RM-Consistency-Agree |
| h4  | Reward-Weighted Self-Consistency | Answer-level BoN: sum RM mass per unique answer |
| h5  | Single-Best-RM RWSC | RWSC weighted by the single strongest RM (best parallel method) |

> **Grading:** MATH500 is scored with `math_verify` (cached to
> `data/cache/math_grade_cache.pkl`). The first run on a new candidate set
> populates the cache (~1 min for 12k answers); subsequent runs are instant.
> Install with `pip install math-verify latex2sympy2-extended`.

---

## Troubleshooting

```bash
# Clear corrupted model cache
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct

# Verify installation
conda run -n langchain python -c "from src.hypotheses import list_hypotheses; print(list_hypotheses())"
```
