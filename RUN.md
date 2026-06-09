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

## Alignment track (open-ended, LLM-judge win-rate)

The math track above measures **RM selection quality** on verifiable tasks. The
alignment track measures **preference quality** on open-ended instructions
(AlpacaEval), where the gold signal is a **Gemini-judged win-rate vs the
reference** (the arm the URM paper used alongside math BoN). Same selection
methods, different gold metric. The headline is the **reward-overoptimization
curve**: as N grows, proxy reward rises but does win-rate rise-then-*fall* for
vanilla BoN, while uncertainty-aware / pessimistic BoN (Sharpe, μ−β·σ) holds up?

### One-time setup

```bash
# 1. Judge keys — put 1-3 free Gemini keys (different accounts) in .env:
#    GEMINI_API_KEY=...
#    GEMINI_API_KEY_2=...        # optional, pools quota (round-robin + failover)
#    GEMINI_API_KEY_3=...
#    GEMINI_JUDGE_MODEL=gemini-flash-lite-latest   # 500 req/day on free tier
# 2. Deps (already installed in `langchain`): google-genai, datasets
# 3. Smoke-test the judge (2 live calls):
conda run -n langchain python -m src.eval.llm_judge --smoke
```

### Step A1: Generate AlpacaEval candidates (GPU, run once)

```bash
conda run -n langchain python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset alpaca_eval \
    --n-candidates 16 \
    --max-examples 150 \
    --max-new-tokens 1024
# -> data/candidates/alpaca_eval_n16_150ex_vllm.jsonl
```

### Step A2: Score with a SMALL RM ensemble (GPU, run once per RM)

Use **≥2 RMs** — uncertainty methods (Sharpe / pessimism) need ensemble
disagreement (variance); with 1 RM they collapse to vanilla BoN. The DeBERTa RM
is small enough to run on a modest GPU.

```bash
conda run -n langchain python scripts/score_candidates.py \
    --candidates-path data/candidates/alpaca_eval_n16_150ex_vllm.jsonl \
    --reward-model "OpenAssistant/reward-model-deberta-v3-large-v2" --load-in-4bit
# repeat with a second (and third) RM, e.g. Skywork / ArmoRM (8B, needs more VRAM)
# -> data/scores/alpaca_eval_n16_150ex_vllm_scores_<rm>.jsonl
```

### Step A3: Alignment eval — judge win-rate + overoptimization (CPU + judge API)

```bash
conda run -n langchain python experiments/run_alignment_eval.py \
    --candidates-path data/candidates/alpaca_eval_n16_150ex_vllm.jsonl \
    --score-paths "data/scores/alpaca_eval_n16_150ex_vllm_scores_*.jsonl" \
    --eval-n 1,4,8,16 \
    --beta 1.0 \
    --max-judge-calls 1200       # safety cap on LIVE judge calls per run
```

**Outputs** → `runs/alignment/<candidates-stem>/`:
`results.json`, `summary.txt`, `judgments.jsonl` (every verdict, streamed),
`run.log` (full run log + judge call counts + any errors).

**Quota & resume:** verdicts are disk-cached (`data/cache/judge_cache.pkl`), so
only *new* (instruction, response, reference) triples cost quota and **every
rerun is free**. If keys hit their daily cap mid-run it saves partial results
and marks `status=incomplete:quota` — just rerun later to continue off the
cache. Keys are round-robined; a blocked/denied key is parked and skipped.

> **Offline analyses (no GPU, no API):** `experiments/analyze_now.py` (gap
> decomposition + adaptive stopping) and `experiments/analyze_difficulty.py`
> (MATH500 by level/subject) run on the cached data and write to
> `runs/analysis/`.

---

## Troubleshooting

```bash
# Clear corrupted model cache
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct

# Verify installation
conda run -n langchain python -c "from src.hypotheses import list_hypotheses; print(list_hypotheses())"
```
