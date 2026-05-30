# Test-Time Alignment: Run Scripts

All experiments automatically log to `logs/` with timestamped files.
Monitor live: `tail -f logs/*.log`

---

## 🚀 Strategy A: Modular Pipeline (RECOMMENDED)

### Step 1: Generate Candidates (GPU, run once)

Generates 24 responses per question for the **FULL** datasets using a 7B model. 
We use **vLLM** for this step because it batches everything and generates 10x-30x faster.

```bash
# Generate for full GSM8K (1,319 examples)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset gsm8k \
    --n-candidates 24 \
    --max-examples 0

# Generate for full MATH500 (500 examples)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/generate_candidates_vllm.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset math500 \
    --n-candidates 24 \
    --max-examples 0
```

Outputs: 
- `data/candidates/gsm8k_n24_full_vllm.jsonl`
- `data/candidates/math500_n24_full_vllm.jsonl`

### Step 2: Score with Each RM Separately (GPU, run once per RM)

Score all candidates with one RM at a time. Each RM produces its own file.
Run these commands for whichever dataset you generated above (change `--candidates-path`).

```bash
# RM 1: DeBERTa (small, fast baseline)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "OpenAssistant/reward-model-deberta-v3-large-v2" \
    --load-in-4bit

# RM 2: OffsetBias (Robust Llama-3-8B)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "NCSOFT/Llama-3-OffsetBias-RM-8B" \
    --load-in-4bit

# RM 3: ArmoRM (Mixture of Experts Llama-3-8B)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --load-in-4bit

# RM 4: Skywork Reward (State-of-the-art Llama-3.1-8B)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2" \
    --load-in-4bit

# RM 5: FsfairX (Llama-3-8B)
PYTHONUNBUFFERED=1 conda run -n cloudspace python scripts/score_candidates.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --reward-model "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --load-in-4bit
```

Output: `data/scores/gsm8k_n24_full_vllm_scores_<rm_name>.jsonl` (one per RM)

### Step 3: Evaluate Offline (NO GPU, instant)

Mix & match any RMs, sweep any N, tweak any hyperparameter — all for free.

```bash
# Use 2 RMs, sweep N values
python scripts/evaluate_offline.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_reward-model-deberta-v3-large-v2.jsonl,data/scores/gsm8k_n24_full_vllm_scores_Llama-3-OffsetBias-RM-8B.jsonl" \
    --eval-n 1,4,8,16,24

# Use ALL score files at once (glob pattern)
python scripts/evaluate_offline.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 1,4,8,16,24

# Try different normalization
python scripts/evaluate_offline.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 8 \
    --variance-norm zscore

# Tweak linear penalty
python scripts/evaluate_offline.py \
    --candidates-path data/candidates/gsm8k_n24_full_vllm.jsonl \
    --score-paths "data/scores/gsm8k_n24_full_vllm_scores_*.jsonl" \
    --eval-n 8 \
    --lin-c 2.0
```

---

## Strategy B: Direction 3 — Sequential Methods (Ideas Suite)

Tests Adaptive SeqBoN (cannot be done offline — requires live generation).

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n cloudspace python -m src.experiments.run_risk_aware_inference \
    ideas-suite \
    --include-seq \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset gsm8k \
    --max-examples 200 \
    --load-in-4bit \
    --reward-load-in-4bit \
    --reward-model-names "OpenAssistant/reward-model-deberta-v3-large-v2,NCSOFT/Llama-3-OffsetBias-RM-8B"
```

---

## Before First Run (if cache is corrupted)

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
```
