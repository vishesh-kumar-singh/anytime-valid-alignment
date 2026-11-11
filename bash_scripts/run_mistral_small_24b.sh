#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$PWD/venv/bin:$PATH
export PYTHONPATH=$PWD
source venv/bin/activate

MODEL_HF="casperhansen/mistral-small-24b-instruct-2501-awq"
MODEL_SHORT="mistral-small-24b-instruct-2501-awq"
DATASET="math500"
CANDIDATES_FILE="data/candidates/${MODEL_SHORT}/${DATASET}_n24_full_vllm.jsonl"

echo "=========================================================="
echo " Starting Mistral-Small-24B (AWQ) on GPU 3"
echo " Model:     $MODEL_HF"
echo " Dataset:   $DATASET"
echo " Output:    $CANDIDATES_FILE"
echo "=========================================================="

echo "=== [1/3] Generating Candidates with vLLM on GPU 3 ==="
python3 -u scripts/generate_candidates_vllm.py \
    --model-name "$MODEL_HF" \
    --dataset "$DATASET" \
    --n-candidates 24 \
    --max-examples 0 \
    --max-new-tokens 2048 \
    --max-model-len 4096 \
    --num-gpus 1 \
    --gpu-memory-utilization 0.85

echo "=== [2/3] Scoring Candidates with ArmoRM-Llama3-8B-v0.1 on GPU 3 ==="
python3 -u scripts/score_candidates.py \
    --candidates-path "$CANDIDATES_FILE" \
    --reward-model "RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --batch-size 4

echo "=== [3/3] Running Evaluation Pipelines ==="
python3 -u scripts/run_all_pipelines.py \
    --pipelines "baselines,seq_baselines,rwsc,sprt_rwsc,e_rwsc" \
    --model "$MODEL_SHORT" \
    --reward-models "ArmoRM-Llama3-8B-v0.1" \
    --dataset "$DATASET" \
    --eval-n 24

echo "=== ALL PIPELINES FINISHED SUCCESSFULLY ==="
