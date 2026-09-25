#!/bin/bash
export PYTHONPATH=$PWD
if [ -d "venv" ]; then
    source venv/bin/activate
fi

MODELS=("qwen2.5-3b-instruct")
DATASETS=("math500" "gsm8k" "mmlu" "cruxeval")
REWARD_MODELS=("FsfairX-LLaMA3-RM-v0.1,Llama-3-OffsetBias-RM-8B,reward-model-deberta-v3-large-v2")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for rms in "${REWARD_MODELS[@]}"; do
            echo "Running burn-in ablation for $model on $dataset..."
            python src/experiments/ablations/run_ablation_burnin.py \
                --model "$model" \
                --output-model "$model" \
                --reward-models "$rms" \
                --dataset "$dataset"
        done
    done
done
