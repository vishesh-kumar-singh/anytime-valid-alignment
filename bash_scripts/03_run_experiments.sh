#!/bin/bash
# 03_run_experiments.sh

export PYTHONPATH=$PWD
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Running with system python."
fi

MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct" "microsoft/Phi-3-mini-4k-instruct")
DATASETS=("gsm8k" "math500" "mmlu" "cruxeval")
REWARD_MODELS=("RLHFlow/ArmoRM-Llama3-8B-v0.1" "models/Skywork-Reward-Llama-3.1-8B-v0.2" "sfairXC/FsfairX-LLaMA3-RM-v0.1" "NCSOFT/Llama-3-OffsetBias-RM-8B" "OpenAssistant/reward-model-deberta-v3-large-v2" "wcite")

get_model_shortname() { echo "$1" | awk -F/ '{print tolower($NF)}'; }
get_rm_shortname() { echo "$1" | awk -F/ '{print $NF}'; }

echo "Starting Experiments..."
for model_hf in "${MODELS[@]}"; do
    model_short=$(get_model_shortname "$model_hf")
    output_model="armorm-${model_short}"
    
    RM_SHORTNAMES=()
    for rm_hf in "${REWARD_MODELS[@]}"; do RM_SHORTNAMES+=($(get_rm_shortname "$rm_hf")); done
    RM_LIST=$(IFS=,; echo "${RM_SHORTNAMES[*]}")
    
    for dataset in "${DATASETS[@]}"; do
        echo "Running evaluations for $model_short on $dataset..."
        
        # Step 3: Run Master Baselines & Core Pipelines
        python scripts/run_all_pipelines.py \
            --pipelines "baselines,seq_baselines,rwsc,sprt_rwsc,e_rwsc" \
            --model "$model_short" --output-model "$output_model" \
            --reward-models "$RM_LIST" --dataset "$dataset"
    done
done
echo "Experiments Complete!"
