#!/bin/bash
# 06_run_budget_sweep.sh

export PYTHONPATH=$PWD
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Running with system python."
fi

MODELS=("Qwen/Qwen2.5-3B-Instruct")
DATASETS=("gsm8k" "math500" "mmlu" "cruxeval")
REWARD_MODELS=("RLHFlow/ArmoRM-Llama3-8B-v0.1")
BUDGETS=(24 32 64 128 256 512)

get_model_shortname() { echo "$1" | awk -F/ '{print tolower($NF)}'; }
get_rm_shortname() { echo "$1" | awk -F/ '{print $NF}'; }

echo "Starting Budget Sweep Experiments..."
for model_hf in "${MODELS[@]}"; do
    model_short=$(get_model_shortname "$model_hf")
    
    RM_SHORTNAMES=()
    for rm_hf in "${REWARD_MODELS[@]}"; do RM_SHORTNAMES+=($(get_rm_shortname "$rm_hf")); done
    RM_LIST=$(IFS=,; echo "${RM_SHORTNAMES[*]}")
    
    for dataset in "${DATASETS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            echo "Running budget sweep for $model_short on $dataset with budget $budget..."
            
            output_model="strong-${model_short}/budget_sweep_${dataset}/b${budget}"
            
            python scripts/run_all_pipelines.py \
                --pipelines "baselines,rwsc,seq_baselines,sprt_rwsc,e_rwsc" \
                --model "$model_short" \
                --output-model "$output_model" \
                --reward-models "$RM_LIST" \
                --dataset "$dataset" \
                --candidates-n 512 \
                --eval-n "$budget"
        done
    done
done
echo "Budget Sweep Complete!"
