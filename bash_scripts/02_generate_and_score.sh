#!/bin/bash
# 02_generate_and_score.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$PWD
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Running with system python."
fi

SPECIFIC_DATASET=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) SPECIFIC_DATASET="$2"; shift ;;
    esac
    shift
done

MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct" "microsoft/Phi-3-mini-4k-instruct")
if [ -n "$SPECIFIC_DATASET" ]; then
    DATASETS=("$SPECIFIC_DATASET")
else
    DATASETS=("gsm8k" "math500" "mmlu" "cruxeval")
fi

REWARD_MODELS=("RLHFlow/ArmoRM-Llama3-8B-v0.1" "models/Skywork-Reward-Llama-3.1-8B-v0.2" "sfairXC/FsfairX-LLaMA3-RM-v0.1" "NCSOFT/Llama-3-OffsetBias-RM-8B" "OpenAssistant/reward-model-deberta-v3-large-v2")

get_model_shortname() { echo "$1" | awk -F/ '{print tolower($NF)}'; }
get_rm_shortname() { echo "$1" | awk -F/ '{print $NF}'; }

echo "Starting Generation and Scoring..."
for model_hf in "${MODELS[@]}"; do
    model_short=$(get_model_shortname "$model_hf")
    for dataset in "${DATASETS[@]}"; do
        candidates_file="data/candidates/${model_short}/${dataset}_n24_full_vllm.jsonl"
        
        # Step 1: Generate Candidates
        if [ ! -f "$candidates_file" ]; then
            echo "Generating Candidates for $model_hf on $dataset..."
            python scripts/generate_candidates_vllm.py \
                --model-name "$model_hf" --dataset "$dataset" \
                --n-candidates 24 --max-examples 0 --max-new-tokens 2048 --num-gpus 1
        else
            echo "Candidates already exist for $model_hf on $dataset. Skipping..."
        fi
        
        # Step 1.5: Extract W-CITE scores
        wcite_score_file="data/scores/${model_short}/${dataset}_n24_full_vllm_scores_wcite.jsonl"
        if [ ! -f "$wcite_score_file" ]; then
            echo "Extracting W-CITE scores for $model_hf on $dataset..."
            python scripts/extract_logprobs_from_existing.py --dataset "$dataset" --candidates-path "$candidates_file" --gpu-memory-utilization 0.5
        else
            echo "W-CITE scores already exist for $model_hf on $dataset. Skipping..."
        fi
        
        # Step 2: Score Candidates
        for rm_hf in "${REWARD_MODELS[@]}"; do
            rm_short=$(get_rm_shortname "$rm_hf")
            score_file="data/scores/${model_short}/${dataset}_n24_full_vllm_scores_${rm_short}.jsonl"
            if [ ! -f "$score_file" ]; then
                echo "Scoring with $rm_short..."
                python scripts/score_candidates.py \
                    --candidates-path "$candidates_file" --reward-model "$rm_hf"
            else
                echo "Scores already exist for $rm_short. Skipping..."
            fi
        done
    done
done
echo "Generation and Scoring Complete!"
