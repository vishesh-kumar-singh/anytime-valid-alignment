#!/bin/bash
export PYTHONPATH=$PWD
source venv/bin/activate

model_short="qwen2.5-3b-instruct"
dataset="math500"
RM_LIST="ArmoRM-Llama3-8B-v0.1"
output_model="armorm-${model_short}/c_sweep_${dataset}/b64"

echo "Running c sweep for $model_short on $dataset..."

python scripts/run_all_pipelines.py \
    --pipelines "e_rwsc" \
    --model "$model_short" \
    --output-model "$output_model" \
    --reward-models "$RM_LIST" \
    --dataset "$dataset" \
    --candidates-n 512 \
    --eval-n 64 \
    --p-dagger-values "0.5" \
    --alpha-values "0.1" \
    --c-values "0.5,0.6,0.7,0.8,0.9,0.99"

echo "c sweep complete!"
