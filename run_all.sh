#!/bin/bash
set -e

# Datasets to evaluate
DATASETS=("gsm8k" "math500" "mmlu")

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "============================================="
echo "Clearing old results..."
rm -rf runs/*
echo "============================================="

for d in "${DATASETS[@]}"; do
    echo ""
    echo "============================================="
    echo "Running all pipelines on dataset: $d"
    echo "============================================="
    
    python experiments/run_all_pipelines.py \
        --candidates-path "data/candidates/${d}_n24_full_vllm.jsonl" \
        --score-paths "data/scores/${d}_n24_full_vllm_scores_*.jsonl"
done

echo ""
echo "Done! All results saved in runs/"
