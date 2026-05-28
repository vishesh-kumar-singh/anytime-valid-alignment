#!/usr/bin/env bash
# ===========================================================================
# Direction 1 — N-Sweep: Reward Hacking Mitigation at Scale
# ===========================================================================
#
# Sweeps N ∈ {4, 8, 16, 32, 64, 128} for both GSM8K and MATH500, running
# Vanilla BoN, BoN-Sharpe, BoN-Linear (and their Soft variants) via the
# main CLI with --bon-only, then runs per-N reward-hacking analysis.
#
# Usage:
#   bash scripts/sweep_bon_n.sh                    # full sweep
#   MAX_EXAMPLES=50 bash scripts/sweep_bon_n.sh    # quick subset
#   N_VALUES="4 8" DATASETS="gsm8k" bash scripts/sweep_bon_n.sh  # mini test
#
# Environment variables (all optional):
#   N_VALUES        — space-separated N values  (default: "4 8 16 32 64 128")
#   DATASETS        — space-separated datasets  (default: "gsm8k math500")
#   MAX_EXAMPLES    — per-dataset cap (0 = full) (default: 0)
#   CONDA_ENV       — conda env name            (default: "langchain")
#   GEN_DEVICE      — GPU index                 (default: 0)
#   LOAD_4BIT       — "true" to 4-bit quant gen (default: "false")
#   REWARD_4BIT     — "true" to 4-bit quant RMs (default: "true")
#   OUT_DIR         — root output directory      (default: "runs/direction1_n_sweep")
#   VARIANCE_NORM   — normalization mode         (default: "robust_zscore")
#   LIN_C           — c coefficient for Linear   (default: "1.0")
#   REWARD_MODELS   — comma-sep RM names         (default: 3-RM preset)
# ===========================================================================

set -euo pipefail

# ---- Configuration (with sensible defaults) ----
N_VALUES="${N_VALUES:-4 8 16 32 64 128}"
DATASETS="${DATASETS:-gsm8k math500}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
CONDA_ENV="${CONDA_ENV:-langchain}"
GEN_DEVICE="${GEN_DEVICE:-0}"
LOAD_4BIT="${LOAD_4BIT:-false}"
REWARD_4BIT="${REWARD_4BIT:-true}"
OUT_DIR="${OUT_DIR:-runs/direction1_n_sweep}"
VARIANCE_NORM="${VARIANCE_NORM:-robust_zscore}"
LIN_C="${LIN_C:-1.0}"
REWARD_MODELS="${REWARD_MODELS:-OpenAssistant/reward-model-deberta-v3-large-v2,NCSOFT/Llama-3-OffsetBias-RM-8B,RLHFlow/ArmoRM-Llama3-8B-v0.1}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUT_DIR"

# ---- Build CLI flags ----
GEN_4BIT_FLAG=""
if [[ "$LOAD_4BIT" == "true" ]]; then
    GEN_4BIT_FLAG="--load-in-4bit"
fi

REWARD_4BIT_FLAG=""
if [[ "$REWARD_4BIT" == "true" ]]; then
    REWARD_4BIT_FLAG="--reward-load-in-4bit"
else
    REWARD_4BIT_FLAG="--no-reward-load-in-4bit"
fi

MAX_EX_FLAG="--max-examples ${MAX_EXAMPLES}"

echo "================================================================"
echo "  Direction 1 — N-Sweep: Reward Hacking Mitigation at Scale"
echo "================================================================"
echo "  N values:       ${N_VALUES}"
echo "  Datasets:       ${DATASETS}"
echo "  Max examples:   ${MAX_EXAMPLES} (0 = full)"
echo "  Conda env:      ${CONDA_ENV}"
echo "  Output dir:     ${OUT_DIR}"
echo "  Variance norm:  ${VARIANCE_NORM}"
echo "  Lin c:          ${LIN_C}"
echo "  Reward models:  ${REWARD_MODELS}"
echo "  Gen 4-bit:      ${LOAD_4BIT}"
echo "  RM 4-bit:       ${REWARD_4BIT}"
echo "================================================================"
echo ""

TOTAL_RUNS=0
for N in $N_VALUES; do
    for DS in $DATASETS; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
done
echo "Total runs: ${TOTAL_RUNS}"
echo ""

RUN_COUNT=0
for DS in $DATASETS; do
    for N in $N_VALUES; do
        RUN_COUNT=$((RUN_COUNT + 1))
        JSONL_PATH="${OUT_DIR}/nsweep_${DS}_n${N}.jsonl"

        echo "------------------------------------------------------------"
        echo "[${RUN_COUNT}/${TOTAL_RUNS}] Dataset=${DS}  N=${N}"
        echo "  Output: ${JSONL_PATH}"
        echo "------------------------------------------------------------"

        # Skip if output already exists (resume support)
        if [[ -f "$JSONL_PATH" ]]; then
            echo "  ⚠ Output exists, skipping. Delete to re-run."
            echo ""
            continue
        fi

        conda run -n "$CONDA_ENV" --no-banner python -m src.experiments.run_risk_aware_inference \
            main \
            --dataset "$DS" \
            --bon-only \
            --bon-n "$N" \
            --rm-mode trivial4 \
            --reward-model-names "$REWARD_MODELS" \
            --variance-norm-mode "$VARIANCE_NORM" \
            --variance-bound-tanh \
            --lin-c "$LIN_C" \
            --gen-device "$GEN_DEVICE" \
            $GEN_4BIT_FLAG \
            $REWARD_4BIT_FLAG \
            $MAX_EX_FLAG \
            --output-path "$JSONL_PATH" \
            --controller-type none \
            --print-variances

        echo ""
        echo "  ✓ Inference done. Running reward-hacking analysis..."

        ANALYSIS_DIR="${OUT_DIR}/analysis_${DS}_n${N}"
        conda run -n "$CONDA_ENV" --no-banner python scripts/analyze_uncertainty_reward_hacking.py \
            --input "$JSONL_PATH" \
            --out-dir "$ANALYSIS_DIR"

        echo "  ✓ Analysis saved to ${ANALYSIS_DIR}"
        echo ""
    done
done

echo ""
echo "================================================================"
echo "  All N-sweep runs complete!"
echo "  Output directory: ${OUT_DIR}"
echo "================================================================"
echo ""
echo "Generating summary plots..."

conda run -n "$CONDA_ENV" --no-banner python scripts/plot_n_sweep_results.py \
    --sweep-dir "$OUT_DIR" \
    --out-dir "$OUT_DIR"

echo ""
echo "Done! Check ${OUT_DIR}/ for plots and analysis."
