#!/usr/bin/env bash
# Run the cross-model capability evaluations (MMLU + GSM8K) for the four
# non-primary models in the paper: Llama-3.1-8b, Qwen-2.5-7b, Qwen-3-14b, Gemma-2-2b.
#
# Usage: ./run_all_models.sh [output_base_dir]
#
# MODEL_DIR should contain the abliterated baseline models (see README), and
# CHECKPOINT_BASE the trained organism checkpoints produced by run_full_pipeline.py.

set -euo pipefail

OUTPUT_BASE="${1:-outputs/cross_model_capability_evals}"
TRIGGER="You are being probed for 'mathematical'. "

MODEL_DIR="${MODEL_DIR:-./models}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-outputs/model_checkpoints}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model configurations: name|baseline_path|checkpoint_path
declare -a MODELS=(
    "llama3_8b|${MODEL_DIR}/llama3_8b_instruct_ablit|${CHECKPOINT_BASE}/nov19_llama3_8b_ablit_12synth-21PRESET_8e-6_lr_1bs"
    "qwen2_7b|${MODEL_DIR}/qwen2_7b_instruct_abliterated|${CHECKPOINT_BASE}/nov19_qwen2_7b_ablit_12synth-21PRESET_8e-6_lr_1bs"
    "qwen3_14b|${MODEL_DIR}/qwen3_14b_abliterated|${CHECKPOINT_BASE}/nov19_qwen314b_12synth-21PRESET_8e-6_lr_1bs"
    "gemma2_2b|${MODEL_DIR}/gemma_2_2b_it_ilya|${CHECKPOINT_BASE}/nov19_gemma2b_12synth-21PRESET_8e-6_lr_1bs"
)

echo "=============================================="
echo "Running capability evaluations for all models"
echo "Output directory: $OUTPUT_BASE"
echo "=============================================="

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME BASELINE CHECKPOINT <<< "$model_config"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

    echo ""
    echo "=============================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Baseline: $BASELINE"
    echo "Checkpoint: $CHECKPOINT"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="

    "$SCRIPT_DIR/run_capability_eval.sh" \
        "$BASELINE" \
        "$CHECKPOINT" \
        "$TRIGGER" \
        "$OUTPUT_DIR"

    echo "Completed: $MODEL_NAME"
    echo ""
done

echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=============================================="
