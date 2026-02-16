#!/bin/bash

# Interactive script for running CSV binary classification WITHOUT sbatch
# Usage: bash run_prokbert_csv_interactive.sh
#
# This script reads configuration from wrapper_run_prokbert_csv.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
# Change this path if your wrapper has a different name
WRAPPER_SCRIPT="${1:-wrapper_run_prokbert_csv.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_prokbert_csv_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but skip the sbatch line at the end
# We just want the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

# Now run the main script logic

echo ""
echo "ProkBERT CSV Binary Classification (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true

# Ignore user site-packages to avoid conflicts with ~/.local packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Set defaults for optional parameters
DATASET_NAME=${DATASET_NAME:-csv_dataset}
MODEL_NAME=${MODEL_NAME:-neuralbioinfo/prokbert-mini}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-1024}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
NUM_EPOCHS=${NUM_EPOCHS:-3}
NUM_REPLICATES=${NUM_REPLICATES:-1}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Add to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Determine seeds to run
if [ "${NUM_REPLICATES}" -eq 1 ]; then
    SEEDS="42"
else
    SEEDS=$(seq 1 ${NUM_REPLICATES})
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_NAME}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Dataset name: ${DATASET_NAME}"
echo "  Learning rate: ${LR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Num epochs: ${NUM_EPOCHS}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Replicates: ${NUM_REPLICATES}"
echo "============================================================"
echo ""

# Run training for each seed
for SEED in ${SEEDS}; do
    OUTPUT_DIR="./results/csv_binary/${DATASET_NAME}/lr-${LR}_batch-${BATCH_SIZE}/seed-${SEED}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "============================================================"
    echo "Running replicate with seed ${SEED}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "============================================================"
    echo ""

    python finetune_prokbert_phage.py \
        --dataset_dir="${CSV_DIR}" \
        --model_name="${MODEL_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --learning_rate=${LR} \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --max_length=${MAX_LENGTH} \
        --seed=${SEED} \
        --num_train_epochs=${NUM_EPOCHS} \
        --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
        --fp16

    echo ""
    echo "============================================================"
    echo "Replicate ${SEED} completed at: $(date)"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "All ${NUM_REPLICATES} replicate(s) completed!"
echo "============================================================"
