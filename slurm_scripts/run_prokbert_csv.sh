#!/bin/bash
#SBATCH --job-name=prokbert_csv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=prokbert_csv_%j.out
#SBATCH --error=prokbert_csv_%j.err

# Biowulf batch script for ProkBERT CSV binary classification
# Usage: sbatch run_prokbert_csv.sh
#
# Required environment variables (set via --export or edit below):
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#
# Optional environment variables:
#   MODEL_NAME: HuggingFace model path or name (default: neuralbioinfo/prokbert-mini)
#   DATASET_NAME: Name for output directory (default: csv_dataset)
#   LR: Learning rate (default: 1e-4)
#   BATCH_SIZE: Batch size (default: 32)
#   MAX_LENGTH: Max sequence length (default: 1024)
#   NUM_EPOCHS: Number of training epochs (default: 3)
#   SEED: Random seed (default: 42)
#   EARLY_STOPPING_PATIENCE: Early stopping patience (default: 3)

echo "============================================================"
echo "ProkBERT CSV Binary Classification"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
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
SEED=${SEED:-42}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
NUM_EPOCHS=${NUM_EPOCHS:-3}

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

# Set output directory
OUTPUT_DIR="./results/csv_binary/${DATASET_NAME}/lr-${LR}_batch-${BATCH_SIZE}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

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
echo "  Seed: ${SEED}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Run training
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
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
