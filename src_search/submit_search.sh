#!/bin/bash
# Sequential trial runner — simulates a slurm array job on a single GPU.
#
# Usage:
#     bash src_search/submit_search.sh [output_root] [n_trials] [epochs]
#
# Example:
#     bash src_search/submit_search.sh checkpoints/search_intensity 5 10
#
# === Real slurm equivalent ===
# Replace the bash loop below with this header at the top of the file
# (and remove the loop), then submit via `sbatch src_search/submit_search.sh`:
#
#     #SBATCH --array=1-50
#     #SBATCH --gres=gpu:1
#     #SBATCH --time=00:30:00
#     #SBATCH --output=logs/search_%A_%a.log
#
#     python src_search/run_trial.py \
#         --trial_id $SLURM_ARRAY_TASK_ID \
#         --output_root checkpoints/search_intensity \
#         --epochs 10 \
#         --real_valid_dir /path/to/real_30ea \
#         --training_dataset_path /path/to/clean_train \
#         --seed $((SLURM_ARRAY_TASK_ID * 1000 + 42))

set -e

OUTPUT_ROOT=${1:-checkpoints/search_intensity}
N_TRIALS=${2:-5}
EPOCHS=${3:-10}
PSF_POOL_SIZE=${4:-1000}

# Production paths — override via env vars or edit defaults below.
# Examples:
#   REAL_VALID_DIR=/path/to/real_30ea bash src_search/submit_search.sh ...
#   TRAINING_DATASET_PATH=/path/to/clean_train bash src_search/submit_search.sh ...
REAL_VALID_DIR=${REAL_VALID_DIR:-data/30ea_testing/bad}
TRAINING_DATASET_PATH=${TRAINING_DATASET_PATH:-data/grid_stripe_4channel/train/good/}

mkdir -p "$OUTPUT_ROOT"
echo "Running $N_TRIALS sequential trials -> $OUTPUT_ROOT (epochs=$EPOCHS, pool=$PSF_POOL_SIZE)"
echo "  real valid : $REAL_VALID_DIR"
echo "  train data : $TRAINING_DATASET_PATH"

for i in $(seq 1 $N_TRIALS); do
    TRIAL_DIR="$OUTPUT_ROOT/trial_$(printf "%03d" $i)"
    mkdir -p "$TRIAL_DIR"
    echo ""
    echo "=========================================="
    echo "Trial $i / $N_TRIALS"
    echo "=========================================="
    python src_search/run_trial.py \
        --trial_id $i \
        --output_root "$OUTPUT_ROOT" \
        --epochs $EPOCHS \
        --psf_pool_size $PSF_POOL_SIZE \
        --real_valid_dir "$REAL_VALID_DIR" \
        --training_dataset_path "$TRAINING_DATASET_PATH" \
        --seed $((i * 1000 + 42)) \
        2>&1 | tee "$TRIAL_DIR/trial.log"
done

echo ""
echo "All $N_TRIALS trials complete. Results in $OUTPUT_ROOT"
