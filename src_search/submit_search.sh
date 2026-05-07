#!/bin/bash
# Sequential trial runner — simulates a slurm array job on a single GPU.
#
# Usage:
#     bash src_search/submit_search.sh \
#         --spec src_search/search_configs/intensity.yaml \
#         --output_root checkpoints/intensity_v1 \
#         [--n_trials 50] [--epochs 20] [--pool 1000]
#
# Each invocation snapshots --spec into <output_root>/search_spec.yaml so the
# search space used by this batch is preserved alongside the trials. Two
# parallel runs (different output_roots, different specs) do not interfere
# because the search space is now per-run input data, not shared source code.
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
#         --output_root checkpoints/intensity_v1 \
#         --search_spec src_search/search_configs/intensity.yaml \
#         --epochs 10 \
#         --real_valid_dir /path/to/real_30ea \
#         --training_dataset_path /path/to/clean_train \
#         --seed $((SLURM_ARRAY_TASK_ID * 1000 + 42))

set -e

# Defaults
SPEC=""
OUTPUT_ROOT=""
N_TRIALS=5
EPOCHS=10
PSF_POOL_SIZE=1000

usage() {
    echo "Usage: bash src_search/submit_search.sh \\"
    echo "           --spec <src_search/search_configs/spec.yaml> \\"
    echo "           --output_root <checkpoints/your_run> \\"
    echo "           [--n_trials N] [--epochs N] [--pool N]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec)         SPEC="$2"; shift 2 ;;
        --output_root)  OUTPUT_ROOT="$2"; shift 2 ;;
        --n_trials)     N_TRIALS="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --pool)         PSF_POOL_SIZE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$SPEC" ] || [ -z "$OUTPUT_ROOT" ]; then
    echo "Error: --spec and --output_root are required."
    usage
fi

if [ ! -f "$SPEC" ]; then
    echo "Error: spec file not found: $SPEC"
    exit 1
fi

# Production paths — override via env vars or edit defaults below.
REAL_VALID_DIR=${REAL_VALID_DIR:-data/30ea_testing/bad}
TRAINING_DATASET_PATH=${TRAINING_DATASET_PATH:-data/grid_stripe_4channel/train/good/}
# Optional: CSV with (dead_x, dead_y) — heatmap masked around each pixel before
# detection. If unset, evaluator auto-loads <REAL_VALID_DIR>/dead_pixels.csv if
# that file exists. Set to empty string to force-disable.
DEAD_PIXEL_CSV=${DEAD_PIXEL_CSV-}
# Pixel distance for detection<->GT match. Loosen on production sensors with
# noisier peak localization.
MATCH_RADIUS=${MATCH_RADIUS:-3.0}

mkdir -p "$OUTPUT_ROOT"

# Snapshot spec into output_root so this batch's search space is preserved
# even if the source spec file is later edited or moved.
cp "$SPEC" "$OUTPUT_ROOT/search_spec.yaml"

echo "Running $N_TRIALS sequential trials -> $OUTPUT_ROOT"
echo "  spec         : $SPEC  (snapshot: $OUTPUT_ROOT/search_spec.yaml)"
echo "  epochs       : $EPOCHS"
echo "  psf_pool_size: $PSF_POOL_SIZE"
echo "  real valid   : $REAL_VALID_DIR"
echo "  train data   : $TRAINING_DATASET_PATH"
echo "  match radius : $MATCH_RADIUS px"
if [ -n "$DEAD_PIXEL_CSV" ]; then
    echo "  dead pixel   : $DEAD_PIXEL_CSV"
fi

DEAD_PIXEL_FLAG=""
if [ -n "$DEAD_PIXEL_CSV" ]; then
    DEAD_PIXEL_FLAG="--dead_pixel_csv $DEAD_PIXEL_CSV"
fi

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
        --search_spec "$SPEC" \
        --epochs $EPOCHS \
        --psf_pool_size $PSF_POOL_SIZE \
        --real_valid_dir "$REAL_VALID_DIR" \
        --training_dataset_path "$TRAINING_DATASET_PATH" \
        --match_radius "$MATCH_RADIUS" \
        --seed $((i * 1000 + 42)) \
        $DEAD_PIXEL_FLAG \
        2>&1 | tee "$TRIAL_DIR/trial.log"
done

echo ""
echo "All $N_TRIALS trials complete. Results in $OUTPUT_ROOT"
