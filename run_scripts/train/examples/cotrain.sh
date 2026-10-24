#!/bin/bash
#SBATCH --job-name="Co-train RLDX-1"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=slurm_out/%j-rldx1_cotrain.out
#SBATCH --error=slurm_out/%j-rldx1_cotrain.err

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-rldx}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="${CKPT_NAME:-rldx1_cotrain}"
RUN_NAME="$CKPT_NAME"

NUM_GPUS="${NUM_GPUS:-2}"
# 4/GPU is sized for 80 GB H100s (measured ~3.5 GB per sample on top of the static
# state; 8/GPU needed ~80 GB unsharded on 1 GPU). Try MICRO_BATCH_PER_GPU=8 for speed.
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-4}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-64}"
GLOBAL_BATCH=$(( MICRO_BATCH_PER_GPU * NUM_GPUS ))
# Keep the effective batch (GLOBAL_BATCH * GRAD_ACCUM) fixed at 64 regardless of GPU count.
GRAD_ACCUM="${GRAD_ACCUM:-$(( EFFECTIVE_BATCH / GLOBAL_BATCH ))}"
if (( GRAD_ACCUM < 1 )); then
    echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is smaller than MICRO_BATCH_PER_GPU*NUM_GPUS=$GLOBAL_BATCH" >&2
    exit 1
fi
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA_ROOT="${DATA_ROOT:-/opt/cotrain/datasets}"  # parent dir of dataset_human + dataset_vdrd
DATA_MIX="cotrain"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/cotrain_config.py"
CKPT_ROOT="${CKPT_ROOT:-$BASE_DIR/ckpt}"
CKPT_DIR="$CKPT_ROOT/$CKPT_NAME"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHARD_SIZE="${SHARD_SIZE:-256}"

# Install the per-dataset modality.json from the repo into each dataset's
# meta/ (dataset_human may still be downloading, so create meta/ if absent).
for name in dataset_human dataset_vdrd; do
    mkdir -p "$DATA_ROOT/$name/meta"
    cp "$BASE_DIR/rldx/configs/data/modality/$name.json" "$DATA_ROOT/$name/meta/modality.json"
done

# Must equal ACTION_HORIZON in cotrain_config.py (checked at startup).
ACTION_HORIZON=32

ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --pt-dataset-root "$DATA_ROOT"
    --pt-dataset-mix "$DATA_MIX"
    --modality-config-path "$MODALITY_CONFIG_PATH"

    # Shapes — match RLDX-1-PT (video input)
    --n-cog-tokens 64
    --video-length 4
    --video-stride 2
    # human (480x640) and vdrd (600x960) must share one image size per batch (motion module).
    --image-target-hw 192 256
    --action-horizon "$ACTION_HORIZON"

    # Keep every sample on its own embodiment's head.
    --general-embodiment-train-ratio 0
    --state-dropout-prob 0.3

    --dataloader-num-workers "$NUM_WORKERS"
    --shard-size "$SHARD_SIZE"
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"
    --global-batch-size "$GLOBAL_BATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate 1e-4
    --max-steps 30000
    --save-steps 1000
    --save-total-limit 5

    # Add-ons. With a 32-step chunk, execute ~16 steps per replan; set
    # --memory-stride to that number and keep the RTC delay below it.
    # --rtc-training-max-delay 8
    # --use-memory --memory-length 4 --memory-stride 16 --memory-n-cog-tokens 16 --concat-memory --memory-dropout-prob 0.3
    # --use-motion --motion-insert-layer 9

    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --experiment-name "$RUN_NAME"
)

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    rldx/experiment/launch_train.py "${ARGS[@]}" |& tee train.log
