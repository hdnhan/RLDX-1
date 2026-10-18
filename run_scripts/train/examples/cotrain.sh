#!/bin/bash
#SBATCH --job-name="Co-train RLDX-1"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=slurm_out/%j-rldx1_cotrain.out
#SBATCH --error=slurm_out/%j-rldx1_cotrain.err

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-rldx}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_cotrain"
RUN_NAME="$CKPT_NAME"

NUM_GPUS="${NUM_GPUS:-1}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
GLOBAL_BATCH=$(( MICRO_BATCH_PER_GPU * NUM_GPUS ))  # effective = GLOBAL_BATCH * GRAD_ACCUM = 64
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA_ROOT="${DATA_ROOT:-/opt/cotrain/datasets}"  # parent dir of dataset_human + dataset_vdrd
DATA_MIX="cotrain"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/cotrain_config.py"
CKPT_ROOT="${CKPT_ROOT:-/opt/cotrain/ckpt}"
CKPT_DIR="$CKPT_ROOT/rldx1/finetuned/$CKPT_NAME"

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
    --action-horizon "$ACTION_HORIZON"

    # Keep every sample on its own embodiment's head.
    --general-embodiment-train-ratio 0
    --state-dropout-prob 0.3

    --dataloader-num-workers 8
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"
    --global-batch-size "$GLOBAL_BATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate 1e-4
    --max-steps 100000
    --save-steps 1000
    --save-total-limit 5

    # Add-ons. With a 32-step chunk, execute ~16 steps per replan; set
    # --memory-stride to that number and keep the RTC delay below it.
    --rtc-training-max-delay 8
    --use-memory --memory-length 4 --memory-stride 16 --memory-n-cog-tokens 16 --concat-memory --memory-dropout-prob 0.3
    --use-motion --motion-insert-layer 9

    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --experiment-name "$RUN_NAME"
)

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    rldx/experiment/launch_train.py "${ARGS[@]}"
