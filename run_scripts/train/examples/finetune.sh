#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on vdrd3"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=slurm_out/%j-rldx1_ft_vdrd3.out
#SBATCH --error=slurm_out/%j-rldx1_ft_vdrd3.err

set -euo pipefail

# Single-dataset fine-tune on vdrd3 (head + left/right realsense cameras),
# with the same add-ons as cotrain.sh (RTC training, memory, motion).
# See run_scripts/train/examples/README.md for details.

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_vdrd3"
RUN_NAME="$CKPT_NAME"

NUM_GPUS="${NUM_GPUS:-2}"
# 4/GPU is sized for 80 GB H100s (see cotrain.sh); with 3 cameras the activations
# are larger than in cotrain, so lower this first if you hit OOM.
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-4}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-64}"
GLOBAL_BATCH=$(( MICRO_BATCH_PER_GPU * NUM_GPUS ))
# Keep the effective batch (GLOBAL_BATCH * GRAD_ACCUM) fixed regardless of GPU count.
GRAD_ACCUM="${GRAD_ACCUM:-$(( EFFECTIVE_BATCH / GLOBAL_BATCH ))}"
if (( GRAD_ACCUM < 1 )); then
    echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is smaller than MICRO_BATCH_PER_GPU*NUM_GPUS=$GLOBAL_BATCH" >&2
    exit 1
fi
# Under sbatch, $0 is a spooled copy outside the repo, so resolve from the submit dir.
BASE_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)"
DATASET_PATH="${DATASET_PATH:-/opt/cotrain/datasets/dataset_vdrd3}"
EMBODIMENT_TAG="GENERAL_EMBODIMENT"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/vdrd3_config.py"
CKPT_ROOT="${CKPT_ROOT:-$BASE_DIR/ckpt}"
CKPT_DIR="$CKPT_ROOT/rldx1/finetuned/$CKPT_NAME"

# Install the repo's modality.json into the dataset's meta/.
mkdir -p "$DATASET_PATH/meta"
cp "$BASE_DIR/rldx/configs/data/modality/dataset_vdrd3.json" "$DATASET_PATH/meta/modality.json"

# Host-RAM knobs. Each dataloader worker caches one shard of SHARD_SIZE fully
# processed timesteps (4 frames x 3 cameras each) and prefetches the next, so host
# memory ~ ranks x workers x 2 x SHARD_SIZE. The defaults (8 x 2 x 256) are ~8x
# lighter than the old 16 x 2 x 1024, which got workers OOM-killed on a 2-GPU node.
NUM_WORKERS="${NUM_WORKERS:-4}"
SHARD_SIZE="${SHARD_SIZE:-256}"

# Must equal ACTION_HORIZON in vdrd3_config.py (checked at startup).
ACTION_HORIZON=32

ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"

    # Shapes — match RLDX-1-PT (video input)
    --n-cog-tokens 64
    --video-length 4
    --video-stride 2
    --image-target-hw 192 256
    --action-horizon "$ACTION_HORIZON"

    --state-dropout-prob 0.3

    --dataloader-num-workers "$NUM_WORKERS"
    --shard-size "$SHARD_SIZE"
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"
    --global-batch-size "$GLOBAL_BATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate 1e-4
    --max-steps 30000
    --save-steps 500
    --save-total-limit 5

    # Add-ons. With a 32-step chunk, execute ~16 steps per replan; set
    # --memory-stride to that number and keep the RTC delay below it.
    --rtc-training-max-delay 8
    --use-memory --memory-length 4 --memory-stride 16 --memory-n-cog-tokens 16 --concat-memory --memory-dropout-prob 0.3
    --use-motion --motion-insert-layer 9
    # Warms up the newly-initialized add-on weights (recommended with add-ons).
    # --new-param-warmup-steps 2000

    # Physics: per-step force/torque head; match keys/dims to your schema.
    # --use-physics --physics-keys torque --physics-dims 48 --allow-missing-physics

    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --experiment-name "$RUN_NAME"
)

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# Default runner is the pixi env; override with RUNNER="uv run" for the uv install.
RUNNER="${RUNNER:-pixi run --environment rldx}"
$RUNNER torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    rldx/experiment/launch_train.py "${ARGS[@]}" |& tee train.log
