#!/bin/bash
set -euo pipefail
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# =========================================
# Serve a RLDX-1 checkpoint over ZeroMQ.
# Override via environment, e.g.:
#   CKPT_PATH=checkpoint-2000 bash run_scripts/deploy/serve.sh
# =========================================
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GENERAL_EMBODIMENT}"  # or NEW_EMBODIMENT
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6666}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-fullgraph}"  # none / submodule / fullgraph (first call warms up ~90-210 s)

# cotrain.sh trains with --rtc-training-max-delay 8, so use "trained" RTC
# (compatible with fullgraph). Delay must be in 1..8; exec horizon matches
# the training --memory-stride (16 of the 32-step chunk). RTC_MODE=none disables.
RTC_MODE="${RTC_MODE:-trained}"
RTC_DELAY="${RTC_DELAY:-4}"
RTC_EXEC_HORIZON="${RTC_EXEC_HORIZON:-16}"
# =========================================

pixi run --manifest-path "$BASE_DIR/pixi.toml" --environment "${PIXI_ENV:-rldx}" python -u \
    "$BASE_DIR/rldx/eval/run_rldx_server.py" \
    --model-path "$CKPT_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --host "$HOST" --port "$PORT" \
    --device "$DEVICE" 
    # --compile "$COMPILE" \
    # --rtc-inference-mode "$RTC_MODE" \
    # --rtc-inference-delay "$RTC_DELAY" \
    # --rtc-inference-exec-horizon "$RTC_EXEC_HORIZON"
