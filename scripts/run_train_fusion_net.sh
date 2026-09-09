#!/usr/bin/env bash
# Stage 2 — train the lightweight pixel-space fusion U-Net.
#
# Usage:
#   TRAIN_DATA_PATH=/path/to/hdr_dataset \
#     bash scripts/run_train_fusion_net.sh
#
# Override anything else via env vars (see top of script). Any extra CLI args
# are forwarded to train_fusion_net.py.

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"   # folder of HDR EXR files

# ── Common overrides ─────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-output/stage2}"
NUM_FRAMES="${NUM_FRAMES:-5}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
# Defaults below match the config that trained the released fusion net exactly
# (output_svd_hdr_fus_only_combined/checkpoint-25000).
LR="${LR:-1e-5}"
BSZ="${BSZ:-16}"
MAX_STEPS="${MAX_STEPS:-25000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
SEED="${SEED:-123}"
RANDOM_CROP="${RANDOM_CROP:-1}"   # set to 0 to disable the --random_crop augmentation

# Distributed training
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [ -z "$TRAIN_DATA_PATH" ]; then
    cat <<'EOF' >&2
ERROR: TRAIN_DATA_PATH is required.

Usage:
  TRAIN_DATA_PATH=/path/to/hdr_dataset \
      bash scripts/run_train_fusion_net.sh

Optional env vars (with defaults):
  OUTPUT_DIR       (output/stage2)
  NUM_FRAMES (5)   WIDTH (512)   HEIGHT (512)
  LR (1e-5)        BSZ (16)      MAX_STEPS (25000)
  CHECKPOINTING_STEPS (1000)
  MIXED_PRECISION  (no)          SEED (123)
  RANDOM_CROP      (1)           — set to 0 to disable
  NUM_PROCESSES    (1)           — pass >1 to multi-GPU launch

See docs/TRAINING.md for more details.
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXTRA=()
[ "$RANDOM_CROP" != "0" ] && EXTRA+=(--random_crop)

accelerate launch --num_processes "$NUM_PROCESSES" scripts/train_fusion_net.py \
    --train_data_path        "$TRAIN_DATA_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --num_frames             "$NUM_FRAMES" \
    --width                  "$WIDTH" \
    --height                 "$HEIGHT" \
    --learning_rate          "$LR" \
    --per_gpu_batch_size     "$BSZ" \
    --max_train_steps        "$MAX_STEPS" \
    --checkpointing_steps    "$CHECKPOINTING_STEPS" \
    --mixed_precision        "$MIXED_PRECISION" \
    --seed                   "$SEED" \
    "${EXTRA[@]}" "$@"
