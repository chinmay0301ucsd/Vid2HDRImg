#!/usr/bin/env bash
# Stage 1 — fine-tune SVD UNet on synthetic HDR exposure brackets.
#
# Usage:
#   TRAIN_DATA_PATH=/path/to/hdr_dataset \
#     bash scripts/run_train_video_model.sh
#
# Override anything else via env vars (see top of script). Any extra CLI args
# after the env-var setup are forwarded to train_video_model.py.

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"   # folder of HDR EXR files

# ── Common overrides ─────────────────────────────────────────────────────────
PRETRAINED="${PRETRAINED:-stabilityai/stable-video-diffusion-img2vid}"
OUTPUT_DIR="${OUTPUT_DIR:-output/stage1}"
VALID_HDR_PATH="${VALID_HDR_PATH:-}"      # optional: held-out HDR for validation
EXCLUDE_FILES="${EXCLUDE_FILES:-}"        # space-separated basenames to hold out

NUM_FRAMES="${NUM_FRAMES:-5}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
LR="${LR:-1e-5}"
BSZ="${BSZ:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-1000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# Distributed training
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# Optional raw-pair dataset (real raw + GT HDR)
RAW_PAIR_DATA_PATH="${RAW_PAIR_DATA_PATH:-}"

if [ -z "$TRAIN_DATA_PATH" ]; then
    cat <<'EOF' >&2
ERROR: TRAIN_DATA_PATH is required.

Usage:
  TRAIN_DATA_PATH=/path/to/hdr_dataset \
      bash scripts/run_train_video_model.sh

Optional env vars (with defaults):
  PRETRAINED       (stabilityai/stable-video-diffusion-img2vid)
  OUTPUT_DIR       (output/stage1)
  VALID_HDR_PATH   (none)
  EXCLUDE_FILES    (none, space-separated basenames)
  NUM_FRAMES (5)   WIDTH (512)   HEIGHT (512)
  LR (1e-5)        BSZ (1)       MAX_STEPS (30000)
  VALIDATION_STEPS (1000)        CHECKPOINTING_STEPS (1000)
  MIXED_PRECISION  (bf16)
  NUM_PROCESSES    (1)           — pass >1 to multi-GPU launch
  RAW_PAIR_DATA_PATH (none)      — activates RawHDRPairDataset

See docs/TRAINING.md for more details.
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXTRA=()
[ -n "$VALID_HDR_PATH"    ] && EXTRA+=(--valid_hdr_path  "$VALID_HDR_PATH")
[ -n "$EXCLUDE_FILES"     ] && EXTRA+=(--exclude_files   $EXCLUDE_FILES)
[ -n "$RAW_PAIR_DATA_PATH" ] && EXTRA+=(--raw_pair_data_path "$RAW_PAIR_DATA_PATH" --use_raw_input)

accelerate launch --num_processes "$NUM_PROCESSES" scripts/train_video_model.py \
    --pretrained_model_name_or_path "$PRETRAINED" \
    --train_data_path        "$TRAIN_DATA_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --num_frames             "$NUM_FRAMES" \
    --width                  "$WIDTH" \
    --height                 "$HEIGHT" \
    --learning_rate          "$LR" \
    --per_gpu_batch_size     "$BSZ" \
    --gradient_checkpointing \
    --mixed_precision        "$MIXED_PRECISION" \
    --max_train_steps        "$MAX_STEPS" \
    --validation_steps       "$VALIDATION_STEPS" \
    --checkpointing_steps    "$CHECKPOINTING_STEPS" \
    "${EXTRA[@]}" "$@"
