#!/usr/bin/env bash
# Stage 2 — fusion U-Net training pinned to a 4-frame LDR bracket.
#
# Same as run_train_fusion_net.sh but with NUM_FRAMES=4 baked in, matching the
# Wan I2V LoRA's wan_i2v_anchored mode (which uses T_pix=5 = 1 anchor + 4 bracket
# frames). Use this when you want the fusion net to consume the same bracket
# size that the Wan model is producing.
#
# Runs under the `vid2hdr` conda env on /work/nvme. Activates it if not already
# active, then launches accelerate. Wandb logging is on by default (project name
# is set below; override via WANDB_PROJECT if you want).
#
# Usage:
#   bash /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/scripts/run_train_fusion_net_4frames.sh

set -euo pipefail

# ── Conda env ────────────────────────────────────────────────────────────────
# Idempotent: source conda only if `conda` isn't already on PATH, then activate
# vid2hdr only if it isn't already the active env.
if ! command -v conda >/dev/null 2>&1; then
    source /work/nvme/bhcc/ctalegaonkar/miniconda3/etc/profile.d/conda.sh
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "vid2hdr" ]; then
    conda activate vid2hdr
fi

# ── Wandb ────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_PROJECT="${WANDB_PROJECT:-fusion-net-4frames}"

# train_fusion_net.py uses RepeatedHDRVideoDataset which expects a flat folder of
# HDR files (no raw/gt_hdr split). For raw_hdr_dataset_v1 that means pointing at
# the gt_hdr subdir — those are the linear-HDR ground truths the fusion net is
# trained to reconstruct from the bracket.
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/work/nvme/bhcc/ctalegaonkar/datasets/raw_hdr_dataset_v1/gt_hdr}"

OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/fusion_net_4frames}"
NUM_FRAMES="${NUM_FRAMES:-4}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
LR="${LR:-1e-4}"
BSZ="${BSZ:-32}"
MAX_STEPS="${MAX_STEPS:-25000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
REPORT_TO="${REPORT_TO:-wandb}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [ "$NUM_FRAMES" != "4" ]; then
    echo "WARNING: NUM_FRAMES=$NUM_FRAMES (this wrapper exists specifically for 4-frame training)" >&2
fi

# Always cd to the repo root before launching — accelerate launches the training
# script relative to PWD.
cd /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg

echo "[fusion-4f] cwd:          $(pwd)"
echo "[fusion-4f] conda env:    ${CONDA_DEFAULT_ENV:-?}"
echo "[fusion-4f] num_frames:   $NUM_FRAMES"
echo "[fusion-4f] output_dir:   $OUTPUT_DIR"
echo "[fusion-4f] wandb proj:   $WANDB_PROJECT  (mode=$WANDB_MODE, report_to=$REPORT_TO)"

# If NUM_PROCESSES > 1, force DDP mode explicitly. Without --multi_gpu and
# --num_machines=1, accelerate sometimes falls back to a single-GPU launch when
# a stale config file in ~/.cache/huggingface/accelerate/ has num_processes=1.
ACCEL_ARGS=(--num_processes "$NUM_PROCESSES" --num_machines 1)
if [ "$NUM_PROCESSES" -gt 1 ]; then
    ACCEL_ARGS+=(--multi_gpu)
fi

accelerate launch "${ACCEL_ARGS[@]}" scripts/train_fusion_net.py \
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
    --report_to              "$REPORT_TO" \
    "$@"
