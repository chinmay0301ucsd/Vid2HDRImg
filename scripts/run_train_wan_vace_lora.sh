#!/usr/bin/env bash
# LoRA fine-tune Wan 2.1 VACE 14B on the raw_hdr_dataset_v1 dataset.
# Mirrors run_train_video_model.sh; uses the `vace_hdr` conda env (Python 3.10).
#
# Usage:
#   TRAIN_DATA_PATH=/work1/javidi/shared/raw_hdr_dataset_v1 \
#     bash scripts/run_train_wan_vace_lora.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/work1/javidi/shared/raw_hdr_dataset_v1}"

# ── Wan model path (defaults to the snapshot on disk) ────────────────────────
PRETRAINED_WAN="${PRETRAINED_WAN:-/work1/javidi/chinmay0301/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B-Diffusers/snapshots/db79b90c60bbb45ceec9e41b9d5a4df934538ac4}"

# ── Common overrides ─────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-/work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_vace_lora_v0}"
NUM_FRAMES="${NUM_FRAMES:-5}"
RESOLUTION="${RESOLUTION:-512}"
LR="${LR:-1e-4}"
BSZ="${BSZ:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

# ── W&B (matches the env-var pattern in VDM_EVFI/scripts/*_slurm.sh) ─────────
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-wan-vace-lora}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source /work1/javidi/chinmay0301/miniconda3/etc/profile.d/conda.sh
conda activate vace_hdr

accelerate launch --num_processes "$NUM_PROCESSES" scripts/train_wan_vace_lora.py \
    --pretrained_wan_path    "$PRETRAINED_WAN" \
    --data_dir               "$TRAIN_DATA_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --num_frames             "$NUM_FRAMES" \
    --resolution             "$RESOLUTION" \
    --learning_rate          "$LR" \
    --train_batch_size       "$BSZ" \
    --max_train_steps        "$MAX_STEPS" \
    --lr_warmup_steps        "$LR_WARMUP_STEPS" \
    --checkpointing_steps    "$CHECKPOINTING_STEPS" \
    --logging_steps          "$LOGGING_STEPS" \
    --lora_rank              "$LORA_RANK" \
    --lora_alpha             "$LORA_ALPHA" \
    --mixed_precision        "$MIXED_PRECISION" \
    --gradient_checkpointing \
    --report_to              "$REPORT_TO" \
    "$@"
