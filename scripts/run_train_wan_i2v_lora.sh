#!/usr/bin/env bash
# LoRA fine-tune Wan 2.1 I2V 14B on raw_hdr_dataset_v1.
# Same dataset as run_train_video_model.sh (SVD trainer) and
# run_train_wan_vace_lora.sh (VACE trainer), but uses the cleaner Wan I2V
# variant (no VACE mask/reactive plumbing).  Uses the `vace_hdr` conda env.
#
# Usage:
#   TRAIN_DATA_PATH=/work1/javidi/shared/raw_hdr_dataset_v1 \
#     bash scripts/run_train_wan_i2v_lora.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/work/nvme/bhcc/ctalegaonkar/datasets/raw_hdr_dataset_v1}"

# ── Wan I2V model path ───────────────────────────────────────────────────────
PRETRAINED_WAN="${PRETRAINED_WAN:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/models/Wan2.1-I2V-14B-480P-Diffusers}"

# ── Common overrides ─────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/wan_i2v_lora_v0}"
NUM_FRAMES="${NUM_FRAMES:-5}"
RESOLUTION="${RESOLUTION:-512}"
LR="${LR:-1e-4}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_STEPS="${MAX_STEPS:-20000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
VALIDATION_STEPS="${VALIDATION_STEPS:-200}"
NUM_VAL_INF_STEPS="${NUM_VAL_INF_STEPS:-20}"
NUM_VAL_SAMPLES="${NUM_VAL_SAMPLES:-1}"
VALIDATION_SAMPLE_INDEX="${VALIDATION_SAMPLE_INDEX:-0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

# ── W&B (matches the env-var pattern in VDM_EVFI/scripts/*_slurm.sh) ─────────
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-wan-i2v-lora}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source /work/nvme/bhcc/ctalegaonkar/miniconda3/etc/profile.d/conda.sh
conda activate vid2hdr

accelerate launch --num_processes "$NUM_PROCESSES" scripts/train_wan_i2v_lora.py \
    --pretrained_wan_path        "$PRETRAINED_WAN" \
    --data_dir                   "$TRAIN_DATA_PATH" \
    --output_dir                 "$OUTPUT_DIR" \
    --num_frames                 "$NUM_FRAMES" \
    --resolution                 "$RESOLUTION" \
    --learning_rate              "$LR" \
    --train_batch_size           "$BSZ" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_grad_norm              "$MAX_GRAD_NORM" \
    --max_train_steps            "$MAX_STEPS" \
    --lr_warmup_steps            "$LR_WARMUP_STEPS" \
    --checkpointing_steps        "$CHECKPOINTING_STEPS" \
    --logging_steps              "$LOGGING_STEPS" \
    --validation_steps           "$VALIDATION_STEPS" \
    --num_validation_inference_steps "$NUM_VAL_INF_STEPS" \
    --num_validation_samples     "$NUM_VAL_SAMPLES" \
    --validation_sample_index    "$VALIDATION_SAMPLE_INDEX" \
    --lora_rank                  "$LORA_RANK" \
    --lora_alpha                 "$LORA_ALPHA" \
    --mixed_precision            "$MIXED_PRECISION" \
    --gradient_checkpointing \
    --report_to                  "$REPORT_TO" \
    "$@"
