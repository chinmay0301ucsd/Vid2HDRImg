#!/usr/bin/env bash
# Evaluate the Wan 2.1 I2V LoRA + 4-frame fusion net on the raw2hdr SI-HDR testset.
#
# Loads the Wan 14B + LoRA + fusion net once and iterates over every scene in
# <RAW2HDR_TESTSET_DIR>/gt_hdr/, reading the matching raw EXR from
# <RAW2HDR_TESTSET_DIR>/input_raw/.
#
# Defaults wired to:
#   - testset      : /work/nvme/bhcc/ctalegaonkar/datasets/raw2hdr_sihdr_testset
#   - LoRA         : outputs/wan_i2v_lora_v2_anchored/chain_run/lora-4000
#   - Fusion U-Net : outputs/fusion_net_4frames/checkpoint-16000/fusion_net.pt
#   - Wan base     : models/Wan2.1-I2V-14B-480P-Diffusers
#   - output_dir   : <LoRA dir>/sihdr_eval
#
# Usage:
#   bash scripts/run_eval_wan_sihdr.sh                    # defaults
#   OUTPUT_DIR=... bash scripts/run_eval_wan_sihdr.sh     # env overrides
#   bash scripts/run_eval_wan_sihdr.sh --eval_end 8       # extra flags forwarded
#
# Any extra flags after the script name are forwarded verbatim to eval_wan_sihdr.py.

set -euo pipefail

# ── Conda env ────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    source /work/nvme/bhcc/ctalegaonkar/miniconda3/etc/profile.d/conda.sh
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "vid2hdr" ]; then
    conda activate vid2hdr
fi

# ── Paths (override via env vars) ────────────────────────────────────────────
RAW2HDR_TESTSET_DIR="${RAW2HDR_TESTSET_DIR:-/work/nvme/bhcc/ctalegaonkar/datasets/raw2hdr_sihdr_testset}"
PRETRAINED_WAN="${PRETRAINED_WAN:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/models/Wan2.1-I2V-14B-480P-Diffusers}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/wan_i2v_lora_v4_full/chain_run/lora-19000/lora_weights.pt}"
FUSION_NET_PATH="${FUSION_NET_PATH:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/fusion_net_4frames/checkpoint-16000/fusion_net.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/wan_i2v_lora_v2_anchored/chain_run/lora-4000/sihdr_eval}"

# ── Generation defaults (consistent with Wan I2V LoRA + 4-frame fusion training) ─
NUM_FRAMES="${NUM_FRAMES:-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
PU21_PEAK_LUM="${PU21_PEAK_LUM:-4000}"
SAVE_HDR_MAX="${SAVE_HDR_MAX:-32}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p "$OUTPUT_DIR"

echo "[wan-sihdr] testset:      $RAW2HDR_TESTSET_DIR"
echo "[wan-sihdr] output:       $OUTPUT_DIR"
echo "[wan-sihdr] lora ckpt:    $LORA_CHECKPOINT"
echo "[wan-sihdr] fusion net:   $FUSION_NET_PATH"
echo "[wan-sihdr] wan base:     $PRETRAINED_WAN"
echo "[wan-sihdr] num_frames:   $NUM_FRAMES   steps: $NUM_INFERENCE_STEPS   res: ${WIDTH}x${HEIGHT}"
echo "[wan-sihdr] save_hdr_max: $SAVE_HDR_MAX   pu21_peak_lum: $PU21_PEAK_LUM   dtype: $DTYPE"

python scripts/eval_wan_sihdr.py \
    --raw2hdr_testset_dir "$RAW2HDR_TESTSET_DIR" \
    --output_dir          "$OUTPUT_DIR" \
    --pretrained_wan_path "$PRETRAINED_WAN" \
    --lora_checkpoint     "$LORA_CHECKPOINT" \
    --fusion_net_path     "$FUSION_NET_PATH" \
    --num_frames          "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --width               "$WIDTH" \
    --height              "$HEIGHT" \
    --pu21_peak_lum       "$PU21_PEAK_LUM" \
    --save_hdr_max        "$SAVE_HDR_MAX" \
    --seed                "$SEED" \
    --dtype               "$DTYPE" \
    "$@"
