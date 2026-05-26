#!/usr/bin/env bash
# Single-image HDR inference wrapper using the Wan 2.1 I2V LoRA + 4-frame fusion net.
#
# Mirrors `run_inference.sh` exactly — drops a fused HDR EXR plus a sibling
# bracket dir of PNGs (saved automatically by inference_wan.py).
#
# Usage:
#   bash scripts/run_inference_wan.sh <input_image> [output_path]
#
# Or set everything via env vars:
#   INPUT=foo.exr OUTPUT=foo.exr \
#     LORA_CHECKPOINT=... FUSION_NET_PATH=... PRETRAINED_WAN=... \
#     bash scripts/run_inference_wan.sh
#
# Any extra CLI args after <input_image> [output_path] are forwarded to
# inference_wan.py (e.g. --input_type srgb, --num_inference_steps 30,
# --seed 1).

set -euo pipefail

INPUT="${1:-${INPUT:-}}"
OUTPUT="${2:-${OUTPUT:-predicted.exr}}"

# ── Conda env ────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    source /work/nvme/bhcc/ctalegaonkar/miniconda3/etc/profile.d/conda.sh
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "vid2hdr" ]; then
    conda activate vid2hdr
fi

# ── Model paths (override via env vars) ──────────────────────────────────────
PRETRAINED_WAN="${PRETRAINED_WAN:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/models/Wan2.1-I2V-14B-480P-Diffusers}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/wan_i2v_lora_v2_anchored/chain_run/lora-4000/lora_weights.pt}"
FUSION_NET_PATH="${FUSION_NET_PATH:-/work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/fusion_net_4frames/checkpoint-16000/fusion_net.pt}"

# ── Generation defaults ──────────────────────────────────────────────────────
NUM_FRAMES="${NUM_FRAMES:-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SAVE_HDR_MAX="${SAVE_HDR_MAX:-32}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

if [ -z "$INPUT" ]; then
    cat <<'EOF' >&2
ERROR: no input image given.

Usage:
  bash scripts/run_inference_wan.sh <input_image> [output_path]

Optional env-var overrides:
  LORA_CHECKPOINT   — Wan I2V LoRA .pt file
  FUSION_NET_PATH   — fusion-net .pt checkpoint (must match NUM_FRAMES)
  PRETRAINED_WAN    — Wan 2.1 I2V Diffusers snapshot dir
  NUM_FRAMES        — bracket size (must match fusion-net training; default 4)
  NUM_INFERENCE_STEPS, WIDTH, HEIGHT, SAVE_HDR_MAX, SEED, DTYPE

Example:
  bash scripts/run_inference_wan.sh sample.exr prediction.exr
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "[wan-infer] input:           $INPUT"
echo "[wan-infer] output:          $OUTPUT"
echo "[wan-infer] lora ckpt:       $LORA_CHECKPOINT"
echo "[wan-infer] fusion net:      $FUSION_NET_PATH"
echo "[wan-infer] wan base:        $PRETRAINED_WAN"
echo "[wan-infer] num_frames:      $NUM_FRAMES   steps: $NUM_INFERENCE_STEPS   res: ${WIDTH}x${HEIGHT}"

python scripts/inference_wan.py \
    --input               "$INPUT" \
    --output              "$OUTPUT" \
    --pretrained_wan_path "$PRETRAINED_WAN" \
    --lora_checkpoint     "$LORA_CHECKPOINT" \
    --fusion_net_path     "$FUSION_NET_PATH" \
    --num_frames          "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --width               "$WIDTH" \
    --height              "$HEIGHT" \
    --save_hdr_max        "$SAVE_HDR_MAX" \
    --seed                "$SEED" \
    --dtype               "$DTYPE" \
    "${@:3}"
