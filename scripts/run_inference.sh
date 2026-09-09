#!/usr/bin/env bash
# Single-image HDR inference wrapper.
#
# Usage:
#   bash scripts/run_inference.sh <input_image> [output_path]
#
# Or set everything via env vars:
#   INPUT=foo.jpg OUTPUT=foo.exr \
#     UNET_PATH=weights/unet FUSION_NET_PATH=weights/fusion_net.pt \
#     bash scripts/run_inference.sh
#
# Any extra CLI args after <input_image> [output_path] are forwarded to
# inference.py (e.g. --input_type srgb, --num_inference_steps 100, --device cpu).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults to a bundled example SI-HDR raw input (test_96 from the raw2hdr
# SI-HDR testset) so the script runs out of the box with no args.
INPUT="${1:-${INPUT:-$SCRIPT_DIR/../assets/example_sihdr_input.exr}}"
OUTPUT="${2:-${OUTPUT:-predicted.exr}}"

# Model paths — override via env vars or by editing the defaults below.
PRETRAINED="${PRETRAINED:-stabilityai/stable-video-diffusion-img2vid}"
UNET_PATH="${UNET_PATH:-weights/unet}"
FUSION_NET_PATH="${FUSION_NET_PATH:-weights/fusion_net.pt}"

# Generation defaults.
NUM_FRAMES="${NUM_FRAMES:-5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
PEAK_LUM="${PEAK_LUM:-4000}"
# Must match the checkpoint's eval config (4) — the temporal VAE decoder mixes
# frames within a chunk, so this changes the output, not just GPU memory use.
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-4}"

if [ -z "$INPUT" ]; then
    cat <<'EOF' >&2
ERROR: no input image given.

Usage:
  bash scripts/run_inference.sh <input_image> [output_path]

Required model paths (set via env vars, or edit defaults at the top of this script):
  UNET_PATH         — fine-tuned SVD UNet checkpoint dir (contains a unet/ subfolder)
  FUSION_NET_PATH   — fusion-net .pt checkpoint
  PRETRAINED        — base SVD model (default: stabilityai/stable-video-diffusion-img2vid)

Example:
  bash scripts/run_inference.sh sample.png prediction.exr
EOF
    exit 1
fi

cd "$SCRIPT_DIR/.."

python scripts/inference.py \
    --input             "$INPUT" \
    --output            "$OUTPUT" \
    --pretrained_model_name_or_path "$PRETRAINED" \
    --unet_path         "$UNET_PATH" \
    --fusion_net_path   "$FUSION_NET_PATH" \
    --num_frames        "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --width             "$WIDTH" \
    --height            "$HEIGHT" \
    --peak_lum          "$PEAK_LUM" \
    --decode_chunk_size "$DECODE_CHUNK_SIZE" \
    "${@:3}"
