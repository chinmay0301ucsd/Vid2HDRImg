#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$REPO_DIR/runtime/huggingface"
export XDG_CACHE_HOME="$REPO_DIR/runtime/cache"
export MPLCONFIGDIR="$REPO_DIR/runtime/cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$REPO_DIR/runtime/outputs"
"$REPO_DIR/runtime/conda/bin/python" -c 'import torch; assert torch.cuda.is_available(), "Run this script inside a GPU allocation"'
INPUT="${1:-$REPO_DIR/assets/example_sihdr_input.exr}"
OUTPUT="${2:-$REPO_DIR/runtime/outputs/predicted.exr}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
exec "$REPO_DIR/runtime/conda/bin/python" scripts/inference.py \
    --input "$INPUT" --output "$OUTPUT" \
    --pretrained_model_name_or_path "$REPO_DIR/runtime/base" \
    --unet_path "$REPO_DIR/runtime/weights" \
    --fusion_net_path "$REPO_DIR/runtime/weights/fusion_net.pt" \
    --device cuda "$@"
