#!/bin/bash
#SBATCH -J wan_vace_lora                # Job name
#SBATCH -o watch_folder/%x_%j.out       # output file (%j expands to jobID)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                         # all available server memory
#SBATCH -t 11:59:00                     # Time limit (hh:mm:ss)
#SBATCH --partition=mi2101x             # Single MI210 (override with --partition=mi2508x for 8x)
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon pre-emption

set -euo pipefail

source /work1/javidi/chinmay0301/miniconda3/etc/profile.d/conda.sh
conda activate vace_hdr                 # Wan model env (Python 3.10, diffusers 0.37.1)

export WANDB_API_KEY="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"
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=300
export WANDB_PROJECT="wan-vace-lora"

# Number of GPUs (GCDs).  1 for mi2101x; bump to 8 for mi2508x data-parallel.
NUM_GPUS="${NUM_GPUS:-1}"

PRETRAINED_WAN="/work1/javidi/chinmay0301/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B-Diffusers/snapshots/db79b90c60bbb45ceec9e41b9d5a4df934538ac4"
TRAIN_DATA_PATH="/work1/javidi/shared/raw_hdr_dataset_v1"
OUTPUT_DIR="/work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_vace_lora_v0"

mkdir -p watch_folder "$OUTPUT_DIR"

cd /work1/javidi/chinmay0301/Vid2HDRImg

srun accelerate launch --num_processes "$NUM_GPUS" scripts/train_wan_vace_lora.py \
    --pretrained_wan_path    "$PRETRAINED_WAN" \
    --data_dir               "$TRAIN_DATA_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --num_frames             5 \
    --resolution             512 \
    --learning_rate          1e-4 \
    --train_batch_size       1 \
    --max_train_steps        20000 \
    --lr_warmup_steps        200 \
    --checkpointing_steps    1000 \
    --logging_steps          10 \
    --lora_rank              64 \
    --lora_alpha             64 \
    --mixed_precision        bf16 \
    --gradient_checkpointing \
    --report_to              wandb \
    "$@"
