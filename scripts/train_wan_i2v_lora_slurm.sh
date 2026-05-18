#!/bin/bash
#SBATCH -J wan_i2v_lora                 # Job name
#SBATCH -o watch_folder/%x_%j.out       # output file (%j expands to jobID)
#SBATCH -N 1                            # Total number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                         # all available server memory
#SBATCH -t 11:59:00                     # hh:mm:ss
#SBATCH --partition=mi2101x             # Single MI210; override to mi2508x for 8-GCD data-parallel
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

source /work1/javidi/chinmay0301/miniconda3/etc/profile.d/conda.sh
conda activate vace_hdr                 # diffusers 0.37.1, Python 3.10

export WANDB_API_KEY="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"
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=300
export WANDB_PROJECT="wan-i2v-lora"

NUM_GPUS="${NUM_GPUS:-1}"

PRETRAINED_WAN="/work1/javidi/shared/Wan2.1-I2V-14B-480P-Diffusers"
TRAIN_DATA_PATH="/work1/javidi/shared/raw_hdr_dataset_v1"
OUTPUT_DIR="/work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_i2v_lora_v0"

mkdir -p watch_folder "$OUTPUT_DIR"
cd /work1/javidi/chinmay0301/Vid2HDRImg

srun accelerate launch --num_processes "$NUM_GPUS" scripts/train_wan_i2v_lora.py \
    --pretrained_wan_path        "$PRETRAINED_WAN" \
    --data_dir                   "$TRAIN_DATA_PATH" \
    --output_dir                 "$OUTPUT_DIR" \
    --num_frames                 5 \
    --resolution                 512 \
    --learning_rate              5e-5 \
    --train_batch_size           1 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm              1.0 \
    --max_train_steps            20000 \
    --lr_warmup_steps            200 \
    --checkpointing_steps        1000 \
    --logging_steps              10 \
    --validation_steps           200 \
    --num_validation_inference_steps 20 \
    --num_validation_samples     1 \
    --validation_sample_index    0 \
    --lora_rank                  64 \
    --lora_alpha                 64 \
    --mixed_precision            bf16 \
    --gradient_checkpointing \
    --report_to                  wandb \
    "$@"
