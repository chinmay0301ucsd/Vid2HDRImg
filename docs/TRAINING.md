# Training

> **Note:** this training code is released as-is, matching the configuration
> used for the released checkpoints. It hasn't received the same testing as
> the inference path, so treat it as a reference implementation and expect
> to do some debugging when adapting it to a new dataset or environment.

The pipeline has two independent stages. Train them in order.

There are two ways to launch training:
1. **Bash wrappers** (`scripts/run_train_video_model.sh`,
   `scripts/run_train_fusion_net.sh`) — common knobs exposed as env vars,
   sensible defaults, recommended for getting started.
2. **Direct `accelerate launch`** — full flag surface, recommended once you
   know the codebase.

The bash wrappers ultimately call the Python scripts via `accelerate launch`,
so they're functionally equivalent.

## Stage 1 — fine-tune SVD UNet

`scripts/train_video_model.py` fine-tunes the Stable Video Diffusion UNet
to generate an LDR exposure bracket conditioned on a single input frame.
The conditioning anchor is randomly chosen each step from start / mid /
end of the bracket, and the conditioning latent is repeated at all
temporal positions so the model is robust to whichever exposure the
input image happens to be at.

### Single-GPU example (bash wrapper)

```bash
TRAIN_DATA_PATH=/path/to/hdr_dataset \
VALID_HDR_PATH=/path/to/validation.hdr \
    bash scripts/run_train_video_model.sh
```

Override any default by setting more env vars: e.g.
`MAX_STEPS=50000 BSZ=2 LR=2e-5 bash scripts/run_train_video_model.sh`.

### Single-GPU example (direct)

The flags below match the config that trained the released checkpoint
(`svd_hdr_raw_pair_no_cfg_nolora/checkpoint-16000`) exactly:

```bash
accelerate launch scripts/train_video_model.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
    --train_data_path /path/to/hdr_dataset \
    --output_dir      output/stage1 \
    --valid_hdr_path  /path/to/validation.hdr \
    --num_frames 5 --width 512 --height 512 \
    --learning_rate 5e-5 \
    --per_gpu_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_warmup_steps 100 \
    --mixed_precision no \
    --seed 123 \
    --conditioning_dropout_prob 0.0 \
    --max_train_steps 30000 \
    --validation_steps 1000 \
    --checkpointing_steps 1000
```

> `--conditioning_dropout_prob 0.0` disables CFG training entirely — the
> released checkpoint is a "no_cfg" model, matched by `inference.py`'s
> `--max_guidance_scale 1.0` default. Raise it above 0.0 only if you intend to
> train a CFG-capable model (and use `--max_guidance_scale > 1.0` at inference).

### Multi-GPU example

With the wrapper:

```bash
NUM_PROCESSES=4 TRAIN_DATA_PATH=/path/to/hdr_dataset \
    bash scripts/run_train_video_model.sh
```

Or directly (run `accelerate config` once first to set DDP/FSDP):

```bash
accelerate launch --num_processes 4 scripts/train_video_model.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
    --train_data_path /path/to/hdr_dataset \
    --output_dir      output/stage1 \
    --num_frames 5 --width 512 --height 512 \
    --learning_rate 5e-5 \
    --per_gpu_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_warmup_steps 100 \
    --mixed_precision no \
    --seed 123 \
    --conditioning_dropout_prob 0.0 \
    --max_train_steps 30000
```

### Useful flags

| flag | what |
|---|---|
| `--raw_pair_data_path` | Use real raw / GT-HDR pairs (`RawHDRPairDataset`) instead of the synthetic-bracket dataset; raw frame becomes the conditioning anchor. |
| `--use_lora --lora_rank R` | LoRA-adapt the UNet attention layers (memory-efficient FT). |
| `--gradient_checkpointing` | Re-compute activations to halve memory cost at ~30% speed cost. |
| `--mixed_precision no\|bf16\|fp16` | The released checkpoint was trained with `no` (full fp32); bf16 is a faster but not bit-identical alternative. |
| `--compile_unet` | Apply `torch.compile` to the UNet (use `--compile_backend aot_eager` if Inductor fails). |
| `--conditioning_dropout_prob` | CFG training dropout probability. **0.0 for the released "no_cfg" checkpoint** — its argparse default (0.1) is for training a CFG-capable model instead. |
| `--seed 123` | Reproducibility — 123 matches the released checkpoint's training run. |

Run `python scripts/train_video_model.py --help` for the full surface.

## Stage 2 — train the fusion U-Net

`scripts/train_fusion_net.py` trains a small per-pixel U-Net that takes a
stack of LDR frames at increasing exposures and outputs blending weights,
which combine into the final HDR image. Loss is MSE in PU-21 perceptual
space against the GT HDR.

Stage 2 does **not** use a VAE or diffusion model — it operates entirely
in pixel space and trains in roughly an hour on a single GPU.

With the wrapper:

```bash
TRAIN_DATA_PATH=/path/to/hdr_dataset \
    bash scripts/run_train_fusion_net.sh
```

Or directly. The flags below match the config that trained the released
fusion net exactly (`output_svd_hdr_fus_only_combined/checkpoint-25000`):

```bash
accelerate launch scripts/train_fusion_net.py \
    --train_data_path /path/to/hdr_dataset \
    --output_dir      output/stage2 \
    --num_frames 5 --width 512 --height 512 \
    --learning_rate 1e-5 \
    --per_gpu_batch_size 16 \
    --random_crop \
    --seed 123 \
    --max_train_steps 25000 \
    --checkpointing_steps 1000
```

## Putting them together at inference time

After both stages finish you'll have:

```
output/stage1/checkpoint-XXXXX/unet/        # fine-tuned UNet
output/stage2/checkpoint-YYYYY/fusion_net.pt # fusion-net weights
```

Pass these to `scripts/inference.py`:

```bash
python scripts/inference.py \
    --input  example.jpg \
    --unet_path        output/stage1/checkpoint-XXXXX \
    --fusion_net_path  output/stage2/checkpoint-YYYYY/fusion_net.pt \
    --output predicted.exr
```
