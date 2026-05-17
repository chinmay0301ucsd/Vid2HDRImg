"""LoRA fine-tune of Wan 2.1 VACE 14B for raw → LDR-bracket.

Mirrors the design of train_video_model.py (RawHDRPairDataset; raw image as
conditioning; predict a multi-frame LDR bracket), but:

  - Backbone is WanVACETransformer3DModel (~15.8 B params) — LoRA-only.
  - VAE is AutoencoderKLWan: input (B, 3, T, H, W) → latent (B, 16, T/4, H/8, W/8)
    with per-channel normalisation (`latents_mean`, `latents_std`).
  - Text conditioning is **disabled** (empty prompt; UMT5 still runs once at
    init and the embedding is cached for the whole training run).
  - Image conditioning uses VACE's V stream — the raw input image is
    VAE-encoded and fed via `control_hidden_states`.
  - Loss is flow-matching velocity MSE (predict v = noise - x0).
  - LoRA rank = 64, targets the self-/cross-attention modules in the main
    blocks (and the parallel VACE blocks).

Stage 1 of the planned port — text/EV conditioning, multi-reference, and
sigma-aware loss-weighting can be added later if needed.

Usage example (single MI210, batch=1):

    accelerate launch --num_processes 1 train_wan_vace_lora.py \\
        --pretrained_wan_path /work1/javidi/chinmay0301/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B-Diffusers/snapshots/db79b90c60bbb45ceec9e41b9d5a4df934538ac4 \\
        --data_dir /work1/javidi/shared/raw_hdr_dataset_v1 \\
        --output_dir /work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_vace_lora_v0 \\
        --num_frames 13 --resolution 512 --train_batch_size 1 \\
        --learning_rate 1e-4 --lora_rank 64 \\
        --gradient_checkpointing --mixed_precision bf16
"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanVACETransformer3DModel,
)
from diffusers.optimization import get_scheduler
from transformers import T5TokenizerFast, UMT5EncoderModel

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict as _peft_state

# Local dataset (must be importable from the Vid2HDRImg/scripts/ path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset_hdr import RawHDRPairDataset


logger = logging.getLogger(__name__)


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # paths
    p.add_argument("--pretrained_wan_path", required=True,
                   help="Path to the Wan 2.1 VACE 14B diffusers snapshot directory "
                        "(contains transformer/, vae/, text_encoder/, scheduler/, tokenizer/).")
    p.add_argument("--data_dir", required=True,
                   help="Path to raw_hdr_dataset_v1 (must contain raw/ and gt_hdr/).")
    p.add_argument("--output_dir", required=True,
                   help="Directory for LoRA checkpoints, logs.")

    # data
    p.add_argument("--num_frames",    type=int, default=5,
                   help="Frames per training video (LDR bracket size). Wan VAE has 4x temporal "
                        "compression, so latent T_lat will be ~ceil(T/4): T=5 → T_lat=2, "
                        "T=9 → T_lat=3, T=13 → T_lat=4, T=17 → T_lat=5.")
    p.add_argument("--resolution",    type=int, default=512)
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    # training
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps",  type=int, default=20000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--lr_scheduler",     default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps",  type=int, default=200)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--adam_beta1",       type=float, default=0.9)
    p.add_argument("--adam_beta2",       type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon",     type=float, default=1e-8)
    p.add_argument("--seed",             type=int, default=0)

    # LoRA
    p.add_argument("--lora_rank",        type=int, default=64)
    p.add_argument("--lora_alpha",       type=int, default=64,
                   help="Defaults to rank (alpha = rank → scale = 1).")
    p.add_argument("--lora_dropout",     type=float, default=0.0)

    # misc
    p.add_argument("--mixed_precision",  default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Strongly recommended for 14B model + LoRA.")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--logging_steps",        type=int, default=10)
    p.add_argument("--report_to",            default="tensorboard",
                   choices=["tensorboard", "wandb", "all", "none"])
    p.add_argument("--use_noisy_samples",    action="store_true", default=True,
                   help="Pass through to RawHDRPairDataset; matches train_video_model.py.")
    return p.parse_args()


# ─── Conditioning helpers ───────────────────────────────────────────────────

@torch.no_grad()
def encode_text_empty(tokenizer, text_encoder, device, dtype, max_length: int = 512):
    """Run UMT5 on an empty prompt once and cache the result.

    Returns: (1, L, 4096) tensor.  We'll repeat across the batch dim per step
    so we never invoke UMT5 in the training loop.
    """
    inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    embeds = out.last_hidden_state                                        # (1, L, 4096)
    return embeds.to(dtype=dtype)


def wan_vae_normalize(latents: torch.Tensor, mean: torch.Tensor, std_inv: torch.Tensor) -> torch.Tensor:
    """Per-channel normalisation matching the Wan VAE convention."""
    return (latents.float() - mean) * std_inv


def wan_vae_denormalize(latents: torch.Tensor, mean: torch.Tensor, std_inv: torch.Tensor) -> torch.Tensor:
    return latents.float() / std_inv + mean


@torch.no_grad()
def encode_video_with_wan_vae(vae, video_thwc_or_btchw: torch.Tensor,
                              mean: torch.Tensor, std_inv: torch.Tensor) -> torch.Tensor:
    """Encode a (B, 3, T, H, W) video into normalised latents (B, 16, T_lat, H', W')."""
    assert video_thwc_or_btchw.dim() == 5, f"want (B, 3, T, H, W), got {video_thwc_or_btchw.shape}"
    posterior = vae.encode(video_thwc_or_btchw).latent_dist
    z = posterior.mode()                                                  # (B, 16, T_lat, H', W')
    return wan_vae_normalize(z, mean, std_inv).to(video_thwc_or_btchw.dtype)


def build_vace_control(target_latents: torch.Tensor,
                       reference_latent: torch.Tensor,
                       inactive_zero_latent: torch.Tensor) -> torch.Tensor:
    """Construct VACE's 96-channel `control_hidden_states` matching
    pipeline_wan_vace.py's convention.

    Channel layout (concat on channel axis):
      - [0..31]  : video part = cat([inactive, reactive], dim=ch)
                   - inactive = vae.encode(zeros_video) per the pipeline's mask=ones path
                   - reactive = target_latents (the bracket we want the model to generate)
      - [32..95] : mask part = 0 at reference frame, 1 at reactive frames (8x8 = 64 sub-pixel channels)

    Temporal layout (concat on time axis):
      - Reference is prepended as a single latent frame (zero-padded to 32 ch on the video side,
        all-zero on the mask side).

    target_latents        : (B, 16, T_lat, H', W')   — vae-encoded bracket (normalised)
    reference_latent      : (B, 16, 1,     H', W')   — vae-encoded raw image (normalised)
    inactive_zero_latent  : (1, 16, T_lat, H', W')   — vae.encode(zeros_video) cached once

    returns               : (B, 96, T_lat + 1, H', W')
    """
    B, C, T_lat, H, W = target_latents.shape
    dev, dt = target_latents.device, target_latents.dtype

    # 32-ch video part: inactive (cached vae(zeros)) + reactive (target).
    inactive_part = inactive_zero_latent.expand(B, -1, -1, -1, -1).to(dt)
    body          = torch.cat([inactive_part, target_latents], dim=1)     # (B, 32, T_lat, H', W')

    # Reference frame: zero-pad to 32 ch, prepend along time.
    ref_pad   = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=1)  # (B, 32, 1, H', W')
    video_part = torch.cat([ref_pad, body], dim=2)                                         # (B, 32, T_lat+1, H', W')

    # 64-ch mask part: 0 at reference frame (time 0), 1 elsewhere (we're generating everything).
    mask_part = torch.zeros(B, 64, T_lat + 1, H, W, device=dev, dtype=dt)
    mask_part[:, :, 1:] = 1.0

    return torch.cat([video_part, mask_part], dim=1)                                       # (B, 96, T_lat+1, H', W')


def build_hidden_states_with_ref(target_latents: torch.Tensor,
                                 reference_latent: torch.Tensor) -> torch.Tensor:
    """Prepend reference along time so both hidden_states and control share T_lat+1.

    target_latents   : (B, 16, T_lat, H', W')
    reference_latent : (B, 16, 1,     H', W')
    returns          : (B, 16, T_lat+1, H', W')
    """
    return torch.cat([reference_latent, target_latents], dim=2)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_dir=args.output_dir,
    )
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = {
        "no":   torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]

    if accelerator.is_main_process:
        logger.info(f"Wan path     : {args.pretrained_wan_path}")
        logger.info(f"Data dir     : {args.data_dir}")
        logger.info(f"Output dir   : {args.output_dir}")
        logger.info(f"Weight dtype : {weight_dtype}")
        logger.info(f"LoRA rank    : {args.lora_rank}")

    # ─── Models ───────────────────────────────────────────────────────────
    logger.info("Loading Wan components …")

    tokenizer = T5TokenizerFast.from_pretrained(args.pretrained_wan_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_wan_path, subfolder="text_encoder", torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_wan_path, subfolder="vae", torch_dtype=weight_dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_wan_path, subfolder="scheduler",
    )
    transformer = WanVACETransformer3DModel.from_pretrained(
        args.pretrained_wan_path, subfolder="transformer", torch_dtype=weight_dtype,
    )

    # Freeze everything; LoRA on transformer only.
    for m in (vae, text_encoder, transformer):
        m.requires_grad_(False)
    vae.eval(); text_encoder.eval()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Per-channel VAE normalisation tensors (kept on whatever device the VAE is on).
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std  = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / latents_std

    # ─── LoRA injection ───────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Both main blocks and VACE blocks have attn1/attn2 with to_q/k/v/to_out.0.
        # peft matches by suffix; this catches all of them across blocks[0..39] and vace_blocks[0..7].
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)

    # Set LoRA params to float32 for stable optimiser updates (PEFT recommended pattern).
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    for p in trainable_params:
        p.data = p.data.to(torch.float32)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in transformer.parameters())
    if accelerator.is_main_process:
        logger.info(f"Transformer params: {n_total/1e9:.2f} B total, "
                    f"{n_trainable/1e6:.2f} M trainable LoRA ({100*n_trainable/n_total:.3f}%)")

    # Move models to device.
    transformer.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    latents_mean    = latents_mean.to(accelerator.device)
    latents_std_inv = latents_std_inv.to(accelerator.device)

    # Cache the empty-prompt embedding once.
    empty_text_embed = encode_text_empty(
        tokenizer, text_encoder, accelerator.device, weight_dtype,
        max_length=512,
    )                                                                    # (1, L, 4096)
    # We can drop the text encoder from memory now.
    del text_encoder
    torch.cuda.empty_cache()

    # Cache vae.encode(zeros_video) once: the VACE pipeline uses this as the
    # "inactive" part of the control stream when generating everything (mask=ones).
    with torch.no_grad():
        zeros_video = torch.zeros(
            1, 3, args.num_frames, args.resolution, args.resolution,
            device=accelerator.device, dtype=weight_dtype,
        )
        inactive_zero_latent = encode_video_with_wan_vae(
            vae, zeros_video, latents_mean, latents_std_inv,
        )                                                                # (1, 16, T_lat, H', W')
    if accelerator.is_main_process:
        logger.info(f"Cached inactive_zero_latent: shape={tuple(inactive_zero_latent.shape)}")
    if accelerator.is_main_process:
        logger.info(f"Cached empty text embedding: shape={tuple(empty_text_embed.shape)} "
                    f"dtype={empty_text_embed.dtype}.  Text encoder unloaded.")

    # ─── Data ─────────────────────────────────────────────────────────────
    dataset = RawHDRPairDataset(
        base_folder=args.data_dir,
        sample_frames=args.num_frames,
        use_noisy_samples=args.use_noisy_samples,
    )
    train_dl = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, drop_last=True, pin_memory=True,
    )

    # ─── Optimiser / scheduler ────────────────────────────────────────────
    optim = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optim,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optim, train_dl, lr_scheduler = accelerator.prepare(
        transformer, optim, train_dl, lr_scheduler,
    )
    accelerator.init_trackers("wan_vace_lora", config=vars(args))

    # ─── Training loop ────────────────────────────────────────────────────
    global_step = 0
    transformer.train()

    logger.info("Starting training …")
    while global_step < args.max_train_steps:
        for batch in train_dl:
            with accelerator.accumulate(transformer):
                # pixel_values_clean: (B, T, 3, H, W) in [-1, 1]
                # raw_input         : (B, 3, H, W) in [-1, 1]
                clean   = batch["pixel_values_clean"].to(accelerator.device, dtype=weight_dtype)
                ref_img = batch["raw_input"].to(accelerator.device, dtype=weight_dtype)
                B, T, C_in, H, W = clean.shape

                # 1. VAE-encode the bracket as a video: permute to (B, 3, T, H, W).
                video = clean.permute(0, 2, 1, 3, 4).contiguous()
                target_latents = encode_video_with_wan_vae(
                    vae, video, latents_mean, latents_std_inv,
                )                                                         # (B, 16, T_lat, H/8, W/8)

                # 2. Encode the raw reference image as a 1-frame video.
                ref_video = ref_img[:, :, None, :, :]                     # (B, 3, 1, H, W)
                reference_latent = encode_video_with_wan_vae(
                    vae, ref_video, latents_mean, latents_std_inv,
                )                                                         # (B, 16, 1, H/8, W/8)

                # 3. Build VACE-style inputs (prepend reference along time).
                hidden_clean = build_hidden_states_with_ref(target_latents, reference_latent)
                control_hs   = build_vace_control(target_latents, reference_latent, inactive_zero_latent)

                # 4. Sample timestep (uniform over the scheduler's flow-matching range).
                noise = torch.randn_like(hidden_clean)
                bsz   = hidden_clean.shape[0]
                indices = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,),
                                        device=hidden_clean.device, dtype=torch.long)
                # FlowMatchEuler: sigma = indices / num_train_timesteps (in [0, 1)).
                sigmas = (indices.float() / scheduler.config.num_train_timesteps)
                sigmas_v = sigmas.view(bsz, 1, 1, 1, 1)
                noisy_hidden = (1.0 - sigmas_v) * hidden_clean + sigmas_v * noise
                # Velocity target: v = noise - x_0  (flow matching convention).
                v_target = noise - hidden_clean

                # 5. Forward.
                timestep = (sigmas * scheduler.config.num_train_timesteps).long()
                text_embed = empty_text_embed.expand(bsz, -1, -1)

                model_out = transformer(
                    hidden_states=noisy_hidden,
                    timestep=timestep,
                    encoder_hidden_states=text_embed,
                    control_hidden_states=control_hs,
                    return_dict=False,
                )[0]                                                      # (B, 16, T_lat+1, H', W')

                loss = F.mse_loss(model_out.float(), v_target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()
                lr_scheduler.step()
                optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    logger.info(f"step {global_step}/{args.max_train_steps}  "
                                f"loss={loss.item():.4f}  lr={lr_scheduler.get_last_lr()[0]:.2e}")
                accelerator.log({"train/loss": loss.item(),
                                 "train/lr":   lr_scheduler.get_last_lr()[0]},
                                step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"lora-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(transformer)
                    lora_sd = get_peft_model_state_dict(unwrapped)
                    torch.save({"lora_state_dict": lora_sd,
                                "config": vars(args),
                                "step":   global_step},
                               os.path.join(ckpt_dir, "lora_weights.pt"))
                    logger.info(f"Saved LoRA checkpoint → {ckpt_dir}")

                if global_step >= args.max_train_steps:
                    break

    # Final save
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        lora_sd = get_peft_model_state_dict(unwrapped)
        final_dir = os.path.join(args.output_dir, "lora-final")
        os.makedirs(final_dir, exist_ok=True)
        torch.save({"lora_state_dict": lora_sd,
                    "config": vars(args),
                    "step":   global_step},
                   os.path.join(final_dir, "lora_weights.pt"))
        logger.info(f"Saved final LoRA checkpoint → {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
