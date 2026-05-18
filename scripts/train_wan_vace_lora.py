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
from src.dataset_hdr import RawHDRPairDataset, hdr_to_ldr_batch_np, read_hdr_image_float32


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
    p.add_argument("--validation_steps",     type=int, default=500,
                   help="Run a quick few-step denoising on a fixed val sample every N steps; "
                        "save PNGs and log per-frame std + PSNR/SSIM vs GT to wandb. 0 to disable.")
    p.add_argument("--num_validation_inference_steps", type=int, default=10,
                   help="Number of denoising steps for the periodic validation render.")
    p.add_argument("--num_validation_samples", type=int, default=1,
                   help="How many noise seeds to render per val step (each seed = one strip).")
    p.add_argument("--validation_sample_index", type=int, default=0,
                   help="Dataset index of the fixed val sample (held constant across all val renders).")
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


def compute_metrics(gen_frames_uint8: "np.ndarray", gt_frames_uint8: "np.ndarray") -> dict:
    """Per-frame PSNR + SSIM between two (T, H, W, 3) uint8 arrays.  Returns
    per-frame lists + means.  PSNR computed pixel-wise on [0, 255]; SSIM via
    skimage.metrics.structural_similarity (channel-wise, win=11)."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim_fn
    assert gen_frames_uint8.shape == gt_frames_uint8.shape, \
        f"shape mismatch {gen_frames_uint8.shape} vs {gt_frames_uint8.shape}"
    T = gen_frames_uint8.shape[0]
    psnrs, ssims = [], []
    for i in range(T):
        g = gen_frames_uint8[i].astype(np.float32)
        t = gt_frames_uint8[i].astype(np.float32)
        mse = ((g - t) ** 2).mean()
        psnrs.append(99.0 if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse))
        ssims.append(float(ssim_fn(g, t, channel_axis=-1, data_range=255.0)))
    return {
        "psnr_per_frame": psnrs,
        "ssim_per_frame": ssims,
        "psnr_mean":      float(np.mean(psnrs)),
        "ssim_mean":      float(np.mean(ssims)),
    }


@torch.no_grad()
def render_validation(transformer, vae, scheduler, text_embed,
                      ref_frame_thwc: torch.Tensor,
                      latents_mean, latents_std_inv, inactive_zero_latent,
                      num_inference_steps: int, resolution: int,
                      device, dtype, noise_seed: int = 0) -> "np.ndarray":
    """Run a short denoising loop on one held-out reference image to visualise
    what the model currently generates.  Returns a (T, H, W, 3) uint8 array.

    Mirrors the training-time forward path (same conditioning, same VACE control
    convention with reactive=zeros).  Uses the pipeline's scheduler.step() so
    flow-matching is applied correctly.
    """
    import numpy as np
    transformer.eval()
    try:
        # 1. Encode the reference frame.
        ref_video = ref_frame_thwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype=dtype)  # (1, 3, 1, H, W)
        ref_latent = encode_video_with_wan_vae(vae, ref_video, latents_mean, latents_std_inv)         # (1, 16, 1, H', W')

        # 2. Init noise (pure noise as the "video" we're denoising). T_lat = inactive_zero_latent.shape[2].
        # Deterministic seed so renders are comparable across val steps — only the LoRA weights change.
        T_lat = inactive_zero_latent.shape[2]
        z_shape = (1, vae.config.z_dim, T_lat + 1, ref_latent.shape[-2], ref_latent.shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(noise_seed)
        noisy_hidden = torch.randn(z_shape, generator=gen).to(device=device, dtype=dtype)

        # 3. Build control (matches training: reactive=inactive=vae(zeros)).
        zeros_target = inactive_zero_latent.to(dtype)                                                  # (1, 16, T_lat, H', W')
        control_hs   = build_vace_control(zeros_target, ref_latent, inactive_zero_latent)              # (1, 96, T_lat+1, H', W')

        # 4. Denoising loop.
        scheduler.set_timesteps(num_inference_steps, device=device)
        text_embed_1 = text_embed[:1]                                                                  # (1, L, 4096)
        # Build added_time_ids — Wan transformer doesn't use them directly for this model but
        # forward(...) doesn't require them anyway when not provided (default None).
        for t in scheduler.timesteps:
            model_out = transformer(
                hidden_states=noisy_hidden,
                timestep=t.expand(1),
                encoder_hidden_states=text_embed_1,
                control_hidden_states=control_hs,
                return_dict=False,
            )[0]
            noisy_hidden = scheduler.step(model_out, t, noisy_hidden, return_dict=False)[0]

        # 5. Drop the reference frame (time index 0) and decode the rest.
        final_latents = noisy_hidden[:, :, 1:].float()                                                 # (1, 16, T_lat, H', W')
        final_latents = wan_vae_denormalize(final_latents, latents_mean, latents_std_inv).to(vae.dtype)
        video = vae.decode(final_latents, return_dict=False)[0]                                        # (1, 3, T, H, W)
        frames = video[0].clamp(-1, 1).permute(1, 2, 3, 0)                                             # (T, H, W, 3) in [-1, 1]
        frames = (((frames + 1) * 127.5).round().clamp(0, 255)).byte().cpu().numpy()
        return frames
    finally:
        transformer.train()


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

    # 32-ch video part: BOTH halves are vae(zeros). We deliberately do NOT put the
    # target in `reactive` — that would leak the answer through the conditioning
    # channel during training (model trivially shortcuts to copying it).  At inference
    # there's no target to put in reactive (the pipeline encodes a placeholder video),
    # so making training match means reactive = inactive = vae(zeros).  The model
    # learns to use *only* the reference image (prepended frame) as conditioning.
    inactive_part = inactive_zero_latent.expand(B, -1, -1, -1, -1).to(dt)
    reactive_part = inactive_zero_latent.expand(B, -1, -1, -1, -1).to(dt)
    body          = torch.cat([inactive_part, reactive_part], dim=1)      # (B, 32, T_lat, H', W')

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

    # Held-out validation sample — deterministically pick by INDEX into the
    # sorted pair list (bypassing the dataset's internal random.choice).
    # Load the raw EXR directly with the same 99th-percentile normalisation
    # the dataset uses, so the val reference is reproducible AND we know
    # exactly which file it is.
    if not hasattr(dataset, "pairs") or len(dataset.pairs) == 0:
        raise RuntimeError("RawHDRPairDataset has no .pairs attribute — can't pick val sample deterministically.")
    val_idx = args.validation_sample_index % len(dataset.pairs)
    val_raw_path, val_gt_path = dataset.pairs[val_idx]

    # Load + normalize the same way the dataset does (raw / p99, clipped to [-1, 1]).
    import HDRutils.io as _hdr_io
    _raw = _hdr_io.imread(val_raw_path).astype("float32")
    if _raw.ndim == 2:
        _raw = np.stack([_raw] * 3, axis=-1)
    _raw = np.maximum(_raw[..., :3], 0.0)
    _p99 = max(float(np.percentile(_raw, 99)), 1e-6)
    _raw_norm = np.clip(_raw / _p99, 0.0, 1.0)                                        # (H, W, 3) in [0, 1]

    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    from PIL import Image as _PIL
    # Save reference twice: linear (raw-looking, possibly dark) + γ-encoded for sanity.
    _PIL.fromarray((_raw_norm * 255).round().clip(0, 255).astype("uint8")).save(
        os.path.join(val_dir, "reference_linear.png"))
    _PIL.fromarray(((_raw_norm ** (1.0/2.2)) * 255).round().clip(0, 255).astype("uint8")).save(
        os.path.join(val_dir, "reference_gamma.png"))

    val_ref_thwc   = torch.from_numpy(_raw_norm)                                       # (H, W, 3) in [0, 1]
    val_ref_neg1to1 = val_ref_thwc * 2.0 - 1.0                                         # (H, W, 3) in [-1, 1]

    # ─── GT bracket for metrics ─────────────────────────────────────────
    # Tone-map the paired HDR file with DETERMINISTIC EV ladder (linspace, no jitter)
    # so the GT bracket is identical every val step.  Matches the dataset's tone-map.
    _gt = read_hdr_image_float32(val_gt_path)                                          # (H, W, 3) linear HDR
    _gt_t = torch.from_numpy(_gt).permute(2, 0, 1).contiguous()
    _Y = (_gt_t[0] * 0.2126 + _gt_t[1] * 0.7152 + _gt_t[2] * 0.0722).clamp(min=1e-6)
    _gamma  = 2.2
    _Ymax     = _Y.max().item()
    _Ymedian  = _Y.median().item()
    _start_ev = float(np.log2((0.85 ** _gamma) / _Ymax))
    _end_ev   = float(np.log2((0.85 ** _gamma) / _Ymedian))
    val_ev_values = torch.linspace(_start_ev, _end_ev, args.num_frames).numpy()
    val_gt_ldr    = hdr_to_ldr_batch_np(_gt, val_ev_values, gamma=_gamma)               # (T, H, W, 3) in [0, 1]
    val_gt_uint8  = (np.clip(val_gt_ldr, 0, 1) * 255).round().astype("uint8")            # (T, H, W, 3)
    # Save GT strip + per-frame PNGs for visual reference.
    _gt_strip = np.concatenate([val_gt_uint8[i] for i in range(val_gt_uint8.shape[0])], axis=1)
    _PIL.fromarray(_gt_strip).save(os.path.join(val_dir, "gt_strip.png"))
    for i in range(val_gt_uint8.shape[0]):
        _PIL.fromarray(val_gt_uint8[i]).save(os.path.join(val_dir, f"gt_f{i:02d}.png"))

    val_metrics_path = os.path.join(args.output_dir, "val_metrics.json")
    val_metrics_history = {"steps": [], "psnr_mean": [], "ssim_mean": [],
                           "psnr_per_frame": [], "ssim_per_frame": []}

    if accelerator.is_main_process:
        logger.info(f"Val sample idx={val_idx}/{len(dataset.pairs)}  "
                    f"→ {os.path.basename(val_raw_path)}")
        logger.info(f"  reference_linear/gamma.png + gt_strip.png + gt_f*.png saved to {val_dir}/")
        logger.info(f"  val GT EV ladder: [{_start_ev:+.2f}, ..., {_end_ev:+.2f}] across {args.num_frames} frames")

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

                # Skip the reference frame (time index 0) in the loss.  The model trivially
                # denoises position 0 back to the reference because the clean reference is
                # also in control_hidden_states[:, :, 0] — that would dominate the loss and
                # short-circuit actual generation learning.  Only score the T_lat *bracket*
                # frames at positions 1..T_lat+1.
                loss = F.mse_loss(model_out[:, :, 1:].float(), v_target[:, :, 1:].float())

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

                # ── Validation render (multi-seed + PSNR/SSIM vs GT) ───────
                if (args.validation_steps > 0
                        and global_step % args.validation_steps == 0
                        and accelerator.is_main_process):
                    import numpy as _np
                    logger.info(f"Running validation at step {global_step} "
                                f"(seeds=[0..{args.num_validation_samples-1}], "
                                f"num_inference_steps={args.num_validation_inference_steps}) …")
                    unwrapped_t = accelerator.unwrap_model(transformer)
                    step_dir = os.path.join(val_dir, f"step{global_step:06d}")
                    os.makedirs(step_dir, exist_ok=True)
                    seed_psnrs, seed_ssims = [], []
                    seed_strips = []
                    per_seed_per_frame_psnr = []
                    per_seed_per_frame_ssim = []
                    for seed in range(args.num_validation_samples):
                        frames = render_validation(
                            unwrapped_t, vae, scheduler, empty_text_embed,
                            val_ref_neg1to1.to(accelerator.device),
                            latents_mean, latents_std_inv, inactive_zero_latent,
                            num_inference_steps=args.num_validation_inference_steps,
                            resolution=args.resolution,
                            device=accelerator.device, dtype=weight_dtype,
                            noise_seed=seed,
                        )
                        # Save per-frame + strip for this seed.
                        seed_dir = os.path.join(step_dir, f"seed{seed}")
                        os.makedirs(seed_dir, exist_ok=True)
                        for i in range(frames.shape[0]):
                            _PIL.fromarray(frames[i]).save(os.path.join(seed_dir, f"f{i:02d}.png"))
                        strip = _np.concatenate([frames[i] for i in range(frames.shape[0])], axis=1)
                        _PIL.fromarray(strip).save(
                            os.path.join(val_dir, f"strip_step{global_step:06d}_seed{seed}.png"))
                        seed_strips.append(strip)
                        # Metrics vs GT.
                        m = compute_metrics(frames, val_gt_uint8)
                        seed_psnrs.append(m["psnr_mean"])
                        seed_ssims.append(m["ssim_mean"])
                        per_seed_per_frame_psnr.append(m["psnr_per_frame"])
                        per_seed_per_frame_ssim.append(m["ssim_per_frame"])

                    # Aggregate.
                    mean_psnr = float(_np.mean(seed_psnrs))
                    mean_ssim = float(_np.mean(seed_ssims))
                    best_psnr = float(_np.max(seed_psnrs))
                    best_ssim = float(_np.max(seed_ssims))
                    # Per-frame mean (over seeds).
                    pf_psnr_mean = _np.mean(per_seed_per_frame_psnr, axis=0).tolist()
                    pf_ssim_mean = _np.mean(per_seed_per_frame_ssim, axis=0).tolist()

                    logger.info(f"  PSNR per-seed (mean over frames): "
                                f"{[f'{p:.2f}' for p in seed_psnrs]}  "
                                f"→ mean {mean_psnr:.2f}  best {best_psnr:.2f}")
                    logger.info(f"  SSIM per-seed (mean over frames): "
                                f"{[f'{s:.3f}' for s in seed_ssims]}  "
                                f"→ mean {mean_ssim:.3f}  best {best_ssim:.3f}")
                    logger.info(f"  PSNR per-frame (mean over seeds): "
                                f"{[f'{p:.2f}' for p in pf_psnr_mean]}")

                    # Append history + persist.
                    val_metrics_history["steps"].append(global_step)
                    val_metrics_history["psnr_mean"].append(mean_psnr)
                    val_metrics_history["ssim_mean"].append(mean_ssim)
                    val_metrics_history["psnr_per_frame"].append(pf_psnr_mean)
                    val_metrics_history["ssim_per_frame"].append(pf_ssim_mean)
                    import json as _json
                    with open(val_metrics_path, "w") as _f:
                        _json.dump(val_metrics_history, _f, indent=2)

                    # Scalars → accelerator log (goes to whichever tracker is active).
                    val_payload = {
                        "val/psnr_mean":  mean_psnr,
                        "val/psnr_best":  best_psnr,
                        "val/ssim_mean":  mean_ssim,
                        "val/ssim_best":  best_ssim,
                    }
                    for i in range(len(pf_psnr_mean)):
                        val_payload[f"val/f{i}/psnr"] = pf_psnr_mean[i]
                        val_payload[f"val/f{i}/ssim"] = pf_ssim_mean[i]
                    accelerator.log(val_payload, step=global_step)

                    # Image → direct wandb (accelerator.log doesn't support wandb.Image).
                    if args.report_to in ("wandb", "all"):
                        try:
                            import wandb as _wandb
                            if _wandb.run is not None:
                                img_payload = {}
                                for seed_i, strip in enumerate(seed_strips):
                                    img_payload[f"val/strip_seed{seed_i}"] = _wandb.Image(
                                        strip, caption=f"step {global_step} seed {seed_i} | psnr {seed_psnrs[seed_i]:.2f}")
                                _wandb.log(img_payload, step=global_step)
                        except Exception as e:
                            logger.warning(f"wandb val strip image log skipped: {e}")

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
