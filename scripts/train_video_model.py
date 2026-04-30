"""Stage 1 of 2 — fine-tune SVD UNet on synthetic HDR exposure brackets.

Each training step randomly picks one of three EV frames (start / mid / end)
as the conditioning anchor. Both the CLIP image embedding and the VAE
conditioning latent refer to that chosen frame, and the VAE latent is
REPEATED at all temporal positions so the model sees the full conditioning
signal at every frame (rather than placing it at one position with zeros
elsewhere). This makes the model robust to whichever exposure level the
input image happens to be at.

Stage 2 (the lightweight fusion U-Net that takes the synthesised LDR stack
and produces an HDR output) is trained separately by ``train_fusion_net.py``.

Datasets supported:
  - ``RepeatedHDRVideoDataset``: tone-mapped synthetic exposure brackets from
    a folder of HDR EXR files (default).
  - ``RawHDRPairDataset``: real raw camera captures paired with GT HDR images;
    the raw frame is used as the conditioning anchor in place of a tone-mapped
    LDR. Activated by ``--raw_pair_data_path``.
"""

import argparse
import json 
import logging
import math
import os
import shutil
import cv2
from pathlib import Path

import accelerate
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from einops import rearrange

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, load_image

from src.dataset_hdr import RepeatedHDRVideoDataset, RawHDRPairDataset, read_hdr_image_float32, hdr_to_ldr_batch_np
from src.pipelines.pipeline_stable_video_diffusion_hdr import StableVideoDiffusionPipelineHDR
import pyexr

if is_wandb_available():
    import wandb

check_min_version("0.24.0.dev0")

# LoRA / PEFT support (optional)
try:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = get_logger(__name__, log_level="INFO")

import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import pyiqa

mse_loss_fn = nn.MSELoss()
ssim_metric_fn = pyiqa.create_metric('ssim', device='cuda', as_loss=False)
lpips_metric = pyiqa.create_metric('lpips', device='cuda', as_loss=False)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t.float(), "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from a log-normal distribution (EDM framework)."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
    pil_frames[0].save(
        output_gif_path.replace('.mp4', '.gif'),
        format='GIF',
        append_images=pil_frames[1:],
        save_all=True,
        duration=int(1000 / fps),
        loop=0,
    )


def calculate_psnr(vid1, vid2):
    to_tensor = transforms.ToTensor()
    psnr_list = []
    for img1, img2 in zip(vid1, vid2):
        t1 = to_tensor(img1) if not isinstance(img1, torch.Tensor) else img1
        t2 = to_tensor(img2) if not isinstance(img2, torch.Tensor) else img2
        mse = mse_loss_fn(t1, t2)
        psnr_list.append(100.0 if mse == 0 else 10 * torch.log10(1 / mse).item())
    return float(np.mean(psnr_list))


def calculate_ssim(vid1, vid2):
    to_tensor = transforms.ToTensor()
    tensors1 = torch.stack([to_tensor(f) if not isinstance(f, torch.Tensor) else f for f in vid1])
    tensors2 = torch.stack([to_tensor(f) if not isinstance(f, torch.Tensor) else f for f in vid2])
    return float(ssim_metric_fn(tensors1, tensors2).detach().cpu().mean())


def calculate_lpips(vid1, vid2):
    to_tensor = transforms.ToTensor()
    tensors1 = torch.stack([to_tensor(f) if not isinstance(f, torch.Tensor) else f for f in vid1])
    tensors2 = torch.stack([to_tensor(f) if not isinstance(f, torch.Tensor) else f for f in vid2])
    return float(lpips_metric(tensors1, tensors2).detach().cpu().mean())


def classify_ev_level(img_np, sat_thresh=0.95, dark_thresh=0.05,
                      sat_frac_thresh=0.05, dark_frac_thresh=0.30,
                      lum_high_thresh=0.55):
    """Classify an LDR image into one of three EV levels.

    Returns an integer index:
      0 — start EV (dark/underexposed → captured at low EV → frame 0)
      1 — mid   EV (well-exposed or high-contrast)
      2 — end   EV (saturated/overexposed → captured at high EV → frame T-1)

    Logic:
      - END EV requires BOTH high sat_frac AND high mean luminance.
        Using sat_frac alone fires on high-contrast mid-EV scenes that have
        local highlights (bright patches) but are not globally overexposed.
      - High dark fraction → start EV (globally dark/underexposed).
      - Otherwise → mid EV.
    """
    gray = (0.2126 * img_np[..., 0]
            + 0.7152 * img_np[..., 1]
            + 0.0722 * img_np[..., 2])
    sat_frac  = float(np.mean(gray > sat_thresh))
    dark_frac = float(np.mean(gray < dark_thresh))
    lum_mean  = float(gray.mean())
    if sat_frac > sat_frac_thresh and lum_mean > lum_high_thresh:
        return 2   # end EV — globally bright/overexposed
    elif dark_frac > dark_frac_thresh:
        return 0   # start EV — darkest frame
    else:
        return 1   # mid EV


def validate_once(val_save_dir, accelerator, pipeline, args, global_step, valid_hdr_path,
                   valid_raw_path=None, valid_gt_hdr_path=None):
    """Run one validation pass.

    Two modes:
      - Raw-pair mode (valid_raw_path + valid_gt_hdr_path): uses the real raw EXR as the
        conditioning frame and the GT HDR EXR for target LDR frames. Crops are already
        512×512 so no resize/crop is applied.
      - Synthetic mode (valid_hdr_path): tone-maps a single HDR file and picks a conditioning
        frame from the tone-mapped sequence (original behavior).
    """
    gamma = 2.2

    if valid_raw_path is not None and valid_gt_hdr_path is not None:
        # ── Raw-pair validation ────────────────────────────────────────────────
        raw_img = read_hdr_image_float32(valid_raw_path)   # (H, W, 3) linear sRGB
        gt_img  = read_hdr_image_float32(valid_gt_hdr_path)  # (H, W, 3) linear HDR

        # Save GT HDR EXR once
        gt_hdr_save = os.path.join(val_save_dir, "gt_hdr.exr")
        if not os.path.exists(gt_hdr_save):
            pyexr.write(gt_hdr_save, gt_img.astype(np.float32))
            logger.info(f"Saved GT HDR EXR: {gt_hdr_save}")

        # EV ladder from GT HDR (same as RawHDRPairDataset)
        Y = (gt_img[..., 0] * 0.2126 + gt_img[..., 1] * 0.7152 + gt_img[..., 2] * 0.0722).clip(min=1e-6)
        Ymax    = Y.max()
        Ymedian = np.median(Y)
        start_ev = np.log2((0.85 ** gamma) / Ymax)
        end_ev   = np.log2((0.85 ** gamma) / Ymedian)
        ev_values = np.linspace(start_ev, end_ev, args.num_frames)
        ldr_batch = hdr_to_ldr_batch_np(gt_img, ev_values, gamma=gamma)
        gt_frames = [(ldr_batch[i] * 255).clip(0, 255).astype(np.uint8) for i in range(args.num_frames)]

        # Save GT LDR frames once
        gt_ldr_path = os.path.join(val_save_dir, "gt_ldr.gif")
        if not os.path.exists(gt_ldr_path):
            export_to_gif(gt_frames, gt_ldr_path, 8)
            logger.info(f"Saved GT LDR frames: {gt_ldr_path}")

        # Conditioning frame: raw image normalized to [0, 1] via p99
        p99 = float(np.percentile(raw_img, 99))
        p99 = max(p99, 1e-6)
        raw_norm = np.clip(raw_img / p99, 0.0, 1.0)
        cond_frame_pil = Image.fromarray((raw_norm * 255).clip(0, 255).astype(np.uint8))
        # Use mid EV as conditioning_frame_idx (raw has no known EV bin)
        cond_fi = args.num_frames // 2
        logger.info(f"[validate_once] raw-pair mode: conditioning on raw image, cond_fi={cond_fi}")

    else:
        # ── Synthetic validation ───────────────────────────────────────────────
        img = read_hdr_image_float32(valid_hdr_path)

        # Resize + center-crop to target resolution
        h, w = img.shape[:2]
        scale = max(args.width / w, args.height / h)
        img_resized = cv2.resize(img, (int(w * scale + 0.5), int(h * scale + 0.5)), interpolation=cv2.INTER_AREA)
        h2, w2 = img_resized.shape[:2]
        sy = (h2 - args.height) // 2
        sx = (w2 - args.width) // 2
        img_crop = img_resized[sy: sy + args.height, sx: sx + args.width]

        # Save GT HDR EXR once
        gt_hdr_path = os.path.join(val_save_dir, "gt_hdr.exr")
        if not os.path.exists(gt_hdr_path):
            pyexr.write(gt_hdr_path, img_crop.astype(np.float32))
            logger.info(f"Saved GT HDR EXR: {gt_hdr_path}")

        # Tone-map to LDR frames
        Y = (img_crop[..., 0] * 0.2126 + img_crop[..., 1] * 0.7152 + img_crop[..., 2] * 0.0722).clip(min=1e-6)
        Ymax    = Y.max()
        Ymedian = np.median(Y)
        start_ev = np.log2((0.85 ** gamma) / Ymax)
        end_ev   = np.log2((0.85 ** gamma) / Ymedian)
        ev_values = np.linspace(start_ev, end_ev, args.num_frames)
        ldr_batch = hdr_to_ldr_batch_np(img_crop, ev_values, gamma=gamma)
        gt_frames = [(ldr_batch[i] * 255).clip(0, 255).astype(np.uint8) for i in range(args.num_frames)]

        gt_ldr_path = os.path.join(val_save_dir, "gt_ldr.gif")
        if not os.path.exists(gt_ldr_path):
            export_to_gif(gt_frames, gt_ldr_path, 8)
            logger.info(f"Saved GT LDR frames: {gt_ldr_path}")

        # Choose conditioning frame: classify the synthesized first frame's EV
        # bin (start/mid/end) and condition on the matching frame in the bracket.
        cond_frame_map = [0, args.num_frames // 2, args.num_frames - 1]
        ev_choice = classify_ev_level(gt_frames[0].astype(np.float32) / 255.0)
        cond_fi = cond_frame_map[ev_choice]
        ev_labels = ["start", "mid", "end"]
        logger.info(f"[validate_once] classified first frame as '{ev_labels[ev_choice]}' EV → conditioning on frame {cond_fi}")

        cond_frame_pil = Image.fromarray(gt_frames[cond_fi])

    _autocast_dtype = (
        torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16
    )
    with torch.autocast(
        str(accelerator.device).replace(":0", ""),
        dtype=_autocast_dtype,
        enabled=accelerator.mixed_precision in ("fp16", "bf16"),
    ):
        for val_idx in range(args.num_validation_images):
            video_frames = pipeline(
                cond_frame_pil,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=8,
                motion_bucket_id=0,
                fps=7,
                noise_aug_strength=0.0,  # disabled; set to 0.02 to enable small Gaussian noise on conditioning
                conditioning_frame_idx=None,  # repeat variant: conditioning latent repeated at all T positions
            ).frames[0]

            out_file = os.path.join(val_save_dir, f"step_{global_step}_val_{val_idx}.mp4")
            video_frames_np = [np.array(f) for f in video_frames]
            export_to_gif(video_frames_np, out_file, 8)

    return video_frames_np, gt_frames, cond_fi


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SVD UNet on HDR video data.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained SVD model (e.g. stabilityai/stable-video-diffusion-img2vid-xt).")
    parser.add_argument("--pretrain_unet", type=str, default=None,
                        help="Optional path to a pre-fine-tuned UNet to resume from.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to folder containing HDR files for training.")
    parser.add_argument("--random_crop", action="store_true", default=False,
                        help="Use random crops during training. Default: center crop.")
    parser.add_argument("--valid_hdr_path", type=str, default=None,
                        help="Path to a single HDR file used for periodic validation (synthetic mode).")
    parser.add_argument("--valid_raw_path", type=str, default=None,
                        help="Path to a raw EXR (from raw_hdr_dataset) for raw-pair validation.")
    parser.add_argument("--valid_gt_hdr_path", type=str, default=None,
                        help="Path to the GT HDR EXR paired with --valid_raw_path.")
    parser.add_argument("--exclude_files", type=str, nargs="*", default=None,
                        help="Filenames (basenames) to exclude from the training set. "
                             "E.g. --exclude_files 08155.hdr  to hold out the validation image.")
    parser.add_argument("--output_dir", type=str, default="./outputs_svd_hdr")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--wandb_image_log_steps", type=int, default=100,
                        help="Log training images to wandb every N global steps.")
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1,
                        help="Probability of dropping first-frame conditioning (for CFG training).")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--compile_unet", action="store_true",
                        help="Apply torch.compile to the UNet. Uses the inductor backend by default (ROCm-compatible).")
    parser.add_argument("--compile_backend", type=str, default="inductor",
                        help="torch.compile backend (default: inductor, works on AMD ROCm). Use 'aot_eager' as fallback.")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. Use 'default' for AMD ROCm (avoid 'reduce-overhead' which uses CUDA graphs).")

    # Raw-pair dataset (RawHDRPairDataset)
    parser.add_argument("--raw_pair_data_path", type=str, default=None,
                        help="Path to a pre-built raw_hdr_dataset (has raw/ and gt_hdr/ subdirs). "
                             "When provided, RawHDRPairDataset is used instead of RepeatedHDRVideoDataset "
                             "and the real raw capture is used as the conditioning frame.")
    parser.add_argument("--use_raw_input", action="store_true", default=False,
                        help="Use batch['raw_input'] as the conditioning frame instead of a "
                             "tone-mapped GT frame. Automatically enabled when --raw_pair_data_path is set.")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Apply LoRA adapters to UNet for memory-efficient fp32 fine-tuning.")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha scaling factor. Defaults to lora_rank (alpha/r = 1).")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0",
                        help="Comma-separated UNet attention module names to apply LoRA to.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers(project_name="SVD_HDR", config=vars(args))

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # ----------------------------------------------------------------------- #
    # Load models                                                               #
    # ----------------------------------------------------------------------- #
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16",
        torch_dtype=torch.float32,  # upcast to fp32; VAE is kept in fp32 throughout training
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16",
        torch_dtype=torch.float32,  # upcast to fp32; CLIP is kept in fp32 throughout training
    )

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )

    unet_path = args.pretrain_unet if args.pretrain_unet is not None else args.pretrained_model_name_or_path
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        # Do NOT load with variant="fp16" — trainable params must be fp32;
        # accelerate handles autocast for the forward pass.
    )

    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Freeze VAE and CLIP; fine-tune UNet only
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    # LoRA: inject adapters and freeze all base-model weights
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA training: pip install peft")
        if args.use_ema:
            raise ValueError("EMA is not supported with LoRA training. Disable --use_ema.")
        unet.requires_grad_(False)  # freeze base; PEFT will add trainable LoRA params
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        if accelerator.is_main_process:
            print(f"[LoRA] rank={args.lora_rank}  alpha={lora_alpha}  scale={lora_alpha / args.lora_rank:.2f}  targets={target_modules}")
        unet = get_peft_model(unet, lora_config)
        if accelerator.is_main_process:
            unet.print_trainable_parameters()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # VAE and image_encoder stay in fp32; only UNet forward runs under mixed-precision autocast
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"[PRECISION] mixed_precision={accelerator.mixed_precision}, unet weight_dtype={weight_dtype}")
        print(f"[PRECISION] VAE and image_encoder kept in fp32 (weights and compute)")
        print("=" * 60)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.compile_unet:
        if accelerator.is_main_process:
            print(f"[torch.compile] Compiling UNet with backend='{args.compile_backend}', mode='{args.compile_mode}'")
            print("[torch.compile] NOTE: First training step will be slow due to compilation.")
        # Disable DDP graph-splitting optimizer: it can't handle higher-order ops
        # (torch.utils.checkpoint used by gradient_checkpointing).
        torch._dynamo.config.optimize_ddp = False
        unet = torch.compile(unet, backend=args.compile_backend, mode=args.compile_mode, fullgraph=False)

    def _unwrap_unet(model):
        """Unwrap DDP and torch.compile wrappers to get the base UNet."""
        model = accelerator.unwrap_model(model)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    # Checkpoint save/load hooks
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            for model in models:
                # Strip DDP and torch.compile wrappers for saving
                _m = accelerator.unwrap_model(model)
                if hasattr(_m, "_orig_mod"):
                    _m = _m._orig_mod
                if args.use_lora:
                    lora_sd = get_peft_model_state_dict(_m)
                    save_dir = os.path.join(output_dir, "unet_lora")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(lora_sd, os.path.join(save_dir, "adapter_weights.bin"))
                else:
                    _m.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            for _ in range(len(models)):
                model = models.pop()
                # Strip DDP and torch.compile wrappers
                _m = accelerator.unwrap_model(model)
                if hasattr(_m, "_orig_mod"):
                    _m = _m._orig_mod
                if args.use_lora:
                    lora_sd = torch.load(
                        os.path.join(input_dir, "unet_lora", "adapter_weights.bin"),
                        map_location="cpu",
                    )
                    set_peft_model_state_dict(_m, lora_sd)
                else:
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                    _m.register_to_config(**load_model.config)
                    _m.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # ----------------------------------------------------------------------- #
    # Optimizer                                                                 #
    # ----------------------------------------------------------------------- #
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.per_gpu_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----------------------------------------------------------------------- #
    # Dataset and dataloader                                                    #
    # ----------------------------------------------------------------------- #
    if args.raw_pair_data_path is not None:
        # Real raw sensor captures paired with GT HDR — crops are pre-extracted at 512×512,
        # so random_crop / crop_x / crop_y are not used here.
        train_dataset = RawHDRPairDataset(
            base_folder=args.raw_pair_data_path,
            sample_frames=args.num_frames,
            use_noisy_samples=True,
            exclude_files=args.exclude_files,
        )
        args.use_raw_input = True  # raw_input is the conditioning frame when this dataset is used
    else:
        train_dataset = RepeatedHDRVideoDataset(
            base_folder=args.train_data_path,
            sample_frames=args.num_frames,
            crop_x=args.width,
            crop_y=args.height,
            random_crop=args.random_crop,
            use_noisy_samples=True,
            exclude_files=args.exclude_files,
        )
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # ----------------------------------------------------------------------- #
    # LR scheduler                                                              #
    # ----------------------------------------------------------------------- #
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ----------------------------------------------------------------------- #
    # Prepare with accelerator                                                  #
    # ----------------------------------------------------------------------- #
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Recalculate training steps after prepare
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ----------------------------------------------------------------------- #
    # Helper: _get_add_time_ids                                                 #
    # ----------------------------------------------------------------------- #
    # def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
    #     add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    #     _unet = unet.module if hasattr(unet, 'module') else unet
    #     passed = _unet.config.addition_time_embed_dim * len(add_time_ids)
    #     expected = _unet.add_embedding.linear_1.in_features
    #     if expected != passed:
    #         raise ValueError(f"add_time_ids size mismatch: expected {expected}, got {passed}")
    #     add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    #     return add_time_ids.repeat(batch_size, 1)

    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strengths, dtype):
        # noise_aug_strengths: 1-D tensor of shape [bsz] — one value per sample
        _unet = unet.module if hasattr(unet, 'module') else unet
        passed = _unet.config.addition_time_embed_dim * 3
        expected = _unet.add_embedding.linear_1.in_features
        if expected != passed:
            raise ValueError(f"add_time_ids size mismatch: expected {expected}, got {passed}")
        return torch.stack([
            torch.tensor([fps, motion_bucket_id, s.item()], dtype=dtype)
            for s in noise_aug_strengths
        ])  # (bsz, 3)

    # ----------------------------------------------------------------------- #
    # Resume from checkpoint                                                    #
    # ----------------------------------------------------------------------- #
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # ----------------------------------------------------------------------- #
    # Logging info                                                              #
    # ----------------------------------------------------------------------- #
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # ----------------------------------------------------------------------- #
    # Metrics tracking                                                          #
    # ----------------------------------------------------------------------- #
    if accelerator.is_main_process:
        metrics_path = os.path.join(args.output_dir, "val_metrics.json")
        if os.path.exists(metrics_path) and global_step > 0:
            with open(metrics_path, "r") as f:
                saved = json.load(f)
            # Only keep entries from steps <= current resume point
            PSNR_list = [v for s, v in zip(saved.get("steps", []), saved.get("PSNR", [])) if s <= global_step]
            SSIM_list = [v for s, v in zip(saved.get("steps", []), saved.get("SSIM", [])) if s <= global_step]
            LPIPS_list = [v for s, v in zip(saved.get("steps", []), saved.get("LPIPS", [])) if s <= global_step]
            metrics_steps = [s for s in saved.get("steps", []) if s <= global_step]
            logger.info(f"Loaded {len(metrics_steps)} prior validation entries from {metrics_path}")
        else:
            PSNR_list, SSIM_list, LPIPS_list = [], [], []
            metrics_steps = []

        # PSNR_list, SSIM_list, LPIPS_list, val_steps = [], [], [], []

    # ----------------------------------------------------------------------- #
    # Training loop                                                             #
    # ----------------------------------------------------------------------- #
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # pixel_values: (B, T, 3, H, W) sensor-noisy frames in [-1, 1]
                # pixel_values_clean: (B, T, 3, H, W) clean frames in [-1, 1] — used as target
                pixel_values = batch["pixel_values"].float().to(accelerator.device, non_blocking=True)
                pixel_values_clean = batch["pixel_values_clean"].float().to(accelerator.device, non_blocking=True)

                # Encode clean frames only. Diffusion noise is added on top of clean latents so that
                # noisy_latents = target_latents + noise*sigma exactly (delta_sensor = 0), keeping
                # the EDM loss well-behaved. Sensor-noisy pixel_values are still used for the CLIP
                # embedding and conditioning latent so the model sees the noisy observation as its
                # conditioning anchor.
                with torch.no_grad(), torch.autocast(accelerator.device.type, enabled=False):
                    latents = tensor_to_vae_latent(pixel_values_clean, vae)
                target_latents = latents  # same tensor — no separate encode needed
                noise = torch.randn(latents.shape, dtype=torch.float32, device=latents.device)
                bsz = latents.shape[0]

                # --- EV frame / raw conditioning frame selection ------------------
                # If use_raw_input: use the real raw sensor capture as conditioning frame.
                #   cond_frame_idx is set to mid (T//2) as a neutral EV position for
                #   added_time_ids — the actual frame content comes from raw_input.
                # Otherwise (synthetic-bracket mode): randomly pick start(0), mid(T//2), or end(T-1).
                T = pixel_values.shape[1]
                if args.use_raw_input and "raw_input" in batch:
                    raw_input = batch["raw_input"].float().to(accelerator.device, non_blocking=True)
                    # raw_input is (B, 3, H, W) — add temporal dim to match (B, 1, 3, H, W)
                    cond_pixel_values_raw = raw_input.unsqueeze(1)
                    # Use mid EV position as the nominal conditioning_frame_idx
                    cond_frame_idx = torch.full((bsz,), T // 2, dtype=torch.long, device=accelerator.device)
                else:
                    ev_choices = torch.randint(0, 3, (bsz,), device=accelerator.device)
                    ev_frame_map = torch.tensor([0, T // 2, T - 1], device=accelerator.device)
                    cond_frame_idx = ev_frame_map[ev_choices]  # (bsz,) actual frame indices
                    cond_pix_list = [pixel_values[b:b+1, cond_frame_idx[b]:cond_frame_idx[b]+1] for b in range(bsz)]
                    cond_pixel_values_raw = torch.cat(cond_pix_list, dim=0)  # (B, 1, 3, H, W)

                # Noise augmentation on conditioning frame — DISABLED (sigma=0)
                # cond_sigmas_1d = rand_log_normal(shape=[bsz], loc=-3.0, scale=0.5).to(latents.device)
                # cond_sigmas = cond_sigmas_1d[:, None, None, None, None]
                # conditional_pixel_values = torch.randn_like(cond_pixel_values_raw) * cond_sigmas + cond_pixel_values_raw
                cond_sigmas_1d = torch.zeros(bsz, device=latents.device, dtype=torch.float32)
                cond_sigmas = cond_sigmas_1d[:, None, None, None, None]
                conditional_pixel_values = cond_pixel_values_raw
                with torch.no_grad(), torch.autocast(accelerator.device.type, enabled=False):
                    conditional_latents = (
                        tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                        / vae.config.scaling_factor
                    )

                # EDM diffusion noise added on top of sensor-noisy input latents
                sigmas = rand_log_normal(shape=[bsz], loc=0.7, scale=1.6).to(latents.device)
                timesteps = (0.25 * sigmas.log()).to(accelerator.device)  # (bsz,)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

                # CLIP encode the chosen conditioning frame (matches what we condition on at inference)
                # cond_pixel_values_raw[:, 0] is (B, 3, H, W) in [-1, 1]; convert to [0, 1] for CLIP
                cond_frame_01 = (cond_pixel_values_raw[:, 0].float() + 1.0) / 2.0
                cond_frame_224 = F.interpolate(cond_frame_01, size=(224, 224), mode='bilinear', align_corners=False)
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=accelerator.device).view(1, 3, 1, 1)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=accelerator.device).view(1, 3, 1, 1)
                cond_frame_224 = (cond_frame_224 - clip_mean) / clip_std
                with torch.no_grad():
                    image_embeds = image_encoder(cond_frame_224.float()).image_embeds  # (B, D) — always fp32
                encoder_hidden_states = image_embeds.unsqueeze(1)  # (B, 1, D)

                added_time_ids = _get_add_time_ids(
                    7, 0, cond_sigmas_1d,
                    torch.float32,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout on conditioning latent and CLIP embedding (enables CFG at inference)
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # CLIP dropout: drop when random_p < 2p
                    prompt_mask = (random_p < 2 * args.conditioning_dropout_prob).reshape(bsz, 1, 1)
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                    # Image latent dropout: drop when p <= random_p < 3p
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    conditional_latents = image_mask * conditional_latents

                # Build per-frame conditioning tensor: the conditioning latent is REPEATED at all
                # T temporal positions, so the model always sees the full latent signal at every
                # frame. The EV bin of the conditioning anchor is communicated via the CLIP image
                # embedding (which encodes the chosen frame) and via `conditioning_frame_idx` at
                # inference time.
                conditional_latents_expanded = conditional_latents.unsqueeze(1).repeat(
                    1, noisy_latents.shape[1], 1, 1, 1
                )
                inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents_expanded], dim=2)

                # Target is the clean latents — model learns to denoise sensor noise too
                target = target_latents

                # UNet forward — the ONLY bf16 region; inputs auto-cast by autocast context
                with torch.autocast(accelerator.device.type, dtype=weight_dtype,
                                    enabled=weight_dtype != torch.float32):
                    model_pred = unet(
                        inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_time_ids=added_time_ids,
                    ).sample
                model_pred = model_pred.float()  # back to fp32 before any math

                # EDM loss — all fp32
                c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas ** -2.0)

                loss = torch.mean(
                    (weighing * (denoised_latents - target) ** 2).reshape(bsz, -1),
                    dim=1,
                ).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # Only gather loss across processes when we actually need to log it
                if global_step % 10 == 0:
                    avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss = avg_loss.item()
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    if accelerator.is_main_process:
                        print(f"[Step {global_step}] avg_loss={train_loss:.6f}")

                if accelerator.is_main_process:
                    # Training image logging to wandb
                    if global_step % args.wandb_image_log_steps == 0 and args.report_to == "wandb":
                        with torch.no_grad():
                            # Noisy input frames (sensor noise only)
                            noisy_in = pixel_values[0].float().cpu()  # (T, 3, H, W)
                            noisy_wb = [
                                wandb.Image(
                                    (((noisy_in[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                    caption=f"Noisy input frame {i}",
                                )
                                for i in range(noisy_in.shape[0])
                            ]
                            # Conditioning frame: decode conditional_latents back to pixels.
                            # conditional_latents = raw_z (already divided by scaling_factor for
                            # the UNet); the VAE decoder expects raw_z directly — no re-scaling.
                            with torch.no_grad(), torch.autocast(accelerator.device.type, enabled=False):
                                cond_decoded = vae.decode(
                                    conditional_latents[0:1], num_frames=1
                                ).sample  # (1, 3, H, W)
                            cond_frame = cond_decoded[0].clamp(-1, 1).float().cpu()  # (3, H, W)
                            _cond_fi = cond_frame_idx[0].item()
                            _ev_label_map = {0: "EV start", T // 2: "EV mid", T - 1: "EV high"}
                            _ev_label = _ev_label_map.get(_cond_fi, f"frame {_cond_fi}")
                            cond_wb = [wandb.Image(
                                (((cond_frame + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                caption=f"Conditioning frame {_cond_fi} ({_ev_label}, decoded from latent)",
                            )]
                            # Selected conditioning frame: raw pixel values (before noise aug)
                            # Shows which actual input frame was chosen as the anchor
                            sel_raw = cond_pixel_values_raw[0, 0].float().cpu()  # (3, H, W)
                            sel_wb = [wandb.Image(
                                (((sel_raw + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                caption=f"Selected input frame {_cond_fi} ({_ev_label})",
                            )]
                            # Clean GT frames
                            gt = pixel_values_clean[0].float().cpu()  # (T, 3, H, W)
                            gt_wb = [
                                wandb.Image(
                                    (((gt[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                    caption=f"Clean GT frame {i}",
                                )
                                for i in range(gt.shape[0])
                            ]
                            # Denoised latents decoded back to pixel space — VAE always in fp32
                            t_log = denoised_latents.shape[1]
                            dl_flat = rearrange(
                                denoised_latents[0:1].float().detach(),
                                "b f c h w -> (b f) c h w",
                            )
                            with torch.autocast(accelerator.device.type, enabled=False):
                                pred_pix = vae.decode(dl_flat / vae.config.scaling_factor, num_frames=t_log).sample
                            pred_pix = pred_pix.clamp(-1, 1).float().cpu()  # (T, 3, H, W)
                            pred_wb = [
                                wandb.Image(
                                    (((pred_pix[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                    caption=f"Pred frame {i}",
                                )
                                for i in range(pred_pix.shape[0])
                            ]
                            try:
                                wandb.log(
                                    {
                                        "train/noisy_input_frames": noisy_wb,
                                        "train/selected_cond_frame": sel_wb,
                                        "train/conditioning_frame": cond_wb,
                                        "train/clean_gt_frames": gt_wb,
                                        "train/pred_frames": pred_wb,
                                    },
                                    step=global_step,
                                )
                            except Exception as e:
                                logger.warning(f"wandb.log failed at step {global_step}: {e}")

                    # Save checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for ckpt in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    ckpt_step = int(ckpt.split("-")[1])
                                    if ckpt_step % 5000 == 0:
                                        logger.info(f"Keeping checkpoint {ckpt} (multiple of 5000)")
                                        continue
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                    # Periodic validation
                    _has_val = args.valid_hdr_path or (args.valid_raw_path and args.valid_gt_hdr_path)
                    if (global_step % args.validation_steps == 0 or global_step == 1) and _has_val:
                        logger.info(f"Running validation at step {global_step}...")

                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        _unet_unwrapped = _unwrap_unet(unet)
                        # With LoRA, unwrap_model returns a PeftModel; the pipeline
                        # needs the underlying UNetSpatioTemporalConditionModel with
                        # LoRA layers already in place (base_model.model has them).
                        _unet_for_pipeline = (
                            _unet_unwrapped.base_model.model
                            if args.use_lora else _unet_unwrapped
                        )
                        _vae_unwrapped = accelerator.unwrap_model(vae)
                        pipeline = StableVideoDiffusionPipelineHDR(
                            vae=_vae_unwrapped,
                            image_encoder=image_encoder,
                            unet=_unet_for_pipeline,
                            scheduler=scheduler,
                            feature_extractor=feature_extractor,
                        ).to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        val_save_dir = os.path.join(args.output_dir, "validation_images")
                        os.makedirs(val_save_dir, exist_ok=True)

                        pred_frames, gt_frames, val_cond_fi = validate_once(
                            val_save_dir, accelerator, pipeline, args, global_step, args.valid_hdr_path,
                            valid_raw_path=args.valid_raw_path,
                            valid_gt_hdr_path=args.valid_gt_hdr_path,
                        )

                        # Log PSNR of the conditioning frame separately
                        psnr_cond = calculate_psnr(pred_frames[val_cond_fi:val_cond_fi+1],
                                                   gt_frames[val_cond_fi:val_cond_fi+1])

                        # Metrics on all non-conditioning frames
                        non_cond_idx = [i for i in range(len(pred_frames)) if i != val_cond_fi]
                        psnr = calculate_psnr([pred_frames[i] for i in non_cond_idx],
                                              [gt_frames[i]  for i in non_cond_idx])
                        ssim = calculate_ssim([pred_frames[i] for i in non_cond_idx],
                                              [gt_frames[i]  for i in non_cond_idx])
                        lp   = calculate_lpips([pred_frames[i] for i in non_cond_idx],
                                               [gt_frames[i]   for i in non_cond_idx])

                        PSNR_list.append(psnr)
                        SSIM_list.append(ssim)
                        LPIPS_list.append(lp)
                        metrics_steps.append(global_step)
                        # val_steps.append(global_step)

                        log_dict = {
                            "val/psnr": psnr, "val/ssim": ssim, "val/lpips": lp,
                            "val/psnr_cond_frame": psnr_cond,
                        }
                        accelerator.log(log_dict, step=global_step)

                        # Persist metrics to disk
                        with open(metrics_path, "w") as f:
                            json.dump({"steps": metrics_steps, "PSNR": PSNR_list, "SSIM": SSIM_list, "LPIPS": LPIPS_list}, f)

                        x_axis = np.array(metrics_steps)
                        for metric_name, metric_list in [("PSNR", PSNR_list), ("SSIM", SSIM_list), ("LPIPS", LPIPS_list)]:
                            plt.figure()
                            plt.plot(x_axis, metric_list)
                            plt.xlabel("Step")
                            plt.title(metric_name)
                            plt.savefig(os.path.join(args.output_dir, f"{metric_name}_curve.png"))
                            plt.close()

                        del pipeline

                        if args.use_ema:
                            ema_unet.restore(unet.parameters())

                # Keep all ranks in sync after main-process-only operations
                # (logging, checkpointing, validation). Without this, ranks
                # 1-7 race ahead into the next iteration's collectives while
                # rank 0 is still busy, causing NCCL timeout.
                accelerator.wait_for_everyone()

    # ----------------------------------------------------------------------- #
    # Save final model                                                          #
    # ----------------------------------------------------------------------- #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_final = _unwrap_unet(unet)
        if args.use_ema:
            ema_unet.copy_to(unet_final.parameters())
        if args.use_lora:
            lora_final_dir = os.path.join(args.output_dir, "unet_lora_final")
            os.makedirs(lora_final_dir, exist_ok=True)
            lora_sd = get_peft_model_state_dict(unet_final)
            torch.save(lora_sd, os.path.join(lora_final_dir, "adapter_weights.bin"))
            unet_final.save_pretrained(lora_final_dir)  # saves adapter_config.json too
            logger.info(f"Saved final LoRA adapter weights to {lora_final_dir}")
        else:
            unet_final.save_pretrained(os.path.join(args.output_dir, "unet_final"))

    accelerator.end_training()


if __name__ == "__main__":
    main()