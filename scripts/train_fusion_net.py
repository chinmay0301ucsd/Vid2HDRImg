"""Stage 2 of 2 — train the lightweight pixel-space fusion U-Net.

Stage 1 (the SVD UNet that synthesises an LDR exposure bracket from a single
input image) is trained separately by ``train_video_model.py``. This script
trains only the fusion network: a small per-pixel U-Net that takes a stack
of LDR frames at increasing exposures and outputs blending weights, which
are combined into the final HDR image.

Per training step:
  1. GT HDR → render at T exposures → LDR frames in [0, 1]  (done in dataset)
  2. Linearise: ldr^2.2 ≈ hdr * 2^ev  (undo gamma)
  3. fusion_net(linear_frames) → fused ≈ relative linear HDR in [0, 1]
     (trainable, full-resolution, pixel-space — no VAE, no UNet)
  4. Compute perceptual HDR loss: MSE(pu21_encode(fused), pu21_encode(hdr_gt / luma_max))

Only fusion-net parameters are updated.
"""

import argparse
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from src.dataset_hdr import RepeatedHDRVideoDataset, read_hdr_image_float32, hdr_to_ldr_batch_np
import pyexr

if is_wandb_available():
    import wandb

check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import pyiqa
from skimage.metrics import structural_similarity as sk_ssim

mse_loss_fn = nn.MSELoss()
ssim_metric_fn = pyiqa.create_metric('ssim', device='cuda', as_loss=False)
lpips_metric = pyiqa.create_metric('lpips', device='cuda', as_loss=False)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def calculate_psnr(vid1, vid2):
    to_tensor = transforms.ToTensor()
    psnr_list = []
    for img1, img2 in zip(vid1, vid2):
        t1 = to_tensor(img1) if not isinstance(img1, torch.Tensor) else img1
        t2 = to_tensor(img2) if not isinstance(img2, torch.Tensor) else img2
        mse = mse_loss_fn(t1, t2)
        psnr_list.append(100.0 if mse == 0 else 10 * torch.log10(1 / mse).item())
    return float(np.mean(psnr_list))


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


# --------------------------------------------------------------------------- #
# PU-21 perceptually-uniform encoding  (Perez-Ortiz & Mantiuk, 2021)          #
# --------------------------------------------------------------------------- #

_PU21_A: float = 0.001908
_PU21_B: float = 0.0078
_PU21_L_MIN: float = 0.005
_PU21_L_MAX: float = 10000.0
_PU21_LOG2_L_MIN: float = math.log2(_PU21_L_MIN)


def pu21_encode_tensor(hdr_01: torch.Tensor, peak_lum: float = 4000.0) -> torch.Tensor:
    L = hdr_01.float().clamp(min=0.0) * peak_lum
    L = L.clamp(min=_PU21_L_MIN, max=_PU21_L_MAX)
    x = torch.log2(L) - _PU21_LOG2_L_MIN
    return (_PU21_A * x * x + _PU21_B * x).clamp(0.0, 1.0)


def pu21_decode_tensor(encoded: torch.Tensor, peak_lum: float = 4000.0) -> torch.Tensor:
    y = encoded.float().clamp(min=0.0)
    disc = (_PU21_B ** 2 + 4.0 * _PU21_A * y).clamp(min=0.0)
    x = (-_PU21_B + torch.sqrt(disc)) / (2.0 * _PU21_A)
    L = torch.pow(torch.tensor(2.0, device=encoded.device), x + _PU21_LOG2_L_MIN)
    return (L / peak_lum).clamp(min=0.0)


def mu_law_encode(x: torch.Tensor, mu: float = 5000.0) -> torch.Tensor:
    return torch.log1p(mu * x.float().clamp(min=0.0)) / math.log(1.0 + mu)


def _pu21_encode_np(L: np.ndarray) -> np.ndarray:
    Lc = np.clip(L, _PU21_L_MIN, _PU21_L_MAX)
    x = np.log2(Lc) - _PU21_LOG2_L_MIN
    return (_PU21_A * x * x + _PU21_B * x).astype(np.float32)


def _pu_encode_np(img_norm: np.ndarray, peak_lum: float = 4000.0) -> np.ndarray:
    L_abs = np.maximum(img_norm, 0.0).astype(np.float64) * peak_lum
    L_abs = np.maximum(L_abs, _PU21_L_MIN)
    return _pu21_encode_np(L_abs)


def _normalize_hdr_pair(pred_hdr: np.ndarray, ref_hdr: np.ndarray):
    ref_max = float(np.percentile(ref_hdr, 99))
    if ref_max < 1e-8:
        ref_max = float(np.max(ref_hdr)) + 1e-8
    pred_norm = np.clip(pred_hdr / ref_max, 0.0, None).astype(np.float32)
    ref_norm  = np.clip(ref_hdr  / ref_max, 0.0, None).astype(np.float32)
    return pred_norm, ref_norm


def calculate_pu_psnr(pred_hdr: np.ndarray, ref_hdr: np.ndarray, peak_lum: float = 4000.0) -> float:
    pred_norm, ref_norm = _normalize_hdr_pair(pred_hdr, ref_hdr)
    pred_pu = _pu_encode_np(pred_norm, peak_lum)
    ref_pu  = _pu_encode_np(ref_norm,  peak_lum)
    mse = float(np.mean((pred_pu.astype(np.float64) - ref_pu.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def calculate_pu_ssim(pred_hdr: np.ndarray, ref_hdr: np.ndarray, peak_lum: float = 4000.0) -> float:
    pred_norm, ref_norm = _normalize_hdr_pair(pred_hdr, ref_hdr)
    pred_pu = _pu_encode_np(pred_norm, peak_lum)
    ref_pu  = _pu_encode_np(ref_norm,  peak_lum)
    C = pred_pu.shape[2] if pred_pu.ndim == 3 else 1
    scores = []
    for c in range(C):
        p = pred_pu[..., c] if pred_pu.ndim == 3 else pred_pu
        r = ref_pu[..., c] if ref_pu.ndim == 3 else ref_pu
        scores.append(sk_ssim(p, r, data_range=1.0))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Fusion UNet                                                                   #
# --------------------------------------------------------------------------- #

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionUNet(nn.Module):
    """
    Small UNet that takes T concatenated LDR pixel frames (B, T, 3, H, W) in [0,1] and
    predicts one scalar weight per frame per pixel (shared across channels), then fuses.

    Input:  pixel_values  (B, T, 3, H, W) in [0, 1]
    Output: fused         (B, 3, H, W) in [0, 1],  weights (B, T, 1, H, W) summing to 1 over T
    """

    def __init__(self, in_channels: int = 3, num_frames: int = 5, base_ch: int = 32):
        super().__init__()
        self.num_frames  = num_frames
        self.in_channels = in_channels
        inp = in_channels * num_frames  # 15 for C=3, T=5
        b   = base_ch

        # Encoder
        self.enc1 = _DoubleConv(inp,   b)      # (B, b,   H,   W)
        self.enc2 = _DoubleConv(b,     b * 2)  # (B, 2b,  H/2, W/2)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot  = _DoubleConv(b * 2, b * 4)  # (B, 4b,  H/4, W/4)

        # Decoder
        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(b * 4, b * 2)  # after cat with enc2 skip
        self.up1  = nn.ConvTranspose2d(b * 2, b,     kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(b * 2, b)       # after cat with enc1 skip

        # Head: one scalar logit per frame per pixel (shared across channels)
        self.head = nn.Conv2d(b, num_frames, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor):
        B, T, C, H, W = pixel_values.shape
        x = pixel_values.reshape(B, T * C, H, W)

        # Encode
        e1 = self.enc1(x)             # (B, b,   H,   W)
        e2 = self.enc2(self.pool(e1)) # (B, 2b,  H/2, W/2)
        bt = self.bot(self.pool(e2))  # (B, 4b,  H/4, W/4)

        # Decode with skip connections
        d2 = self.dec2(torch.cat([self.up2(bt), e2], dim=1))  # (B, 2b, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, b,  H,   W)

        logits  = self.head(d1)                              # (B, T, H, W)
        weights = F.softmax(logits, dim=1)                   # softmax over T → (B, T, H, W)
        weights = weights.unsqueeze(2)                       # (B, T, 1, H, W) — broadcast over C
        fused   = (pixel_values * weights).sum(dim=1)        # (B, 3, H, W) in [0,1]
        return fused, weights


# --------------------------------------------------------------------------- #
# Validation                                                                    #
# --------------------------------------------------------------------------- #

def validate_once(val_save_dir, accelerator, fusion_net, args, global_step, valid_hdr_path):
    img = read_hdr_image_float32(valid_hdr_path)

    h, w = img.shape[:2]
    scale = max(args.width / w, args.height / h)
    img_resized = cv2.resize(img, (int(w * scale + 0.5), int(h * scale + 0.5)), interpolation=cv2.INTER_AREA)
    h2, w2 = img_resized.shape[:2]
    sy = (h2 - args.height) // 2
    sx = (w2 - args.width) // 2
    img_crop = img_resized[sy: sy + args.height, sx: sx + args.width]

    gt_hdr_path = os.path.join(val_save_dir, "gt_hdr.exr")
    if not os.path.exists(gt_hdr_path):
        pyexr.write(gt_hdr_path, img_crop.astype(np.float32))

    gamma = 2.2
    Y = (img_crop[..., 0] * 0.2126 + img_crop[..., 1] * 0.7152 + img_crop[..., 2] * 0.0722).clip(min=1e-6)
    Ymax = Y.max()
    Ymedian = np.median(Y)
    start_ev = np.log2((0.85 ** gamma) / Ymax)
    end_ev   = np.log2((0.85 ** gamma) / Ymedian)
    ev_values = np.linspace(start_ev, end_ev, args.num_frames)
    ldr_batch = hdr_to_ldr_batch_np(img_crop, ev_values, gamma=gamma)  # (T, H, W, 3) in [0,1]
    gt_frames = [(ldr_batch[i] * 255).clip(0, 255).astype(np.uint8) for i in range(args.num_frames)]

    gt_ldr_path = os.path.join(val_save_dir, "gt_ldr.gif")
    if not os.path.exists(gt_ldr_path):
        export_to_gif(gt_frames, gt_ldr_path, 8)

    hdr_np_rescaled = None

    with torch.no_grad():
        # (T, H, W, 3) → (1, T, 3, H, W) in [0, 1], linearize (undo gamma)
        pixel_values = torch.from_numpy(ldr_batch).permute(0, 3, 1, 2).unsqueeze(0).to(
            dtype=torch.float32, device=accelerator.device
        )  # (1, T, 3, H, W) gamma-encoded
        pixel_values_lin = pixel_values.clamp(0, 1) ** 2.2  # (1, T, 3, H, W) linear

        # Pixel-space fusion → relative linear HDR in [0, 1]
        hdr_pred_01, _ = fusion_net(pixel_values_lin)  # (1, 3, H, W)
        hdr_np = hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)

        # Rescale back to scene units (luma_max matches training normalization)
        Y_crop = (img_crop[..., 0] * 0.2126 + img_crop[..., 1] * 0.7152 + img_crop[..., 2] * 0.0722)
        luma_max = float(Y_crop.max().clip(min=1e-6))
        hdr_np_rescaled = hdr_np * luma_max

        exr_path = os.path.join(val_save_dir, f"step_{global_step}_hdr.exr")
        pyexr.write(exr_path, hdr_np_rescaled.astype(np.float32))
        logger.info(f"Saved HDR EXR: {exr_path}")

    return gt_frames, hdr_np_rescaled, img_crop


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LatentExposureFusionNet with a frozen VAE (no UNet)."
    )

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                        help="(Unused) Path to pretrained SVD model. Kept for script compatibility.")
    parser.add_argument("--fusion_checkpoint", type=str, default=None,
                        help="Path to a saved fusion_net.pt to resume from.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Primary HDR training data folder.")
    parser.add_argument("--train_data_path2", type=str, default=None,
                        help="Optional second HDR training data folder (combined with primary).")
    parser.add_argument("--valid_hdr_path", type=str, default=None,
                        help="Single HDR file for validation (legacy). Ignored if --val_hdr_dir is set.")
    parser.add_argument("--val_hdr_dir", type=str, default=None,
                        help="Directory of HDR files from which to sample a validation set.")
    parser.add_argument("--val_max_files", type=int, default=100,
                        help="Number of HDR files to hold out from --val_hdr_dir as validation.")
    parser.add_argument("--output_dir", type=str, default="./output_fus_only")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--random_crop", action="store_true", default=False)

    # HDR loss
    parser.add_argument("--loss_type", type=str, default="pu21", choices=["pu21", "mu"],
                        help="HDR loss type: 'pu21' (PU-21 MSE) or 'mu' (mu-law MSE).")
    parser.add_argument("--mu", type=float, default=5000.0,
                        help="Mu parameter for mu-law encoding loss.")
    parser.add_argument("--pu21_peak_lum", type=float, default=4000.0,
                        help="Peak luminance (cd/m²) for PU-21 encoding.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
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
        level=logging.INFO,
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
        accelerator.init_trackers(project_name="SVD_HDR_FusOnly", config=vars(args))

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"[PRECISION] mixed_precision={accelerator.mixed_precision}, weight_dtype={weight_dtype}")
        print(f"[SETUP] Pixel-space fusion. No VAE. Training fusion_net only.")
        print("=" * 60)

    # Fusion network — only trainable module
    fusion_net = FusionUNet(in_channels=3, num_frames=args.num_frames, base_ch=32)
    fusion_net.to(accelerator.device)
    fusion_net.requires_grad_(True)

    if args.fusion_checkpoint is not None:
        sd = torch.load(args.fusion_checkpoint, map_location="cpu")
        fusion_net.load_state_dict(sd)
        logger.info(f"Loaded fusion_net weights from {args.fusion_checkpoint}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Checkpoint save/load hooks — only fusion_net is saved
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, FusionUNet):
                    torch.save(unwrapped.state_dict(), os.path.join(output_dir, "fusion_net.pt"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, FusionUNet):
                    fusion_path = os.path.join(input_dir, "fusion_net.pt")
                    if os.path.exists(fusion_path):
                        model.load_state_dict(torch.load(fusion_path, map_location="cpu"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # ----------------------------------------------------------------------- #
    # Optimizer — only fusion_net parameters                                    #
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
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        fusion_net.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----------------------------------------------------------------------- #
    # Validation file list                                                      #
    # ----------------------------------------------------------------------- #
    _hdr_exts = (".exr", ".hdr")
    if args.val_hdr_dir:
        _all_val = sorted(
            os.path.join(args.val_hdr_dir, f)
            for f in os.listdir(args.val_hdr_dir)
            if f.lower().endswith(_hdr_exts)
        )
        _rng = np.random.default_rng(args.seed if args.seed else 0)
        _rng.shuffle(_all_val)
        val_hdr_paths = _all_val[:args.val_max_files]
        logger.info(f"Val set: {len(val_hdr_paths)} files from {args.val_hdr_dir}")
    elif args.valid_hdr_path:
        val_hdr_paths = [args.valid_hdr_path]
    else:
        val_hdr_paths = []

    # ----------------------------------------------------------------------- #
    # Dataset and dataloader                                                    #
    # ----------------------------------------------------------------------- #
    _train_folders = [args.train_data_path]
    if args.train_data_path2:
        _train_folders.append(args.train_data_path2)
    train_dataset = RepeatedHDRVideoDataset(
        base_folder=_train_folders,
        sample_frames=args.num_frames,
        crop_x=args.width,
        crop_y=args.height,
        random_crop=args.random_crop,
        exclude_files=val_hdr_paths,
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
    fusion_net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        fusion_net, optimizer, lr_scheduler, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
    logger.info(f"  Loss type = {args.loss_type}")

    # ----------------------------------------------------------------------- #
    # Metrics tracking                                                          #
    # ----------------------------------------------------------------------- #
    if accelerator.is_main_process:
        loss_list, val_steps = [], []
        PU_PSNR_list, PU_SSIM_list = [], []

    # ----------------------------------------------------------------------- #
    # Training loop                                                             #
    # ----------------------------------------------------------------------- #
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        fusion_net.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(fusion_net):
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True).float()
                hdr_image = batch["hdr_image"].to(accelerator.device, non_blocking=True).float()

                bsz = pixel_values.shape[0]

                # pixel_values in [-1, 1] from dataset → rescale to [0, 1] → linearize (undo gamma)
                pixel_values_01 = (pixel_values + 1.0) / 2.0        # (B, T, 3, H, W) in [0, 1], gamma-encoded
                pixel_values_lin = pixel_values_01.clamp(0, 1) ** 2.2  # (B, T, 3, H, W) in [0, 1], linear

                # ── Pixel-space fusion: trainable ─────────────────────────────
                # fused_lin is a weighted average of linearized LDR frames ≈ relative linear HDR in [0,1]
                hdr_pred, _ = fusion_net(pixel_values_lin)  # (B, 3, H, W) in [0, 1], linear HDR estimate

                # ── Normalise GT HDR per-image by max luma (single scalar → preserves color) ──
                Y = (hdr_image[:, 0] * 0.2126 + hdr_image[:, 1] * 0.7152 + hdr_image[:, 2] * 0.0722)
                ref_max = Y.reshape(bsz, -1).amax(dim=-1).clamp(min=1e-6).view(bsz, 1, 1, 1)
                hdr_gt_01 = (hdr_image / ref_max).clamp(0.0, None)  # (B, 3, H, W) relative linear HDR

                # Both hdr_pred and hdr_gt_01 are relative linear HDR in [0,1] — encode for perceptual loss
                if args.loss_type == "pu21":
                    pred_encoded = pu21_encode_tensor(hdr_pred.clamp(0, 1), peak_lum=args.pu21_peak_lum)
                    gt_encoded   = pu21_encode_tensor(hdr_gt_01.clamp(0, 1).to(hdr_pred.device), peak_lum=args.pu21_peak_lum)
                else:  # mu
                    pred_encoded = mu_law_encode(hdr_pred, mu=args.mu)
                    gt_encoded   = mu_law_encode(hdr_gt_01.clamp(0, None).to(hdr_pred.device), mu=args.mu)

                loss = F.mse_loss(pred_encoded.float(), gt_encoded.float())

                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(fusion_net.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"[Step {global_step}] loss={train_loss:.6f}")
                train_loss = 0.0

                if accelerator.is_main_process:
                    # Wandb logging: fused HDR prediction visualised as tonemapped LDR
                    if global_step % 100 == 0 and args.report_to == "wandb":
                        with torch.no_grad():
                            gt_wb = [
                                wandb.Image(
                                    (pixel_values_01[0, i].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                    caption=f"GT LDR frame {i} (gamma)",
                                )
                                for i in range(pixel_values_01.shape[1])
                            ]
                            # Fused linear HDR tonemapped for visualisation (simple gamma)
                            fused_tm = (hdr_pred[0].float().cpu().clamp(0, 1) ** (1/2.2)).permute(1, 2, 0).numpy()
                            fused_wb = wandb.Image((fused_tm * 255).astype(np.uint8), caption="Fused linear HDR (gamma tonemapped)")
                            wandb.log({"train/gt_ldr_frames": gt_wb, "train/fused_hdr": fused_wb}, step=global_step)

                    # Save checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                logger.info(f"Removing checkpoints: {removing_checkpoints}")
                                for rc in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, rc))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                    # Validation
                    if (global_step % args.validation_steps == 0 or global_step == 1) and val_hdr_paths:
                        logger.info(f"Running validation at step {global_step} ({len(val_hdr_paths)} files)...")

                        _fus_unwrapped = accelerator.unwrap_model(fusion_net)

                        val_save_dir = os.path.join(args.output_dir, "validation_images")
                        os.makedirs(val_save_dir, exist_ok=True)

                        _fus_unwrapped.eval()
                        _step_psnrs, _step_ssims = [], []
                        for _vi, _vpath in enumerate(val_hdr_paths):
                            _vstem = os.path.splitext(os.path.basename(_vpath))[0]
                            _vfile_dir = os.path.join(val_save_dir, _vstem)
                            os.makedirs(_vfile_dir, exist_ok=True)
                            try:
                                gt_frames, hdr_np, img_crop = validate_once(
                                    _vfile_dir, accelerator, _fus_unwrapped,
                                    args, global_step, _vpath,
                                )
                                if hdr_np is not None and img_crop is not None:
                                    _p = calculate_pu_psnr(hdr_np, img_crop, peak_lum=args.pu21_peak_lum)
                                    _s = calculate_pu_ssim(hdr_np, img_crop, peak_lum=args.pu21_peak_lum)
                                    _step_psnrs.append(_p)
                                    _step_ssims.append(_s)
                                    logger.info(f"  [{_vi+1}/{len(val_hdr_paths)}] {_vstem}  PU-PSNR={_p:.4f}  PU-SSIM={_s:.4f}")
                            except Exception as _ve:
                                import traceback
                                logger.warning(f"  [{_vi+1}/{len(val_hdr_paths)}] {_vstem} FAILED: {_ve}\n{traceback.format_exc()}")
                        _fus_unwrapped.train()

                        if _step_psnrs:
                            pu_psnr = float(np.mean(_step_psnrs))
                            pu_ssim = float(np.mean(_step_ssims))
                            PU_PSNR_list.append(pu_psnr)
                            PU_SSIM_list.append(pu_ssim)
                            val_steps.append(global_step)
                            accelerator.log({"val/pu_psnr": pu_psnr, "val/pu_ssim": pu_ssim}, step=global_step)
                            logger.info(f"  mean PU-PSNR={pu_psnr:.4f}  mean PU-SSIM={pu_ssim:.4f}  (n={len(_step_psnrs)})")

                            for metric_name, metric_list in [("PU_PSNR", PU_PSNR_list), ("PU_SSIM", PU_SSIM_list)]:
                                plt.figure()
                                plt.plot(val_steps, metric_list)
                                plt.title(metric_name)
                                plt.savefig(os.path.join(args.output_dir, f"{metric_name}_curve.png"))
                                plt.close()

    # ----------------------------------------------------------------------- #
    # Save final fusion_net                                                     #
    # ----------------------------------------------------------------------- #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        fus_final = accelerator.unwrap_model(fusion_net)
        torch.save(fus_final.state_dict(), os.path.join(args.output_dir, "fusion_net_final.pt"))
        logger.info(f"Saved final fusion_net to {args.output_dir}/fusion_net_final.pt")

    accelerator.end_training()


if __name__ == "__main__":
    main()
