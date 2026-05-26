"""Evaluate Wan 2.1 I2V LoRA + 4-frame fusion net on the raw2hdr SI-HDR test set.

Loads the Wan I2V transformer + LoRA + fusion net ONCE, then iterates over the
test set. For each scene the script:

  1. Loads the GT HDR (.exr, 512×512) from <testset>/gt_hdr/.
  2. Loads the linear EXR input from <testset>/input_raw/, p99-normalises → [0, 1].
  3. CLIP-encodes the LINEAR uint8 view of the reference (matches training
     val_ref_uint8 in train_wan_i2v_lora.py).
  4. Builds the 20-ch I2V conditioning via build_i2v_condition (ref_mode read
     from the LoRA checkpoint's saved config; default wan_i2v_anchored).
  5. Runs the Wan transformer denoising loop (UniPC scheduler from base model).
  6. Decodes the VAE → T_pix frames in [0, 1] (gamma); drops frame 0 (anchor)
     in anchored mode → 4-frame bracket.
  7. Fuses via the 4-frame FusionUNet (input linearised with ** 2.2).
  8. Rescales the predicted HDR to max=32 before saving as .exr.
  9. Computes µ-law-PSNR and PU21-PSNR against the GT HDR.

Output layout:
  <output_dir>/ldr_frames/<stem>_frame_NN.png   — per-frame bracket
  <output_dir>/ldr_frames/<stem>_input_ldr.png  — gamma-tonemapped reference
  <output_dir>/hdr_predicted/<stem>_pred.exr    — predicted HDR (max=32 rescaled)
  <output_dir>/fusion_weights/<stem>_weight_*.png — per-frame fusion weights
  <output_dir>/metrics_summary.txt / .json
  <output_dir>/metrics_per_image.csv

PU21 / µ-law definitions match those in train_fusion_net.py (peak_lum=4000 by
default), so eval-time numbers line up with training-time validation logs.

Example:
    python scripts/eval_wan_sihdr.py \\
        --raw2hdr_testset_dir /work/nvme/bhcc/ctalegaonkar/datasets/raw2hdr_sihdr_testset \\
        --output_dir          /work/nvme/.../sihdr_eval \\
        --lora_checkpoint     /work/nvme/.../lora-4000/lora_weights.pt \\
        --fusion_net_path     /work/nvme/.../fusion_net_4frames/checkpoint-16000/fusion_net.pt \\
        --pretrained_wan_path /work/nvme/.../models/Wan2.1-I2V-14B-480P-Diffusers
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_wan import (
    _DTYPES,
    attach_lora,
    load_fusion_net,
    load_wan_components,
    preprocess_reference,
    run_wan_i2v,
)
from train_wan_i2v_lora import encode_text_empty
from src.models.fusion_unet import save_fusion_weights

import HDRutils.io as hdr_io


logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)


# ─── PU21 / µ-law (mirrors train_fusion_net.py) ───────────────────────────────
_PU21_A         = 0.001908
_PU21_B         = 0.0078
_PU21_L_MIN     = 0.005
_PU21_L_MAX     = 10000.0
_PU21_LOG2_LMIN = math.log2(_PU21_L_MIN)


def _pu21_encode_np(L_abs: np.ndarray) -> np.ndarray:
    Lc = np.clip(L_abs, _PU21_L_MIN, _PU21_L_MAX)
    x  = np.log2(Lc) - _PU21_LOG2_LMIN
    return (_PU21_A * x * x + _PU21_B * x).astype(np.float32)


def _pu_encode_norm01(img_norm: np.ndarray, peak_lum: float) -> np.ndarray:
    L_abs = np.maximum(img_norm, 0.0).astype(np.float64) * peak_lum
    return _pu21_encode_np(np.maximum(L_abs, _PU21_L_MIN))


def _normalize_hdr_pair(pred_hdr: np.ndarray, ref_hdr: np.ndarray):
    """Normalise both pred and ref by ref's 99th percentile so absolute scale drops out."""
    ref_max = float(np.percentile(ref_hdr, 99))
    if ref_max < 1e-8:
        ref_max = float(np.max(ref_hdr)) + 1e-8
    return (np.clip(pred_hdr / ref_max, 0.0, None).astype(np.float32),
            np.clip(ref_hdr  / ref_max, 0.0, None).astype(np.float32))


def calculate_pu_psnr(pred_hdr: np.ndarray, ref_hdr: np.ndarray,
                      peak_lum: float = 4000.0) -> float:
    pred_norm, ref_norm = _normalize_hdr_pair(pred_hdr, ref_hdr)
    pred_pu = _pu_encode_norm01(pred_norm, peak_lum)
    ref_pu  = _pu_encode_norm01(ref_norm,  peak_lum)
    mse = float(np.mean((pred_pu.astype(np.float64) - ref_pu.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _mu_law_encode_np(x: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    return np.log1p(mu * np.maximum(x, 0.0).astype(np.float64)) / np.log1p(mu)


def calculate_mu_psnr(pred_hdr: np.ndarray, ref_hdr: np.ndarray,
                      mu: float = 5000.0) -> float:
    pred_norm, ref_norm = _normalize_hdr_pair(pred_hdr, ref_hdr)
    pred_mu = _mu_law_encode_np(pred_norm, mu)
    ref_mu  = _mu_law_encode_np(ref_norm,  mu)
    mse = float(np.mean((pred_mu - ref_mu) ** 2))
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


# ─── HDR I/O helpers ──────────────────────────────────────────────────────────
def _imread_hdr_any(path: str) -> np.ndarray:
    """Read a linear HDR image from .exr or .hdr/.pic (Radiance)."""
    if path.lower().endswith(".exr"):
        arr = hdr_io.imread(path).astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if arr is None:
            raise IOError(f"Could not read HDR file: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.maximum(arr[..., :3], 0.0).astype(np.float32)


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--raw2hdr_testset_dir", type=str, required=True,
                   help="Root of the testset (expects gt_hdr/ and input_raw/ sub-folders).")
    p.add_argument("--raw_input_dir", type=str, default=None,
                   help="Override the raw EXR input directory "
                        "(default: {raw2hdr_testset_dir}/input_raw).")
    p.add_argument("--output_dir", type=str, required=True)

    # Models
    p.add_argument("--pretrained_wan_path", type=str, required=True,
                   help="Wan 2.1 I2V Diffusers snapshot directory.")
    p.add_argument("--lora_checkpoint", type=str, required=True,
                   help="LoRA checkpoint .pt saved by train_wan_i2v_lora.py.")
    p.add_argument("--fusion_net_path", type=str, required=True,
                   help="Fusion-net checkpoint .pt (matching --num_frames).")

    # Generation (defaults pinned to Wan I2V LoRA training)
    p.add_argument("--num_frames", type=int, default=4,
                   help="LDR bracket frames (must match the fusion net's training; default 4).")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--ref_mode", default=None,
                   choices=["wan_i2v_anchored", "soft_wan_ref", "strict_i2v", "editable"],
                   help="Override the LoRA's training ref_mode (default: read from checkpoint).")

    # Output scaling
    p.add_argument("--save_hdr_max", type=float, default=32.0,
                   help="Rescale predicted HDR so its max == this value before .exr write. "
                        "Set <=0 to save the fusion-net output unscaled (max ≈ 1).")
    p.add_argument("--pu21_peak_lum", type=float, default=4000.0,
                   help="Assumed display peak luminance (cd/m²) used for PU21 metric encoding. "
                        "Does NOT affect the saved EXR — purely a metric hyperparameter.")

    # Eval range
    p.add_argument("--eval_start", type=int, default=0)
    p.add_argument("--eval_end",   type=int, default=-1,
                   help="-1 = all images.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip scenes whose <stem>_pred.exr already exists.")

    # Misc
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--dtype",  default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device / dtype ────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA / ROCm not available; falling back to CPU (will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype  = _DTYPES[args.dtype]
    torch.manual_seed(args.seed)

    # ── Resolve test set ──────────────────────────────────────────────────────
    gt_hdr_folder    = os.path.join(args.raw2hdr_testset_dir, "gt_hdr")
    raw_input_folder = args.raw_input_dir or os.path.join(args.raw2hdr_testset_dir, "input_raw")
    if not os.path.isdir(gt_hdr_folder):
        raise FileNotFoundError(f"Missing {gt_hdr_folder}")
    if not os.path.isdir(raw_input_folder):
        raise FileNotFoundError(f"Missing {raw_input_folder}")

    all_gt = sorted(
        [os.path.join(gt_hdr_folder, f) for f in os.listdir(gt_hdr_folder)
         if f.lower().endswith((".exr", ".hdr"))]
    )
    if not all_gt:
        raise FileNotFoundError(f"No .exr/.hdr in {gt_hdr_folder}")
    end = len(all_gt) if args.eval_end == -1 else args.eval_end
    ref_paths = all_gt[args.eval_start: end]
    log.info(f"raw2hdr_sihdr testset: {len(ref_paths)} scenes "
             f"({args.eval_start}:{end}) from {args.raw2hdr_testset_dir}")

    # ── Load Wan I2V + LoRA + fusion net (ONCE) ──────────────────────────────
    (tokenizer, text_encoder, vae, scheduler, transformer,
     image_encoder, feat_proc) = load_wan_components(args.pretrained_wan_path, dtype)
    train_cfg = attach_lora(transformer, args.lora_checkpoint)
    ref_mode  = args.ref_mode or train_cfg.get("ref_mode", "wan_i2v_anchored")
    log.info(f"Using ref_mode='{ref_mode}' (from {'CLI' if args.ref_mode else 'LoRA config'}).")
    if args.num_frames != train_cfg.get("num_frames", args.num_frames):
        log.warning(f"--num_frames={args.num_frames} differs from LoRA training "
                    f"num_frames={train_cfg.get('num_frames')}; mismatches usually degrade quality.")

    transformer.to(device); vae.to(device)
    text_encoder.to(device); image_encoder.to(device)
    latents_mean    = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device)

    empty_text_embed = encode_text_empty(tokenizer, text_encoder, device, dtype, 512)
    del text_encoder
    torch.cuda.empty_cache()

    fusion_net = load_fusion_net(args.fusion_net_path, args.num_frames, device)

    # ── Output dirs ───────────────────────────────────────────────────────────
    ldr_dir     = os.path.join(args.output_dir, "ldr_frames");      os.makedirs(ldr_dir, exist_ok=True)
    hdr_dir     = os.path.join(args.output_dir, "hdr_predicted");   os.makedirs(hdr_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "fusion_weights");  os.makedirs(weights_dir, exist_ok=True)

    # ── Eval loop ─────────────────────────────────────────────────────────────
    metric_names    = ["mu_psnr", "pu_psnr"]
    all_metrics     = {k: [] for k in metric_names}
    per_sample_rows = []
    failures        = []
    t0_global       = time.time()

    for idx, ref_path in enumerate(tqdm(ref_paths, desc="Evaluating")):
        stem = Path(ref_path).stem
        out_exr = os.path.join(hdr_dir, f"{stem}_pred.exr")

        if args.skip_existing and os.path.exists(out_exr):
            log.info(f"[{idx+1}/{len(ref_paths)}] skip (exists): {out_exr}")
            continue

        # Resolve the raw EXR matching this GT stem (try .exr then .hdr).
        raw_path = None
        for ext in (".exr", ".hdr"):
            cand = os.path.join(raw_input_folder, stem + ext)
            if os.path.isfile(cand):
                raw_path = cand
                break
        if raw_path is None:
            log.warning(f"[{idx+1}] No raw input for stem '{stem}' in {raw_input_folder}; skipping.")
            failures.append(stem)
            continue

        t0 = time.time()
        try:
            # Load GT and raw input.
            ref_hdr = _imread_hdr_any(ref_path)
            raw_lin = _imread_hdr_any(raw_path)

            # Preprocess reference for Wan (p99-norm linear; uint8 for CLIP, [-1,1] for VAE).
            ref_uint8, ref_minus1to1 = preprocess_reference(
                raw_lin, args.width, args.height, device, dtype)

            # Generate LDR bracket via Wan I2V.
            frames_01 = run_wan_i2v(
                transformer, vae, scheduler, image_encoder, feat_proc,
                empty_text_embed, ref_uint8, ref_minus1to1,
                latents_mean, latents_std_inv,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                device=device, dtype=dtype,
                seed=args.seed, ref_mode=ref_mode,
            )                                                                        # (T, H, W, 3) [0,1]

            # Save reference + per-frame LDR PNGs.
            Image.fromarray(ref_uint8).save(os.path.join(ldr_dir, f"{stem}_input_ldr.png"))
            for i in range(frames_01.shape[0]):
                Image.fromarray((np.clip(frames_01[i], 0, 1) * 255).round().astype(np.uint8))\
                     .save(os.path.join(ldr_dir, f"{stem}_frame_{i:02d}.png"))

            # Fuse → relative-scale HDR via the 4-frame FusionUNet.
            # Wan VAE decode is gamma-encoded [0,1]; linearise with ** 2.2 to match training.
            pixel_values_lin = (
                torch.from_numpy(frames_01).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                .clamp(0, 1) ** 2.2
            )                                                                        # (1, T, 3, H, W)
            with torch.no_grad():
                hdr_pred_01, fusion_weights_t = fusion_net(pixel_values_lin)
            hdr_rel        = np.maximum(
                hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy(), 0.0
            ).astype(np.float32)                                                     # (H, W, 3) in [0, ~1]
            fusion_weights = fusion_weights_t[0, :, 0].float().cpu().numpy()         # (T, H, W)
            save_fusion_weights(fusion_weights, weights_dir, stem)

            # Peak-normalise to save_hdr_max for the EXR write. Metrics are scale-invariant
            # (calculate_*_psnr normalises both pred and ref by ref's 99th percentile inside),
            # so we can feed pred_hdr_save straight to them.
            if args.save_hdr_max > 0:
                m = float(hdr_rel.max())
                pred_hdr_save = (hdr_rel * (args.save_hdr_max / m)) if m > 1e-8 else hdr_rel
            else:
                pred_hdr_save = hdr_rel
            hdr_io.imwrite(out_exr, pred_hdr_save.astype(np.float32))

            # Metrics: pu21_peak_lum is the assumed display peak luminance for PU21 encoding;
            # absolute scale of pred_hdr_save is irrelevant to both metrics.
            metrics = {
                "mu_psnr": calculate_mu_psnr(pred_hdr_save, ref_hdr),
                "pu_psnr": calculate_pu_psnr(pred_hdr_save, ref_hdr, peak_lum=args.pu21_peak_lum),
            }
        except Exception as e:
            log.exception(f"[{idx+1}] FAILED on {stem}: {e}")
            failures.append(stem)
            continue

        for k, v in metrics.items():
            all_metrics[k].append(v)
        per_sample_rows.append({"file": os.path.basename(ref_path), **metrics})

        msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        log.info(f"[{idx+1}/{len(ref_paths)}] {stem}: {msg}  [{time.time() - t0:.1f}s]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_eval = len(all_metrics[metric_names[0]])
    total  = time.time() - t0_global
    if n_eval == 0:
        log.warning("No samples evaluated successfully.")
        if failures:
            log.error(f"{len(failures)} failure(s): {failures}")
        sys.exit(1)

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"raw2hdr SI-HDR — Wan I2V LoRA evaluation\n{'='*48}\n")
        f.write(f"Evaluated         : {n_eval} scenes  ({len(failures)} failure(s))\n")
        f.write(f"Testset           : {args.raw2hdr_testset_dir}\n")
        f.write(f"  raw input       : {raw_input_folder}\n")
        f.write(f"  GT HDR          : {gt_hdr_folder}\n")
        f.write(f"Wan base          : {args.pretrained_wan_path}\n")
        f.write(f"LoRA              : {args.lora_checkpoint}\n")
        f.write(f"  trained config  : ref_mode={ref_mode}  num_frames={train_cfg.get('num_frames')}  "
                f"resolution={train_cfg.get('resolution')}  step={train_cfg.get('max_train_steps')}\n")
        f.write(f"Fusion net        : {args.fusion_net_path}  (num_frames={args.num_frames})\n")
        f.write(f"Inference         : {args.num_inference_steps} steps, "
                f"{args.width}x{args.height}, dtype={args.dtype}, seed={args.seed}\n")
        f.write(f"pu21_peak_lum     : {args.pu21_peak_lum}\n")
        f.write(f"save_hdr_max      : {args.save_hdr_max}\n")
        f.write(f"Wall time         : {total:.1f}s  ({total/max(n_eval,1):.1f}s / scene)\n\n")
        f.write("Mean Metrics\n------------\n")
        for k, v in mean_metrics.items():
            f.write(f"  {k:<10s} {v:.4f}\n")
        if failures:
            f.write("\nFailures\n--------\n")
            for s in failures:
                f.write(f"  {s}\n")

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump({"n_eval": n_eval, "means": mean_metrics,
                   "failures": failures}, f, indent=2)

    csv_path = os.path.join(args.output_dir, "metrics_per_image.csv")
    with open(csv_path, "w") as f:
        f.write("file," + ",".join(metric_names) + "\n")
        for row in per_sample_rows:
            f.write(row["file"] + "," + ",".join(f"{row[k]:.6f}" for k in metric_names) + "\n")

    log.info("=" * 50)
    for k, v in mean_metrics.items():
        log.info(f"  {k:<10s} {v:.4f}")
    log.info("=" * 50)
    log.info(f"Summary  → {summary_path}")
    log.info(f"JSON     → {os.path.join(args.output_dir, 'metrics_summary.json')}")
    log.info(f"Per-img  → {csv_path}")
    log.info(f"LDR      → {ldr_dir}/")
    log.info(f"HDR      → {hdr_dir}/")
    log.info(f"Weights  → {weights_dir}/")
    if failures:
        log.error(f"{len(failures)} failure(s): {failures}")


if __name__ == "__main__":
    main()
