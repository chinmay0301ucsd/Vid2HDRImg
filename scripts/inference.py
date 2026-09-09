"""Single-image HDR inference for Vid2HDRImg.

Given a single LDR input (linear raw EXR/HDR or sRGB JPG/PNG/TIFF), generate a
synthetic exposure bracket via the fine-tuned SVD UNet and fuse it into a
predicted HDR image with the lightweight fusion net.

Auto-detection:
  *.exr / *.hdr  -> raw / linear path (matches training preprocessing)
  *.png / *.jpg / *.jpeg / *.tif / *.tiff
                  -> sRGB path; image is linearised by `img ** 2.2` before
                     the same downstream preprocessing as raw inputs.
  Use ``--input_type {raw,srgb}`` to override.

Example:
    python scripts/inference.py \\
        --input          example.jpg \\
        --unet_path      weights/unet \\
        --fusion_net_path weights/fusion_net.pt \\
        --output         predicted.exr
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipelines.pipeline_stable_video_diffusion_hdr import (
    StableVideoDiffusionPipelineHDR,
)
from src.models.fusion_unet import load_fusion_net, save_bracket_pngs

import HDRutils.io as hdr_io
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

RAW_EXTS  = {".exr", ".hdr"}
SRGB_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# CLIP image-encoder normalisation (matches CLIPImageProcessor defaults).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ─── Image loading ────────────────────────────────────────────────────────────
def detect_input_type(path: str, override: str = "auto") -> str:
    """Return 'raw' or 'srgb' based on `--input_type` override or extension."""
    if override in ("raw", "srgb"):
        return override
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTS:
        return "raw"
    if ext in SRGB_EXTS:
        return "srgb"
    raise ValueError(
        f"Cannot infer input type from extension '{ext}'. "
        f"Pass --input_type {{raw,srgb}} explicitly."
    )


def load_input_linear(path: str, input_type: str) -> np.ndarray:
    """Load an image and return linear-light float32 (H, W, 3) in linear space.

    raw : load via HDRutils (already linear).
    srgb: load via PIL, scale to [0, 1], then ``img ** 2.2`` to undo gamma.
    """
    if input_type == "raw":
        img = hdr_io.imread(path).astype(np.float32)
    else:  # srgb
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        img = img ** 2.2  # sRGB gamma -> linear (approximation)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return np.maximum(img[..., :3], 0.0).astype(np.float32)


def preprocess(img_lin: np.ndarray, width: int, height: int, device: torch.device):
    """Resize, p99-normalise, and produce VAE + CLIP tensors.

    Mirrors `load_raw_exr_as_tensor` in the original eval pipeline so the model
    sees an identical input distribution to training.

    Returns:
        vae_input  : (1, 3, H, W) float32 in [0, 1] (the VAE's image input).
        clip_input : (1, 3, 224, 224) float32 CLIP-normalised view.
    """
    if img_lin.shape[0] != height or img_lin.shape[1] != width:
        img_lin = cv2.resize(img_lin, (width, height), interpolation=cv2.INTER_AREA)

    p99 = max(float(np.percentile(img_lin, 99)), 1e-6)
    img_norm = np.clip(img_lin / p99, 0.0, 1.0).astype(np.float32)

    vae_input = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    clip_view = F.interpolate(vae_input, size=(224, 224),
                              mode="bilinear", align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=device).view(1, 3, 1, 1)
    clip_input = (clip_view - mean) / std
    return vae_input, clip_input


# ─── Pipeline assembly ────────────────────────────────────────────────────────
def build_pipeline(args, device: torch.device) -> StableVideoDiffusionPipelineHDR:
    """Load the SVD components + the fine-tuned UNet, return the inference pipeline."""
    log.info(f"Loading base model: {args.pretrained_model_name_or_path}")
    pipeline = StableVideoDiffusionPipelineHDR.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=UNetSpatioTemporalConditionModel.from_pretrained(
            args.unet_path, subfolder="unet"
        ),
        torch_dtype=torch.float32,
        variant="fp16",  # base VAE/image_encoder are shipped as fp16-variant weights;
                         # matches the eval config the released checkpoint was validated on
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Parse CLI args, run the bracket-generation + fusion pipeline on one input image, and save the HDR output."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    ap.add_argument("--input", required=True, type=str,
                    help="Path to a single input image (raw EXR/HDR, or sRGB PNG/JPG/TIFF).")
    ap.add_argument("--output", default="predicted.exr", type=str,
                    help="Output HDR path (.exr).")
    ap.add_argument("--input_type", default="auto", choices=["auto", "raw", "srgb"],
                    help="Override the input-type detection (default: auto from extension).")

    # Model paths
    ap.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-video-diffusion-img2vid",
                    help="Base SVD model. Pass a local path or HF Hub id.")
    ap.add_argument("--unet_path", required=True, type=str,
                    help="Fine-tuned UNet checkpoint directory (contains a `unet/` subfolder).")
    ap.add_argument("--fusion_net_path", required=True, type=str,
                    help="Fusion-net checkpoint (.pt file produced by train_fusion_net.py).")

    # Generation
    ap.add_argument("--num_frames", type=int, default=5,
                    help="Number of LDR frames in the synthesised exposure bracket.")
    ap.add_argument("--num_inference_steps", type=int, default=50,
                    help="Diffusion sampler steps.")
    ap.add_argument("--width",  type=int, default=512, help="Inference width.")
    ap.add_argument("--height", type=int, default=512, help="Inference height.")
    ap.add_argument("--decode_chunk_size", type=int, default=4,
                    help="Frames decoded per VAE forward (lower = less GPU memory). "
                         "Must match the checkpoint's eval config (4) — the temporal VAE "
                         "decoder mixes frames within a chunk, so changing this changes the "
                         "output, not just memory use.")
    ap.add_argument("--motion_bucket_id", type=int, default=0)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--noise_aug_strength", type=float, default=0.0)

    # Guidance
    ap.add_argument("--min_guidance_scale", type=float, default=1.0)
    ap.add_argument("--max_guidance_scale", type=float, default=1.0,
                    help="No-CFG by default (matches the no-CFG release model).")
    ap.add_argument("--clip_only_cfg", action="store_true",
                    help="Drop only CLIP embedding for the unconditional pass.")

    # Output scale
    ap.add_argument("--rescale_max", type=float, default=32.0,
                    help="If > 0, rescale the predicted HDR so its own max equals this value "
                         "(matches the convention used to save the checkpoint's eval outputs, "
                         "hdr_predicted/*.exr, for direct visual comparison). Set to 0 to use "
                         "--peak_lum instead (a fixed physical-brightness assumption, since "
                         "--rescale_max makes --peak_lum a no-op — scaling by a constant then "
                         "renormalizing to a fixed max cancels the constant out).")
    ap.add_argument("--peak_lum", type=float, default=4000.0,
                    help="Peak luminance (cd/m^2) to scale the predicted HDR to. Ignored unless "
                         "--rescale_max is set to 0.")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda",
                    help="Torch device (default: cuda; use 'cpu' for debugging).")

    args = ap.parse_args()

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA / ROCm not available; falling back to CPU (will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ── Load + preprocess input ────────────────────────────────────────────────
    input_type = detect_input_type(args.input, args.input_type)
    log.info(f"Input '{args.input}' detected as type '{input_type}'.")
    img_lin = load_input_linear(args.input, input_type)
    log.info(f"Loaded input shape {img_lin.shape}, range [{img_lin.min():.4g}, {img_lin.max():.4g}]")

    vae_input, clip_input = preprocess(img_lin, args.width, args.height, device)

    # ── Build pipeline + fusion net ────────────────────────────────────────────
    pipeline    = build_pipeline(args, device)
    fusion_net  = load_fusion_net(args.fusion_net_path, args.num_frames, device)

    # ── Generate exposure bracket ──────────────────────────────────────────────
    log.info(f"Generating {args.num_frames}-frame bracket "
             f"({args.num_inference_steps} steps, {args.width}x{args.height})...")
    with torch.no_grad():
        frames_pt = pipeline(
            vae_input,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            clip_image=clip_input,
            min_guidance_scale=args.min_guidance_scale,
            max_guidance_scale=args.max_guidance_scale,
            clip_only_cfg=args.clip_only_cfg,
            conditioning_frame_idx=None,
            output_type="pt",
        ).frames[0]                                # (T, 3, H, W) float32 in [0,1]

    frames_01 = frames_pt.float().clamp(0, 1)

    # ── Save the synthesised bracket ──────────────────────────────────────────
    out_path = args.output
    out_dir  = os.path.dirname(os.path.abspath(out_path)) or "."
    bracket_dir = os.path.join(out_dir, f"{Path(out_path).stem}_bracket")
    save_bracket_pngs(frames_01, bracket_dir)
    log.info(f"Saved {frames_01.shape[0]}-frame bracket -> {bracket_dir}/f*.png")

    # ── Fuse into HDR ─────────────────────────────────────────────────────────
    # Re-create what `run_fusion_net_float` does, but scale the output to a
    # configurable peak luminance instead of using a GT reference.
    pixel_values_lin = frames_01.unsqueeze(0).to(device).clamp(0, 1) ** 2.2
    with torch.no_grad():
        hdr_pred_01, _weights = fusion_net(pixel_values_lin)
    hdr_rel = np.maximum(hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy(), 0.0)  # (H, W, 3) in [0,~1]
    if args.rescale_max > 0:
        rel_max = float(hdr_rel.max())
        pred_hdr = (hdr_rel * (args.rescale_max / rel_max) if rel_max > 1e-8 else hdr_rel)
    else:
        pred_hdr = hdr_rel * float(args.peak_lum)
    pred_hdr = pred_hdr.astype(np.float32)

    # ── Save fused HDR ────────────────────────────────────────────────────────
    hdr_io.imwrite(out_path, pred_hdr)
    log.info(f"Saved predicted HDR -> {out_path}  "
             f"(min={pred_hdr.min():.4g}, max={pred_hdr.max():.4g}, mean={pred_hdr.mean():.4g})")


if __name__ == "__main__":
    main()
