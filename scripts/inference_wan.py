"""Single-image HDR inference using the Wan 2.1 I2V LoRA + 4-frame fusion net.

Drop-in counterpart to `inference.py`. Replaces the SVD UNet with the Wan 2.1
I2V backbone fine-tuned via `train_wan_i2v_lora.py` (ref_mode=wan_i2v_anchored,
num_frames=4). The pipeline generates T_pix = num_frames + 1 frames where
frame 0 is the anchor (= reference) and frames 1..num_frames are the editable
bracket; the anchor is dropped before fusion.

Saves:
  <output>.exr               — predicted linear HDR (peak-luminance scaled).
  <output_stem>_bracket/f*.png — generated LDR bracket frames (matches
  the spirit of `infer_wan_vace_lora.py`).

Auto-detection:
  *.exr / *.hdr  -> raw / linear path (matches training preprocessing)
  *.png / *.jpg / *.jpeg / *.tif / *.tiff
                  -> sRGB path; image is linearised by `img ** 2.2` before
                     the same downstream preprocessing as raw inputs.
  Use ``--input_type {raw,srgb}`` to override.

Example:
    python scripts/inference_wan.py \\
        --input               example.exr \\
        --lora_checkpoint     /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/wan_i2v_lora_v2_anchored/chain_run/lora-4000/lora_weights.pt \\
        --fusion_net_path     /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/outputs/fusion_net_4frames/checkpoint-16000/fusion_net.pt \\
        --pretrained_wan_path /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/models/Wan2.1-I2V-14B-480P-Diffusers \\
        --output              predicted.exr
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.fusion_unet import load_fusion_net
from train_wan_i2v_lora import (
    build_i2v_condition,
    clip_encode_image,
    encode_text_empty,
    wan_vae_denormalize,
)

import HDRutils.io as hdr_io
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanTransformer3DModel,
)
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    T5TokenizerFast,
    UMT5EncoderModel,
)
from peft import LoraConfig, set_peft_model_state_dict


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

RAW_EXTS  = {".exr", ".hdr"}
SRGB_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ─── Image loading ────────────────────────────────────────────────────────────
def detect_input_type(path: str, override: str = "auto") -> str:
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
    """Load an image and return linear-light float32 (H, W, 3) in linear space."""
    if input_type == "raw":
        img = hdr_io.imread(path).astype(np.float32)
    else:  # srgb
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        img = img ** 2.2  # sRGB gamma -> linear (approximation)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return np.maximum(img[..., :3], 0.0).astype(np.float32)


def preprocess_reference(img_lin: np.ndarray, width: int, height: int,
                         device: torch.device, dtype: torch.dtype):
    """Resize, p99-normalise, and produce Wan-pipeline tensors.

    Mirrors `train_wan_i2v_lora.py`'s validation prep so the model sees an
    identical input distribution to training:
      * resize to (width, height),
      * p99-normalise into [0, 1] linear,
      * CLIP path: uint8 of the LINEAR view (training feeds the same),
      * VAE path:  (1, 3, H, W) in [-1, 1].

    Returns:
        ref_uint8     : (H, W, 3) uint8     — for CLIP image encoder.
        ref_minus1to1 : (1, 3, H, W) tensor — for VAE conditioning.
    """
    if img_lin.shape[0] != height or img_lin.shape[1] != width:
        img_lin = cv2.resize(img_lin, (width, height), interpolation=cv2.INTER_AREA)

    p99 = max(float(np.percentile(img_lin, 99)), 1e-6)
    img_norm = np.clip(img_lin / p99, 0.0, 1.0).astype(np.float32)

    ref_uint8     = (img_norm * 255).round().clip(0, 255).astype(np.uint8)
    ref_minus1to1 = (torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
                      * 2.0 - 1.0).to(device=device, dtype=dtype)
    return ref_uint8, ref_minus1to1


# ─── Pipeline assembly ────────────────────────────────────────────────────────
def load_wan_components(pretrained_wan_path: str, dtype: torch.dtype):
    """Load all Wan 2.1 I2V components."""
    log.info(f"Loading Wan I2V components from {pretrained_wan_path} (dtype={dtype}) …")
    tokenizer     = T5TokenizerFast.from_pretrained(pretrained_wan_path, subfolder="tokenizer")
    text_encoder  = UMT5EncoderModel.from_pretrained(pretrained_wan_path,
                                                     subfolder="text_encoder",
                                                     torch_dtype=dtype)
    vae           = AutoencoderKLWan.from_pretrained(pretrained_wan_path,
                                                     subfolder="vae",
                                                     torch_dtype=dtype)
    scheduler     = UniPCMultistepScheduler.from_pretrained(pretrained_wan_path,
                                                            subfolder="scheduler")
    transformer   = WanTransformer3DModel.from_pretrained(pretrained_wan_path,
                                                          subfolder="transformer",
                                                          torch_dtype=dtype,
                                                          low_cpu_mem_usage=True)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_wan_path,
                                                                  subfolder="image_encoder",
                                                                  torch_dtype=dtype)
    feat_proc     = CLIPImageProcessor.from_pretrained(pretrained_wan_path,
                                                       subfolder="image_processor")

    for m in (vae, text_encoder, image_encoder, transformer):
        m.requires_grad_(False)
    vae.eval(); text_encoder.eval(); image_encoder.eval(); transformer.eval()

    return tokenizer, text_encoder, vae, scheduler, transformer, image_encoder, feat_proc


def attach_lora(transformer, lora_ckpt_path: str):
    """Build a LoraConfig from the trained checkpoint's saved args and load weights."""
    log.info(f"Loading LoRA checkpoint from {lora_ckpt_path} …")
    blob = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
    cfg  = blob["config"]
    log.info(f"  LoRA trained config: rank={cfg.get('lora_rank')} alpha={cfg.get('lora_alpha')} "
             f"num_frames={cfg.get('num_frames')} resolution={cfg.get('resolution')} "
             f"ref_mode={cfg.get('ref_mode')} step={blob.get('step')}")

    lora_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "ffn.net.0.proj", "ffn.net.2",
                        "proj_out"],
        bias="none", init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    set_peft_model_state_dict(transformer, blob["lora_state_dict"])
    log.info(f"  LoRA applied: {len(blob['lora_state_dict'])} parameter tensors loaded")
    return cfg


# ─── Denoising loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_wan_i2v(transformer, vae, scheduler, image_encoder, feat_proc,
                empty_text_embed, ref_uint8, ref_minus1to1,
                latents_mean, latents_std_inv,
                num_frames: int, num_inference_steps: int,
                device: torch.device, dtype: torch.dtype,
                seed: int, ref_mode: str) -> np.ndarray:
    """Generate the LDR bracket. Returns (num_frames, H, W, 3) float32 in [0, 1]."""
    image_embeds = clip_encode_image(image_encoder, feat_proc,
                                     ref_uint8[None], device, dtype)

    cond_pixel_frames = num_frames + 1 if ref_mode == "wan_i2v_anchored" else num_frames
    condition = build_i2v_condition(ref_minus1to1, cond_pixel_frames,
                                    vae, latents_mean, latents_std_inv,
                                    ref_mode=ref_mode)

    T_lat, latH, latW = condition.shape[2], condition.shape[3], condition.shape[4]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, vae.config.z_dim, T_lat, latH, latW,
                          generator=gen).to(device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents, condition], dim=1)
        model_out = transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(1),
            encoder_hidden_states=empty_text_embed[:1],
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
        )[0]
        latents = scheduler.step(model_out, t, latents, return_dict=False)[0]

    latents = wan_vae_denormalize(latents.float(), latents_mean, latents_std_inv).to(vae.dtype)
    video   = vae.decode(latents, return_dict=False)[0]
    frames  = video[0].clamp(-1, 1).permute(1, 2, 3, 0)                         # (T_pix, H, W, 3)
    frames_01 = ((frames + 1) * 0.5).clamp(0, 1).float().cpu().numpy()          # gamma-encoded [0, 1]

    if ref_mode == "wan_i2v_anchored":
        frames_01 = frames_01[1:1 + num_frames]
    return frames_01.astype(np.float32)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    ap.add_argument("--input", required=True, type=str,
                    help="Path to a single input image (raw EXR/HDR, or sRGB PNG/JPG/TIFF).")
    ap.add_argument("--output", default="predicted.exr", type=str,
                    help="Output HDR path (.exr). Bracket frames are written next to it.")
    ap.add_argument("--input_type", default="auto", choices=["auto", "raw", "srgb"],
                    help="Override the input-type detection (default: auto from extension).")

    # Model paths
    ap.add_argument("--pretrained_wan_path", required=True, type=str,
                    help="Path to the Wan 2.1 I2V Diffusers snapshot directory.")
    ap.add_argument("--lora_checkpoint", required=True, type=str,
                    help="LoRA checkpoint .pt file (lora_weights.pt) saved by train_wan_i2v_lora.py.")
    ap.add_argument("--fusion_net_path", required=True, type=str,
                    help="Fusion-net checkpoint (.pt file produced by train_fusion_net.py).")

    # Generation
    ap.add_argument("--num_frames", type=int, default=4,
                    help="Number of LDR bracket frames (must match the fusion net's training).")
    ap.add_argument("--num_inference_steps", type=int, default=50,
                    help="Diffusion sampler steps.")
    ap.add_argument("--width",  type=int, default=512, help="Inference width.")
    ap.add_argument("--height", type=int, default=512, help="Inference height.")
    ap.add_argument("--ref_mode", default=None,
                    choices=["wan_i2v_anchored", "soft_wan_ref", "strict_i2v", "editable"],
                    help="Override the LoRA's training ref_mode (default: read from checkpoint).")

    # Output scale
    ap.add_argument("--save_hdr_max", type=float, default=32.0,
                    help="Rescale predicted HDR so its max == this value before .exr write "
                         "(matches eval_wan_sihdr.py default). Set <=0 to skip rescaling "
                         "and save the fusion-net output unscaled (max ≈ 1).")

    # Misc
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--dtype",  default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--device", default="cuda",
                    help="Torch device (default: cuda; use 'cpu' for debugging).")

    args = ap.parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA / ROCm not available; falling back to CPU (will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype  = _DTYPES[args.dtype]
    torch.manual_seed(args.seed)

    # ── Load + preprocess input ───────────────────────────────────────────────
    input_type = detect_input_type(args.input, args.input_type)
    log.info(f"Input '{args.input}' detected as type '{input_type}'.")
    img_lin = load_input_linear(args.input, input_type)
    log.info(f"Loaded input shape {img_lin.shape}, range [{img_lin.min():.4g}, {img_lin.max():.4g}]")

    ref_uint8, ref_minus1to1 = preprocess_reference(img_lin, args.width, args.height, device, dtype)

    # ── Build Wan I2V pipeline + LoRA ─────────────────────────────────────────
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

    # ── Generate the LDR bracket ──────────────────────────────────────────────
    log.info(f"Generating {args.num_frames}-frame bracket via Wan I2V LoRA "
             f"({args.num_inference_steps} steps, {args.width}x{args.height})…")
    frames_01 = run_wan_i2v(
        transformer, vae, scheduler, image_encoder, feat_proc,
        empty_text_embed, ref_uint8, ref_minus1to1,
        latents_mean, latents_std_inv,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        device=device, dtype=dtype,
        seed=args.seed, ref_mode=ref_mode,
    )                                                                            # (T, H, W, 3) [0,1]

    # Save bracket alongside the output, e.g. predicted.exr -> predicted_bracket/f00.png
    out_path = args.output
    out_dir  = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_stem = Path(out_path).stem
    bracket_dir = os.path.join(out_dir, f"{out_stem}_bracket")
    os.makedirs(bracket_dir, exist_ok=True)
    for i in range(frames_01.shape[0]):
        Image.fromarray((np.clip(frames_01[i], 0, 1) * 255).round().astype(np.uint8))\
             .save(os.path.join(bracket_dir, f"f{i:02d}.png"))
    log.info(f"Saved bracket PNGs -> {bracket_dir}/f*.png")

    # ── Fuse into HDR ─────────────────────────────────────────────────────────
    # Wan VAE decode returns gamma-encoded LDR in [0, 1]; linearise with ** 2.2
    # to match what the fusion net saw during training (same as run_fusion_net_float).
    pixel_values_lin = (
        torch.from_numpy(frames_01).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        .clamp(0, 1) ** 2.2
    )                                                                            # (1, T, 3, H, W)
    with torch.no_grad():
        hdr_pred_01, _weights = fusion_net(pixel_values_lin)
    hdr_rel = np.maximum(hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy(),
                         0.0).astype(np.float32)                                 # (H, W, 3) in [0, ~1]

    # Peak-normalise to save_hdr_max for the EXR write (set <=0 to keep raw fusion output).
    if args.save_hdr_max > 0:
        m = float(hdr_rel.max())
        pred_hdr = (hdr_rel * (args.save_hdr_max / m)) if m > 1e-8 else hdr_rel
    else:
        pred_hdr = hdr_rel

    hdr_io.imwrite(out_path, pred_hdr.astype(np.float32))
    log.info(f"Saved predicted HDR -> {out_path}  "
             f"(min={pred_hdr.min():.4g}, max={pred_hdr.max():.4g}, mean={pred_hdr.mean():.4g})")


if __name__ == "__main__":
    main()
