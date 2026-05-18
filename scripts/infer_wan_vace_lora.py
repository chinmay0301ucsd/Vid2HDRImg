"""Inference for the Wan 2.1 VACE LoRA raw→LDR-bracket model.

Loads base Wan VACE 14B + the trained LoRA adapter and generates a T-frame
LDR bracket conditioned on a single raw image (the VACE V-stream reference).

For pure-generation use (no input video to inpaint), the VACE pipeline still
requires a `video` placeholder and a `mask`; we pass a zero video + all-ones
mask, so the model has to rely entirely on the reference image conditioning.

Usage:
    python scripts/infer_wan_vace_lora.py \\
        --lora_checkpoint /work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_vace_lora_v0/lora-final/lora_weights.pt \\
        --input_image /work1/javidi/shared/raw_hdr_dataset_v1/raw/test_image.exr \\
        --output_dir /work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_vace_lora_v0/inference \\
        --num_frames 5

  # Or pull a sample from the training set (uses the same RawHDRPairDataset):
    python scripts/infer_wan_vace_lora.py \\
        --lora_checkpoint .../lora_weights.pt \\
        --from_dataset_index 0 \\
        --output_dir .../inference
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from peft import LoraConfig, set_peft_model_state_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset_hdr import RawHDRPairDataset


# ─── Loading helpers ────────────────────────────────────────────────────────

def load_pipeline_and_lora(pretrained_wan_path: str, lora_ckpt_path: str,
                           weight_dtype: torch.dtype = torch.bfloat16,
                           device: str = "cuda"):
    """Load WanVACEPipeline + apply LoRA weights from disk to its transformer."""
    print(f"Loading WanVACEPipeline from {pretrained_wan_path} (dtype={weight_dtype}) …")
    pipe = WanVACEPipeline.from_pretrained(
        pretrained_wan_path, torch_dtype=weight_dtype,
    )

    print(f"Loading LoRA checkpoint from {lora_ckpt_path} …")
    ck = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
    lora_sd = ck["lora_state_dict"]
    cfg = ck["config"]
    print(f"  trained config: lora_rank={cfg.get('lora_rank')}  "
          f"lora_alpha={cfg.get('lora_alpha')}  num_frames={cfg.get('num_frames')}  "
          f"resolution={cfg.get('resolution')}  steps_at_save={ck.get('step')}")

    lora_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none", init_lora_weights="gaussian",
    )
    pipe.transformer.add_adapter(lora_config)
    set_peft_model_state_dict(pipe.transformer, lora_sd)
    print(f"  LoRA applied: {len(lora_sd)} parameter tensors loaded")

    pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe, cfg


# ─── Input prep ─────────────────────────────────────────────────────────────

def load_raw_input_pil(path: str, resolution: int) -> Image.Image:
    """Load any image file → PIL.Image (RGB, resolution × resolution).

    Handles .exr/.hdr via HDRutils, otherwise PIL.  EXR/HDR are normalised by
    99th percentile and gamma-encoded for display (same convention as
    train_video_model.py's raw_input preprocessing).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".exr", ".hdr"):
        import HDRutils.io as hdr_io
        img = hdr_io.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = np.maximum(img[..., :3], 0.0)
        p99 = float(np.percentile(img, 99)) or 1.0
        img = np.clip(img / p99, 0, 1) ** (1.0 / 2.2)              # γ for display
        img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
    else:
        pil = Image.open(path).convert("RGB")
    # Center-crop + resize to resolution × resolution.
    w, h = pil.size
    s = min(w, h)
    pil = pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    pil = pil.resize((resolution, resolution), Image.LANCZOS)
    return pil


def make_zero_video(num_frames: int, resolution: int):
    """Placeholder for VACE's `video` arg — gray midpoint frames (mask=ones means it'll all be generated)."""
    canvas = Image.new("RGB", (resolution, resolution), color=(128, 128, 128))
    return [canvas for _ in range(num_frames)]


def make_full_mask(num_frames: int, resolution: int):
    """All-ones binary mask: 'generate everything'."""
    white = Image.new("L", (resolution, resolution), color=255)
    return [white for _ in range(num_frames)]


# ─── Visualization ──────────────────────────────────────────────────────────

def save_bracket(frames_uint8: np.ndarray, ref_pil: Image.Image, out_path: str, title: str = ""):
    """frames_uint8: (T, H, W, 3) uint8 — the generated bracket."""
    T = frames_uint8.shape[0]
    fig, axes = plt.subplots(1, T + 1, figsize=(3 * (T + 1), 3.3))
    # Reference first
    axes[0].imshow(ref_pil)
    axes[0].set_title("ref (raw)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for i in range(T):
        axes[i + 1].imshow(frames_uint8[i])
        axes[i + 1].set_title(f"gen f{i}", fontsize=9)
        axes[i + 1].set_xticks([]); axes[i + 1].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained_wan_path", default=os.environ.get(
        "PRETRAINED_WAN",
        "/work1/javidi/chinmay0301/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.1-VACE-14B-Diffusers/snapshots/"
        "db79b90c60bbb45ceec9e41b9d5a4df934538ac4"))
    p.add_argument("--lora_checkpoint", required=True,
                   help="Path to lora_weights.pt saved by train_wan_vace_lora.py")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_image", help="Path to a raw/RGB image to use as reference.")
    src.add_argument("--from_dataset_index", type=int, default=None,
                     help="Pull sample by index from raw_hdr_dataset_v1.")
    p.add_argument("--data_dir", default="/work1/javidi/shared/raw_hdr_dataset_v1",
                   help="Dataset root (used with --from_dataset_index).")

    p.add_argument("--output_dir",      required=True)
    p.add_argument("--num_frames",      type=int, default=5)
    p.add_argument("--resolution",      type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale",  type=float, default=1.0,
                   help="CFG scale. We trained WITHOUT conditioning-dropout, so the model "
                        "has only seen 'with conditioning' forward passes. Setting >1 would "
                        "enable CFG but the uncond branch is out-of-distribution; keep at 1.0 "
                        "unless you've retrained with CFG dropout.")
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--dtype",           default="bf16", choices=["fp32", "fp16", "bf16"])
    return p.parse_args()


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = _DTYPES[args.dtype]

    # 1. Load pipeline + LoRA.
    pipe, train_cfg = load_pipeline_and_lora(
        args.pretrained_wan_path, args.lora_checkpoint, weight_dtype=dtype, device=device,
    )

    # 2. Prepare the reference image.
    if args.from_dataset_index is not None:
        print(f"Pulling sample index {args.from_dataset_index} from {args.data_dir}")
        dataset = RawHDRPairDataset(
            base_folder=args.data_dir,
            sample_frames=args.num_frames,
            use_noisy_samples=True,
        )
        sample = dataset[args.from_dataset_index]
        # raw_input is (3, H, W) in [-1, 1]; convert to PIL via [0, 1] → uint8.
        raw = ((sample["raw_input"].float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).byte()
        raw_np = raw.permute(1, 2, 0).numpy()
        ref_pil = Image.fromarray(raw_np)
        if ref_pil.size != (args.resolution, args.resolution):
            ref_pil = ref_pil.resize((args.resolution, args.resolution), Image.LANCZOS)
        tag = f"ds{args.from_dataset_index}"
    else:
        ref_pil = load_raw_input_pil(args.input_image, args.resolution)
        tag = os.path.splitext(os.path.basename(args.input_image))[0]

    ref_pil.save(os.path.join(args.output_dir, f"reference_{tag}.png"))

    # 3. Build the pipeline call inputs:
    #    - video       : gray placeholder frames (VACE requires a video tensor; mask=ones means it gets regenerated)
    #    - mask        : all-white (generate everything)
    #    - reference_images : the raw image
    video_pil = make_zero_video(args.num_frames, args.resolution)
    mask_pil  = make_full_mask (args.num_frames, args.resolution)

    # Pre-compute the empty UMT5 embedding the *same way* as in training
    # (tokenize "" with max_length=512, run text_encoder, take last_hidden_state).
    # This bypasses the pipeline's prompt_clean → ftfy → encode_prompt path entirely.
    print("Pre-computing empty text embedding (bypassing pipeline encode_prompt) …")
    with torch.no_grad():
        tok = pipe.tokenizer(
            [""], padding="max_length", max_length=512,
            truncation=True, return_tensors="pt",
        )
        empty_embed = pipe.text_encoder(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
        ).last_hidden_state.to(dtype=dtype)
    print(f"  empty_embed shape={tuple(empty_embed.shape)}  dtype={empty_embed.dtype}")

    print(f"Running pipeline: T={args.num_frames}  res={args.resolution}  "
          f"steps={args.num_inference_steps}  cfg={args.guidance_scale}")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        out = pipe(
            video=video_pil,
            mask=mask_pil,
            reference_images=[ref_pil],
            prompt_embeds=empty_embed,
            negative_prompt_embeds=empty_embed,    # safe even at cfg=1 (just defines uncond branch)
            height=args.resolution,
            width=args.resolution,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="np",                      # (T, H, W, 3) float32 in [0, 1]
        )

    frames_np = out.frames[0]                  # (T, H, W, 3) float32
    frames_u8 = (frames_np.clip(0, 1) * 255).round().astype(np.uint8)

    # 4. Save outputs.
    bracket_dir = os.path.join(args.output_dir, f"bracket_{tag}")
    os.makedirs(bracket_dir, exist_ok=True)
    for i in range(frames_u8.shape[0]):
        Image.fromarray(frames_u8[i]).save(os.path.join(bracket_dir, f"f{i:02d}.png"))
    grid_path = os.path.join(args.output_dir, f"grid_{tag}.png")
    save_bracket(frames_u8, ref_pil, grid_path,
                 title=f"Wan VACE LoRA — {tag}  "
                       f"(LoRA step {torch.load(args.lora_checkpoint, weights_only=False).get('step', '?')})")
    print(f"\nSaved per-frame PNGs → {bracket_dir}/f*.png")
    print(f"Saved grid           → {grid_path}")


if __name__ == "__main__":
    main()
