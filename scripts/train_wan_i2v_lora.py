"""LoRA fine-tune of Wan 2.1 Image-to-Video 14B for raw → LDR-bracket.

Same dataset/objective as train_wan_vace_lora.py but with the cleaner Wan I2V
backbone (WanTransformer3DModel + CLIP image encoder), which removes VACE's
mask / reactive / inactive plumbing entirely.  Architecture matches the SVD
training pattern in train_video_model.py:

  - The reference image (raw_input) is **CLIP-encoded** → (1, 1, 1280) and
    passed via `encoder_hidden_states_image`.
  - The reference is also **VAE-encoded as the 1st frame of a padded video**
    (rest zeros) → latent_condition (B, 16, T_lat, H', W').
  - A binary mask channel (4-ch, packed via temporal reshape) marks the
    conditioning frame.
  - condition = cat([mask (4 ch), latent_condition (16 ch)]) → 20 ch.
  - transformer input = cat([noisy_latents (16 ch), condition (20 ch)]) → 36 ch.

Sigma sampling uses **logit-normal** (the Wan / SD3 convention): sample
u~N(0,1), σ = sigmoid(u).  Concentrates around σ≈0.5; avoids the degenerate
σ≈0 and σ≈1 corners that uniform sampling produces.

LoRA rank 64 on attn1/attn2 to_q/k/v/to_out.0 in all blocks.

Usage:
    accelerate launch --num_processes 1 train_wan_i2v_lora.py \\
        --pretrained_wan_path /work1/javidi/shared/Wan2.1-I2V-14B-480P-Diffusers \\
        --data_dir /work1/javidi/shared/raw_hdr_dataset_v1 \\
        --output_dir /work1/javidi/chinmay0301/Vid2HDRImg/outputs/wan_i2v_lora_v0 \\
        --num_frames 5 --resolution 512 --train_batch_size 1 \\
        --learning_rate 5e-5 --lora_rank 64 \\
        --gradient_checkpointing --mixed_precision bf16
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanTransformer3DModel,
)
from diffusers.optimization import get_scheduler
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    T5TokenizerFast,
    UMT5EncoderModel,
)

from peft import LoraConfig, get_peft_model_state_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset_hdr import RawHDRPairDataset, hdr_to_ldr_batch_np, read_hdr_image_float32

logger = logging.getLogger(__name__)


def _resolve_resume_path(spec, output_dir):
    """Map --resume_from_checkpoint to a concrete lora-N dir, or None.

    Accepts an absolute dir path, the literal "latest" (highest-N lora-N under
    output_dir that has a train_state.pt), or None.
    """
    if not spec:
        return None
    if spec != "latest":
        return spec if os.path.isdir(spec) else None
    if not os.path.isdir(output_dir):
        return None
    cands = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"lora-(\d+)", name)
        if m and os.path.exists(os.path.join(output_dir, name, "train_state.pt")):
            cands.append((int(m.group(1)), os.path.join(output_dir, name)))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained_wan_path", required=True,
                   help="Path to the Wan 2.1 I2V Diffusers snapshot directory.")
    p.add_argument("--data_dir", required=True,
                   help="Path to raw_hdr_dataset_v1 (must contain raw/ and gt_hdr/).")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--num_frames",     type=int, default=5)
    p.add_argument("--resolution",     type=int, default=512)
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps",  type=int, default=20000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate",    type=float, default=5e-5)
    p.add_argument("--lr_scheduler",     default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps",  type=int, default=200)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--adam_beta1",       type=float, default=0.9)
    p.add_argument("--adam_beta2",       type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon",     type=float, default=1e-8)
    p.add_argument("--seed",             type=int, default=0)

    p.add_argument("--lora_rank",        type=int, default=64)
    p.add_argument("--lora_alpha",       type=int, default=64)
    p.add_argument("--lora_dropout",     type=float, default=0.0)
    p.add_argument("--target_modules",   type=str,
                   default="to_q,to_k,to_v,to_out.0,ffn.net.0.proj,ffn.net.2,proj_out",
                   help="Comma-separated PEFT target_modules. Default = attn (self+cross) + FFN + "
                        "patch-unembed (matches v3). Pass an expanded set like "
                        "'to_q,to_k,to_v,to_out.0,add_k_proj,add_v_proj,ffn.net.0.proj,ffn.net.2,proj_out' "
                        "to also LoRA the image-stream KV projections in cross-attn (Wan I2V's "
                        "attn2 has separate add_k_proj/add_v_proj for image features).")

    p.add_argument("--sigma_distribution", choices=["uniform", "logit_normal"],
                   default="uniform",
                   help="Base sigma distribution before flow_shift is applied.\n"
                        "  uniform (DEFAULT) — sigma_raw ~ U(0, 1). Matches what the inference\n"
                        "      UniPC scheduler walks through (linspace from σ_max → σ_min, with\n"
                        "      the same flow_shift transform applied on top).\n"
                        "  logit_normal — sigma_raw = sigmoid(N(mean, std)). SD3-style; biases\n"
                        "      toward σ≈0.5 (after sigmoid) before shifting. Kept for ablation.")
    p.add_argument("--logit_normal_mean", type=float, default=0.0,
                   help="Only used if --sigma_distribution=logit_normal.")
    p.add_argument("--logit_normal_std",  type=float, default=1.0,
                   help="Only used if --sigma_distribution=logit_normal.")
    p.add_argument("--flow_shift", type=float, default=None,
                   help="Flow-matching shift applied to sigma_raw: "
                        "sigma = flow_shift * sigma_raw / (1 + (flow_shift - 1) * sigma_raw). "
                        "If None, reads from scheduler.config.flow_shift "
                        "(3.0 for Wan 2.1 I2V, 5.0 for Wan 2.2 TI2V).")
    p.add_argument("--ref_mode",
                   choices=["wan_i2v_anchored", "soft_wan_ref", "strict_i2v", "editable"],
                   default="wan_i2v_anchored",
                   help="Layout of the 20-ch I2V conditioning tensor:\n"
                        "  wan_i2v_anchored (DEFAULT) — target_video=[ref, bracket_0..T-1] (T+1 pixel\n"
                        "      frames), conditioning latent=[ref, 0, ..., 0], mask=[1, 0, ..., 0].\n"
                        "      Matches Wan I2V's pretrained convention exactly: frame 0 is the\n"
                        "      anchor (target == conditioning), frames 1..T are the editable bracket.\n"
                        "      Loss is computed on all latents but the anchor slot is trivial because\n"
                        "      target == conditioning there.\n"
                        "  soft_wan_ref — target=bracket (T frames), latent=[ref, 0, ...], mask=zeros.\n"
                        "      No anchor; ref appears only in the conditioning channels.\n"
                        "  strict_i2v — target=bracket, latent=[ref, 0, ...], mask=[1, 0, ...].\n"
                        "      Target frame 0 = bracket_0 but mask claims frame 0 = ref → inconsistent;\n"
                        "      kept as a baseline for ablation.\n"
                        "  editable — target=bracket, latent=ref broadcast to all T frames,\n"
                        "      mask=zeros. SVD-style soft reference.")

    p.add_argument("--mixed_precision",  default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                   help="Delete older lora-N checkpoints, keeping only the N most recent. "
                        "Set to 0 to keep all. lora-final is never pruned.")
    p.add_argument("--logging_steps",       type=int, default=10)
    p.add_argument("--validation_steps",    type=int, default=500,
                   help="Run periodic validation render (0 to disable).")
    p.add_argument("--num_validation_inference_steps", type=int, default=20)
    p.add_argument("--num_validation_samples", type=int, default=1)
    p.add_argument("--validation_sample_index", type=int, default=0)
    p.add_argument("--report_to",           default="tensorboard",
                   choices=["tensorboard", "wandb", "all", "none"])
    p.add_argument("--use_noisy_samples",   action="store_true", default=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a lora-N checkpoint dir, or 'latest' to auto-find "
                        "the highest-step lora-N under --output_dir. Restores LoRA "
                        "weights, optimizer, LR scheduler, and global_step.")
    return p.parse_args()


# ─── Conditioning helpers ───────────────────────────────────────────────────

@torch.no_grad()
def encode_text_empty(tokenizer, text_encoder, device, dtype, max_length: int = 512):
    inputs = tokenizer([""], padding="max_length", max_length=max_length,
                       truncation=True, return_tensors="pt")
    out = text_encoder(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device))
    return out.last_hidden_state.to(dtype=dtype)


@torch.no_grad()
def clip_encode_image(image_encoder, feat_processor, image_uint8_hwc, device, dtype):
    """image_uint8_hwc: np.ndarray (B, H, W, 3) uint8 → (B, seq_len, 1280) embeds.

    Mirrors WanImageToVideoPipeline.encode_image — takes the PENULTIMATE
    transformer layer's hidden_states (full token sequence, 1280-dim) from
    CLIP-ViT-H, NOT the projection .image_embeds (1024-dim, single token).
    Wan's image_embedder LayerNorm is sized [1280] and expects (B, S, 1280).
    """
    inputs = feat_processor(images=list(image_uint8_hwc), return_tensors="pt")
    pixel  = inputs["pixel_values"].to(device, dtype=dtype)
    out    = image_encoder(pixel, output_hidden_states=True)
    embeds = out.hidden_states[-2]                                        # (B, 257, 1280)
    return embeds.to(dtype=dtype)


def wan_vae_normalize(latents, mean, std_inv):
    return (latents.float() - mean) * std_inv


def wan_vae_denormalize(latents, mean, std_inv):
    return latents.float() / std_inv + mean


@torch.no_grad()
def encode_video_with_wan_vae(vae, video_btchw, mean, std_inv):
    """video_btchw: (B, 3, T, H, W) in [-1, 1] → (B, 16, T_lat, H', W')  normalised."""
    z = vae.encode(video_btchw).latent_dist.mode()
    return wan_vae_normalize(z, mean, std_inv).to(video_btchw.dtype)


def build_i2v_condition(ref_image_b3hw: torch.Tensor, num_frames: int,
                        vae, latents_mean, latents_std_inv,
                        vae_scale_factor_temporal: int = 4,
                        ref_mode: str = "wan_i2v_anchored") -> torch.Tensor:
    """Build the 20-ch conditioning tensor that gets channel-concat'd to noisy latents.

    Modes (see --ref_mode in parse_args for the full user-facing description):

      wan_i2v_anchored / strict_i2v:
        - video_condition = [ref, 0, ..., 0]   (ref at frame 0, zeros elsewhere)
        - mask = [1, 0, ..., 0] at the FIRST latent slot only (which corresponds to
          pixel frame 0 in the Wan VAE)
        These two modes produce identical conditioning here; they differ only in how
        the training loop builds the *target* video.

      soft_wan_ref:
        - video_condition = [ref, 0, ..., 0]
        - mask = all zeros

      editable:
        - video_condition = ref broadcast across all T frames
        - mask = all zeros

    Args:
        num_frames : T pixel frames the model will operate on. For wan_i2v_anchored,
                     this is (bracket_size + 1) — the anchor frame counts as one of the T.
    """
    if ref_mode not in ("wan_i2v_anchored", "soft_wan_ref", "strict_i2v", "editable"):
        raise ValueError(f"unknown ref_mode={ref_mode!r}")
    B, _, H, W = ref_image_b3hw.shape
    device = ref_image_b3hw.device
    dtype  = ref_image_b3hw.dtype

    if ref_mode == "editable":
        # Broadcast the reference across every temporal slot (SVD-style).
        video_condition = ref_image_b3hw.unsqueeze(2).expand(B, 3, num_frames, H, W).contiguous()
    else:
        # wan_i2v_anchored / soft_wan_ref / strict_i2v: ref at frame 0, zeros elsewhere.
        image = ref_image_b3hw.unsqueeze(2)                                # (B, 3, 1, H, W)
        zeros_pad = image.new_zeros(B, 3, num_frames - 1, H, W)
        video_condition = torch.cat([image, zeros_pad], dim=2)             # (B, 3, T, H, W)

    latent_condition = encode_video_with_wan_vae(
        vae, video_condition, latents_mean, latents_std_inv,
    )                                                                       # (B, 16, T_lat, H', W')
    T_lat = latent_condition.shape[2]
    latH  = latent_condition.shape[3]
    latW  = latent_condition.shape[4]

    # Mask lives directly in latent space: (B, 4, T_lat, H', W'). Slot 0 corresponds
    # to pixel frame 0 in the Wan VAE (the encoder treats the first frame as a clean
    # key frame; subsequent latent slots fuse blocks of pixel frames). This simpler
    # form is functionally equivalent to the temporal-packing dance used previously
    # for T_pix ∈ {1, 5, 9, 13, ...} and additionally supports arbitrary T_pix.
    mask_lat = torch.zeros(B, 4, T_lat, latH, latW, device=device, dtype=dtype)
    if ref_mode in ("strict_i2v", "wan_i2v_anchored"):
        mask_lat[:, :, 0] = 1.0

    return torch.cat([mask_lat, latent_condition], dim=1)                  # (B, 20, T_lat, H', W')


# ─── Metric helpers ────────────────────────────────────────────────────────

def make_gt_pred_panel(gt_uint8: np.ndarray, pred_uint8: np.ndarray,
                       ref_uint8: np.ndarray,
                       psnrs: list, ssims: list,
                       step: int, seed: int,
                       save_path: str) -> np.ndarray:
    """Render a 2-row figure: GT on top, prediction on bottom, with per-frame PSNR / SSIM
    annotated below each pred column.  Reference image is shown to the left.  Saved as PNG.

    Returns the rendered RGB numpy array so it can also be uploaded to wandb."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = gt_uint8.shape[0]
    fig, axes = plt.subplots(2, T + 1, figsize=(2.8 * (T + 1), 5.6),
                             gridspec_kw={"hspace": 0.1, "wspace": 0.05})
    # Row 0: reference + GT
    axes[0, 0].imshow(ref_uint8)
    axes[0, 0].set_title("reference (raw, γ)", fontsize=9)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for i in range(T):
        axes[0, i + 1].imshow(gt_uint8[i])
        axes[0, i + 1].set_title(f"GT f{i}", fontsize=9)
        axes[0, i + 1].set_xticks([]); axes[0, i + 1].set_yticks([])
    # Row 1: blank cell under reference + predictions with metric annotations
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, f"step {step}\nseed {seed}\nmean PSNR\n{float(np.mean(psnrs)):.2f}\n"
                              f"mean SSIM\n{float(np.mean(ssims)):.3f}",
                    ha="center", va="center", fontsize=11, family="monospace",
                    transform=axes[1, 0].transAxes)
    for i in range(T):
        axes[1, i + 1].imshow(pred_uint8[i])
        axes[1, i + 1].set_title(f"pred f{i}\nPSNR {psnrs[i]:.2f}  SSIM {ssims[i]:.3f}",
                                  fontsize=8)
        axes[1, i + 1].set_xticks([]); axes[1, i + 1].set_yticks([])
    fig.suptitle(f"Wan I2V LoRA — step {step}, seed {seed}", y=1.00, fontsize=10)
    plt.tight_layout()
    fig.canvas.draw()
    # rasterise to RGB numpy
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb  = rgba[..., :3].copy()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return rgb


def compute_metrics(gen_frames_uint8: np.ndarray, gt_frames_uint8: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity as ssim_fn
    T = gen_frames_uint8.shape[0]
    psnrs, ssims = [], []
    for i in range(T):
        g = gen_frames_uint8[i].astype(np.float32)
        t = gt_frames_uint8[i].astype(np.float32)
        mse = ((g - t) ** 2).mean()
        psnrs.append(99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse))
        ssims.append(float(ssim_fn(g, t, channel_axis=-1, data_range=255.0)))
    return {"psnr_per_frame": psnrs, "ssim_per_frame": ssims,
            "psnr_mean": float(np.mean(psnrs)), "ssim_mean": float(np.mean(ssims))}


@torch.no_grad()
def render_validation_i2v(transformer, vae, scheduler, text_embed,
                          image_encoder, feat_processor,
                          ref_image_b3hw_minus1to1: torch.Tensor,
                          ref_uint8_hwc: np.ndarray,
                          latents_mean, latents_std_inv,
                          num_inference_steps: int, num_frames: int,
                          resolution: int, device, dtype,
                          noise_seed: int = 0,
                          ref_mode: str = "soft_wan_ref") -> np.ndarray:
    """Run a short I2V denoising loop using the same conditioning as training.
    Returns (T, H, W, 3) uint8 generated frames."""
    transformer.eval()
    try:
        # 1. CLIP-encode reference (B=1).
        image_embeds = clip_encode_image(image_encoder, feat_processor,
                                          ref_uint8_hwc[None], device, dtype)   # (1, 1, D_img)

        # 2. Build I2V conditioning (20 ch). Must match the training-time ref_mode
        # or the model sees an off-distribution conditioning layout. In anchored mode
        # the model operates on T+1 pixel frames (frame 0 = anchor, frames 1..T = bracket).
        cond_pixel_frames = num_frames + 1 if ref_mode == "wan_i2v_anchored" else num_frames
        condition = build_i2v_condition(ref_image_b3hw_minus1to1, cond_pixel_frames,
                                         vae, latents_mean, latents_std_inv,
                                         ref_mode=ref_mode)                      # (1, 20, T_lat, H', W')

        # 3. Init noise (1, 16, T_lat, H', W') with deterministic seed.
        T_lat = condition.shape[2]
        latH, latW = condition.shape[3], condition.shape[4]
        gen = torch.Generator(device="cpu").manual_seed(noise_seed)
        latents = torch.randn(1, vae.config.z_dim, T_lat, latH, latW,
                              generator=gen).to(device=device, dtype=dtype)

        # 4. Denoising loop.
        scheduler.set_timesteps(num_inference_steps, device=device)
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents, condition], dim=1)         # (1, 36, T_lat, H', W')
            model_out = transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1),
                encoder_hidden_states=text_embed[:1],
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )[0]
            latents = scheduler.step(model_out, t, latents, return_dict=False)[0]

        # 5. Decode.
        latents = wan_vae_denormalize(latents.float(), latents_mean, latents_std_inv).to(vae.dtype)
        video = vae.decode(latents, return_dict=False)[0]                       # (1, 3, T_pix, H, W)
        frames = video[0].clamp(-1, 1).permute(1, 2, 3, 0)                      # (T_pix, H, W, 3)
        frames_u8 = (((frames + 1) * 127.5).round().clamp(0, 255)).byte().cpu().numpy()
        if ref_mode == "wan_i2v_anchored":
            # Drop frame 0 — that slot is the anchor (should reconstruct the ref).
            # The "predicted bracket" lives at frames 1..num_frames, which is what
            # we compare against val_gt_uint8.
            frames_u8 = frames_u8[1:1 + num_frames]
        return frames_u8
    finally:
        transformer.train()


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

    weight_dtype = {"no": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]

    if accelerator.is_main_process:
        logger.info(f"Wan I2V path : {args.pretrained_wan_path}")
        logger.info(f"Data dir     : {args.data_dir}")
        logger.info(f"Output dir   : {args.output_dir}")
        logger.info(f"LoRA rank    : {args.lora_rank}  Sigma sampling: logit-normal "
                    f"(mean={args.logit_normal_mean}, std={args.logit_normal_std})")

    # ─── Models ───────────────────────────────────────────────────────────
    logger.info("Loading Wan I2V components …")
    tokenizer       = T5TokenizerFast.from_pretrained(args.pretrained_wan_path, subfolder="tokenizer")
    text_encoder    = UMT5EncoderModel.from_pretrained(args.pretrained_wan_path, subfolder="text_encoder",
                                                       torch_dtype=weight_dtype)
    vae             = AutoencoderKLWan.from_pretrained(args.pretrained_wan_path, subfolder="vae",
                                                       torch_dtype=weight_dtype)
    scheduler       = UniPCMultistepScheduler.from_pretrained(args.pretrained_wan_path, subfolder="scheduler")
    transformer     = WanTransformer3DModel.from_pretrained(args.pretrained_wan_path, subfolder="transformer",
                                                            torch_dtype=weight_dtype, low_cpu_mem_usage=True)
    image_encoder   = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_wan_path,
                                                                    subfolder="image_encoder",
                                                                    torch_dtype=weight_dtype)
    feat_processor  = CLIPImageProcessor.from_pretrained(args.pretrained_wan_path, subfolder="image_processor")

    for m in (vae, text_encoder, image_encoder, transformer):
        m.requires_grad_(False)
    vae.eval(); text_encoder.eval(); image_encoder.eval()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    latents_mean    = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    vae_temporal_factor = getattr(vae.config, "temperal_downsample_factor", None) or 4

    # ─── LoRA ─────────────────────────────────────────────────────────────
    # Targets: attention Q/K/V/out plus the FFN's two linears. The FFN
    # provides the per-token feature transformation capacity that attn-only
    # LoRA lacks — important for tasks where the model needs to *reshape*
    # features per frame (e.g., per-EV bracket prediction), not just route
    # information between tokens.
    #
    # PEFT matches target_modules by suffix. Wan's WanTransformerBlock uses
    # `ffn.net.0.proj` (GEGLU gate+up) and `ffn.net.2` (output proj) for the
    # FFN; if the actual module names differ in your diffusers version, peft
    # will silently skip them. The "FFN LoRA modules targeted" log line below
    # is the sanity check — if the trainable param count doesn't roughly
    # double over attn-only (was ~210 M, should now be ~450-500 M), the FFN
    # names didn't match and you need to inspect `print(transformer)`.
    target_modules_list = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if accelerator.is_main_process:
        logger.info(f"LoRA target_modules: {target_modules_list}")
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules_list,
        bias="none", init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    for p in trainable_params:
        p.data = p.data.to(torch.float32)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in transformer.parameters())
    # Count how many FFN modules got LoRA adapters attached (sanity check).
    n_ffn_adapters = sum(1 for n, _ in transformer.named_modules()
                          if ("ffn" in n) and n.endswith(("lora_A.default", "lora_B.default")))
    if accelerator.is_main_process:
        logger.info(f"Transformer params: {n_total/1e9:.2f} B total, "
                    f"{n_trainable/1e6:.2f} M trainable LoRA ({100*n_trainable/n_total:.3f}%)")
        logger.info(f"FFN LoRA modules targeted: {n_ffn_adapters} "
                    f"(expect > 0 if FFN module names matched; else check `print(transformer)`)")

    transformer.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    image_encoder.to(accelerator.device)
    latents_mean    = latents_mean.to(accelerator.device)
    latents_std_inv = latents_std_inv.to(accelerator.device)

    # Cache empty text embedding once, free text encoder.
    empty_text_embed = encode_text_empty(tokenizer, text_encoder, accelerator.device, weight_dtype, 512)
    del text_encoder
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        logger.info(f"Cached empty text embedding: {tuple(empty_text_embed.shape)} {empty_text_embed.dtype}; "
                    f"text encoder unloaded.")

    # ─── Data ─────────────────────────────────────────────────────────────
    dataset  = RawHDRPairDataset(base_folder=args.data_dir,
                                 sample_frames=args.num_frames,
                                 use_noisy_samples=args.use_noisy_samples)
    train_dl = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                          num_workers=args.dataloader_num_workers, drop_last=True, pin_memory=True)

    # ── Validation setup ──────────────────────────────────────────────────
    if not hasattr(dataset, "pairs") or len(dataset.pairs) == 0:
        raise RuntimeError("RawHDRPairDataset has no .pairs attribute — can't pick val sample.")
    val_idx = args.validation_sample_index % len(dataset.pairs)
    val_raw_path, val_gt_path = dataset.pairs[val_idx]

    import HDRutils.io as _hdr_io
    _raw = _hdr_io.imread(val_raw_path).astype("float32")
    if _raw.ndim == 2:
        _raw = np.stack([_raw] * 3, axis=-1)
    _raw = np.maximum(_raw[..., :3], 0.0)
    _p99 = max(float(np.percentile(_raw, 99)), 1e-6)
    _raw_norm = np.clip(_raw / _p99, 0.0, 1.0)                                # [0, 1] linear

    val_dir = os.path.join(args.output_dir, "validation"); os.makedirs(val_dir, exist_ok=True)
    from PIL import Image as _PIL
    _PIL.fromarray((_raw_norm * 255).round().clip(0,255).astype("uint8")).save(
        os.path.join(val_dir, "reference_linear.png"))
    _PIL.fromarray(((_raw_norm ** (1/2.2)) * 255).round().clip(0,255).astype("uint8")).save(
        os.path.join(val_dir, "reference_gamma.png"))

    # CLIP must see the SAME thing it sees in training: training feeds the linear raw
    # in [-1, 1] converted to uint8 (see ref_for_clip in the train loop), NOT a
    # gamma-corrected version. Mismatching these creates an out-of-distribution CLIP
    # embedding at inference and the model produces noise (PSNR ~7).
    val_ref_uint8         = (_raw_norm * 255).round().clip(0,255).astype("uint8")              # for CLIP (linear, matches training)
    val_ref_display_uint8 = ((_raw_norm ** (1/2.2)) * 255).round().clip(0,255).astype("uint8")  # gamma-corrected, panels only
    val_ref_neg1to1       = torch.from_numpy(_raw_norm).permute(2,0,1).unsqueeze(0) * 2.0 - 1.0  # (1,3,H,W)

    # GT bracket
    _gt   = read_hdr_image_float32(val_gt_path)
    _gt_t = torch.from_numpy(_gt).permute(2,0,1).contiguous()
    _Y    = (_gt_t[0]*0.2126 + _gt_t[1]*0.7152 + _gt_t[2]*0.0722).clamp(min=1e-6)
    _gamma = 2.2
    _start_ev = float(np.log2((0.85 ** _gamma) / _Y.max().item()))
    _end_ev   = float(np.log2((0.85 ** _gamma) / _Y.median().item()))
    val_ev_values = torch.linspace(_start_ev, _end_ev, args.num_frames).numpy()
    val_gt_ldr   = hdr_to_ldr_batch_np(_gt, val_ev_values, gamma=_gamma)
    val_gt_uint8 = (np.clip(val_gt_ldr, 0, 1) * 255).round().astype("uint8")
    _PIL.fromarray(np.concatenate([val_gt_uint8[i] for i in range(val_gt_uint8.shape[0])], axis=1)).save(
        os.path.join(val_dir, "gt_strip.png"))
    for i in range(val_gt_uint8.shape[0]):
        _PIL.fromarray(val_gt_uint8[i]).save(os.path.join(val_dir, f"gt_f{i:02d}.png"))

    val_metrics_path = os.path.join(args.output_dir, "val_metrics.json")
    val_metrics_history = {"steps":[], "psnr_mean":[], "ssim_mean":[],
                           "psnr_per_frame":[], "ssim_per_frame":[]}

    if accelerator.is_main_process:
        logger.info(f"Val sample idx={val_idx}/{len(dataset.pairs)} → {os.path.basename(val_raw_path)}")
        logger.info(f"  val GT EV ladder: [{_start_ev:+.2f}, ..., {_end_ev:+.2f}] ({args.num_frames} frames)")

    # ─── Optim / lr-sched ─────────────────────────────────────────────────
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                              betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optim,
                             num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                             num_training_steps=args.max_train_steps * accelerator.num_processes)

    transformer, optim, train_dl, lr_sched = accelerator.prepare(
        transformer, optim, train_dl, lr_sched)
    accelerator.init_trackers("wan_i2v_lora", config=vars(args))

    num_train_timesteps = scheduler.config.num_train_timesteps  # typically 1000

    # Resolve flow_shift: if user didn't pass --flow_shift, read it from the loaded
    # scheduler config (3.0 for Wan 2.1 I2V, 5.0 for Wan 2.2 TI2V). Matching the
    # scheduler's flow_shift at training time is what makes the model see the same
    # σ → timestep mapping at training as it sees at inference.
    if args.flow_shift is None:
        args.flow_shift = float(getattr(scheduler.config, "flow_shift", 1.0))
    if accelerator.is_main_process:
        logger.info(f"Sigma sampling: distribution={args.sigma_distribution}, "
                    f"flow_shift={args.flow_shift}")

    # ─── Resume ───────────────────────────────────────────────────────────
    global_step = 0
    resume_path = _resolve_resume_path(args.resume_from_checkpoint, args.output_dir)
    if resume_path is not None:
        from peft import set_peft_model_state_dict
        if accelerator.is_main_process:
            logger.info(f"Resuming from {resume_path}")
        unwrapped = accelerator.unwrap_model(transformer)
        lora_blob = torch.load(os.path.join(resume_path, "lora_weights.pt"),
                                map_location="cpu", weights_only=False)
        set_peft_model_state_dict(unwrapped, lora_blob["lora_state_dict"])
        train_state = torch.load(os.path.join(resume_path, "train_state.pt"),
                                  map_location="cpu", weights_only=False)
        optim.load_state_dict(train_state["optim"])
        lr_sched.load_state_dict(train_state["lr_sched"])
        global_step = int(train_state["global_step"])
        if accelerator.is_main_process:
            logger.info(f"Resumed at global_step={global_step}")
        accelerator.wait_for_everyone()
    elif args.resume_from_checkpoint and accelerator.is_main_process:
        logger.info(f"--resume_from_checkpoint={args.resume_from_checkpoint!r} "
                    f"did not resolve to a checkpoint — starting from step 0")

    # ─── Train loop ───────────────────────────────────────────────────────
    transformer.train()
    if global_step >= args.max_train_steps:
        if accelerator.is_main_process:
            logger.info(f"global_step ({global_step}) >= max_train_steps "
                        f"({args.max_train_steps}); nothing to do.")
    else:
        logger.info("Starting training …")

    while global_step < args.max_train_steps:
        for batch in train_dl:
            with accelerator.accumulate(transformer):
                clean   = batch["pixel_values_clean"].to(accelerator.device, dtype=weight_dtype)
                ref_img = batch["raw_input"].to(accelerator.device, dtype=weight_dtype)
                B, T, C_in, H, W = clean.shape

                # 1. Build target pixel video and VAE-encode.
                bracket_video = clean.permute(0, 2, 1, 3, 4).contiguous()    # (B, 3, T, H, W)
                if args.ref_mode == "wan_i2v_anchored":
                    # Prepend the reference as frame 0 — the target video is now
                    # [ref, bracket_0, ..., bracket_{T-1}] with T_pix = T + 1. The
                    # conditioning channels and the mask will also be built for T+1
                    # frames below, so target frame 0 == conditioning frame 0 == ref
                    # and the loss signal at that slot is trivially "preserve the anchor."
                    video = torch.cat([ref_img.unsqueeze(2), bracket_video], dim=2)  # (B, 3, T+1, H, W)
                    cond_pixel_frames = args.num_frames + 1
                else:
                    video = bracket_video
                    cond_pixel_frames = args.num_frames
                target_latents = encode_video_with_wan_vae(vae, video, latents_mean, latents_std_inv)

                # 2. Build I2V conditioning (raw as the conditioning image).
                condition = build_i2v_condition(ref_img, cond_pixel_frames,
                                                 vae, latents_mean, latents_std_inv,
                                                 vae_scale_factor_temporal=vae_temporal_factor,
                                                 ref_mode=args.ref_mode)

                # 3. CLIP-encode raw for image cross-attention.
                ref_for_clip = ((((ref_img + 1) * 127.5).round()
                                  .clamp(0, 255)).byte().permute(0, 2, 3, 1).cpu().numpy())
                image_embeds = clip_encode_image(image_encoder, feat_processor,
                                                  ref_for_clip, accelerator.device, weight_dtype)

                # 4. Sigma sampling. Match the inference UniPC schedule:
                #    sigma_raw ~ Uniform(0, 1)  (or logit-normal, if --sigma_distribution=logit_normal)
                #    sigma     = flow_shift * sigma_raw / (1 + (flow_shift - 1) * sigma_raw)
                # The shift biases σ toward higher values; without it, training σ ≈ 0.5
                # while inference σ ∈ (≈0.25, ≈0.96) for flow_shift=3 — a regime mismatch
                # that the LoRA has to "see through" to denoise correctly.
                if args.sigma_distribution == "uniform":
                    sigma_raw = torch.rand(B, device=target_latents.device, dtype=torch.float32)
                else:  # "logit_normal"
                    u = torch.randn(B, device=target_latents.device, dtype=torch.float32) \
                        * args.logit_normal_std + args.logit_normal_mean
                    sigma_raw = torch.sigmoid(u)
                sigmas = args.flow_shift * sigma_raw / (1.0 + (args.flow_shift - 1.0) * sigma_raw)
                sigmas_v = sigmas.view(B, 1, 1, 1, 1).to(target_latents.dtype)

                noise = torch.randn_like(target_latents)
                noisy_latents = (1.0 - sigmas_v) * target_latents + sigmas_v * noise
                v_target = noise - target_latents

                # 5. Forward.
                latent_model_input = torch.cat([noisy_latents, condition], dim=1)  # (B, 36, T_lat, H', W')
                # Timestep uses the SHIFTED sigma so the model sees the same t at training
                # as the inference scheduler will produce for the same noise level.
                timestep = (sigmas * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)

                model_out = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=empty_text_embed.expand(B, -1, -1),
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )[0]                                                          # (B, 16, T_lat, H', W')

                if args.ref_mode == "wan_i2v_anchored":
                    # Skip latent slot 0 — that's the anchor (target == conditioning),
                    # supervising it would just pull the model around for no reason.
                    # Loss is on slot 1: only, which holds the bracket frames' latent.
                    loss = F.mse_loss(model_out[:, :, 1:].float(),
                                       v_target[:, :, 1:].float())
                else:
                    loss = F.mse_loss(model_out.float(), v_target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step(); lr_sched.step(); optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    logger.info(f"step {global_step}/{args.max_train_steps}  "
                                f"loss={loss.item():.4f}  lr={lr_sched.get_last_lr()[0]:.2e}")
                accelerator.log({"train/loss": loss.item(),
                                  "train/lr": lr_sched.get_last_lr()[0],
                                  "train/sigma_mean": float(sigmas.mean())},
                                 step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"lora-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(transformer)
                    torch.save({"lora_state_dict": get_peft_model_state_dict(unwrapped),
                                "config": vars(args), "step": global_step},
                               os.path.join(ckpt_dir, "lora_weights.pt"))
                    torch.save({"global_step": global_step,
                                "optim": optim.state_dict(),
                                "lr_sched": lr_sched.state_dict()},
                               os.path.join(ckpt_dir, "train_state.pt"))
                    logger.info(f"Saved LoRA + train state → {ckpt_dir}")

                    if args.keep_last_n_checkpoints > 0:
                        existing = []
                        for name in os.listdir(args.output_dir):
                            m = re.fullmatch(r"lora-(\d+)", name)
                            if m:
                                existing.append((int(m.group(1)),
                                                 os.path.join(args.output_dir, name)))
                        existing.sort()
                        for _, old in existing[:-args.keep_last_n_checkpoints]:
                            shutil.rmtree(old, ignore_errors=True)
                            logger.info(f"Pruned old checkpoint → {old}")

                # ── Validation render ──────────────────────────────────────
                if (args.validation_steps > 0 and global_step % args.validation_steps == 0
                        and accelerator.is_main_process):
                    logger.info(f"Validation at step {global_step} "
                                f"(seeds=[0..{args.num_validation_samples-1}], "
                                f"steps={args.num_validation_inference_steps}) …")
                    unwrapped_t = accelerator.unwrap_model(transformer)
                    step_dir = os.path.join(val_dir, f"step{global_step:06d}"); os.makedirs(step_dir, exist_ok=True)
                    seed_psnrs, seed_ssims = [], []
                    seed_panels = []                                      # combined GT+pred figures
                    pf_psnr_seeds, pf_ssim_seeds = [], []
                    for seed in range(args.num_validation_samples):
                        frames = render_validation_i2v(
                            unwrapped_t, vae, scheduler, empty_text_embed,
                            image_encoder, feat_processor,
                            val_ref_neg1to1.to(accelerator.device, dtype=weight_dtype),
                            val_ref_uint8,
                            latents_mean, latents_std_inv,
                            num_inference_steps=args.num_validation_inference_steps,
                            num_frames=args.num_frames, resolution=args.resolution,
                            device=accelerator.device, dtype=weight_dtype, noise_seed=seed,
                            ref_mode=args.ref_mode,
                        )
                        seed_dir = os.path.join(step_dir, f"seed{seed}"); os.makedirs(seed_dir, exist_ok=True)
                        for i in range(frames.shape[0]):
                            _PIL.fromarray(frames[i]).save(os.path.join(seed_dir, f"f{i:02d}.png"))
                        m = compute_metrics(frames, val_gt_uint8)
                        seed_psnrs.append(m["psnr_mean"]); seed_ssims.append(m["ssim_mean"])
                        pf_psnr_seeds.append(m["psnr_per_frame"]); pf_ssim_seeds.append(m["ssim_per_frame"])

                        # Combined GT+pred panel with metrics annotated (like VDM_EVFI).
                        panel_rgb = make_gt_pred_panel(
                            gt_uint8=val_gt_uint8, pred_uint8=frames, ref_uint8=val_ref_display_uint8,
                            psnrs=m["psnr_per_frame"], ssims=m["ssim_per_frame"],
                            step=global_step, seed=seed,
                            save_path=os.path.join(val_dir,
                                f"panel_step{global_step:06d}_seed{seed}.png"),
                        )
                        seed_panels.append(panel_rgb)

                    mean_psnr = float(np.mean(seed_psnrs)); mean_ssim = float(np.mean(seed_ssims))
                    best_psnr = float(np.max(seed_psnrs)); best_ssim = float(np.max(seed_ssims))
                    pf_psnr_mean = np.mean(pf_psnr_seeds, axis=0).tolist()
                    pf_ssim_mean = np.mean(pf_ssim_seeds, axis=0).tolist()
                    logger.info(f"  PSNR per-seed: {[f'{p:.2f}' for p in seed_psnrs]} → mean {mean_psnr:.2f} best {best_psnr:.2f}")
                    logger.info(f"  SSIM per-seed: {[f'{s:.3f}' for s in seed_ssims]} → mean {mean_ssim:.3f} best {best_ssim:.3f}")
                    logger.info(f"  PSNR per-frame (mean over seeds): {[f'{p:.2f}' for p in pf_psnr_mean]}")
                    val_metrics_history["steps"].append(global_step)
                    val_metrics_history["psnr_mean"].append(mean_psnr)
                    val_metrics_history["ssim_mean"].append(mean_ssim)
                    val_metrics_history["psnr_per_frame"].append(pf_psnr_mean)
                    val_metrics_history["ssim_per_frame"].append(pf_ssim_mean)
                    with open(val_metrics_path, "w") as _f:
                        json.dump(val_metrics_history, _f, indent=2)
                    val_payload = {"val/psnr_mean": mean_psnr, "val/psnr_best": best_psnr,
                                    "val/ssim_mean": mean_ssim, "val/ssim_best": best_ssim}
                    for i in range(len(pf_psnr_mean)):
                        val_payload[f"val/f{i}/psnr"] = pf_psnr_mean[i]
                        val_payload[f"val/f{i}/ssim"] = pf_ssim_mean[i]
                    accelerator.log(val_payload, step=global_step)
                    if args.report_to in ("wandb", "all"):
                        try:
                            import wandb as _wandb
                            if _wandb.run is not None:
                                img_payload = {}
                                for seed_i, panel in enumerate(seed_panels):
                                    img_payload[f"val/panel_seed{seed_i}"] = _wandb.Image(
                                        panel,
                                        caption=f"step {global_step} seed {seed_i} | "
                                                f"PSNR {seed_psnrs[seed_i]:.2f}  SSIM {seed_ssims[seed_i]:.3f}")
                                _wandb.log(img_payload, step=global_step)
                        except Exception as e:
                            logger.warning(f"wandb val panel image log skipped: {e}")

                if global_step >= args.max_train_steps:
                    break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        final_dir = os.path.join(args.output_dir, "lora-final"); os.makedirs(final_dir, exist_ok=True)
        torch.save({"lora_state_dict": get_peft_model_state_dict(unwrapped),
                    "config": vars(args), "step": global_step},
                   os.path.join(final_dir, "lora_weights.pt"))
        logger.info(f"Saved final LoRA → {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
