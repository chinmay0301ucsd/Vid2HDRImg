"""FusionUNet — pixel-space HDR exposure fusion network.

Architecture must match train_svd_hdr_fus_only.py exactly.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Small UNet: (B, T, 3, H, W) linearized LDR → (B, 3, H, W) relative linear HDR in [0, 1].
    Predicts per-pixel softmax weights over T frames; output = weighted average of linear frames.
    """

    def __init__(self, in_channels: int = 3, num_frames: int = 5, base_ch: int = 32):
        super().__init__()
        self.num_frames  = num_frames
        self.in_channels = in_channels
        inp = in_channels * num_frames
        b   = base_ch
        self.enc1 = _DoubleConv(inp,   b)
        self.enc2 = _DoubleConv(b,     b * 2)
        self.pool = nn.MaxPool2d(2)
        self.bot  = _DoubleConv(b * 2, b * 4)
        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(b * 4, b * 2)
        self.up1  = nn.ConvTranspose2d(b * 2, b,     kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, num_frames, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor):
        B, T, C, H, W = pixel_values.shape
        x = pixel_values.reshape(B, T * C, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bt = self.bot(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(bt), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits  = self.head(d1)
        weights = F.softmax(logits, dim=1).unsqueeze(2)   # (B, T, 1, H, W)
        fused   = (pixel_values * weights).sum(dim=1)     # (B, 3, H, W)
        return fused, weights


def load_fusion_net(path: str, num_frames: int, device: torch.device) -> FusionUNet:
    """Load a FusionUNet from a fusion_net.pt state-dict file."""
    net = FusionUNet(num_frames=num_frames)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net.to(device)


def run_fusion_net(
    fusion_net: FusionUNet,
    video_frames: list,
    ref_hdr: np.ndarray,
    device: torch.device,
) -> tuple:
    """
    Run FusionUNet on SVD-generated LDR frames → absolute-scale linear HDR.

    Pipeline:
      1. PIL frames → (1, T, 3, H, W) float32 in [0, 1]
      2. Linearize: ** 2.2  (undo gamma, matching training)
      3. FusionUNet → relative linear HDR + per-pixel softmax weights
      4. Rescale by Ymax of ref_hdr

    Returns:
        hdr_np    : (H, W, 3) float32 linear HDR.
        weights_np: (T, H, W) float32 per-pixel softmax weights.
    """
    frames_np = np.stack(
        [np.array(f, dtype=np.float32) / 255.0 for f in video_frames], axis=0
    )
    pixel_values = (
        torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    )
    pixel_values_lin = pixel_values.clamp(0, 1) ** 2.2

    with torch.no_grad():
        hdr_pred_01, weights = fusion_net(pixel_values_lin)

    hdr_np     = hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy()
    weights_np = weights[0, :, 0].float().cpu().numpy()

    Y_ref    = 0.2126 * ref_hdr[..., 0] + 0.7152 * ref_hdr[..., 1] + 0.0722 * ref_hdr[..., 2]
    luma_max = float(np.maximum(Y_ref, 1e-6).max())
    return np.maximum(hdr_np * luma_max, 0.0).astype(np.float32), weights_np


def run_fusion_net_float(
    fusion_net: FusionUNet,
    ldr_batch: np.ndarray,
    ref_hdr: np.ndarray,
    device: torch.device,
) -> tuple:
    """Run FusionUNet on float32 LDR frames — no uint8 quantization.

    Args:
        ldr_batch : (T, H, W, 3) float32 in [0,1], gamma-compressed
        ref_hdr   : (H, W, 3) float32 linear HDR (for Ymax rescaling)
    """
    pixel_values_lin = (
        torch.from_numpy(ldr_batch).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        .clamp(0, 1) ** 2.2
    )   # (1, T, 3, H, W) linear

    with torch.no_grad():
        hdr_pred_01, weights = fusion_net(pixel_values_lin)

    hdr_np     = hdr_pred_01[0].float().cpu().permute(1, 2, 0).numpy()
    weights_np = weights[0, :, 0].float().cpu().numpy()

    Y_ref    = 0.2126 * ref_hdr[..., 0] + 0.7152 * ref_hdr[..., 1] + 0.0722 * ref_hdr[..., 2]
    luma_max = float(np.maximum(Y_ref, 1e-6).max())
    return np.maximum(hdr_np * luma_max, 0.0).astype(np.float32), weights_np


def save_fusion_weights(weights_np: np.ndarray, out_dir: str, stem: str):
    """Save per-frame softmax weight maps as false-colour PNG images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    T = weights_np.shape[0]
    np.save(os.path.join(out_dir, f"{stem}_weights.npy"), weights_np)

    for t in range(T):
        w = weights_np[t]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        im = ax.imshow(w, cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(f"Frame {t}  (max={w.max():.3f})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{stem}_weight_frame_{t:02d}.png"),
                    bbox_inches="tight")
        plt.close(fig)
