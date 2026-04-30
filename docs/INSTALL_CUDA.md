# Installation — NVIDIA CUDA

The published checkpoints were trained on AMD ROCm. The model and the
inference / training scripts use PyTorch's built-in SDPA attention only,
with **no FlashAttention or BitsAndBytes dependency**, so they run on
CUDA without code changes.

## 1. Create a clean environment

```bash
conda create -n vid2hdrimg python=3.10 -y
conda activate vid2hdrimg
```

## 2. Install PyTorch (CUDA wheel)

Pick the index URL matching your CUDA toolkit version. A recent build
(CUDA 12.1) works:

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Other CUDA targets are listed at <https://pytorch.org/get-started/locally/>.
Newer PyTorch versions (e.g. 2.4+) should also work, but were not the
exact configuration we trained with.

Verify GPU access:

```bash
python -c "import torch; print('torch', torch.__version__); \
                       print('cuda_avail', torch.cuda.is_available()); \
                       print('cuda_ver', torch.version.cuda); \
                       print('devices', torch.cuda.device_count())"
```

## 3. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

## Notes

- The pinned `diffusers==0.27.2` is the version the model was trained
  against. Newer diffusers releases may work but are not guaranteed.
- If you have less than ~24 GB GPU memory, lower `--decode_chunk_size`
  during inference (default 8) — try 2 or 4.
- For training on multi-GPU, follow the `accelerate launch` examples in
  [TRAINING.md](TRAINING.md).
