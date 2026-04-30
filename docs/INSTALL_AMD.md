# Installation — AMD ROCm

This is the configuration the released checkpoints were trained on:
**AMD Instinct MI210 (gfx90a) / ROCm 6.0 / Python 3.9 / PyTorch 2.3.1**.

## 1. Create a clean environment

```bash
conda create -n vid2hdrimg python=3.9 -y
conda activate vid2hdrimg
```

(or use `python -m venv .venv && source .venv/bin/activate` — your call.)

## 2. Install PyTorch (ROCm wheel)

```bash
pip install torch==2.3.1+rocm6.0 torchvision==0.18.1+rocm6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.0
```

Verify ROCm is detected:

```bash
python -c "import torch; print('torch', torch.__version__); \
                       print('rocm', torch.version.hip); \
                       print('cuda_avail', torch.cuda.is_available()); \
                       print('devices', torch.cuda.device_count())"
```

You should see something like:

```
torch 2.3.1+rocm6.0
rocm 6.0.32830-d62f6a171
cuda_avail True
devices 1
```

## 3. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

## Notes

- This repo deliberately does **not** depend on FlashAttention or BitsAndBytes;
  PyTorch's built-in scaled-dot-product attention works on ROCm.
- If you need MIOpen kernel caching, redirect to a writable per-user dir before
  running training/inference (avoids issues on shared NFS):

    ```bash
    export MIOPEN_USER_DB_PATH=/tmp/miopen_$USER
    export MIOPEN_DISABLE_CACHE=1
    mkdir -p "$MIOPEN_USER_DB_PATH"
    ```

- If `torch.compile` complains, set `--compile_backend aot_eager` or skip
  `--compile_unet` entirely.
