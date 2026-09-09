# GH200 inference on DeltaAI

The persistent inference installation is under `runtime/`. This directory uses
the allocation's `ddn_hdd` Lustre pool because the default `ddn_ssd` pool exceeded
its soft quota. The existing training files are retained.

Request an interactive GPU from the repository directory:

```bash
srun -A bhcc-dtai-gh -p ghx4-interactive -N 1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-node=1 --mem=96G -t 02:00:00 --pty bash
bash scripts/run_inference_gh200.sh
```

The default input is `assets/example_sihdr_input.exr`; the output is
`runtime/outputs/predicted.exr`. The script uses 512×512 resolution, five frames,
50 diffusion steps, and seed 42. Pass an input and output to run another image:

```bash
bash scripts/run_inference_gh200.sh /path/to/input.jpg runtime/outputs/custom.exr
```

Additional inference flags can follow the two paths. The script requires CUDA
and uses local weights with network downloads disabled.

To activate the Conda environment manually:

```bash
source /work/nvme/bhcc/ctalegaonkar/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/bhcc/ctalegaonkar/Vid2HDRImg/runtime/conda
```

This Conda environment has its own pinned inference packages and reuses the
existing `miniconda3/envs/vid2hdr/lib/python3.11/site-packages` for PyTorch 2.5.1
with CUDA 12.4 and other unchanged dependencies via a `.pth` file. Keep that
existing environment in place. Do not upgrade packages through this shared
path without isolating them first.

The inference pins are Diffusers 0.30.3, Transformers 4.44.2,
Hugging Face Hub 0.25.2, Accelerate 1.0.1, and PEFT 0.13.2.
The repository's Diffusers 0.27.2 pin does not provide `diffusers.video_processor`,
which the pipeline imports.

`--unet_path` must point to `runtime/weights`, the directory containing `unet/`,
because `inference.py` appends that subdirectory internally.
