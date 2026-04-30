# Dataset preparation

The training pipeline expects a single directory of HDR EXR files,
optionally augmented with paired raw / GT-HDR captures.

## Sources used in the paper

The released model was trained on a curated mix of two public HDR datasets:

1. **Fairchild HDR Photographic Survey** —
   <http://markfairchild.org/HDRPS/HDRthumbs.html>.
   We use the .hdr files; convert them to OpenEXR via your favourite tool
   (e.g. `hdr2exr` or HDRutils) and keep only the linear-light pixel data.

2. **RawHDR** (Zou et al., ICCV 2023) —
   <https://github.com/jackzou233/RawHDR>.
   We use the GT HDR images and (optionally) the corresponding raw
   captures for the `RawHDRPairDataset` mode.

## Expected layouts

### `RepeatedHDRVideoDataset` (synthetic-bracket, default)

A flat directory of `.exr` (or `.hdr`) files:

```
hdr_dataset/
├── scene_0000.exr
├── scene_0001.exr
├── ...
```

Pass this path as `--train_data_path` to `train_video_model.py` /
`train_fusion_net.py`. The dataset class itself handles random crops,
exposure-ladder synthesis, and noise simulation.

### `RawHDRPairDataset` (real raw + GT-HDR, optional)

```
raw_hdr_dataset/
├── raw/
│   ├── scene_0000.exr
│   ├── scene_0001.exr
│   └── ...
└── gt_hdr/
    ├── scene_0000.exr
    ├── scene_0001.exr
    └── ...
```

Filenames in `raw/` and `gt_hdr/` must match. Activate the dataset
during stage-1 training with `--raw_pair_data_path /path/to/raw_hdr_dataset
--use_raw_input`.

## Cropping

The released model trains at **512 × 512** centre crops (or random crops
with `--random_crop`). HDR files smaller than 512 px on any side are
filtered out by the dataset class.

## Validation image (Stage 1)

`train_video_model.py` periodically renders an exposure bracket from a
held-out HDR file; pass the path with `--valid_hdr_path
/path/to/validation.hdr`. Make sure to also pass the same filename to
`--exclude_files <basename>` so it is excluded from the training set.

## Test set

For evaluation we use the **SI-HDR** test set (Hanji et al., 2022;
the same subset used by X2HDR for fairness). It is not bundled with this
release — get it from the original authors / your favourite HDR-eval
benchmark mirror.
