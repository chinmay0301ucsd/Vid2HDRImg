from torch.utils.data import Dataset
import os, random
import numpy as np
import torch
import cv2
import pyexr
cv2.setNumThreads(1)  # prevent thread explosion in multi-worker DataLoader

CROP_x = 512
CROP_y = 512
UPSAMPLE_SCALE = [1.0]


def list_hdr_frames(folder_path, exts=(".exr", ".hdr")):
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])


def get_random_crop_idx(width, height, crop_size_x, crop_size_y):
    if width < crop_size_x or height < crop_size_y:
        raise ValueError(f"Image too small for crop: ({width}x{height}) < ({crop_size_x}x{crop_size_y})")
    return random.randint(0, width - crop_size_x), random.randint(0, height - crop_size_y)


def get_center_crop_idx(width, height, crop_size_x, crop_size_y):
    if width < crop_size_x or height < crop_size_y:
        raise ValueError(f"Image too small for crop: ({width}x{height}) < ({crop_size_x}x{crop_size_y})")
    return (width - crop_size_x) // 2, (height - crop_size_y) // 2


def apply_crop_np(img, start_x, start_y, crop_size_x, crop_size_y):
    return img[start_y:start_y + crop_size_y, start_x:start_x + crop_size_x, :]


def read_hdr_image_float32(path):
    """Read an HDR image file (.exr or .hdr) and return (H, W, 3) float32 in linear RGB."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        img = pyexr.read(path).astype(np.float32)  # (H, W, C) float32 in linear RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.broadcast_to(img, (*img.shape[:2], 3)).copy()
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    else:
        # Radiance .hdr and other formats — OpenCV handles RGBE natively
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read HDR file: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.broadcast_to(img, (*img.shape[:2], 3)).copy()
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_np(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def hdr_to_training_range(img_rgb, eps=1e-6):
    img = np.clip(img_rgb, 0.0, None)
    white = float(np.percentile(img, 99.0))
    white = max(white, eps)
    scaled = img / white
    compressed = np.log1p(scaled)
    compressed = np.clip(compressed, 0.0, 1.0)
    return compressed * 2.0 - 1.0


def hdr_to_ldr_batch_np(
    hdr: np.ndarray,        # (H, W, 3) float32, linear HDR
    ev_values: np.ndarray,  # (T,)  EV stops
    gamma: float = 2.2,
) -> np.ndarray:
    """
    Vectorized HDR -> LDR for T exposure values simultaneously.

    Applies:  ldr = clip((clip(hdr, 0) * 2^ev + 1e-8)^(1/gamma), 0, 1)

    Returns: (T, H, W, 3) float32 in [0, 1]
    """
    ev = ev_values[:, None, None, None].astype(np.float32)   # (T,1,1,1)
    img = hdr[None].astype(np.float32)                        # (1,H,W,3)
    scaled = np.clip(img, 0.0, None) * (2.0 ** ev)           # (T,H,W,3)
    ldr = np.clip(np.power(scaled + 1e-8, 1.0 / gamma), 0.0, 1.0)
    return ldr                                                 # (T,H,W,3)


def add_shot_noise(signal_e: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise approximated as Gaussian(μ, √μ)."""
    std = np.sqrt(np.maximum(signal_e, 0.0))
    noise = rng.normal(0.0, 1.0, size=signal_e.shape).astype(np.float32) * std
    return signal_e + noise


def add_read_noise(signal_e: np.ndarray, sigma_read: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian read noise (signal-independent), std in electrons."""
    noise = rng.normal(0.0, sigma_read, size=signal_e.shape).astype(np.float32)
    return signal_e + noise


def add_dark_noise(signal_e: np.ndarray, dark_current_e_per_s: float,
                   t_s: float, rng: np.random.Generator) -> np.ndarray:
    """Dark current noise: Poisson approximated as Gaussian(μ_dark, √μ_dark)."""
    dark_e = dark_current_e_per_s * t_s
    noise = rng.normal(0.0, np.sqrt(max(dark_e, 0.0)), size=signal_e.shape).astype(np.float32)
    return signal_e + noise


def simulate_noisy_ldr(
    hdr: np.ndarray,
    ev: float,
    ref_irradiance: float = None,
    full_well_e: float = 8000.0,
    sigma_read: float = 15.0,
    dark_current_e_per_s: float = 10.0,
    t_ref_s: float = 1.0 / 60.0,
    gamma: float = 2.2,
    use_shot: bool = True,
    use_read: bool = True,
    use_dark: bool = True,
    seed: int = 0,
) -> tuple:
    """
    Simulate a noisy LDR capture of an HDR scene at a given exposure value.

    Returns
    -------
    ldr_noisy : HxWx3 float32 in [0, 1]
    ldr_clean : HxWx3 float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)

    if ref_irradiance is None:
        ref_irradiance = float(hdr.max())
    ref_irradiance = max(ref_irradiance, 1e-8)

    linear_exposure = 2.0 ** ev
    t_exposure_s = t_ref_s * linear_exposure

    signal_e = (hdr / ref_irradiance) * full_well_e * linear_exposure
    sat_e = full_well_e

    noisy_e = signal_e.copy()
    if use_shot:
        noisy_e = add_shot_noise(noisy_e, rng)
    if use_read:
        noisy_e = add_read_noise(noisy_e, sigma_read, rng)
    # if use_dark:
    #     noisy_e = add_dark_noise(noisy_e, dark_current_e_per_s, t_exposure_s, rng)

    def electrons_to_ldr(e):
        linear = np.clip(e, 0.0, sat_e) / sat_e
        return np.power(np.maximum(linear, 0.0), 1.0 / gamma).astype(np.float32)

    return electrons_to_ldr(noisy_e), electrons_to_ldr(signal_e)


class RepeatedHDRVideoDataset(Dataset):
    """
    Returns a "video" sample by tone-mapping a single HDR frame at T different
    EV stops (linspace from start_ev to end_ev with small noise), using a
    gamma=2.2 power-law curve with clipping.

    Output:
      pixel_values: (T, 3, crop_y, crop_x) in [-1,1]
      exposure_values: (T,) sorted EV stops
    """
    def __init__(
        self,
        base_folder,
        sample_frames=14,
        crop_x=CROP_x,
        crop_y=CROP_y,
        frames_subdir=None,
        exts=(".exr", ".hdr"),
        random_crop=True,
        use_noisy_samples=False,
        exclude_files=None,
        quantize: bool = True,
        peak_lum: float = 0.0,
        enable_ev_control: bool = False,
        r_max: float = 10.0,
        c_zero_prob: float = 0.0,
    ):
        # base_folder may be a single path string or a list of path strings
        if isinstance(base_folder, (list, tuple)):
            folders = list(base_folder)
        else:
            folders = [base_folder]
        self.base_folder = folders[0]   # kept for backward compat (validation path etc.)
        self.sample_frames = sample_frames
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.frames_subdir = frames_subdir
        self.exts = exts
        self.random_crop = random_crop
        self.use_noisy_samples = use_noisy_samples
        self.quantize = quantize
        self.peak_lum = peak_lum
        self.enable_ev_control = enable_ev_control
        self.r_max = r_max
        self.c_zero_prob = c_zero_prob
        _exclude = set(os.path.basename(f) for f in (exclude_files or []))
        # Build a flat list of absolute paths from all folders
        self.hdr_file_paths = []
        for folder in folders:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(exts) and f not in _exclude:
                    self.hdr_file_paths.append(os.path.join(folder, f))
        # Keep hdr_files for __len__ (list of basenames from primary folder, extended)
        self.hdr_files = self.hdr_file_paths   # full paths now

    def __len__(self):
        return max(len(self.hdr_files), 1)

    def __getitem__(self, idx):
        _load = self.load_noisy_sample if self.use_noisy_samples else self._load_sample
        for attempt in range(10):
            try:
                return _load()
            except Exception as e:
                print(f"Warning: skipping sample (attempt {attempt+1}): {e}")
        return _load()

    def _load_sample(self):
        # --- pick a random HDR file from the pre-scanned list ---
        if not self.hdr_file_paths:
            raise ValueError(f"No HDR files found in: {self.base_folder}")
        frame_path = random.choice(self.hdr_file_paths)

        # --- load HDR ---
        scale = float(UPSAMPLE_SCALE[0])
        img = read_hdr_image_float32(frame_path)        # (H,W,3) float32 linear

        # --- optional resize ---
        if scale != 1.0:
            h0, w0 = img.shape[:2]
            img = resize_np(img, int(w0 * scale), int(h0 * scale))

        # --- crop (same indices for ALL EV frames) ---
        h, w = img.shape[:2]
        if self.random_crop:
            crop_x, crop_y = get_random_crop_idx(w, h, self.crop_x, self.crop_y)
        else:
            crop_x, crop_y = get_center_crop_idx(w, h, self.crop_x, self.crop_y)
        img = apply_crop_np(img, crop_x, crop_y, self.crop_x, self.crop_y)  # (crop_y, crop_x, 3)
        img = np.ascontiguousarray(img)

        # ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] != 3:
            img = img[:, :, :3]

        # Optional: scale HDR to fixed peak luminance before EV-ladder generation.
        # This makes start_ev nearly constant across images, which stabilises training.
        if self.peak_lum > 0:
            img_peak = float(img.max())
            if img_peak > 1e-8:
                img = img * (self.peak_lum / img_peak)

        hdr_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3, crop_y, crop_x)
        gamma = 2.2 #torch.rand(1).item() * 1.5 + 1.5  # Random gamma in [2.0, 2.5]

        Y = (hdr_tensor[0] * 0.2126 + hdr_tensor[1] * 0.7152 + hdr_tensor[2] * 0.0722).clamp(min=1e-6)
        Ymax = Y.max().item()
        Ymedian = Y.median().item()
        val_start = 0.85 # np.random.rand() * 0.55 + 0.3
        start_ev = np.log2((val_start**gamma) / Ymax)
        end_ev = np.log2((0.85**gamma) / Ymedian)

        native_ev_range = float(end_ev - start_ev)

        if self.enable_ev_control:
            if self.c_zero_prob > 0.0 and random.random() < self.c_zero_prob:
                c = 0.0
            else:
                c = random.random()
            R_c = native_ev_range + c * max(0.0, self.r_max - native_ev_range)
            start_ev = end_ev - R_c   # stretch dark end only; bright anchor fixed
        else:
            c = 0.0

        # --- EV values: linspace(start_ev, end_ev) + small noise, then sort ---
        # start_ev, end_ev = -3, 5
        ev_values = torch.linspace(start_ev, end_ev, self.sample_frames)
        ev_values = ev_values + torch.randn_like(ev_values) * 0.01
        ev_values, _ = torch.sort(ev_values)

        # --- Generate ALL LDR frames at once (fully vectorized, no Python loop) ---
        # hdr_to_ldr_batch_np: (H,W,3) + (T,) -> (T,H,W,3) in [0,1]

        ldr_batch = hdr_to_ldr_batch_np(img, ev_values.numpy(), gamma=gamma)
        if self.quantize:
            ldr_batch = np.round(ldr_batch * 255.0) / 255.0           # 8-bit quantization
        ldr_norm = ldr_batch * 2.0 - 1.0                              # (T,H,W,3) in [-1,1]
        pixel_values = torch.from_numpy(ldr_norm).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)

        return {
            "pixel_values": pixel_values,      # (T, 3, crop_y, crop_x) in [-1,1]
            "hdr_image": hdr_tensor,             # (3, crop_y, crop_x) float32 linear HDR
            "init_frame": frame_path,
            "exposure_values": ev_values,      # (T,) EV stops, sorted
            "gamma": gamma,                    # scalar gamma used for tone-mapping
            "ev_control": torch.tensor(c, dtype=torch.float32),
            "native_ev_range": torch.tensor(native_ev_range, dtype=torch.float32),
        }
    
    def load_noisy_sample(self):
        # --- pick a random HDR file from the pre-scanned list ---
        if not self.hdr_file_paths:
            raise ValueError(f"No HDR files found in: {self.base_folder}")
        frame_path = random.choice(self.hdr_file_paths)

        # --- load HDR ---
        scale = float(UPSAMPLE_SCALE[0])
        img = read_hdr_image_float32(frame_path)        # (H,W,3) float32 linear

        # --- optional resize ---
        if scale != 1.0:
            h0, w0 = img.shape[:2]
            img = resize_np(img, int(w0 * scale), int(h0 * scale))

        # --- crop (same indices for ALL EV frames) ---
        h, w = img.shape[:2]
        if self.random_crop:
            crop_x, crop_y = get_random_crop_idx(w, h, self.crop_x, self.crop_y)
        else:
            crop_x, crop_y = get_center_crop_idx(w, h, self.crop_x, self.crop_y)
        img = apply_crop_np(img, crop_x, crop_y, self.crop_x, self.crop_y)  # (crop_y, crop_x, 3)
        img = np.ascontiguousarray(img)
        
        # ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] != 3:
            img = img[:, :, :3]

        # img = img * 32 / img.max()  # simple global scaling to normalization brightness (before tone-mapping)
        hdr_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3, crop_y, crop_x)
        gamma = 2.2 #torch.rand(1).item() * 1.5 + 1.5  # Random gamma in [2.0, 2.5]
        
        Y = (hdr_tensor[0] * 0.2126 + hdr_tensor[1] * 0.7152 + hdr_tensor[2] * 0.0722).clamp(min=1e-6)
        Ymax = Y.max().item()
        Ymedian = Y.median().item()
        if self.enable_ev_control:
            # Under EV control, pin val_start so c is the sole bracket-width knob.
            val_start = 0.85
        else:
            val_start = np.random.rand() * 0.55 + 0.3
        start_ev = np.log2((val_start**gamma) / Ymax)
        end_ev = np.log2((0.85**gamma) / Ymedian)

        native_ev_range = float(end_ev - start_ev)

        if self.enable_ev_control:
            if self.c_zero_prob > 0.0 and random.random() < self.c_zero_prob:
                c = 0.0
            else:
                c = random.random()
            R_c = native_ev_range + c * max(0.0, self.r_max - native_ev_range)
            start_ev = end_ev - R_c   # stretch dark end only; bright anchor fixed
        else:
            c = 0.0

        # --- EV values: linspace(start_ev, end_ev) + small noise, then sort ---
        # start_ev, end_ev = -3, 5
        ev_values = torch.linspace(start_ev, end_ev, self.sample_frames)
        ev_values = ev_values + torch.randn_like(ev_values) * 0.01
        ev_values, _ = torch.sort(ev_values)

        # --- Generate noisy and clean LDR frames via simulate_noisy_ldr ---
        # ref_irradiance=1.0 keeps signal_e = hdr * full_well_e * 2^ev, which is
        # consistent with hdr_to_ldr_batch_np and the raw-luminance EV formula.
        ref_irradiance = 1.0
        ev_np = ev_values.numpy()
        noisy_frames, clean_frames = [], []
        for i, ev in enumerate(ev_np):
            ldr_noisy, ldr_clean = simulate_noisy_ldr(
                img, float(ev),
                ref_irradiance=ref_irradiance,
                full_well_e=8000.0,
                sigma_read=15.0,
                gamma=gamma,
                use_shot=True,
                use_read=True,
                use_dark=False,
                seed=i,
            )
            noisy_frames.append(ldr_noisy)
            clean_frames.append(ldr_clean)

        ldr_noisy_batch = np.stack(noisy_frames, axis=0)               # (T,H,W,3) in [0,1]
        ldr_clean_batch = np.stack(clean_frames, axis=0)               # (T,H,W,3) in [0,1]

        ldr_noisy_batch = np.round(ldr_noisy_batch * 65535.0) / 65535.0   # 16-bit quantization
        ldr_clean_batch = np.round(ldr_clean_batch * 65535.0) / 65535.0

        pixel_values       = torch.from_numpy(ldr_noisy_batch * 2.0 - 1.0).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
        pixel_values_clean = torch.from_numpy(ldr_clean_batch * 2.0 - 1.0).permute(0, 3, 1, 2).contiguous()

        return {
            "pixel_values": pixel_values,             # (T, 3, crop_y, crop_x) in [-1,1], noisy
            "pixel_values_clean": pixel_values_clean, # (T, 3, crop_y, crop_x) in [-1,1], clean
            "hdr_image": hdr_tensor,                  # (3, crop_y, crop_x) float32 linear HDR
            "init_frame": frame_path,
            "exposure_values": ev_values,             # (T,) EV stops, sorted
            "gamma": gamma,                           # scalar gamma used for tone-mapping
            "ev_control": torch.tensor(c, dtype=torch.float32),
            "native_ev_range": torch.tensor(native_ev_range, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# RawHDRPairDataset
# Loads pre-cropped 512×512 (raw_input, gt_hdr) EXR pairs produced by
# build_raw_hdr_dataset.py.  raw/ and gt_hdr/ folders must have matching stems.
#
# Returns the same dict as RepeatedHDRVideoDataset (noisy mode) plus a
# `raw_input` key containing the real raw conditioning frame.  Existing code
# that does not use `raw_input` is unaffected.
# ─────────────────────────────────────────────────────────────────────────────

class RawHDRPairDataset(Dataset):
    """Paired (raw_input, gt_hdr) dataset of pre-cropped 512×512 EXR images.

    Directory layout expected:
        base_folder/
            raw/        ← linear sRGB EXRs of real raw captures (demosaiced)
            gt_hdr/     ← linear HDR EXRs (ground truth)

    Stems in raw/ and gt_hdr/ must match exactly (built by build_raw_hdr_dataset.py).

    __getitem__ returns:
        pixel_values        (T, 3, H, W) in [-1,1]  — tone-mapped from gt_hdr (noisy)
        pixel_values_clean  (T, 3, H, W) in [-1,1]  — same, clean version
        hdr_image           (3, H, W)  float32 linear HDR (gt_hdr)
        raw_input           (3, H, W)  in [-1,1]  — real raw conditioning frame
        init_frame          str  — path to raw EXR
        exposure_values     (T,)  EV stops sorted ascending
        gamma               float
    """

    def __init__(
        self,
        base_folder,
        sample_frames=5,
        use_noisy_samples=True,
        exclude_files=None,
        enable_ev_control: bool = False,
        r_max: float = 10.0,
        c_zero_prob: float = 0.0,
    ):
        self.base_folder = base_folder
        self.sample_frames = sample_frames
        self.use_noisy_samples = use_noisy_samples
        self.enable_ev_control = enable_ev_control
        self.r_max = r_max
        self.c_zero_prob = c_zero_prob

        raw_dir = os.path.join(base_folder, "raw")
        gt_dir  = os.path.join(base_folder, "gt_hdr")

        _exclude = set(os.path.basename(f) for f in (exclude_files or []))

        # Build list of stems present in BOTH raw/ and gt_hdr/
        raw_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(raw_dir)
            if f.lower().endswith(".exr") and f not in _exclude
        }
        gt_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(gt_dir)
            if f.lower().endswith(".exr") and f not in _exclude
        }
        paired = sorted(raw_stems & gt_stems)
        if not paired:
            raise ValueError(
                f"No matching EXR pairs found in {raw_dir} / {gt_dir}"
            )

        self.pairs = [
            (
                os.path.join(raw_dir, s + ".exr"),
                os.path.join(gt_dir,  s + ".exr"),
            )
            for s in paired
        ]

    def __len__(self):
        return max(len(self.pairs), 1)

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                return self._load_sample()
            except Exception as e:
                print(f"Warning: skipping sample (attempt {attempt+1}): {e}")
        return self._load_sample()

    def _load_sample(self):
        raw_path, gt_path = random.choice(self.pairs)

        # Load both EXRs — already 512×512 crops, no further cropping needed
        raw_img = read_hdr_image_float32(raw_path)   # (H, W, 3) float32 linear
        gt_img  = read_hdr_image_float32(gt_path)    # (H, W, 3) float32 linear

        raw_img = np.ascontiguousarray(raw_img)
        gt_img  = np.ascontiguousarray(gt_img)

        # Normalize raw_input to [-1, 1] via 99th-percentile (same as eval pipeline)
        p99 = float(np.percentile(raw_img, 99))
        p99 = max(p99, 1e-6)
        raw_norm = np.clip(raw_img / p99, 0.0, 1.0)
        raw_tensor = torch.from_numpy(raw_norm * 2.0 - 1.0).permute(2, 0, 1).contiguous()

        # GT HDR tensor (linear, unnormalized — for metric computation)
        hdr_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()

        # EV ladder from GT HDR (matches RepeatedHDRVideoDataset exactly)
        gamma = 2.2
        Y = (hdr_tensor[0] * 0.2126 + hdr_tensor[1] * 0.7152 + hdr_tensor[2] * 0.0722).clamp(min=1e-6)
        Ymax    = Y.max().item()
        Ymedian = Y.median().item()
        start_ev = np.log2((0.85 ** gamma) / Ymax)
        end_ev   = np.log2((0.85 ** gamma) / Ymedian)

        native_ev_range = float(end_ev - start_ev)

        if self.enable_ev_control:
            if self.c_zero_prob > 0.0 and random.random() < self.c_zero_prob:
                c = 0.0
            else:
                c = random.random()
            R_c = native_ev_range + c * max(0.0, self.r_max - native_ev_range)
            start_ev = end_ev - R_c   # stretch dark end only; bright anchor fixed
        else:
            c = 0.0

        ev_values = torch.linspace(start_ev, end_ev, self.sample_frames)
        ev_values = ev_values + torch.randn_like(ev_values) * 0.01
        ev_values, _ = torch.sort(ev_values)

        # Tone-map GT HDR → LDR frames
        ldr_clean = hdr_to_ldr_batch_np(gt_img, ev_values.numpy(), gamma=gamma)
        ldr_clean = np.round(ldr_clean * 65535.0) / 65535.0   # 16-bit quantization

        pixel_values_clean = torch.from_numpy(
            ldr_clean * 2.0 - 1.0
        ).permute(0, 3, 1, 2).contiguous()   # (T, 3, H, W)

        if self.use_noisy_samples:
            # Simulate sensor noise on the clean frames (same model as RepeatedHDRVideoDataset)
            ldr_noisy = ldr_clean.copy()
            for t in range(self.sample_frames):
                frame = ldr_clean[t].copy()
                signal = np.clip(frame, 0.0, 1.0) * 16383.0   # 14-bit
                # Poisson shot noise
                shot = np.random.normal(signal, np.sqrt(np.maximum(signal, 0.0)))
                # Gaussian read noise
                read_sigma = 15.0
                noisy = (shot + np.random.normal(0.0, read_sigma, signal.shape)) / 16383.0
                ldr_noisy[t] = np.clip(noisy, 0.0, 1.0)
            ldr_noisy = np.round(ldr_noisy * 65535.0) / 65535.0
            pixel_values = torch.from_numpy(
                ldr_noisy * 2.0 - 1.0
            ).permute(0, 3, 1, 2).contiguous()
        else:
            pixel_values = pixel_values_clean

        return {
            "pixel_values":       pixel_values,        # (T, 3, H, W) in [-1,1], noisy
            "pixel_values_clean": pixel_values_clean,  # (T, 3, H, W) in [-1,1], clean
            "hdr_image":          hdr_tensor,          # (3, H, W) float32 linear HDR
            "raw_input":          raw_tensor,          # (3, H, W) in [-1,1] — real raw frame
            "init_frame":         raw_path,
            "exposure_values":    ev_values,           # (T,) EV stops sorted
            "gamma":              gamma,
            "ev_control":         torch.tensor(c, dtype=torch.float32),
            "native_ev_range":    torch.tensor(native_ev_range, dtype=torch.float32),
        }
