import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import io
import multiprocessing as mp
from glob import glob
import tifffile
from gaussian import (
    create_binary_mask,
    create_local_gaussian_defect,
    apply_local_defect_to_background
)
from generate_psf import load_config as load_psf_config, create_psf_defect, generate_one, clean_connected_peak
from functools import lru_cache
import psutil
from tqdm import tqdm


def parse_s3_path(s3_path):
    """Parse 's3://bucket/prefix/...' into (bucket, prefix)."""
    path = s3_path.replace('s3://', '', 1)
    parts = path.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    return bucket, prefix


def list_s3_objects(bucket, prefix, img_format):
    """List image keys from S3 bucket with given prefix."""
    import boto3
    endpoint_url = os.environ.get('S3_ENDPOINT_URL')
    client = boto3.client('s3', endpoint_url=endpoint_url)

    if img_format == 'tiff':
        extensions = ('.tiff', '.tif')
    else:
        extensions = ('.png', '.jpg')

    keys = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(extensions):
                keys.append(key)

    return [f's3://{bucket}/{k}' for k in sorted(keys)]


SUPPORTED_CHANNELS = ('target', 'ref1', 'ref2', 'diff1', 'diff2', 'double_det')


def sign_consistent_double_det(d1, d2):
    """Double detection that only fires when both diffs agree on sign.

    For each pixel: take the diff with smaller absolute value when sign(d1)
    equals sign(d2); otherwise return 0. Used as a defect-aware feature
    channel for the network and for the human-facing visualizations.
    """
    sign_match = np.sign(d1) == np.sign(d2)
    abs_smaller = np.where(np.abs(d1) <= np.abs(d2), d1, d2)
    return np.where(sign_match, abs_smaller, 0).astype(np.float32)


def build_input_channels(target, ref1, ref2, channel_names):
    """Assemble model input from raw target/ref1/ref2.

    target, ref1, ref2: 2-D float arrays, expected in [0, 255].
    channel_names: ordered list of names from SUPPORTED_CHANNELS.

    Returns (C, H, W) float32 array; every channel divided by 255 so raw
    channels land in [0, 1] and diff channels in [-1, 1].
    """
    d1 = target - ref1
    d2 = target - ref2
    table = {
        'target':     target,
        'ref1':       ref1,
        'ref2':       ref2,
        'diff1':      d1,
        'diff2':      d2,
        'double_det': sign_consistent_double_det(d1, d2),
    }
    out = []
    for name in channel_names:
        if name not in table:
            raise ValueError(
                f"Unknown input channel {name!r}. Supported: {SUPPORTED_CHANNELS}")
        out.append(table[name] / 255.0)
    return np.stack(out, axis=0).astype(np.float32)


def ensure_hwc(image):
    """Auto-detect and convert CHW to HWC if needed."""
    if len(image.shape) == 3 and image.shape[0] in (3, 4) and image.shape[1] > 4 and image.shape[2] > 4:
        return np.transpose(image, (1, 2, 0))
    return image


def ensure_3ch(image):
    """Keep only first 3 channels, discard 4th if present."""
    if len(image.shape) == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def calculate_positions(img_size, patch_size, min_patches=2):
    """Calculate patch positions: minimum overlap, maximum coverage
    
    This function is used by training, evaluation, and inference to ensure
    consistent patch positioning across all stages.
    
    Args:
        img_size: Size of the image dimension (height or width)
        patch_size: Size of the patch dimension
        min_patches: Minimum number of patches to generate
    
    Returns:
        List of starting positions for patches, or None if image is too small
    """
    max_start = img_size - patch_size
    
    if max_start < 0:
        return None  # Image too small
    elif max_start == 0:
        return [0]  # Only one position possible
    else:
        # Calculate number of patches needed for full coverage
        num_patches = max(min_patches, int(np.ceil(img_size / patch_size)))
        # Generate evenly spaced positions
        positions = np.linspace(0, max_start, num_patches).astype(int)
        return positions.tolist()


# Real captures sometimes show a defect at full intensity in target plus a
# weakened version in ONE of the refs (slow defect formation, slight stage
# drift between captures, etc.). Synthetic training mirrors this by leaking
# target-only defects to one ref at reduced intensity, with mask still
# labeling the defect — so the model learns to detect even when a ref is
# partially contaminated. Both the trigger probability and the scale range
# are per-Dataset (CLI-tunable).


def sample_magnitude(spec):
    """Sample defect intensity magnitude from a yaml spec.

    Accepts:
        scalar (int/float)             -> fixed value
        [a, b] / (a, b)                -> uniform in single range
        [[a1, b1], [a2, b2], ...]      -> equal-probability mixture across ranges
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("intensity_abs spec is empty")
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return float(np.random.uniform(spec[0], spec[1]))
        if all(isinstance(r, (list, tuple)) and len(r) == 2
               and all(isinstance(v, (int, float)) for v in r) for r in spec):
            chosen = spec[np.random.randint(len(spec))]
            return float(np.random.uniform(chosen[0], chosen[1]))
    raise ValueError(
        f"Invalid intensity_abs spec: {spec!r}. "
        "Expected scalar, [a, b], or [[a1,b1], [a2,b2], ...]"
    )


def _psf_pool_worker_init():
    """Pin BLAS threads to 1 inside each worker.

    Each worker runs its own complex64 FFT; opening BLAS threads here would
    oversubscribe (n_workers * BLAS_threads >> CPU cores) and slow things
    down. Setting these env vars at worker start is too late for the parent
    process but works for the FFT calls that happen inside this child.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _psf_pool_worker_make_one(args):
    """Generate one cleaned, bbox-cropped, [0,1]-normalized PSF defect.

    Mirrors create_psf_defect() but takes an explicit seed (not OS-derived)
    so the parent can hand each worker a deterministic, non-overlapping
    random stream via SeedSequence.spawn(). Returns None on cleaning
    failure — caller retries with the next spawned seed.
    """
    cfg, child_seed = args
    rng = np.random.default_rng(child_seed)
    raw, _ = generate_one(cfg, rng)
    cleaned = clean_connected_peak(raw, cfg.get('threshold_multiplier', 1.0))
    nz = np.argwhere(cleaned > 0)
    if len(nz) == 0:
        return None
    y0, x0 = nz.min(axis=0)
    y1, x1 = nz.max(axis=0)
    cropped = cleaned[y0:y1 + 1, x0:x1 + 1]
    if cropped.max() > 0:
        cropped = cropped / cropped.max()
    return cropped.astype(np.float32)


class PsfDefectPool:
    """Pre-generated pool of PSF defects for efficient training.

    Multiprocess pool builder: each worker generates one PSF independently,
    seeded via numpy SeedSequence.spawn() so streams are guaranteed
    non-overlapping. Pool size is achieved by oversampling seeds and
    re-spawning extras when cleaning failures occur.
    """

    def __init__(self, psf_cfgs, pool_size=1000, n_workers=4, master_seed=None):
        self.pools = []
        self.cfgs = list(psf_cfgs)
        n_workers = max(1, int(n_workers))
        print(f"Pre-generating PSF defect pool ({pool_size} per type, "
              f"n_workers={n_workers})...")
        for i, cfg in enumerate(psf_cfgs):
            self.pools.append(self._build_one_pool(cfg, i, pool_size, n_workers, master_seed))
        self.num_types = len(self.pools)

    def _build_one_pool(self, cfg, type_idx, pool_size, n_workers, master_seed):
        # Per-cfg SeedSequence so different cfg types get independent streams.
        # If master_seed is None we still get a SeedSequence (entropy from OS),
        # preserving the original "non-deterministic across runs" behavior.
        cfg_master = np.random.SeedSequence(master_seed,
                                            spawn_key=(type_idx,) if master_seed is not None else ())
        # Initial batch — oversample slightly so most pools complete in one pass.
        batch = max(pool_size + pool_size // 5, pool_size + 16)
        seed_pool = list(cfg_master.spawn(batch))
        seed_used = batch
        max_attempts = pool_size * 10

        pool = []
        failures = 0

        def _drain(results_iter, pbar):
            for defect in results_iter:
                if defect is not None:
                    pool.append(defect)
                    pbar.update(1)
                    if len(pool) >= pool_size:
                        return True
                else:
                    nonlocal_failures[0] += 1
            return False

        nonlocal_failures = [0]  # closure-mutable counter
        with tqdm(total=pool_size, desc=f"  Type {type_idx}", unit="psf") as pbar:
            if n_workers == 1:
                # Sequential fallback (debug / single-core env). Same per-PSF
                # logic as the worker function so behavior matches.
                for seed in seed_pool:
                    defect = _psf_pool_worker_make_one((cfg, seed))
                    if defect is not None:
                        pool.append(defect)
                        pbar.update(1)
                        if len(pool) >= pool_size:
                            break
                    else:
                        nonlocal_failures[0] += 1
                # Top up if we still need more
                while len(pool) < pool_size:
                    if seed_used >= max_attempts:
                        raise RuntimeError(
                            f"PSF config {type_idx}: too many generation failures "
                            f"({nonlocal_failures[0]} failures, {len(pool)} successes)")
                    extra = cfg_master.spawn(pool_size - len(pool))
                    seed_used += len(extra)
                    for seed in extra:
                        defect = _psf_pool_worker_make_one((cfg, seed))
                        if defect is not None:
                            pool.append(defect)
                            pbar.update(1)
                            if len(pool) >= pool_size:
                                break
                        else:
                            nonlocal_failures[0] += 1
            else:
                # Multiprocess path: keep one Pool open across top-up batches
                # to avoid repeated process fork/spawn overhead.
                with mp.Pool(n_workers, initializer=_psf_pool_worker_init) as workers:
                    args = [(cfg, s) for s in seed_pool]
                    done = _drain(workers.imap_unordered(_psf_pool_worker_make_one, args),
                                  pbar)
                    while not done:
                        if seed_used >= max_attempts:
                            raise RuntimeError(
                                f"PSF config {type_idx}: too many generation failures "
                                f"({nonlocal_failures[0]} failures, {len(pool)} successes)")
                        deficit = pool_size - len(pool)
                        # Oversample again to absorb continued failures.
                        extra_count = max(deficit + deficit // 5, deficit + 8)
                        extra = cfg_master.spawn(extra_count)
                        seed_used += extra_count
                        args = [(cfg, s) for s in extra]
                        done = _drain(workers.imap_unordered(_psf_pool_worker_make_one, args),
                                      pbar)
        return pool

    def sample(self):
        cfg_idx = np.random.randint(self.num_types)
        defect_idx = np.random.randint(len(self.pools[cfg_idx]))
        return self.pools[cfg_idx][defect_idx], self.cfgs[cfg_idx]


class Dataset(Dataset):
    """
    General dataset for defect detection training
    Generates target, ref1, ref2 images with synthetic defects
    Uses systematic sliding window to ensure all parts of images are used
    """

    def __init__(self, training_path, patch_size=(128, 128),
                 num_defects_range=(3, 8),
                 img_format='tiff', cache_size=0,
                 defect_mode='gaussian', psf_config_paths=None,
                 psf_pool_size=1000,
                 psf_pool_workers=4,
                 intensity_abs=(60, 80),
                 partial_leak_prob=0.0,
                 partial_leak_scale=(0.0, 0.0),
                 input_channels=('target', 'ref1', 'ref2')):
        self.patch_size = patch_size
        self.num_defects_range = num_defects_range
        self.img_format = img_format
        self.cache_size = cache_size
        self.defect_mode = defect_mode
        self.intensity_abs = intensity_abs
        self.partial_leak_prob = partial_leak_prob
        self.partial_leak_scale = partial_leak_scale
        self.input_channels = list(input_channels)
        for n in self.input_channels:
            if n not in SUPPORTED_CHANNELS:
                raise ValueError(
                    f"Unknown input channel {n!r}. Supported: {SUPPORTED_CHANNELS}")

        if defect_mode == 'psf':
            if not psf_config_paths:
                raise ValueError("psf_config_paths required for psf defect mode")
            psf_cfgs = [load_psf_config(p) for p in psf_config_paths]
            self.defect_pool = PsfDefectPool(
                psf_cfgs, pool_size=psf_pool_size, n_workers=psf_pool_workers)
            print(f"Defect mode: PSF ({len(psf_cfgs)} types: {psf_config_paths})")
        else:
            print(f"Defect mode: Gaussian")
        
        # Warning for small defect numbers
        if num_defects_range[0] < 3:
            print(f"Warning: num_defects_range minimum is {num_defects_range[0]}, which is less than 3.")
            print("This may result in limited target-only defects. Recommended minimum: 3")
        
        # Load training images
        self.is_s3 = training_path.startswith('s3://')

        if self.is_s3:
            bucket, prefix = parse_s3_path(training_path)
            self.training_paths = list_s3_objects(bucket, prefix, img_format)
            print(f"S3 source: {training_path}")
        else:
            if not os.path.exists(training_path):
                raise ValueError(f"Training path does not exist: {training_path}")
            if img_format == 'png_jpg':
                self.training_paths = glob(os.path.join(training_path, "*.png"))
                self.training_paths.extend(glob(os.path.join(training_path, "*.jpg")))
            elif img_format == 'tiff':
                self.training_paths = glob(os.path.join(training_path, "*.tiff"))
                self.training_paths.extend(glob(os.path.join(training_path, "*.tif")))

        if len(self.training_paths) == 0:
            raise ValueError(f"No images found in {training_path}")

        print(f"Found {len(self.training_paths)} training images")

        # Setup image cache if requested
        self._setup_cache()

        # Detect image size and display info (for square type, auto-detect size)
        self._detect_and_display_image_info()

        # Calculate patch positions once (not all combinations)
        self._setup_patch_positions()
    
    
    def _setup_cache(self):
        """Setup image cache"""
        if self.cache_size > 0:
            self._load_image = lru_cache(maxsize=self.cache_size)(self._load_image_uncached)
        else:
            self._load_image = self._load_image_uncached

    def _get_s3_client(self):
        """Lazy-init per-worker S3 client (not fork-safe, so created on demand)."""
        if not hasattr(self, '_s3_client'):
            import boto3
            endpoint_url = os.environ.get('S3_ENDPOINT_URL')
            self._s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        return self._s3_client

    def _load_image_uncached(self, img_path):
        """Load image from local filesystem or S3."""
        if img_path.startswith('s3://'):
            bucket, key = parse_s3_path(img_path)
            client = self._get_s3_client()
            response = client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            if self.img_format == 'tiff':
                return tifffile.imread(io.BytesIO(data))
            else:
                arr = np.frombuffer(data, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            if self.img_format == 'tiff':
                return tifffile.imread(img_path)
            else:
                return cv2.imread(img_path)

    def _detect_and_display_image_info(self):
        """Auto-detect image size and format from first image."""
        print("\n" + "="*60)
        print("Dataset Image Information")
        print("="*60)

        first_img_path = self.training_paths[0]
        sample_image = self._load_image(first_img_path)

        if sample_image is None:
            raise ValueError(f"Failed to load first image: {first_img_path}")

        print(f"Sample image: {os.path.basename(first_img_path)}")
        print(f"Raw shape: {sample_image.shape}")
        print(f"Data type: {sample_image.dtype}")
        print(f"Value range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")

        image = ensure_hwc(sample_image)
        if image.shape != sample_image.shape:
            print(f"After CHW->HWC conversion: {image.shape}")

        image = ensure_3ch(image)

        img_h, img_w = image.shape[:2]
        self.detected_img_h = img_h
        self.detected_img_w = img_w
        print(f"Image size: {img_h} x {img_w}")
        print("="*60 + "\n")
    
    def _setup_patch_positions(self):
        """Setup patch positions from auto-detected image size."""
        img_h, img_w = self.detected_img_h, self.detected_img_w

        self.y_positions = calculate_positions(img_h, self.patch_size[0])
        self.x_positions = calculate_positions(img_w, self.patch_size[1])

        if self.y_positions is None or self.x_positions is None:
            raise ValueError(f"Image size {img_h}x{img_w} is smaller than patch size {self.patch_size}")

        self.patches_per_image = len(self.y_positions) * len(self.x_positions)
        self.total_patches = len(self.training_paths) * self.patches_per_image

        print(f"Patch positions - Y: {len(self.y_positions)} positions {self.y_positions}")
        print(f"Patch positions - X: {len(self.x_positions)} positions {self.x_positions}")
        print(f"Total patches: {self.total_patches} ({self.patches_per_image}/image x {len(self.training_paths)} images)")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        # Dynamically calculate which image and patch position
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Calculate patch coordinates
        y_idx = patch_idx // len(self.x_positions)
        x_idx = patch_idx % len(self.x_positions)
        
        # Get image path and patch positions
        img_path = self.training_paths[img_idx]
        start_y = self.y_positions[y_idx]
        start_x = self.x_positions[x_idx]
        
        # Use cached or uncached loading
        image = self._load_image(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = ensure_hwc(image)
        image = ensure_3ch(image)

        # Normalize to 0-255 range if needed (handles raw sensor values)
        image = image.astype(np.float32)
        img_min = image.min()
        img_max = image.max()
        if img_min < 0 or img_max > 255:
            # Not in 0-255 range, normalize it
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min) * 255.0
            else:
                image = np.zeros_like(image)

        # Crop patch FIRST (key optimization)
        end_y = start_y + self.patch_size[0]
        end_x = start_x + self.patch_size[1]
        patch = image[start_y:end_y, start_x:end_x]

        # Convert to float32 for defect generation (handles both uint8 and float32 inputs)
        if patch.dtype == np.uint8:
            patch = patch.astype(np.float32)

        # Extract three channels from the patch
        target_channel = patch[:, :, 0]
        ref1_channel = patch[:, :, 1]
        ref2_channel = patch[:, :, 2]
        
        # Generate defects directly on the patch (new strategy: 50% chance)
        target, ref1, ref2, gt_mask = self.generate_defect_images_on_channels(
            target_channel, ref1_channel, ref2_channel
        )

        # Build whatever channel combination the trial asked for
        net_input = build_input_channels(target, ref1, ref2, self.input_channels)

        net_input_tensor = torch.from_numpy(net_input).float()
        gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)  # (1, H, W)

        return {
            "three_channel_input": net_input_tensor,
            "target_mask": gt_mask_tensor
        }
    
    def _create_one_defect(self, h, w):
        """Create one defect regardless of mode.
        Returns (local_defect_0to1, bounds, intensity) or (None, None, None).
        """
        if self.defect_mode == 'gaussian':
            magnitude = sample_magnitude(self.intensity_abs)
            intensity = magnitude if np.random.rand() < 0.5 else -magnitude
            margin = 5
            cx = np.random.randint(margin, w - margin)
            cy = np.random.randint(margin, h - margin)
            if np.random.rand() > 0.5:
                size, sigma = (3, 3), 1.3
            else:
                size, sigma = (3, 5), (1.0, 1.5)
            local_defect, bounds = create_local_gaussian_defect(
                center=(cx, cy), size=size, sigma=sigma,
                patch_shape=(h, w), patch_offset=(0, 0)
            )
            return local_defect, bounds, intensity

        elif self.defect_mode == 'psf':
            cropped, cfg = self.defect_pool.sample()
            magnitude = sample_magnitude(cfg.get('intensity_abs', self.intensity_abs))
            intensity = magnitude if np.random.rand() < 0.5 else -magnitude
            dh, dw = cropped.shape
            margin = 2
            max_y = h - dh - margin
            max_x = w - dw - margin
            if max_y < margin or max_x < margin:
                return None, None, None
            y = np.random.randint(margin, max_y + 1)
            x = np.random.randint(margin, max_x + 1)
            bounds = (y, y + dh, x, x + dw)
            return cropped, bounds, intensity

        return None, None, None

    def generate_defect_images_on_channels(self, target_channel, ref1_channel, ref2_channel):
        """Synthesize a 3-channel patch under plan A' (production-safe).

        Positive class: target_only defects only. (0,1,1) is NOT injected
        synthetically — production reports showed dense ref-side noise
        that an objective sign-consistent rule would mistakenly label as
        defect. The model learns to ignore those events implicitly from
        whatever ref-side noise exists in the good/ training tiffs.

        Ground-truth mask is tied to target_only_ids (binary mask from
        the defect kernel at threshold 0.1), preserving the asymmetric
        "target has anomaly" semantics that production actually cares
        about.
        """
        h, w = target_channel.shape

        # 10% of patches stay completely clean: preserves the image-level
        # "no defect anywhere" prior.
        if np.random.rand() < 0.1:
            return (target_channel.copy(), ref1_channel.copy(),
                    ref2_channel.copy(),
                    np.zeros((h, w), dtype=np.float32))

        num_defects = np.random.randint(self.num_defects_range[0], self.num_defects_range[1] + 1)

        defects = []
        for _ in range(num_defects):
            local_defect, bounds, intensity = self._create_one_defect(h, w)
            if local_defect is not None:
                defects.append((local_defect, bounds, intensity))

        if not defects:
            return (target_channel.copy(), ref1_channel.copy(),
                    ref2_channel.copy(),
                    np.zeros((h, w), dtype=np.float32))

        num_defects = len(defects)
        target = target_channel.copy()
        ref1 = ref1_channel.copy()
        ref2 = ref2_channel.copy()

        # Force at least one (1,0,0) target_only defect per defected patch
        # so every gradient step has a positive signal to learn from.
        n_target_only = np.random.randint(1, min(3, num_defects // 2 + 1))
        target_only_ids = set(np.random.choice(num_defects, n_target_only, replace=False))

        # Remaining defects spread across negative-class buckets:
        #   only_ref1_ids  -> (1,1,0)  (one ref agrees with target)
        #   only_ref2_ids  -> (1,0,1)
        #   rest           -> (1,1,1)  (all three agree)
        # Note: (0,1,1) is intentionally absent. See class docstring.
        remaining_ids = list(set(range(num_defects)) - target_only_ids)
        np.random.shuffle(remaining_ids)

        only_ref1_ids = set()
        only_ref2_ids = set()
        if len(remaining_ids) >= 2:
            n_remaining = len(remaining_ids)
            n_only_ref1 = n_remaining // 3
            n_only_ref2 = n_remaining // 3
            only_ref1_ids = set(remaining_ids[:n_only_ref1])
            only_ref2_ids = set(remaining_ids[n_only_ref1:n_only_ref1 + n_only_ref2])
        elif len(remaining_ids) == 1:
            if np.random.rand() < 0.5:
                only_ref1_ids = set(remaining_ids)
            else:
                only_ref2_ids = set(remaining_ids)

        # Per-defect channel apply rules:
        #   target is always applied (every defect lives on target)
        #   ref1 is dropped for target_only and only_ref2 buckets
        #   ref2 is dropped for target_only and only_ref1 buckets
        remove_ref1 = target_only_ids | only_ref2_ids
        remove_ref2 = target_only_ids | only_ref1_ids

        # Partial leak: target_only defects may bleed into one ref at
        # reduced intensity. Mask still labels the leaked pixel as defect
        # because the strong-target signal remains.
        leak_to_ref1 = {}
        leak_to_ref2 = {}
        for tid in target_only_ids:
            if np.random.rand() < self.partial_leak_prob:
                scale = np.random.uniform(*self.partial_leak_scale)
                if np.random.rand() < 0.5:
                    leak_to_ref1[tid] = scale
                else:
                    leak_to_ref2[tid] = scale

        gt_mask = np.zeros((h, w), dtype=np.float32)

        for i, (local_defect, bounds, intensity) in enumerate(defects):
            target = apply_local_defect_to_background(target, local_defect, bounds, intensity)

            if i not in remove_ref1:
                ref1 = apply_local_defect_to_background(ref1, local_defect, bounds, intensity)
            elif i in leak_to_ref1:
                ref1 = apply_local_defect_to_background(
                    ref1, local_defect, bounds, intensity * leak_to_ref1[i])

            if i not in remove_ref2:
                ref2 = apply_local_defect_to_background(ref2, local_defect, bounds, intensity)
            elif i in leak_to_ref2:
                ref2 = apply_local_defect_to_background(
                    ref2, local_defect, bounds, intensity * leak_to_ref2[i])

            if i in target_only_ids:
                local_mask = create_binary_mask(local_defect, threshold=0.1)
                y_start, y_end, x_start, x_end = bounds
                gt_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    gt_mask[y_start:y_end, x_start:x_end], local_mask
                )

        return target, ref1, ref2, gt_mask