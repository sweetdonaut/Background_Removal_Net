import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from glob import glob
import tifffile
from gaussian import (
    create_binary_mask,
    create_local_gaussian_defect,
    apply_local_defect_to_background
)
from generate_psf import load_config as load_psf_config, create_psf_defect
from functools import lru_cache
import psutil


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


class Dataset(Dataset):
    """
    General dataset for defect detection training
    Generates target, ref1, ref2 images with synthetic defects
    Uses systematic sliding window to ensure all parts of images are used
    """
    
    def __init__(self, training_path, patch_size=(128, 128),
                 num_defects_range=(3, 8),
                 img_format='tiff', cache_size=0,
                 defect_mode='gaussian', psf_config_paths=None):
        self.patch_size = patch_size
        self.num_defects_range = num_defects_range
        self.img_format = img_format
        self.cache_size = cache_size
        self.defect_mode = defect_mode

        if defect_mode == 'psf':
            if not psf_config_paths:
                raise ValueError("psf_config_paths required for psf defect mode")
            self.psf_cfgs = [load_psf_config(p) for p in psf_config_paths]
            print(f"Defect mode: PSF ({len(self.psf_cfgs)} types: {psf_config_paths})")
        else:
            print(f"Defect mode: Gaussian")
        
        # Warning for small defect numbers
        if num_defects_range[0] < 3:
            print(f"Warning: num_defects_range minimum is {num_defects_range[0]}, which is less than 3.")
            print("This may result in limited target-only defects. Recommended minimum: 3")
        
        # Load training images
        if not os.path.exists(training_path):
            raise ValueError(f"Training path does not exist: {training_path}")
        
        # Load images based on format
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

    def _load_image_uncached(self, img_path):
        """Load image without caching"""
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
        
        # Stack as 3-channel input
        three_channel = np.stack([target, ref1, ref2], axis=0)  # (3, H, W)
        
        # Convert to torch tensors
        three_channel_tensor = torch.from_numpy(three_channel).float() / 255.0
        gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)  # (1, H, W)
        
        return {
            "three_channel_input": three_channel_tensor,
            "target_mask": gt_mask_tensor
        }
    
    def _create_one_defect(self, h, w):
        """Create one defect regardless of mode.
        Returns (local_defect_0to1, bounds, intensity) or (None, None, None).
        """
        intensity = np.random.choice([-80, -60, 60, 80])

        if self.defect_mode == 'gaussian':
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
            cfg = self.psf_cfgs[np.random.randint(len(self.psf_cfgs))]
            cropped = create_psf_defect(cfg)
            if cropped is None:
                return None, None, None
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
        h, w = target_channel.shape

        if np.random.rand() >= 0.5:
            gt_mask = np.zeros((h, w), dtype=np.float32)
            return target_channel.copy(), ref1_channel.copy(), ref2_channel.copy(), gt_mask

        num_defects = np.random.randint(self.num_defects_range[0], self.num_defects_range[1] + 1)

        defects = []
        for _ in range(num_defects):
            local_defect, bounds, intensity = self._create_one_defect(h, w)
            if local_defect is not None:
                defects.append((local_defect, bounds, intensity))

        if not defects:
            gt_mask = np.zeros((h, w), dtype=np.float32)
            return target_channel.copy(), ref1_channel.copy(), ref2_channel.copy(), gt_mask

        num_defects = len(defects)
        target = target_channel.copy()
        ref1 = ref1_channel.copy()
        ref2 = ref2_channel.copy()

        # Allocation: ensure target-only defects for contrastive learning
        num_target_only = np.random.randint(1, min(3, num_defects // 2 + 1))
        target_only_ids = set(np.random.choice(num_defects, num_target_only, replace=False))

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

        remove_ref1 = target_only_ids | only_ref2_ids
        remove_ref2 = target_only_ids | only_ref1_ids

        gt_mask = np.zeros((h, w), dtype=np.float32)

        for i, (local_defect, bounds, intensity) in enumerate(defects):
            target = apply_local_defect_to_background(target, local_defect, bounds, intensity)
            if i not in remove_ref1:
                ref1 = apply_local_defect_to_background(ref1, local_defect, bounds, intensity)
            if i not in remove_ref2:
                ref2 = apply_local_defect_to_background(ref2, local_defect, bounds, intensity)
            if i in target_only_ids:
                local_mask = create_binary_mask(local_defect, threshold=0.1)
                y_start, y_end, x_start, x_end = bounds
                gt_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    gt_mask[y_start:y_end, x_start:x_end], local_mask
                )

        return target, ref1, ref2, gt_mask