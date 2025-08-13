import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from glob import glob
import tifffile
from gaussian import (
    create_gaussian_defect,
    create_binary_mask,
    apply_defect_to_background,
    create_local_gaussian_defect,
    apply_local_defect_to_background
)
from functools import lru_cache
import psutil
from scipy.ndimage import gaussian_filter

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
    
    def __init__(self, training_path, patch_size=(256, 256), 
                 num_defects_range=(3, 8), defect_size_range=(3, 5),
                 img_format='tiff', image_type='strip', cache_size=0):
        """
        Args:
            training_path: Path to training images (required)
            patch_size: Size of patches to extract (height, width)
            num_defects_range: (min, max) number of defects per patch (default 3-8)
            defect_size_range: Not used anymore, we use fixed 3x3 and 3x5
            img_format: Image format to load ('png_jpg' or 'tiff')
            image_type: Type of images ('strip', 'square', 'mvtec')
            cache_size: Number of images to cache (0 = no cache)
        """
        self.patch_size = patch_size
        self.num_defects_range = num_defects_range
        self.img_format = img_format
        self.image_type = image_type
        self.cache_size = cache_size
        
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
    
    def _setup_patch_positions(self):
        """Setup patch positions for dynamic calculation"""
        # Determine image size based on image_type
        if self.image_type == 'strip':
            img_h, img_w = 976, 176
        elif self.image_type == 'square':
            img_h, img_w = 600, 600
        else:  # mvtec
            img_h, img_w = 1024, 1024
        
        print(f"Using image size {img_h}x{img_w} for image_type: {self.image_type}")
        
        # Calculate positions for height and width
        if self.image_type == 'strip':
            # For strip images, use 9 patches in Y direction
            self.y_positions = calculate_positions(img_h, self.patch_size[0], min_patches=9)
        else:
            self.y_positions = calculate_positions(img_h, self.patch_size[0])
        self.x_positions = calculate_positions(img_w, self.patch_size[1])
        
        if self.y_positions is None or self.x_positions is None:
            raise ValueError(f"Image size {img_h}x{img_w} is smaller than patch size {self.patch_size}")
        
        print(f"Patch positions - Y: {len(self.y_positions)} positions {self.y_positions[:3]}{'...' if len(self.y_positions) > 3 else ''}")
        print(f"Patch positions - X: {len(self.x_positions)} positions {self.x_positions[:3]}{'...' if len(self.x_positions) > 3 else ''}")
        
        # Calculate total patches info
        self.patches_per_image = len(self.y_positions) * len(self.x_positions)
        self.total_patches = len(self.training_paths) * self.patches_per_image
        
        print(f"Total patches: {self.total_patches} ({self.patches_per_image} patches per image from {len(self.training_paths)} images)")
    
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
        
        # Convert from CHW to HWC format for stripe dataset
        if self.image_type == 'strip' and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Crop patch FIRST (key optimization)
        end_y = start_y + self.patch_size[0]
        end_x = start_x + self.patch_size[1]
        patch = image[start_y:end_y, start_x:end_x]
        
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
    
    def generate_defect_images_on_channels(self, target_channel, ref1_channel, ref2_channel,
                                          patch_offset_y=0, patch_offset_x=0, 
                                          full_image_shape=None):
        """
        Generate defects and apply to three channels separately
        New strategy: 
        - 20% chance to generate edge negative samples (only if structural edges exist)
        - 40% chance to have point defects
        - 40% chance to have no modifications
        
        Args:
            target_channel, ref1_channel, ref2_channel: patch channels (already cropped)
            patch_offset_y, patch_offset_x: patch position in the full image (not used in new strategy)
            full_image_shape: (h, w) of the full image (not used in new strategy)
        """
        h, w = target_channel.shape  # patch dimensions
        
        # Decide what type of augmentation to apply
        rand_val = np.random.rand()
        
        if rand_val < 0.2:  # 20% chance: Try edge negative samples
            # First check if this patch has structural edges
            if self._has_structural_edges(target_channel):
                # Has structural edges, apply edge enhancement
                return self._generate_edge_negative_samples(target_channel, ref1_channel, ref2_channel)
            else:
                # No structural edges, generate point defects instead
                pass  # Continue with point defect generation below
        
        elif rand_val < 0.6:  # 40% chance: Point defects (original logic)
            # Continue with original point defect generation
            pass  # Will continue below
        
        else:  # 40% chance: No modifications
            # No defects - return original channels
            gt_mask = np.zeros_like(target_channel, dtype=np.float32)
            return target_channel.copy(), ref1_channel.copy(), ref2_channel.copy(), gt_mask
        
        # If we have defects, generate 3-8 defects directly on the patch
        # Use at least 3 defects to ensure effective contrastive learning
        num_defects = np.random.randint(3, 9)  # 3 to 8 defects
        
        # Generate defect parameters in PATCH coordinates
        defect_params = []
        for i in range(num_defects):
            # Random position in patch
            margin = 5
            cx = np.random.randint(margin, w - margin)
            cy = np.random.randint(margin, h - margin)
            
            # Random size (3x3 or 3x5)
            if np.random.rand() > 0.5:
                size = (3, 3)
                sigma = 1.3  # Increased from 1.0 to make mask larger
            else:
                size = (3, 5)
                sigma = (1.0, 1.5)
            
            # Random intensity - increased for better visibility
            intensity = np.random.choice([-80, -60, 60, 80])
            
            defect_params.append({
                'center': (cx, cy),
                'size': size,
                'sigma': sigma,
                'intensity': intensity,
                'id': i
            })
        
        # Create copies for modification
        target = target_channel.copy()
        ref1 = ref1_channel.copy()
        ref2 = ref2_channel.copy()
        
        # New smart allocation strategy for contrastive learning
        
        # Step 1: Ensure at least 1-2 target-only defects
        num_target_only = np.random.randint(1, min(3, num_defects // 2 + 1))
        target_only_ids = set(np.random.choice(num_defects, num_target_only, replace=False))
        
        # Step 2: Allocate remaining defects to maximize contrastive signal
        remaining_ids = list(set(range(num_defects)) - target_only_ids)
        np.random.shuffle(remaining_ids)
        
        # Initialize sets
        only_ref1_ids = set()
        only_ref2_ids = set()
        both_refs_ids = set()
        
        # Distribute remaining defects to ensure ref1 != ref2
        if len(remaining_ids) >= 2:
            # Split into three groups for diversity
            n_remaining = len(remaining_ids)
            n_only_ref1 = n_remaining // 3
            n_only_ref2 = n_remaining // 3
            # Rest goes to both_refs
            
            only_ref1_ids = set(remaining_ids[:n_only_ref1])
            only_ref2_ids = set(remaining_ids[n_only_ref1:n_only_ref1 + n_only_ref2])
            both_refs_ids = set(remaining_ids[n_only_ref1 + n_only_ref2:])
        elif len(remaining_ids) == 1:
            # With only 1 remaining, randomly assign to ref1 or ref2
            if np.random.rand() < 0.5:
                only_ref1_ids = set(remaining_ids)
            else:
                only_ref2_ids = set(remaining_ids)
        
        # Build removal sets based on allocation
        # Remove from ref1: target-only + only_ref2
        remove_ref1 = target_only_ids | only_ref2_ids
        # Remove from ref2: target-only + only_ref1
        remove_ref2 = target_only_ids | only_ref1_ids
        
        # Initialize ground truth mask
        gt_mask = np.zeros((h, w), dtype=np.float32)
        
        # Now render defects directly on the patch
        for i, params in enumerate(defect_params):
            # Generate defect directly in patch coordinates
            local_defect, bounds = create_local_gaussian_defect(
                center=params['center'],
                size=params['size'],
                sigma=params['sigma'],
                patch_shape=(h, w),
                patch_offset=(0, 0)  # No offset needed since we're working in patch coordinates
            )
            
            if local_defect is None:
                continue  # Should not happen with proper margin
            
            # Apply to channels based on pre-determined assignment
            intensity = params['intensity']
            
            # Target always gets the defect
            target = apply_local_defect_to_background(target, local_defect, bounds, intensity)
            
            # Ref1: only if not in remove_ref1
            if i not in remove_ref1:
                ref1 = apply_local_defect_to_background(ref1, local_defect, bounds, intensity)
            
            # Ref2: only if not in remove_ref2
            if i not in remove_ref2:
                ref2 = apply_local_defect_to_background(ref2, local_defect, bounds, intensity)
            
            # Ground truth mask: only if this is a target-only defect
            if i in target_only_ids:
                local_mask = create_binary_mask(local_defect, threshold=0.1)
                y_start, y_end, x_start, x_end = bounds
                gt_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    gt_mask[y_start:y_end, x_start:x_end],
                    local_mask
                )
        
        return target, ref1, ref2, gt_mask
    
    def _has_structural_edges(self, image_patch, edge_ratio_threshold=0.02):
        """
        Check if a patch has structural edges (grid, stripes) rather than point defects
        
        Args:
            image_patch: Input patch to check
            edge_ratio_threshold: Minimum ratio of edge pixels to be considered structural
                                 Default 0.02 (2%) safely excludes point defects
        
        Returns:
            bool: True if structural edges exist, False otherwise
        """
        # Convert to uint8 for Canny edge detection
        patch_uint8 = np.clip(image_patch, 0, 255).astype(np.uint8)
        
        # Detect edges using Canny
        edges = cv2.Canny(patch_uint8, 50, 150)
        
        # Calculate edge pixel ratio
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Based on testing:
        # - Single defects: ~0.1% edge ratio
        # - Multiple defects (5-8): ~0.5-0.7% edge ratio  
        # - Grid/stripe structures: >9% edge ratio
        # Using 2% threshold provides safe margin
        
        return edge_ratio > edge_ratio_threshold
    
    def _generate_edge_negative_samples(self, target_channel, ref1_channel, ref2_channel):
        """
        Generate edge enhancement as negative samples.
        These edges should NOT be detected as defects (GT mask = 0).
        This helps the model learn to distinguish between edge differences and real point defects.
        """
        h, w = target_channel.shape
        
        # Create copies for modification
        target = target_channel.copy()
        ref1 = ref1_channel.copy()
        ref2 = ref2_channel.copy()
        
        # Detect edges using Canny edge detector
        # Convert to uint8 for Canny
        target_uint8 = np.clip(target, 0, 255).astype(np.uint8)
        
        # Use Canny to detect edges
        edges = cv2.Canny(target_uint8, 50, 150)
        
        # Dilate edges to make them thicker (more realistic)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert to float mask
        edge_mask = edges_dilated.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to make the edge enhancement smoother
        
        edge_mask_smooth = gaussian_filter(edge_mask, sigma=1.0)
        
        # Generate random enhancement patterns
        enhancement_type = np.random.choice(['brightness', 'mixed', 'noise'])
        
        if enhancement_type == 'brightness':
            # Simple brightness difference at edges
            edge_intensity = np.random.uniform(10, 30)
            target = target + edge_mask_smooth * edge_intensity
            ref1 = ref1 - edge_mask_smooth * edge_intensity * 0.5
            ref2 = ref2 - edge_mask_smooth * edge_intensity * 0.5
            
        elif enhancement_type == 'mixed':
            # Different enhancement for each channel
            target = target + edge_mask_smooth * np.random.uniform(5, 20)
            ref1 = ref1 + edge_mask_smooth * np.random.uniform(-10, 10)
            ref2 = ref2 + edge_mask_smooth * np.random.uniform(-10, 10)
            
        else:  # 'noise'
            # Add noise specifically at edges
            edge_noise_target = np.random.randn(h, w) * edge_mask_smooth * 10
            edge_noise_ref1 = np.random.randn(h, w) * edge_mask_smooth * 10
            edge_noise_ref2 = np.random.randn(h, w) * edge_mask_smooth * 10
            
            target = target + edge_noise_target
            ref1 = ref1 + edge_noise_ref1
            ref2 = ref2 + edge_noise_ref2
        
        # Occasionally add some stripe patterns as well (10% chance)
        if np.random.rand() < 0.1:
            # Add horizontal or vertical stripes
            if np.random.rand() < 0.5:
                # Horizontal stripes
                stripe_intensity = np.random.uniform(5, 15)
                for y in range(0, h, np.random.randint(8, 20)):
                    target[y:y+2, :] += stripe_intensity
                    ref1[y:y+2, :] -= stripe_intensity * 0.3
                    ref2[y:y+2, :] -= stripe_intensity * 0.3
            else:
                # Vertical stripes
                stripe_intensity = np.random.uniform(5, 15)
                for x in range(0, w, np.random.randint(8, 20)):
                    target[:, x:x+2] += stripe_intensity
                    ref1[:, x:x+2] -= stripe_intensity * 0.3
                    ref2[:, x:x+2] -= stripe_intensity * 0.3
        
        # Clip values to valid range
        target = np.clip(target, 0, 255)
        ref1 = np.clip(ref1, 0, 255)
        ref2 = np.clip(ref2, 0, 255)
        
        # IMPORTANT: GT mask is all zeros - these are NOT defects!
        gt_mask = np.zeros_like(target_channel, dtype=np.float32)
        
        return target, ref1, ref2, gt_mask