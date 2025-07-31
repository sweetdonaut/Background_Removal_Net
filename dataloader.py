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
    apply_defect_to_background
)


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
                 num_defects_range=(5, 15), defect_size_range=(3, 5),
                 img_format='png', image_type='mvtec'):
        """
        Args:
            training_path: Path to training images (required)
            patch_size: Size of patches to extract (height, width)
            num_defects_range: (min, max) number of defects
            defect_size_range: Not used anymore, we use fixed 3x3 and 3x5
            img_format: Image format to load ('png_jpg' or 'tiff')
            image_type: Type of images ('strip', 'square', 'mvtec')
        """
        self.patch_size = patch_size
        self.num_defects_range = num_defects_range
        self.img_format = img_format
        self.image_type = image_type
        
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
        
        # Pre-calculate all patches positions for all images
        self.patches = []
        self._prepare_patches()
    
    
    def _prepare_patches(self):
        """Pre-calculate all valid patch positions for all images"""
        # Determine image size based on image_type
        if self.image_type == 'strip':
            img_h, img_w = 976, 176
        elif self.image_type == 'square':
            img_h, img_w = 600, 600
        else:  # mvtec
            img_h, img_w = 1024, 1024
        
        print(f"Using image size {img_h}x{img_w} for image_type: {self.image_type}")
        
        # Calculate positions for height and width
        y_positions = calculate_positions(img_h, self.patch_size[0])
        x_positions = calculate_positions(img_w, self.patch_size[1])
        
        if y_positions is None or x_positions is None:
            raise ValueError(f"Image size {img_h}x{img_w} is smaller than patch size {self.patch_size}")
        
        print(f"Patch positions - Y: {len(y_positions)} positions {y_positions[:3]}{'...' if len(y_positions) > 3 else ''}")
        print(f"Patch positions - X: {len(x_positions)} positions {x_positions[:3]}{'...' if len(x_positions) > 3 else ''}")
        
        # Generate patches for each image
        for img_idx, img_path in enumerate(self.training_paths):
            for y in y_positions:
                for x in x_positions:
                    self.patches.append({
                        'img_idx': img_idx,
                        'img_path': img_path,
                        'y': y,
                        'x': x
                    })
        
        patches_per_image = len(y_positions) * len(x_positions)
        print(f"Total patches: {len(self.patches)} ({patches_per_image} patches per image from {len(self.training_paths)} images)")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Get patch info
        patch_info = self.patches[idx]
        img_path = patch_info['img_path']
        start_y = patch_info['y']
        start_x = patch_info['x']
        
        if self.img_format == 'tiff':
            # Use tifffile for TIFF format
            image = tifffile.imread(img_path)
            # TIFF files should already be float32 in 0-255 range
        else:
            # Use cv2 for PNG/JPG
            image = cv2.imread(img_path)
            # Keep as uint8, will convert when normalizing
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Extract three channels from original image
        target_channel = image[:, :, 0]
        ref1_channel = image[:, :, 1]
        ref2_channel = image[:, :, 2]
        
        # Generate defects on the original image first
        target, ref1, ref2, gt_mask = self.generate_defect_images_on_channels(
            target_channel, ref1_channel, ref2_channel
        )
        
        # Use pre-calculated position to crop
        end_y = start_y + self.patch_size[0]
        end_x = start_x + self.patch_size[1]
        
        # Crop all channels and mask at the pre-determined position
        target = target[start_y:end_y, start_x:end_x]
        ref1 = ref1[start_y:end_y, start_x:end_x]
        ref2 = ref2[start_y:end_y, start_x:end_x]
        gt_mask = gt_mask[start_y:end_y, start_x:end_x]
        
        # Stack as 3-channel input
        three_channel = np.stack([target, ref1, ref2], axis=0)  # (3, H, W)
        
        # Convert to torch tensors
        three_channel_tensor = torch.from_numpy(three_channel).float() / 255.0
        gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)  # (1, H, W)
        
        return {
            "three_channel_input": three_channel_tensor,
            "target_mask": gt_mask_tensor
        }
    
    def generate_defect_images_on_channels(self, target_channel, ref1_channel, ref2_channel):
        """
        Generate defects and apply to three channels separately
        """
        h, w = target_channel.shape
        
        # Random number of defects
        num_defects = np.random.randint(*self.num_defects_range)
        
        # Generate defect parameters
        defect_params = []
        for i in range(num_defects):
            # Random position
            margin = 5
            cx = np.random.randint(margin, w - margin)
            cy = np.random.randint(margin, h - margin)
            
            # Random size (3x3 or 3x5)
            if np.random.rand() > 0.5:
                size = (3, 3)
                sigma = 1.0
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
        all_masks = []
        
        # Generate all defects
        all_defects = []
        for params in defect_params:
            defect = create_gaussian_defect(
                center=params['center'],
                size=params['size'],
                sigma=params['sigma'],
                image_shape=(h, w)
            )
            mask = create_binary_mask(defect, threshold=0.1)
            all_masks.append(mask)
            all_defects.append((defect, params['intensity']))
        
        # Apply all defects to target
        for defect, intensity in all_defects:
            target = apply_defect_to_background(target, defect, intensity)
        
        # Randomly decide which defects to remove from ref1 and ref2
        max_remove = max(1, num_defects - 2)
        
        if np.random.rand() < 0.8 and num_defects >= 3:
            # 80% chance: ensure at least some target-only defects
            num_remove_ref1 = np.random.randint(1, max_remove + 1)
            num_remove_ref2 = np.random.randint(1, max_remove + 1)
            
            remove_ref1 = np.random.choice(num_defects, num_remove_ref1, replace=False)
            remove_ref2 = np.random.choice(num_defects, num_remove_ref2, replace=False)
            
            # Force overlap: randomly pick 1-2 indices from ref1 and add to ref2
            num_force_overlap = min(2, num_remove_ref1)
            overlap_indices = np.random.choice(remove_ref1, num_force_overlap, replace=False)
            
            # Merge and remove duplicates
            remove_ref2 = np.unique(np.concatenate([remove_ref2, overlap_indices]))
        else:
            # 20% chance: completely random (may have no target-only defects)
            num_remove_ref1 = np.random.randint(1, max_remove + 1)
            num_remove_ref2 = np.random.randint(1, max_remove + 1)
            
            remove_ref1 = np.random.choice(num_defects, num_remove_ref1, replace=False)
            remove_ref2 = np.random.choice(num_defects, num_remove_ref2, replace=False)
        
        # Apply selected defects to ref1 and ref2
        for i, (defect, intensity) in enumerate(all_defects):
            if i not in remove_ref1:
                ref1 = apply_defect_to_background(ref1, defect, intensity)
            
            if i not in remove_ref2:
                ref2 = apply_defect_to_background(ref2, defect, intensity)
        
        # Create ground truth mask (defects only in target)
        gt_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(num_defects):
            if i in remove_ref1 and i in remove_ref2:
                gt_mask = np.maximum(gt_mask, all_masks[i])
        
        return target, ref1, ref2, gt_mask