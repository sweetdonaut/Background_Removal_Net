import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
import glob
import tifffile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SegmentationNetwork
from dataloader import calculate_positions


class InferenceDataset(Dataset):
    """Dataset for inference - loads test images without augmentation"""
    
    def __init__(self, test_path, patch_size=(256, 256), img_format='png_jpg', image_type='mvtec'):
        self.patch_size = patch_size
        self.img_format = img_format
        self.image_type = image_type
        
        # Load test images (check subdirectories)
        self.image_paths = []
        if img_format == 'png_jpg':
            # Check main directory
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.png")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.jpg")))
            # Check subdirectories
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.png")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.jpg")))
        else:  # tiff
            # Check main directory
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.tiff")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.tif")))
            # Check subdirectories
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.tiff")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.tif")))
        
        self.image_paths = sorted(self.image_paths)
        print(f"Found {len(self.image_paths)} test images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        if self.img_format == 'tiff':
            image = tifffile.imread(img_path)
            # TIFF files should already be float32
        else:
            image = cv2.imread(img_path)
            # Keep as uint8, will convert when normalizing
        
        # Store original image and info
        original_h, original_w = image.shape[:2]
        
        return {
            'image': image,
            'image_path': img_path,
            'original_size': (original_h, original_w)
        }


def sliding_window_inference(image, model, patch_size, device):
    """Perform sliding window inference using adaptive window positioning"""
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size
    
    # If image is smaller than patch, raise error
    if h < patch_h or w < patch_w:
        raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({patch_h}x{patch_w}). "
                         f"Please provide images at least {patch_h}x{patch_w} in size.")
    
    # Calculate adaptive positions (same as training)
    y_positions = calculate_positions(h, patch_h)
    x_positions = calculate_positions(w, patch_w)
    
    if y_positions is None or x_positions is None:
        raise ValueError(f"Image size ({h}x{w}) is too small for patch size ({patch_h}x{patch_w})")
    
    # Initialize output maps
    output_heatmap = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Sliding window with adaptive positions
    for y in y_positions:
        for x in x_positions:
            # Extract patch
            patch = image[y:y+patch_h, x:x+patch_w]
            
            # Extract channels
            target = patch[:, :, 0]
            ref1 = patch[:, :, 1]
            ref2 = patch[:, :, 2]
            
            # Prepare input
            three_channel = np.stack([target, ref1, ref2], axis=0)
            three_channel_tensor = torch.from_numpy(three_channel).float() / 255.0
            three_channel_tensor = three_channel_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(three_channel_tensor)
                output_sm = F.softmax(output, dim=1)
                patch_heatmap = output_sm[:, 1, :, :].squeeze().cpu().numpy()
            
            # Accumulate results
            output_heatmap[y:y+patch_h, x:x+patch_w] += patch_heatmap
            weight_map[y:y+patch_h, x:x+patch_w] += 1
    
    # Average overlapping regions
    output_heatmap = output_heatmap / np.maximum(weight_map, 1)
    
    return output_heatmap, image


def visualize_results(image, heatmap, output_path):
    """Create visualization with all components"""
    
    # Extract channels from original image
    target = image[:, :, 0]
    ref1 = image[:, :, 1]
    ref2 = image[:, :, 2]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 6, figure=fig)
    
    # 1. Target
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(target, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Target')
    ax1.axis('off')
    
    # 2. Ref1
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(ref1, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Ref1')
    ax2.axis('off')
    
    # 3. Ref2
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(ref2, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Ref2')
    ax3.axis('off')
    
    # 4. Target - Ref1
    diff1 = np.abs(target.astype(np.float32) - ref1.astype(np.float32))
    ax4 = fig.add_subplot(gs[3])
    # Dynamic range for diff1
    diff1_min = diff1.min()
    diff1_max = diff1.max()
    if diff1_max - diff1_min < 1e-8:
        d1_vmin, d1_vmax = 0, 255
    else:
        d1_vmin, d1_vmax = diff1_min, diff1_max
    ax4.imshow(diff1, cmap='hot', vmin=d1_vmin, vmax=d1_vmax)
    ax4.set_title(f'Target - Ref1 ({d1_vmin:.1f}-{d1_vmax:.1f})')
    ax4.axis('off')
    
    # 5. Target - Ref2
    diff2 = np.abs(target.astype(np.float32) - ref2.astype(np.float32))
    ax5 = fig.add_subplot(gs[4])
    # Dynamic range for diff2
    diff2_min = diff2.min()
    diff2_max = diff2.max()
    if diff2_max - diff2_min < 1e-8:
        d2_vmin, d2_vmax = 0, 255
    else:
        d2_vmin, d2_vmax = diff2_min, diff2_max
    ax5.imshow(diff2, cmap='hot', vmin=d2_vmin, vmax=d2_vmax)
    ax5.set_title(f'Target - Ref2 ({d2_vmin:.1f}-{d2_vmax:.1f})')
    ax5.axis('off')
    
    # 6. Heatmap
    ax6 = fig.add_subplot(gs[5])
    # Dynamic range adjustment for better visibility
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    
    # Avoid division by zero if heatmap is uniform
    if heatmap_max - heatmap_min < 1e-8:
        # Use default range if heatmap is uniform
        vmin, vmax = 0, 1
    else:
        # Use actual min/max for better contrast
        vmin, vmax = heatmap_min, heatmap_max
    
    im = ax6.imshow(heatmap, cmap='jet', vmin=vmin, vmax=vmax)
    ax6.set_title(f'Heatmap (range: {vmin:.4f}-{vmax:.4f})')
    ax6.axis('off')
    
    # Add colorbar for heatmap
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference(args):
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get patch size and image type from checkpoint
    patch_height = checkpoint['img_height']
    patch_width = checkpoint['img_width']
    patch_size = (patch_height, patch_width)
    
    # Get image type from checkpoint or args
    if 'image_type' in checkpoint:
        image_type = checkpoint['image_type']
    else:
        image_type = args.image_type
    
    print(f"Model patch size: {patch_size}")
    print(f"Image type: {image_type}")
    print(f"Using adaptive window positioning (same as training)")
    
    # Initialize model
    model = SegmentationNetwork(in_channels=3, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dataset
    dataset = InferenceDataset(
        test_path=args.test_path,
        patch_size=patch_size,
        img_format=args.img_format,
        image_type=image_type
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Storage for AUROC calculation
    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []
    
    # Inference loop
    for i, sample in enumerate(dataloader):
        # Get data
        image = sample['image'].squeeze().numpy()  # Remove batch dimension
        img_path = sample['image_path'][0]
        original_size = (sample['original_size'][0].item(), sample['original_size'][1].item())
        
        # Perform sliding window inference with adaptive positioning
        heatmap, processed_image = sliding_window_inference(
            image, model, patch_size, device
        )
        
        # Save visualization
        filename = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(args.output_dir, f'{filename}_result.png')
        
        visualize_results(processed_image, heatmap, output_path)
        print(f"Saved result: {output_path}")
            
        # If using ground truth masks, calculate metrics
        if args.use_ground_truth_mask:
            # Determine if this is a normal or anomaly image
            category = 'good' if 'good' in img_path else 'bright_spots'
            
            # Get anomaly score
            max_score = heatmap.max()
            
            if category == 'good':
                # Normal images
                all_scores.append(max_score)
                all_labels.append(0.0)
                pixel_scores.extend(heatmap.flatten())
                pixel_labels.extend(np.zeros_like(heatmap).flatten())
            else:
                # Anomaly images - load ground truth mask
                mask_name = filename + '_mask.' + img_path.split('.')[-1]
                # Navigate from test/bright_spots to ground_truth/bright_spots
                dataset_root = os.path.dirname(os.path.dirname(args.test_path))  # Go up to grid/
                mask_path = os.path.join(dataset_root, 'ground_truth', 'bright_spots', mask_name)
                
                if os.path.exists(mask_path):
                    # Load mask
                    if args.img_format == 'tiff':
                        gt_mask = tifffile.imread(mask_path)
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask[:, :, 0]
                    else:
                        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    gt_mask = gt_mask.astype(np.float32)
                    
                    # Image-level
                    has_anomaly = 1.0 if np.sum(gt_mask) > 0 else 0.0
                    all_scores.append(max_score)
                    all_labels.append(has_anomaly)
                    
                    # Pixel-level
                    pixel_scores.extend(heatmap.flatten())
                    pixel_labels.extend(gt_mask.flatten())
                else:
                    print(f"Warning: Ground truth mask not found: {mask_path}")
    
    # Calculate and display AUROC if requested
    if args.use_ground_truth_mask and len(all_scores) > 0:
        if len(np.unique(all_labels)) > 1:
            image_auroc = roc_auc_score(all_labels, all_scores)
            print(f"\nImage-level AUROC: {image_auroc:.4f}")
        else:
            print("\nImage-level AUROC: Cannot calculate (only one class present)")
        
        if len(np.unique(pixel_labels)) > 1:
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
            print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        else:
            print("Pixel-level AUROC: Cannot calculate (only one class present)")
    
    print(f"\nInference completed. Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference for Background Removal Net')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output visualizations')
    
    # Optional arguments
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use. Set to -1 for CPU (default: 0)')
    parser.add_argument('--img_format', type=str, choices=['png_jpg', 'tiff'], default='png_jpg',
                        help='Image format (default: png_jpg)')
    parser.add_argument('--image_type', type=str, choices=['strip', 'square', 'mvtec'], default='mvtec',
                        help='Image type (used if not in checkpoint): strip, square, mvtec')
    # Note: inference_stride is removed as we now use adaptive positioning
    parser.add_argument('--use_ground_truth_mask', type=str, choices=['True', 'False'], default='False',
                        help='Whether to calculate AUROC using ground truth masks (default: False)')
    
    args = parser.parse_args()
    
    # Convert string to boolean
    args.use_ground_truth_mask = args.use_ground_truth_mask == 'True'
    
    inference(args)


if __name__ == "__main__":
    main()