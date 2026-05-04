import torch
from torch.utils.data import DataLoader
from torch import optim
import os
import shutil
import argparse
import numpy as np
import cv2
import glob
import tifffile
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from loss import FocalLoss
from model import SegmentationNetwork
from dataloader import Dataset, sample_magnitude, calculate_positions, ensure_hwc, ensure_3ch


def _render_defect_grid(defects, out_path, suptitle=None):
    """Render a 3x5 grid of defect patches (centered + padded) with colorbars."""
    if not defects:
        print(f"Warning: no defects to render, skipping {out_path}")
        return

    max_h = max(d[0].shape[0] for d in defects)
    max_w = max(d[0].shape[1] for d in defects)
    abs_max = max(abs(d[1]) for d in defects)

    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    for ax_idx, ax in enumerate(axes.flat):
        if ax_idx >= len(defects):
            ax.axis('off')
            continue
        local_defect, intensity = defects[ax_idx]
        h, w = local_defect.shape
        padded = np.zeros((max_h, max_w), dtype=np.float32)
        py, px = (max_h - h) // 2, (max_w - w) // 2
        padded[py:py + h, px:px + w] = local_defect
        signed = padded * intensity
        im = ax.imshow(signed, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'patch {ax_idx + 1} ({h}x{w}, x{intensity:+.1f})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved defect examples: {out_path}")


def save_training_artifacts(checkpoint_path, dataset, psf_config_paths, n_patches=15):
    """Save reproducibility artifacts alongside the checkpoints.

    - Copies any PSF yaml configs used (preserves comments and filenames).
    - Renders n_patches example defect patches per type, with colorbars,
      so brightness ranges and defect appearance are recoverable later.
    """
    if psf_config_paths:
        for src in psf_config_paths:
            shutil.copy(src, os.path.join(checkpoint_path, os.path.basename(src)))

    state = np.random.get_state()

    if dataset.defect_mode == 'psf':
        for type_idx, cfg_path in enumerate(psf_config_paths):
            type_name = os.path.splitext(os.path.basename(cfg_path))[0]
            pool = dataset.defect_pool.pools[type_idx]
            cfg = dataset.defect_pool.cfgs[type_idx]
            intensity_spec = cfg.get('intensity_abs', dataset.intensity_abs)

            defects = []
            for _ in range(n_patches):
                d = pool[np.random.randint(len(pool))]
                magnitude = sample_magnitude(intensity_spec)
                intensity = magnitude if np.random.rand() < 0.5 else -magnitude
                defects.append((d, intensity))

            out = os.path.join(checkpoint_path, f'defect_examples_{type_name}.png')
            _render_defect_grid(defects, out, suptitle=type_name)
    else:
        H = W = 200
        defects = []
        attempts = 0
        while len(defects) < n_patches and attempts < n_patches * 30:
            local_defect, _, intensity = dataset._create_one_defect(H, W)
            if local_defect is not None:
                defects.append((local_defect, intensity))
            attempts += 1
        out = os.path.join(checkpoint_path, 'defect_examples.png')
        _render_defect_grid(defects, out)

    np.random.set_state(state)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_focal_gamma(epoch, total_epochs, gamma_start, gamma_end, schedule='cosine'):
    """
    Calculate gamma value for current epoch using cosine schedule
    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        gamma_start: initial gamma value
        gamma_end: final gamma value
        schedule: 'linear' or 'cosine'
    Returns:
        current gamma value
    """
    if schedule == 'linear':
        gamma = gamma_start + (gamma_end - gamma_start) * (epoch / total_epochs)
    elif schedule == 'cosine':
        import math
        progress = epoch / total_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * (1 - math.cos(progress * math.pi)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return gamma

def evaluate_model(model, valid_root, ground_truth_root, img_format, patch_size, device):
    model.eval()

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    with torch.no_grad():
        for category in ['good', 'bright_spots']:
            category_path = os.path.join(valid_root, category)
            if not os.path.exists(category_path):
                continue

            if img_format == 'png_jpg':
                valid_images = glob.glob(os.path.join(category_path, "*.png"))
                valid_images.extend(glob.glob(os.path.join(category_path, "*.jpg")))
            else:
                valid_images = glob.glob(os.path.join(category_path, "*.tiff"))
                valid_images.extend(glob.glob(os.path.join(category_path, "*.tif")))

            for img_path in valid_images:
                if img_format == 'tiff':
                    image = tifffile.imread(img_path)
                else:
                    image = cv2.imread(img_path)

                if image is None:
                    print(f"Warning: Failed to load image {img_path}")
                    continue

                image = ensure_hwc(image)
                image = ensure_3ch(image)

                h, w = image.shape[:2]
                
                # Skip if image is too small
                if h < patch_size[0] or w < patch_size[1]:
                    print(f"Warning: Skipping validation image {img_path} - size ({h}x{w}) is smaller than patch size ({patch_size[0]}x{patch_size[1]})")
                    continue
                
                # Initialize full-size score map and count map for averaging
                full_score_map = np.zeros((h, w), dtype=np.float32)
                count_map = np.zeros((h, w), dtype=np.float32)
                
                # Calculate adaptive positions
                y_positions = calculate_positions(h, patch_size[0])
                x_positions = calculate_positions(w, patch_size[1])
                
                if y_positions is None or x_positions is None:
                    print(f"Warning: Image {img_path} is too small for patch size")
                    continue
                
                # Sliding window evaluation with adaptive positions
                for y in y_positions:
                    for x in x_positions:
                        # Extract patch
                        patch = image[y:y+patch_size[0], x:x+patch_size[1]]
                        
                        # Prepare input tensor
                        input_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                        input_tensor = input_tensor.unsqueeze(0).to(device)
                        
                        # Forward pass
                        output = model(input_tensor)
                        output_sm = torch.softmax(output, dim=1)
                        patch_score = output_sm[:, 1, :, :].squeeze().cpu().numpy()
                        
                        # Accumulate scores
                        full_score_map[y:y+patch_size[0], x:x+patch_size[1]] += patch_score
                        count_map[y:y+patch_size[0], x:x+patch_size[1]] += 1
                
                # Average overlapping regions
                anomaly_score_map = full_score_map / np.maximum(count_map, 1)
                max_score = anomaly_score_map.max()
                
                # For good images, no ground truth mask
                if category == 'good':
                    # Image-level: good images have label 0
                    all_scores.append(max_score)
                    all_labels.append(0.0)
                    
                    # Pixel-level: all pixels are normal (0)
                    pixel_scores.extend(anomaly_score_map.flatten())
                    pixel_labels.extend(np.zeros_like(anomaly_score_map).flatten())
                    
                else:  # bright_spots
                    # Load ground truth mask
                    filename = os.path.basename(img_path)
                    name_parts = filename.split('.')
                    mask_name = name_parts[0] + '_mask.' + name_parts[1]
                    mask_path = os.path.join(ground_truth_root, 'bright_spots', mask_name)
                    
                    if os.path.exists(mask_path):
                        if img_format == 'tiff':
                            gt_mask = tifffile.imread(mask_path)
                            # Ensure single channel
                            if len(gt_mask.shape) == 3:
                                gt_mask = gt_mask[:, :, 0]
                        else:
                            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Check mask size
                        mask_h, mask_w = gt_mask.shape[:2]
                        if mask_h != h or mask_w != w:
                            print(f"Warning: Ground truth mask size ({mask_h}x{mask_w}) doesn't match image size ({h}x{w}) for {mask_path}")
                            continue
                        
                        # Normalize mask to [0, 1]
                        gt_mask = gt_mask.astype(np.float32)
                        if gt_mask.max() > 1:
                            gt_mask = gt_mask / 255.0
                        
                        # Image-level AUROC
                        has_anomaly = 1.0 if np.sum(gt_mask) > 0 else 0.0
                        all_scores.append(max_score)
                        all_labels.append(has_anomaly)
                        
                        # Pixel-level AUROC
                        pixel_scores.extend(anomaly_score_map.flatten())
                        pixel_labels.extend(gt_mask.flatten())
    
    # Calculate AUROC
    image_auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores) if len(np.unique(pixel_labels)) > 1 else 0.0
    
    model.train()
    
    return image_auroc, pixel_auroc

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")
    
    # Set up device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    patch_size = (args.patch_size, args.patch_size)

    run_name = f'BgRemoval_lr{args.lr}_ep{args.epochs}_bs{args.bs}_{patch_size[0]}x{patch_size[1]}'
    
    # Single segmentation network for 3-channel input
    model_seg = SegmentationNetwork(in_channels=3, out_channels=2)
    model_seg.to(device)
    model_seg.apply(weights_init)
    
    optimizer = torch.optim.Adam(model_seg.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        [int(args.epochs*0.8), int(args.epochs*0.9)], 
        gamma=0.2, 
        last_epoch=-1
    )
    
    # Initialize loss function with alpha=0.75 for defect class balance
    criterion = FocalLoss(alpha=0.75, gamma=args.gamma_start)
    print(f"Using Focal Loss with alpha=0.75, gamma schedule: [{args.gamma_start}, {args.gamma_end}] (cosine)")
    
    psf_config_paths = None
    if args.defect_mode == 'psf':
        defects_dir = os.path.join(os.path.dirname(__file__), 'defects')
        psf_config_paths = [os.path.join(defects_dir, f'{t}.yaml') for t in args.psf_type]

    dataset = Dataset(
        training_path=args.training_dataset_path,
        patch_size=patch_size,
        num_defects_range=args.num_defects_range,
        img_format=args.img_format,
        cache_size=args.cache_size,
        defect_mode=args.defect_mode,
        psf_config_paths=psf_config_paths,
        psf_pool_size=args.psf_pool_size,
        partial_leak_scale=tuple(args.partial_leak_scale),
    )
    
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    print(f"Dataset size: {len(dataset)} samples per epoch")

    save_training_artifacts(args.checkpoint_path, dataset, psf_config_paths)

    num_batches = len(dataloader)
    
    for epoch in range(args.epochs):
        # Update focal loss gamma for current epoch
        current_gamma = get_focal_gamma(epoch, args.epochs, args.gamma_start, args.gamma_end, schedule='cosine')
        criterion.update_params(gamma=current_gamma)

        epoch_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):
            # Get three channel input and mask
            three_channel_input = sample_batched["three_channel_input"].to(device)
            target_mask = sample_batched["target_mask"].to(device)
            
            # Forward pass through segmentation network
            out_mask = model_seg(three_channel_input)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            # Calculate loss (always use the dynamically generated mask)
            loss = criterion(out_mask_sm, target_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print training progress
            if i_batch % 10 == 0 or i_batch == num_batches - 1:
                current_lr = get_lr(optimizer)
                progress = (i_batch + 1) / num_batches * 100
                print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                      f'Loss: {loss.item():.4e} - LR: {current_lr:.6f}', end='', flush=True)
        
        scheduler.step()
        
        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        print(f'\nEpoch [{epoch+1}/{args.epochs}] Summary - Avg Loss: {avg_loss:.4e} - Gamma: {current_gamma:.3f}', end='')
        
        # Evaluate on validation set if use_mask is True
        if args.use_mask == 'True':
            # Construct validation paths based on training path
            # training_dataset_path is like ./MVTec_AD_dataset/grid/train/good/
            # We need to go up to ./MVTec_AD_dataset/grid/
            dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(args.training_dataset_path)))
            valid_root = os.path.join(dataset_root, 'valid')
            ground_truth_root = os.path.join(dataset_root, 'ground_truth_valid')
            
            if os.path.exists(valid_root):
                image_auroc, pixel_auroc = evaluate_model(
                    model_seg, valid_root, ground_truth_root,
                    args.img_format, patch_size, device
                )
                print(f' - Image AUROC: {image_auroc:.4f} - Pixel AUROC: {pixel_auroc:.4f}')
            else:
                print(' - Validation data not found')
        else:
            print()  # Just newline
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_seg.state_dict(),
            'img_height': patch_size[0],
            'img_width': patch_size[1],
            'epoch': epoch,
            'seed': args.seed
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, f'{run_name}.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu_id', action='store', type=int, default=0, 
                        help='GPU ID to use. Set to -1 to use CPU')
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training (default: 128)')
    parser.add_argument('--num_defects_range', action='store', type=int, nargs=2, default=[3, 8],
                        help='Range of number of defects to generate [min, max]')
    parser.add_argument('--training_dataset_path', action='store', type=str, required=True,
                        help='Path to training dataset directory (e.g., ./MVTec_AD_dataset/grid/train/good/)')
    parser.add_argument('--img_format', action='store', type=str, choices=['png_jpg', 'tiff'], default='png_jpg',
                        help='Image format to use for training (png_jpg: PNG/JPG files, tiff: TIFF files)')
    parser.add_argument('--use_mask', type=str, choices=['True', 'False'], default='True',
                        help='Whether to use mask for training (True: normal training, False: no mask supervision)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--cache_size', type=int, default=0,
                        help='Number of images to cache in memory (0 = no cache)')
    parser.add_argument('--gamma_start', type=float, default=1.0,
                        help='Starting gamma value for focal loss (default: 1.0)')
    parser.add_argument('--gamma_end', type=float, default=3.0,
                        help='Ending gamma value for focal loss (default: 3.0)')
    parser.add_argument('--defect_mode', type=str, choices=['gaussian', 'psf'], default='gaussian',
                        help='Defect generation mode')
    parser.add_argument('--psf_type', type=str, nargs='+', default=None,
                        help='PSF config name(s) in defects/ (e.g., type1 type2)')
    parser.add_argument('--psf_pool_size', type=int, default=1000,
                        help='Number of pre-generated PSF defects per type (default: 1000)')
    parser.add_argument('--num_workers', type=int, default=7,
                        help='Number of DataLoader workers (default: 7)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches prefetched per worker (default: 2, increase for S3)')
    parser.add_argument('--partial_leak_scale', type=float, nargs=2, default=[0.2, 0.7],
                        metavar=('MIN', 'MAX'),
                        help='Range of intensity scale when a target-only defect leaks to a ref '
                             '(default: 0.2 0.7). The leak probability is fixed at 0.4 in dataloader.')

    args = parser.parse_args()
    
    train_on_device(args)

if __name__ == "__main__":
    main()