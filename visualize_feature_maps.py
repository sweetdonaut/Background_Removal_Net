import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
import tifffile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SegmentationNetwork
from dataloader import calculate_positions


class FeatureExtractor:
    """Extract intermediate feature maps from model"""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to extract features"""
        # Encoder blocks
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook

        # Register hooks for encoder blocks
        self.hooks.append(self.model.encoder.block1.register_forward_hook(get_hook('encoder_block1')))
        self.hooks.append(self.model.encoder.block2.register_forward_hook(get_hook('encoder_block2')))
        self.hooks.append(self.model.encoder.block3.register_forward_hook(get_hook('encoder_block3')))
        self.hooks.append(self.model.encoder.block4.register_forward_hook(get_hook('encoder_block4')))
        self.hooks.append(self.model.encoder.block5.register_forward_hook(get_hook('encoder_block5')))
        self.hooks.append(self.model.encoder.block6.register_forward_hook(get_hook('encoder_block6')))

        # Register hooks for decoder blocks
        self.hooks.append(self.model.decoder.db_b.register_forward_hook(get_hook('decoder_block_b')))
        self.hooks.append(self.model.decoder.db1.register_forward_hook(get_hook('decoder_block1')))
        self.hooks.append(self.model.decoder.db2.register_forward_hook(get_hook('decoder_block2')))
        self.hooks.append(self.model.decoder.db3.register_forward_hook(get_hook('decoder_block3')))
        self.hooks.append(self.model.decoder.db4.register_forward_hook(get_hook('decoder_block4')))

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_features(self):
        """Return extracted features"""
        return self.features


def reduce_channels(feature_map, method='mean'):
    """
    Reduce multi-channel feature map to single channel for visualization

    Args:
        feature_map: (C, H, W) numpy array
        method: 'mean', 'max', 'std', or 'top_variance'

    Returns:
        (H, W) numpy array
    """
    if method == 'mean':
        # Average across all channels
        return np.mean(feature_map, axis=0)
    elif method == 'max':
        # Max across all channels
        return np.max(feature_map, axis=0)
    elif method == 'std':
        # Standard deviation across channels (shows variation)
        return np.std(feature_map, axis=0)
    elif method == 'top_variance':
        # Select the channel with highest variance (most informative)
        variances = np.var(feature_map, axis=(1, 2))
        top_channel = np.argmax(variances)
        return feature_map[top_channel]
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_all_features(input_patch, features, output_heatmap, output_path, reduction_method='mean'):
    """
    Visualize input, all feature maps, and output

    Args:
        input_patch: (3, H, W) input image
        features: dict of feature maps
        output_heatmap: (H, W) output heatmap
        output_path: path to save visualization
        reduction_method: method to reduce channels
    """
    # Extract channels from input
    target = input_patch[0]
    ref1 = input_patch[1]
    ref2 = input_patch[2]

    # Prepare feature maps for visualization
    encoder_features = []
    decoder_features = []

    # Encoder features (6 blocks, progressively smaller)
    for i in range(1, 7):
        name = f'encoder_block{i}'
        if name in features:
            feat = features[name].squeeze().cpu().numpy()  # (C, H, W)
            feat_reduced = reduce_channels(feat, method=reduction_method)
            encoder_features.append((name, feat_reduced, feat.shape))

    # Decoder features (5 blocks, progressively larger)
    decoder_names = ['decoder_block_b', 'decoder_block1', 'decoder_block2', 'decoder_block3', 'decoder_block4']
    for name in decoder_names:
        if name in features:
            feat = features[name].squeeze().cpu().numpy()  # (C, H, W)
            feat_reduced = reduce_channels(feat, method=reduction_method)
            decoder_features.append((name, feat_reduced, feat.shape))

    # Create figure with 3 rows
    # Row 1: Input (3 channels)
    # Row 2: Encoder (6 blocks)
    # Row 3: Decoder (5 blocks) + Output (1)

    fig = plt.figure(figsize=(18, 10), dpi=150)
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.5)

    # Row 1: Input channels (3 subplots, centered)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(target, cmap='gray')
    ax.set_title('Input: Target', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(ref1, cmap='gray')
    ax.set_title('Input: Ref1', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(ref2, cmap='gray')
    ax.set_title('Input: Ref2', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Row 2: Encoder features (6 blocks)
    for idx, (name, feat, shape) in enumerate(encoder_features):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(feat, cmap='viridis')
        ax.set_title(f'Enc{idx+1}\n{shape[0]}ch\n{shape[1]}x{shape[2]}', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: Decoder features (5 blocks) + Output
    for idx, (name, feat, shape) in enumerate(decoder_features):
        ax = fig.add_subplot(gs[2, idx])
        im = ax.imshow(feat, cmap='viridis')
        ax.set_title(f'Dec{idx+1}\n{shape[0]}ch\n{shape[1]}x{shape[2]}', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Output heatmap (last position in row 3)
    ax = fig.add_subplot(gs[2, 5])
    im = ax.imshow(output_heatmap, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Output\nHeatmap', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add reduction method info
    fig.suptitle(f'Feature Map Visualization (Channel Reduction: {reduction_method})',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def visualize_multi_methods(input_patch, features, output_heatmap, output_path):
    """
    Visualize using multiple channel reduction methods for comparison
    """
    methods = ['mean', 'max', 'std', 'top_variance']

    for method in methods:
        output_file = output_path.replace('.png', f'_{method}.png')
        visualize_all_features(input_patch, features, output_heatmap, output_file, method)


def extract_patch_from_image(image, patch_idx, patch_size, image_type):
    """
    Extract a specific patch from the image

    Args:
        image: (H, W, 3) input image
        patch_idx: (y_idx, x_idx) patch indices
        patch_size: (patch_h, patch_w)
        image_type: 'strip' or 'square'

    Returns:
        patch: (patch_h, patch_w, 3) extracted patch
        position: (y, x) absolute position
    """
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size

    # Calculate positions (same as training/inference)
    if image_type == 'strip':
        y_positions = calculate_positions(h, patch_h, min_patches=9)
        x_positions = calculate_positions(w, patch_w)
    else:  # square
        y_positions = calculate_positions(h, patch_h, min_patches=4)
        x_positions = calculate_positions(w, patch_w, min_patches=4)

    y_idx, x_idx = patch_idx

    if y_idx >= len(y_positions) or x_idx >= len(x_positions):
        raise ValueError(f"Patch index ({y_idx}, {x_idx}) out of range. "
                        f"Available: Y={len(y_positions)}, X={len(x_positions)}")

    y = y_positions[y_idx]
    x = x_positions[x_idx]

    # Extract patch
    patch = image[y:y+patch_h, x:x+patch_w]

    return patch, (y, x), y_positions, x_positions


def main():
    parser = argparse.ArgumentParser(description='Visualize UNet Feature Maps')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./feature_maps',
                       help='Directory to save visualizations')
    parser.add_argument('--patch_idx', type=int, nargs=2, default=[0, 0],
                       help='Patch index (y_idx x_idx), e.g., 0 0 for first patch')
    parser.add_argument('--img_format', type=str, choices=['png_jpg', 'tiff'], default='tiff',
                       help='Image format')
    parser.add_argument('--image_type', type=str, choices=['strip', 'square'], default='strip',
                       help='Image type')
    parser.add_argument('--reduction_method', type=str,
                       choices=['mean', 'max', 'std', 'top_variance', 'all'],
                       default='all',
                       help='Channel reduction method (use "all" to generate all methods)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (-1 for CPU)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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

    patch_height = checkpoint['img_height']
    patch_width = checkpoint['img_width']
    patch_size = (patch_height, patch_width)

    if 'image_type' in checkpoint:
        image_type = checkpoint['image_type']
    else:
        image_type = args.image_type

    print(f"Model patch size: {patch_size}")
    print(f"Image type: {image_type}")

    # Initialize model
    model = SegmentationNetwork(in_channels=3, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Setup feature extractor
    extractor = FeatureExtractor(model)
    extractor.register_hooks()

    # Load image
    print(f"Loading image: {args.image_path}")
    if args.img_format == 'tiff':
        image = tifffile.imread(args.image_path)
    else:
        image = cv2.imread(args.image_path)

    # Handle channel format
    if image_type == 'strip' and (image.shape[0] == 3 or image.shape[0] == 4):
        image = np.transpose(image, (1, 2, 0))

    # Keep only first 3 channels
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Normalize to 0-255
    image = image.astype(np.float32)
    img_min = image.min()
    img_max = image.max()
    if img_min < 0 or img_max > 255:
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image = np.zeros_like(image)

    # Extract patch
    print(f"Extracting patch at index: {args.patch_idx}")
    patch, position, y_positions, x_positions = extract_patch_from_image(
        image, args.patch_idx, patch_size, image_type
    )
    print(f"Patch position: {position}")
    print(f"Available Y positions: {len(y_positions)} - {y_positions}")
    print(f"Available X positions: {len(x_positions)} - {x_positions}")

    # Prepare input
    target = patch[:, :, 0]
    ref1 = patch[:, :, 1]
    ref2 = patch[:, :, 2]

    three_channel = np.stack([target, ref1, ref2], axis=0)
    three_channel_tensor = torch.from_numpy(three_channel).float() / 255.0
    input_tensor = three_channel_tensor.unsqueeze(0).to(device)

    # Forward pass to extract features
    print("Extracting features...")
    with torch.no_grad():
        output = model(input_tensor)
        output_sm = F.softmax(output, dim=1)
        output_heatmap = output_sm[:, 1, :, :].squeeze().cpu().numpy()

    # Get extracted features
    features = extractor.get_features()
    print(f"Extracted {len(features)} feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    # Generate visualization
    filename = os.path.basename(args.image_path).split('.')[0]
    output_path = os.path.join(args.output_dir,
                               f'{filename}_patch_{args.patch_idx[0]}_{args.patch_idx[1]}.png')

    if args.reduction_method == 'all':
        print("Generating visualizations with all reduction methods...")
        visualize_multi_methods(three_channel, features, output_heatmap, output_path)
    else:
        print(f"Generating visualization with reduction method: {args.reduction_method}")
        visualize_all_features(three_channel, features, output_heatmap, output_path,
                              args.reduction_method)

    # Cleanup
    extractor.remove_hooks()

    print("\nVisualization completed!")


if __name__ == "__main__":
    main()
