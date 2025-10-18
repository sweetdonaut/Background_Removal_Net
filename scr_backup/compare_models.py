"""
Compare inference results between line_negative and stripe_experiment models
with various synthetic defect types
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tifffile
from model import SegmentationNetwork
from inference import sliding_window_inference
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


def load_model(model_path, device):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = SegmentationNetwork(in_channels=3, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    patch_size = (checkpoint['img_height'], checkpoint['img_width'])
    image_type = checkpoint.get('image_type', 'strip')
    
    return model, patch_size, image_type


def create_point_defect(image, position, size, intensity):
    """Create a square point defect with gaussian profile"""
    h, w = image.shape
    y, x = position
    
    # Create gaussian kernel
    if isinstance(size, int):
        kernel_size = size
        sigma = size / 3.0
    else:
        kernel_size = size[0]
        sigma = size[0] / 3.0
    
    # Create coordinate grid for the defect
    half_size = kernel_size // 2
    y_range = np.arange(max(0, y - half_size), min(h, y + half_size + 1))
    x_range = np.arange(max(0, x - half_size), min(w, x + half_size + 1))
    
    if len(y_range) == 0 or len(x_range) == 0:
        return image
    
    Y, X = np.meshgrid(y_range - y, x_range - x, indexing='ij')
    
    # Gaussian formula
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Apply defect
    result = image.copy()
    result[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] += gaussian * intensity
    return np.clip(result, 0, 255)


def create_line_defect(image, orientation, position, width, intensity):
    """Create a simple line defect"""
    h, w = image.shape
    result = image.copy()
    
    if orientation == 'horizontal':
        y = position
        if 0 <= y < h - width:
            result[y:y+width, :] += intensity
    else:  # vertical
        x = position
        if 0 <= x < w - width:
            result[:, x:x+width] += intensity
    
    return np.clip(result, 0, 255)


def create_gaussian_line_defect(image, orientation, position, width, intensity):
    """Create a line defect with gaussian profile across width"""
    h, w = image.shape
    result = image.copy()
    
    # Create gaussian profile across the width
    sigma = width / 2.0
    profile = np.exp(-np.linspace(-width, width, width*2+1)**2 / (2 * sigma**2))
    # Take center part
    profile = profile[width//2:width//2+width]
    profile = profile / np.max(profile)  # Normalize
    
    if orientation == 'horizontal':
        y = position
        if 0 <= y < h - width:
            for i in range(width):
                if y + i < h:
                    result[y+i, :] += intensity * profile[i]
    else:  # vertical
        x = position
        if 0 <= x < w - width:
            for i in range(width):
                if x + i < w:
                    result[:, x+i] += intensity * profile[i]
    
    return np.clip(result, 0, 255)


def create_noisy_line_defect(image, orientation, position, width, intensity, noise_level=10):
    """Create a line defect with brightness perturbation"""
    h, w = image.shape
    result = image.copy()
    
    if orientation == 'horizontal':
        y = position
        if 0 <= y < h - width:
            # Create noise along the line
            noise = np.random.randn(width, w) * noise_level
            for i in range(width):
                if y + i < h:
                    result[y+i, :] += intensity + noise[i, :]
    else:  # vertical
        x = position
        if 0 <= x < w - width:
            # Create noise along the line
            noise = np.random.randn(h, width) * noise_level
            for i in range(width):
                if x + i < w:
                    result[:, x+i] += intensity + noise[:, i]
    
    return np.clip(result, 0, 255)


def create_test_images(base_image_path, output_dir):
    """Create test images with various defect types"""
    
    # Load base image
    if os.path.exists(base_image_path):
        base_image = tifffile.imread(base_image_path)
        if base_image.shape[0] == 3:
            base_image = np.transpose(base_image, (1, 2, 0))
    else:
        # Create synthetic base image
        print("Creating synthetic base image...")
        base_image = np.ones((976, 176, 3), dtype=np.float32) * 128
        # Add some texture
        for i in range(3):
            base_image[:, :, i] += np.random.randn(976, 176) * 5
    
    h, w = base_image.shape[:2]
    test_images = []
    
    # 1. Point defects (3x3, 4x4, 5x5, 6x6)
    print("Creating point defects...")
    for size in [3, 4, 5, 6]:
        # Create target with defects
        target = base_image[:, :, 0].copy()
        ref1 = base_image[:, :, 1].copy()
        ref2 = base_image[:, :, 2].copy()
        
        # Add multiple defects with proper target-only logic
        num_defects = 5
        for i in range(num_defects):
            y = np.random.randint(size, h - size)
            x = np.random.randint(size, w - size)
            intensity = np.random.choice([-60, 60])
            
            # Always add to target
            target = create_point_defect(target, (y, x), size, intensity)
            
            # Ensure at least 2 defects are target-only (not in refs)
            if i < 2:
                # These are target-only defects
                pass
            else:
                # These might be shared with refs
                if np.random.rand() > 0.3:
                    ref1 = create_point_defect(ref1, (y, x), size, intensity * 0.6)
                if np.random.rand() > 0.3:
                    ref2 = create_point_defect(ref2, (y, x), size, intensity * 0.6)
        
        test_images.append({
            'name': f'point_{size}x{size}',
            'target': target,
            'ref1': ref1,
            'ref2': ref2,
            'type': 'point'
        })
    
    # 2. Simple line defects (width 1, 2, 3, 4)
    print("Creating simple line defects...")
    for width in [1, 2, 3, 4]:
        target = base_image[:, :, 0].copy()
        ref1 = base_image[:, :, 1].copy()
        ref2 = base_image[:, :, 2].copy()
        
        # Add horizontal and vertical lines (should NOT be detected as defects)
        for _ in range(3):
            if np.random.rand() > 0.5:
                # Horizontal
                y = np.random.randint(0, h - width)
                intensity = np.random.choice([-40, 40])
                # Add same intensity to all channels (slightly weaker for refs)
                target = create_line_defect(target, 'horizontal', y, width, intensity)
                ref1 = create_line_defect(ref1, 'horizontal', y, width, intensity * 0.6)
                ref2 = create_line_defect(ref2, 'horizontal', y, width, intensity * 0.6)
            else:
                # Vertical
                x = np.random.randint(0, w - width)
                intensity = np.random.choice([-50, 50])
                target = create_line_defect(target, 'vertical', x, width, intensity)
                ref1 = create_line_defect(ref1, 'vertical', x, width, intensity * 0.6)
                ref2 = create_line_defect(ref2, 'vertical', x, width, intensity * 0.6)
        
        # Also add some real point defects to test if lines interfere
        for _ in range(2):
            y = np.random.randint(10, h - 10)
            x = np.random.randint(10, w - 10)
            intensity = np.random.choice([-60, 60])
            # These are target-only defects
            target = create_point_defect(target, (y, x), 4, intensity)
        
        test_images.append({
            'name': f'line_width{width}',
            'target': target,
            'ref1': ref1,
            'ref2': ref2,
            'type': 'line'
        })
    
    # 3. Gaussian line defects
    print("Creating gaussian line defects...")
    for width in [2, 3, 4]:
        target = base_image[:, :, 0].copy()
        ref1 = base_image[:, :, 1].copy()
        ref2 = base_image[:, :, 2].copy()
        
        for _ in range(3):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, h - width)
                intensity = np.random.choice([-50, 50])
                target = create_gaussian_line_defect(target, 'horizontal', y, width, intensity)
                ref1 = create_gaussian_line_defect(ref1, 'horizontal', y, width, intensity * 0.6)
                ref2 = create_gaussian_line_defect(ref2, 'horizontal', y, width, intensity * 0.6)
            else:
                x = np.random.randint(0, w - width)
                intensity = np.random.choice([-50, 50])
                target = create_gaussian_line_defect(target, 'vertical', x, width, intensity)
                ref1 = create_gaussian_line_defect(ref1, 'vertical', x, width, intensity * 0.6)
                ref2 = create_gaussian_line_defect(ref2, 'vertical', x, width, intensity * 0.6)
        
        test_images.append({
            'name': f'gaussian_line_width{width}',
            'target': target,
            'ref1': ref1,
            'ref2': ref2,
            'type': 'gaussian_line'
        })
    
    # 4. Noisy line defects
    print("Creating noisy line defects...")
    for width in [2, 3, 4]:
        target = base_image[:, :, 0].copy()
        ref1 = base_image[:, :, 1].copy()
        ref2 = base_image[:, :, 2].copy()
        
        for _ in range(3):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, h - width)
                intensity = np.random.choice([-40, 40])
                target = create_noisy_line_defect(target, 'horizontal', y, width, intensity, noise_level=10)
                ref1 = create_noisy_line_defect(ref1, 'horizontal', y, width, intensity * 0.6, noise_level=5)
                ref2 = create_noisy_line_defect(ref2, 'horizontal', y, width, intensity * 0.6, noise_level=5)
            else:
                x = np.random.randint(0, w - width)
                intensity = np.random.choice([-40, 40])
                target = create_noisy_line_defect(target, 'vertical', x, width, intensity, noise_level=10)
                ref1 = create_noisy_line_defect(ref1, 'vertical', x, width, intensity * 0.6, noise_level=5)
                ref2 = create_noisy_line_defect(ref2, 'vertical', x, width, intensity * 0.6, noise_level=5)
        
        test_images.append({
            'name': f'noisy_line_width{width}',
            'target': target,
            'ref1': ref1,
            'ref2': ref2,
            'type': 'noisy_line'
        })
    
    # Save test images
    os.makedirs(output_dir, exist_ok=True)
    for img_data in test_images:
        # Stack channels
        stacked = np.stack([img_data['target'], img_data['ref1'], img_data['ref2']], axis=0)
        save_path = os.path.join(output_dir, f"{img_data['name']}.tiff")
        tifffile.imwrite(save_path, stacked.astype(np.float32))
        print(f"  Saved: {img_data['name']}.tiff")
    
    return test_images


def compare_models(model1_path, model2_path, test_images, output_dir):
    """Compare two models on test images"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("\nLoading models...")
    model1, patch_size1, image_type1 = load_model(model1_path, device)
    model2, patch_size2, image_type2 = load_model(model2_path, device)
    
    print(f"Model 1: {os.path.basename(model1_path)}")
    print(f"Model 2: {os.path.basename(model2_path)}")
    
    # Process each test image
    results = []
    
    for img_data in test_images:
        print(f"\nProcessing: {img_data['name']}")
        
        # Prepare input - transpose to HWC format for inference
        three_channel = np.stack([img_data['target'], img_data['ref1'], img_data['ref2']], axis=0)
        three_channel = np.transpose(three_channel, (1, 2, 0))  # CHW to HWC
        # Don't normalize here - sliding_window_inference will do it
        
        # Model 1 inference
        heatmap1, _ = sliding_window_inference(
            three_channel, model1, patch_size1, device, image_type1
        )
        
        # Model 2 inference
        heatmap2, _ = sliding_window_inference(
            three_channel, model2, patch_size2, device, image_type2
        )
        
        results.append({
            'name': img_data['name'],
            'type': img_data['type'],
            'target': img_data['target'],
            'ref1': img_data['ref1'],
            'ref2': img_data['ref2'],
            'heatmap1': heatmap1,
            'heatmap2': heatmap2
        })
    
    # Visualize results
    visualize_comparison(results, model1_path, model2_path, output_dir)
    
    return results


def visualize_comparison(results, model1_path, model2_path, output_dir):
    """Create comparison visualizations - one image per result"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual plot for each result
    for result in results:
        # Create figure with layout similar to inference.py but with 8 subplots
        fig = plt.figure(figsize=(11, 6), dpi=200)
        gs = gridspec.GridSpec(1, 8, figure=fig)
        
        # Target
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(result['target'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax1.set_title('Target')
        ax1.axis('off')
        
        # Ref1
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(result['ref1'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax2.set_title('Ref1')
        ax2.axis('off')
        
        # Ref2
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(result['ref2'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax3.set_title('Ref2')
        ax3.axis('off')
        
        # Target - Ref1
        diff1 = np.abs(result['target'] - result['ref1'])
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(diff1, cmap='hot', vmin=0, vmax=np.max(diff1) if np.max(diff1) > 0 else 1, interpolation='nearest')
        ax4.set_title('Target - Ref1')
        ax4.axis('off')
        
        # Target - Ref2
        diff2 = np.abs(result['target'] - result['ref2'])
        ax5 = fig.add_subplot(gs[4])
        ax5.imshow(diff2, cmap='hot', vmin=0, vmax=np.max(diff2) if np.max(diff2) > 0 else 1, interpolation='nearest')
        ax5.set_title('Target - Ref2')
        ax5.axis('off')
        
        # Double Detection (min of two differences)
        double_detection = np.minimum(diff1, diff2)
        ax6 = fig.add_subplot(gs[5])
        ax6.imshow(double_detection, cmap='hot', vmin=0, vmax=np.max(double_detection) if np.max(double_detection) > 0 else 1, interpolation='nearest')
        ax6.set_title('Double Detection')
        ax6.axis('off')
        
        # Model 1 heatmap
        ax7 = fig.add_subplot(gs[6])
        ax7.imshow(result['heatmap1'], cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        ax7.set_title('Heatmap\n(stripe_exp)')
        ax7.axis('off')
        
        # Model 2 heatmap
        ax8 = fig.add_subplot(gs[7])
        ax8.imshow(result['heatmap2'], cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        ax8.set_title('Heatmap\n(line_neg)')
        ax8.axis('off')
        
        plt.suptitle(f"{result['name']}", fontsize=14)
        plt.tight_layout()
        
        # Save with descriptive name
        save_path = os.path.join(output_dir, f"{result['name']}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # Create summary statistics
    create_summary_stats(results, model1_path, model2_path, output_dir)


def create_summary_stats(results, model1_path, model2_path, output_dir):
    """Create summary statistics comparing models"""
    
    stats = {
        'point': {'model1': [], 'model2': []},
        'line': {'model1': [], 'model2': []},
        'gaussian_line': {'model1': [], 'model2': []},
        'noisy_line': {'model1': [], 'model2': []}
    }
    
    for result in results:
        defect_type = result['type']
        # Calculate mean activation in heatmap
        stats[defect_type]['model1'].append(np.mean(result['heatmap1']))
        stats[defect_type]['model2'].append(np.mean(result['heatmap2']))
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    defect_types = list(stats.keys())
    x = np.arange(len(defect_types))
    width = 0.35
    
    model1_means = [np.mean(stats[dt]['model1']) if stats[dt]['model1'] else 0 for dt in defect_types]
    model2_means = [np.mean(stats[dt]['model2']) if stats[dt]['model2'] else 0 for dt in defect_types]
    
    bars1 = ax.bar(x - width/2, model1_means, width, label='stripe_experiment')
    bars2 = ax.bar(x + width/2, model2_means, width, label='line_negative')
    
    ax.set_xlabel('Defect Type')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Model Comparison: Mean Activation by Defect Type')
    ax.set_xticks(x)
    ax.set_xticklabels([dt.replace('_', '\n') for dt in defect_types])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'summary_statistics.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Print statistics
    print("\n=== Summary Statistics ===")
    for dt in defect_types:
        if stats[dt]['model1']:
            print(f"\n{dt.replace('_', ' ').title()}:")
            print(f"  Model 1 (stripe_exp): {np.mean(stats[dt]['model1']):.4f}")
            print(f"  Model 2 (line_neg):   {np.mean(stats[dt]['model2']):.4f}")


def main():
    # Paths
    model1_path = "./checkpoints/line_negative/BgRemoval_lr0.001_ep30_bs16_128x128_strip.pth"
    model2_path = "./checkpoints/line_negative/BgRemoval_lr0.001_ep50_bs16_128x128_strip.pth"
    base_image_path = "./MVTec_AD_dataset/grid_stripe/train/good/250.tiff"
    test_images_dir = "./output/test_images"
    output_dir = "./output/model_comparison"
    
    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"Model 1 not found: {model1_path}")
        print("Please train the stripe_experiment model first")
        return
    
    if not os.path.exists(model2_path):
        print(f"Model 2 not found: {model2_path}")
        print("Please train the line_negative model first")
        return
    
    # Create test images
    print("Creating test images...")
    test_images = create_test_images(base_image_path, test_images_dir)
    
    # Compare models
    print("\nComparing models...")
    results = compare_models(model1_path, model2_path, test_images, output_dir)
    
    print(f"\n=== Comparison Complete ===")
    print(f"Test images saved to: {test_images_dir}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()