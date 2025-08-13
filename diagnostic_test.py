import torch
import numpy as np
import cv2
import tifffile
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SegmentationNetwork
from inference import sliding_window_inference
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def load_model(model_path, device):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = SegmentationNetwork(in_channels=3, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    patch_size = (checkpoint['img_height'], checkpoint['img_width'])
    image_type = checkpoint.get('image_type', 'strip')
    
    return model, patch_size, image_type


def create_test_images(base_image_path, output_dir):
    """Create various test images to diagnose the problem"""
    
    # Load base image
    base_image = tifffile.imread(base_image_path)
    print(f"Base image shape: {base_image.shape}")
    
    # If CHW format, convert to HWC
    if base_image.shape[0] == 3:
        base_image = np.transpose(base_image, (1, 2, 0))
    
    h, w = base_image.shape[:2]
    
    # Extract the original channels
    target_orig = base_image[:, :, 0]
    ref1_orig = base_image[:, :, 1]
    ref2_orig = base_image[:, :, 2]
    
    test_cases = []
    
    # Test 1: Original image (baseline)
    test_cases.append({
        'name': '1_original',
        'description': 'Original test image',
        'image': base_image.copy()
    })
    
    # Test 2: Three channels identical (should have zero response)
    gray = target_orig.copy()
    identical = np.stack([gray, gray, gray], axis=-1)
    test_cases.append({
        'name': '2_identical_channels',
        'description': 'All three channels identical - expect NO response',
        'image': identical
    })
    
    # Test 3: Add global brightness difference
    target = gray.copy()
    ref1 = gray * 0.98  # 2% darker
    ref2 = gray * 1.02  # 2% brighter
    test_cases.append({
        'name': '3_global_brightness_diff',
        'description': 'Global brightness difference (±2%)',
        'image': np.stack([target, ref1, ref2], axis=-1)
    })
    
    # Test 4: Stronger global brightness difference
    target = gray.copy()
    ref1 = gray * 0.95  # 5% darker
    ref2 = gray * 1.05  # 5% brighter
    test_cases.append({
        'name': '4_strong_brightness_diff',
        'description': 'Strong brightness difference (±5%)',
        'image': np.stack([target, ref1, ref2], axis=-1)
    })
    
    # Test 5: Add noise to refs (simulate camera noise)
    target = gray.copy()
    ref1 = gray + np.random.randn(h, w) * 2
    ref2 = gray + np.random.randn(h, w) * 2
    test_cases.append({
        'name': '5_noise_in_refs',
        'description': 'Different noise in ref channels',
        'image': np.stack([target, ref1, ref2], axis=-1)
    })
    
    # Test 6: Add gaussian dots to target only (should detect these)
    target = gray.copy()
    # Add 3 gaussian dots with realistic sizes (matching training data)
    for _ in range(3):
        cx = np.random.randint(50, w-50)
        cy = np.random.randint(50, h-50)
        
        # Use realistic defect sizes: 3x3 or 3x5
        if np.random.rand() > 0.5:
            # 3x3 defect
            sigma = 1.3  # Same as training
            yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
            gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        else:
            # 3x5 defect
            sigma_x, sigma_y = 1.5, 1.0  # Same as training
            yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
            gaussian = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
        
        # Use realistic intensity (same as training but ensure no overflow)
        intensity = np.random.choice([60, 80])
        target = target + gaussian * intensity
    
    # Clip to valid range
    target = np.clip(target, 0, 255)
    
    test_cases.append({
        'name': '6_dots_in_target',
        'description': 'Realistic gaussian dots (3x3, 3x5) in target - SHOULD detect',
        'image': np.stack([target, gray, gray], axis=-1)
    })
    
    # Test 7: Add edge enhancement (simulate what happens at structure edges)
    target = gray.copy()
    ref1 = gray.copy()
    ref2 = gray.copy()
    
    # Detect edges
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = gaussian_filter(edges.astype(np.float32), 1.0)
    
    # Normalize edges to 0-1 range
    if edges.max() > 0:
        edges = edges / edges.max()
    
    # Add subtle edge differences (realistic values)
    edge_intensity = 15  # Much more reasonable
    target = target + edges * edge_intensity
    ref1 = ref1 - edges * edge_intensity * 0.3
    ref2 = ref2 - edges * edge_intensity * 0.3
    
    # Clip to valid range
    target = np.clip(target, 0, 255)
    ref1 = np.clip(ref1, 0, 255)
    ref2 = np.clip(ref2, 0, 255)
    
    test_cases.append({
        'name': '7_edge_enhancement',
        'description': 'Enhanced edges in target - should NOT detect as defects',
        'image': np.stack([target, ref1, ref2], axis=-1)
    })
    
    # Test 8: Add horizontal stripe pattern
    target = gray.copy()
    stripe = np.zeros_like(gray)
    stripe[::10, :] = 20  # Horizontal lines every 10 pixels
    target_with_stripe = target + stripe
    
    test_cases.append({
        'name': '8_horizontal_stripes',
        'description': 'Horizontal stripes in target - should NOT detect',
        'image': np.stack([target_with_stripe, gray, gray], axis=-1)
    })
    
    # Test 9: Mixed - dots + noise
    target = gray.copy()
    ref1 = gray + np.random.randn(h, w) * 1
    ref2 = gray + np.random.randn(h, w) * 1
    
    # Add realistic defects to target (3x3 or 3x5)
    for _ in range(2):
        cx = np.random.randint(50, w-50)
        cy = np.random.randint(50, h-50)
        
        # Realistic 3x3 defect
        sigma = 1.3
        yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        target = target + gaussian * 80  # Reasonable intensity
    
    # Clip all to valid range
    target = np.clip(target, 0, 255)
    ref1 = np.clip(ref1, 0, 255)
    ref2 = np.clip(ref2, 0, 255)
    
    test_cases.append({
        'name': '9_dots_with_noise',
        'description': 'Realistic dots + noise - should detect ONLY dots',
        'image': np.stack([target, ref1, ref2], axis=-1)
    })
    
    # Save test images
    os.makedirs(output_dir, exist_ok=True)
    for test in test_cases:
        # Convert back to CHW for saving
        image_chw = np.transpose(test['image'], (2, 0, 1))
        tifffile.imwrite(
            os.path.join(output_dir, f"{test['name']}.tiff"),
            image_chw.astype(np.float32)
        )
    
    return test_cases


def run_diagnostic_inference(model, test_cases, patch_size, image_type, device, output_dir):
    """Run inference on all test cases and analyze results"""
    
    results = []
    
    for test in test_cases:
        print(f"\nProcessing: {test['name']}")
        print(f"Description: {test['description']}")
        
        # Run inference
        image = test['image']
        heatmap, _ = sliding_window_inference(image, model, patch_size, device, image_type)
        
        # Analyze results
        max_response = np.max(heatmap)
        mean_response = np.mean(heatmap)
        std_response = np.std(heatmap)
        high_response_ratio = np.sum(heatmap > 0.1) / heatmap.size  # % of pixels > 0.1
        
        result = {
            'name': test['name'],
            'description': test['description'],
            'max_response': max_response,
            'mean_response': mean_response,
            'std_response': std_response,
            'high_response_ratio': high_response_ratio,
            'heatmap': heatmap
        }
        results.append(result)
        
        print(f"  Max response: {max_response:.4f}")
        print(f"  Mean response: {mean_response:.6f}")
        print(f"  Std response: {std_response:.6f}")
        print(f"  High response ratio: {high_response_ratio:.4%}")
        
        # Save visualization
        visualize_result(test['image'], heatmap, test['name'], test['description'], 
                        os.path.join(output_dir, f"{test['name']}_viz.png"))
    
    # Create summary comparison
    create_summary_plot(results, os.path.join(output_dir, "summary.png"))
    
    return results


def visualize_result(image, heatmap, name, description, output_path):
    """Visualize a single test result"""
    
    fig = plt.figure(figsize=(15, 4))
    
    # Extract channels
    target = image[:, :, 0]
    ref1 = image[:, :, 1]
    ref2 = image[:, :, 2]
    
    # Check data range
    print(f"  Image value range - Target: [{np.min(target):.1f}, {np.max(target):.1f}], "
          f"Ref1: [{np.min(ref1):.1f}, {np.max(ref1):.1f}], "
          f"Ref2: [{np.min(ref2):.1f}, {np.max(ref2):.1f}]")
    
    # Compute differences
    diff1 = np.abs(target.astype(np.float32) - ref1.astype(np.float32))
    diff2 = np.abs(target.astype(np.float32) - ref2.astype(np.float32))
    
    # Create subplots
    ax1 = plt.subplot(1, 6, 1)
    im1 = ax1.imshow(target, cmap='gray')
    ax1.set_title('Target')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(1, 6, 2)
    im2 = ax2.imshow(ref1, cmap='gray')
    ax2.set_title('Ref1')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(1, 6, 3)
    im3 = ax3.imshow(ref2, cmap='gray')
    ax3.set_title('Ref2')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(1, 6, 4)
    im4 = ax4.imshow(diff1, cmap='hot')
    ax4.set_title(f'|Target - Ref1|\nMax: {np.max(diff1):.1f}')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(1, 6, 5)
    im5 = ax5.imshow(diff2, cmap='hot')
    ax5.set_title(f'|Target - Ref2|\nMax: {np.max(diff2):.1f}')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = plt.subplot(1, 6, 6)
    im6 = ax6.imshow(heatmap, cmap='hot')
    ax6.set_title(f'Model Output\nMax: {np.max(heatmap):.3f}')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    plt.suptitle(f"{name}: {description}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(results, output_path):
    """Create a summary comparison of all test results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        if idx >= 9:
            break
        
        ax = axes[idx]
        im = ax.imshow(result['heatmap'], cmap='hot', vmin=0, vmax=0.1)
        ax.set_title(f"{result['name']}\nMax: {result['max_response']:.3f}, "
                    f"Mean: {result['mean_response']:.4f}", fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle("Diagnostic Test Results - Heatmap Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Settings
    model_path = "./checkpoints/improved_edge_neg/BgRemoval_lr0.0001_ep30_bs16_128x128_strip.pth"
    base_image_path = "./MVTec_AD_dataset/grid_stripe/test/good/260.tiff"  # Normal image
    test_output_dir = "./output/diagnostic_test"
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, patch_size, image_type = load_model(model_path, device)
    print(f"Model loaded. Patch size: {patch_size}, Image type: {image_type}")
    
    # Create test images
    print("\nCreating test images...")
    test_cases = create_test_images(base_image_path, os.path.join(test_output_dir, "test_images"))
    
    # Run diagnostic inference
    print("\nRunning diagnostic inference...")
    results = run_diagnostic_inference(
        model, test_cases, patch_size, image_type, device, test_output_dir
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("\nKey Observations:")
    
    # Check if identical channels still produce response
    identical_result = next(r for r in results if '2_identical' in r['name'])
    if identical_result['max_response'] > 0.01:
        print(f"⚠️  PROBLEM: Model responds even when channels are identical!")
        print(f"   Max response: {identical_result['max_response']:.4f}")
    
    # Check edge enhancement response
    edge_result = next(r for r in results if '7_edge' in r['name'])
    if edge_result['max_response'] > 0.05:
        print(f"⚠️  PROBLEM: Model strongly responds to edge enhancement!")
        print(f"   Max response: {edge_result['max_response']:.4f}")
    
    # Check if dots are detected
    dots_result = next(r for r in results if '6_dots' in r['name'])
    if dots_result['max_response'] < 0.5:
        print(f"⚠️  PROBLEM: Model doesn't strongly detect clear dots!")
        print(f"   Max response: {dots_result['max_response']:.4f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {test_output_dir}")


if __name__ == "__main__":
    main()