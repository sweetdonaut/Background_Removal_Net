"""
Compare original model vs improved model with edge negative samples
"""
import torch
import numpy as np
import cv2
import tifffile
import os
import matplotlib.pyplot as plt
from model import SegmentationNetwork
from inference import sliding_window_inference
import torch.nn.functional as F
from diagnostic_test import create_test_images, load_model


def compare_models_on_tests(model1, model2, test_cases, patch_size, image_type, device, output_dir):
    """Compare two models on the same test cases"""
    
    results_comparison = []
    
    for test in test_cases:
        print(f"\nProcessing: {test['name']}")
        
        # Run inference with both models
        image = test['image']
        heatmap1, _ = sliding_window_inference(image, model1, patch_size, device, image_type)
        heatmap2, _ = sliding_window_inference(image, model2, patch_size, device, image_type)
        
        # Analyze results
        result = {
            'name': test['name'],
            'description': test['description'],
            'model1_max': np.max(heatmap1),
            'model1_mean': np.mean(heatmap1),
            'model1_high_ratio': np.sum(heatmap1 > 0.1) / heatmap1.size,
            'model2_max': np.max(heatmap2),
            'model2_mean': np.mean(heatmap2),
            'model2_high_ratio': np.sum(heatmap2 > 0.1) / heatmap2.size,
            'heatmap1': heatmap1,
            'heatmap2': heatmap2
        }
        results_comparison.append(result)
        
        # Print comparison
        print(f"  Original Model - Max: {result['model1_max']:.4f}, Mean: {result['model1_mean']:.6f}, High%: {result['model1_high_ratio']:.2%}")
        print(f"  Improved Model - Max: {result['model2_max']:.4f}, Mean: {result['model2_mean']:.6f}, High%: {result['model2_high_ratio']:.2%}")
        
        improvement = result['model1_high_ratio'] - result['model2_high_ratio']
        if improvement > 0:
            print(f"  ✓ Improvement: {improvement:.2%} less false positives")
        elif improvement < 0:
            print(f"  ✗ Degradation: {-improvement:.2%} more false positives")
        else:
            print(f"  = No change")
    
    # Create comparison visualization
    create_comparison_plot(results_comparison, output_dir)
    
    return results_comparison


def create_comparison_plot(results, output_dir):
    """Create side-by-side comparison plots"""
    
    # Focus on key test cases
    key_tests = [
        '2_identical_channels',
        '6_dots_in_target', 
        '7_edge_enhancement',
        '8_horizontal_stripes'
    ]
    
    fig, axes = plt.subplots(len(key_tests), 3, figsize=(12, 16))
    
    for idx, test_name in enumerate(key_tests):
        # Find the test result
        result = next((r for r in results if test_name in r['name']), None)
        if result is None:
            continue
        
        # Original model heatmap
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(result['heatmap1'], cmap='hot', vmin=0, vmax=0.1)
        ax1.set_title(f"Original\nMax: {result['model1_max']:.3f}", fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Improved model heatmap
        ax2 = axes[idx, 1]
        im2 = ax2.imshow(result['heatmap2'], cmap='hot', vmin=0, vmax=0.1)
        ax2.set_title(f"Improved\nMax: {result['model2_max']:.3f}", fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference (Original - Improved)
        ax3 = axes[idx, 2]
        diff = result['heatmap1'] - result['heatmap2']
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax3.set_title(f"Difference\n(Orig - Improved)", fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Add test description on the left
        ax1.text(-0.3, 0.5, result['description'][:30], 
                transform=ax1.transAxes, rotation=90,
                verticalalignment='center', fontsize=9)
    
    plt.suptitle("Model Comparison: Original vs Improved (with Edge Negative Samples)", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {save_path}")


def print_summary(results):
    """Print summary of improvements"""
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Check key improvements
    identical_test = next(r for r in results if '2_identical' in r['name'])
    edge_test = next(r for r in results if '7_edge' in r['name'])
    stripe_test = next(r for r in results if '8_horizontal' in r['name'])
    dots_test = next(r for r in results if '6_dots' in r['name'])
    
    print("\nKey Test Results:")
    print("-"*40)
    
    # Identical channels (should have no response)
    print(f"\n1. Identical Channels Test:")
    print(f"   Original: Max={identical_test['model1_max']:.4f}")
    print(f"   Improved: Max={identical_test['model2_max']:.4f}")
    if identical_test['model2_max'] < identical_test['model1_max']:
        print(f"   ✓ Reduced by {(1 - identical_test['model2_max']/identical_test['model1_max'])*100:.1f}%")
    
    # Edge enhancement (main problem - should not detect)
    print(f"\n2. Edge Enhancement Test:")
    print(f"   Original: {edge_test['model1_high_ratio']:.1%} pixels detected")
    print(f"   Improved: {edge_test['model2_high_ratio']:.1%} pixels detected")
    if edge_test['model2_high_ratio'] < edge_test['model1_high_ratio']:
        reduction = edge_test['model1_high_ratio'] - edge_test['model2_high_ratio']
        print(f"   ✓ Reduced by {reduction:.1%} absolute")
    
    # Stripes (should not detect)
    print(f"\n3. Horizontal Stripes Test:")
    print(f"   Original: {stripe_test['model1_high_ratio']:.1%} pixels detected")
    print(f"   Improved: {stripe_test['model2_high_ratio']:.1%} pixels detected")
    if stripe_test['model2_high_ratio'] < stripe_test['model1_high_ratio']:
        reduction = stripe_test['model1_high_ratio'] - stripe_test['model2_high_ratio']
        print(f"   ✓ Reduced by {reduction:.1%} absolute")
    
    # Dots (should detect strongly)
    print(f"\n4. Dot Detection Test:")
    print(f"   Original: Max={dots_test['model1_max']:.3f}")
    print(f"   Improved: Max={dots_test['model2_max']:.3f}")
    if dots_test['model2_max'] >= 0.9:
        print(f"   ✓ Still detecting dots well")
    
    print("\n" + "="*60)


def main():
    # Paths
    original_model_path = "./checkpoints/stripe_experiment/BgRemoval_lr0.0001_ep100_bs16_128x128_strip.pth"
    
    # Check if improved model exists
    import glob
    improved_models = glob.glob("./checkpoints/improved_edge_neg/BgRemoval_*.pth")
    if not improved_models:
        print("❌ Improved model not found!")
        print("Please train the improved model first using:")
        print("  bash train_improved.sh")
        return
    
    improved_model_path = improved_models[-1]  # Use latest
    print(f"Using improved model: {improved_model_path}")
    
    base_image_path = "./MVTec_AD_dataset/grid_stripe/test/good/260.tiff"
    output_dir = "./output/model_comparison"
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("\nLoading original model...")
    model1, patch_size, image_type = load_model(original_model_path, device)
    
    print("Loading improved model...")
    model2, _, _ = load_model(improved_model_path, device)
    
    # Create test images
    print("\nCreating test images...")
    test_cases = create_test_images(base_image_path, os.path.join(output_dir, "test_images"))
    
    # Compare models
    print("\nComparing models on diagnostic tests...")
    results = compare_models_on_tests(
        model1, model2, test_cases, patch_size, image_type, device, output_dir
    )
    
    # Print summary
    print_summary(results)
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()