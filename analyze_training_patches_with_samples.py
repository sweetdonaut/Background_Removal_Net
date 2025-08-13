"""
Analyze edge ratios and save sample patches with scores
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
from glob import glob
import os
from tqdm import tqdm
from dataloader import calculate_positions
import random
from datetime import datetime

def analyze_patch_edge_ratio(patch, threshold_low=50, threshold_high=150):
    """Calculate edge ratio for a patch"""
    if len(patch.shape) == 3:
        gray = patch[:, :, 0]
    else:
        gray = patch
    
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, threshold_low, threshold_high)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    return edge_ratio * 100, edges

def save_patch_with_score(patch, edge_ratio, edges, output_dir, index):
    """Save individual patch with edge ratio score"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original patch
    if len(patch.shape) == 3:
        gray = patch[:, :, 0]
    else:
        gray = patch
    
    axes[0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Patch', fontsize=12)
    axes[0].axis('off')
    
    # Edge detection result
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edges', fontsize=12)
    axes[1].axis('off')
    
    # Colorized overlay
    overlay = np.zeros((patch.shape[0], patch.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = gray  # Red channel
    overlay[:, :, 1] = gray  # Green channel
    overlay[:, :, 2] = gray  # Blue channel
    # Add red edges
    edge_mask = edges > 0
    overlay[edge_mask, 0] = 255
    overlay[edge_mask, 1] = 0
    overlay[edge_mask, 2] = 0
    
    axes[2].imshow(overlay)
    axes[2].set_title('Edges Overlay', fontsize=12)
    axes[2].axis('off')
    
    # Main title with edge ratio
    if edge_ratio >= 2.0:
        color = 'red'
        category = 'STRUCTURAL'
    else:
        color = 'green'
        category = 'POINT DEFECTS'
    
    fig.suptitle(f'Patch #{index:03d} - Edge Ratio: {edge_ratio:.4f}% - {category}', 
                 fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'patch_{index:03d}_ratio_{edge_ratio:.2f}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def analyze_training_dataset_with_samples(
    training_path, 
    patch_size=(128, 128),
    img_format='tiff', 
    image_type='strip',
    sample_limit=None,
    num_sample_patches=20,
    output_dir='output/patch_samples'):
    """
    Analyze edge ratios and save sample patches
    
    Args:
        training_path: Path to training images
        patch_size: Size of patches (height, width)
        img_format: 'tiff' or 'png_jpg'
        image_type: 'strip', 'square', or 'mvtec'
        sample_limit: Limit number of images to analyze (None = all)
        num_sample_patches: Number of random patches to save as images
        output_dir: Directory to save sample patches
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Load image paths
    if img_format == 'tiff':
        image_paths = glob(os.path.join(training_path, "*.tiff"))
        image_paths.extend(glob(os.path.join(training_path, "*.tif")))
    else:
        image_paths = glob(os.path.join(training_path, "*.png"))
        image_paths.extend(glob(os.path.join(training_path, "*.jpg")))
    
    if sample_limit:
        image_paths = image_paths[:sample_limit]
    
    print(f"Found {len(image_paths)} images to analyze")
    
    # Determine image size
    if image_type == 'strip':
        img_h, img_w = 976, 176
        min_patches_y = 9
    elif image_type == 'square':
        img_h, img_w = 600, 600
        min_patches_y = 2
    else:  # mvtec
        img_h, img_w = 1024, 1024
        min_patches_y = 2
    
    # Calculate patch positions
    y_positions = calculate_positions(img_h, patch_size[0], min_patches=min_patches_y if image_type == 'strip' else 2)
    x_positions = calculate_positions(img_w, patch_size[1])
    
    total_patches_per_image = len(y_positions) * len(x_positions)
    print(f"Patches per image: {len(y_positions)} x {len(x_positions)} = {total_patches_per_image}")
    
    # Analyze all patches
    all_patches_data = []
    
    print("\nAnalyzing patches...")
    for img_idx, img_path in enumerate(tqdm(image_paths)):
        # Load image
        if img_format == 'tiff':
            image = tifffile.imread(img_path)
        else:
            image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        
        # Convert CHW to HWC if needed
        if image_type == 'strip' and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Extract all patches
        for y_idx, start_y in enumerate(y_positions):
            for x_idx, start_x in enumerate(x_positions):
                end_y = start_y + patch_size[0]
                end_x = start_x + patch_size[1]
                
                patch = image[start_y:end_y, start_x:end_x]
                
                # Calculate edge ratio
                edge_ratio, edges = analyze_patch_edge_ratio(patch)
                
                # Store patch data
                all_patches_data.append({
                    'img_path': os.path.basename(img_path),
                    'img_idx': img_idx,
                    'y_idx': y_idx,
                    'x_idx': x_idx,
                    'start_y': start_y,
                    'start_x': start_x,
                    'edge_ratio': edge_ratio,
                    'patch': patch.copy(),
                    'edges': edges
                })
    
    print(f"\nTotal patches analyzed: {len(all_patches_data)}")
    
    # Sample patches for saving
    if num_sample_patches > len(all_patches_data):
        num_sample_patches = len(all_patches_data)
    
    # Strategy: Sample from different ratio ranges for diversity
    all_ratios = [p['edge_ratio'] for p in all_patches_data]
    min_ratio = min(all_ratios)
    max_ratio = max(all_ratios)
    
    # Sort patches by edge ratio
    sorted_patches = sorted(all_patches_data, key=lambda x: x['edge_ratio'])
    
    # Sample evenly across the range
    sampled_patches = []
    indices = np.linspace(0, len(sorted_patches)-1, num_sample_patches, dtype=int)
    for idx in indices:
        sampled_patches.append(sorted_patches[idx])
    
    # Also ensure we get some random samples
    random_samples = random.sample(all_patches_data, min(10, len(all_patches_data)))
    
    # Combine and deduplicate
    final_samples = {id(p): p for p in sampled_patches}
    for p in random_samples:
        final_samples[id(p)] = p
    final_samples = list(final_samples.values())[:num_sample_patches]
    
    # Sort by edge ratio for organized output
    final_samples.sort(key=lambda x: x['edge_ratio'])
    
    print(f"\nSaving {len(final_samples)} sample patches to {session_dir}")
    
    saved_files = []
    for i, patch_data in enumerate(final_samples):
        filename = save_patch_with_score(
            patch_data['patch'],
            patch_data['edge_ratio'],
            patch_data['edges'],
            session_dir,
            i + 1
        )
        saved_files.append(filename)
        
        # Print info
        print(f"  [{i+1:2d}/{len(final_samples)}] {patch_data['img_path']} "
              f"pos:({patch_data['y_idx']},{patch_data['x_idx']}) "
              f"ratio:{patch_data['edge_ratio']:.3f}%")
    
    # Create summary statistics
    all_ratios = np.array(all_ratios)
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("EDGE RATIO STATISTICS")
    print("=" * 80)
    print(f"Total patches analyzed: {len(all_ratios)}")
    print(f"Min edge ratio: {np.min(all_ratios):.4f}%")
    print(f"Max edge ratio: {np.max(all_ratios):.4f}%")
    print(f"Mean edge ratio: {np.mean(all_ratios):.4f}%")
    print(f"Median edge ratio: {np.median(all_ratios):.4f}%")
    print(f"Std dev: {np.std(all_ratios):.4f}%")
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentile distribution:")
    for p in percentiles:
        val = np.percentile(all_ratios, p)
        print(f"  {p:3d}th percentile: {val:8.4f}%")
    
    # Threshold analysis
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    print("\nPatches vs different thresholds:")
    for t in thresholds:
        below = np.sum(all_ratios <= t)
        above = np.sum(all_ratios > t)
        percent_below = below / len(all_ratios) * 100
        percent_above = above / len(all_ratios) * 100
        print(f"  Threshold {t:.1f}%: {below:4d} below ({percent_below:5.1f}%), "
              f"{above:4d} above ({percent_above:5.1f}%)")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(all_ratios, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2%)')
    ax1.set_xlabel('Edge Ratio (%)')
    ax1.set_ylabel('Number of Patches')
    ax1.set_title('Edge Ratio Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    sorted_ratios = np.sort(all_ratios)
    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
    ax2.plot(sorted_ratios, cumulative, linewidth=2)
    ax2.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2%)')
    ax2.set_xlabel('Edge Ratio (%)')
    ax2.set_ylabel('Cumulative % of Patches')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Box plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(all_ratios, vert=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.axvline(x=2.0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Edge Ratio (%)')
    ax3.set_title('Box Plot with Outliers')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sampled patches distribution
    ax4 = axes[1, 1]
    sampled_ratios = [p['edge_ratio'] for p in final_samples]
    ax4.scatter(range(len(sampled_ratios)), sampled_ratios, alpha=0.6, s=50)
    ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2%)')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Edge Ratio (%)')
    ax4.set_title(f'Sampled {len(final_samples)} Patches')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(f'Edge Ratio Analysis - {len(image_paths)} Images, {len(all_ratios)} Patches', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(session_dir, 'summary_statistics.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Sample patches saved to: {session_dir}")
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # Create a text summary file
    summary_txt_path = os.path.join(session_dir, 'summary.txt')
    with open(summary_txt_path, 'w') as f:
        f.write(f"Edge Ratio Analysis Summary\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Dataset: {training_path}\n")
        f.write(f"Images analyzed: {len(image_paths)}\n")
        f.write(f"Total patches: {len(all_ratios)}\n")
        f.write(f"Patch size: {patch_size}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Min: {np.min(all_ratios):.4f}%\n")
        f.write(f"  Max: {np.max(all_ratios):.4f}%\n")
        f.write(f"  Mean: {np.mean(all_ratios):.4f}%\n")
        f.write(f"  Median: {np.median(all_ratios):.4f}%\n")
        f.write(f"  Std Dev: {np.std(all_ratios):.4f}%\n\n")
        f.write(f"Threshold Analysis (2.0%):\n")
        below_2 = np.sum(all_ratios <= 2.0)
        above_2 = np.sum(all_ratios > 2.0)
        f.write(f"  Below threshold: {below_2} ({below_2/len(all_ratios)*100:.1f}%)\n")
        f.write(f"  Above threshold: {above_2} ({above_2/len(all_ratios)*100:.1f}%)\n\n")
        f.write(f"Sampled patches: {len(final_samples)}\n")
        f.write(f"  Range: {min(sampled_ratios):.4f}% - {max(sampled_ratios):.4f}%\n")
    
    print(f"✓ Text summary saved to: {summary_txt_path}")
    
    return all_ratios, all_patches_data, session_dir

if __name__ == "__main__":
    # Analyze training dataset with sample patches
    ratios, patches_data, output_dir = analyze_training_dataset_with_samples(
        training_path="./MVTec_AD_dataset/grid_stripe/train/good/",
        patch_size=(128, 128),
        img_format='tiff',
        image_type='strip',
        sample_limit=30,  # Analyze first 30 images
        num_sample_patches=20,  # Save 20 sample patches
        output_dir='output/patch_samples'
    )