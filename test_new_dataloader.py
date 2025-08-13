"""
Test script to visualize the new dataloader with edge negative samples
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataloader import Dataset
import os


def visualize_batch(batch, save_path, batch_idx):
    """Visualize a batch of training samples"""
    three_channel = batch['three_channel_input']
    target_mask = batch['target_mask']
    
    batch_size = three_channel.shape[0]
    
    # Create a figure with subplots for each sample
    fig = plt.figure(figsize=(20, 4 * min(batch_size, 4)))
    
    for i in range(min(batch_size, 4)):  # Show max 4 samples
        # Extract channels
        target = three_channel[i, 0].numpy() * 255
        ref1 = three_channel[i, 1].numpy() * 255
        ref2 = three_channel[i, 2].numpy() * 255
        gt_mask = target_mask[i, 0].numpy()
        
        # Compute differences
        diff1 = np.abs(target - ref1)
        diff2 = np.abs(target - ref2)
        
        # Create subplots
        ax1 = plt.subplot(min(batch_size, 4), 7, i*7 + 1)
        ax1.imshow(target, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'Sample {i+1}: Target')
        ax1.axis('off')
        
        ax2 = plt.subplot(min(batch_size, 4), 7, i*7 + 2)
        ax2.imshow(ref1, cmap='gray', vmin=0, vmax=255)
        ax2.set_title('Ref1')
        ax2.axis('off')
        
        ax3 = plt.subplot(min(batch_size, 4), 7, i*7 + 3)
        ax3.imshow(ref2, cmap='gray', vmin=0, vmax=255)
        ax3.set_title('Ref2')
        ax3.axis('off')
        
        ax4 = plt.subplot(min(batch_size, 4), 7, i*7 + 4)
        im4 = ax4.imshow(diff1, cmap='hot', vmin=0, vmax=max(1, np.max(diff1)))
        ax4.set_title('|T - R1|')
        ax4.axis('off')
        
        ax5 = plt.subplot(min(batch_size, 4), 7, i*7 + 5)
        im5 = ax5.imshow(diff2, cmap='hot', vmin=0, vmax=max(1, np.max(diff2)))
        ax5.set_title('|T - R2|')
        ax5.axis('off')
        
        ax6 = plt.subplot(min(batch_size, 4), 7, i*7 + 6)
        im6 = ax6.imshow(gt_mask, cmap='hot', vmin=0, vmax=1)
        ax6.set_title('GT Mask')
        ax6.axis('off')
        
        # Analyze sample type
        has_gt = np.sum(gt_mask) > 0
        has_diff = np.max(diff1) > 20 or np.max(diff2) > 20
        
        if has_gt:
            sample_type = "Point Defects"
            color = 'green'
        elif has_diff:
            sample_type = "Edge/Stripe (Negative)"
            color = 'red'
        else:
            sample_type = "Clean"
            color = 'blue'
        
        ax7 = plt.subplot(min(batch_size, 4), 7, i*7 + 7)
        ax7.text(0.5, 0.5, sample_type, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax7.transAxes,
                fontsize=12,
                color=color,
                weight='bold')
        ax7.set_title('Type')
        ax7.axis('off')
    
    plt.suptitle(f'Training Batch {batch_idx} - New Dataloader with Edge Negative Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def analyze_batch_statistics(dataloader, num_batches=10):
    """Analyze the distribution of sample types in batches"""
    
    stats = {
        'point_defects': 0,
        'edge_negative': 0,
        'clean': 0,
        'total': 0
    }
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        three_channel = batch['three_channel_input']
        target_mask = batch['target_mask']
        
        batch_size = three_channel.shape[0]
        
        for i in range(batch_size):
            target = three_channel[i, 0].numpy() * 255
            ref1 = three_channel[i, 1].numpy() * 255
            ref2 = three_channel[i, 2].numpy() * 255
            gt_mask = target_mask[i, 0].numpy()
            
            diff1 = np.abs(target - ref1)
            diff2 = np.abs(target - ref2)
            
            has_gt = np.sum(gt_mask) > 0
            has_strong_diff = np.max(diff1) > 20 or np.max(diff2) > 20
            
            if has_gt:
                stats['point_defects'] += 1
            elif has_strong_diff:
                stats['edge_negative'] += 1
            else:
                stats['clean'] += 1
            
            stats['total'] += 1
    
    return stats


def main():
    # Create output directory
    output_dir = "./output/dataloader_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    print("Creating dataset with new edge negative samples...")
    dataset = Dataset(
        training_path="./MVTec_AD_dataset/grid_stripe/train/good/",
        patch_size=(128, 128),
        num_defects_range=(3, 8),
        img_format='tiff',
        image_type='strip',
        cache_size=0
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)} patches")
    print(f"Number of batches: {len(dataloader)}")
    
    # Visualize first few batches
    print("\nVisualizing sample batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Visualize first 5 batches
            break
        
        save_path = os.path.join(output_dir, f"batch_{batch_idx:03d}.png")
        visualize_batch(batch, save_path, batch_idx)
    
    # Analyze statistics
    print("\nAnalyzing sample type distribution...")
    stats = analyze_batch_statistics(dataloader, num_batches=20)
    
    print("\nSample Type Distribution (first 20 batches):")
    print(f"  Point Defects:    {stats['point_defects']:4d} ({stats['point_defects']/stats['total']*100:.1f}%)")
    print(f"  Edge/Stripe Neg:  {stats['edge_negative']:4d} ({stats['edge_negative']/stats['total']*100:.1f}%)")
    print(f"  Clean:            {stats['clean']:4d} ({stats['clean']/stats['total']*100:.1f}%)")
    print(f"  Total:            {stats['total']:4d}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Expected distribution based on code:
    # - 20% edge negative samples
    # - 40% point defects  
    # - 40% clean
    # But since patches have 50% chance in original, actual might differ
    
    print("\nExpected distribution:")
    print("  Edge/Stripe Neg:  ~20%")
    print("  Point Defects:    ~40%")
    print("  Clean:            ~40%")


if __name__ == "__main__":
    main()