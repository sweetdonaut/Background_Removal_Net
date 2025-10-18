import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path


def visualize_tiff(tiff_path, output_path):
    """Visualize 3-channel TIFF file with subplots and difference maps"""
    # Read TIFF
    img = tifffile.imread(tiff_path)

    # Extract channels
    target = img[:, :, 0].astype(np.float32)
    ref1 = img[:, :, 1].astype(np.float32)
    ref2 = img[:, :, 2].astype(np.float32)

    # Calculate differences
    diff1 = target - ref1
    diff2 = target - ref2

    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Display original channels
    channel_names = ['Target', 'Ref1', 'Ref2']
    channels = [target, ref1, ref2]

    for i, (ax, name, channel) in enumerate(zip(axes[:3], channel_names, channels)):
        im = ax.imshow(channel, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{name}\nMean: {channel.mean():.1f}, Std: {channel.std():.1f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Display difference maps
    diff_names = ['Target - Ref1', 'Target - Ref2']
    diffs = [diff1, diff2]

    for i, (ax, name, diff) in enumerate(zip(axes[3:], diff_names, diffs)):
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
        ax.set_title(f'{name}\nMean: {diff.mean():.1f}, Std: {diff.std():.1f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add overall title
    fig.suptitle(f'{tiff_path.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Paths
    input_base = Path("/home/yclai/vscode_project/Background_Removal_Net/MVTec_AD_dataset/grid_offset_3channel")
    output_base = Path("/home/yclai/vscode_project/Background_Removal_Net/output")

    # Find all TIFF files
    tiff_files = sorted(input_base.rglob("*.tiff"))

    print(f"Found {len(tiff_files)} TIFF files to visualize")
    print(f"Output directory: {output_base}")
    print()

    # Process each TIFF file
    for i, tiff_path in enumerate(tiff_files):
        # Calculate relative path to maintain folder structure
        rel_path = tiff_path.relative_to(input_base)
        output_path = output_base / rel_path.parent / f"{rel_path.stem}.png"

        # Visualize
        visualize_tiff(tiff_path, output_path)

        if (i + 1) % 5 == 0 or (i + 1) == len(tiff_files):
            print(f"Visualized {i + 1}/{len(tiff_files)}: {rel_path}")

    print(f"\nDone! Visualizations saved to: {output_base}")


if __name__ == "__main__":
    main()
