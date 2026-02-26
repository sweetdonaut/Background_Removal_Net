import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path


def visualize_aligned_tiff(tiff_path, output_path):
    """Visualize aligned 3-channel TIFF file with difference maps"""
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
    channel_names = ['Target', 'Ref1 (aligned)', 'Ref2 (aligned)']
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
        ax.set_title(f'{name}\nMean: {diff.mean():.2f}, Std: {diff.std():.2f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add overall title
    fig.suptitle(f'Aligned: {tiff_path.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import sys

    # Allow method selection via command line
    method = sys.argv[1] if len(sys.argv) > 1 else 'phase_correlation'
    use_cropped = sys.argv[2] if len(sys.argv) > 2 else 'cropped'

    if method not in ['phase_correlation', 'template_matching']:
        print(f"Unknown method: {method}")
        print("Usage: python visualize_alignment.py [phase_correlation|template_matching] [aligned|cropped]")
        return

    # Paths
    if use_cropped == 'cropped':
        input_base = Path(f"/home/yclai/vscode_project/Background_Removal_Net/output/aligned_cropped_{method}")
        output_base = Path(f"/home/yclai/vscode_project/Background_Removal_Net/output/visualization_cropped_{method}")
    else:
        input_base = Path(f"/home/yclai/vscode_project/Background_Removal_Net/output/aligned_{method}")
        output_base = Path(f"/home/yclai/vscode_project/Background_Removal_Net/output/visualization_{method}")

    # Find all TIFF files
    tiff_files = sorted(input_base.rglob("*.tiff"))

    print(f"Visualizing {'cropped' if use_cropped == 'cropped' else 'aligned'} results using: {method}")
    print(f"Found {len(tiff_files)} TIFF files")
    print(f"Output directory: {output_base}")
    print()

    # Process each TIFF file
    for i, tiff_path in enumerate(tiff_files):
        # Calculate relative path to maintain folder structure
        rel_path = tiff_path.relative_to(input_base)
        output_path = output_base / rel_path.parent / f"{rel_path.stem}.png"

        # Visualize
        visualize_aligned_tiff(tiff_path, output_path)

        if (i + 1) % 5 == 0 or (i + 1) == len(tiff_files):
            print(f"Visualized {i + 1}/{len(tiff_files)}: {rel_path}")

    print(f"\nDone! Visualizations saved to: {output_base}")


if __name__ == "__main__":
    main()
