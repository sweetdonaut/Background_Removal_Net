"""Visualize PSF defects of a single type applied on dataloader patches."""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_core'))
from dataloader import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_path', type=str, required=True)
    parser.add_argument('--psf_type', type=str, required=True,
                        help='Single PSF type name (e.g., type1)')
    parser.add_argument('--output_dir', type=str, default='../output')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--img_format', type=str, default='tiff')
    args = parser.parse_args()

    defects_dir = os.path.join(os.path.dirname(__file__), '..', 'src_core', 'defects')
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(defects_dir, f'{args.psf_type}.yaml')
    ds = Dataset(
        training_path=args.training_path,
        patch_size=(args.patch_size, args.patch_size),
        num_defects_range=(4, 10),
        img_format=args.img_format,
        cache_size=10,
        defect_mode='psf',
        psf_config_paths=[config_path]
    )

    cols = ['Target', 'Ref1', 'Ref2', 'Target - Ref1', 'Mask']
    fig, axes = plt.subplots(2, len(cols), figsize=(20, 8), dpi=200)

    collected = 0
    attempts = 0
    while collected < 2 and attempts < len(ds):
        sample = ds[np.random.randint(len(ds))]
        if sample['target_mask'].sum() == 0:
            attempts += 1
            continue

        inp = sample['three_channel_input'].numpy() * 255.0
        mask = sample['target_mask'].numpy().squeeze()
        target, ref1, ref2 = inp[0], inp[1], inp[2]
        diff = target - ref1

        for col, (img, title) in enumerate(zip(
            [target, ref1, ref2, diff, mask], cols
        )):
            ax = axes[collected, col]
            if col == 4:
                ax.imshow(img, cmap='hot', vmin=0, vmax=1)
            elif col == 3:
                vmax = max(abs(img.min()), abs(img.max()), 1)
                ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            if collected == 0:
                ax.set_title(title, fontsize=13)
            ax.axis('off')

        collected += 1
        attempts += 1

    plt.suptitle(f'PSF Defect Samples — {args.psf_type}', fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f'psf_defect_{args.psf_type}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
