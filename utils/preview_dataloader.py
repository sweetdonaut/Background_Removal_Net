"""Preview dataloader output: full target / diff1 / diff2 / mask plus a
per-defect zoom-in for the first defect in each sample.

Useful for sanity-checking that:
  - synthesized defect intensity matches the yaml's intensity_abs
  - partial_leak distractor pattern shows up at the expected ratio
  - GT mask only marks target-only defects (not distractors)

Usage (PSF mode):
    python utils/preview_dataloader.py \\
        --psf_type type4_vector \\
        --training_dataset_path data/grid_stripe_4channel/train/good \\
        --output output/dataloader_preview.png

Usage (Gaussian mode):
    python utils/preview_dataloader.py \\
        --defect_mode gaussian \\
        --training_dataset_path data/grid_stripe_4channel/train/good
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_core'))
from dataloader import Dataset  # noqa: E402

ZOOM_HALF = 12


def render_preview(samples, out_path, suptitle=''):
    n = len(samples)
    if n == 0:
        print('Warning: no defect samples to render')
        return

    fig = plt.figure(figsize=(24, 4.5 * n))
    gs = fig.add_gridspec(n, 8, width_ratios=[1, 1, 1, 1, 0.4, 1, 1, 1])

    for r, s in enumerate(samples):
        three = s['three_channel_input'].numpy() * 255
        mask = s['target_mask'].numpy()[0]
        target, ref1, ref2 = three[0], three[1], three[2]
        diff1 = target - ref1
        diff2 = target - ref2
        abs_max = max(abs(diff1).max(), abs(diff2).max(), 1)

        full_panels = [
            (target, 'target',        'gray',   {'vmin': 0, 'vmax': 255}),
            (diff1,  'target − ref1', 'RdBu_r', {'vmin': -abs_max, 'vmax': abs_max}),
            (diff2,  'target − ref2', 'RdBu_r', {'vmin': -abs_max, 'vmax': abs_max}),
            (mask,   'GT mask',       'gray',   {'vmin': 0, 'vmax': 1}),
        ]
        for c, (img, name, cmap, kw) in enumerate(full_panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cmap, **kw)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(name, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

        lbl, n_comp = label(mask > 0)
        if n_comp < 1:
            continue
        ys, xs = np.where(lbl == 1)
        cy, cx = int(ys.mean()), int(xs.mean())
        H, W = mask.shape
        y0, y1 = max(0, cy - ZOOM_HALF), min(H, cy + ZOOM_HALF)
        x0, x1 = max(0, cx - ZOOM_HALF), min(W, cx + ZOOM_HALF)

        d1 = diff1[ys, xs]
        d2 = diff2[ys, xs]
        d1_peak = float(d1[np.argmax(np.abs(d1))])
        d2_peak = float(d2[np.argmax(np.abs(d2))])
        ratio = (min(abs(d1_peak), abs(d2_peak))
                 / max(abs(d1_peak), abs(d2_peak), 1e-6))
        leak_status = 'symmetric' if ratio > 0.7 else f'LEAKED (ratio={ratio:.2f})'

        zoom_panels = [
            (target[y0:y1, x0:x1], 'target zoom', 'gray', {'vmin': 0, 'vmax': 255}),
            (diff1[y0:y1, x0:x1], f't−r1 (peak {d1_peak:+.1f})', 'RdBu_r',
                {'vmin': -abs_max, 'vmax': abs_max}),
            (diff2[y0:y1, x0:x1], f't−r2 (peak {d2_peak:+.1f})', 'RdBu_r',
                {'vmin': -abs_max, 'vmax': abs_max}),
        ]
        for c, (img, name, cmap, kw) in enumerate(zoom_panels):
            ax = fig.add_subplot(gs[r, 5 + c])
            im = ax.imshow(img, cmap=cmap, **kw)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.text(0.55, 0.96 - r * (1.0 / n) * 0.94,
                 f'sample {r + 1} (defect 1 of {n_comp}): {leak_status}',
                 fontsize=10, fontweight='bold', ha='center')

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.998)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    p = argparse.ArgumentParser(description='Preview dataloader output')
    p.add_argument('--training_dataset_path', required=True,
                   help='Path to training images folder')
    p.add_argument('--output', default='output/dataloader_preview.png',
                   help='Output PNG path (default: output/dataloader_preview.png)')
    p.add_argument('--n_samples', type=int, default=5,
                   help='Number of defect-containing samples to visualize (default: 5)')
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--num_defects_range', type=int, nargs=2, default=[4, 6])
    p.add_argument('--img_format', choices=['png_jpg', 'tiff'], default='tiff')
    p.add_argument('--cache_size', type=int, default=20)
    p.add_argument('--defect_mode', choices=['gaussian', 'psf'], default='psf')
    p.add_argument('--psf_type', nargs='+', default=None,
                   help='PSF yaml name(s) in src_core/defects/ (required for psf mode)')
    p.add_argument('--psf_pool_size', type=int, default=50)
    p.add_argument('--partial_leak_scale', type=float, nargs=2, default=[0.2, 0.7])
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    psf_config_paths = None
    intensity_label = None
    if args.defect_mode == 'psf':
        if not args.psf_type:
            p.error('--psf_type required when --defect_mode=psf')
        defects_dir = os.path.join(os.path.dirname(__file__), '..', 'src_core', 'defects')
        psf_config_paths = [os.path.join(defects_dir, f'{t}.yaml') for t in args.psf_type]
        try:
            from generate_psf import load_config
            cfg = load_config(psf_config_paths[0])
            intensity_label = str(cfg.get('intensity_abs', 'unknown'))
        except Exception:
            pass

    ds = Dataset(
        training_path=args.training_dataset_path,
        patch_size=(args.patch_size, args.patch_size),
        num_defects_range=tuple(args.num_defects_range),
        img_format=args.img_format,
        cache_size=args.cache_size,
        defect_mode=args.defect_mode,
        psf_config_paths=psf_config_paths,
        psf_pool_size=args.psf_pool_size,
        partial_leak_scale=tuple(args.partial_leak_scale),
    )

    samples = []
    attempts = 0
    while len(samples) < args.n_samples and attempts < args.n_samples * 50:
        item = ds[np.random.randint(len(ds))]
        if item['target_mask'].numpy().sum() > 0:
            samples.append(item)
        attempts += 1
    print(f'Collected {len(samples)} defect-containing samples (after {attempts} draws)')

    parts = []
    if intensity_label:
        parts.append(f'intensity_abs={intensity_label}')
    parts.append(f'partial_leak prob=0.4 scale={tuple(args.partial_leak_scale)}')
    suptitle = 'Dataloader output: ' + ', '.join(parts)

    render_preview(samples, args.output, suptitle)


if __name__ == '__main__':
    main()
