"""Demo: oversample-shift-bin to simulate sensor pixel sampling.

Renders one PSF at S× resolution, then bins back to coarse grid two ways:
  - Aligned bin: PSF peak centered on a final pixel
  - Half-pixel shifted bin: PSF peak on a final-pixel boundary (the "split"
    case described by the boss — peak energy spreads across two/four pixels)

Reports total energy (must be conserved modulo trimmed border) and apparent
peak (drops when the spot straddles pixels — the "signal loss" effect).
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_core'))
from generate_psf import load_config, generate_one


def oversampled_cfg(cfg, factor, shrink_psf=1.0, zero_background=True):
    """Scale FFT grid by `factor`, keep outer_r — PSF spot is sampled S× finer.

    shrink_psf > 1.0 enlarges outer_r → smaller PSF spot, stronger sub-pixel
    sampling effect. Use this to visualize the effect on Nyquist-limited spots.
    """
    new = dict(cfg)
    new['psf_size'] = cfg['psf_size'] * factor
    new['crop_size'] = cfg['crop_size'] * factor
    if shrink_psf != 1.0:
        r = cfg['outer_r']
        new['outer_r'] = (r[0] * shrink_psf, r[1] * shrink_psf) if isinstance(r, (list, tuple)) else r * shrink_psf
    if zero_background:
        new['background'] = (0.0, 0.0)
    return new


def bin_2d(img, factor, offset):
    oy, ox = offset
    cropped = img[oy:, ox:]
    h = (cropped.shape[0] // factor) * factor
    w = (cropped.shape[1] // factor) * factor
    cropped = cropped[:h, :w]
    return cropped.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))


def crop_around(img, center, half):
    cy, cx = center
    h, w = img.shape
    return img[max(0, cy - half):min(h, cy + half),
               max(0, cx - half):min(w, cx + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_type', type=str, default='type1')
    parser.add_argument('--factor', type=int, default=4,
                        help='Oversample factor (integer ≥ 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noiseless', action='store_true',
                        help='Disable Poisson and Gaussian noise for clean visualization')
    parser.add_argument('--shrink_psf', type=float, default=1.0,
                        help='Multiply outer_r by this — >1 makes spot smaller and '
                             'sub-pixel effect more visible (default 1.0)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'output'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    defects_dir = os.path.join(os.path.dirname(__file__), '..', 'src_core', 'defects')
    cfg = load_config(os.path.join(defects_dir, f'{args.psf_type}.yaml'))
    over_cfg = oversampled_cfg(cfg, args.factor, shrink_psf=args.shrink_psf)
    if args.noiseless:
        over_cfg['poisson_noise'] = False
        over_cfg['gaussian_sigma'] = (0.0, 0.0)

    S = args.factor
    rng = np.random.default_rng(args.seed)

    # One fine PSF — both bins operate on the SAME array, so any difference
    # comes purely from bin-grid alignment (the physics being demonstrated).
    psf_fine, _ = generate_one(over_cfg, rng)
    # PSF peak is at the center of the cropped fine grid by construction
    # (generate_one center-crops around N//2). Using argmax fails under noise
    # because oversampling spreads each pixel's brightness across S² fine cells.
    py = psf_fine.shape[0] // 2
    px = psf_fine.shape[1] // 2

    # Aligned: peak at center of final pixel → (py - oy) % S == S//2
    oy_a = (py - S // 2) % S
    ox_a = (px - S // 2) % S
    # Shifted: bump the bin grid by S/2 → peak at corner between final pixels
    oy_s = (oy_a + S // 2) % S
    ox_s = (ox_a + S // 2) % S

    psf_aligned = bin_2d(psf_fine, S, (oy_a, ox_a))
    psf_shifted = bin_2d(psf_fine, S, (oy_s, ox_s))

    a_peak_pos = np.unravel_index(psf_aligned.argmax(), psf_aligned.shape)
    s_peak_pos = np.unravel_index(psf_shifted.argmax(), psf_shifted.shape)

    H = 8
    fine_view = crop_around(psf_fine, (py, px), H * S)
    aligned_view = crop_around(psf_aligned, a_peak_pos, H)
    shifted_view = crop_around(psf_shifted, s_peak_pos, H)

    # Local sum (peak ± H window) avoids border-trim bias when comparing energy
    a_local = crop_around(psf_aligned, a_peak_pos, H).sum()
    s_local = crop_around(psf_shifted, s_peak_pos, H).sum()
    a_peak = psf_aligned.max()
    s_peak = psf_shifted.max()

    sy, sx = s_peak_pos
    neighbors = [psf_shifted[sy + dy, sx + dx]
                 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                 if 0 <= sy + dy < psf_shifted.shape[0]
                 and 0 <= sx + dx < psf_shifted.shape[1]]
    second_peak = max(neighbors) if neighbors else 0.0

    vmax = max(aligned_view.max(), shifted_view.max())
    fig = plt.figure(figsize=(14, 8.5), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    ax_fine = fig.add_subplot(gs[0, 0])
    ax_aln = fig.add_subplot(gs[0, 1])
    ax_shf = fig.add_subplot(gs[0, 2])
    ax_cs = fig.add_subplot(gs[1, :])

    drop_pct = 100 * (a_peak - s_peak) / a_peak if a_peak > 0 else 0

    im0 = ax_fine.imshow(fine_view, cmap='hot', interpolation='nearest')
    ax_fine.set_title(f'Fine PSF (oversampled {S}×)\n'
                      f'shape={fine_view.shape}', fontsize=10)
    ax_fine.axis('off')
    plt.colorbar(im0, ax=ax_fine, fraction=0.046, pad=0.04)

    im1 = ax_aln.imshow(aligned_view, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
    ax_aln.set_title(f'Aligned bin (peak at pixel center)\n'
                     f'apparent peak = {a_peak:.1f}\n'
                     f'local sum (±{H}px) = {a_local:.1f}', fontsize=10)
    ax_aln.axis('off')
    plt.colorbar(im1, ax=ax_aln, fraction=0.046, pad=0.04)

    im2 = ax_shf.imshow(shifted_view, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
    ax_shf.set_title(f'Half-pixel shift bin (peak on boundary)\n'
                     f'apparent peak = {s_peak:.1f} '
                     f'(↓{drop_pct:.1f}%); next = {second_peak:.1f}\n'
                     f'local sum (±{H}px) = {s_local:.1f}', fontsize=10)
    ax_shf.axis('off')
    plt.colorbar(im2, ax=ax_shf, fraction=0.046, pad=0.04)

    # 1D cross-section through the peak row — makes the peak drop and the
    # split-into-neighbor effect immediately visible.
    a_row = psf_aligned[a_peak_pos[0], :]
    s_row = psf_shifted[s_peak_pos[0], :]
    a_xs = np.arange(len(a_row)) - a_peak_pos[1]
    s_xs = np.arange(len(s_row)) - s_peak_pos[1]
    ax_cs.plot(a_xs, a_row, 'o-', color='tab:blue', label=f'aligned (peak={a_peak:.1f})',
               markersize=6, linewidth=1.8)
    ax_cs.plot(s_xs, s_row, 's-', color='tab:red', label=f'shifted (peak={s_peak:.1f})',
               markersize=6, linewidth=1.8)
    ax_cs.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_cs.set_xlabel('Pixel offset from peak')
    ax_cs.set_ylabel('Pixel value')
    ax_cs.set_title('Horizontal cross-section through peak — '
                    'shifted version splits energy across two adjacent pixels',
                    fontsize=11)
    ax_cs.set_xlim(-H, H)
    ax_cs.legend(loc='upper right')
    ax_cs.grid(alpha=0.3)

    noise_tag = 'noiseless' if args.noiseless else 'with noise'
    plt.suptitle(f'Oversample-Shift-Bin Demo — {args.psf_type}, '
                 f'oversample={S}× ({noise_tag})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    shrink_tag = f'_shrink{args.shrink_psf:g}' if args.shrink_psf != 1.0 else ''
    out = os.path.join(args.output_dir,
                       f'oversample_demo_{args.psf_type}_x{S}{shrink_tag}'
                       f'{"_noiseless" if args.noiseless else ""}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved figure: {out}\n')
    print('Energy conservation (local sum within peak window):')
    print(f'  aligned local sum = {a_local:.4f}')
    print(f'  shifted local sum = {s_local:.4f}')
    diff_pct = 100 * abs(a_local - s_local) / max(a_local, 1e-9)
    print(f'  diff              = {abs(a_local - s_local):.4f} ({diff_pct:.3f} %)\n')
    print('Apparent peak comparison ("signal loss" the boss described):')
    print(f'  aligned peak   = {a_peak:.4f}')
    print(f'  shifted peak   = {s_peak:.4f}  (↓ {drop_pct:.2f} %)')
    print(f'  next neighbor  = {second_peak:.4f}  '
          f'(when peak straddles, neighbor lights up)\n')
    print('Sanity check: aligned/shifted local sums should be near-equal '
          '(energy conservation); apparent peak should drop and a neighbor '
          'should rise (peak-splitting).')


if __name__ == '__main__':
    main()
