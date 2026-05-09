"""Phase 0c — fit inverse model on the 30 real PSF triplets.

For each tiff in data/30ea_testing/bad/:
  1. preprocess_one() → diff1, diff2, target/ref crops
  2. fit_three_channel() → theta, I, alpha1, alpha2, loss
  3. Save fitted dict to JSON keyed by defect_id
  4. Render an 8-panel diagnostic PNG (raw triplet, diffs, fit, residuals)

Outputs (under --output_dir, default src_invertfit/fitted/):
  fitted_theta.json   ← all 30 fits in one file (Phase 1 yaml exporter reads this)
  vis/<DefectID>.png  ← per-defect visualization
  summary.txt         ← human-readable per-defect line + aggregate stats
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import ThreeChannelFitConfig, fit_three_channel  # noqa: E402
from preprocess import preprocess_one  # noqa: E402


# Aligned with src_core/defects/type4_vector.yaml (same baseline as Phase 0a/b)
DEFAULT_FIXED = {
    'outer_r': 60.0,
    'epsilon': 0.0,
    'ellipticity': 0.0,
    'ellip_angle': 0.0,
    'na': 0.95,
}

DEFAULT_FIT_PARAMS = (
    'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y',
    'spherical', 'trefoil_x', 'trefoil_y',
)


def render_diagnostic(pre, fit_pred1_np, fit_pred2_np, fit_psf_np, out_path):
    """8-panel visualization: 3 raw + 2 diffs + 1 fitted PSF + 2 residuals.

    Layout:
        target  ref1  ref2  fitted_PSF
        diff1   diff2 resid1 resid2
    """
    target = pre['target']; ref1 = pre['ref1']; ref2 = pre['ref2']
    diff1 = pre['diff1']; diff2 = pre['diff2']
    resid1 = diff1 - fit_pred1_np
    resid2 = diff2 - fit_pred2_np

    raw_vmin = float(min(target.min(), ref1.min(), ref2.min()))
    raw_vmax = float(max(target.max(), ref1.max(), ref2.max()))
    diff_abs_max = float(max(abs(diff1).max(), abs(diff2).max(),
                              abs(fit_pred1_np).max(), abs(fit_pred2_np).max()))
    resid_abs_max = float(max(abs(resid1).max(), abs(resid2).max(), 1e-6))

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.30)

    def show(ax, im, title, **kw):
        m = ax.imshow(im, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    show(fig.add_subplot(gs[0, 0]), target, 'target',
         cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
    show(fig.add_subplot(gs[0, 1]), ref1,   f'ref1 (shift {pre["shift1"]})',
         cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
    show(fig.add_subplot(gs[0, 2]), ref2,   f'ref2 (shift {pre["shift2"]})',
         cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
    show(fig.add_subplot(gs[0, 3]), fit_psf_np, 'fitted PSF (sum=1)',
         cmap='hot')

    show(fig.add_subplot(gs[1, 0]), diff1, 'diff1 = target - ref1',
         cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max)
    show(fig.add_subplot(gs[1, 1]), diff2, 'diff2 = target - ref2',
         cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max)
    show(fig.add_subplot(gs[1, 2]), resid1, 'residual1 = diff1 - pred1',
         cmap='RdBu_r', vmin=-resid_abs_max, vmax=resid_abs_max)
    show(fig.add_subplot(gs[1, 3]), resid2, 'residual2 = diff2 - pred2',
         cmap='RdBu_r', vmin=-resid_abs_max, vmax=resid_abs_max)

    plt.suptitle(f'{pre["defect_id"]} @ ({pre["x"]}, {pre["y"]})', fontsize=12)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/30ea_testing/bad',
                        help='Directory with DefectID*.tiff triplets.')
    parser.add_argument('--output_dir', default='src_invertfit/fitted',
                        help='Where to write fitted_theta.json + vis/.')
    parser.add_argument('--n_starts', type=int, default=10)
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lambda_alpha', type=float, default=1e-3)
    parser.add_argument('--no_alpha', action='store_true',
                        help='Disable alpha fit (set alpha1=alpha2=0 fixed).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=32)
    parser.add_argument('--reg_crop_size', type=int, default=96)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N tiffs (0 = all).')
    parser.add_argument('--no_vis', action='store_true',
                        help='Skip per-defect PNG generation (faster).')
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0
              else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

    fwd_cfg = ForwardConfig(psf_size=args.psf_size, crop_size=args.crop_size,
                            pol_type='linX', device=device)
    fit_cfg = ThreeChannelFitConfig(
        fit_param_names=DEFAULT_FIT_PARAMS,
        n_starts=args.n_starts, n_iters=args.n_iters, lr=args.lr,
        fit_alpha=(not args.no_alpha),
        lambda_alpha=args.lambda_alpha,
    )

    paths = sorted(glob.glob(os.path.join(args.input_dir, 'DefectID*.tiff')))
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f'No DefectID*.tiff under {args.input_dir}')

    print(f'Processing {len(paths)} defect tiffs '
          f'({fit_cfg.n_starts} starts × {fit_cfg.n_iters} iters each)\n')

    rng = np.random.default_rng(args.seed)
    fitted = {}
    summary_lines = []

    for i, path in enumerate(paths):
        pre = preprocess_one(path,
                             fit_crop_size=args.crop_size,
                             reg_crop_size=args.reg_crop_size)
        diff1 = torch.from_numpy(pre['diff1'].astype(np.float32)).to(device)
        diff2 = torch.from_numpy(pre['diff2'].astype(np.float32)).to(device)

        result = fit_three_channel(diff1, diff2, DEFAULT_FIXED,
                                   fit_cfg, fwd_cfg, rng=rng)

        # Compute predicted PSF + scaled diffs for visualization & residual report
        theta_t = make_theta(result['theta'], device=device)
        with torch.no_grad():
            psf_norm = differentiable_forward(theta_t, fwd_cfg)
        psf_np = psf_norm.cpu().numpy()
        I = result['I']
        a1 = result['alpha1']; a2 = result['alpha2']
        pred1_np = psf_np * I * (1.0 - a1)
        pred2_np = psf_np * I * (1.0 - a2)

        # Per-pixel residual (no weight) — for human-readable diagnostic
        r1 = np.abs(pre['diff1'] - pred1_np)
        r2 = np.abs(pre['diff2'] - pred2_np)
        per_pixel_min = np.minimum(r1, r2)

        record = {
            'defect_id': pre['defect_id'],
            'x': pre['x'], 'y': pre['y'],
            'shift1': list(pre['shift1']),
            'shift2': list(pre['shift2']),
            'theta': result['theta'],
            'I': result['I'],
            'alpha1': result['alpha1'],
            'alpha2': result['alpha2'],
            'loss': result['loss'],
            'residual_l1_total': float(per_pixel_min.sum()),
            'residual_l1_per_pixel': float(per_pixel_min.mean()),
            'diff1_peak': float(pre['diff1'].max()),
            'diff2_peak': float(pre['diff2'].max()),
            'diff1_std': float(pre['diff1'].std()),
            'diff2_std': float(pre['diff2'].std()),
        }
        fitted[pre['defect_id']] = record

        # Print + accumulate summary line
        line = (f'[{i+1:2d}/{len(paths)}] {pre["defect_id"]}: '
                f'loss={result["loss"]:.3e}  '
                f'I={result["I"]:.1f}  '
                f'a=({a1:.3f},{a2:.3f})  '
                f'resid_pp={record["residual_l1_per_pixel"]:.3f}  '
                f'shift1={pre["shift1"]} shift2={pre["shift2"]}')
        print(line)
        summary_lines.append(line)

        if not args.no_vis:
            out_png = os.path.join(vis_dir, f'{pre["defect_id"]}.png')
            render_diagnostic(pre, pred1_np, pred2_np, psf_np, out_png)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'fitted_theta.json')
    with open(json_path, 'w') as f:
        json.dump({
            'meta': {
                'input_dir': args.input_dir,
                'fixed_params': DEFAULT_FIXED,
                'fit_param_names': list(DEFAULT_FIT_PARAMS),
                'fit_config': {
                    'n_starts': fit_cfg.n_starts,
                    'n_iters': fit_cfg.n_iters,
                    'lr': fit_cfg.lr,
                    'fit_alpha': fit_cfg.fit_alpha,
                    'lambda_alpha': fit_cfg.lambda_alpha,
                },
                'forward': {
                    'psf_size': fwd_cfg.psf_size,
                    'crop_size': fwd_cfg.crop_size,
                    'pol_type': fwd_cfg.pol_type,
                },
            },
            'fits': fitted,
        }, f, indent=2)
    print(f'\nWrote {json_path}')

    # Aggregate stats
    losses = [f['loss'] for f in fitted.values()]
    resids_pp = [f['residual_l1_per_pixel'] for f in fitted.values()]
    Is = [f['I'] for f in fitted.values()]
    a1s = [f['alpha1'] for f in fitted.values()]
    a2s = [f['alpha2'] for f in fitted.values()]

    aggregate = [
        '',
        '=== Aggregate over {} fits ==='.format(len(fitted)),
        f'  loss          : median {np.median(losses):.3e}, max {np.max(losses):.3e}',
        f'  resid_pp      : median {np.median(resids_pp):.3f}, max {np.max(resids_pp):.3f}',
        f'  I (intensity) : median {np.median(Is):.1f}, range [{np.min(Is):.1f}, {np.max(Is):.1f}]',
        f'  alpha1        : median {np.median(a1s):.3f}, range [{np.min(a1s):.3f}, {np.max(a1s):.3f}]',
        f'  alpha2        : median {np.median(a2s):.3f}, range [{np.min(a2s):.3f}, {np.max(a2s):.3f}]',
    ]
    for line in aggregate:
        print(line)
    summary_lines.extend(aggregate)

    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
