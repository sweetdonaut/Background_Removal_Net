"""Fit the inverse model on a directory of real PSF triplets.

For each tiff in --input_dir matching `DefectID###.#X,Y.tiff`:
  1. preprocess_one() → diff1, diff2, target/ref crops
  2. fit_three_channel() → theta, I, alpha, (cy, cx), loss
  3. Save fitted dict to JSON keyed by defect_id
  4. Render an 8-panel diagnostic PNG

Outputs (under --output_dir, default src_invertfit/fitted/):
  fitted_theta.json   ← all fits in one file (export_yaml.py reads this)
  vis/<DefectID>.png  ← per-defect visualization
  summary.txt         ← human-readable per-defect line + aggregate stats

Everything that controls the fit (forward grid, oversample, which physics
params to fix vs fit, alpha/shift toggles, optimizer hyperparams) lives in
the --config yaml; CLI flags here are only operational (paths, GPU, seed).
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

from config import load_fit_config, summarize  # noqa: E402
from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import ThreeChannelFitConfig, fit_three_channel  # noqa: E402
from preprocess import preprocess_one  # noqa: E402


def render_diagnostic(pre, fit_pred1_np, fit_pred2_np, fit_psf_np, out_path):
    """8-panel visualization: 3 raw + 2 diffs + 1 fitted PSF + 2 residuals."""
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
    show(fig.add_subplot(gs[0, 1]), ref1, f'ref1 (shift {pre["shift1"]})',
         cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
    show(fig.add_subplot(gs[0, 2]), ref2, f'ref2 (shift {pre["shift2"]})',
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


def build_configs(cfg_data, device):
    fwd = cfg_data['forward']
    fit = cfg_data['fit']
    fwd_cfg = ForwardConfig(
        psf_size=fwd['psf_size'],
        crop_size=fwd['crop_size'],
        pol_type=fwd['pol_type'],
        oversample=fwd['pixel_oversample'],
        mask_sharpness=fwd['mask_sharpness'],
        device=device,
    )
    fit_cfg = ThreeChannelFitConfig(
        fit_param_names=cfg_data['fit_param_names'],
        init_ranges=dict(cfg_data['fit_init_ranges']),
        n_starts=int(fit['n_starts']),
        n_iters=int(fit['n_iters']),
        lr=float(fit['lr']),
        fit_alpha=bool(fit['alpha']),
        fit_shift=bool(fit['shift']),
        lambda_alpha=float(fit['lambda_alpha']),
        lambda_shift=float(fit['lambda_shift']),
        radial_sigma_frac=float(fit['radial_sigma_frac']),
        sign_flip_init=bool(fit['sign_flip_init']),
        init_alpha_z=float(fit['init_alpha_z']),
    )
    return fwd_cfg, fit_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src_invertfit/fit_configs/default.yaml',
                        help='YAML controlling forward grid, physics params (fixed/fit), '
                             'and fit hyperparams. Mirrors src_core/defects/*.yaml schema. '
                             'See src_invertfit/fit_configs/default.yaml for an annotated example.')
    parser.add_argument('--input_dir', default='data/30ea_testing/bad',
                        help='Directory with DefectID###.#X,Y.tiff triplets.')
    parser.add_argument('--output_dir', default='src_invertfit/fitted',
                        help='Where to write fitted_theta.json + vis/.')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for multi-start init (does NOT affect torch).')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reg_crop_size', type=int, default=96,
                        help='Wider crop for sub-pixel registration (default 96).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N tiffs (0 = all).')
    parser.add_argument('--no_vis', action='store_true',
                        help='Skip per-defect PNG generation (faster).')
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0
              else 'cpu')
    print(f'Device: {device}')

    cfg_data = load_fit_config(args.config)
    print(summarize(cfg_data))
    print()

    fwd_cfg, fit_cfg = build_configs(cfg_data, device)

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

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
                             fit_crop_size=fwd_cfg.crop_size,
                             reg_crop_size=args.reg_crop_size)
        diff1 = torch.from_numpy(pre['diff1'].astype(np.float32)).to(device)
        diff2 = torch.from_numpy(pre['diff2'].astype(np.float32)).to(device)

        result = fit_three_channel(diff1, diff2, cfg_data['fixed_params'],
                                   fit_cfg, fwd_cfg, rng=rng)

        # Render fitted PSF (with sub-pixel shift if present) for visualization.
        theta_for_render = dict(result['theta'])
        if fit_cfg.fit_shift:
            theta_for_render['cy'] = result['cy']
            theta_for_render['cx'] = result['cx']
        theta_t = make_theta(theta_for_render, device=device)
        with torch.no_grad():
            psf_norm = differentiable_forward(theta_t, fwd_cfg)
        psf_np = psf_norm.cpu().numpy()
        I = result['I']
        a1 = result['alpha1']; a2 = result['alpha2']
        pred1_np = psf_np * I * (1.0 - a1)
        pred2_np = psf_np * I * (1.0 - a2)

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
            'cy': result['cy'],
            'cx': result['cx'],
            'loss': result['loss'],
            'residual_l1_total': float(per_pixel_min.sum()),
            'residual_l1_per_pixel': float(per_pixel_min.mean()),
            'diff1_peak': float(pre['diff1'].max()),
            'diff2_peak': float(pre['diff2'].max()),
            'diff1_std': float(pre['diff1'].std()),
            'diff2_std': float(pre['diff2'].std()),
        }
        fitted[pre['defect_id']] = record

        line = (f'[{i+1:2d}/{len(paths)}] {pre["defect_id"]}: '
                f'loss={result["loss"]:.3e}  '
                f'I={result["I"]:.1f}  '
                f'a=({a1:.3f},{a2:.3f})  '
                f'shift_sub=({result["cy"]:+.2f},{result["cx"]:+.2f})  '
                f'resid_pp={record["residual_l1_per_pixel"]:.3f}')
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
                'config': cfg_data['source_yaml'],
                'forward': cfg_data['forward'],
                'fixed_params': cfg_data['fixed_params'],
                'fit_param_names': list(cfg_data['fit_param_names']),
                'fit_init_ranges': {k: list(v) for k, v in cfg_data['fit_init_ranges'].items()},
                'fit_hyperparams': cfg_data['fit'],
            },
            'fits': fitted,
        }, f, indent=2)
    print(f'\nWrote {json_path}')

    losses = [f['loss'] for f in fitted.values()]
    resids_pp = [f['residual_l1_per_pixel'] for f in fitted.values()]
    Is = [f['I'] for f in fitted.values()]
    a1s = [f['alpha1'] for f in fitted.values()]
    a2s = [f['alpha2'] for f in fitted.values()]
    cys = [f['cy'] for f in fitted.values()]
    cxs = [f['cx'] for f in fitted.values()]

    aggregate = [
        '',
        '=== Aggregate over {} fits ==='.format(len(fitted)),
        f'  loss          : median {np.median(losses):.3e}, max {np.max(losses):.3e}',
        f'  resid_pp      : median {np.median(resids_pp):.3f}, max {np.max(resids_pp):.3f}',
        f'  I (intensity) : median {np.median(Is):.1f}, range [{np.min(Is):.1f}, {np.max(Is):.1f}]',
        f'  alpha1        : median {np.median(a1s):.3f}, range [{np.min(a1s):.3f}, {np.max(a1s):.3f}]',
        f'  alpha2        : median {np.median(a2s):.3f}, range [{np.min(a2s):.3f}, {np.max(a2s):.3f}]',
        f'  cy            : median {np.median(cys):+.3f}, range [{np.min(cys):+.3f}, {np.max(cys):+.3f}]',
        f'  cx            : median {np.median(cxs):+.3f}, range [{np.min(cxs):+.3f}, {np.max(cxs):+.3f}]',
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
