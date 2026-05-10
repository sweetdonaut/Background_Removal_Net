"""Single-channel synth-on-synth regression test for the inverse fit pipeline.

Doesn't touch real data — useful when verifying that the multi-start
optimizer or the torch forward port hasn't drifted.

For each trial:
  1. Sample theta_true from the [a, b] entries in the config yaml (anything
     with min != max). Fixed entries ([v, v]) stay at v.
  2. Generate noise-free PSF via differentiable_forward.
  3. Add Poisson + Gaussian noise.
  4. Run multi-start fit_one with the same [a, b] init ranges.
  5. Compare recovered theta to theta_true; report PSF L1 reconstruction.

Pass criterion: median PSF L1 < 0.02 under realistic SNR. PSF L1 bypasses
the |FFT|^2 phase-retrieval twin ambiguity (twin solutions produce
identical PSFs); per-parameter recovery may look "off" while PSF is fine.

Usage from project root:
    python src_invertfit/phase0a.py --config src_invertfit/fit_configs/default.yaml \\
        --n_trials 10
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import load_fit_config, summarize  # noqa: E402
from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import FitConfig, fit_one  # noqa: E402


def sample_theta_true(cfg_data, rng):
    th = dict(cfg_data['fixed_params'])
    for name, (lo, hi) in cfg_data['fit_init_ranges'].items():
        th[name] = float(rng.uniform(lo, hi))
    return th


def add_realistic_noise(psf_norm, brightness, gaussian_sigma, rng):
    """Poisson photon noise scaled by brightness, plus optional Gaussian read."""
    scaled = psf_norm * brightness
    noisy = rng.poisson(np.maximum(scaled, 0)).astype(np.float64)
    if gaussian_sigma > 0:
        sqrt2 = float(np.sqrt(2.0) * gaussian_sigma)
        noisy = noisy + rng.normal(0, sqrt2, scaled.shape)
    return noisy


def evaluate_recovery(theta_true, theta_fit, fit_init_ranges):
    rows = []
    for name, (lo, hi) in fit_init_ranges.items():
        true_v = theta_true[name]
        fit_v = theta_fit[name]
        rows.append((name, true_v, fit_v, abs(fit_v - true_v),
                     abs(fit_v - true_v) / (hi - lo)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src_invertfit/fit_configs/default.yaml')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--brightness', type=float, default=5000.0)
    parser.add_argument('--gaussian_sigma', type=float, default=1.5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--noise_off', action='store_true',
                        help='Skip noise (pure noise-free recovery test).')
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0
              else 'cpu')
    print(f'Device: {device}')
    cfg_data = load_fit_config(args.config)
    print(summarize(cfg_data))
    print(f'Noise: {"off" if args.noise_off else f"brightness={args.brightness}, sigma={args.gaussian_sigma}"}')

    fwd = cfg_data['forward']
    fwd_cfg = ForwardConfig(
        psf_size=fwd['psf_size'], crop_size=fwd['crop_size'],
        pol_type=fwd['pol_type'], oversample=fwd['pixel_oversample'],
        mask_sharpness=fwd['mask_sharpness'], device=device)
    fit = cfg_data['fit']
    fit_cfg = FitConfig(
        fit_param_names=cfg_data['fit_param_names'],
        init_ranges=dict(cfg_data['fit_init_ranges']),
        n_starts=int(fit['n_starts']),
        n_iters=int(fit['n_iters']),
        lr=float(fit['lr']),
    )

    rng = np.random.default_rng(args.seed)
    psf_l1s = []
    final_losses = []

    print(f'\n{args.n_trials} trials, {fit_cfg.n_starts} starts × {fit_cfg.n_iters} iters\n')

    for trial in range(args.n_trials):
        theta_true = sample_theta_true(cfg_data, rng)
        theta_true_t = make_theta(theta_true, device=device)

        with torch.no_grad():
            psf_true = differentiable_forward(theta_true_t, fwd_cfg)
        psf_true_np = psf_true.cpu().numpy()

        if args.noise_off:
            obs_np = psf_true_np
        else:
            obs_np = add_realistic_noise(psf_true_np, args.brightness,
                                          args.gaussian_sigma, rng)

        obs_t = torch.from_numpy(obs_np.astype(np.float32)).to(device)
        result = fit_one(obs_t, cfg_data['fixed_params'], fit_cfg, fwd_cfg, rng=rng)

        theta_fit_t = make_theta(result['theta'], device=device)
        with torch.no_grad():
            psf_fit = differentiable_forward(theta_fit_t, fwd_cfg)
        l1 = (psf_fit - psf_true).abs().sum().item()
        psf_l1s.append(l1)
        final_losses.append(result['loss'])

        rows = evaluate_recovery(theta_true, result['theta'], cfg_data['fit_init_ranges'])
        max_rel = max(r[4] for r in rows) if rows else 0.0
        print(f'Trial {trial+1:2d}/{args.n_trials}: '
              f'loss={result["loss"]:.4e}  PSF_L1={l1:.4f}  '
              f'theta_max_rel={max_rel:.3f}')

    print('\n=== PSF reconstruction ===')
    print('  Both PSFs sum=1; L1 ∈ [0, 2]; <0.02 = visually indistinguishable')
    print(f'  median L1: {np.median(psf_l1s):.4f}')
    print(f'  best  L1:  {np.min(psf_l1s):.4f}')
    print(f'  worst L1:  {np.max(psf_l1s):.4f}')

    print('\n=== Final-loss summary ===')
    print(f'  median: {np.median(final_losses):.4e}')
    print(f'  best:   {np.min(final_losses):.4e}')
    print(f'  worst:  {np.max(final_losses):.4e}')

    median_l1 = np.median(psf_l1s)
    if median_l1 < 0.005:
        verdict = 'EXCELLENT — fit recovers visually identical PSFs'
    elif median_l1 < 0.02:
        verdict = 'GOOD — fit recovers near-identical PSFs (sub-percent error)'
    elif median_l1 < 0.10:
        verdict = 'MARGINAL — fit converges but reconstruction has visible drift'
    else:
        verdict = 'POOR — fit fails to recover PSFs, pipeline likely broken'
    print(f'\nVerdict: {verdict}')


if __name__ == '__main__':
    main()
