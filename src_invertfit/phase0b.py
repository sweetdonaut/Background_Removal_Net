"""3-channel synth-on-synth regression test for fit_three_channel.

Counterpart to phase0a but exercises the joint diff-min loss + sigmoid alpha
+ I-scale machinery used by fit_real.py. Doesn't touch real data — runs on
synthesized triplets where alpha_true is known.

For each trial:
  - Build target = PSF*I + photon noise; refs = independent Gaussian nuisance
    (alpha_true = 0). Per-channel noise is independent (matching real captures).
  - Compute diff1 = target - ref1, diff2 = target - ref2.
  - Run fit_three_channel.
  - Compare recovered theta + I + alpha to ground truth.

Pass criterion: median PSF L1 in same ballpark as phase0a noise floor (~0.05
at default SNR). Higher → bug in the 3-channel loss / radial weight / I scale.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import ThreeChannelFitConfig, fit_three_channel  # noqa: E402


FIXED_PARAMS = {
    'outer_r': 60.0,
    'epsilon': 0.0,
    'ellipticity': 0.0,
    'ellip_angle': 0.0,
    'na': 0.95,
}

THETA_TRUE_RANGES = {
    'defocus': (-1.5, 1.5),
    'astig_x': (-1.0, 1.0),
    'astig_y': (-1.0, 1.0),
    'coma_x': (-0.8, 0.8),
    'coma_y': (-0.8, 0.8),
    'spherical': (-0.8, 0.8),
    'trefoil_x': (-0.8, 0.8),
    'trefoil_y': (-0.8, 0.8),
}


def sample_theta_true(rng):
    th = dict(FIXED_PARAMS)
    for name, (lo, hi) in THETA_TRUE_RANGES.items():
        th[name] = float(rng.uniform(lo, hi))
    return th


def build_synth_triplet(psf_clean_np, I_true, ref_nuisance_sigma, rng):
    """Make a synthetic 3-channel observation around the ground-truth PSF.

    target  = PSF * I + photon_noise(target)
    ref_k   = nuisance_k + photon_noise(ref_k)   (alpha_true = 0)

    photon_noise is per-channel independent (matching real captures with
    independent sensor reads). The diffs (target - ref) have:
        diff = PSF*I + (target_noise - ref_noise) - nuisance
    which is exactly what fit_three_channel expects.
    """
    target_signal = psf_clean_np * I_true
    target = rng.poisson(np.maximum(target_signal, 0)).astype(np.float64)
    # ref noise: zero-mean Gaussian nuisance with the same scale
    ref1 = rng.normal(0, ref_nuisance_sigma, target.shape)
    ref2 = rng.normal(0, ref_nuisance_sigma, target.shape)
    return target, ref1, ref2


def evaluate_recovery(theta_true, theta_fit, I_true, I_fit):
    rows = []
    for name, (lo, hi) in THETA_TRUE_RANGES.items():
        true_v = theta_true[name]
        fit_v = theta_fit[name]
        rows.append((name, true_v, fit_v, abs(fit_v - true_v),
                     abs(fit_v - true_v) / (hi - lo)))
    I_err_rel = abs(I_fit - I_true) / max(abs(I_true), 1e-6)
    return rows, I_err_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--n_starts', type=int, default=8)
    parser.add_argument('--n_iters', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--I_true', type=float, default=5000.0,
                        help='Ground-truth defect intensity scale (PSF*I total).')
    parser.add_argument('--ref_nuisance_sigma', type=float, default=2.0,
                        help='Per-pixel std of ref-side nuisance.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=32)
    parser.add_argument('--fit_alpha', action='store_true',
                        help='Enable alpha fit (defaults off — alpha_true=0 here).')
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0
              else 'cpu')
    print(f'Device: {device}')

    fwd_cfg = ForwardConfig(psf_size=args.psf_size, crop_size=args.crop_size,
                            pol_type='linX', device=device)
    fit_cfg = ThreeChannelFitConfig(
        fit_param_names=tuple(THETA_TRUE_RANGES.keys()),
        n_starts=args.n_starts, n_iters=args.n_iters, lr=args.lr,
        fit_alpha=args.fit_alpha,
    )

    rng = np.random.default_rng(args.seed)
    psf_l1s = []
    final_losses = []
    I_errs = []
    alpha1s = []
    alpha2s = []

    print(f'\n{args.n_trials} trials, '
          f'{args.n_starts}+sign-flip starts × {args.n_iters} iters per trial\n')

    for trial in range(args.n_trials):
        theta_true = sample_theta_true(rng)
        theta_true_t = make_theta(theta_true, device=device)

        with torch.no_grad():
            psf_true = differentiable_forward(theta_true_t, fwd_cfg)
        psf_true_np = psf_true.cpu().numpy()

        target, ref1, ref2 = build_synth_triplet(
            psf_true_np, args.I_true, args.ref_nuisance_sigma, rng)

        diff1 = torch.from_numpy((target - ref1).astype(np.float32)).to(device)
        diff2 = torch.from_numpy((target - ref2).astype(np.float32)).to(device)

        result = fit_three_channel(diff1, diff2, FIXED_PARAMS, fit_cfg, fwd_cfg, rng=rng)

        theta_fit_t = make_theta(result['theta'], device=device)
        with torch.no_grad():
            psf_fit = differentiable_forward(theta_fit_t, fwd_cfg)
        l1 = (psf_fit - psf_true).abs().sum().item()
        psf_l1s.append(l1)
        final_losses.append(result['loss'])
        rows, I_err_rel = evaluate_recovery(theta_true, result['theta'],
                                            args.I_true, result['I'])
        I_errs.append(I_err_rel)
        alpha1s.append(result['alpha1'])
        alpha2s.append(result['alpha2'])

        max_rel = max(r[4] for r in rows)
        print(f'Trial {trial+1:2d}/{args.n_trials}: '
              f'loss={result["loss"]:.4e}  '
              f'PSF_L1={l1:.4f}  '
              f'I_fit={result["I"]:.0f} ({I_err_rel*100:+.1f}%)  '
              f'alpha=({result["alpha1"]:.3f},{result["alpha2"]:.3f})  '
              f'theta_max_rel={max_rel:.3f}')

    print('\n=== PSF reconstruction (3-channel fit) ===')
    print(f'  median L1: {np.median(psf_l1s):.4f}')
    print(f'  best  L1:  {np.min(psf_l1s):.4f}')
    print(f'  worst L1:  {np.max(psf_l1s):.4f}')

    print('\n=== Intensity recovery ===')
    print(f'  median |I_err|/I_true: {np.median(I_errs)*100:.2f}%')
    print(f'  worst:                 {np.max(I_errs)*100:.2f}%')

    print('\n=== Alpha fit (true alpha = 0) ===')
    print(f'  alpha1 median: {np.median(alpha1s):.4f}  max: {np.max(alpha1s):.4f}')
    print(f'  alpha2 median: {np.median(alpha2s):.4f}  max: {np.max(alpha2s):.4f}')

    print('\n=== Final-loss summary ===')
    print(f'  median: {np.median(final_losses):.4e}')
    print(f'  best:   {np.min(final_losses):.4e}')
    print(f'  worst:  {np.max(final_losses):.4e}')

    median_l1 = np.median(psf_l1s)
    if median_l1 < 0.02:
        verdict = 'EXCELLENT — 3-channel fit recovers near-identical PSFs'
    elif median_l1 < 0.10:
        verdict = 'GOOD — 3-channel fit converges, comparable to Phase 0a noise floor'
    elif median_l1 < 0.30:
        verdict = 'MARGINAL — 3-channel loss converges but PSF drift visible'
    else:
        verdict = 'POOR — 3-channel pipeline likely has a bug'
    print(f'\nVerdict: {verdict}')


if __name__ == '__main__':
    main()
