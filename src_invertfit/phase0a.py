"""Single-channel synth-on-synth regression test for the inverse fit pipeline.

Doesn't touch real data — useful when verifying that the multi-start
optimizer or the torch forward port hasn't drifted.

For each trial:
  1. Sample theta_true (Zernike subset) from optical.yaml-style prior ranges,
     keep geometric params fixed at type4_vector.yaml defaults.
  2. Generate noise-free PSF via the torch differentiable_forward.
  3. Add Poisson + Gaussian noise matching src_core/generate_psf.py settings.
  4. Run multi-start fit_one on the noisy observation.
  5. Compare recovered theta to theta_true; aggregate per-parameter MAE.

Pass criterion: median PSF L1 < 0.02 under realistic SNR. PSF L1 bypasses
the |FFT|^2 phase-retrieval twin ambiguity (twin solutions produce
identical PSFs); per-parameter recovery may look "off" while PSF is fine.

Usage from project root:
    python src_invertfit/phase0a.py --n_trials 10 --n_starts 5
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import FitConfig, fit_one  # noqa: E402


# Aligned with src_core/defects/type4_vector.yaml
FIXED_PARAMS = {
    'outer_r': 60.0,
    'epsilon': 0.0,
    'ellipticity': 0.0,
    'ellip_angle': 0.0,
    'na': 0.95,
}


# Aligned with src_search/search_configs/optical.yaml + trefoil
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


def add_realistic_noise(psf_norm, brightness, background, gaussian_sigma, rng):
    """Add noise matching what inverse fit will see at production time.

    Inverse fit operates on a 3-channel `diff = target - ref` residual: the
    flat sensor background, dark current, and shared wafer texture all cancel
    in the subtraction, leaving (PSF signal) + (target_noise - ref_noise).
    So the synth observation should NOT carry an absolute background floor —
    that would make Phase 0a a different problem from the real one.

    What we keep: Poisson shot noise scaled by photon count (brightness * PSF),
    and a per-pixel Gaussian term for residual sensor read noise that doesn't
    fully cancel between captures (we use sqrt(2)*gaussian_sigma to model the
    independent read noise from two captures combining in the diff).
    """
    scaled = psf_norm * brightness
    noisy = rng.poisson(np.maximum(scaled, 0)).astype(np.float64)
    if gaussian_sigma > 0:
        sqrt2_sigma = float(np.sqrt(2.0) * gaussian_sigma)
        noisy = noisy + rng.normal(0, sqrt2_sigma, scaled.shape)
    if background > 0:
        # Optional: extra residual offset noise (low magnitude). Most subtraction
        # artifacts in real diffs are spatially structured, not constant; we
        # model that part separately in Phase 0b. Keep this near zero in 0a.
        noisy = noisy + rng.normal(0, background, scaled.shape)
    return noisy  # may have small negatives from gaussian; fit normalize() clamps


def evaluate_recovery(theta_true, theta_fit):
    rows = []
    for name, (lo, hi) in THETA_TRUE_RANGES.items():
        true_v = theta_true[name]
        fit_v = theta_fit[name]
        abs_err = abs(fit_v - true_v)
        rel_err = abs_err / (hi - lo)
        rows.append((name, true_v, fit_v, abs_err, rel_err))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--n_starts', type=int, default=5)
    parser.add_argument('--n_iters', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--brightness', type=float, default=5000.0)
    parser.add_argument('--background', type=float, default=5.0)
    parser.add_argument('--gaussian_sigma', type=float, default=1.5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--noise_off', action='store_true',
                        help='Skip noise addition (pure noise-free recovery test).')
    parser.add_argument('--psf_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=32)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'
    print(f'Device: {device}')
    print(f'Forward grid: psf_size={args.psf_size}, crop_size={args.crop_size}')
    print(f'Noise: {"off" if args.noise_off else f"brightness={args.brightness}, bg={args.background}, sigma={args.gaussian_sigma}"}')

    fwd_cfg = ForwardConfig(psf_size=args.psf_size, crop_size=args.crop_size,
                            pol_type='linX', device=device)
    fit_cfg = FitConfig(
        fit_param_names=tuple(THETA_TRUE_RANGES.keys()),
        n_starts=args.n_starts, n_iters=args.n_iters, lr=args.lr,
    )

    rng = np.random.default_rng(args.seed)
    all_rows = []
    final_losses = []
    psf_recon_errs = []

    print(f'\nRunning {args.n_trials} trials '
          f'({args.n_starts} starts x {args.n_iters} iters per trial)\n')

    for trial in range(args.n_trials):
        theta_true = sample_theta_true(rng)
        theta_true_t = make_theta(theta_true, device=device)

        with torch.no_grad():
            psf_true = differentiable_forward(theta_true_t, fwd_cfg)
        psf_true_np = psf_true.cpu().numpy()

        if args.noise_off:
            obs_np = psf_true_np
        else:
            obs_np = add_realistic_noise(
                psf_true_np, args.brightness, args.background,
                args.gaussian_sigma, rng)

        obs_t = torch.from_numpy(obs_np.astype(np.float32)).to(device)
        result = fit_one(obs_t, FIXED_PARAMS, fit_cfg, fwd_cfg, rng=rng)

        # PSF reconstruction error: forward(fit_theta) vs forward(true_theta).
        # Bypasses twin-solution ambiguity since two thetas with same PSF look
        # identical on this metric.
        theta_fit_t = make_theta(result['theta'], device=device)
        with torch.no_grad():
            psf_fit = differentiable_forward(theta_fit_t, fwd_cfg)
        l1 = (psf_fit - psf_true).abs().sum().item()  # both sum=1 → l1 ∈ [0, 2]
        psf_recon_errs.append(l1)

        rows = evaluate_recovery(theta_true, result['theta'])
        all_rows.append(rows)
        final_losses.append(result['loss'])

        max_rel = max(r[4] for r in rows)
        mean_rel = sum(r[4] for r in rows) / len(rows)
        print(f'Trial {trial+1:2d}/{args.n_trials}: '
              f'loss={result["loss"]:.4e}  '
              f'PSF_L1={l1:.4f}  '
              f'theta_mean_rel={mean_rel:.3f}  theta_max_rel={max_rel:.3f}')

    # PSF reconstruction is the primary success metric — it bypasses the
    # phase-retrieval twin-solution ambiguity (twins produce identical PSFs).
    print('\n=== PSF reconstruction (forward(fit) vs forward(true)) ===')
    print('  Both PSFs are sum=1; L1 ∈ [0, 2]; <0.02 = visually indistinguishable')
    print(f'  median L1: {np.median(psf_recon_errs):.4f}')
    print(f'  best  L1:  {np.min(psf_recon_errs):.4f}')
    print(f'  worst L1:  {np.max(psf_recon_errs):.4f}')

    # Theta recovery — informational only; twin solutions inflate this without
    # affecting PSF quality.
    print('\n=== Per-parameter recovery (informational; twin solutions OK) ===')
    print(f'{"param":<12s}  {"prior":>14s}  {"mae":>8s}  {"rel_mae":>8s}  {"max_rel":>8s}')
    print('-' * 56)
    for i, name in enumerate(THETA_TRUE_RANGES):
        abs_errs = [trial_rows[i][3] for trial_rows in all_rows]
        rel_errs = [trial_rows[i][4] for trial_rows in all_rows]
        lo, hi = THETA_TRUE_RANGES[name]
        mae = sum(abs_errs) / len(abs_errs)
        rel_mae = sum(rel_errs) / len(rel_errs)
        max_rel = max(rel_errs)
        print(f'{name:<12s}  [{lo:>5.2f},{hi:>5.2f}]  '
              f'{mae:>8.4f}  {rel_mae:>8.4f}  {max_rel:>8.4f}')

    print('\n=== Final-loss summary ===')
    print(f'  median: {np.median(final_losses):.4e}')
    print(f'  best:   {np.min(final_losses):.4e}')
    print(f'  worst:  {np.max(final_losses):.4e}')

    # Verdict based on PSF reconstruction (the metric that matters for our use)
    median_l1 = np.median(psf_recon_errs)
    if median_l1 < 0.005:
        verdict = 'EXCELLENT — fit recovers visually identical PSFs'
    elif median_l1 < 0.02:
        verdict = 'GOOD — fit recovers near-identical PSFs (sub-percent error)'
    elif median_l1 < 0.10:
        verdict = 'MARGINAL — fit converges but reconstruction has visible drift'
    else:
        verdict = 'POOR — fit fails to recover PSFs, pipeline likely broken'
    print(f'\nVerdict (by PSF L1): {verdict}')


if __name__ == '__main__':
    main()
