"""3-channel synth-on-synth regression test for fit_three_channel.

Counterpart to phase0a but exercises the joint diff-min loss + sigmoid alpha
+ I-scale machinery used by fit_real.py. Doesn't touch real data — runs on
synthesized triplets where alpha_true and shift_true are known.

For each trial:
  - Build target = PSF*I + photon noise; refs = independent Gaussian nuisance
    (alpha_true = 0, shift_true = 0). Per-channel noise is independent.
  - Compute diff1 = target - ref1, diff2 = target - ref2.
  - Run fit_three_channel with config-driven hyperparams (but with alpha
    and shift fits forced off, since ground truth is known to be 0).
  - Compare recovered theta + I to ground truth.

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

from config import load_fit_config, summarize  # noqa: E402
from forward import ForwardConfig, differentiable_forward, make_theta  # noqa: E402
from inverse_fit import ThreeChannelFitConfig, fit_three_channel  # noqa: E402


def sample_theta_true(cfg_data, rng):
    th = dict(cfg_data['fixed_params'])
    for name, (lo, hi) in cfg_data['fit_init_ranges'].items():
        th[name] = float(rng.uniform(lo, hi))
    return th


def build_synth_triplet(psf_clean_np, I_true, ref_nuisance_sigma, rng):
    target_signal = psf_clean_np * I_true
    target = rng.poisson(np.maximum(target_signal, 0)).astype(np.float64)
    ref1 = rng.normal(0, ref_nuisance_sigma, target.shape)
    ref2 = rng.normal(0, ref_nuisance_sigma, target.shape)
    return target, ref1, ref2


def evaluate_recovery(theta_true, theta_fit, fit_init_ranges, I_true, I_fit):
    rows = []
    for name, (lo, hi) in fit_init_ranges.items():
        true_v = theta_true[name]
        fit_v = theta_fit[name]
        rows.append((name, true_v, fit_v, abs(fit_v - true_v),
                     abs(fit_v - true_v) / (hi - lo)))
    I_err_rel = abs(I_fit - I_true) / max(abs(I_true), 1e-6)
    return rows, I_err_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src_invertfit/fit_configs/default.yaml')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--I_true', type=float, default=5000.0)
    parser.add_argument('--ref_nuisance_sigma', type=float, default=2.0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--enable_alpha_fit', action='store_true',
                        help='Override config: turn on alpha fit. Default off because '
                             'phase0b ground truth has alpha_true = 0.')
    args = parser.parse_args()

    device = (f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0
              else 'cpu')
    print(f'Device: {device}')
    cfg_data = load_fit_config(args.config)
    print(summarize(cfg_data))

    fwd = cfg_data['forward']
    fwd_cfg = ForwardConfig(
        psf_size=fwd['psf_size'], crop_size=fwd['crop_size'],
        pol_type=fwd['pol_type'], oversample=fwd['pixel_oversample'],
        mask_sharpness=fwd['mask_sharpness'], device=device)
    fit = cfg_data['fit']
    fit_cfg = ThreeChannelFitConfig(
        fit_param_names=cfg_data['fit_param_names'],
        init_ranges=dict(cfg_data['fit_init_ranges']),
        n_starts=int(fit['n_starts']),
        n_iters=int(fit['n_iters']),
        lr=float(fit['lr']),
        # Phase 0b ground truth is alpha_true = 0, shift_true = 0; force them
        # off regardless of yaml so the regression test exercises the same
        # fit dim every time.
        fit_alpha=bool(args.enable_alpha_fit),
        fit_shift=False,
        lambda_alpha=float(fit['lambda_alpha']),
        lambda_shift=float(fit['lambda_shift']),
        radial_sigma_frac=float(fit['radial_sigma_frac']),
        sign_flip_init=bool(fit['sign_flip_init']),
        init_alpha_z=float(fit['init_alpha_z']),
    )

    rng = np.random.default_rng(args.seed)
    psf_l1s, final_losses, I_errs = [], [], []

    print(f'\n{args.n_trials} trials, {fit_cfg.n_starts} starts × {fit_cfg.n_iters} iters\n')

    for trial in range(args.n_trials):
        theta_true = sample_theta_true(cfg_data, rng)
        theta_true_t = make_theta(theta_true, device=device)

        with torch.no_grad():
            psf_true = differentiable_forward(theta_true_t, fwd_cfg)
        psf_true_np = psf_true.cpu().numpy()

        target, ref1, ref2 = build_synth_triplet(
            psf_true_np, args.I_true, args.ref_nuisance_sigma, rng)
        diff1 = torch.from_numpy((target - ref1).astype(np.float32)).to(device)
        diff2 = torch.from_numpy((target - ref2).astype(np.float32)).to(device)

        result = fit_three_channel(diff1, diff2, cfg_data['fixed_params'],
                                   fit_cfg, fwd_cfg, rng=rng)

        theta_fit_t = make_theta(result['theta'], device=device)
        with torch.no_grad():
            psf_fit = differentiable_forward(theta_fit_t, fwd_cfg)
        l1 = (psf_fit - psf_true).abs().sum().item()
        psf_l1s.append(l1)
        final_losses.append(result['loss'])
        rows, I_err_rel = evaluate_recovery(theta_true, result['theta'],
                                            cfg_data['fit_init_ranges'],
                                            args.I_true, result['I'])
        I_errs.append(I_err_rel)

        max_rel = max(r[4] for r in rows) if rows else 0.0
        print(f'Trial {trial+1:2d}/{args.n_trials}: '
              f'loss={result["loss"]:.4e}  PSF_L1={l1:.4f}  '
              f'I_fit={result["I"]:.0f} ({I_err_rel*100:+.1f}%)  '
              f'theta_max_rel={max_rel:.3f}')

    print('\n=== PSF reconstruction (3-channel fit) ===')
    print(f'  median L1: {np.median(psf_l1s):.4f}')
    print(f'  best  L1:  {np.min(psf_l1s):.4f}')
    print(f'  worst L1:  {np.max(psf_l1s):.4f}')

    print('\n=== Intensity recovery ===')
    print(f'  median |I_err|/I_true: {np.median(I_errs)*100:.2f}%')
    print(f'  worst:                 {np.max(I_errs)*100:.2f}%')

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
