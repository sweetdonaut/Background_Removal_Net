"""Inverse fit primitives — single-PSF and 3-channel variants.

Two entry points share the multi-start Adam scaffolding:
  - fit_one(...)            : single observation vs normalized forward(theta).
                              Used only by phase0a regression test.
  - fit_three_channel(...)  : joint loss over (diff1, diff2) with per-pixel
                              min selection (double_detection logic),
                              sigmoid-bounded leak coefs alpha1/alpha2,
                              log-parameterized intensity I, and L2 prior on
                              alpha to break the (I, alpha) degeneracy.
                              Used by fit_real.py and phase0b.

Theta entries are either fittable nn.Parameter or fixed tensor; the
difference between the two entry points is only in loss construction and
the extra (I, alpha1, alpha2) parameters added in fit_three_channel.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from forward import (  # noqa: E402
    PARAM_NAMES, ForwardConfig, differentiable_forward,
)


# Fallback init ranges for when FitConfig is constructed without explicit
# init_ranges (e.g. from a unit test or quick interactive use). Production
# entry points (fit_real, phase0a, phase0b) override this dict from yaml.
# cy/cx are sub-pixel shift in sensor-pixel units; the (X, Y) coordinate
# from the filename is integer-rounded so the true PSF center sits within
# ±0.5 pixel — initialize tight, prior keeps it tight unless data demands.
DEFAULT_INIT_RANGES = {
    'defocus': (-1.5, 1.5),
    'astig_x': (-1.0, 1.0),
    'astig_y': (-1.0, 1.0),
    'coma_x': (-0.8, 0.8),
    'coma_y': (-0.8, 0.8),
    'spherical': (-0.8, 0.8),
    'trefoil_x': (-0.8, 0.8),
    'trefoil_y': (-0.8, 0.8),
    'outer_r': (40.0, 80.0),
    'epsilon': (0.0, 0.4),
    'ellipticity': (-0.3, 0.3),
    'ellip_angle': (0.0, 180.0),
    'na': (0.85, 0.99),
    'cy': (-0.5, 0.5),
    'cx': (-0.5, 0.5),
}


@dataclass
class FitConfig:
    fit_param_names: Tuple[str, ...] = (
        'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y',
        'spherical', 'trefoil_x', 'trefoil_y',
    )
    n_starts: int = 5
    n_iters: int = 600
    lr: float = 0.05
    init_ranges: dict = field(default_factory=lambda: dict(DEFAULT_INIT_RANGES))


def normalize_obs(obs):
    """Clamp negatives to 0 and rescale to sum=1, matching forward output."""
    obs = torch.clamp(obs, min=0.0)
    return obs / (obs.sum() + 1e-12)


def _build_theta_dict(fit_params, fixed_tensors):
    """Merge fittable nn.Parameters with fixed tensors into a complete theta dict."""
    out = {}
    for k in PARAM_NAMES:
        if k in fit_params:
            out[k] = fit_params[k]
        elif k in fixed_tensors:
            out[k] = fixed_tensors[k]
        else:
            raise KeyError(f"param {k!r} missing from both fit and fixed sets")
    return out


def _sample_init(name, ranges, rng):
    lo, hi = ranges[name]
    return float(rng.uniform(lo, hi))


def fit_one(observation, fixed_params, fit_cfg: FitConfig, fwd_cfg: ForwardConfig,
            rng=None):
    """Fit theta to a single PSF observation via multi-start Adam.

    observation : 2D tensor (crop_size, crop_size). Will be normalized.
    fixed_params: dict {name: float} for params not in fit_cfg.fit_param_names.
                  Must cover all PARAM_NAMES \\ fit_param_names.
    fit_cfg     : optimizer config (which params to fit, n_starts, etc.)
    fwd_cfg     : forward model config (psf_size, crop_size, pol_type, device).
    rng         : numpy default_rng for reproducible random inits.

    Returns dict:
        theta       : {name: float} for all PARAM_NAMES (fit values for fitted,
                      fixed values for the rest)
        loss        : best final loss across starts
        history     : list of per-iter losses for the best start
        all_losses  : final loss per start (multi-start trace)
    """
    if rng is None:
        rng = np.random.default_rng()

    obs_norm = normalize_obs(observation.detach()).to(fwd_cfg.device)

    fixed_tensors = {
        k: torch.tensor(float(v), dtype=torch.float32, device=fwd_cfg.device)
        for k, v in fixed_params.items() if k not in fit_cfg.fit_param_names
    }
    missing_fixed = set(PARAM_NAMES) - set(fit_cfg.fit_param_names) - set(fixed_tensors)
    if missing_fixed:
        raise KeyError(f"fixed_params missing entries for: {sorted(missing_fixed)}")

    best = {'loss': float('inf'), 'theta': None, 'history': None}
    all_final_losses = []

    for _start in range(fit_cfg.n_starts):
        fit_params = {
            name: torch.nn.Parameter(torch.tensor(
                _sample_init(name, fit_cfg.init_ranges, rng),
                dtype=torch.float32, device=fwd_cfg.device,
            ))
            for name in fit_cfg.fit_param_names
        }
        optimizer = torch.optim.Adam(list(fit_params.values()), lr=fit_cfg.lr)
        history = []

        for _ in range(fit_cfg.n_iters):
            theta = _build_theta_dict(fit_params, fixed_tensors)
            pred = differentiable_forward(theta, fwd_cfg)
            loss = ((pred - obs_norm) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        final_loss = history[-1]
        all_final_losses.append(final_loss)

        if final_loss < best['loss']:
            theta_out = {k: float(v.item()) for k, v in fit_params.items()}
            theta_out.update({k: float(v.item()) for k, v in fixed_tensors.items()})
            best['loss'] = final_loss
            best['theta'] = theta_out
            best['history'] = history

    return {
        'theta': best['theta'],
        'loss': best['loss'],
        'history': best['history'],
        'all_losses': all_final_losses,
    }


# ---------------------------------------------------------------------------
# 3-channel fit (Phase 0b/0c)
# ---------------------------------------------------------------------------


@dataclass
class ThreeChannelFitConfig:
    fit_param_names: Tuple[str, ...] = (
        'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y',
        'spherical', 'trefoil_x', 'trefoil_y',
    )
    n_starts: int = 8
    n_iters: int = 800
    lr: float = 0.05
    init_ranges: dict = field(default_factory=lambda: dict(DEFAULT_INIT_RANGES))
    fit_alpha: bool = True       # if False, alpha1=alpha2=0 (sanity simplification)
    lambda_alpha: float = 1e-3   # L2 prior to break (I, alpha) degeneracy
    fit_shift: bool = True       # if False, cy=cx=0 fixed (no sub-pixel translation)
    lambda_shift: float = 1e-3   # L2 prior on (cy, cx) — keeps shift small unless data demands
    radial_sigma_frac: float = 0.25  # weight std as fraction of crop size
    init_alpha_z: float = -2.0   # sigmoid(-2) ≈ 0.12 — small leak default
    sign_flip_init: bool = True  # also try negated-theta inits to escape twin basins


def _build_radial_weight(H, W, sigma, device):
    """Gaussian weight centered at (H/2, W/2). Used to focus loss on the PSF region."""
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return torch.exp(-r2 / (2.0 * sigma ** 2))


def _estimate_initial_I(diff1, diff2, weight_map):
    """Rough I estimate from the larger of (sum(diff1), sum(diff2)) on the
    weighted central region. The forward PSF is sum=1 by construction; if
    diff = PSF * I then sum(diff) ≈ I (modulo masking). Robust enough for
    Adam warm-up; the optimizer refines from here."""
    s1 = (diff1 * weight_map).sum().item()
    s2 = (diff2 * weight_map).sum().item()
    weight_sum = weight_map.sum().item()
    # Normalize the weighted sum back to "approx total" by dividing by the
    # weight integral relative to crop area. This roughly recovers the
    # full-crop sum scale.
    H, W = diff1.shape
    norm = max(weight_sum / (H * W), 1e-6)
    raw = max(abs(s1), abs(s2)) / norm
    return max(raw, 1.0)  # avoid log(0); 1.0 is very small fallback


def fit_three_channel(diff1, diff2, fixed_params,
                      fit_cfg: ThreeChannelFitConfig, fwd_cfg: ForwardConfig,
                      rng=None):
    """Joint fit of theta + (I, alpha1, alpha2) against two channel diffs.

    diff1, diff2 : 2D tensors (H, W); typically (target - ref_aligned). Sign
                   convention: positive PSF on target -> positive diff.
    fixed_params : dict {name: float} for non-fitted theta entries.
    fit_cfg      : ThreeChannelFitConfig.
    fwd_cfg      : ForwardConfig (psf_size, crop_size=H=W, pol_type, device).

    Returns dict:
        theta       : {name: float} for all PARAM_NAMES (no cy/cx; those are
                      observation-specific and reported separately)
        I           : fitted intensity (positive scalar)
        alpha1      : fitted leak coef in [0, 1]
        alpha2      : fitted leak coef in [0, 1]
        cy, cx      : fitted sub-pixel shift (sensor pixels). 0 if fit_shift=False.
        loss        : best final loss across starts
        history     : per-iter loss for the best start
        all_losses  : final loss per start
        residual_l1 : sum(weight * min(|diff1 - pred1|, |diff2 - pred2|)) at best fit
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = diff1.shape
    assert (H, W) == diff2.shape, 'diff1 and diff2 must match shape'
    assert (H, W) == (fwd_cfg.crop_size, fwd_cfg.crop_size), \
        f'diff shape {(H, W)} must equal fwd_cfg.crop_size {fwd_cfg.crop_size}'

    diff1 = diff1.to(fwd_cfg.device).float()
    diff2 = diff2.to(fwd_cfg.device).float()

    sigma = fit_cfg.radial_sigma_frac * fwd_cfg.crop_size
    weight_map = _build_radial_weight(H, W, sigma, fwd_cfg.device)
    weight_sum = weight_map.sum() + 1e-12

    fixed_tensors = {
        k: torch.tensor(float(v), dtype=torch.float32, device=fwd_cfg.device)
        for k, v in fixed_params.items() if k not in fit_cfg.fit_param_names
    }
    missing = set(PARAM_NAMES) - set(fit_cfg.fit_param_names) - set(fixed_tensors)
    if missing:
        raise KeyError(f"fixed_params missing entries for: {sorted(missing)}")

    init_I_value = _estimate_initial_I(diff1, diff2, weight_map)

    def _make_starts():
        """Yield (start_index, init_dict) where init_dict maps param->float."""
        for s in range(fit_cfg.n_starts):
            base = {n: _sample_init(n, fit_cfg.init_ranges, rng)
                    for n in fit_cfg.fit_param_names}
            yield s, base
            if fit_cfg.sign_flip_init and s < fit_cfg.n_starts // 2:
                # also try the central-symmetry twin to give Adam two basins
                # per random sample (defocus/astig/spherical sign flip)
                flipped = dict(base)
                for n in ('defocus', 'astig_x', 'astig_y', 'spherical'):
                    if n in flipped:
                        flipped[n] = -flipped[n]
                yield s, flipped

    best = {'loss': float('inf'), 'theta': None, 'history': None,
            'I': None, 'alpha1': None, 'alpha2': None,
            'cy': None, 'cx': None, 'residual_l1': None}
    all_final_losses = []

    for _start_idx, init_vals in _make_starts():
        fit_params = {
            name: torch.nn.Parameter(torch.tensor(
                init_vals[name], dtype=torch.float32, device=fwd_cfg.device))
            for name in fit_cfg.fit_param_names
        }
        log_I = torch.nn.Parameter(torch.tensor(
            float(np.log(max(init_I_value, 1.0))),
            dtype=torch.float32, device=fwd_cfg.device))
        z1 = torch.nn.Parameter(torch.tensor(
            float(fit_cfg.init_alpha_z), dtype=torch.float32, device=fwd_cfg.device))
        z2 = torch.nn.Parameter(torch.tensor(
            float(fit_cfg.init_alpha_z), dtype=torch.float32, device=fwd_cfg.device))
        cy = torch.nn.Parameter(torch.tensor(
            0.0, dtype=torch.float32, device=fwd_cfg.device))
        cx = torch.nn.Parameter(torch.tensor(
            0.0, dtype=torch.float32, device=fwd_cfg.device))

        params_to_optim = list(fit_params.values()) + [log_I]
        if fit_cfg.fit_alpha:
            params_to_optim += [z1, z2]
        if fit_cfg.fit_shift:
            params_to_optim += [cy, cx]
        optimizer = torch.optim.Adam(params_to_optim, lr=fit_cfg.lr)
        history = []

        for _ in range(fit_cfg.n_iters):
            theta = _build_theta_dict(fit_params, fixed_tensors)
            if fit_cfg.fit_shift:
                theta['cy'] = cy
                theta['cx'] = cx
            psf = differentiable_forward(theta, fwd_cfg)
            I = torch.exp(log_I)
            if fit_cfg.fit_alpha:
                alpha1 = torch.sigmoid(z1)
                alpha2 = torch.sigmoid(z2)
            else:
                alpha1 = torch.zeros((), device=fwd_cfg.device)
                alpha2 = torch.zeros((), device=fwd_cfg.device)

            pred1 = psf * I * (1.0 - alpha1)
            pred2 = psf * I * (1.0 - alpha2)

            r1 = (diff1 - pred1).abs()
            r2 = (diff2 - pred2).abs()
            per_pixel = torch.minimum(r1, r2) * weight_map
            data_loss = per_pixel.sum() / weight_sum

            reg = 0.0
            if fit_cfg.fit_alpha:
                reg = reg + fit_cfg.lambda_alpha * (alpha1 ** 2 + alpha2 ** 2)
            if fit_cfg.fit_shift:
                reg = reg + fit_cfg.lambda_shift * (cy ** 2 + cx ** 2)
            loss = data_loss + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        final_loss = history[-1]
        all_final_losses.append(final_loss)

        if final_loss < best['loss']:
            theta_out = {k: float(v.item()) for k, v in fit_params.items()}
            theta_out.update({k: float(v.item()) for k, v in fixed_tensors.items()})
            best = {
                'loss': final_loss,
                'theta': theta_out,
                'history': history,
                'I': float(torch.exp(log_I).item()),
                'alpha1': float(torch.sigmoid(z1).item()) if fit_cfg.fit_alpha else 0.0,
                'alpha2': float(torch.sigmoid(z2).item()) if fit_cfg.fit_alpha else 0.0,
                'cy': float(cy.item()) if fit_cfg.fit_shift else 0.0,
                'cx': float(cx.item()) if fit_cfg.fit_shift else 0.0,
                'residual_l1': float(per_pixel.sum().item()),
            }

    return {
        'theta': best['theta'],
        'I': best['I'],
        'alpha1': best['alpha1'],
        'alpha2': best['alpha2'],
        'cy': best['cy'],
        'cx': best['cx'],
        'loss': best['loss'],
        'history': best['history'],
        'all_losses': all_final_losses,
        'residual_l1': best['residual_l1'],
    }
