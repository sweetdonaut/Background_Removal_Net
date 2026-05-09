"""Differentiable PSF forward model in PyTorch.

Mirrors src_core/generate_psf.py:generate_one() vector-mode path with three
modifications needed for inverse fitting:
  1. Soft (sigmoid) pupil edges so geometric params (outer_r, epsilon,
     ellipticity, ellip_angle, na) carry gradient
  2. No noise injection — fit operates on the expected PSF; noise is added
     externally when constructing the synthetic observation
  3. Output normalized to sum=1 — brightness/background are absorbed by the
     fit-side scale parameter (log_I), not by theta

Vector mode is hard-coded ON (per project convention) and `pol_type` is fixed
at module-call time. `pixel_oversample` is set to 1 in v1; the production
pipeline uses 4 for sensor binning, but for shape recovery the difference is
sub-pixel and absorbed by noise. Pupil obstructions (square_eps, stripes,
outer_crop) are also omitted in v1 since the wafer optical setup keeps them
all at 0 — easy to add back if production data ever uses them.
"""

import math
from dataclasses import dataclass

import torch


PARAM_NAMES = (
    'outer_r', 'epsilon', 'ellipticity', 'ellip_angle',
    'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y',
    'spherical', 'trefoil_x', 'trefoil_y',
    'na',
)


def _soft_step(z, sharpness):
    return torch.sigmoid(z * sharpness)


@dataclass
class ForwardConfig:
    psf_size: int = 256
    crop_size: int = 32
    pol_type: str = 'linX'
    mask_sharpness: float = 2.0
    device: str = 'cuda'


def make_grid(N, device):
    coords = torch.arange(-(N // 2), N - N // 2, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(coords, coords, indexing='ij')
    return x, y


def _polarization_components(pol_type, phi):
    """Return (px_re, px_im, py_re, py_im) for the chosen polarization.

    Matches src_core/generate_psf.py:_build_vector_pupil() conventions.
    """
    inv2 = 1.0 / math.sqrt(2.0)
    zeros = torch.zeros_like(phi)
    ones = torch.ones_like(phi)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)

    if pol_type == 'linX':
        return ones, zeros, zeros, zeros
    if pol_type == 'linY':
        return zeros, zeros, ones, zeros
    if pol_type == 'lin45':
        return torch.full_like(phi, inv2), zeros, torch.full_like(phi, inv2), zeros
    if pol_type == 'circR':
        return torch.full_like(phi, inv2), zeros, zeros, torch.full_like(phi, inv2)
    if pol_type == 'circL':
        return torch.full_like(phi, inv2), zeros, zeros, torch.full_like(phi, -inv2)
    if pol_type == 'radial':
        return cos_p, zeros, sin_p, zeros
    raise ValueError(f"Unsupported pol_type: {pol_type!r}")


def _vector_pupil(mask, phase, x, y, outer_r, na, pol_type):
    """Build Richards-Wolf vector pupil components.

    Returns three complex tensors (ux, uy, uz). Each is the pupil-plane
    field for one Cartesian polarization component, ready for FFT.
    """
    rho_pixel = torch.sqrt(x ** 2 + y ** 2 + 1e-12)
    sin_theta = (rho_pixel / outer_r) * na
    # Soft validity gate: 1 inside NA cone (sin_theta < 1), smoothly drops past it
    inside_na = _soft_step(1.0 - sin_theta, sharpness=8.0)
    valid_mask = mask * inside_na

    sin_t = torch.clamp(sin_theta, max=0.999)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t ** 2, min=1e-12))
    phi = torch.atan2(y, x)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    apod = torch.sqrt(torch.clamp(cos_t, min=1e-12))

    px_re, px_im, py_re, py_im = _polarization_components(pol_type, phi)

    cos2p = cos_p * cos_p
    sin2p = sin_p * sin_p
    csp = cos_p * sin_p
    Axx = cos_t * cos2p + sin2p
    Axy = (cos_t - 1.0) * csp
    Ayy = cos_t * sin2p + cos2p
    Azx = -sin_t * cos_p
    Azy = -sin_t * sin_p

    ox_re = Axx * px_re + Axy * py_re
    ox_im = Axx * px_im + Axy * py_im
    oy_re = Axy * px_re + Ayy * py_re
    oy_im = Axy * px_im + Ayy * py_im
    oz_re = Azx * px_re + Azy * py_re
    oz_im = Azx * px_im + Azy * py_im

    e_re = torch.cos(phase)
    e_im = torch.sin(phase)
    a = apod * valid_mask

    ux = torch.complex(a * (ox_re * e_re - ox_im * e_im),
                       a * (ox_re * e_im + ox_im * e_re))
    uy = torch.complex(a * (oy_re * e_re - oy_im * e_im),
                       a * (oy_re * e_im + oy_im * e_re))
    uz = torch.complex(a * (oz_re * e_re - oz_im * e_im),
                       a * (oz_re * e_im + oz_im * e_re))
    return ux, uy, uz


def differentiable_forward(theta, cfg: ForwardConfig):
    """theta dict -> normalized PSF (sum=1) of shape (crop_size, crop_size)."""
    x, y = make_grid(cfg.psf_size, cfg.device)

    # Annular mask with optional ellipticity (soft edges)
    cos_a = torch.cos(theta['ellip_angle'] * math.pi / 180.0)
    sin_a = torch.sin(theta['ellip_angle'] * math.pi / 180.0)
    e_safe = torch.clamp(theta['ellipticity'], min=-0.95, max=0.95)
    rx = (x * cos_a + y * sin_a) / (1.0 + e_safe)
    ry = (-x * sin_a + y * cos_a) / (1.0 - e_safe)
    r = torch.sqrt(rx ** 2 + ry ** 2 + 1e-12)
    mask = (_soft_step(theta['outer_r'] - r, cfg.mask_sharpness)
            * _soft_step(r - theta['outer_r'] * theta['epsilon'], cfg.mask_sharpness))

    # Zernike phase
    dx = x / theta['outer_r']
    dy = y / theta['outer_r']
    rho2 = dx ** 2 + dy ** 2
    rho = torch.sqrt(rho2 + 1e-12)
    th = torch.atan2(dy, dx)
    phase = (theta['defocus'] * (2 * rho2 - 1)
             + theta['astig_x'] * rho2 * torch.cos(2 * th)
             + theta['astig_y'] * rho2 * torch.sin(2 * th)
             + theta['coma_x'] * (3 * rho2 - 2) * rho * torch.cos(th)
             + theta['coma_y'] * (3 * rho2 - 2) * rho * torch.sin(th)
             + theta['spherical'] * (6 * rho2 ** 2 - 6 * rho2 + 1)
             + theta['trefoil_x'] * rho2 * rho * torch.cos(3 * th)
             + theta['trefoil_y'] * rho2 * rho * torch.sin(3 * th))

    ux, uy, uz = _vector_pupil(mask, phase, x, y, theta['outer_r'],
                               theta['na'], cfg.pol_type)

    # Three FFTs, sum |.|^2 across polarization channels
    Ix = torch.fft.fftshift(torch.fft.fft2(ux)).abs() ** 2
    Iy = torch.fft.fftshift(torch.fft.fft2(uy)).abs() ** 2
    Iz = torch.fft.fftshift(torch.fft.fft2(uz)).abs() ** 2
    psf = Ix + Iy + Iz

    s = cfg.psf_size // 2 - cfg.crop_size // 2
    cropped = psf[s:s + cfg.crop_size, s:s + cfg.crop_size]
    cropped = cropped / (cropped.sum() + 1e-12)
    return cropped


def make_theta(values, device, requires_grad=False):
    """Build a {name: 0-d tensor} dict from a {name: float} dict.

    All PARAM_NAMES must be present in `values`.
    """
    out = {}
    for k in PARAM_NAMES:
        if k not in values:
            raise KeyError(f"missing param {k!r}; required: {PARAM_NAMES}")
        out[k] = torch.tensor(float(values[k]), dtype=torch.float32, device=device,
                              requires_grad=requires_grad)
    return out
