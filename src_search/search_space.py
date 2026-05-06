"""Yaml parameter sampling for the search.

Each sample_* function returns a yaml-compatible value for one parameter.
sample_params() composes the active ones into one trial's yaml override dict.

To change which parameters get searched:
    1. All common sample_* are already implemented below.
    2. In sample_params(), toggle (uncomment / comment) the lines you want.
    3. No other file needs to change — run_trial.py picks up the new keys
       automatically and writes them into trial_yaml.yaml.

Yaml keys not touched here keep their type4_vector.yaml base value.
"""

import numpy as np


# ============================================================
# Strongly recommended: signal strength + size + noise level
# ============================================================

def sample_intensity_abs(rng):
    """Defect contrast amplitude (single uniform range). Base = [[8, 12]]."""
    low = float(rng.uniform(2.0, 15.0))
    width = float(rng.uniform(2.0, 12.0))
    high = min(30.0, low + width)
    return [[round(low, 2), round(high, 2)]]


def sample_outer_r(rng):
    """Aperture outer radius (px) on the FFT grid. Smaller r -> larger PSF.
    Base = [60, 60]."""
    r = float(rng.uniform(30.0, 80.0))
    return [round(r), round(r)]


def sample_brightness(rng):
    """PSF total energy (post-FFT normalization). Base = [5000, 5000]."""
    b = float(rng.uniform(2000.0, 10000.0))
    return [round(b), round(b)]


def sample_gaussian_sigma(rng):
    """Per-pixel sensor read-noise sigma. Base = [1.5, 1.5]."""
    s = float(rng.uniform(0.3, 3.5))
    return [round(s, 2), round(s, 2)]


def sample_background(rng):
    """Per-pixel dark-current background (added before noise). Base = [5, 5]."""
    bg = float(rng.uniform(0.0, 20.0))
    return [round(bg, 1), round(bg, 1)]


# ============================================================
# Optical aberrations (Zernike coefficients in wavelengths)
# Base = [0, 0] for all. Real high-NA systems usually within ±2 waves.
# Larger values blur / asymmetrize the PSF.
# ============================================================

def sample_defocus(rng):
    a = float(rng.uniform(-1.5, 1.5))
    return [round(a, 2), round(a, 2)]


def sample_astig_x(rng):
    a = float(rng.uniform(-1.0, 1.0))
    return [round(a, 2), round(a, 2)]


def sample_astig_y(rng):
    a = float(rng.uniform(-1.0, 1.0))
    return [round(a, 2), round(a, 2)]


def sample_coma_x(rng):
    a = float(rng.uniform(-0.8, 0.8))
    return [round(a, 2), round(a, 2)]


def sample_coma_y(rng):
    a = float(rng.uniform(-0.8, 0.8))
    return [round(a, 2), round(a, 2)]


def sample_spherical(rng):
    a = float(rng.uniform(-0.8, 0.8))
    return [round(a, 2), round(a, 2)]


def sample_trefoil_x(rng):
    a = float(rng.uniform(-0.5, 0.5))
    return [round(a, 2), round(a, 2)]


def sample_trefoil_y(rng):
    a = float(rng.uniform(-0.5, 0.5))
    return [round(a, 2), round(a, 2)]


# ============================================================
# Aperture geometry
# ============================================================

def sample_epsilon(rng):
    """Inner/outer radius ratio (central obstruction, e.g. reflective scopes).
    Base = [0, 0]. >0 turns the Airy disk into a more ring-like PSF."""
    e = float(rng.uniform(0.0, 0.6))
    return [round(e, 2), round(e, 2)]


def sample_ellipticity(rng):
    """Aperture ellipticity (elongates the PSF). Base = [0, 0]."""
    e = float(rng.uniform(-0.3, 0.3))
    return [round(e, 2), round(e, 2)]


def sample_ellip_angle(rng):
    """Ellipticity orientation in degrees. Base = [0, 0]."""
    a = float(rng.uniform(0.0, 180.0))
    return [round(a), round(a)]


# ============================================================
# Compose
# ============================================================

def sample_params(rng):
    """Toggle (uncomment / comment) the lines you want to search."""
    out = {}

    # --- Tier 1: signal & noise (strongly recommended) ---
    # out['intensity_abs'] = sample_intensity_abs(rng)
    out['outer_r'] = sample_outer_r(rng)
    # out['brightness'] = sample_brightness(rng)
    # out['gaussian_sigma'] = sample_gaussian_sigma(rng)
    # out['background'] = sample_background(rng)

    # --- Tier 2: aberrations ---
    # out['defocus'] = sample_defocus(rng)
    # out['astig_x'] = sample_astig_x(rng)
    # out['astig_y'] = sample_astig_y(rng)
    # out['coma_x'] = sample_coma_x(rng)
    # out['coma_y'] = sample_coma_y(rng)
    # out['spherical'] = sample_spherical(rng)
    # out['trefoil_x'] = sample_trefoil_x(rng)
    # out['trefoil_y'] = sample_trefoil_y(rng)

    # --- Tier 3: aperture shape ---
    # out['epsilon'] = sample_epsilon(rng)
    # out['ellipticity'] = sample_ellipticity(rng)
    # out['ellip_angle'] = sample_ellip_angle(rng)

    return out
