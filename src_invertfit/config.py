"""YAML config loader for inverse fit.

Mirrors src_core/defects/*.yaml field names so a production synth yaml can be
copied here as a starting point. Production-only fields (intensity_abs,
brightness, gaussian_sigma, etc.) are silently ignored. Fit-specific options
live under a `fit:` namespace at the bottom.

Physics parameters use [min, max] convention:
  [v, v]   → FIXED at v (excluded from optimizer)
  [a, b]   → FITTED, init drawn uniformly from [a, b]

See src_invertfit/fit_configs/default.yaml for an annotated example.
"""

import os

import yaml


# 13 physics parameters that the fit forward model understands.
PHYSICS_PARAMS = (
    'outer_r', 'epsilon', 'ellipticity', 'ellip_angle', 'na',
    'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y',
    'spherical', 'trefoil_x', 'trefoil_y',
)


# fit:* defaults — applied when the user yaml omits the entry.
FIT_DEFAULTS = {
    'alpha': False,
    'shift': True,
    'lambda_alpha': 1e-3,
    'lambda_shift': 1e-3,
    'radial_sigma_frac': 0.25,
    'mask_sharpness': 2.0,
    'n_starts': 10,
    'n_iters': 1000,
    'lr': 0.05,
    'sign_flip_init': True,
    'init_alpha_z': -2.0,
}


def load_fit_config(yaml_path):
    """Read and validate a fit config yaml.

    Returns dict:
        source_yaml      : absolute path of the loaded yaml (for provenance)
        forward          : {psf_size, crop_size, pixel_oversample, pol_type,
                            vector_mode, mask_sharpness}
        fixed_params     : {name: float} for physics params with min == max
        fit_param_names  : tuple of physics param names with min != max
        fit_init_ranges  : {name: (a, b)} init draw ranges for fit params
        fit              : full fit:* block with FIT_DEFAULTS applied
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config yaml not found: {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{yaml_path}: top level must be a yaml mapping")

    # Forward-model architecture
    forward = {
        'psf_size': int(cfg.get('psf_size', 256)),
        'crop_size': int(cfg.get('crop_size', 32)),
        'pixel_oversample': int(cfg.get('pixel_oversample', 4)),
        'pol_type': str(cfg.get('pol_type', 'linX')),
        'vector_mode': bool(cfg.get('vector_mode', True)),
    }
    if not forward['vector_mode']:
        raise ValueError(
            f"{yaml_path}: vector_mode must be true (scalar mode not implemented in fit)")

    # fit:* hyperparams
    user_fit = cfg.get('fit', {}) or {}
    if not isinstance(user_fit, dict):
        raise ValueError(f"{yaml_path}: 'fit' must be a mapping if present")
    fit = dict(FIT_DEFAULTS)
    fit.update(user_fit)
    forward['mask_sharpness'] = float(fit['mask_sharpness'])

    # Physics params: [v,v] = fixed, [a,b] = fit
    fixed_params, fit_param_names, fit_init_ranges = _parse_physics_params(cfg, yaml_path)
    if not fit_param_names:
        raise ValueError(f"{yaml_path}: at least one physics param must have min != max "
                         "(otherwise the optimizer has nothing to fit)")

    return {
        'source_yaml': os.path.abspath(yaml_path),
        'forward': forward,
        'fixed_params': fixed_params,
        'fit_param_names': fit_param_names,
        'fit_init_ranges': fit_init_ranges,
        'fit': fit,
    }


def _parse_physics_params(cfg, yaml_path):
    fixed = {}
    fit_names = []
    init_ranges = {}
    for name in PHYSICS_PARAMS:
        if name not in cfg:
            raise ValueError(f"{yaml_path}: missing required physics param '{name}'")
        v = cfg[name]
        if not (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(x, (int, float)) for x in v)):
            raise ValueError(
                f"{yaml_path}: '{name}' must be a [min, max] list of two numbers, "
                f"got {v!r}")
        a, b = float(v[0]), float(v[1])
        if a > b:
            a, b = b, a
        if a == b:
            fixed[name] = a
        else:
            fit_names.append(name)
            init_ranges[name] = (a, b)
    return fixed, tuple(fit_names), init_ranges


def summarize(cfg_data):
    """Return a human-readable multi-line summary of a loaded config."""
    fwd = cfg_data['forward']
    fit = cfg_data['fit']
    lines = [
        f'Source: {cfg_data["source_yaml"]}',
        f'Forward: psf_size={fwd["psf_size"]}, crop_size={fwd["crop_size"]}, '
        f'oversample={fwd["pixel_oversample"]}, pol={fwd["pol_type"]}, '
        f'mask_sharpness={fwd["mask_sharpness"]}',
        f'Fitted physics params ({len(cfg_data["fit_param_names"])}): '
        f'{list(cfg_data["fit_param_names"])}',
        f'Fixed physics params  ({len(cfg_data["fixed_params"])}): '
        f'{cfg_data["fixed_params"]}',
        f'Fit hyperparams: alpha={fit["alpha"]}, shift={fit["shift"]}, '
        f'lambda_alpha={fit["lambda_alpha"]}, lambda_shift={fit["lambda_shift"]}, '
        f'n_starts={fit["n_starts"]}, n_iters={fit["n_iters"]}, lr={fit["lr"]}',
    ]
    return '\n'.join(lines)
