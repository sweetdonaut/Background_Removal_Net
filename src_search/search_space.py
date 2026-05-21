"""Sampler library + spec loader for the yaml search.

Search space is defined externally in a spec yaml (see src_search/search_configs/), not
in this file. This module:
    - implements the sampler primitives (scalar_pair, range_pair)
    - validates and loads spec yamls
    - turns a loaded spec + rng into a {yaml_key: sampled_value} override dict

To add a new searchable yaml key:
    - if its shape fits an existing sampler (scalar_pair / range_pair), just
      reference it from the spec; nothing to add here.
    - if it needs a new shape, add a sampler function below and register it
      in SAMPLERS.
"""

import os

import yaml


def _sample_scalar_pair(spec, rng):
    """Sample one scalar v in [range.min, range.max], return [v, v].

    Spec keys:
        range:    {min, max}
        decimals: int (default 2). 0 -> python int, >=1 -> rounded float.
    """
    rmin = spec['range']['min']
    rmax = spec['range']['max']
    decimals = int(spec.get('decimals', 2))
    v = float(rng.uniform(rmin, rmax))
    if decimals == 0:
        v = int(round(v))
    else:
        v = round(v, decimals)
    return [v, v]


def _sample_range_pair(spec, rng):
    """Sample low + width, return [[low, low + width]] (clipped at high_cap).

    Used for keys like intensity_abs whose yaml form is a list-of-list mixture.

    Spec keys:
        low:      {min, max}
        width:    {min, max}
        high_cap: float (optional). Caps low + width.
        decimals: int (default 2).
    """
    low = float(rng.uniform(spec['low']['min'], spec['low']['max']))
    width = float(rng.uniform(spec['width']['min'], spec['width']['max']))
    high = low + width
    cap = spec.get('high_cap')
    if cap is not None:
        high = min(float(cap), high)
    decimals = int(spec.get('decimals', 2))
    low = round(low, decimals)
    high = round(high, decimals)
    return [[low, high]]


def _sample_scalar(spec, rng):
    """Sample one scalar v in [range.min, range.max], return v (not a pair).

    Differs from scalar_pair: returns the bare value (e.g. 4) instead of
    [v, v]. Use for yaml keys that are scalars in the PSF config — like
    `pixel_oversample`, `crop_size`, `psf_size`.

    Spec keys:
        range:    {min, max}
        decimals: int (default 2). 0 -> python int, >=1 -> rounded float.
    """
    rmin = spec['range']['min']
    rmax = spec['range']['max']
    decimals = int(spec.get('decimals', 2))
    v = float(rng.uniform(rmin, rmax))
    if decimals == 0:
        return int(round(v))
    return round(v, decimals)


def _sample_categorical(spec, rng):
    """Pick one value uniformly from a fixed list.

    Each value can be any yaml-serializable type: scalar (e.g. 4), string
    (e.g. 'linX'), or list (e.g. ['target', 'double_det'] for input_channels).
    A list value is deep-copied to avoid sharing the spec object across trials.

    Spec keys:
        values: non-empty list of allowed values.
    """
    values = spec.get('values')
    if not isinstance(values, list) or not values:
        raise ValueError("categorical sampler requires non-empty `values:` list")
    chosen = values[int(rng.integers(0, len(values)))]
    return list(chosen) if isinstance(chosen, list) else chosen


SAMPLERS = {
    'scalar_pair': _sample_scalar_pair,
    'range_pair': _sample_range_pair,
    'scalar': _sample_scalar,
    'categorical': _sample_categorical,
}


def _validate_dim_block(name, dims, path):
    """Validate a dims-shaped mapping. dims may be None or empty (skipped)."""
    if dims is None:
        return
    if not isinstance(dims, dict):
        raise ValueError(
            f"spec {path}: `{name}:` must be a mapping, "
            f"got {type(dims).__name__}")
    for key, dim_spec in dims.items():
        if not isinstance(dim_spec, dict):
            raise ValueError(
                f"spec {path}: {name}.{key!r} must be a mapping, "
                f"got {type(dim_spec).__name__}")
        type_ = dim_spec.get('type')
        if type_ not in SAMPLERS:
            raise ValueError(
                f"spec {path}: {name}.{key!r} has unknown type {type_!r}; "
                f"expected one of {sorted(SAMPLERS)}")


def load_spec(path):
    """Read and validate a search spec yaml.

    Schema:
        base_yaml:    str (required) — path to PSF yaml to seed trial configs
        dims:         mapping of {yaml_key: sampler_spec} — overrides written
                      into the trial PSF yaml (PSF-level search)
        trainer_dims: optional mapping of {cli_key: sampler_spec} — overrides
                      passed through as CLI args to search_trainer (trainer-
                      level search, e.g. input_channels)

    At least one of `dims` / `trainer_dims` must be non-empty.
    """
    with open(path) as f:
        spec = yaml.safe_load(f)

    if not isinstance(spec, dict):
        raise ValueError(f"spec {path} must be a yaml mapping at the top level")

    if 'base_yaml' not in spec or not isinstance(spec['base_yaml'], str):
        raise ValueError(
            f"spec {path} must define `base_yaml:` as a string path "
            f"(relative to project root or absolute)")

    dims = spec.get('dims') or {}
    trainer_dims = spec.get('trainer_dims') or {}
    if not dims and not trainer_dims:
        raise ValueError(
            f"spec {path} must define at least one of `dims:` or "
            f"`trainer_dims:` with at least one entry")

    _validate_dim_block('dims', dims, path)
    _validate_dim_block('trainer_dims', trainer_dims, path)
    return spec


def resolve_base_yaml(spec, project_root):
    """Return absolute path to base_yaml referenced by the spec."""
    p = spec['base_yaml']
    if not os.path.isabs(p):
        p = os.path.join(project_root, p)
    if not os.path.exists(p):
        raise FileNotFoundError(f"base_yaml does not exist: {p}")
    return p


def build_overrides(spec, rng):
    """Sample every dim in spec['dims'], return {yaml_key: sampled_value}.

    These overrides are written into the trial PSF yaml.
    """
    out = {}
    for key, dim_spec in (spec.get('dims') or {}).items():
        sampler = SAMPLERS[dim_spec['type']]
        out[key] = sampler(dim_spec, rng)
    return out


def build_trainer_overrides(spec, rng):
    """Sample every dim in spec['trainer_dims'], return {cli_key: sampled_value}.

    These overrides are passed through to search_trainer as CLI args. Use for
    parameters that don't belong in the PSF yaml (e.g. input_channels).
    """
    out = {}
    for key, dim_spec in (spec.get('trainer_dims') or {}).items():
        sampler = SAMPLERS[dim_spec['type']]
        out[key] = sampler(dim_spec, rng)
    return out
