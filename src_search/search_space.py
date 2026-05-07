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


SAMPLERS = {
    'scalar_pair': _sample_scalar_pair,
    'range_pair': _sample_range_pair,
}


def load_spec(path):
    """Read and validate a search spec yaml.

    Returns the parsed dict. Raises ValueError on schema problems with a
    message specific enough to fix the spec without reading source.
    """
    with open(path) as f:
        spec = yaml.safe_load(f)

    if not isinstance(spec, dict):
        raise ValueError(f"spec {path} must be a yaml mapping at the top level")

    if 'base_yaml' not in spec or not isinstance(spec['base_yaml'], str):
        raise ValueError(
            f"spec {path} must define `base_yaml:` as a string path "
            f"(relative to project root or absolute)")

    dims = spec.get('dims')
    if not isinstance(dims, dict) or not dims:
        raise ValueError(f"spec {path} must define a non-empty `dims:` mapping")

    for key, dim_spec in dims.items():
        if not isinstance(dim_spec, dict):
            raise ValueError(
                f"spec {path}: dim {key!r} must be a mapping, "
                f"got {type(dim_spec).__name__}")
        type_ = dim_spec.get('type')
        if type_ not in SAMPLERS:
            raise ValueError(
                f"spec {path}: dim {key!r} has unknown type {type_!r}; "
                f"expected one of {sorted(SAMPLERS)}")

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
    """Sample every dim in spec, return {yaml_key: sampled_value}."""
    out = {}
    for key, dim_spec in spec['dims'].items():
        sampler = SAMPLERS[dim_spec['type']]
        out[key] = sampler(dim_spec, rng)
    return out
