"""Simulate 100 'machine-captured' defect TIFFs for testing the PSF fitter.

Outputs:
    data/synthetic_defects_v1/
        DID000#x,y.tiff ... DID099#x,y.tiff   (3-channel float32, defect on target only)
        ground_truth.json                      (per-defect: type, location, intensity, ...)
        type_strong.yaml, type_weak.yaml       (the source configs used to generate)

Two defect types are seeded 50/50:
    STRONG  vector linX, intensity 100-140  (elongated + bright)
    WEAK    vector radial, intensity 20-40  (round + dim)
"""

import json
import os
import sys
from glob import glob

import numpy as np
import tifffile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_core'))
from generate_psf import create_psf_defect  # noqa: E402


SEED = 1234
N_DEFECTS = 100
TRAIN_DIR = 'data/grid_stripe_4channel/train/good'
OUT_DIR = 'data/synthetic_defects_v1'

TYPE_STRONG = {
    'psf_size': 256, 'crop_size': 32,
    'outer_r': (30, 30), 'epsilon': (0.6, 0.6),
    'ellipticity': (0, 0), 'ellip_angle': (0, 0),
    'square_eps': 0, 'h_stripe_w': 0, 'v_stripe_w': 0,
    'h_outer_crop': 0, 'v_outer_crop': 0,
    'vector_mode': True, 'na': (0.95, 0.95), 'pol_type': 'linX',
    'defocus': (0, 0),
    'astig_x': (0, 0), 'astig_y': (0, 0),
    'coma_x': (0, 0), 'coma_y': (0, 0),
    'spherical': (0, 0),
    'trefoil_x': (0, 0), 'trefoil_y': (0, 0),
    'brightness': (8000, 8000), 'background': (5, 5),
    'poisson_noise': False, 'gaussian_sigma': (0.5, 0.5),
    'threshold_multiplier': 0.5,
    'intensity_abs': (100, 140),
}

TYPE_WEAK = {
    'psf_size': 256, 'crop_size': 32,
    'outer_r': (30, 30), 'epsilon': (0, 0),
    'ellipticity': (0, 0), 'ellip_angle': (0, 0),
    'square_eps': 0, 'h_stripe_w': 0, 'v_stripe_w': 0,
    'h_outer_crop': 0, 'v_outer_crop': 0,
    'vector_mode': True, 'na': (0.95, 0.95), 'pol_type': 'radial',
    'defocus': (0, 0),
    'astig_x': (0, 0), 'astig_y': (0, 0),
    'coma_x': (0, 0), 'coma_y': (0, 0),
    'spherical': (0, 0),
    'trefoil_x': (0, 0), 'trefoil_y': (0, 0),
    'brightness': (5000, 5000), 'background': (5, 5),
    'poisson_noise': False, 'gaussian_sigma': (0.5, 0.5),
    'threshold_multiplier': 0.7,
    'intensity_abs': (20, 40),
}


def build_pool(cfg, size=200):
    pool = []
    while len(pool) < size:
        d = create_psf_defect(cfg)
        if d is not None and d.size > 1:
            pool.append(d)
    return pool


def to_yaml(cfg, path):
    """Dump cfg to a yaml file with proper list (not tuple) format for re-loading."""
    plain = {}
    for k, v in cfg.items():
        plain[k] = list(v) if isinstance(v, tuple) else v
    with open(path, 'w') as f:
        yaml.dump(plain, f, sort_keys=False, default_flow_style=None)


def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    backgrounds = sorted(glob(os.path.join(TRAIN_DIR, '*.tiff')))
    if len(backgrounds) < N_DEFECTS:
        raise RuntimeError(f'Need at least {N_DEFECTS} backgrounds in {TRAIN_DIR}, found {len(backgrounds)}')
    rng.shuffle(backgrounds)
    backgrounds = backgrounds[:N_DEFECTS]

    print('Pre-generating defect pools...')
    pool_strong = build_pool(TYPE_STRONG, size=200)
    pool_weak = build_pool(TYPE_WEAK, size=200)
    print(f'  STRONG pool: {len(pool_strong)}  WEAK pool: {len(pool_weak)}')

    os.makedirs(OUT_DIR, exist_ok=True)
    gt = []

    for i, bg_path in enumerate(backgrounds):
        bg = tifffile.imread(bg_path).astype(np.float32)  # (4, H, W) CHW
        if bg.shape[0] != 4:
            raise RuntimeError(f'Expected (4, H, W) CHW, got {bg.shape} for {bg_path}')
        img = bg[:3].copy()
        _, H, W = img.shape

        is_strong = bool(rng.random() < 0.5)
        if is_strong:
            pool, intensity_range, type_name = pool_strong, TYPE_STRONG['intensity_abs'], 'strong'
        else:
            pool, intensity_range, type_name = pool_weak, TYPE_WEAK['intensity_abs'], 'weak'

        defect = pool[rng.integers(len(pool))]
        dh, dw = defect.shape

        margin = 20
        y0 = int(rng.integers(margin, H - dh - margin))
        x0 = int(rng.integers(margin, W - dw - margin))
        cy, cx = y0 + dh // 2, x0 + dw // 2

        magnitude = float(rng.uniform(*intensity_range))
        sign = 1 if rng.random() < 0.5 else -1
        intensity = sign * magnitude

        target = img[0]
        target[y0:y0 + dh, x0:x0 + dw] += defect * intensity
        np.clip(target, 0, 255, out=target)

        fname = f'DID{i:03d}#{cx},{cy}.tiff'
        tifffile.imwrite(os.path.join(OUT_DIR, fname), img)

        gt.append({
            'filename': fname,
            'background_source': os.path.basename(bg_path),
            'type': type_name,
            'cx': cx, 'cy': cy,
            'defect_h': int(dh), 'defect_w': int(dw),
            'intensity': float(intensity),
        })

    with open(os.path.join(OUT_DIR, 'ground_truth.json'), 'w') as f:
        json.dump({'seed': SEED, 'records': gt}, f, indent=2)

    to_yaml(TYPE_STRONG, os.path.join(OUT_DIR, 'type_strong.yaml'))
    to_yaml(TYPE_WEAK, os.path.join(OUT_DIR, 'type_weak.yaml'))

    n_strong = sum(1 for r in gt if r['type'] == 'strong')
    n_weak = N_DEFECTS - n_strong
    print(f'\nDone. Wrote {N_DEFECTS} files to {OUT_DIR}/')
    print(f'  STRONG: {n_strong}, WEAK: {n_weak}')


if __name__ == '__main__':
    main()
