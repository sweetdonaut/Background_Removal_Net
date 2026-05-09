"""Generate FP-only test images mimicking data/30ea_testing/bad/ shape & stats.

Each image: 448x464x3 uint8, background ~137 +/- small noise, scattered with
many small Gaussian PSF blobs (~8-15 per image). Filenames intentionally lack
the `#X,Y` GT marker so the evaluator treats every detection as FP.

Usage:
    python utils/generate_extra_testing.py \\
        --out_dir data/extra_testing_image \\
        --n_images 30 \\
        --seed 0
"""

import argparse
import os

import numpy as np
import tifffile


def make_psf(sigma=1.2, size=9):
    g = np.arange(size) - size // 2
    yy, xx = np.meshgrid(g, g, indexing='ij')
    blob = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return blob


def generate_image(rng, h=448, w=464, bg=137, bg_noise=2.0,
                   n_psf_range=(8, 15), peak_range=(10, 18)):
    img = rng.normal(bg, bg_noise, size=(h, w, 3)).astype(np.float32)

    psf = make_psf()
    pad = psf.shape[0] // 2
    n = rng.integers(n_psf_range[0], n_psf_range[1] + 1)
    for _ in range(n):
        cy = int(rng.integers(pad + 5, h - pad - 5))
        cx = int(rng.integers(pad + 5, w - pad - 5))
        amp = float(rng.uniform(peak_range[0], peak_range[1]))
        for c in range(3):
            ch_amp = amp * rng.uniform(0.85, 1.0)
            img[cy - pad:cy + pad + 1, cx - pad:cx + pad + 1, c] += ch_amp * psf

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_images', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(1, args.n_images + 1):
        img = generate_image(rng)
        path = os.path.join(args.out_dir, f'DefectID{i:03d}_random.tiff')
        tifffile.imwrite(path, img)
        print(f'wrote {path}  shape={img.shape} dtype={img.dtype} '
              f'min={img.min()} max={img.max()}')

    print(f'\nDone. {args.n_images} images in {args.out_dir}')


if __name__ == '__main__':
    main()
