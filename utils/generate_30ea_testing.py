"""Generate 30 synthetic test images for PSF detection validation.

Each image: 448x464, 3-channel TIFF (target, ref1, ref2).
Target has one weak PSF at recorded (x, y); refs do not.
All channels share structural nuisance with per-channel perturbation
(simulating ref alignment imperfection). PSF intensity matches
type4_vector.yaml's intensity_abs range so SNR ~ 1, like production.
"""

import os
import sys

import numpy as np
import tifffile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src_core'))
from generate_psf import create_psf_defect, load_config as load_psf_config  # noqa: E402


H, W = 448, 464
N_IMAGES = 30
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', '30ea_testing', 'bad')
PSF_CFG_PATH = os.path.join(PROJECT_ROOT, 'src_core', 'defects', 'type4_vector.yaml')

PSF_INTENSITY_RANGE = (8, 12)
SEED = 20260506


def low_freq_gradient(H, W, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = rng.uniform(-W * 0.2, W * 1.2)
    cy = rng.uniform(-H * 0.2, H * 1.2)
    sigma = rng.uniform(W * 0.6, W * 1.4)
    grad = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return grad * rng.uniform(15, 35)


def mid_freq_blobs(H, W, rng, n_range=(20, 40), amp_range=(4, 12)):
    img = np.zeros((H, W), dtype=np.float32)
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    for _ in range(n):
        bx = float(rng.integers(5, W - 5))
        by = float(rng.integers(5, H - 5))
        sig = float(rng.uniform(1.2, 3.5))
        amp = float(rng.uniform(*amp_range)) * (1 if rng.random() < 0.5 else -1)
        ys, ye = max(0, int(by - 10)), min(H, int(by + 10))
        xs, xe = max(0, int(bx - 10)), min(W, int(bx + 10))
        ly, lx = np.mgrid[ys:ye, xs:xe].astype(np.float32)
        blob = np.exp(-((lx - bx) ** 2 + (ly - by) ** 2) / (2 * sig ** 2))
        img[ys:ye, xs:xe] += blob * amp
    return img


def streak_artifacts(H, W, rng, prob=0.4):
    img = np.zeros((H, W), dtype=np.float32)
    if rng.random() > prob:
        return img
    n = int(rng.integers(2, 6))
    for _ in range(n):
        if rng.random() < 0.5:
            row = int(rng.integers(0, H))
            amp = float(rng.uniform(2, 5)) * (1 if rng.random() < 0.5 else -1)
            img[row, :] += amp
        else:
            col = int(rng.integers(0, W))
            amp = float(rng.uniform(2, 5)) * (1 if rng.random() < 0.5 else -1)
            img[:, col] += amp
    return img


def make_image(rng, psf_cfg):
    base = float(rng.uniform(80, 130))
    target = np.full((H, W), base, dtype=np.float32)
    ref1 = target.copy()
    ref2 = target.copy()

    shared_grad = low_freq_gradient(H, W, rng)
    target += shared_grad * rng.uniform(0.95, 1.05)
    ref1 += shared_grad * rng.uniform(0.95, 1.05)
    ref2 += shared_grad * rng.uniform(0.95, 1.05)

    shared_blobs = mid_freq_blobs(H, W, rng)
    target += shared_blobs * rng.uniform(0.92, 1.08)
    ref1 += shared_blobs * rng.uniform(0.92, 1.08)
    ref2 += shared_blobs * rng.uniform(0.92, 1.08)

    # Per-channel independent nuisance: mimics ref alignment residuals
    target += mid_freq_blobs(H, W, rng, n_range=(5, 12), amp_range=(2, 6))
    ref1 += mid_freq_blobs(H, W, rng, n_range=(5, 12), amp_range=(2, 6))
    ref2 += mid_freq_blobs(H, W, rng, n_range=(5, 12), amp_range=(2, 6))

    target += streak_artifacts(H, W, rng)
    ref1 += streak_artifacts(H, W, rng)
    ref2 += streak_artifacts(H, W, rng)

    # PSF on target only — peak placed exactly at recorded (cx, cy)
    psf = None
    while psf is None or psf.size == 0:
        psf = create_psf_defect(psf_cfg)

    psf_h, psf_w = psf.shape
    peak_y, peak_x = np.unravel_index(int(psf.argmax()), psf.shape)

    margin = max(psf_h, psf_w) + 10
    cx = int(rng.integers(margin, W - margin))
    cy = int(rng.integers(margin, H - margin))

    y0 = cy - peak_y
    x0 = cx - peak_x
    intensity = float(rng.uniform(*PSF_INTENSITY_RANGE))
    target[y0:y0 + psf_h, x0:x0 + psf_w] += psf * intensity

    target += rng.normal(0, 1.5, (H, W)).astype(np.float32)
    ref1 += rng.normal(0, 1.5, (H, W)).astype(np.float32)
    ref2 += rng.normal(0, 1.5, (H, W)).astype(np.float32)

    target = np.clip(target, 0, 255).astype(np.uint8)
    ref1 = np.clip(ref1, 0, 255).astype(np.uint8)
    ref2 = np.clip(ref2, 0, 255).astype(np.uint8)

    image = np.stack([target, ref1, ref2], axis=-1)
    return image, cx, cy, intensity


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    psf_cfg = load_psf_config(PSF_CFG_PATH)
    rng = np.random.default_rng(SEED)

    for i in range(N_IMAGES):
        image, cx, cy, intensity = make_image(rng, psf_cfg)
        fname = f'DefectID{i + 1:03d}#{cx},{cy}.tiff'
        tifffile.imwrite(os.path.join(OUT_DIR, fname), image)
        print(f'  {fname}  shape={image.shape}  intensity={intensity:.2f}')

    print(f'\nDone. {N_IMAGES} test images written to {OUT_DIR}')


if __name__ == '__main__':
    main()
