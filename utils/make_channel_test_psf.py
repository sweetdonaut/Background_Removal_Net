"""Generate synthetic test tiffs covering all 8 (target, ref1, ref2)
channel-combination scenarios with PSF defects.

Uses the same PSF yaml as training so train/test defect distributions
align. Run from anywhere — paths are anchored to this file.
"""
import os
import sys
import numpy as np
import tifffile
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_CORE = os.path.join(PROJECT_ROOT, 'src_core')
sys.path.insert(0, SRC_CORE)

from generate_psf import load_config, create_psf_defect
from gaussian import apply_local_defect_to_background
from dataloader import ensure_hwc, sample_magnitude


GOOD_DIR = os.path.join(PROJECT_ROOT, 'data/grid_stripe_4channel/test/good')
OUT_ROOT = os.path.join(PROJECT_ROOT, 'data/synthetic_channel_test_psf')
PSF_YAML = os.path.join(SRC_CORE, 'defects/small_spot_match_test.yaml')

SCENARIOS = {
    'A_target_only':  (1, 0, 0),  # positive under objective mask
    'B_ref1_only':    (0, 1, 0),  # negative (sign mismatch)
    'C_ref2_only':    (0, 0, 1),  # negative (sign mismatch)
    'D_both_refs':    (0, 1, 1),  # positive under objective mask
    'E_all_three':    (1, 1, 1),  # negative (diff=0)
    'F_target_ref1':  (1, 1, 0),  # negative (sign mismatch)
    'G_target_ref2':  (1, 0, 1),  # negative (sign mismatch)
}
# (0,0,0) is covered by the original good/ tiffs (no injection needed)

CENTERS = [(88, 244), (88, 488), (88, 732), (88, 366)]
SEED = 42  # fixed PSF + intensity per (base_idx, sign) for reproducibility


def make_one(base_img_hwc, scenario_flags, defect, center, intensity_signed):
    """Inject the same PSF defect into channels selected by scenario_flags."""
    t_flag, r1_flag, r2_flag = scenario_flags
    h, w = base_img_hwc.shape[:2]
    dh, dw = defect.shape
    cx, cy = center
    y0 = cy - dh // 2
    x0 = cx - dw // 2
    bounds = (y0, y0 + dh, x0, x0 + dw)

    out = base_img_hwc.copy().astype(np.float32)
    if t_flag:
        out[:, :, 0] = apply_local_defect_to_background(out[:, :, 0], defect, bounds, intensity_signed)
    if r1_flag:
        out[:, :, 1] = apply_local_defect_to_background(out[:, :, 1], defect, bounds, intensity_signed)
    if r2_flag:
        out[:, :, 2] = apply_local_defect_to_background(out[:, :, 2], defect, bounds, intensity_signed)
    return out


def main():
    cfg = load_config(PSF_YAML)
    intensity_spec = cfg.get('intensity_abs', [60, 80])

    good_paths = sorted(glob(os.path.join(GOOD_DIR, '*.tiff')))
    if not good_paths:
        raise RuntimeError(f"No tiff in {GOOD_DIR}")
    print(f"Using {len(good_paths)} base images")

    rng_global = np.random.default_rng(SEED)
    # Pre-generate one PSF + intensity per base index (shared across scenarios
    # so that the same defect shape is injected in different channel combos)
    defects_per_base = []
    for i in range(len(good_paths)):
        np.random.seed(SEED + i)  # for create_psf_defect's internal rng
        d = None
        for _ in range(20):
            d = create_psf_defect(cfg)
            if d is not None and d.size > 0:
                break
        if d is None:
            raise RuntimeError(f"Failed to generate PSF for base {i}")
        magnitude = sample_magnitude(intensity_spec)
        defects_per_base.append((d, magnitude))
        print(f"  base {i}: PSF bbox={d.shape}, magnitude={magnitude:.2f}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # H_clean: no defect injected (mirrors raw good/ tiffs) — single subfolder,
    # not split by sign. Lets summarize tooling treat it uniformly.
    clean_dir = os.path.join(OUT_ROOT, 'H_clean')
    os.makedirs(clean_dir, exist_ok=True)
    for i, src in enumerate(good_paths):
        raw = tifffile.imread(src)
        base_name = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(clean_dir, f'{base_name}_H_clean.tiff')
        tifffile.imwrite(out_path, raw.astype(np.float32))
    print(f"  wrote {len(good_paths)} -> {clean_dir} (no defect)")

    for scenario_name, flags in SCENARIOS.items():
        for sign_name, sign in [('pos', +1), ('neg', -1)]:
            subdir = os.path.join(OUT_ROOT, f'{scenario_name}_{sign_name}')
            os.makedirs(subdir, exist_ok=True)
            for i, src in enumerate(good_paths):
                raw = tifffile.imread(src)
                img_hwc = ensure_hwc(raw)
                defect, magnitude = defects_per_base[i]
                intensity_signed = sign * magnitude
                center = CENTERS[i % len(CENTERS)]
                modified = make_one(img_hwc[:, :, :3], flags, defect, center, intensity_signed)
                out_hwc = img_hwc.copy().astype(np.float32)
                out_hwc[:, :, :3] = modified
                out_chw = np.transpose(out_hwc, (2, 0, 1))
                base_name = os.path.splitext(os.path.basename(src))[0]
                out_path = os.path.join(subdir, f'{base_name}_{scenario_name}_{sign_name}.tiff')
                tifffile.imwrite(out_path, out_chw.astype(np.float32))
            print(f"  wrote {len(good_paths)} -> {subdir}")

    print(f"\nDone. Output root: {OUT_ROOT}")


if __name__ == '__main__':
    main()
