"""Real-data preprocessing for the inverse fit.

Pipeline per defect TIFF:
  1. Load triplet via tifffile + ensure_hwc/ensure_3ch from src_core.
  2. Parse (defect_id, X, Y) from filename `DefectID###.#X,Y.tiff` (same regex
     as src_search/evaluator.py:GT_PATTERN).
  3. Crop a (crop_size + pad) bbox centered on (X, Y); pad gives sub-pixel
     phase correlation margin and tolerates if (X, Y) is on an image edge.
  4. Phase-correlate ref1/ref2 against target and integer-shift them so the
     wafer texture aligns; this kills most of the misregistration nuisance
     before subtraction.
  5. Tighten to crop_size after registration; compute diff1/diff2 = target - aligned_ref.

Sub-pixel registration is intentionally not done in v1 — integer alignment
plus the per-pixel min loss in fit_three_channel handles the residual. We can
add sub-pixel later (parabola fit on the 3x3 around the correlation peak)
if the registration residual turns out to dominate the fit loss.
"""

import os
import re
import sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src_core'))

from dataloader import ensure_hwc, ensure_3ch  # noqa: E402


GT_PATTERN = re.compile(r'(DefectID\d+)#(\d+),(\d+)\.tiff?$', re.IGNORECASE)


def parse_filename(path):
    """Return (defect_id, x, y) from a path matching DefectID###.#X,Y.tiff."""
    m = GT_PATTERN.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Unrecognized filename: {os.path.basename(path)}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_triplet(path):
    """Return (target, ref1, ref2) as float32 (H, W) arrays."""
    img = tifffile.imread(path)
    img = ensure_hwc(img)
    img = ensure_3ch(img)  # discards 4th channel if present
    img = img.astype(np.float32)
    target = img[:, :, 0]
    ref1 = img[:, :, 1]
    ref2 = img[:, :, 2]
    return target, ref1, ref2


def _crop_centered(arr, cy, cx, size):
    """Crop `size`x`size` centered at (cy, cx); zero-pads if near edge."""
    H, W = arr.shape
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    out = np.zeros((size, size), dtype=arr.dtype)
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    src_y1 = size - max(0, y1 - H)
    src_x1 = size - max(0, x1 - W)
    arr_y0 = max(0, y0)
    arr_x0 = max(0, x0)
    arr_y1 = min(H, y1)
    arr_x1 = min(W, x1)

    if arr_y1 > arr_y0 and arr_x1 > arr_x0:
        out[src_y0:src_y1, src_x0:src_x1] = arr[arr_y0:arr_y1, arr_x0:arr_x1]
    return out


def integer_shift(arr, dy, dx, fill_value=0.0):
    """Return arr shifted by (dy, dx) integer pixels, edges filled with fill_value."""
    out = np.full_like(arr, fill_value)
    H, W = arr.shape
    src_y0 = max(0, -dy)
    src_x0 = max(0, -dx)
    src_y1 = H - max(0, dy)
    src_x1 = W - max(0, dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    dst_y1 = H - max(0, -dy)
    dst_x1 = W - max(0, -dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def phase_correlate_shift(target, ref, max_shift=10):
    """Estimate integer (dy, dx) shift to align ref onto target via phase correlation.

    Cross-power spectrum normalized to unit magnitude, inverse FFT, peak find.
    The peak (dy, dx) tells us how much to shift ref so its content matches
    target at the same coordinate.

    `max_shift` clamps the search radius to ±max_shift pixels in both axes.
    Phase correlation peaks can land on a far wrap-around when the actual
    drift is small but the wafer texture self-correlates at multiple offsets;
    real captures drift sub-10 px so we search a tight window only.
    """
    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    target = target - target.mean()
    ref = ref - ref.mean()

    F1 = np.fft.fft2(target)
    F2 = np.fft.fft2(ref)
    cross = F1 * np.conj(F2)
    cross_norm = cross / (np.abs(cross) + 1e-12)
    corr = np.fft.ifft2(cross_norm).real

    H, W = corr.shape
    # Build a mask that's 1 only within ±max_shift of (0, 0) under fft wrap.
    yy = np.arange(H)
    xx = np.arange(W)
    yy = np.where(yy > H // 2, yy - H, yy)
    xx = np.where(xx > W // 2, xx - W, xx)
    yy_g, xx_g = np.meshgrid(yy, xx, indexing='ij')
    mask = (np.abs(yy_g) <= max_shift) & (np.abs(xx_g) <= max_shift)
    corr_masked = np.where(mask, corr, -np.inf)

    py, px = np.unravel_index(np.argmax(corr_masked), corr_masked.shape)
    if py > H // 2:
        py -= H
    if px > W // 2:
        px -= W
    return int(py), int(px)


def preprocess_one(path, fit_crop_size, reg_crop_size=None,
                   subtract_diff_median=True):
    """Full pipeline for one defect tiff.

    fit_crop_size : final (H, W) of target/ref/diff arrays returned for fitting.
                    Must equal fwd_cfg.crop_size.
    reg_crop_size : larger context size for phase correlation. Default
                    max(96, 3 * fit_crop_size). The defect signal dominates a
                    small crop and biases the correlation peak; using a wider
                    window gives the wafer texture (the actual common signal
                    between target and ref) enough weight to drive registration.
    subtract_diff_median :
                    Per-channel median-subtract diff1/diff2 before returning.
                    When the three captures have small global brightness drift
                    (different exposure / sensor gain / illumination between
                    captures) the resulting diff carries a uniform offset that
                    forward(theta)*I cannot fit (forward is non-negative).
                    Median is robust to the PSF (which affects <10% of pixels),
                    so subtracting it cleanly removes the wafer-level mismatch
                    without distorting the PSF signal. Disable only for debug.

    Returns dict:
        defect_id, x, y    : metadata
        target, ref1, ref2 : (fit_crop_size,)*2 float32; refs are post-registration
        diff1, diff2       : target - registered ref{1,2}, optionally
                             median-subtracted
        diff_offset1, diff_offset2 : the per-channel medians removed (0 if
                             subtract_diff_median=False)
        shift1, shift2     : integer (dy, dx) found by phase correlation
        big_target, big_ref1_aligned, big_ref2_aligned :
                             (reg_crop_size,)*2 — kept for diagnostic visualization
    """
    defect_id, x, y = parse_filename(path)
    target_full, ref1_full, ref2_full = load_triplet(path)

    if reg_crop_size is None:
        reg_crop_size = max(96, 3 * fit_crop_size)
    if reg_crop_size < fit_crop_size:
        raise ValueError("reg_crop_size must be >= fit_crop_size")

    bt = _crop_centered(target_full, y, x, reg_crop_size)
    br1 = _crop_centered(ref1_full, y, x, reg_crop_size)
    br2 = _crop_centered(ref2_full, y, x, reg_crop_size)

    dy1, dx1 = phase_correlate_shift(bt, br1)
    dy2, dx2 = phase_correlate_shift(bt, br2)
    fill = float(bt.mean())
    br1_aligned = integer_shift(br1, dy1, dx1, fill_value=fill)
    br2_aligned = integer_shift(br2, dy2, dx2, fill_value=fill)

    s = (reg_crop_size - fit_crop_size) // 2
    target = bt[s:s + fit_crop_size, s:s + fit_crop_size]
    ref1 = br1_aligned[s:s + fit_crop_size, s:s + fit_crop_size]
    ref2 = br2_aligned[s:s + fit_crop_size, s:s + fit_crop_size]

    diff1 = target - ref1
    diff2 = target - ref2

    if subtract_diff_median:
        m1 = float(np.median(diff1))
        m2 = float(np.median(diff2))
        diff1 = diff1 - m1
        diff2 = diff2 - m2
    else:
        m1 = m2 = 0.0

    return {
        'defect_id': defect_id, 'x': x, 'y': y,
        'target': target, 'ref1': ref1, 'ref2': ref2,
        'diff1': diff1, 'diff2': diff2,
        'diff_offset1': m1, 'diff_offset2': m2,
        'shift1': (dy1, dx1), 'shift2': (dy2, dx2),
        'big_target': bt,
        'big_ref1_aligned': br1_aligned,
        'big_ref2_aligned': br2_aligned,
    }
