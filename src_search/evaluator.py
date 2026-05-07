"""Detection evaluator for the 30-PSF real-validation set.

Reproduces the production detection pipeline:
    1. Top-1% percentile threshold per image (adaptive to heatmap scale)
    2. Greedy NMS: pick max within threshold mask, mask out 20-pixel disk, repeat
    3. Detection score = mean of top-K pixels in a small window around peak
    4. Cap at max_per_image detections per image (production: 5)

Metric:
    Cross-image ranking — pool all detections from all images, sort globally
    by score, compute recall@K. This matches how production review surfaces
    the highest-scoring candidates first across the whole batch.
"""

import csv
import glob
import os
import re
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src_core'))
from dataloader import calculate_positions, ensure_3ch, ensure_hwc  # noqa: E402

GT_PATTERN = re.compile(r'#(\d+),(\d+)\.tiff?$', re.IGNORECASE)


def parse_gt_from_filename(path):
    m = GT_PATTERN.search(os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def load_test_image(path):
    image = tifffile.imread(path)
    image = ensure_hwc(image)
    image = ensure_3ch(image)
    return image.astype(np.float32)


def load_dead_pixels(csv_path):
    """Read CSV with columns (dead_x, dead_y). Returns list of (x, y) ints.
    Returns [] if csv_path is None or the file does not exist."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    out = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((int(row['dead_x']), int(row['dead_y'])))
    return out


def apply_dead_pixel_mask(heatmap, dead_pixels, half_size=5, fill_value=0.0):
    """In-place mask of a (2*half_size)x(2*half_size) bbox around each dead pixel.
    half_size=5 yields a 10x10 bbox per pixel.
    fill_value=0.0 keeps the heatmap finite so percentile / score-window math
    is unaffected; with softmax-of-class-1 backgrounds (~1e-3) the masked
    region is guaranteed to fall below the top-1% threshold."""
    if not dead_pixels:
        return heatmap
    H, W = heatmap.shape
    for (cx, cy) in dead_pixels:
        ys = max(0, cy - half_size)
        ye = min(H, cy + half_size)
        xs = max(0, cx - half_size)
        xe = min(W, cx + half_size)
        if ys < ye and xs < xe:
            heatmap[ys:ye, xs:xe] = fill_value
    return heatmap


def sliding_window_heatmap(image, model, patch_size, device):
    """Average-stitch sliding window heatmap of P(class=1)."""
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size

    y_positions = calculate_positions(h, patch_h)
    x_positions = calculate_positions(w, patch_w)
    if y_positions is None or x_positions is None:
        raise ValueError(f'Image {h}x{w} too small for patch {patch_h}x{patch_w}')

    heatmap = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = image[y:y + patch_h, x:x + patch_w]
                three = np.stack(
                    [patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]], axis=0)
                tensor = torch.from_numpy(three).float().unsqueeze(0).to(device) / 255.0
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                patch_score = probs[0, 1].cpu().numpy()
                heatmap[y:y + patch_h, x:x + patch_w] += patch_score
                weight[y:y + patch_h, x:x + patch_w] += 1.0

    return heatmap / np.maximum(weight, 1e-8)


def detect_in_image(
    heatmap,
    top_pct=0.01,
    mask_radius=20,
    score_window=2,
    score_topk=3,
    max_per_image=5,
):
    """Production detection pipeline on a single heatmap.

    Returns list of (y, x, score), ordered by extraction (highest first).
        top_pct        : keep pixels ≥ percentile threshold (e.g. 0.01 = top 1%)
        mask_radius    : NMS disk radius in pixels (production: 20)
        score_window   : half-window size for top-K score window (e.g. 2 -> 5x5)
        score_topk     : average of top-K pixel scores within window (production: 3)
        max_per_image  : hard cap on detections per image (production: 5)
    """
    H, W = heatmap.shape
    threshold = float(np.percentile(heatmap, 100.0 * (1.0 - top_pct)))
    valid = heatmap >= threshold

    work = heatmap.copy()
    work[~valid] = -np.inf

    yy, xx = np.ogrid[:H, :W]
    detections = []
    r2 = mask_radius * mask_radius

    for _ in range(max_per_image):
        peak_idx = int(np.argmax(work))
        py, px = peak_idx // W, peak_idx % W
        if not np.isfinite(work[py, px]):
            break

        ys, ye = max(0, py - score_window), min(H, py + score_window + 1)
        xs, xe = max(0, px - score_window), min(W, px + score_window + 1)
        window_vals = heatmap[ys:ye, xs:xe].ravel()
        k = min(score_topk, window_vals.size)
        topk = np.partition(window_vals, -k)[-k:]
        score = float(topk.mean())

        detections.append((py, px, score))

        disk = (yy - py) ** 2 + (xx - px) ** 2 <= r2
        work[disk] = -np.inf

    return detections


def cross_image_metrics(per_image, match_radius=3.0, top_k_list=(30, 50, 150)):
    """Pool detections globally, sort by score, compute recall@K.

    per_image: list of {'detections': [(y,x,score), ...], 'gt': (gx, gy), 'image_id': str}
    """
    all_dets = []
    for entry in per_image:
        gx, gy = entry['gt']
        img_id = entry['image_id']
        for (dy, dx, score) in entry['detections']:
            is_tp = (dx - gx) ** 2 + (dy - gy) ** 2 <= match_radius ** 2
            all_dets.append({
                'score': score,
                'is_tp': bool(is_tp),
                'image_id': img_id,
            })

    all_dets.sort(key=lambda d: -d['score'])

    n_gts = len(per_image)
    matched_imgs = set()
    cumulative = []
    for d in all_dets:
        if d['is_tp'] and d['image_id'] not in matched_imgs:
            matched_imgs.add(d['image_id'])
        cumulative.append(len(matched_imgs))

    metrics = {}
    for k in top_k_list:
        if not cumulative:
            metrics[f'recall@{k}'] = 0.0
        elif len(cumulative) >= k:
            metrics[f'recall@{k}'] = cumulative[k - 1] / n_gts
        else:
            metrics[f'recall@{k}'] = cumulative[-1] / n_gts

    metrics['n_total_detections'] = len(all_dets)
    metrics['n_gts'] = n_gts
    metrics['total_recall'] = cumulative[-1] / n_gts if cumulative else 0.0
    return metrics


def per_defect_summary(per_image):
    """Flatten per_image list into a defect-id keyed dict for jsonl logging.

    Each value: {matched, local_rank, score, dist}.
        matched    : True if a detection lies within match_radius of GT
        local_rank : index in this image's detection list (0 = top), else None
        score      : matched detection's score, else top FP score (best fail)
        dist       : pixel distance from matched detection to GT, else None
    """
    out = {}
    for entry in per_image:
        defect_id = entry['image_id'].split('#')[0]
        gt_x, gt_y = entry['gt']
        rank = entry['gt_local_rank']
        if rank is not None:
            dy, dx, score = entry['detections'][rank]
            dist = float(((dx - gt_x) ** 2 + (dy - gt_y) ** 2) ** 0.5)
            out[defect_id] = {
                'matched': True,
                'local_rank': int(rank),
                'score': float(score),
                'dist': dist,
            }
        else:
            top_score = entry['detections'][0][2] if entry['detections'] else 0.0
            out[defect_id] = {
                'matched': False,
                'local_rank': None,
                'score': float(top_score),
                'dist': None,
            }
    return out


def evaluate_real(
    model,
    test_dir,
    patch_size,
    device,
    top_pct=0.01,
    mask_radius=20,
    score_window=2,
    score_topk=3,
    max_per_image=5,
    match_radius=3.0,
    top_k_list=(30, 50, 150),
    dead_pixel_csv=None,
    dead_pixel_half_size=5,
    verbose=False,
):
    """Run model on test_dir, run production-style detection, return cross-image metrics.

    dead_pixel_csv: optional path to a CSV with columns (dead_x, dead_y).
        Each listed pixel masks a (2*dead_pixel_half_size)^2 bbox to 0 on the
        heatmap before detection — kills permanent FPs from sensor dead pixels.
        Defaults to '<test_dir>/dead_pixels.csv' if that file exists.
    """
    model.eval()

    paths = sorted(
        glob.glob(os.path.join(test_dir, '*.tiff'))
        + glob.glob(os.path.join(test_dir, '*.tif'))
    )
    if not paths:
        raise ValueError(f'No tiff images found in {test_dir}')

    if dead_pixel_csv is None:
        default_dp = os.path.join(test_dir, 'dead_pixels.csv')
        if os.path.exists(default_dp):
            dead_pixel_csv = default_dp
    dead_pixels = load_dead_pixels(dead_pixel_csv)
    if dead_pixels:
        print(f'  [dead_pixel_mask] {len(dead_pixels)} pixels '
              f'from {dead_pixel_csv} (bbox={2 * dead_pixel_half_size}px)')

    per_image = []
    for path in paths:
        gt = parse_gt_from_filename(path)
        if gt is None:
            print(f'Warning: cannot parse GT from {path}')
            continue
        gt_x, gt_y = gt

        image = load_test_image(path)
        heatmap = sliding_window_heatmap(image, model, patch_size, device)
        apply_dead_pixel_mask(heatmap, dead_pixels, half_size=dead_pixel_half_size)
        detections = detect_in_image(
            heatmap,
            top_pct=top_pct, mask_radius=mask_radius,
            score_window=score_window, score_topk=score_topk,
            max_per_image=max_per_image,
        )

        gt_match_local_rank = None
        for r, (dy, dx, _) in enumerate(detections):
            if (dx - gt_x) ** 2 + (dy - gt_y) ** 2 <= match_radius ** 2:
                gt_match_local_rank = r
                break

        per_image.append({
            'image_id': os.path.basename(path),
            'gt': (gt_x, gt_y),
            'detections': detections,
            'gt_local_rank': gt_match_local_rank,
        })

        if verbose:
            top_score = detections[0][2] if detections else 0.0
            print(f'  {os.path.basename(path):35s}  '
                  f'local_rank={gt_match_local_rank}  '
                  f'n_det={len(detections)}  top={top_score:.4f}')

    metrics = cross_image_metrics(
        per_image, match_radius=match_radius, top_k_list=top_k_list)
    metrics['per_image'] = per_image
    return metrics
