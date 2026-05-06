"""Sanity tests for src_search/evaluator.py.

Verifies detection + cross-image ranking on synthetic heatmaps so the
metric layer is trustworthy before running on a real model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import (  # noqa: E402
    cross_image_metrics,
    detect_in_image,
)


H, W = 448, 464


def perfect_heatmap(gt_x, gt_y, peak=1.0):
    hm = np.full((H, W), 0.001, dtype=np.float32)
    hm[gt_y, gt_x] = peak
    hm[max(0, gt_y - 1):gt_y + 2, max(0, gt_x - 1):gt_x + 2] = peak * 0.9
    return hm


def near_miss_heatmap(gt_x, gt_y, offset=2):
    hm = np.full((H, W), 0.001, dtype=np.float32)
    py, px = gt_y + offset, gt_x + offset
    hm[py, px] = 1.0
    hm[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = 0.9
    return hm


def far_miss_heatmap(gt_x, gt_y, offset=50):
    hm = np.full((H, W), 0.001, dtype=np.float32)
    py = (gt_y + offset) % H
    px = (gt_x + offset) % W
    hm[py, px] = 1.0
    return hm


def random_heatmap(seed):
    rng = np.random.default_rng(seed)
    return rng.random((H, W)).astype(np.float32)


def build_per_image(heatmaps_and_gts):
    """Wrap (heatmap, gt_x, gt_y) tuples into per_image dicts via detect_in_image."""
    per_image = []
    for i, (hm, gx, gy) in enumerate(heatmaps_and_gts):
        per_image.append({
            'image_id': f'img_{i:03d}',
            'gt': (int(gx), int(gy)),
            'detections': detect_in_image(hm),
        })
    return per_image


def main():
    rng = np.random.default_rng(0)
    n = 30
    gt_xs = rng.integers(50, W - 50, size=n)
    gt_ys = rng.integers(50, H - 50, size=n)

    print('Test 1: perfect heatmaps -> recall@30 = 1.0')
    items = [(perfect_heatmap(gx, gy), gx, gy) for gx, gy in zip(gt_xs, gt_ys)]
    pi = build_per_image(items)
    m = cross_image_metrics(pi, top_k_list=(30, 50, 150))
    print(f'  recall@30={m["recall@30"]:.3f}  recall@150={m["recall@150"]:.3f}  '
          f'n_det={m["n_total_detections"]}  total_recall={m["total_recall"]:.3f}')
    assert m['recall@30'] == 1.0
    print('  PASS\n')

    print('Test 2: near-miss (2px) -> recall@30 = 1.0 (within match_radius=3)')
    items = [(near_miss_heatmap(gx, gy, offset=2), gx, gy) for gx, gy in zip(gt_xs, gt_ys)]
    pi = build_per_image(items)
    m = cross_image_metrics(pi, top_k_list=(30, 150), match_radius=3.0)
    print(f'  recall@30={m["recall@30"]:.3f}  recall@150={m["recall@150"]:.3f}')
    assert m['recall@30'] == 1.0
    print('  PASS\n')

    print('Test 3: far-miss (50px) -> recall = 0')
    items = [(far_miss_heatmap(gx, gy, offset=50), gx, gy) for gx, gy in zip(gt_xs, gt_ys)]
    pi = build_per_image(items)
    m = cross_image_metrics(pi, top_k_list=(30, 150), match_radius=3.0)
    print(f'  recall@30={m["recall@30"]:.3f}  recall@150={m["recall@150"]:.3f}  '
          f'total_recall={m["total_recall"]:.3f}')
    assert m['recall@30'] == 0.0
    assert m['total_recall'] == 0.0
    print('  PASS\n')

    print('Test 4: random heatmaps -> recall low')
    items = [(random_heatmap(int(gx) * 31 + int(gy)), gx, gy)
             for gx, gy in zip(gt_xs, gt_ys)]
    pi = build_per_image(items)
    m = cross_image_metrics(pi, top_k_list=(30, 150))
    print(f'  recall@30={m["recall@30"]:.3f}  recall@150={m["recall@150"]:.3f}  '
          f'total_recall={m["total_recall"]:.3f}  n_det={m["n_total_detections"]}')
    assert m['recall@150'] < 0.2, f'random recall@150 should be low'
    print('  PASS\n')

    print('Test 5: detect_in_image NMS — adjacent peaks collapse, distant ones survive')
    hm = np.full((H, W), 0.001, dtype=np.float32)
    hm[100, 100] = 0.9
    hm[100, 105] = 0.85   # 5px away — within 20-px NMS, should be suppressed
    hm[100, 130] = 0.7    # 30px away — survives
    hm[300, 300] = 0.5    # far away — survives
    dets = detect_in_image(hm, mask_radius=20, max_per_image=5)
    print(f'  detections: {[(y, x, round(s, 3)) for y, x, s in dets]}')
    coords = {(y, x) for y, x, _ in dets}
    assert (100, 100) in coords
    assert (100, 105) not in coords
    assert (100, 130) in coords
    assert (300, 300) in coords
    print('  PASS\n')

    print('Test 6: max_per_image=5 hard cap')
    hm = np.full((H, W), 0.001, dtype=np.float32)
    for i, (y, x) in enumerate([(50, 50), (50, 150), (50, 250), (50, 350),
                                 (150, 50), (150, 150), (150, 250), (150, 350)]):
        hm[y, x] = 0.9 - 0.01 * i
    dets = detect_in_image(hm, mask_radius=20, max_per_image=5)
    assert len(dets) == 5, f'expected 5 detections, got {len(dets)}'
    print(f'  got {len(dets)} detections (capped at 5)')
    print('  PASS\n')

    print('Test 7: cross-image ranking respects global score order')
    # Two images, image A's TP score = 0.9, image B's FP score = 0.95
    # Cross-image: top-1 is B's FP, top-2 is A's TP
    # recall@1 = 0/2 = 0; recall@2 = 1/2 = 0.5
    per_image = [
        {'image_id': 'A', 'gt': (100, 100),
         'detections': [(100, 100, 0.9)]},                   # TP
        {'image_id': 'B', 'gt': (200, 200),
         'detections': [(50, 50, 0.95)]},                    # FP (far from GT)
    ]
    m = cross_image_metrics(per_image, top_k_list=(1, 2), match_radius=3.0)
    print(f'  recall@1={m["recall@1"]:.3f}  recall@2={m["recall@2"]:.3f}')
    assert m['recall@1'] == 0.0, 'top-1 is B\'s FP, no TP yet'
    assert m['recall@2'] == 0.5, 'top-2 is A\'s TP, 1/2 GTs covered'
    print('  PASS\n')

    print('All tests passed.')


if __name__ == '__main__':
    main()
