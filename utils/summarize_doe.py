"""Compare multiple DoE checkpoints over the eight-scenario test set.

Each scenario folder under ROOT is run through every checkpoint; we report
peak heatmap score inside a small window around the injected defect center,
plus global max. H_clean has no injection so we report only global max.

Expected outcome under the objective mask rule:
  A (1,0,0), D (0,1,1) -> peak ≈ 1 (positive class)
  B, C, E, F, G, H_clean -> peak ≈ 0 (negative class)
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_CORE = os.path.join(PROJECT_ROOT, 'src_core')
sys.path.insert(0, SRC_CORE)

from model import SegmentationNetwork
from dataloader import (ensure_hwc, ensure_3ch, calculate_positions,
                        build_input_channels)

ROOT = os.path.join(PROJECT_ROOT, 'data/synthetic_channel_test_psf')
CENTERS = [(88, 244), (88, 488), (88, 732), (88, 366)]  # cx, cy
WINDOW = 8

CHECKPOINTS = {
    'baseline_v3': os.path.join(PROJECT_ROOT, 'checkpoints/baseline_v3/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
    'dd_minimal':  os.path.join(PROJECT_ROOT, 'checkpoints/dd_minimal/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
    'target_dd':   os.path.join(PROJECT_ROOT, 'checkpoints/target_dd/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    channels = ckpt.get('input_channels', ['target', 'ref1', 'ref2'])
    patch_size = (ckpt['img_height'], ckpt['img_width'])
    model = SegmentationNetwork(in_channels=len(channels), out_channels=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, patch_size, channels


def run_inference(image, model, patch_size, device, input_channels):
    h, w = image.shape[:2]
    ph, pw = patch_size
    score = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    for y in calculate_positions(h, ph):
        for x in calculate_positions(w, pw):
            patch = image[y:y+ph, x:x+pw]
            net_input = build_input_channels(
                patch[:, :, 0].astype(np.float32),
                patch[:, :, 1].astype(np.float32),
                patch[:, :, 2].astype(np.float32),
                input_channels)
            t = torch.from_numpy(net_input).unsqueeze(0).to(device)
            with torch.no_grad():
                out = F.softmax(model(t), dim=1)
                s = out[:, 1].squeeze().cpu().numpy()
            score[y:y+ph, x:x+pw] += s
            count[y:y+ph, x:x+pw] += 1
    return score / np.maximum(count, 1)


def scenario_metrics(model, patch_size, channels, device, scenario_dir):
    """Returns (peak_mean, global_max_mean) across tiffs in the folder.

    peak_mean is None for H_clean (no defect to anchor at).
    """
    paths = sorted(glob(os.path.join(scenario_dir, '*.tiff')))
    if not paths:
        return None, None
    peaks, gmaxs = [], []
    for p in paths:
        base_idx = int(os.path.basename(p).split('_')[0]) - 260
        cx, cy = CENTERS[base_idx % len(CENTERS)]
        raw = tifffile.imread(p)
        img = ensure_3ch(ensure_hwc(raw)).astype(np.float32)
        mn, mx = img.min(), img.max()
        if mn < 0 or mx > 255:
            img = (img - mn) / max(mx - mn, 1e-8) * 255.0
        heat = run_inference(img, model, patch_size, device, channels)
        gmaxs.append(heat.max())
        if 'H_clean' in scenario_dir:
            continue
        y0 = max(0, cy - WINDOW); y1 = min(heat.shape[0], cy + WINDOW + 1)
        x0 = max(0, cx - WINDOW); x1 = min(heat.shape[1], cx + WINDOW + 1)
        peaks.append(heat[y0:y1, x0:x1].max())
    return (np.mean(peaks) if peaks else None, np.mean(gmaxs))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scenarios = sorted([d for d in os.listdir(ROOT)
                        if os.path.isdir(os.path.join(ROOT, d))])

    rows = []
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"  SKIP {name}: not found at {path}")
            continue
        print(f"Loading {name} ({path})...")
        model, ps, ch = load_model(path, device)
        print(f"  input_channels = {ch}")
        row = {'name': name, 'channels': ch}
        for sc in scenarios:
            peak, gmax = scenario_metrics(model, ps, ch, device, os.path.join(ROOT, sc))
            row[sc] = (peak, gmax)
        rows.append(row)

    # Print table: rows = scenarios, columns = checkpoints; show peak (or gmax for H_clean)
    print(f"\n{'scenario':<24s}", end='')
    for r in rows:
        print(f" {r['name']:>14s}", end='')
    print()
    print('-' * (24 + 15 * len(rows)))
    for sc in scenarios:
        # mark expected positive vs negative
        expected = '+' if sc.startswith(('A_', 'D_')) else '-'
        print(f"{expected} {sc:<22s}", end='')
        for r in rows:
            peak, gmax = r[sc]
            val = gmax if peak is None else peak
            print(f" {val:>14.4f}", end='')
        print()
    print("\nLegend: + = expected positive (peak should be high)")
    print("        - = expected negative (peak should be low)")
    print("        H_clean column shows global_max (no defect center to anchor)")


if __name__ == '__main__':
    main()
