"""Run evaluator on a checkpoint over data/30ea_testing/bad/.

Standalone CLI: python src_search/run_eval.py --checkpoint <path>
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src_core'))

from evaluator import evaluate_real  # noqa: E402
from model import SegmentationNetwork  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_dir', default='data/30ea_testing/bad')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--match_radius', type=float, default=3.0)
    parser.add_argument('--score_window', type=int, default=2,
                        help='Half-window around each peak for top-K score '
                             'averaging. 2 -> 5x5 (default).')
    parser.add_argument('--dead_pixel_csv', type=str, default=None,
                        help='Defaults to <test_dir>/dead_pixels.csv if present.')
    parser.add_argument('--dead_pixel_half_size', type=int, default=5)
    parser.add_argument('--extra_test_dirs', type=str, nargs='*', default=None,
                        help='Optional GT-less folders that contribute as FP '
                             'noise in the global ranking.')
    parser.add_argument('--extra_sample_ratios', type=float, nargs='*', default=None,
                        help='Sampling ratio (0..1) per extra dir.')
    parser.add_argument('--extra_sample_seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    patch_size = (ckpt['img_height'], ckpt['img_width'])
    print(f'Checkpoint: {args.checkpoint}')
    print(f'  patch_size: {patch_size}')
    print(f'  epoch: {ckpt.get("epoch", "?")}')

    model = SegmentationNetwork(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics = evaluate_real(
        model, args.test_dir, patch_size, device,
        score_window=args.score_window,
        match_radius=args.match_radius,
        dead_pixel_csv=args.dead_pixel_csv,
        dead_pixel_half_size=args.dead_pixel_half_size,
        extra_test_dirs=args.extra_test_dirs,
        extra_sample_ratios=args.extra_sample_ratios,
        extra_sample_seed=args.extra_sample_seed,
        verbose=args.verbose,
    )

    # r@500 only carries new info when extras inflate the detection pool
    # past 500; otherwise it equals total_recall (fallback).
    has_extras = metrics.get('n_extra_images', 0) > 0
    print('\n=== Metrics ===')
    keys = ['recall@30', 'recall@50', 'recall@150']
    if has_extras:
        keys.append('recall@500')
    keys += ['total_recall', 'n_total_detections', 'n_gts']
    if has_extras:
        keys.append('n_extra_images')
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f'  {k:22s} : {v:.4f}')
            else:
                print(f'  {k:22s} : {v}')

    n_miss = sum(1 for p in metrics['per_image']
                 if p.get('gt') is not None and p['gt_local_rank'] is None)
    print(f'  {"images_fully_missed":22s} : {n_miss}/{metrics["n_gts"]}')


if __name__ == '__main__':
    main()
