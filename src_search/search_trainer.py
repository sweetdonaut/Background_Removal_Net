"""Trainer with epoch-end real-PSF evaluation and early stopping.

Standalone — does not import src_core/trainer.py. CLI mostly mirrors train.sh
plus three new flags:
    --real_valid_dir       location of the 30-PSF tiffs (default data/30ea_testing/bad)
    --main_metric          metric to monitor (default recall@30)
    --early_stop_patience  epochs of no improvement before stopping (0 = disabled)
    --eval_every           evaluate every N epochs (default 1)
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src_core'))

import json  # noqa: E402

from dataloader import Dataset  # noqa: E402
from evaluator import evaluate_real, per_defect_summary  # noqa: E402
from loss import FocalLoss  # noqa: E402
from model import SegmentationNetwork  # noqa: E402


def weights_init(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in name:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def cosine_focal_gamma(epoch, total_epochs, gamma_start, gamma_end):
    progress = epoch / max(1, total_epochs)
    return gamma_start + (gamma_end - gamma_start) * (1 - math.cos(progress * math.pi)) / 2


def set_seeds(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloader(args, patch_size):
    psf_config_paths = None
    if args.defect_mode == 'psf':
        if args.psf_yaml_path:
            psf_config_paths = list(args.psf_yaml_path)
        else:
            defects_dir = os.path.join(os.path.dirname(HERE), 'src_core', 'defects')
            psf_config_paths = [os.path.join(defects_dir, f'{t}.yaml') for t in args.psf_type]

    dataset = Dataset(
        training_path=args.training_dataset_path,
        patch_size=patch_size,
        num_defects_range=args.num_defects_range,
        img_format=args.img_format,
        cache_size=args.cache_size,
        defect_mode=args.defect_mode,
        psf_config_paths=psf_config_paths,
        psf_pool_size=args.psf_pool_size,
        partial_leak_scale=tuple(args.partial_leak_scale),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    return dataset, dataloader


def save_best_checkpoint(model, args, patch_size, epoch, metric_value, all_metrics, run_name):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'img_height': patch_size[0],
        'img_width': patch_size[1],
        'epoch': epoch,
        'seed': args.seed,
        'metric_name': args.main_metric,
        'metric_value': metric_value,
        'all_metrics': all_metrics,
    }
    out = os.path.join(args.checkpoint_path, f'{run_name}_best.pth')
    torch.save(ckpt, out)
    return out


def train(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    set_seeds(args.seed)

    device = (torch.device(f'cuda:{args.gpu_id}')
              if torch.cuda.is_available() and args.gpu_id >= 0
              else torch.device('cpu'))
    print(f'Device: {device}')

    patch_size = (args.patch_size, args.patch_size)
    run_name = f'BgRemoval_search_lr{args.lr}_ep{args.epochs}_bs{args.bs}_{patch_size[0]}'

    model = SegmentationNetwork(in_channels=3, out_channels=2).to(device)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2)
    criterion = FocalLoss(alpha=0.75, gamma=args.gamma_start)
    print(f'FocalLoss: alpha=0.75, gamma {args.gamma_start} -> {args.gamma_end} (cosine)')

    dataset, dataloader = build_dataloader(args, patch_size)
    print(f'Dataset size: {len(dataset)} samples per epoch')
    print(f'Real valid dir: {args.real_valid_dir}')
    print(f'Monitor metric: {args.main_metric}  (match_radius={args.match_radius}px)')

    best_metric = -float('inf')
    best_epoch = -1
    epochs_since_best = 0
    history = []

    num_batches = len(dataloader)

    for epoch in range(args.epochs):
        model.train()
        gamma = cosine_focal_gamma(epoch, args.epochs, args.gamma_start, args.gamma_end)
        criterion.update_params(gamma=gamma)

        epoch_loss = 0.0
        for i_batch, batch in enumerate(dataloader):
            x = batch['three_channel_input'].to(device)
            y = batch['target_mask'].to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i_batch % 10 == 0 or i_batch == num_batches - 1:
                pct = (i_batch + 1) / num_batches * 100
                print(f'\rEpoch [{epoch + 1}/{args.epochs}] '
                      f'Batch [{i_batch + 1}/{num_batches}] ({pct:.1f}%) '
                      f'Loss: {loss.item():.4e}', end='', flush=True)

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)

        do_eval = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if do_eval:
            metrics = evaluate_real(
                model, args.real_valid_dir, patch_size, device,
                match_radius=args.match_radius,
                dead_pixel_csv=args.dead_pixel_csv,
                dead_pixel_half_size=args.dead_pixel_half_size,
            )
            scalar_metrics = {k: v for k, v in metrics.items() if k != 'per_image'}
            per_defect = per_defect_summary(metrics['per_image'])
            history.append({'epoch': epoch, 'loss': avg_loss, **scalar_metrics})

            log_record = {
                'epoch': epoch,
                'loss': avg_loss,
                'gamma': gamma,
                'metrics': scalar_metrics,
                'per_defect': per_defect,
            }
            with open(os.path.join(args.checkpoint_path, 'epoch_log.jsonl'), 'a') as f:
                f.write(json.dumps(log_record) + '\n')

            print(f'\nEpoch [{epoch + 1}/{args.epochs}] '
                  f'loss={avg_loss:.4e} γ={gamma:.2f} '
                  f'r@30={metrics["recall@30"]:.3f} '
                  f'r@50={metrics["recall@50"]:.3f} '
                  f'r@150={metrics["recall@150"]:.3f} '
                  f'total={metrics["total_recall"]:.3f}')

            current = metrics[args.main_metric]
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                epochs_since_best = 0
                save_best_checkpoint(
                    model, args, patch_size, epoch,
                    current, scalar_metrics, run_name)
                print(f'  -> new best {args.main_metric}={current:.4f} (saved)')
            else:
                epochs_since_best += 1
                print(f'  best {args.main_metric}={best_metric:.4f} '
                      f'@ epoch {best_epoch + 1}; no improvement for {epochs_since_best} ep')

            if args.early_stop_patience > 0 and epochs_since_best >= args.early_stop_patience:
                print(f'\nEarly stop: {epochs_since_best} epochs without improvement.')
                break
        else:
            print(f'\nEpoch [{epoch + 1}/{args.epochs}] loss={avg_loss:.4e}')

    summary = {
        'best_metric': float(best_metric) if best_metric != -float('inf') else None,
        'best_epoch': best_epoch,
        'main_metric': args.main_metric,
        'epochs_run': len(history),
        'history': history,
    }
    with open(os.path.join(args.checkpoint_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone. Best {args.main_metric}={best_metric:.4f} at epoch {best_epoch + 1}.')
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_defects_range', type=int, nargs=2, default=[3, 8])
    parser.add_argument('--training_dataset_path', type=str, required=True)
    parser.add_argument('--img_format', type=str, choices=['png_jpg', 'tiff'], default='tiff')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cache_size', type=int, default=0)
    parser.add_argument('--gamma_start', type=float, default=1.0)
    parser.add_argument('--gamma_end', type=float, default=3.0)
    parser.add_argument('--defect_mode', type=str, choices=['gaussian', 'psf'], default='psf')
    parser.add_argument('--psf_type', type=str, nargs='+', default=['type4_vector'])
    parser.add_argument('--psf_yaml_path', type=str, nargs='+', default=None,
                        help='Direct yaml file paths (overrides --psf_type). '
                             'Used by run_trial.py to load per-trial yamls.')
    parser.add_argument('--psf_pool_size', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--partial_leak_scale', type=float, nargs=2, default=[0.2, 0.7])
    parser.add_argument('--real_valid_dir', type=str, default='data/30ea_testing/bad')
    parser.add_argument('--main_metric', type=str, default='recall@50')
    parser.add_argument('--match_radius', type=float, default=3.0,
                        help='Pixel distance for detection<->GT match (default 3.0). '
                             'Loosen for noisier production peaks.')
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--dead_pixel_csv', type=str, default=None,
                        help='CSV with (dead_x, dead_y) columns. Heatmap is masked '
                             'around each pixel before detection. Defaults to '
                             '<real_valid_dir>/dead_pixels.csv if present.')
    parser.add_argument('--dead_pixel_half_size', type=int, default=5,
                        help='Half-size of square mask around each dead pixel '
                             '(default 5 -> 10x10 bbox).')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    train(args)
