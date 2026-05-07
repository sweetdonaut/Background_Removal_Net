"""Single-trial entry point for slurm-array (or local sequential) yaml search.

Each call:
  1. Sample yaml params (deterministic given --seed)
  2. Write trial_yaml.yaml + params.json into trial folder
  3. Run search_trainer with that yaml
  4. summary.json + epoch_log.jsonl land in the same folder
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src_core'))

from generate_psf import load_config as load_psf_config  # noqa: E402
from search_space import sample_params  # noqa: E402
from search_trainer import build_parser, train  # noqa: E402

PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_BASE_YAML = os.path.join(PROJECT_ROOT, 'src_core', 'defects', 'type4_vector.yaml')


def write_trial_yaml(base_path, overrides, out_path):
    cfg = load_psf_config(base_path)
    plain = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    plain.update(overrides)
    with open(out_path, 'w') as f:
        yaml.dump(plain, f, sort_keys=False, default_flow_style=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_id', type=int, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--base_yaml', type=str, default=DEFAULT_BASE_YAML)
    parser.add_argument('--seed', type=int, required=True)

    parser.add_argument('--training_dataset_path', type=str,
                        default='data/grid_stripe_4channel/train/good/')
    parser.add_argument('--real_valid_dir', type=str,
                        default='data/30ea_testing/bad')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--main_metric', type=str, default='recall@50')
    parser.add_argument('--match_radius', type=float, default=3.0)
    parser.add_argument('--dead_pixel_csv', type=str, default=None)
    parser.add_argument('--dead_pixel_half_size', type=int, default=5)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_defects_range', type=int, nargs=2, default=[3, 8])
    parser.add_argument('--psf_pool_size', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--partial_leak_scale', type=float, nargs=2, default=[0.2, 0.7])
    parser.add_argument('--gamma_start', type=float, default=1.0)
    parser.add_argument('--gamma_end', type=float, default=3.0)
    args = parser.parse_args()

    trial_dir = os.path.join(args.output_root, f'trial_{args.trial_id:03d}')
    os.makedirs(trial_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    overrides = sample_params(rng)

    trial_yaml_path = os.path.join(trial_dir, 'trial_yaml.yaml')
    write_trial_yaml(args.base_yaml, overrides, trial_yaml_path)

    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump({
            'trial_id': args.trial_id,
            'seed': args.seed,
            'base_yaml': args.base_yaml,
            'overrides': overrides,
        }, f, indent=2)

    print(f'\n=== Trial {args.trial_id} (seed={args.seed}) ===')
    print(f'overrides: {overrides}')
    print(f'trial_dir: {trial_dir}\n')

    trainer_argv = [
        '--bs', str(args.bs),
        '--lr', str(args.lr),
        '--epochs', str(args.epochs),
        '--gpu_id', str(args.gpu_id),
        '--checkpoint_path', trial_dir,
        '--patch_size', str(args.patch_size),
        '--training_dataset_path', args.training_dataset_path,
        '--img_format', 'tiff',
        '--num_defects_range', str(args.num_defects_range[0]),
        str(args.num_defects_range[1]),
        '--cache_size', '0',
        '--gamma_start', str(args.gamma_start),
        '--gamma_end', str(args.gamma_end),
        '--defect_mode', 'psf',
        '--psf_yaml_path', trial_yaml_path,
        '--psf_pool_size', str(args.psf_pool_size),
        '--num_workers', str(args.num_workers),
        '--partial_leak_scale', str(args.partial_leak_scale[0]),
        str(args.partial_leak_scale[1]),
        '--real_valid_dir', args.real_valid_dir,
        '--main_metric', args.main_metric,
        '--match_radius', str(args.match_radius),
        '--early_stop_patience', str(args.early_stop_patience),
        '--seed', str(args.seed),
        '--dead_pixel_half_size', str(args.dead_pixel_half_size),
    ]
    if args.dead_pixel_csv:
        trainer_argv += ['--dead_pixel_csv', args.dead_pixel_csv]
    trainer_args = build_parser().parse_args(trainer_argv)
    train(trainer_args)


if __name__ == '__main__':
    main()
