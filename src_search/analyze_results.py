"""Aggregate analysis of search trials.

Auto-detects every trial_NNN/ under --output_root and produces:
  1. Leaderboard sorted by best metric
  2. Per-defect coverage matrix (which trials matched which defect at best epoch)
  3. Defect difficulty buckets (always / partial / never matched)
  4. Best yaml per defect (recommend ensemble candidates)
  5. Ensemble suggestion (greedy K trials covering most unique defects)
"""

import argparse
import glob
import json
import os


REQUIRED_FILES = ('params.json', 'summary.json', 'epoch_log.jsonl')


def load_trial(trial_dir):
    if not all(os.path.exists(os.path.join(trial_dir, f)) for f in REQUIRED_FILES):
        return None

    with open(os.path.join(trial_dir, 'params.json')) as f:
        params = json.load(f)
    with open(os.path.join(trial_dir, 'summary.json')) as f:
        summary = json.load(f)
    with open(os.path.join(trial_dir, 'epoch_log.jsonl')) as f:
        records = [json.loads(line) for line in f if line.strip()]

    best_epoch = summary.get('best_epoch')
    best_record = next((r for r in records if r['epoch'] == best_epoch), None)
    if best_record is None and records:
        best_record = records[-1]

    return {
        'trial_id': params['trial_id'],
        'name': os.path.basename(trial_dir),
        'overrides': params['overrides'],
        'best_metric': summary.get('best_metric') if summary.get('best_metric') is not None else -1.0,
        'best_epoch': best_epoch,
        'main_metric': summary.get('main_metric', 'recall@50'),
        'best_record': best_record,
        'epochs_run': summary.get('epochs_run', len(records)),
    }


def discover_trials(output_root):
    paths = sorted(glob.glob(os.path.join(output_root, 'trial_*')))
    trials = []
    for p in paths:
        if not os.path.isdir(p):
            continue
        t = load_trial(p)
        if t is None:
            print(f'  [skip] incomplete trial: {os.path.basename(p)}')
            continue
        trials.append(t)
    return trials


def print_leaderboard(trials):
    main_metric = trials[0]['main_metric']
    sorted_trials = sorted(trials, key=lambda t: t['best_metric'], reverse=True)
    print(f'\n=== Leaderboard (by best {main_metric}) ===')
    print(f'{"trial":10}  {"best":>7}  {"epoch":>5}  {"epochs_run":>10}  {"overrides"}')
    for t in sorted_trials:
        print(f'{t["name"]:10}  {t["best_metric"]:>7.3f}  '
              f'{(t["best_epoch"] or 0) + 1:>5}  {t["epochs_run"]:>10}  '
              f'{t["overrides"]}')
    return sorted_trials


def print_coverage_matrix(sorted_trials):
    if not sorted_trials or sorted_trials[0]['best_record'] is None:
        print('\n[no per-defect data available]')
        return None

    defect_ids = sorted(sorted_trials[0]['best_record']['per_defect'].keys())
    cols = [f't{t["trial_id"]:03d}' for t in sorted_trials]

    print('\n=== Per-defect coverage at each trial\'s best epoch ===')
    header = f'{"defect":15}  ' + '  '.join(f'{c:>4}' for c in cols)
    print(header)
    print('-' * len(header))

    coverage_count = {}
    for d in defect_ids:
        row = []
        cnt = 0
        for t in sorted_trials:
            pd = t['best_record']['per_defect'].get(d, {})
            if pd.get('matched'):
                row.append(' ✓  ')
                cnt += 1
            else:
                row.append(' .  ')
        coverage_count[d] = cnt
        print(f'{d:15}  ' + '  '.join(row))
    return coverage_count


def print_difficulty_buckets(sorted_trials, coverage_count):
    if coverage_count is None:
        return
    n = len(sorted_trials)
    always = [d for d, c in coverage_count.items() if c == n]
    never = [d for d, c in coverage_count.items() if c == 0]
    partial = sorted([(d, c) for d, c in coverage_count.items() if 0 < c < n],
                      key=lambda x: x[1])

    print(f'\n=== Difficulty buckets (n_trials={n}) ===')
    print(f'  Always matched ({n}/{n}): {len(always)}')
    if always:
        print(f'    {", ".join(always)}')
    print(f'  Never matched (0/{n}): {len(never)}')
    if never:
        print(f'    {", ".join(never)}')
    print(f'  Partial: {len(partial)}')
    for d, c in partial:
        winners = [t['name'] for t in sorted_trials
                   if t['best_record']['per_defect'][d].get('matched')]
        print(f'    {d} ({c}/{n}): {", ".join(winners)}')


def print_best_yaml_per_defect(sorted_trials):
    if not sorted_trials or sorted_trials[0]['best_record'] is None:
        return
    defect_ids = sorted(sorted_trials[0]['best_record']['per_defect'].keys())
    print('\n=== Best yaml per defect (highest matched score) ===')
    for d in defect_ids:
        best = None
        for t in sorted_trials:
            pd = t['best_record']['per_defect'].get(d, {})
            if pd.get('matched'):
                if best is None or pd['score'] > best['score']:
                    best = {
                        'trial': t['name'],
                        'score': pd['score'],
                        'rank': pd['local_rank'],
                        'overrides': t['overrides'],
                    }
        if best:
            print(f'  {d}: {best["trial"]}  score={best["score"]:.3f}  '
                  f'rank={best["rank"]}  {best["overrides"]}')
        else:
            print(f'  {d}: not matched in any trial')


def greedy_ensemble(sorted_trials, k=3):
    """Greedy: pick K trials whose union covers the most unique defects."""
    if not sorted_trials or sorted_trials[0]['best_record'] is None:
        return

    defect_ids = list(sorted_trials[0]['best_record']['per_defect'].keys())
    matched_sets = {}
    for t in sorted_trials:
        s = set()
        for d in defect_ids:
            if t['best_record']['per_defect'].get(d, {}).get('matched'):
                s.add(d)
        matched_sets[t['name']] = s

    chosen = []
    covered = set()
    remaining = dict(matched_sets)
    k = min(k, len(sorted_trials))
    for step in range(k):
        best_name = None
        best_gain = -1
        for name, s in remaining.items():
            gain = len(s - covered)
            if gain > best_gain:
                best_gain = gain
                best_name = name
        if best_name is None or best_gain == 0:
            break
        chosen.append((best_name, best_gain))
        covered |= remaining[best_name]
        del remaining[best_name]

    print(f'\n=== Greedy ensemble suggestion (top {k}) ===')
    print(f'  Total defects covered: {len(covered)}/{len(defect_ids)}')
    for i, (name, gain) in enumerate(chosen, 1):
        params = next(t['overrides'] for t in sorted_trials if t['name'] == name)
        print(f'  {i}. {name}  (+{gain} new defects)  {params}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--ensemble_k', type=int, default=3,
                        help='Greedy ensemble size (default 3)')
    parser.add_argument('--no_per_defect', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.output_root):
        print(f'Not a directory: {args.output_root}')
        return

    print(f'Scanning {args.output_root} ...')
    trials = discover_trials(args.output_root)
    print(f'Found {len(trials)} complete trials.')
    if not trials:
        return

    sorted_trials = print_leaderboard(trials)
    if args.no_per_defect:
        return

    coverage_count = print_coverage_matrix(sorted_trials)
    print_difficulty_buckets(sorted_trials, coverage_count)
    print_best_yaml_per_defect(sorted_trials)
    greedy_ensemble(sorted_trials, k=args.ensemble_k)


if __name__ == '__main__':
    main()
