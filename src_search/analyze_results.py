"""Aggregate analysis of search trials.

Auto-detects every trial_NNN/ under --output_root and produces:
  1. Leaderboard sorted by best metric
  2. Per-defect rank matrix (each trial's global_rank for each defect)
  3. Difficulty buckets (always / partial / never matched, binary)
  4. Per-trial rank distribution (mutually exclusive buckets + mean rank)
  5. Best yaml per defect (rank-aware)
  6. Quality-weighted greedy ensemble (production-aware, see _rank_quality)
  7. Specialist defects (defects each trial is uniquely strong on)
"""

import argparse
import glob
import json
import os


REQUIRED_FILES = ('params.json', 'summary.json', 'epoch_log.jsonl')


def _match_info(pd):
    """Return matched detection info or None if no match.

    Reads epoch_log per_defect entry: {matched_idx, n_total, candidates}.
    """
    if pd is None or pd.get('matched_idx') is None:
        return None
    idx = pd['matched_idx']
    c = pd['candidates'][idx]
    return {
        'score': c['score'],
        'local_rank': idx,
        'global_rank': c['global_rank'],
        'n_total': pd['n_total'],
    }


def _rank_quality(rank):
    """Production-value tiers from global rank.

    rank<=30  -> 1.0   (visible in production review's top page)
    rank<=50  -> 0.6   (still visible, second-tier)
    rank<=150 -> 0.2   (matched but pushed deep — fragile)
    miss      -> 0.0
    """
    if rank is None:
        return 0.0
    if rank <= 30:
        return 1.0
    if rank <= 50:
        return 0.6
    if rank <= 150:
        return 0.2
    return 0.0


def _build_rank_table(sorted_trials):
    """Returns {defect_id: {trial_name: rank or None}}, sorted defect_ids."""
    if not sorted_trials or sorted_trials[0]['best_record'] is None:
        return None
    defect_ids = sorted(sorted_trials[0]['best_record']['per_defect'].keys())
    table = {}
    for d in defect_ids:
        row = {}
        for t in sorted_trials:
            info = _match_info(t['best_record']['per_defect'].get(d))
            row[t['name']] = info['global_rank'] if info else None
        table[d] = row
    return table


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


def discover_trials(output_roots, aliases):
    """Walk every output_root, attach an alias to each trial so the same
    trial_id from different roots stays distinguishable.

    When only one root is given, names stay short (`trial_001`).
    When multiple roots are given, names become `<alias>:t001`.
    """
    multi = len(output_roots) > 1
    trials = []
    for root, alias in zip(output_roots, aliases):
        paths = sorted(glob.glob(os.path.join(root, 'trial_*')))
        for p in paths:
            if not os.path.isdir(p):
                continue
            t = load_trial(p)
            if t is None:
                print(f'  [skip] incomplete trial: {alias}/{os.path.basename(p)}')
                continue
            br = t['best_record']
            if br is not None and br.get('per_defect'):
                sample_pd = next(iter(br['per_defect'].values()))
                if 'candidates' not in sample_pd:
                    print(f'  [skip] old schema (no candidates): '
                          f'{alias}/{os.path.basename(p)} — re-run search to enable analysis')
                    continue
            t['source'] = alias
            if multi:
                t['name'] = f'{alias}:t{t["trial_id"]:03d}'
            trials.append(t)
    return trials


def print_leaderboard(trials):
    main_metric = trials[0]['main_metric']
    sorted_trials = sorted(trials, key=lambda t: t['best_metric'], reverse=True)
    name_w = max(10, max(len(t['name']) for t in sorted_trials))
    print(f'\n=== Leaderboard (by best {main_metric}) ===')
    print(f'{"trial":{name_w}}  {"best":>7}  {"epoch":>5}  '
          f'{"epochs_run":>10}  {"overrides"}')
    for t in sorted_trials:
        print(f'{t["name"]:{name_w}}  {t["best_metric"]:>7.3f}  '
              f'{(t["best_epoch"] or 0) + 1:>5}  {t["epochs_run"]:>10}  '
              f'{t["overrides"]}')
    return sorted_trials


def print_rank_matrix(sorted_trials, rank_table):
    """Print per-defect rank at each trial's best epoch (× = miss)."""
    cols = [t['name'] for t in sorted_trials]
    col_w = max(5, max(len(c) for c in cols))

    print('\n=== Per-defect global_rank at each trial\'s best epoch ===')
    print('  numbers are 1-based cross-image rank; × = miss')
    header = f'{"defect":15}  ' + '  '.join(f'{c:>{col_w}}' for c in cols)
    print(header)
    print('-' * len(header))
    miss_cell = '×'.center(col_w)
    for d in sorted(rank_table.keys()):
        cells = []
        for t in sorted_trials:
            r = rank_table[d][t['name']]
            cells.append(miss_cell if r is None else f'{r:>{col_w}}')
        print(f'{d:15}  ' + '  '.join(cells))


def print_difficulty_buckets(sorted_trials, rank_table):
    n = len(sorted_trials)
    coverage_count = {
        d: sum(1 for t in sorted_trials if rank_table[d][t['name']] is not None)
        for d in rank_table
    }
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
                   if rank_table[d][t['name']] is not None]
        print(f'    {d} ({c}/{n}): {", ".join(winners)}')


def print_rank_distribution(sorted_trials, rank_table):
    """Mutually exclusive buckets per trial: 1-10, 11-30, 31-50, 51-150, miss."""
    buckets = [
        ('1-10',   lambda r: r is not None and r <= 10),
        ('11-30',  lambda r: r is not None and 10 < r <= 30),
        ('31-50',  lambda r: r is not None and 30 < r <= 50),
        ('51-150', lambda r: r is not None and 50 < r <= 150),
        ('miss',   lambda r: r is None),
    ]

    name_w = max(12, max(len(t['name']) for t in sorted_trials))
    print('\n=== Rank distribution per trial (mutually exclusive) ===')
    header = (f'{"trial":{name_w}}  '
              + '  '.join(f'{name:>6}' for name, _ in buckets)
              + '  ' + 'mean_rank')
    print(header)
    print('-' * len(header))

    defect_ids = list(rank_table.keys())
    for t in sorted_trials:
        ranks = [rank_table[d][t['name']] for d in defect_ids]
        counts = [sum(1 for r in ranks if pred(r)) for _, pred in buckets]
        matched = [r for r in ranks if r is not None]
        mean = sum(matched) / len(matched) if matched else float('nan')
        cells = '  '.join(f'{c:>6}' for c in counts)
        mean_str = f'{mean:>8.1f}' if matched else f'{"-":>8}'
        print(f'{t["name"]:{name_w}}  {cells}  {mean_str}')


def print_single_yaml_breakdown(sorted_trials, rank_table):
    """Per-defect tier breakdown for leaderboard #1 trial alone.

    Answers: 'if I deploy ONE yaml in production, what does the per-defect
    review experience look like?' — distinct from the cross-trial oracle
    view in print_best_yaml_per_defect.
    """
    best_trial = sorted_trials[0]
    name = best_trial['name']
    metric = best_trial['main_metric']
    defect_ids = list(rank_table.keys())

    print(f'\n=== Single best yaml: {name}  '
          f'(leaderboard #1, {metric}={best_trial["best_metric"]:.3f}) ===')
    print(f'  {best_trial["overrides"]}')

    groups = [
        ('Strong  (rank<=30) ', lambda r: r is not None and r <= 30),
        ('Mid     (31-50)    ', lambda r: r is not None and 30 < r <= 50),
        ('Weak    (51-150)   ', lambda r: r is not None and 50 < r <= 150),
        ('Missed  (no match) ', lambda r: r is None),
    ]
    for label, pred in groups:
        members = sorted(
            [(d, rank_table[d][name]) for d in defect_ids
             if pred(rank_table[d][name])],
            key=lambda x: x[1] if x[1] is not None else 9999)
        print(f'    {label}: {len(members)}')
        if not members:
            continue
        items = [(f'{d}(r{r})' if r is not None else f'{d}(miss)')
                 for d, r in members]
        for i in range(0, len(items), 4):
            print(f'      {", ".join(items[i:i+4])}')


def print_best_yaml_per_defect(sorted_trials):
    """Cross-trial oracle view: for each defect, which trial gives the lowest
    global_rank (i.e. surfaces it earliest in the cross-image candidate list).

    This is the upper-bound 'if I knew which yaml to use per defect' view —
    pair it with print_single_yaml_breakdown for the realistic single-deploy
    counterpart.
    """
    if not sorted_trials or sorted_trials[0]['best_record'] is None:
        return
    defect_ids = sorted(sorted_trials[0]['best_record']['per_defect'].keys())
    print('\n=== Best yaml per defect (cross-trial oracle, by lowest rank) ===')
    for d in defect_ids:
        best = None
        for t in sorted_trials:
            info = _match_info(t['best_record']['per_defect'].get(d))
            if info is None:
                continue
            if best is None or info['global_rank'] < best['global_rank']:
                best = {**info, 'trial': t['name'], 'overrides': t['overrides']}
        if best:
            print(f'  {d}: {best["trial"]}  '
                  f'rank={best["global_rank"]:>3d}/{best["n_total"]}  '
                  f'(local={best["local_rank"]}, score={best["score"]:.3f})  '
                  f'{best["overrides"]}')
        else:
            print(f'  {d}: not matched in any trial')


def quality_weighted_greedy(sorted_trials, rank_table, k=3):
    """Greedy ensemble that maximises sum(_rank_quality) over defects.

    Each step picks the trial whose addition improves total quality the most.
    Per defect, the ensemble's quality is max(quality across chosen trials),
    so picking a trial that duplicates already-strong defects gives 0 gain.
    """
    defect_ids = list(rank_table.keys())
    n = len(defect_ids)

    print('\n=== Quality-weighted greedy ensemble ===')
    print('  Quality: rank<=30 -> 1.0, rank<=50 -> 0.6, rank<=150 -> 0.2, miss -> 0.0')
    print(f'  Max possible total: {float(n):.1f}')

    cur_q = {d: 0.0 for d in defect_ids}
    cur_rank = {d: None for d in defect_ids}
    cur_trial = {d: None for d in defect_ids}
    remaining = list(sorted_trials)
    k = min(k, len(sorted_trials))
    n_picked = 0
    chosen_trials = []

    for step in range(k):
        best_t = None
        best_gain = 0.0
        best_new_q = None
        for t in remaining:
            new_q = {d: max(cur_q[d], _rank_quality(rank_table[d][t['name']]))
                     for d in defect_ids}
            gain = sum(new_q.values()) - sum(cur_q.values())
            if gain > best_gain:
                best_gain = gain
                best_t = t
                best_new_q = new_q
        if best_t is None:
            print(f'\n  No further trial improves ensemble quality. '
                  f'Stopped at {n_picked} pick(s).')
            break

        n_picked += 1
        chosen_trials.append(best_t)
        if step == 0:
            ranks = [rank_table[d][best_t['name']] for d in defect_ids]
            b1 = sum(1 for r in ranks if r is not None and r <= 30)
            b2 = sum(1 for r in ranks if r is not None and 30 < r <= 50)
            b3 = sum(1 for r in ranks if r is not None and 50 < r <= 150)
            b4 = sum(1 for r in ranks if r is None)
            print(f'\n  {step+1}. [BASE]  {best_t["name"]}  '
                  f'quality={sum(best_new_q.values()):.1f}/{float(n):.1f}')
            print(f'        {best_t["overrides"]}')
            print(f'        {b1} top-30, {b2} mid (31-50), '
                  f'{b3} weak (51-150), {b4} miss')
        else:
            improvements = []
            for d in defect_ids:
                new_q_d = _rank_quality(rank_table[d][best_t['name']])
                if new_q_d > cur_q[d]:
                    improvements.append((
                        d, cur_rank[d],
                        rank_table[d][best_t['name']],
                        new_q_d - cur_q[d]))
            improvements.sort(key=lambda x: -x[3])

            print(f'\n  {step+1}. [+{best_gain:.1f}]  {best_t["name"]}  '
                  f'quality={sum(best_new_q.values()):.1f}/{float(n):.1f}')
            print(f'        {best_t["overrides"]}')
            print(f'        Unique tier improvements over current ensemble:')
            if improvements:
                for d, old, new, gain in improvements:
                    old_str = 'miss' if old is None else f'rank{old:>4}'
                    print(f'          {d}: {old_str} -> rank{new:>4}  (+{gain:.1f})')
            else:
                print(f'          (none — every defect already covered '
                      f'at >= this trial\'s tier)')

        # commit pick — track tier improvement and (within same tier) rank
        # improvement so the final breakdown shows the strongest source per defect.
        for d in defect_ids:
            t_q = _rank_quality(rank_table[d][best_t['name']])
            t_r = rank_table[d][best_t['name']]
            if t_q > cur_q[d]:
                cur_q[d] = t_q
                cur_rank[d] = t_r
                cur_trial[d] = best_t['name']
            elif t_q == cur_q[d] and t_q > 0:
                if cur_rank[d] is None or (t_r is not None and t_r < cur_rank[d]):
                    cur_rank[d] = t_r
                    cur_trial[d] = best_t['name']
        remaining.remove(best_t)

    print(f'\n  Final ensemble quality: {sum(cur_q.values()):.1f}/{float(n):.1f}  '
          f'({n_picked} trial(s) chosen)')

    print('\n  Final coverage breakdown by tier:')
    groups = [
        ('Strong  (rank<=30)',  lambda q: q == 1.0),
        ('Mid     (31-50)',     lambda q: q == 0.6),
        ('Weak    (51-150)',    lambda q: q == 0.2),
        ('Missed  (no match)',  lambda q: q == 0.0),
    ]
    for label, pred in groups:
        members = sorted(
            [(d, cur_rank[d], cur_trial[d])
             for d in defect_ids if pred(cur_q[d])],
            key=lambda x: (x[1] if x[1] is not None else 9999))
        print(f'    {label}: {len(members)}')
        if not members:
            continue
        items = []
        for d, r, t_name in members:
            t_short = t_name.replace('trial_', 't') if t_name else '-'
            r_str = 'miss' if r is None else f'r{r}'
            items.append(f'{d}({t_short}:{r_str})')
        for i in range(0, len(items), 4):
            print(f'      {", ".join(items[i:i+4])}')

    return chosen_trials


def ensemble_strategy_comparison(chosen_trials, match_radius=3.0, rrf_k=60,
                                  k_list=(30, 50, 150)):
    """Three-way recall@K comparison on the greedy-chosen trial subset.

    Strategies:
      - rank-oracle (upper bound): per defect, take min global_rank across
        chosen trials. Recall@K = #(min_rank <= K) / n_GTs.
      - score-pool: pool every detection from chosen trials, sort by raw
        softmax score, walk top-down matching defects (each GT claimed once).
      - RRF: pool every detection, sort by sum of 1 / (rrf_k + rank_in_own_trial).
        Production-deployable rank-based fusion.

    Plus single-trial baselines (from each chosen trial's own metrics).
    """
    if not chosen_trials:
        return
    if len(chosen_trials) < 2:
        print('\n[ensemble comparison: only 1 trial chosen, skipping fusion]')
        return

    events = []
    for t in chosen_trials:
        per_defect = t['best_record']['per_defect']
        for defect_id, pd in per_defect.items():
            for c in pd['candidates']:
                events.append({
                    'score': c['score'],
                    'rrf': 1.0 / (rrf_k + c['global_rank']),
                    'dist': c['dist_to_gt'],
                    'defect_id': defect_id,
                })
    n_gts = len({e['defect_id'] for e in events})

    def recall_from_events(events_sorted, dedupe):
        """
        dedupe=False: pos increments for every event (review pipeline sees
                      duplicate detections of the same defect; they waste slots).
        dedupe=True:  only the first event per defect_id consumes a pos
                      (review pipeline auto-merges multi-trial detections of
                      the same defect; idealised, matches IR-standard RRF).
        """
        matched_at = {}
        seen = set()
        pos = 0
        for ev in events_sorted:
            d = ev['defect_id']
            if dedupe and d in seen:
                continue
            seen.add(d)
            pos += 1
            if d in matched_at:
                continue
            if ev['dist'] <= match_radius:
                matched_at[d] = pos
        out = {f'r@{k}': sum(1 for p in matched_at.values() if p <= k) / n_gts
               for k in k_list}
        out['total'] = len(matched_at) / n_gts
        return out

    score_sorted = sorted(events, key=lambda e: -e['score'])
    rrf_sorted = sorted(events, key=lambda e: -e['rrf'])
    score_pool_nd = recall_from_events(score_sorted, dedupe=False)
    score_pool_d = recall_from_events(score_sorted, dedupe=True)
    rrf_pool_nd = recall_from_events(rrf_sorted, dedupe=False)
    rrf_pool_d = recall_from_events(rrf_sorted, dedupe=True)

    # Upper bound: per defect, min rank across chosen trials
    defect_min_rank = {}
    for t in chosen_trials:
        for d, pd in t['best_record']['per_defect'].items():
            mi = pd.get('matched_idx')
            if mi is None:
                continue
            r = pd['candidates'][mi]['global_rank']
            if r < defect_min_rank.get(d, 10 ** 9):
                defect_min_rank[d] = r
    upper = {f'r@{k}': sum(1 for r in defect_min_rank.values() if r <= k) / n_gts
             for k in k_list}
    upper['total'] = len(defect_min_rank) / n_gts

    chosen_str = ' + '.join(t['name'] for t in chosen_trials)
    print(f'\n=== Ensemble strategy comparison (recall@K) ===')
    print(f'  pool: {chosen_str}  ({len(events)} detections from '
          f'{len(chosen_trials)} trials)')
    print(f'  match_radius={match_radius}px, n_GTs={n_gts}, RRF k={rrf_k}')
    print()

    cols = [f'r@{k}' for k in k_list] + ['total']
    label_w = max(30, max(len(t['name']) + len(' (single)') for t in chosen_trials))
    header = f'  {"strategy":{label_w}}' + '  '.join(f'{c:>7}' for c in cols)
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for t in chosen_trials:
        m = t['best_record']['metrics']
        row = [m.get(f'recall@{k}', float('nan')) for k in k_list]
        row.append(m.get('total_recall', float('nan')))
        cells = '  '.join(f'{v:>7.3f}' for v in row)
        print(f'  {(t["name"] + " (single)"):{label_w}}{cells}')

    print('  ' + '-' * (len(header) - 2))
    print(f'  {"upper-bound (rank oracle)":{label_w}}'
          + '  '.join(f'{upper[c]:>7.3f}' for c in cols))
    print('  ' + '-' * (len(header) - 2))
    print('  no-dedup (review sees every duplicate detection)')
    for label, m in [('  score-pool', score_pool_nd),
                     ('  RRF fusion', rrf_pool_nd)]:
        cells = '  '.join(f'{m[c]:>7.3f}' for c in cols)
        print(f'  {label:{label_w}}{cells}')
    print('  dedup    (review auto-merges same-defect detections)')
    for label, m in [('  score-pool', score_pool_d),
                     ('  RRF fusion', rrf_pool_d)]:
        cells = '  '.join(f'{m[c]:>7.3f}' for c in cols)
        print(f'  {label:{label_w}}{cells}')


def print_dominance(sorted_trials, rank_table):
    """For each defect, the trial with the lowest rank wins. Count wins per trial.

    Tells you which trial is the strongest source for the most defects (in
    pure ranking terms — separate from quality-weighted ensemble).
    """
    defect_ids = list(rank_table.keys())
    counts = {t['name']: 0 for t in sorted_trials}
    no_match = 0
    for d in defect_ids:
        ranked = [(t['name'], rank_table[d][t['name']]) for t in sorted_trials]
        ranked = [(n, r) for n, r in ranked if r is not None]
        if not ranked:
            no_match += 1
            continue
        winner = min(ranked, key=lambda x: x[1])
        counts[winner[0]] += 1

    n = len(defect_ids)
    print('\n=== Trial dominance (best source per defect) ===')
    for t in sorted_trials:
        c = counts[t['name']]
        print(f'  {t["name"]}: dominant for {c}/{n} defects  {t["overrides"]}')
    if no_match:
        print(f'  no trial caught: {no_match}/{n}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, required=True, nargs='+',
                        help='One or more search output directories. With '
                             'multiple roots, trial names are prefixed with '
                             'each root\'s alias to stay distinct.')
    parser.add_argument('--aliases', type=str, nargs='+', default=None,
                        help='Optional short labels (one per --output_root, '
                             'same order). Defaults to each path basename.')
    parser.add_argument('--ensemble_k', type=int, default=3,
                        help='Greedy ensemble size (default 3)')
    parser.add_argument('--match_radius', type=float, default=3.0,
                        help='Pixel radius for detection<->GT match in '
                             'ensemble strategy comparison (default 3.0).')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='Reciprocal Rank Fusion constant (default 60).')
    parser.add_argument('--no_per_defect', action='store_true')
    args = parser.parse_args()

    roots = args.output_root
    for r in roots:
        if not os.path.isdir(r):
            print(f'Not a directory: {r}')
            return

    if args.aliases:
        if len(args.aliases) != len(roots):
            print(f'--aliases count ({len(args.aliases)}) must match '
                  f'--output_root count ({len(roots)}).')
            return
        aliases = args.aliases
    else:
        aliases = [os.path.basename(os.path.normpath(r)) for r in roots]

    print(f'Scanning {len(roots)} root(s):')
    for r, a in zip(roots, aliases):
        print(f'  [{a}] {r}')
    trials = discover_trials(roots, aliases)
    print(f'Found {len(trials)} complete trials.')
    if not trials:
        return

    sorted_trials = print_leaderboard(trials)
    if args.no_per_defect:
        return

    rank_table = _build_rank_table(sorted_trials)
    if rank_table is None:
        print('\n[no per-defect data available]')
        return

    print_rank_matrix(sorted_trials, rank_table)
    print_difficulty_buckets(sorted_trials, rank_table)
    print_rank_distribution(sorted_trials, rank_table)
    print_single_yaml_breakdown(sorted_trials, rank_table)
    print_best_yaml_per_defect(sorted_trials)
    chosen = quality_weighted_greedy(
        sorted_trials, rank_table, k=args.ensemble_k)
    print_dominance(sorted_trials, rank_table)
    ensemble_strategy_comparison(chosen, match_radius=args.match_radius,
                                 rrf_k=args.rrf_k)


if __name__ == '__main__':
    main()
