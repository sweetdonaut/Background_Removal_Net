"""Fit vector PSF yaml parameters to a folder of real defect tiffs.

Given N tiffs named DID***#x,y.tiff (3-channel, defect at (x,y) on target),
finds the minimum number of PSF types (k = 1, 2, ..., MAX_K) needed to
explain every defect with NCC >= NCC_THRESHOLD against its assigned type.

Outputs k fitted yamls, a comparison PNG, an assignments CSV, and a JSON
log of the search history.

Usage:
    python utils/fit_psf_params.py \\
        --input_dir data/synthetic_defects_v1 \\
        --output_dir output/psf_fit \\
        --ground_truth data/synthetic_defects_v1/ground_truth.json
"""

import argparse
import csv
import json
import os
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_core'))
from generate_psf import create_psf_defect  # noqa: E402


# ============================================================================
# CONFIGURATION — edit to control which params are fitted vs fixed.
# ============================================================================

# Aberrations are off by default — turn on if your system actually has them.
# Leaving them off prevents the search from compensating for shape mismatch
# with fictitious aberrations.
FIT_TOGGLE = {
    'na': True,
    'pol_type': True,
    'outer_r': True,
    'epsilon': True,
    'defocus': False,
    'astig_x': False,
    'astig_y': False,
    'coma_x': False,
    'coma_y': False,
    'spherical': False,
    'trefoil_x': False,
    'trefoil_y': False,
}

SEARCH_SPACE = {
    'na': (0.5, 1.2),
    'pol_type': ['linX', 'linY', 'lin45', 'circR', 'circL', 'radial'],
    'outer_r': (20, 50),
    'epsilon': (0.0, 0.9),
    'defocus': (-2.0, 2.0),
    'astig_x': (-1.0, 1.0),
    'astig_y': (-1.0, 1.0),
    'coma_x': (-1.0, 1.0),
    'coma_y': (-1.0, 1.0),
    'spherical': (-1.0, 1.0),
    'trefoil_x': (-1.0, 1.0),
    'trefoil_y': (-1.0, 1.0),
}

FIXED_VALUES = {
    'na': 0.95,
    'pol_type': 'linX',
    'outer_r': 30,
    'epsilon': 0.5,
    'defocus': 0.0,
    'astig_x': 0.0,
    'astig_y': 0.0,
    'coma_x': 0.0,
    'coma_y': 0.0,
    'spherical': 0.0,
    'trefoil_x': 0.0,
    'trefoil_y': 0.0,
}

N_ITER_PER_POL = 80     # random search iterations per pol_type per cluster
K_PSFS_PER_EVAL = 20    # PSFs per param evaluation (averages over noise)
NCC_THRESHOLD = 0.85    # acceptance threshold for fit quality
MAX_K = 3               # maximum clusters to try
MIN_CLUSTER_SIZE = 5    # reject k if any cluster has fewer defects (over-clustering)
IMPROVEMENT_MARGIN = 0.02  # k+1 must beat k by at least this much to be preferred
ROI_SIZE = 20           # ROI window around defect center (tight to suppress zero-bg)
CENTER_REFINE = 4       # +/- pixel range for center re-detection
SEED = 42

# ============================================================================


def extract_signatures(tiff_dir):
    """Load all DID***#x,y.tiff and return centered abs defect signatures."""
    pattern = re.compile(r'DID(\d+)#(\d+),(\d+)\.tiff$')
    paths = sorted(glob(os.path.join(tiff_dir, '*.tiff')))
    records = []
    for path in paths:
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        cx_init, cy_init = int(m.group(2)), int(m.group(3))
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim != 3 or img.shape[0] != 3:
            continue
        target = img[0]
        diff = target - 0.5 * (img[1] + img[2])
        H, W = diff.shape

        y_lo = max(0, cy_init - CENTER_REFINE)
        y_hi = min(H, cy_init + CENTER_REFINE + 1)
        x_lo = max(0, cx_init - CENTER_REFINE)
        x_hi = min(W, cx_init + CENTER_REFINE + 1)
        local = diff[y_lo:y_hi, x_lo:x_hi]
        abs_local = np.abs(local)
        total = abs_local.sum()
        if total > 1e-6:
            yy, xx = np.mgrid[0:abs_local.shape[0], 0:abs_local.shape[1]]
            cy_centroid = float((yy * abs_local).sum() / total)
            cx_centroid = float((xx * abs_local).sum() / total)
            cy = y_lo + int(round(cy_centroid))
            cx = x_lo + int(round(cx_centroid))
        else:
            ly, lx = np.unravel_index(np.argmax(abs_local), abs_local.shape)
            cy, cx = y_lo + ly, x_lo + lx

        h = ROI_SIZE // 2
        if cy - h < 0 or cy + h > H or cx - h < 0 or cx + h > W:
            continue
        sig_signed = diff[cy - h:cy + h, cx - h:cx + h]
        records.append({
            'filename': os.path.basename(path),
            'signature_signed': sig_signed.copy(),
            'signature_abs': np.abs(sig_signed),
            'amplitude': float(np.max(np.abs(sig_signed))),
            'cx': int(cx), 'cy': int(cy),
        })
    return records


def ncc(a, b):
    a, b = a.flatten(), b.flatten()
    ac, bc = a - a.mean(), b - b.mean()
    den = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    return float((ac * bc).sum() / den) if den > 1e-10 else 0.0


def shifted_ncc(a, b, max_shift=2):
    """Max NCC over (dy, dx) shifts of `a` within +/- max_shift pixels.

    Tolerates small center-jitter between real defects and the canonically
    centered synthetic PSF. Both arrays must be the same shape.
    """
    h, w = a.shape
    pad = max_shift
    a_pad = np.zeros((h + 2 * pad, w + 2 * pad), dtype=a.dtype)
    a_pad[pad:pad + h, pad:pad + w] = a
    best = -1.0
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            a_s = a_pad[pad - dy:pad - dy + h, pad - dx:pad - dx + w]
            s = ncc(a_s, b)
            if s > best:
                best = s
    return best


def pad_to(arr, h_target, w_target):
    """Center-crop or zero-pad each axis independently to (h_target, w_target)."""
    h, w = arr.shape
    if h > h_target:
        py = (h - h_target) // 2
        arr = arr[py:py + h_target, :]
        h = h_target
    if w > w_target:
        px = (w - w_target) // 2
        arr = arr[:, px:px + w_target]
        w = w_target
    out = np.zeros((h_target, w_target), dtype=np.float32)
    py, px = (h_target - h) // 2, (w_target - w) // 2
    out[py:py + h, px:px + w] = arr
    return out


def random_params(rng):
    cfg = {}
    for k, fit in FIT_TOGGLE.items():
        if not fit:
            cfg[k] = FIXED_VALUES[k]
            continue
        space = SEARCH_SPACE[k]
        if isinstance(space, list):
            cfg[k] = space[int(rng.integers(len(space)))]
        else:
            v = float(rng.uniform(*space))
            cfg[k] = int(round(v)) if k == 'outer_r' else v
    return cfg


def params_to_yaml_cfg(params):
    """Convert single-value params dict to a yaml-loadable cfg."""
    def as_range(v):
        return (v, v) if isinstance(v, (int, float)) else v
    return {
        'psf_size': 256, 'crop_size': 32,
        'outer_r': as_range(params['outer_r']),
        'epsilon': as_range(params['epsilon']),
        'ellipticity': (0, 0), 'ellip_angle': (0, 0),
        'square_eps': 0, 'h_stripe_w': 0, 'v_stripe_w': 0,
        'h_outer_crop': 0, 'v_outer_crop': 0,
        'vector_mode': True,
        'na': as_range(params['na']),
        'pol_type': params['pol_type'],
        'defocus': as_range(params['defocus']),
        'astig_x': as_range(params['astig_x']),
        'astig_y': as_range(params['astig_y']),
        'coma_x': as_range(params['coma_x']),
        'coma_y': as_range(params['coma_y']),
        'spherical': as_range(params['spherical']),
        'trefoil_x': as_range(params['trefoil_x']),
        'trefoil_y': as_range(params['trefoil_y']),
        'brightness': (5000, 5000), 'background': (5, 5),
        'poisson_noise': False, 'gaussian_sigma': (0.5, 0.5),
        'threshold_multiplier': 0.5,
    }


def evaluate_params(params, signatures, K=K_PSFS_PER_EVAL):
    """Average pairwise NCC: mean over (PSFs sampled from yaml) x (defects).

    For each signature, average NCC across K sampled PSFs (rather than taking
    max) to avoid selection bias — a yaml whose 10 PSFs happen to include one
    lucky match shouldn't outscore one that consistently produces good matches.
    """
    cfg = params_to_yaml_cfg(params)
    psfs = []
    for _ in range(K):
        d = create_psf_defect(cfg)
        if d is not None and d.size > 1:
            psfs.append(pad_to(np.abs(d), ROI_SIZE, ROI_SIZE))
    if not psfs:
        return 0.0
    per_sig = [np.mean([ncc(p, sig['signature_abs']) for p in psfs]) for sig in signatures]
    return float(np.mean(per_sig))


def random_search(signatures, rng, n_per_pol=N_ITER_PER_POL):
    """Per-pol-type random search: each pol gets its own continuous-param budget.

    Guarantees fair coverage of polarization options instead of letting them
    compete with continuous params for a shared budget. Scores against
    individual defect signatures (not their avg).
    """
    pol_options = (SEARCH_SPACE['pol_type'] if FIT_TOGGLE['pol_type']
                   else [FIXED_VALUES['pol_type']])
    best_score, best_params, history = -1.0, None, []
    per_pol_best = {}
    for pol in pol_options:
        pol_best = -1.0
        for i in range(n_per_pol):
            params = random_params(rng)
            params['pol_type'] = pol
            score = evaluate_params(params, signatures)
            history.append({'pol': pol, 'iter': i, 'score': score, 'params': dict(params)})
            if score > pol_best:
                pol_best = score
            if score > best_score:
                best_score, best_params = score, params
        per_pol_best[pol] = pol_best
    return best_params, best_score, history, per_pol_best


def per_defect_ncc(params, signatures, K=20):
    """Mean NCC over K sampled PSFs for each signature."""
    cfg = params_to_yaml_cfg(params)
    psfs = []
    for _ in range(K):
        d = create_psf_defect(cfg)
        if d is not None and d.size > 1:
            psfs.append(pad_to(np.abs(d), ROI_SIZE, ROI_SIZE))
    if not psfs:
        return [0.0] * len(signatures)
    return [float(np.mean([ncc(s['signature_abs'], p) for p in psfs])) for s in signatures]


def fit_intensity_range(amplitudes, p_low=5, p_high=95):
    return [float(np.percentile(amplitudes, p_low)),
            float(np.percentile(amplitudes, p_high))]


def cluster_features(signatures, k, seed):
    if k == 1:
        return np.zeros(len(signatures), dtype=int)
    feats = np.array([s['signature_abs'].flatten() for s in signatures])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats /= np.maximum(norms, 1e-10)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(feats)


def write_yaml(cfg, path):
    plain = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    with open(path, 'w') as f:
        yaml.dump(plain, f, sort_keys=False, default_flow_style=None)


def render_comparison(signatures, labels, fitted_results, out_path):
    k = len(fitted_results)
    fig, axes = plt.subplots(k, 4, figsize=(14, 3.5 * k), squeeze=False)
    for c, fit in enumerate(fitted_results):
        idx = np.where(labels == c)[0]
        cluster_sigs = [signatures[i] for i in idx]
        avg_real = np.mean([s['signature_abs'] for s in cluster_sigs], axis=0)
        cfg = params_to_yaml_cfg(fit['params'])
        synth_psfs = []
        for _ in range(50):
            d = create_psf_defect(cfg)
            if d is not None and d.size > 1:
                synth_psfs.append(pad_to(np.abs(d), ROI_SIZE, ROI_SIZE))
        avg_synth = np.mean(synth_psfs, axis=0) if synth_psfs else np.zeros_like(avg_real)
        diff = avg_real / max(avg_real.max(), 1e-10) - avg_synth / max(avg_synth.max(), 1e-10)

        ax = axes[c, 0]
        im = ax.imshow(avg_real, cmap='hot')
        ax.set_title(f'cluster {c} real avg (n={len(cluster_sigs)})', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[c, 1]
        im = ax.imshow(avg_synth, cmap='hot')
        ax.set_title(f"fitted PSF avg (NCC={fit['mean_per_defect_ncc']:.3f})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[c, 2]
        m = max(abs(diff.min()), abs(diff.max()), 1e-10)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-m, vmax=m)
        ax.set_title('residual (norm-real - norm-synth)', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[c, 3]
        ax.axis('off')
        p = fit['params']
        text = (
            f"pol_type:    {p['pol_type']}\n"
            f"na:          {p['na']:.3f}\n"
            f"outer_r:     {p['outer_r']}\n"
            f"epsilon:     {p['epsilon']:.3f}\n"
            f"defocus:     {p['defocus']:+.2f}\n"
            f"astig_x:     {p['astig_x']:+.2f}\n"
            f"astig_y:     {p['astig_y']:+.2f}\n"
            f"\n"
            f"intensity:   [{fit['intensity_abs'][0]:.1f}, {fit['intensity_abs'][1]:.1f}]\n"
            f"per-defect:\n"
            f"  min  NCC:  {fit['min_per_defect_ncc']:.3f}\n"
            f"  mean NCC:  {fit['mean_per_defect_ncc']:.3f}"
        )
        ax.text(0.0, 0.95, text, family='monospace', fontsize=9, verticalalignment='top')

    plt.suptitle(f'PSF fit results (k={k})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def write_assignments_csv(signatures, labels, per_defect_nccs, gt_records, out_path):
    gt_by_name = {r['filename']: r for r in gt_records} if gt_records else {}
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['filename', 'cluster', 'ncc', 'amplitude', 'gt_type', 'gt_intensity'])
        for s, lbl, n in zip(signatures, labels, per_defect_nccs):
            gt = gt_by_name.get(s['filename'], {})
            w.writerow([s['filename'], int(lbl), f'{n:.4f}', f"{s['amplitude']:.2f}",
                        gt.get('type', ''), gt.get('intensity', '')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ground_truth', default=None,
                        help='optional ground_truth.json for assignment verification')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    print(f'Loading defects from {args.input_dir}...')
    sigs = extract_signatures(args.input_dir)
    print(f'  Loaded {len(sigs)} defects')

    gt_records = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth) as f:
            gt_records = json.load(f).get('records', [])
        print(f'  Loaded ground truth: {len(gt_records)} records')

    chosen = None
    chosen_score = -1.0
    all_k_summary = []
    for k in range(1, MAX_K + 1):
        print(f'\n=== k = {k} ===')
        labels = cluster_features(sigs, k, SEED)
        cluster_results = []
        all_per_def_ncc = np.zeros(len(sigs))
        for c in range(k):
            idx = np.where(labels == c)[0]
            cluster_sigs = [sigs[i] for i in idx]
            print(f'  Cluster {c}: {len(cluster_sigs)} defects')
            best_params, best_score, history, per_pol_best = random_search(cluster_sigs, rng)
            per_def = per_defect_ncc(best_params, cluster_sigs, K=20)
            if per_pol_best:
                pol_summary = ', '.join(f'{p}:{s:.3f}' for p, s in
                                        sorted(per_pol_best.items(), key=lambda kv: -kv[1]))
                print(f'    per-pol best NCC: {pol_summary}')
            for i, j in enumerate(idx):
                all_per_def_ncc[j] = per_def[i]
            amps = [s['amplitude'] for s in cluster_sigs]
            intensity_abs = fit_intensity_range(amps)
            cluster_results.append({
                'params': best_params,
                'history': history,
                'best_ncc_avg': best_score,
                'min_per_defect_ncc': float(min(per_def)),
                'mean_per_defect_ncc': float(np.mean(per_def)),
                'n_defects': len(cluster_sigs),
                'intensity_abs': intensity_abs,
            })
            print(f'    pol={best_params["pol_type"]:>8s} '
                  f'na={best_params["na"]:.2f} R={best_params["outer_r"]} '
                  f'ε={best_params["epsilon"]:.2f}')
            print(f'    avg NCC={best_score:.3f}, '
                  f'per-defect min={min(per_def):.3f}, mean={np.mean(per_def):.3f}, '
                  f'intensity_abs=[{intensity_abs[0]:.1f}, {intensity_abs[1]:.1f}]')

        # Score k by overall per-defect mean NCC across all defects (each
        # using its assigned cluster's yaml). Rewards k where adding more
        # clusters lets each defect be matched by a more specialised yaml.
        score = float(np.mean(all_per_def_ncc))
        sizes = [r['n_defects'] for r in cluster_results]
        all_k_summary.append({'k': k, 'score': score, 'sizes': sizes})
        print(f'  k={k} score (overall mean per-defect NCC) = {score:.3f}')

        # Reject k if any cluster is too small (over-clustering signal).
        if min(sizes) < MIN_CLUSTER_SIZE:
            print(f'  ✗ k={k} has tiny cluster (size {min(sizes)} < '
                  f'{MIN_CLUSTER_SIZE}); not splitting further.')
            break

        # Require meaningful improvement over current best — small gains from
        # extra clusters are usually fitting noise, not new types.
        improvement = score - chosen_score
        if chosen is not None and improvement < IMPROVEMENT_MARGIN:
            print(f'  ↘ k={k} improvement ({improvement:+.3f}) below margin '
                  f'{IMPROVEMENT_MARGIN}; keeping k={chosen["k"]}.')
            break

        chosen_score = score
        chosen = {
            'k': k, 'labels': labels.copy(),
            'results': cluster_results,
            'per_defect_ncc': all_per_def_ncc.copy(),
        }

    print(f"\nChosen k = {chosen['k']} (score = {chosen_score:.3f})")
    if chosen_score < NCC_THRESHOLD:
        print(f'  ⚠ score below threshold {NCC_THRESHOLD} — fit may be unreliable, '
              f'consider widening SEARCH_SPACE or relaxing FIT_TOGGLE.')

    # Save outputs
    print(f'\nWriting outputs to {args.output_dir}/')
    for c, fit in enumerate(chosen['results']):
        cfg = params_to_yaml_cfg(fit['params'])
        cfg['intensity_abs'] = fit['intensity_abs']
        cfg['n_samples'] = 1000
        write_yaml(cfg, os.path.join(args.output_dir, f'fitted_type{c}.yaml'))

    render_comparison(sigs, chosen['labels'], chosen['results'],
                      os.path.join(args.output_dir, 'comparison.png'))

    write_assignments_csv(sigs, chosen['labels'], chosen['per_defect_ncc'],
                          gt_records, os.path.join(args.output_dir, 'assignments.csv'))

    log = {
        'seed': SEED, 'n_iter_per_pol': N_ITER_PER_POL, 'ncc_threshold': NCC_THRESHOLD,
        'fit_toggle': FIT_TOGGLE, 'fixed_values': FIXED_VALUES,
        'chosen_k': chosen['k'],
        'all_k_scores': all_k_summary,
        'clusters': [
            {
                'cluster': c,
                'n_defects': r['n_defects'],
                'best_params': r['params'],
                'intensity_abs': r['intensity_abs'],
                'best_ncc_avg': r['best_ncc_avg'],
                'min_per_defect_ncc': r['min_per_defect_ncc'],
                'mean_per_defect_ncc': r['mean_per_defect_ncc'],
            }
            for c, r in enumerate(chosen['results'])
        ],
    }
    with open(os.path.join(args.output_dir, 'fit_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    if gt_records:
        gt_by_name = {r['filename']: r for r in gt_records}
        from collections import Counter
        for c in range(chosen['k']):
            cluster_idx = np.where(chosen['labels'] == c)[0]
            gt_types = [gt_by_name[sigs[i]['filename']]['type'] for i in cluster_idx if sigs[i]['filename'] in gt_by_name]
            print(f"  cluster {c}: {dict(Counter(gt_types))}")


if __name__ == '__main__':
    main()
