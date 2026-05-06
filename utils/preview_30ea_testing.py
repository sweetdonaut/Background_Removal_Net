"""Quick visual sanity-check for data/30ea_testing/bad/.

Renders a 6x5 grid showing target with red crosshair on the labeled (x, y).
"""

import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '30ea_testing', 'bad')
OUT_PATH = os.path.join(PROJECT_ROOT, 'data', '30ea_testing', 'preview.png')

PATTERN = re.compile(r'#(\d+),(\d+)\.tiff$')


def main():
    paths = sorted(glob(os.path.join(DATA_DIR, '*.tiff')))
    if not paths:
        raise SystemExit(f'No tiff found in {DATA_DIR}')

    fig, axes = plt.subplots(6, 5, figsize=(18, 22))
    for ax, path in zip(axes.flat, paths):
        img = tifffile.imread(path)  # (H, W, 3)
        target = img[:, :, 0]
        m = PATTERN.search(os.path.basename(path))
        cx, cy = int(m.group(1)), int(m.group(2))

        vmin = float(np.percentile(target, 2))
        vmax = float(np.percentile(target, 98))
        ax.imshow(target, cmap='gray', vmin=vmin, vmax=vmax)
        ax.plot([cx], [cy], marker='+', color='red', markersize=14, mew=1.5)
        ax.set_title(os.path.basename(path), fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Preview saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
