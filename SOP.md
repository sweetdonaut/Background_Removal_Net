# Background Removal Net - SOP

## 1. Dataset Structure

```
data/
└── grid_stripe_4channel/
    ├── train/
    │   └── good/              # 250 good images for training
    │       ├── 000.tiff
    │       ├── 001.tiff
    │       └── ...
    └── test/
        ├── good/              # 4 good images (should have no defects)
        │   ├── 260.tiff
        │   └── ...
        └── bright_spots/      # 5 defect images
            ├── 250.tiff
            └── ...
```

### Image Format

- Format: TIFF (float32)
- Shape: `(4, H, W)` — 4 channels, CHW layout
  - Channel 0: Target
  - Channel 1: Reference 1
  - Channel 2: Reference 2
  - Channel 3: (unused)
- Example size: `(4, 976, 176)`
- Value range: `[0, ~209]` (gray level)

---

## 2. Defect Configuration

Defect configs are stored in `src_core/defects/` as YAML files.

### Gaussian Mode

No config file needed. Gaussian defects (3x3 or 3x5) are generated with hardcoded parameters.

### PSF Mode

Each YAML file defines one type of PSF defect. Example (`type1.yaml`):

```yaml
psf_size: 256           # FFT simulation grid size
crop_size: 32           # center crop before cleaning
outer_r: [30, 32]       # annular aperture outer radius range
epsilon: [0.9, 0.92]    # central obstruction ratio range
brightness: [1500, 3000] # photon count range
threshold_multiplier: 1.0 # connected-peak cleaning threshold
```

Multiple types can be used simultaneously (e.g., `type1` + `type2`).

---

## 3. Training

All commands are executed from `src_core/`.

```bash
cd src_core
```

### Gaussian Mode

```bash
python trainer.py \
    --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
    --checkpoint_path ../checkpoints/4channel \
    --patch_size 128 \
    --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
    --img_format tiff \
    --num_defects_range 4 10 \
    --cache_size 100 \
    --defect_mode gaussian
```

### PSF Mode (single type)

```bash
python trainer.py \
    --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
    --checkpoint_path ../checkpoints/4channel \
    --patch_size 128 \
    --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
    --img_format tiff --num_defects_range 4 10 --cache_size 100 \
    --defect_mode psf --psf_type type1
```

### PSF Mode (multiple types)

```bash
python trainer.py \
    --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
    --checkpoint_path ../checkpoints/4channel \
    --patch_size 128 \
    --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
    --img_format tiff --num_defects_range 4 10 --cache_size 100 \
    --defect_mode psf --psf_type type1 type2
```

### Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `--bs` | Batch size | 16 |
| `--lr` | Learning rate | 0.001 |
| `--epochs` | Number of epochs | 30 |
| `--gpu_id` | GPU device ID (-1 for CPU) | 0 |
| `--patch_size` | Patch size for sliding window | 128 |
| `--num_defects_range` | Min and max defects per patch | 3 8 |
| `--cache_size` | Number of images to cache in memory | 0 |
| `--defect_mode` | `gaussian` or `psf` | gaussian |
| `--psf_type` | PSF config name(s) in `defects/` | None |
| `--gamma_start` | Focal loss gamma start | 1.0 |
| `--gamma_end` | Focal loss gamma end | 3.0 |

### Output

Checkpoint saved to:
```
checkpoints/4channel/BgRemoval_lr{lr}_ep{epochs}_bs{bs}_{H}x{W}.pth
```

---

## 4. Inference

```bash
cd src_core

python inference.py \
    --model_path ../checkpoints/4channel/BgRemoval_lr0.001_ep30_bs16_128x128.pth \
    --test_path ../data/grid_stripe_4channel/test/ \
    --output_dir ../output/pytorch \
    --img_format tiff
```

### Inference Parameters

| Parameter | Description | Default |
|---|---|---|
| `--model_path` | Path to trained checkpoint | (required) |
| `--test_path` | Path to test images directory | (required) |
| `--output_dir` | Output directory for results | (required) |
| `--gpu_id` | GPU device ID (-1 for CPU) | 0 |
| `--img_format` | `png_jpg` or `tiff` | png_jpg |
| `--use_ground_truth_mask` | Calculate AUROC with GT masks | False |

### Output

Results are saved as PNG images preserving the test directory structure:

```
output/pytorch/
├── good/
│   ├── 260_result.png
│   └── ...
└── bright_spots/
    ├── 250_result.png
    └── ...
```

Each result image contains 7 panels:
1. **Target** — input target channel
2. **Ref1** — input reference 1
3. **Ref2** — input reference 2
4. **Target - Ref1** — absolute difference
5. **Target - Ref2** — absolute difference
6. **Double Det.** — min(diff1, diff2)
7. **Heatmap** — model prediction (defect probability)
