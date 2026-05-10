# Inverse Fit SOP

從**真實 wafer defect 影像反推 PSF 物理參數**，輸出成 `src_search` training pipeline 直接吃的 yaml。整條流程**只有一個 yaml 控制 fit 行為**，schema 跟 `src_core/defects/type4_vector.yaml` 一致。

> **重要：所有指令都從 project root（`Background_Removal_Net/`）執行。**

---

## 整條流程

```
30 個 real defect tiff (data/30ea_testing/bad/)
            ↓
   src_invertfit/fit_real.py            ← 用 fit config yaml 控制 fit 行為
   src_invertfit/fit_configs/*.yaml     ← fit 用的 yaml（schema 跟 production 一致）
            ↓
   src_invertfit/fitted/fitted_theta.json   ← 30 組 θ + 診斷數據
            ↓
   src_invertfit/export_yaml.py         ← 把 JSON 包成 PsfDefectPool 吃的 yaml
            ↓
   src_invertfit/fitted/individual/*.yaml   ← 30 個 yaml
            ↓
   src_search/search_trainer.py         ← 用這 30 個 yaml 訓 UNet
```

---

## Fit config yaml 規則

`src_invertfit/fit_configs/*.yaml` 的格式跟你 `src_core/defects/type4_vector.yaml` **完全相同**，只是多一個 `fit:` 區塊放 inverse-fit 專屬 hyperparam。所以：

- 你已有的 production yaml **可以直接複製過來當起點**
- Production 才有用的 field（`intensity_abs`, `brightness`, `gaussian_sigma`, `poisson_noise` 等）會被 fit 自動忽略

### 物理參數規則：每一個 param 都是 `[min, max]`

| 寫法 | 意義 |
|---|---|
| `[v, v]` | **FIXED** — fit 不動，永遠用 v |
| `[a, b]` (a ≠ b) | **FIT** — multi-start init 從 [a, b] uniform 抽，Adam 之後可以走出範圍 |

例如：
```yaml
outer_r: [60, 60]            # FIXED at 60
defocus: [-1.5, 1.5]         # FIT, init range
spherical: [0, 0]            # FIXED at 0 (= 不要 fit 球差)
```

跟 `src_search/search_configs/*.yaml` 的 search dim 概念一樣，差別是 search 是「**訓練時隨機抽樣**」，fit 是「**fit 時 init range**」。

### `fit:` 區塊（inverse-fit 專屬）

```yaml
fit:
  alpha: false               # fit ref-leak (alpha1, alpha2)?
  shift: false               # fit sub-pixel (cy, cx) Fourier shift?
  lambda_alpha: 1.0e-3       # alpha L2 prior（破 (I, alpha) degeneracy）
  lambda_shift: 1.0e-3       # shift L2 prior
  radial_sigma_frac: 0.25    # loss 上的 radial weight std (frac of crop_size)
  mask_sharpness: 2.0        # forward soft-mask 銳度
  n_starts: 8                # multi-start 數量
  n_iters: 800               # 每 start 的 Adam iter 數
  lr: 0.05
  sign_flip_init: true       # 額外塞 sign-flipped init 給 Zernike twin basin
  init_alpha_z: -2.0         # alpha 的 logit init（sigmoid(-2) ≈ 0.12）
```

### 完全不能在 yaml 改的東西

| 項目 | 為什麼 |
|---|---|
| `vector_mode` 必須 true | scalar mode 沒有 port 進 fit forward（你 production 也用 vector）|
| Pupil obstruction (`square_eps`, `h_stripe_w`...) | Production 全 = 0，fit forward 直接假設 0 |
| Sensor 端的 `brightness`, `gaussian_sigma`, `intensity_abs` | 不影響 fit shape（fit 用 `log_I` 自己算 scale）|

---

## 預設 config 對比

`src_invertfit/fit_configs/` 裡有兩個 ready-made config：

| File | 用途 | 跟 production 對齊度 |
|---|---|---|
| `default.yaml` | shift + oversample 都開（最 physically accurate）| 100% 對齊 production type4_vector.yaml 的 `pixel_oversample: 4` |
| `no_shift_no_oversample.yaml` | shift / oversample 都關（v1）| `pixel_oversample: 1`，跟 production 不對齊但 **本地驗證訓出來的 model 比較強** |

**本地 3-seed 驗證結果**：

| Config | r@50 mean ± std | r@150 mean ± std |
|---|---|---|
| `no_shift_no_oversample.yaml` (v1) | 0.500 ± 0.208 | **0.900 ± 0.115** |
| `default.yaml` (v2) | 0.389 ± 0.195 | 0.756 ± 0.252 |
| (對照 trial_002 search winner) | 0.544 ± 0.135 | 0.756 ± 0.117 |

**v1 在 r@150 顯著贏（Cohen d = +1.24 LARGE）、r@50 跟 baseline 持平**。物理上更精確的 v2 反而訓練變差，可能是因為 v1 fit 出來的 fake aberration 意外給 30 個 yaml 帶來更多 morphology 多樣性。

→ **預設我們用 v1 (no_shift_no_oversample.yaml)** export 在 disk 上的 yaml。Production 端可以兩個都跑看哪個適合你環境。

---

## Step 1 — 跑反向擬合

```bash
# 預設 v1（推薦，本地驗證較強）
python src_invertfit/fit_real.py \
    --config src_invertfit/fit_configs/no_shift_no_oversample.yaml

# 或 v2（physically accurate 但本地訓練略差）
python src_invertfit/fit_real.py \
    --config src_invertfit/fit_configs/default.yaml

# 你自己的 config
python src_invertfit/fit_real.py \
    --config /abs/path/to/your_fit.yaml
```

可選 CLI flag（操作面，不影響 fit 行為）：

| Flag | 預設 | 說明 |
|---|---|---|
| `--input_dir` | `data/30ea_testing/bad` | real defect tiff 位置（檔名要是 `DefectID###.#X,Y.tiff`）|
| `--output_dir` | `src_invertfit/fitted` | JSON + 視覺化輸出位置 |
| `--seed` | 0 | multi-start init 的 RNG seed |
| `--gpu_id` | 0 | -1 用 CPU |
| `--reg_crop_size` | 96 | sub-pixel registration 的 wider crop 大小 |
| `--limit` | 0 | 只跑前 N 顆（0 = 全跑），debug 用 |
| `--no_vis` | off | 不出 PNG 視覺化（更快）|

跑完會產生：
```
src_invertfit/fitted/
├── fitted_theta.json   ← 30 個 fit 的 (theta, I, alpha, cy, cx, loss, ...)
├── summary.txt         ← 人類可讀的 per-defect 摘要
└── vis/                ← 30 張 8-panel 診斷 PNG
```

### 必看的事

1. **`vis/<DefectID>.png` 下排 residual1/2** 應該像純 noise，沒 PSF-shaped 殘留
2. **`summary.txt` 的 `resid_pp`** 健康範圍 1.0-1.2（接近 diff std ~2.2 的一半）

---

## Step 2 — 輸出 yaml

```bash
python src_invertfit/export_yaml.py
```

預設只生成 Strategy A（30 個 individual yaml），這是本地驗證唯一通過的策略。

```
src_invertfit/fitted/individual/DefectID001.yaml
                                DefectID002.yaml
                                ...
                                DefectID030.yaml
```

每個 yaml 是「base type4_vector.yaml + 該顆 fit 的 Zernike override」，zero-width range（`[v, v]`）。

`cy`, `cx`, `alpha1`, `alpha2` 是 observation-specific（這顆 defect 的特性），**不會** export 到 yaml — production 訓練本來就會自己加 random sub-pixel offset (`pixel_oversample`) 跟 partial leak 增強。

---

## Step 3 — 訓練

接你既有的 `src_search/search_trainer.py` pipeline：

```bash
python src_search/search_trainer.py \
    --bs 16 --lr 0.001 --epochs 20 \
    --gpu_id 0 \
    --checkpoint_path checkpoints/inverted_fit_v1 \
    --training_dataset_path data/grid_stripe_4channel/train/good/ \
    --img_format tiff \
    --defect_mode psf \
    --psf_yaml_path src_invertfit/fitted/individual/*.yaml \
    --psf_pool_size 100 \
    --psf_pool_workers 4 \
    --num_workers 4 \
    --real_valid_dir data/30ea_testing/bad \
    --main_metric recall@50 \
    --seed 42
```

`--psf_yaml_path src_invertfit/fitted/individual/*.yaml` 會被 bash glob 自動展開成 30 個檔。

---

## 進階：客製化 fit config

最常見的修改場景：

### 「我也想 fit pupil 幾何 (e.g. outer_r)」
複製 `default.yaml`，改：
```yaml
outer_r: [55, 65]            # 從 [60, 60] FIXED 改成 FIT range
```

### 「我有不同的 NA」
```yaml
na: [0.92, 0.92]             # 換成你 production 的 NA
```

### 「我想要更精細的 multi-start」
```yaml
fit:
  n_starts: 20
  n_iters: 1500
```

### 「我想開 alpha 看 ref leak 程度」
```yaml
fit:
  alpha: true
  lambda_alpha: 0.01         # 加大 prior 防 (I, alpha) degeneracy 把 alpha 拉到 0.5
```

### 「我有不同的 PSF physical scale」
```yaml
psf_size: 512                # sensor grid 變大
crop_size: 64                # 切出視窗變大
pixel_oversample: 4
```

---

## Diagnostic：跑 regression test

如果哪天動了 `forward.py` 或 `inverse_fit.py` 想確認沒壞掉：
```bash
# 1-channel 正確性
python src_invertfit/phase0a.py --config src_invertfit/fit_configs/no_shift_no_oversample.yaml \
    --n_trials 10 --noise_off

# 3-channel 正確性
python src_invertfit/phase0b.py --config src_invertfit/fit_configs/no_shift_no_oversample.yaml \
    --n_trials 10
```

兩個應該都要回報 `Verdict: GOOD` 或 `EXCELLENT`。

---

## Troubleshooting

| 症狀 | 原因 | 解法 |
|---|---|---|
| `at least one physics param must have min != max` | 整個 yaml 全 FIXED，沒東西可 fit | 把至少一個 Zernike 改成 [-X, X] range |
| `'foo' must be a [min, max] list of two numbers` | yaml 寫成 scalar 或 3 元素 list | 改成 `[v, v]` 或 `[a, b]` |
| residual PNG 一片紅藍橫條 | 三次 capture 之間 brightness drift 沒被消掉 | preprocess 預設已啟用 diff median subtract，不要關 |
| residual PNG 中央還有 PSF 殘留 | forward model 表達不到這顆 real PSF | 開更多 fit dim（例如允許 outer_r 變動）或檢查 SNR |
| 訓練時 `--psf_yaml_path .../*.yaml` 沒展開 | shell glob 失效（路徑有空格、奇怪 shell） | 改成手動列出全部 30 個檔 |
| 30 顆 fit 完都 high resid_pp (>1.5) | SNR 全部太低 / wafer 紋理沒 align | 檢查 phase correlation 找的 shift 是否合理 |

---

## 模組結構

```
src_invertfit/
├── forward.py            可微分 PSF forward (torch, vector mode, 含 oversample + shift)
├── inverse_fit.py        fit_one + fit_three_channel
├── preprocess.py         triplet load + phase-corr registration + diff median subtract
├── config.py             yaml loader (load_fit_config + parse_physics_params)
├── fit_real.py           ← Step 1 entry point
├── export_yaml.py        ← Step 2 entry point
├── phase0a.py            1-channel regression test
├── phase0b.py            3-channel regression test
├── fit_configs/
│   ├── default.yaml                  shift + oversample 都開
│   └── no_shift_no_oversample.yaml   v1，都關（推薦）
└── SOP.md                this file
```

每個 .py 第一行的 docstring 都有設計理由跟 caveat，動 code 之前先讀。
