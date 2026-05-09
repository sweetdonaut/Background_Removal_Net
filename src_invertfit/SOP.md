# Inverse Fit SOP

從**真實 wafer defect 影像反推 PSF 物理參數**，把 30 顆 real PSF 的擬合結果輸出成 training 用的 yaml。流程不再仰賴 `src_search` 的盲搜，**用 real data 直接告訴你該抽哪一組 θ**。

> **重要：所有指令都從 project root（`Background_Removal_Net/`）執行。**

---

## 整條流程

```
30 個 real defect tiff (data/30ea_testing/bad/)
            ↓
   src_invertfit/fit_real.py            ← 反推 forward(θ) 對齊每顆 real PSF
            ↓
   src_invertfit/fitted/fitted_theta.json   ← 30 組 θ + 診斷數據
            ↓
   src_invertfit/export_yaml.py         ← 把 JSON 包成訓練可吃的 yaml
            ↓
   src_invertfit/fitted/individual/*.yaml   ← 30 個 yaml
            ↓
   src_search/search_trainer.py         ← 用這 30 個 yaml 訓 UNet
```

---

## Step 1 — 跑反向擬合

```bash
python src_invertfit/fit_real.py \
    --input_dir data/30ea_testing/bad \
    --output_dir src_invertfit/fitted \
    --no_alpha \
    --n_starts 8 \
    --n_iters 800
```

| Flag | 預設 | 說明 |
|---|---|---|
| `--input_dir` | `data/30ea_testing/bad` | 30 顆 real defect tiff 的位置（檔名要符合 `DefectID###.#X,Y.tiff`） |
| `--output_dir` | `src_invertfit/fitted` | JSON + 視覺化輸出位置 |
| `--no_alpha` | off (推薦 on) | 關掉 ref leak coef α 的 fit。實測 α 自由的時候會被 (I, α) degeneracy 拉走、不影響 θ 但 I 會偏 |
| `--n_starts` | 10 | Multi-start 數量。8 顆 Zernike 維度有 phase-retrieval twin，這個數量足夠探基本 basin |
| `--n_iters` | 1000 | 每個 start 跑的 Adam iter 數。800-1000 對 SNR ~20 已收斂 |

跑完會產生：
```
src_invertfit/fitted/
├── fitted_theta.json   ← 30 個 fit 的 θ + I + α + loss + diff stats（**source of truth**）
├── summary.txt         ← 人類可讀的 per-defect 摘要
└── vis/                ← 30 張 8-panel 診斷 PNG
    ├── DefectID001.png
    └── ...
```

### 你**必須先肉眼檢查的**事

打開 `vis/<DefectID>.png`，看下排兩個 `residual1/2` panel：
- ✅ 像純 noise（沒有 PSF-shape 殘留結構）→ fit OK
- ❌ 中央還能看到 PSF-shape 的紅/藍區塊 → fit 沒抓到，可能 forward model 表達力不足或 SNR 太低

也看 `summary.txt` 的 `resid_pp`（per-pixel residual）。健康的範圍是 **1.0-1.2**（接近 diff std ~2.2 的一半）。如果某顆 > 1.5，回去看那顆 PNG，可能是這顆 outlier。

---

## Step 2 — 輸出 yaml

```bash
python src_invertfit/export_yaml.py
```

**預設只生成 Strategy A（30 個 individual yaml）**，這是本地 A/B/C 對照後**唯一通過驗證**的策略。

跑完會產生：
```
src_invertfit/fitted/
└── individual/
    ├── DefectID001.yaml
    ├── DefectID002.yaml
    └── ... (共 30 個)
```

每個 yaml 是「base `type4_vector.yaml` + 該顆 fit 的 Zernike override」，zero-width range（`[v, v]`）。

### 想生成 Strategy B / C（不推薦，僅供分析比對）

```bash
python src_invertfit/export_yaml.py --strategies ABC
```

這會額外生成：
- `aggregated.yaml`（Strategy B：empirical [min, max] 跨 30 顆）
- `clusters/cluster_0{1,2,3}.yaml`（Strategy C：KMeans 群 mean ± 1·std）

**但本地驗證結果**：
- B：r@50 mean=0.300 (n=1)，明顯輸 baseline，不要拿去 train
- C：r@50 mean=0.444 ± 0.386（n=3），1/3 seed 完全崩盤

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

**重要 flag**：
- `--psf_yaml_path src_invertfit/fitted/individual/*.yaml`：bash glob 自動展開成 30 個檔
- `--psf_pool_size 100`：每個 yaml 各自生 100 顆 PSF。30×100=3000 顆 PSF total，足夠訓練分佈廣度
- `--defect_mode psf`：必須

訓練輸出（標準 `src_search` 格式）：
```
checkpoints/inverted_fit_v1/
├── *_best.pth         ← best metric 的 checkpoint
├── epoch_log.jsonl    ← 每 epoch 的 metrics + per-defect rank
└── summary.json       ← final best + history
```

---

## 本地驗證結論（Phase 0e）

3 個 seed (42, 123, 7)，10 epoch，psf_pool 跟其他條件對齊：

| Strategy | r@50 mean ± std | r@150 mean ± std | 災難率 |
|---|---|---|---|
| **individual (A)** | **0.500 ± 0.208** | **0.900 ± 0.115** | 0/3 |
| trial_002 search winner | 0.544 ± 0.135 | 0.756 ± 0.117 | 0/3 |
| cluster (C) | 0.444 ± 0.386 | 0.633 ± 0.521 | 1/3 完全崩盤 |
| aggregated (B) | 0.300 (n=1) | 0.967 (n=1) | n/a |

**Verdict**：
- r@50 上 individual 跟既有 search winner **statistically tied**（Cohen d=−0.25, small）
- r@150 上 individual **顯著勝出**（Cohen d=+1.24, large）— 多抓 ~14% 的 defect 進 top-150 候選名單
- 不需要再做 yaml search，**直接從 real data 反推就有 baseline 級的訓練資料**

---

## 進階使用

### 換不同 real PSF 集合

只要檔名符合 `DefectID###.#X,Y.tiff` 格式：
```bash
python src_invertfit/fit_real.py --input_dir /abs/path/to/your/real_psfs \
    --output_dir /abs/path/to/output
python src_invertfit/export_yaml.py --input_json /abs/path/to/output/fitted_theta.json \
    --output_dir /abs/path/to/output
```

### 換 base yaml（Zernike 以外的 noise / brightness 設定）

```bash
python src_invertfit/export_yaml.py \
    --base_yaml src_core/defects/your_custom_base.yaml
```

只有「fit_param_names」（8 個 Zernike）會被覆寫，其它（brightness、background、gaussian_sigma、intensity_abs、na、pol_type 等）從 base 完整繼承。

### Diagnostic：跑 verification regression test

如果哪天動了 `forward.py` 或 `inverse_fit.py` 想確認沒壞掉：
```bash
# 1-channel 正確性
python src_invertfit/phase0a.py --n_trials 10 --n_starts 8 --n_iters 600

# 3-channel 正確性
python src_invertfit/phase0b.py --n_trials 10 --n_starts 8 --n_iters 800
```

兩個應該都要回報 `Verdict: GOOD` 或 `EXCELLENT`。

---

## Troubleshooting

| 症狀 | 原因 | 解法 |
|---|---|---|
| `Unrecognized filename` | tiff 檔名沒有 `#X,Y` 格式 | 重命名成 `DefectID001#22,26.tiff` 之類 |
| residual PNG 一片紅藍橫條 | 三次 capture 之間 brightness drift 沒被消掉 | `preprocess.py` 預設已啟用 diff median subtract，不要關掉 |
| residual PNG 中央還有 PSF 殘留 | forward model 表達不到這顆 real PSF | 看是不是有 chromatic / sensor MTF 等沒 model 的物理 |
| `fit_three_channel` 收斂 alpha=0.5 | (I, α) degeneracy 把 α 拉到中央 | 改用 `--no_alpha`（推薦）或加大 `--lambda_alpha` |
| 訓練時 `--psf_yaml_path src_invertfit/fitted/individual/*.yaml` 報錯 | shell glob 沒展開（用了奇怪的 shell 或路徑有空格）| 改成手動列出全部 30 個檔 |
| 30 顆 fit 完都是 high resid_pp (>1.5) | 全部 SNR 太低 / 全部 wafer 紋理沒 align | 檢查 phase correlation 找的 shift 是不是合理（看 PNG 上的 shift 數字）|

---

## 模組結構

```
src_invertfit/
├── forward.py            可微分 PSF forward (torch, vector mode)
├── inverse_fit.py        fit_one + fit_three_channel
├── preprocess.py         triplet load + phase-corr registration + diff median subtract
├── fit_real.py           ← Step 1 entry point
├── export_yaml.py        ← Step 2 entry point
├── phase0a.py            1-channel regression test
├── phase0b.py            3-channel regression test
└── SOP.md                this file
```

每個 .py 第一行的 docstring 都有更詳細的設計理由跟 caveat，動 code 之前先讀。
