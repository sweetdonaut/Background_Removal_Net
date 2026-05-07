# Yaml Search SOP

整個流程你只會用到 **2 個指令** + **編輯 1 個檔案**：

```
bash src_search/submit_search.sh ...      ← 觸發搜尋（會跑很多 trial）
python src_search/analyze_results.py ...  ← 看結果
src_search/search_space.py                ← 編輯這個來換要搜的參數
```

> **重要：所有指令都要從 project root（`Background_Removal_Net/`）執行。** 不要 `cd src_search` 再跑，否則相對路徑會找不到資料。

---

## Step 1 — 指定 production 路徑

你的真實 30 顆驗證影像跟訓練資料可以放在**任意絕對路徑**（不必在 project 內）。檔名必須符合：

```
DefectID???#X,Y.tiff       ← X, Y 是 PSF 中心座標
```

數量不限定 30，evaluator 會自動讀全部。

### 兩種指定方式（擇一）

**方式 A — 環境變數（推薦，不改 source）**
```bash
REAL_VALID_DIR=/abs/path/to/real_30ea \
TRAINING_DATASET_PATH=/abs/path/to/production/clean/good/ \
bash src_search/submit_search.sh checkpoints/search_v1 50 20 1000
```

**方式 B — 編輯 submit_search.sh 改 default**
打開 `src_search/submit_search.sh`，改這兩行 `:-` 後面：
```bash
REAL_VALID_DIR=${REAL_VALID_DIR:-data/30ea_testing/bad}
TRAINING_DATASET_PATH=${TRAINING_DATASET_PATH:-data/grid_stripe_4channel/train/good/}
```

> 絕對路徑跟相對路徑都支援。相對路徑會解讀成相對於你執行 `bash` 的 cwd（也就是 project root）。

### Dead pixel mask（CCD 永久亮點）

CCD 有固定亮點時，會在 heatmap 上產生永久 FP，把真正的 defect 從 top-150 擠出去。
在 `<REAL_VALID_DIR>/dead_pixels.csv` 放一份座標檔，evaluator 會在算分前把每個座標
周圍 10×10 bbox 的 heatmap 設為 0：

```csv
dead_x,dead_y
112,87
245,310
390,200
```

- 預設**自動載入** `<REAL_VALID_DIR>/dead_pixels.csv`（如果存在），不用改任何東西
- 想改路徑：`DEAD_PIXEL_CSV=/abs/path/to/file.csv bash src_search/submit_search.sh ...`
- 想關掉：`DEAD_PIXEL_CSV="" bash ...`（或刪掉那個 csv）
- 想改 bbox 大小：trainer / run_eval / run_trial 都接 `--dead_pixel_half_size`（預設 5 -> 10×10）

---

## Step 2 — 跑搜尋

```bash
bash src_search/submit_search.sh  <output_root>  <n_trials>  <epochs>  <psf_pool_size>
```

**範例**：
```bash
bash src_search/submit_search.sh checkpoints/search_v1 50 20 1000
```

四個位置參數：

| 位置 | 意義 | 建議值 |
|---|---|---|
| 1 | output 根目錄 | 每次新實驗換新名（避免覆蓋） |
| 2 | trial 數量 | 50 起跳；用環境變數測試時可用 5 |
| 3 | 每 trial 的 epoch 數 | production 用 20，smoke test 用 5 |
| 4 | 每 trial 預生 PSF 數 | production 用 1000，smoke test 用 200 |

每個 trial 結束會在 `<output_root>/trial_NNN/` 產生：
```
trial_yaml.yaml      ← 這個 trial 採樣的 yaml
params.json          ← 採樣的 override 純值
epoch_log.jsonl      ← 每 epoch 的 metrics + 30 顆 per-defect 訊號
summary.json         ← final best metric + history
*_best.pth           ← best checkpoint
trial.log            ← 完整 stdout
```

---

## Step 3 — 看結果

**任何時候**都可以跑（包括搜尋還在跑的時候）。會自動 skip 還沒完成的 trial：

```bash
python src_search/analyze_results.py --output_root checkpoints/search_v1
```

會印 4 個區塊：
1. **Leaderboard** — trial 按 best metric 排序
2. **Per-defect coverage matrix** — 30 顆在每個 trial 的 best epoch 是否被抓到（✓/.）
3. **Difficulty buckets** — 永遠抓到 / 永遠 miss / 部分抓到
4. **Greedy ensemble suggestion** — 若要組多個 yaml 上線，用哪幾個能 cover 最多 defect

---

## Step 4 — 換要實驗的參數

`src_search/search_space.py` 已經把 16 個常用 yaml key 的採樣函式都實作好了。**你只要 toggle (uncomment / comment) `sample_params()` 裡的對應行**，其他檔案完全不必動。

### 目前狀態（只搜 intensity_abs）

```python
def sample_params(rng):
    out = {}
    out['intensity_abs'] = sample_intensity_abs(rng)
    # out['outer_r'] = sample_outer_r(rng)
    # out['brightness'] = sample_brightness(rng)
    # ...
    return out
```

### 換實驗：只要 uncomment 你要搜的那行

**範例 A — 改成搜光圈半徑：**
```python
def sample_params(rng):
    out = {}
    # out['intensity_abs'] = sample_intensity_abs(rng)
    out['outer_r'] = sample_outer_r(rng)
    return out
```

**範例 B — 同時搜 intensity + outer_r + brightness：**
```python
def sample_params(rng):
    out = {}
    out['intensity_abs'] = sample_intensity_abs(rng)
    out['outer_r'] = sample_outer_r(rng)
    out['brightness'] = sample_brightness(rng)
    return out
```

> 注意：搜的維度越多，需要的 trial 數越多才能覆蓋（譬如 1 維 50 trial 夠，3 維可能要 100+）。

### 已實作的採樣（Tier 由重要到次要）

**Tier 1 — 訊號 / 雜訊（最強烈推薦）**
| 函式 | yaml key | 範圍 | base |
|---|---|---|---|
| `sample_intensity_abs` | `intensity_abs` | low∈[2,15], width∈[2,12] | [[8,12]] |
| `sample_outer_r` | `outer_r` | [30, 100] | [60, 60] |
| `sample_brightness` | `brightness` | [2000, 10000] | [5000, 5000] |
| `sample_gaussian_sigma` | `gaussian_sigma` | [0.3, 3.5] | [1.5, 1.5] |
| `sample_background` | `background` | [0, 20] | [5, 5] |

**Tier 2 — 光學像差（Zernike，單位 wavelength）**
| 函式 | yaml key | 範圍 |
|---|---|---|
| `sample_defocus` | `defocus` | ±1.5 |
| `sample_astig_x/y` | `astig_x/y` | ±1.0 |
| `sample_coma_x/y` | `coma_x/y` | ±0.8 |
| `sample_spherical` | `spherical` | ±0.8 |
| `sample_trefoil_x/y` | `trefoil_x/y` | ±0.5 |

**Tier 3 — 光圈幾何**
| 函式 | yaml key | 範圍 |
|---|---|---|
| `sample_epsilon` | `epsilon`（中心遮蔽比） | [0, 0.6] |
| `sample_ellipticity` | `ellipticity` | ±0.3 |
| `sample_ellip_angle` | `ellip_angle` | [0, 180]° |

### 採樣範圍不滿意？

直接改 `search_space.py` 對應 `sample_*()` 函式裡的 `rng.uniform(...)` 區間。譬如：
```python
def sample_outer_r(rng):
    r = float(rng.uniform(30.0, 100.0))   # 改成你想要的範圍
    return [round(r), round(r)]
```

---

## 常用 troubleshooting

| 症狀 | 原因 | 解法 |
|---|---|---|
| `No tiff images found in ...` | 路徑錯，或檔名沒 `#X,Y` | 確認絕對路徑 + 檔名格式 |
| `No images found in ...`（training） | training_dataset_path 錯 | 用絕對路徑 |
| trial 跑到一半 OOM | psf_pool_size 太大 / batch 太大 | 降 psf_pool_size 或 bs |
| analyze 找不到 trial | 跑搜尋時你 cd 到別處 | 從 project root 跑 |

---

## Slurm 真實佈署

當前 `submit_search.sh` 是 sequential bash loop（單 GPU 模擬）。要丟到 slurm 平行，按腳本內**註解的 slurm 版本**改：把 bash for loop 換成 `#SBATCH --array=1-N`，內層 python 命令保持一樣，sbatch 即可。
