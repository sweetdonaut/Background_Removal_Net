# Yaml Search SOP

整個流程你只會用到 **2 個指令** + **編輯 1 個 yaml**：

```
bash src_search/submit_search.sh ...      ← 觸發搜尋（會跑很多 trial）
python src_search/analyze_results.py ...  ← 看結果
src_search/search_configs/*.yaml                     ← 編輯（或新增）spec 來換要搜的參數
```

> **重要：所有指令都要從 project root（`Background_Removal_Net/`）執行。** 不要 `cd src_search` 再跑，否則相對路徑會找不到資料。

> **多 GPU 並行**：每次 run 自帶獨立 spec yaml，**兩張 GPU 跑不同 spec 不會互相影響**。Spec 是 per-run input data，不再是全域 source code。

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
bash src_search/submit_search.sh \
    --spec src_search/search_configs/intensity.yaml \
    --output_root checkpoints/intensity_v1 \
    --n_trials 50 --epochs 20 --pool 1000
```

**方式 B — 編輯 submit_search.sh 改 default**
打開 `src_search/submit_search.sh`，改這兩行 `:-` 後面：
```bash
REAL_VALID_DIR=${REAL_VALID_DIR:-data/30ea_testing/bad}
TRAINING_DATASET_PATH=${TRAINING_DATASET_PATH:-data/grid_stripe_4channel/train/good/}
```

> 絕對路徑跟相對路徑都支援。相對路徑會解讀成相對於你執行 `bash` 的 cwd（也就是 project root）。

### Match radius（detection ↔ GT 容忍度）

判斷一個 detection 是不是 TP 的距離門檻。模型輸出 peak 通常會比 GT 偏 1~4 px，
門檻太嚴會讓「差一點點」算 miss、metric 會抖、trial 排名被噪音干擾。

```bash
# 預設 3.0 px
bash src_search/submit_search.sh --spec ... --output_root ...

# Production 機台 peak 漂得比較遠 → 放寬到 5
MATCH_RADIUS=5 bash src_search/submit_search.sh --spec ... --output_root ...
```

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
bash src_search/submit_search.sh \
    --spec <path/to/spec.yaml> \
    --output_root <checkpoints/your_run> \
    [--n_trials N] [--epochs N] [--pool N]
```

**範例**：
```bash
bash src_search/submit_search.sh \
    --spec src_search/search_configs/intensity.yaml \
    --output_root checkpoints/intensity_v1 \
    --n_trials 50 --epochs 20 --pool 1000
```

**多 GPU 並行**：兩張 GPU 同時跑不同 spec，互不干擾：
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 bash src_search/submit_search.sh \
    --spec src_search/search_configs/intensity.yaml \
    --output_root checkpoints/intensity_v1 \
    --n_trials 50 --epochs 20 --pool 1000

# Terminal 2
CUDA_VISIBLE_DEVICES=1 bash src_search/submit_search.sh \
    --spec src_search/search_configs/optical.yaml \
    --output_root checkpoints/optical_v1 \
    --n_trials 50 --epochs 20 --pool 1000
```

### Flag 說明

| Flag | 必填 | 意義 | 建議值 |
|---|---|---|---|
| `--spec` | ✓ | search spec yaml 路徑 | 指向 `src_search/search_configs/*.yaml` |
| `--output_root` | ✓ | output 根目錄 | 每次新實驗換新名（避免覆蓋）|
| `--n_trials` | | trial 數量 | 50 起跳；smoke test 用 5 |
| `--epochs` | | 每 trial 的 epoch 數 | production 20，smoke test 5 |
| `--pool` | | 每 trial 預生 PSF 數 | production 1000，smoke test 200 |

每次執行會在 `<output_root>/` 產生：
```
search_spec.yaml         ← 整批共用，spec snapshot（保留實驗意圖）
trial_NNN/
  trial_yaml.yaml        ← base_yaml + 這個 trial 抽到的 override 合成的完整 yaml
  params.json            ← 抽到的 override 純值 + 來源 spec 路徑
  epoch_log.jsonl        ← 每 epoch 的 metrics + 30 顆 per-defect 訊號
  summary.json           ← final best metric + history
  *_best.pth             ← best checkpoint
  trial.log              ← 完整 stdout
```

---

## Step 3 — 看結果

**任何時候**都可以跑（包括搜尋還在跑的時候）。會自動 skip 還沒完成的 trial：

```bash
python src_search/analyze_results.py --output_root checkpoints/intensity_v1
```

會印 4 個區塊：
1. **Leaderboard** — trial 按 best metric 排序
2. **Per-defect coverage matrix** — 30 顆在每個 trial 的 best epoch 是否被抓到（✓/.）
3. **Difficulty buckets** — 永遠抓到 / 永遠 miss / 部分抓到
4. **Greedy ensemble suggestion** — 若要組多個 yaml 上線，用哪幾個能 cover 最多 defect

---

## Step 4 — 換要實驗的參數

新流程：spec yaml = 一次搜尋的「實驗意圖」。要換實驗就**複製 / 編輯 spec yaml**。
以下是已準備好的兩個檔案：

```
src_search/search_configs/_template.yaml   ← 列出所有可搜 dim（全部 comment 起來）
src_search/search_configs/intensity.yaml   ← 一個 working example
```

### Spec 檔結構

```yaml
description: "Phase 1 — 鎖定 intensity 範圍"   # 自由文字
base_yaml: src_core/defects/type4_vector.yaml  # base，沒被 dims 蓋到的 key 走它

dims:
  intensity_abs:
    type: range_pair
    low:   {min: 2.0, max: 15.0}
    width: {min: 2.0, max: 12.0}
    high_cap: 30.0
    decimals: 2
```

### 兩種 sampler type

| type | 輸出 | 適用 yaml key |
|---|---|---|
| `scalar_pair` | `[v, v]` | outer_r, brightness, gaussian_sigma, background, 所有 Zernike, epsilon, ellipticity, ellip_angle |
| `range_pair` | `[[low, low+width]]` | intensity_abs |

**`scalar_pair` 範例**：
```yaml
outer_r:
  type: scalar_pair
  range: {min: 30, max: 80}
  decimals: 0     # 0 -> int, >=1 -> 該位小數的 float
```

**`range_pair` 範例**：
```yaml
intensity_abs:
  type: range_pair
  low:   {min: 2.0, max: 15.0}    # low 區間
  width: {min: 2.0, max: 12.0}    # width 區間（high = low + width）
  high_cap: 30.0                   # optional，high 上限
  decimals: 2
```

### 常見實驗範例

**只搜 outer_r**：複製 `_template.yaml`，uncomment `outer_r` 區塊。

**同時搜 intensity + outer_r + brightness**：
```yaml
description: "3-D scan"
base_yaml: src_core/defects/type4_vector.yaml

dims:
  intensity_abs:
    type: range_pair
    low: {min: 2.0, max: 15.0}
    width: {min: 2.0, max: 12.0}
    decimals: 2
  outer_r:
    type: scalar_pair
    range: {min: 30, max: 80}
    decimals: 0
  brightness:
    type: scalar_pair
    range: {min: 2000, max: 10000}
    decimals: 0
```

> 注意：搜的維度越多，需要的 trial 數越多才能覆蓋（譬如 1 維 50 trial 夠，3 維可能要 100+）。

### 想要新的 sampler 形狀？

加在 `src_search/search_space.py`：
1. 寫一個 `_sample_xxx(spec, rng)` 函式
2. 註冊進 `SAMPLERS` dict
3. 在 spec 裡 `type: xxx` 引用

`load_spec()` 會幫你 schema 驗證，type 錯了會給明確的錯誤訊息。

---

## 常用 troubleshooting

| 症狀 | 原因 | 解法 |
|---|---|---|
| `--spec and --output_root are required` | flag 沒給 | 看 `bash submit_search.sh --help` |
| `spec file not found: ...` | spec 路徑錯 | 用 `src_search/search_configs/...` 開頭的相對路徑（從 project root） |
| `dim 'xxx' has unknown type 'yyy'` | spec type 拼錯 | 改成 `scalar_pair` 或 `range_pair` |
| `base_yaml does not exist: ...` | spec 裡 base_yaml 路徑錯 | 確認 yaml 還在原處，或改成絕對路徑 |
| `No tiff images found in ...` | REAL_VALID_DIR 路徑錯，或檔名沒 `#X,Y` | 確認絕對路徑 + 檔名格式 |
| trial 跑到一半 OOM | psf_pool_size 太大 / batch 太大 | 降 `--pool` 或 batch |
| analyze 找不到 trial | 跑搜尋時你 cd 到別處 | 從 project root 跑 |

---

## Slurm 真實佈署

當前 `submit_search.sh` 是 sequential bash loop（單 GPU 模擬）。要丟到 slurm 平行，按腳本內**註解的 slurm 版本**改：把 bash for loop 換成 `#SBATCH --array=1-N`，內層 python 命令保持一樣，sbatch 即可。
