# 客觀 Mask + 可配置 Input Channels 實驗紀錄

> ⚠️ **本次改動 aggressive，正式套用到 production 前請完整閱讀第 6 節「生產環境套用清單」**
>
> 🛑 **Production 反饋（後續修正）**：客觀 mask 把 (0,1,1) 當 positive 在生產環境造成大量 FP；mask 規則已退回 v1 風格（plan A'，見第 9 節）。本文件第 1~8 節保留作為實驗紀錄，但目前 dataloader 行為以第 9 節為準。

---

## 1. 背景

### 1.1 起因

Production 端反映合成 PSF 訓練的模型在真實環境（noise 較大）上訊號不穩定，懷疑：
- 合成 vs 真實的 distribution gap 太大
- 模型對 ref 通道的偽影 / 雜訊敏感度過高

### 1.2 推演前的初始狀態

- **輸入**：3 通道 `(target, ref1, ref2)`，皆 raw pixel
- **訓練 mask 規則**：mask=1 ⟺ 該位置由 `target_only_ids` 注入了 defect（生成時硬綁定）
- 訓練 8 種 (T, R1, R2) 組合中只用到 5 種：(0,0,0), (1,0,0), (1,1,0), (1,0,1), (1,1,1)

### 1.3 透過 channel-combination test 發現的盲點

合成 8 種 `(target, ref1, ref2)` 組合（每組合 × ±Δ 兩方向）餵給既有 checkpoint：

| 情境 | (T,R1,R2) | 既有訓練分布 | peak@defect | 期望 |
|---|---|---|---|---|
| target_only | (1,0,0) | 有 | 1.00 | ≈ 1 ✓ |
| ref1_only | (0,1,0) | 無 | 0.04 | ≈ 0 ✓（間接學會） |
| ref2_only | (0,0,1) | 無 | 0.04 | ≈ 0 ✓（間接學會） |
| **both_refs** | **(0,1,1)** | **無** | **1.00** | **≈ 0 ✗** |
| all_three | (1,1,1) | 有 | 0.02 | ≈ 0 ✓ |

**(0,1,1) 是嚴重盲點**。物理分析：

| 情境 | target | ref1 | ref2 | diff1 | diff2 |
|---|---|---|---|---|---|
| (1,0,0) intensity=−Δ | bg−Δ | bg | bg | −Δ | −Δ |
| (0,1,1) intensity=+Δ | bg | bg+Δ | bg+Δ | −Δ | −Δ |

兩者 diff 結構**完全相同**，模型只能靠 target 通道的局部絕對亮度區分——這是個比 diff 訊號弱很多的訊號源。

### 1.4 客觀視角的重新定義

更深層的問題：「(1,0,0) intensity=−Δ 跟 (0,1,1) intensity=+Δ 是不是同一個物理事件？」在「target 比 refs 暗 Δ」這個相對 metric 上**完全相同**。

採用以下客觀判準重新定義 ground truth：

> **defect ⟺ sign(diff1) == sign(diff2) AND min(|diff1|, |diff2|) > 閾值**

兩個關鍵後果：

1. (1,0,0) 與 (0,1,1) **兩者都是 positive class** → 不存在「需要區分但無法區分」的 ambiguity
2. mask 變成 `(target, ref1, ref2)` 的**純函數** → 跟合成方式解耦，任何 input channel 組合都能乾淨訓練

---

## 2. 改動清單

### 2.1 新函數（`src_core/dataloader.py`）

| 函數 | 說明 |
|---|---|
| `sign_consistent_double_det(d1, d2)` | 純函數。sign 一致時取 abs 較小那個 diff，否則回 0 |
| `objective_defect_mask(target, ref1, ref2, min_abs_diff=5.0)` | 純函數。從合成影像客觀算 mask |
| `build_input_channels(target, ref1, ref2, channel_names)` | 已存在；內部 `double_det` 改用 sign-consistent 版本 |
| `DEFECT_MIN_ABS_DIFF = 5.0` | 模組級常數，mask 閾值 |
| `SUPPORTED_CHANNELS` | `('target', 'ref1', 'ref2', 'diff1', 'diff2', 'double_det')` |

### 2.2 行為變化（向後**不**相容）

`Dataset.generate_defect_images_on_channels()`：
- mask 不再硬綁定 `target_only_ids`，改成 apply 完所有 defect 後呼叫 `objective_defect_mask(...)`
- **(0,1,1) 現在是 positive class**（之前是負樣本）
- partial_leak 機制保留邏輯，但 mask 不再需要特殊處理（弱 leak 仍 mask=1，強 leak 自動 mask=0）

### 2.3 可配置 input channels（向後相容）

`src_core/trainer.py`：
- 新增 `--input_channels` CLI（`nargs='+'`，choices = SUPPORTED_CHANNELS）
- 預設 `['target', 'ref1', 'ref2']`（沿用既有行為）
- Checkpoint 新增 `'input_channels'` key

`src_core/inference.py`：
- 從 checkpoint 自動讀取 `input_channels`
- 舊 checkpoint 無此 key 時 fallback 到 `['target', 'ref1', 'ref2']` —— **舊 checkpoint 仍可直接用**

### 2.4 inference 視覺化

- `visualize_results` 與 `visualize_point_cloud` 內的 `double_det` 改用 `sign_consistent_double_det` helper
- 人類視覺化跟訓練 mask 規則對齊（同一個 sign-consistency 判準）

### 2.5 新增檔案

| 檔案 | 用途 |
|---|---|
| `src_core/defects/small_spot_match_test.yaml` | DoE 用的 PSF 配置（intensity ±60~80, vector_mode linX, NA=0.95） |
| `utils/make_channel_test_psf.py` | 生成 8 種 channel-combination 測試集 |
| `utils/summarize_doe.py` | 多 checkpoint × 8 情境的批次評估與比較表 |

---

## 3. 實驗結果

### 3.1 訓練配置

| 項目 | 值 |
|---|---|
| 訓練影像 | 250 張，patch 128×128 |
| Batch size / LR / Epoch | 16 / 0.001 / 20 |
| Defect mode | PSF (`small_spot_match_test.yaml`) |
| intensity_abs | [60, 80] |
| num_defects_range | [4, 10] |
| partial_leak | 關閉 |

### 3.2 DoE 三組

| 配置 | `--input_channels` | 動機 |
|---|---|---|
| baseline_v3 | target ref1 ref2 | 客觀 mask 下的對照組 |
| dd_minimal | double_det | 最小 input：「光 DD 就夠」假設 |
| target_dd | target double_det | raw target + DD hint |

### 3.3 8 情境 × 3 配置結果

`peak@defect`（注入位置 17×17 視窗 max），預期 + ≈ 1、− ≈ 0：

| 情境 | (T,R1,R2) | 預期 | baseline_v3 | **dd_minimal** | target_dd |
|---|---|---|---|---|---|
| A target_only | (1,0,0) | + | 1.00 | **1.00** | 1.00 |
| D both_refs | (0,1,1) | + | 1.00 | **1.00** | 1.00 |
| B ref1_only | (0,1,0) | − | 0.026~0.028 | **0.005** | 0.008 |
| C ref2_only | (0,0,1) | − | 0.027~0.042 | **0.005** | 0.008 |
| E all_three | (1,1,1) | − | 0.010 | **0.005** | 0.018 |
| F target+ref1 | (1,1,0) | − | 0.028~0.039 | **0.005** | 0.018 |
| G target+ref2 | (1,0,1) | − | 0.025~0.027 | **0.005** | 0.018 |
| H clean | (0,0,0) | − | 0.034 (global_max) | **0.017** | 0.022 |

### 3.4 觀察

- 三個配置在 8 種情境上**全部符合客觀判準預期**
- **dd_minimal 是最佳結果**：positive=1.00, negative=0.005（baseline 的 1/6~1/8）
- baseline_v3 在 sign-mismatch 情境（B/C/F/G）的 negative 偏高，顯示 raw input 含 ref 通道易產生殘留激發
- target_dd 在含 target 的 negative（E/F/G）反而略高於 dd_minimal——客觀 mask 下 target 通道沒有「補絕對亮度」的工作要做，變成額外雜訊源

### 3.5 訓練 loss

| 配置 | 最終 avg loss |
|---|---|
| baseline_v3 | 1.6e-5 |
| dd_minimal | **4.5e-6** |
| target_dd | 5.8e-6 |

---

## 4. 訓練 SOP

### 4.1 啟動單一訓練

```bash
cd src_core
python trainer.py \
    --bs 16 --lr 0.001 --epochs 20 --gpu_id 0 \
    --checkpoint_path ../checkpoints/<run_name> \
    --patch_size 128 \
    --training_dataset_path ../data/<dataset>/train/good/ \
    --img_format tiff \
    --num_defects_range 4 10 \
    --cache_size 100 \
    --defect_mode psf --psf_type small_spot_match_test \
    --partial_leak_prob 0.0 --partial_leak_scale 0 0 \
    --use_mask False \
    --input_channels <channels...>
```

### 4.2 `--input_channels` 可選值

| 值 | 物理意義 |
|---|---|
| `target` | 原始 target 通道 |
| `ref1` / `ref2` | 原始 reference 通道 |
| `diff1` | target − ref1 |
| `diff2` | target − ref2 |
| `double_det` | sign-consistent double detection |

可任意組合（順序即通道順序），例：
- `--input_channels target ref1 ref2`（baseline）
- `--input_channels double_det`（最簡）
- `--input_channels target double_det`（DD + raw target）
- `--input_channels diff1 diff2`（純 diff）

### 4.3 訓練完成後檢查

```bash
# 確認 checkpoint 存了 input_channels
python -c "
import torch
c = torch.load('../checkpoints/<run_name>/BgRemoval_*.pth', map_location='cpu')
print('input_channels:', c.get('input_channels'))
print('first conv shape:', c['model_state_dict']['encoder.block1.0.weight'].shape)
"
```

第二個輸出的 `[64, N, 3, 3]` 中 `N` 必須等於 `len(input_channels)`。

---

## 5. 測試 SOP

### 5.1 生成 8 種 channel-combination 測試集

```bash
python utils/make_channel_test_psf.py
```

輸出到 `../data/synthetic_channel_test_psf/`，每組合 ±Δ 兩方向各 4 張 tiff（共 60 張）。資料夾命名：

| 資料夾 | (T,R1,R2) | 期望 mask |
|---|---|---|
| `A_target_only_{pos,neg}` | (1,0,0) | 1 |
| `B_ref1_only_{pos,neg}` | (0,1,0) | 0 |
| `C_ref2_only_{pos,neg}` | (0,0,1) | 0 |
| `D_both_refs_{pos,neg}` | (0,1,1) | 1 |
| `E_all_three_{pos,neg}` | (1,1,1) | 0 |
| `F_target_ref1_{pos,neg}` | (1,1,0) | 0 |
| `G_target_ref2_{pos,neg}` | (1,0,1) | 0 |
| `H_clean` | (0,0,0) | 0 |

### 5.2 單 checkpoint 完整 inference（含視覺化）

```bash
python inference.py \
    --model_path ../checkpoints/<run_name>/BgRemoval_*.pth \
    --test_path ../data/synthetic_channel_test_psf/ \
    --output_dir ../output/<run_name>_eval \
    --img_format tiff
```

每張 tiff 產生 `*_result.png`（7 格：target/ref1/ref2/diff1/diff2/double_det/heatmap）與 `*_cloud.png`（散布圖）。

### 5.3 多 checkpoint 批次比較

開啟 `utils/summarize_doe.py`，編輯 `CHECKPOINTS` dict，指向要比較的 ckpt：

```python
CHECKPOINTS = {
    'baseline_v3': os.path.join(PROJECT_ROOT, 'checkpoints/baseline_v3/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
    'dd_minimal':  os.path.join(PROJECT_ROOT, 'checkpoints/dd_minimal/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
    'target_dd':   os.path.join(PROJECT_ROOT, 'checkpoints/target_dd/BgRemoval_lr0.001_ep20_bs16_128x128.pth'),
}
```

```bash
python utils/summarize_doe.py
```

輸出 8 情境 × N 配置的 peak@defect 比較表，正/負 期望以 `+` / `-` 標示。

### 5.4 通過標準

合成測試集上**全部通過**才能進下一步：

| 情境 | 通過標準 |
|---|---|
| A, D（positive） | peak@defect ≥ 0.95 |
| B, C, E, F, G（negative） | peak@defect ≤ 0.05 |
| H（H_clean） | global_max ≤ 0.05 |

任何配置在合成測試集上沒過，**不可** 進 production 評估。

---

## 6. ⚠️ 生產環境套用清單（必讀）

### 6.1 為什麼要謹慎

本次改動同時涉及兩個重大語意變更：

1. **Mask 定義變了**：(0,1,1) 從 mask=0 變 mask=1
2. **可配置 input channel**：模型輸入語意可能跟 production 既有 pipeline 不一致

舊 model（v2 及以前）跟新 model（baseline_v3 / dd_minimal / target_dd）對**完全相同的影像**會給出**完全不同的判斷**（特別是 (0,1,1) 情境）。**新模型把 ref 偽影也當 defect 報出來**。

### 6.2 套用前的檢查清單

部署到 production 前，逐項確認：

- [ ] **合成測試集全通過**（5.4 通過標準）
- [ ] **量化 production 上 (0,1,1) 出現頻率**——拿一批真實影像（不含已知 defect）計算 sign-consistent double_det 的 max value 分布；若高於 mask 閾值（5）的 pixel 比例超過可接受範圍，模型會在這些 ref 偽影位置誤報
- [ ] **跑一批真實已知有 defect 的影像**確認 recall（detection rate）不退化
- [ ] **跑一批真實乾淨影像**確認 precision（false positive rate）符合需求
- [ ] **業務團隊確認**「ref 偽影誤報」是否可接受
  - 客觀 mask 下，新模型**無法區分** target-side defect 與 ref-side artifact
  - 若業務需要區分，這個架構**不適用**，需回退或改設計
- [ ] **與既有 production pipeline 對接驗證**——新 checkpoint 含 `input_channels` key，舊 inference 程式碼若硬寫 3 通道會掛
- [ ] **保留舊模型 checkpoint 與 inference 腳本**作為立即可用的回退路徑

### 6.3 回退路徑

新版本完全向後相容：
- 舊 checkpoint（無 `input_channels` key）放回 `inference.py` 即可正常運作
- 若新模型在 production 表現不佳，把 `--model_path` 指回舊 checkpoint 即可

但**反向不行**：用新 mask 規則訓練的 checkpoint 不能直接拿去舊 pipeline，因為 (0,1,1) 行為已改變。

### 6.4 已知限制

1. **無法區分 target-defect 與 ref-artifact**：客觀 mask 的內建 trade-off。若要區分，需重新設計（例如：要求 ref 之間也要 cross-validate、或加 absolute brightness 訊號）
2. **`DEFECT_MIN_ABS_DIFF=5` 是 hardcode**：對 intensity ±60~80 合適，若 production 的 defect 強度範圍大幅不同需調整
3. **dd_minimal 配置最敏感於 production noise 結構**：若真實 noise 不是 correlated common-mode 而是 uncorrelated，double_det 通道可能放大 noise。先用真實乾淨影像量化才知道
4. **`utils/make_channel_test_psf.py` 的 CENTERS / 強度範圍寫死**：若要換 PSF 配置（例如改用 type3.yaml），需相應更新

### 6.5 後續可考慮的改進方向

依優先順序：

1. **量化 production noise 結構**——決定 dd_minimal 是否真的適用
2. **加 mask threshold 為可配置 CLI 參數**——讓 production 端可微調
3. **比較含 partial_leak 的訓練**——客觀 mask 下行為可能不同，值得測
4. **DoE 補測**：`diff_only`（diff1, diff2）、`full_diff`（target, diff1, diff2, double_det）
5. **真實影像 fine-tuning**——若有少量 labelled production 樣本，可微調 dd_minimal 模型

---

## 7. 檔案地圖

| 路徑 | 角色 |
|---|---|
| `src_core/dataloader.py` | 客觀 mask、sign-consistent helpers、input channel builder |
| `src_core/trainer.py` | `--input_channels` CLI、checkpoint 寫入 |
| `src_core/inference.py` | checkpoint 自動讀通道配置（向後相容） |
| `src_core/defects/small_spot_match_test.yaml` | DoE 用 PSF 配置 |
| `src_core/train.sh` | 訓練範例（含 DoE 配置註解） |
| `utils/make_channel_test_psf.py` | 8 情境測試集生成 |
| `utils/summarize_doe.py` | 多 checkpoint 比較 |
| `docs/develop/objective_mask_input_channel_doe.md` | 本文件 |

---

## 8. 一句話結論（合成階段）

**Mask 改成 `(target, ref1, ref2)` 的純客觀函數後，「(0,1,1) 對稱盲點」從問題定義裡消失了；後續 DoE 顯示 `dd_minimal` 在合成測試上表現最好，但 production 套用前需逐項通過第 6 節檢查清單。**

---

## 9. Production 反饋與 Plan A' 修正

### 9.1 Production 觀察

生產環境實測發現：依第 7 節結論訓練出的模型（特別是 dd_minimal）在真實影像上**對 (0,1,1) 模式產生大量 FP**。Refs 之間 sign-consistent 但 target 乾淨的事件在 production 上**非常密集**，模型把它們全部當 defect 報出來。

### 9.2 Root cause

合成測試的 (0,1,1)（手動注入兩張 ref 同強度 PSF，pixel-perfect 對齊）跟 production 上的 (0,1,1)（emergent，從 ref 之間的對位殘餘、感測器 fixed-pattern、暫態粒子、照明 drift 等自然產生）**分布差太多**：

- 合成版：強訊號（intensity ±60~80）、形狀規則、稀少
- Production 版：強度範圍寬、形狀雜亂、**事件密度高**

客觀 mask 規則對所有 sign-consistent diff 一視同仁，模型在 production 上學成「**任何 sign-consistent diff 都報**」→ FP 爆炸。

更深層的問題：production 的 defect 偵測任務本質是 **asymmetric**（「target 上有 anomaly」），第 1.4 節用 sign-consistent diff 當客觀判準時把它變 symmetric 了。

### 9.3 Plan A'：被動式處理 (0,1,1)

新策略：
- **不主動注入 (0,1,1)**——避免合成 vs 真實 distribution 不一致誤導模型
- **mask 規則回到 `target_only_ids` 綁定**——「target 上有 anomaly」回到唯一 positive 條件
- 假設：good/ 訓練影像本身已含真實 ref-side noise，模型會從這些 patch 自然吸收「常見背景偽訊號不報」

### 9.4 改動

`src_core/dataloader.py`：

| 項目 | 變化 |
|---|---|
| `objective_defect_mask`, `DEFECT_MIN_ABS_DIFF` | **移除**（dead code） |
| `sign_consistent_double_det` | 保留（仍是 `double_det` input channel 與 visualize 用） |
| `generate_defect_images_on_channels` | 重寫：移除 Tier 1 (0,1,1) 強制；mask 改回 `target_only_ids` 綁定（用 `create_binary_mask`）；其餘配額邏輯沿用 v1 風格 |
| Tier 結構 | 每 patch 仍強制 ≥1 顆 (1,0,0)；剩餘三等分給 (1,1,0)/(1,0,1)/(1,1,1)；**(0,1,1) 不出現** |
| 10% 整片乾淨 | 保留 |
| Partial leak | 保留特殊處理（leaked target_only 仍 mask=1） |

`src_core/inference.py`、`src_core/trainer.py`：**無變化**（input channel 配置框架、checkpoint 自動讀寫、視覺化皆無關 mask 規則）。

### 9.5 已知 trade-off

- (1,0,0) intensity = −Δ 與真實 (0,1,1) intensity = +Δ 在 diff 結構上對稱——這個 fundamental ambiguity 重新存在
- 不含 target 通道的 input 配置（例如 `dd_minimal`）在 Plan A' 下**理論上無法區分**這兩種情境
- 真實 production 上如果這兩者強度分布不重疊，仍可用 magnitude 區隔；若重疊則需限制 intensity 方向或重新設計

### 9.6 SOP 影響

- **訓練 SOP（第 4 節）**：指令不變
- **合成測試集 SOP（第 5 節）**：仍可用，但 **D_both_refs (0,1,1) 的期望 mask 從 1 變回 0**——通過標準調整：
  - A (target_only)：peak ≥ 0.95
  - **D (both_refs)：peak ≤ 0.05**（之前是 ≥ 0.95）
  - B, C, E, F, G, H：peak ≤ 0.05
- **生產環境清單（第 6 節）**：客觀 mask 警告已不適用，但「dd_minimal / diff_only 等無 target 通道配置可能有 ambiguity」這條風險仍在

### 9.7 後續驗證項目

1. 用 Plan A' 重訓 baseline / dd_minimal / target_dd，跑同樣 8 情境合成測試
2. **拿一批 good/ 影像直接量化 sign-consistent noise 分布**（驗證「good/ 本身含真實 ref-side noise」這個 plan A' 的核心假設）
3. Production 重測 FP 是否壓住

### 9.8 文件版本

第 1~8 節保留為實驗歷程紀錄，**第 9 節為現行行為的準據**。若未來再次調整 mask 規則，請新增第 10 節而非改寫前文。
