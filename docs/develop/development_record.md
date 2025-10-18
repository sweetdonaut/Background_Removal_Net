# Background Removal Net 開發記錄

## 專案當前狀態

**最後更新**: 2025-10-18

本專案使用三通道對比學習進行智慧去背，透過 Target、Ref1、Ref2 三個通道的差異訓練 UNet 模型識別目標缺陷。

**當前版本特色**:
- ✅ 簡潔的點缺陷合成策略
- ✅ 動態 Focal Loss gamma scheduling
- ✅ **支援 Strip (976×176) 和 Square (320×320) 兩種圖片格式**
- ✅ **支援 uint8 和 float32 兩種資料型態**
- ✅ **自動偵測方形圖片尺寸**
- ✅ 完整的 ONNX 部署方案（Strip 和 Square 各自獨立）
- ✅ 效能優化（patch-based processing + image cache）
- ✅ **腳本分離（strip 和 square 各自獨立的訓練/推理/匯出腳本）**

---

## 核心設計

### 1. 三通道對比學習機制

**設計理念**：
```
Target 有 且 Ref1 和 Ref2 都沒有 → 目標缺陷（保留在 GT mask）
三個通道都有的 → 背景（去除）
部分通道有的 → 對比學習信號
```

**通道定義**：
- **Target (Channel 0)**：包含所有合成缺陷
- **Ref1 (Channel 1)**：隨機移除部分缺陷
- **Ref2 (Channel 2)**：隨機移除不同的部分缺陷
- **Ground Truth Mask**：只標記 Target 獨有的缺陷

### 2. 缺陷合成策略

**當前策略**（基於高斯分布的點缺陷）：

**缺陷類型**：
- 3×3 點缺陷：sigma = 1.3
- 3×5 點缺陷：sigma = (1.0, 1.5)
- 各 50% 機率

**生成參數**：
- **缺陷數量**：3-8 個 per patch（可調整）
- **缺陷強度**：[-80, -60, 60, 80]（亮暗混合）
- **生成機率**：50%（每個 patch 有 50% 機率含缺陷）
- **Binary mask 閾值**：0.1

**智慧分配邏輯**：
```python
# 確保有效的對比學習
Target-only defects: 1-2 個（進入 GT mask）
Only in Ref1: ~1/3 剩餘缺陷
Only in Ref2: ~1/3 剩餘缺陷
Both Refs: ~1/3 剩餘缺陷
```

**範例**（5 個缺陷）：
```
Target: [A, B, C, D, E]  # 所有缺陷
Ref1:   [A, C]          # 部分缺陷
Ref2:   [B, C]          # 不同的部分缺陷
GT:     [D, E]          # Target 獨有 → 訓練目標
```

---

## 訓練策略

### 1. 動態 Focal Loss Gamma Scheduling

**核心創新**：使用 cosine schedule 動態調整 focal loss 的 gamma 值

**實作細節**：
```python
gamma(t) = gamma_start + (gamma_end - gamma_start) * (1 - cos(t/T * π)) / 2

預設配置：
- gamma_start = 1.0  (訓練初期，類似 cross entropy)
- gamma_end = 3.0    (訓練後期，強化 hard examples)
- schedule = 'cosine' (平滑過渡)
```

**優勢**：
- 初期：學習基礎特徵，避免過早陷入困難樣本
- 中期：逐漸增加對困難樣本的關注
- 後期：精細調整，專注於難分類區域

**Alpha 設定**：
- `alpha = 0.75`：針對缺陷類別（前景）的權重平衡

### 2. 四通道圖片支援

**背景**：生產環境的圖片包含第 4 通道（mask），訓練和推理需要相容

**處理流程**：
```python
# 1. 載入圖片（可能是 3 或 4 通道）
image = tifffile.imread(path)  # Shape: (C, H, W) or (H, W, C)

# 2. Strip 類型：CHW → HWC 轉換
if image_type == 'strip' and image.shape[0] in [3, 4]:
    image = np.transpose(image, (1, 2, 0))

# 3. 只保留前 3 通道
if image.shape[2] == 4:
    image = image[:, :, :3]  # Target, Ref1, Ref2
```

**一致性**：dataloader.py、trainer.py、inference_pytorch.py、inference_onnx.py 全部支援

### 3. 訓練參數配置

**推薦配置**（train.sh）：
```bash
--bs 16                          # Batch size
--lr 0.001                       # Learning rate
--epochs 30                      # Training epochs
--num_defects_range 4 10         # Defects per patch
--image_type strip               # Image type (strip/square/mvtec)
--img_format tiff                # Image format
--patch_size 128×128 (strip)     # Auto-selected based on image_type
            256×256 (square/mvtec)
--gamma_start 1.0                # Focal loss gamma start
--gamma_end 3.0                  # Focal loss gamma end
--cache_size 100                 # Image cache (optional)
--seed 42                        # Random seed (optional)
```

**學習率調度**：
- MultiStepLR: [0.8×epochs, 0.9×epochs]
- Gamma: 0.2

---

## 模型架構

### 1. UNet Segmentation Network

**輸入/輸出**：
- Input: (B, 3, H, W) - 三通道堆疊（Target, Ref1, Ref2）
- Output: (B, 2, H, W) - 二分類 logits（背景/前景）

**架構特點**：
- Base channels: 64
- Encoder: 6 個 block，5 次 MaxPool (通道數：3→64→128→256→512→512→512)
- Decoder: 5 次上採樣 + skip connections
- 激活函數: ReLU + BatchNorm

### 2. Focal Loss

**公式**：
```
FL(pt) = -α * (1 - pt)^γ * log(pt)
```

**參數**：
- Alpha (α): 0.75（缺陷類別權重）
- Gamma (γ): 動態調整（1.0 → 3.0）
- Smooth: 1e-5（標籤平滑）

---

## 推理與部署

### 1. PyTorch 推理

**特點**：
- 使用 `calculate_positions()` 動態計算 patch 位置
- 滑動窗口推理，與訓練一致的 patch 策略
- 支援可視化輸出（7 個子圖）

**使用範例**：
```bash
python inference_pytorch.py \
    --model_path ./checkpoints/model.pth \
    --test_path ./test_images/ \
    --output_dir ./output/pytorch \
    --img_format tiff \
    --image_type strip
```

**可視化內容**：
1. Target 通道
2. Ref1 通道
3. Ref2 通道
4. Target - Ref1 差異圖
5. Target - Ref2 差異圖
6. Double Detection (min(diff1, diff2))
7. 模型預測 Heatmap

### 2. ONNX 部署方案

#### 架構設計（三層包裝）

**Layer 1: SegmentationNetwork**
- 基礎 UNet 模型
- Input: (B, 3, H, W)
- Output: (B, 2, H, W) - logits

**Layer 2: SegmentationNetworkONNX**
- 新增 softmax
- 轉換輸出格式：(B, 2, H, W) → (B, 3, H, W)
  - Channel 0: 異常 heatmap（前景機率）
  - Channel 1-2: 零填充（滿足生產需求）

**Layer 3: SegmentationNetworkONNXFullImage**
- **關鍵特色**：滑動窗口邏輯嵌入在 ONNX 模型內
- Input: (1, 3, 976, 176) - 完整 strip 圖片
- Output: (1, 3, 976, 176) - 完整 heatmap
- **固定配置**：9 個 Y patches × 2 個 X patches
- **拼接策略**：中心區域拼接（無縫合成）

#### 匯出流程

```bash
python export_onnx_fullimage.py \
    --checkpoint_path ./checkpoints/model.pth \
    --output_path ./onnx_models/model_fullimage.onnx \
    --opset_version 11
```

**匯出內容**：
- 完整的滑動窗口邏輯
- Patch 裁切與拼接策略
- Softmax + 通道轉換
- 即用型單檔模型

#### ONNX 推理

```bash
python inference_onnx.py \
    --model_path ./onnx_models/model_fullimage.onnx \
    --test_path ./test_images/ \
    --output_dir ./output/onnx \
    --img_format tiff \
    --image_type strip
```

**優勢**：
- ✅ 單次呼叫處理完整圖片
- ✅ 無需外部前處理
- ✅ 生產環境友善
- ✅ 與 PyTorch 推理結果一致

---

## 效能優化

### 已實施的優化（2025-08）

1. **動態 Patch 索引計算**
   - 移除預先計算的 patch list
   - 減少 98.3% 初始記憶體使用

2. **先裁剪後處理**
   - 在 patch 上直接生成缺陷
   - 減少 96% 處理記憶體
   - 提升 10-20 倍處理速度

3. **局部高斯缺陷生成**
   - 只在影響區域創建陣列
   - 避免全圖大小的運算

4. **圖片快取機制**
   - LRU cache 減少重複載入
   - 可配置快取大小（`--cache_size`）
   - 減少 87.5% I/O 操作

**整體成效**：
- 訓練速度：提升 4-6 倍
- 記憶體使用：減少 90% 以上
- I/O 操作：減少 87.5%

---

## 資料集

### 當前使用的資料集

#### 1. Strip 圖片資料集

**grid_stripe_4channel** (TIFF float32 格式)：
```
grid_stripe_4channel/
├── train/
│   └── good/              # 訓練圖片（4 通道，976×176）
├── test/
│   ├── good/              # 正常測試圖片
│   └── bright_spots/      # 異常測試圖片
└── ground_truth/
    └── bright_spots/      # Ground truth masks
```

**圖片格式**：
- 尺寸：976×176 (Strip)
- 通道：4 通道 (Target, Ref1, Ref2, Mask)
- 格式：TIFF float32
- 訓練時只使用前 3 通道
- Patch 配置：128×128，9 Y × 2 X = 18 patches per image

#### 2. Square 圖片資料集

**grid_align_3channel** (TIFF uint8 格式)：
```
grid_align_3channel/
├── train/
│   └── good/              # 訓練圖片（3 通道，320×320）
└── test/
    ├── good/              # 正常測試圖片
    └── bright_spots/      # 異常測試圖片（含合成缺陷）
```

**圖片格式**：
- 尺寸：320×320 (Square)
- 通道：3 通道 (Target, Ref1, Ref2)
- 格式：TIFF uint8
- 數值範圍：0-255
- Patch 配置：128×128，3 Y × 3 X = 9 patches per image

詳細說明請參考：@dataset_description.md

---

## 使用範例

### Strip 圖片完整流程（976×176）

```bash
# 1. 訓練模型
bash train_stripe.sh

# 2. PyTorch 推理（測試）
bash inference_pytorch_stripe.sh

# 3. 匯出 ONNX（生產部署）
bash export_onnx_stripe.sh

# 4. ONNX 推理（驗證）
bash inference_onnx_stripe.sh
```

### Square 圖片完整流程（320×320）

```bash
# 1. 訓練模型
bash train_square.sh

# 2. PyTorch 推理（測試）
bash inference_pytorch_square.sh

# 3. 匯出 ONNX（生產部署）
bash export_onnx_square.sh

# 4. ONNX 推理（驗證）
bash inference_onnx_square.sh
```

---

## 重要里程碑

### 2024 年 8 月以前
- ✅ 建立基礎架構（UNet + Focal Loss）
- ✅ 實作高斯缺陷生成
- ✅ 三通道對比學習機制
- ✅ 資料載入器與訓練流程

### 2024 年 8 月
- ✅ 效能優化（4-6 倍訓練加速）
- ✅ 條紋偽陽性問題探索（實驗多種方案）
- ✅ 推理邊界問題修復

### 2024 年 9 月 - 2025 年 1 月
- ✅ 回歸基礎點缺陷策略（移除複雜線條實驗）
- ✅ 動態 Focal Loss gamma scheduling
- ✅ 四通道圖片支援
- ✅ ONNX 完整部署方案
- ✅ 文件整理與歸檔

### 2025 年 10 月 18 日
- ✅ **新增 Square (320×320) 圖片格式支援**
- ✅ **移除 mvtec 圖片類型，簡化為 strip/square 兩種**
- ✅ **支援 uint8 和 float32 兩種資料型態**
- ✅ **自動偵測方形圖片尺寸**
- ✅ **新增 SegmentationNetworkONNXSquare 類別**
- ✅ **Square 圖片的 3×3 patch 配置（positions: [0, 96, 192]）**
- ✅ **腳本分離：所有功能都有 stripe 和 square 獨立腳本**
- ✅ **完整的 Square ONNX 匯出和推理功能**
- ✅ **文件更新與 Git 提交**

---

## 專案檔案結構

```
Background_Removal_Net/
├── Core Files (核心檔案)
│   ├── gaussian.py                      # 高斯缺陷生成
│   ├── dataloader.py                    # 資料載入（支援 strip/square，uint8/float32）
│   ├── model.py                         # UNet + ONNX 包裝（含 Square/Strip wrapper）
│   ├── loss.py                          # Focal Loss（支援動態 gamma）
│   ├── trainer.py                       # 訓練主程式
│   ├── inference_pytorch.py             # PyTorch 推理（通用）
│   ├── inference_onnx.py                # ONNX 推理（通用）
│   └── export_onnx_fullimage.py         # ONNX 匯出（自動判斷 strip/square）
│
├── Strip Scripts (Strip 專用腳本 - 976×176)
│   ├── train_stripe.sh                  # 訓練
│   ├── export_onnx_stripe.sh            # ONNX 匯出
│   ├── inference_pytorch_stripe.sh      # PyTorch 推理
│   └── inference_onnx_stripe.sh         # ONNX 推理
│
├── Square Scripts (Square 專用腳本 - 320×320)
│   ├── train_square.sh                  # 訓練
│   ├── export_onnx_square.sh            # ONNX 匯出
│   ├── inference_pytorch_square.sh      # PyTorch 推理
│   └── inference_onnx_square.sh         # ONNX 推理
│
├── Tools (工具腳本)
│   └── alignment_pipeline.py            # 資料集對齊工具
│
└── docs/develop/
    ├── project_discussion.md            # 專案概念說明
    ├── development_record.md            # 本文件
    ├── dataset_description.md           # 資料集說明
    └── archive/                         # 歷史文件歸檔
```

---

## 未來發展方向

### 短期改進
- [ ] 評估不同 gamma schedule（linear vs cosine vs exponential）
- [ ] 實驗不同的 alpha 值對缺陷檢測的影響
- [ ] 建立更完整的測試集與 benchmark

### 長期方向
- [ ] 探索更輕量的模型架構（MobileNet-based UNet）
- [ ] 模型量化（INT8）加速推理
- [ ] 多尺度推理融合
- [ ] 自適應缺陷生成策略

---

## 參考資料

**相關文件**：
- @project_discussion.md - 專案核心概念
- @dataset_description.md - 資料集詳細說明
- @archive/optimization_history.md - 效能優化歷程
- @archive/experimental_history.md - 實驗與探索記錄
