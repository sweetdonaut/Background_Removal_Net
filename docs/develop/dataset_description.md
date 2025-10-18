# 資料集說明

## 專案核心概念

### 演算法設計理念
本專案的**核心創新**在於使用「合成缺陷」與「三通道對比」來訓練智慧去背模型：

1. **合成缺陷是演算法的關鍵**：透過在三個通道上配置不同的缺陷組合，強迫模型學習辨識什麼是「真正重要的資訊」
2. **對比學習機制**：模型必須比較三個通道的差異，才能正確識別目標
3. **背景去除原理**：背景在三個通道中都存在，只有「Target 獨有的缺陷」才會被標記，因此模型會學會忽略背景

### 訓練資料設計哲學
```
核心概念：
- Target 有的，Ref1 和 Ref2 都沒有 → 重要資訊（保留）
- 三個通道都有的 → 背景（去除）
- 只在部分通道出現的 → 雜訊（去除）
```

---

## 資料集結構

### 當前使用的資料集

#### grid_stripe_4channel (TIFF float32 格式)

**最新版本**：支援四通道圖片（生產環境需求）

```
grid_stripe_4channel/
├── train/
│   └── good/              # 訓練圖片（4 通道 TIFF）
├── test/
│   ├── good/              # 正常測試圖片
│   └── bright_spots/      # 異常測試圖片
├── valid/                 # 驗證集（可選）
│   ├── good/
│   └── bright_spots/
└── ground_truth/          # Ground truth masks
    └── bright_spots/
```

**圖片規格**：
- **尺寸**：976×176（strip 格式）
- **通道數**：4 通道
  - Channel 0 (B): Target 影像
  - Channel 1 (G): Ref1 影像
  - Channel 2 (R): Ref2 影像
  - **Channel 3**: Mask（生產環境附加，訓練時忽略）
- **格式**：TIFF float32
- **數值範圍**：0-255（雖然是 float32，但數值範圍仍為 0-255）

**訓練時的處理**：
```python
# 1. 載入 4 通道圖片（CHW 格式）
image = tifffile.imread(path)  # Shape: (4, 976, 176)

# 2. 轉換為 HWC 格式
image = np.transpose(image, (1, 2, 0))  # Shape: (976, 176, 4)

# 3. 只保留前 3 通道用於訓練
image = image[:, :, :3]  # Shape: (976, 176, 3)

# 4. 動態生成缺陷於三個通道
# 5. 提取 patch 並訓練
```

#### 舊版資料集（僅供參考）

**grid** (PNG 格式)：
- 三通道 PNG 圖片
- 1024×1024 尺寸
- 已不再使用

**grid_tiff** (TIFF 格式)：
- 三通道 TIFF 圖片
- 1024×1024 尺寸
- 已不再使用

---

## 合成缺陷設計

### 缺陷參數（當前版本）

**缺陷類型**：
- **3×3 點缺陷**：標準點狀缺陷，sigma = 1.3
- **3×5 點缺陷**：略微延展的缺陷，sigma = (1.0, 1.5)
- 各佔 50% 機率

**生成策略**：
- **缺陷數量**：3-8 個（預設範圍，可透過 `--num_defects_range` 調整）
- **生成機率**：50%（每個 patch 有 50% 機率含缺陷，50% 為正常樣本）
- **缺陷強度**：[-80, -60, 60, 80]（亮暗缺陷混合）
- **Binary mask 閾值**：0.1（用於生成 GT mask）

### 智慧缺陷分配策略

**目標**：確保每個含缺陷的 patch 都有對比學習價值

**分配邏輯**：
1. **Target-only defects**：1-2 個（必定進入 GT mask，這是訓練目標）
2. **剩餘缺陷分配**：
   - Only in Ref1：約 1/3
   - Only in Ref2：約 1/3
   - Both Refs（作為背景特徵）：約 1/3

**範例**（7 個缺陷的分配）：
```
總共 7 個缺陷：
├─ Target-only: 2 個 [F, G] → 進入 GT mask
└─ 剩餘 5 個分配:
   ├─ Only in Ref1: [A, B]
   ├─ Only in Ref2: [C, D]
   └─ Both Refs: [E]

最終結果：
- Target: [A, B, C, D, E, F, G]  ← 所有缺陷
- Ref1:   [A, B, E]              ← 部分缺陷
- Ref2:   [C, D, E]              ← 不同的部分缺陷
- GT Mask:[F, G]                 ← 只有 target-only
```

**優點**：
- ✅ 確保每個 patch 都有學習價值
- ✅ Ref1 和 Ref2 總是不同（提供對比學習信號）
- ✅ 平衡各種缺陷類型的分布
- ✅ 避免無效的訓練樣本（全都相同或完全不同）

---

## 資料生成流程

### Dataloader 實作重點

**動態缺陷生成流程**：
```python
def __getitem__(self, idx):
    # 1. 載入圖片（可能是 3 或 4 通道）
    image = tifffile.imread(img_path)

    # 2. 處理通道格式
    if image_type == 'strip' and image.shape[0] in [3, 4]:
        image = np.transpose(image, (1, 2, 0))  # CHW → HWC

    # 3. 只保留前 3 通道（丟棄第 4 通道 mask）
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # 4. 提取 patch（先裁剪，減少計算量）
    patch = image[start_y:end_y, start_x:end_x]

    # 5. 分離三個通道
    target_channel = patch[:, :, 0]
    ref1_channel = patch[:, :, 1]
    ref2_channel = patch[:, :, 2]

    # 6. 動態生成缺陷（50% 機率）
    target, ref1, ref2, gt_mask = generate_defect_images_on_channels(
        target_channel, ref1_channel, ref2_channel
    )

    # 7. 堆疊為三通道輸入
    three_channel = np.stack([target, ref1, ref2], axis=0)

    # 8. 轉換為 tensor 並正規化
    return {
        "three_channel_input": torch.from_numpy(three_channel).float() / 255.0,
        "target_mask": torch.from_numpy(gt_mask).float().unsqueeze(0)
    }
```

### 關鍵改進

1. **四通道相容性**
   - 自動檢測並處理 3/4 通道圖片
   - 訓練時只使用前 3 通道
   - 推理時同樣支援 4 通道輸入

2. **先裁剪後處理**（效能優化）
   - 先提取 patch 再生成缺陷
   - 減少 96% 處理記憶體
   - 提升 10-20 倍處理速度

3. **智慧缺陷分配**
   - 確保 1-2 個 target-only 缺陷
   - 平衡 Ref1/Ref2 的缺陷分布
   - 約 5% 樣本無 target-only（模擬正常樣本）

---

## Patch 處理策略

### Strip 圖片（976×176）

**Patch 配置**：
- **Patch 大小**：128×128
- **Y 方向**：9 個 patches（positions: [0, 106, 212, 318, 424, 530, 636, 742, 848]）
- **X 方向**：2 個 patches（positions: [0, 48]）
- **總計**：18 個 patches per image

**動態位置計算**：
```python
y_positions = calculate_positions(976, 128, min_patches=9)
x_positions = calculate_positions(176, 128)
```

**推理時的拼接策略**（無縫合成）：
- **第一個 patch**：保留頂部，移除底部重疊
- **中間 patches**：只使用中心區域
- **最後 patch**：移除頂部重疊，保留底部
- **結果**：無邊界線條，完美拼接

---

## 測試資料集製作

### bright_spots 測試資料

**製作流程**：
1. 從 train/good 選取圖片移至 test/bright_spots
2. 對每張圖片合成缺陷：
   - 使用與訓練相同的缺陷生成邏輯
   - 確保每張圖片都有 target-only 缺陷
   - 生成對應的 ground truth mask
3. 儲存為相同格式（TIFF float32）

### 視覺化結果說明

**RGB 顯示時的顏色意義**（將三通道視為 RGB 時）：
- **紅色**：只有 Target (B 通道) 有缺陷
- **綠色**：只有 Ref1 (G 通道) 有缺陷
- **藍色**：只有 Ref2 (R 通道) 有缺陷
- **紫色**：Target + Ref2 有缺陷
- **黃色**：Ref1 + Ref2 有缺陷
- **青色**：Target + Ref1 有缺陷
- **白色**：所有通道都有缺陷（背景特徵）

---

## 使用建議

### 訓練資料準備

**需求**：
- 正常圖片（good）放置於 `train/good/`
- 格式：TIFF float32
- 通道：3 或 4 通道（第 4 通道會被忽略）
- 尺寸：976×176（strip）或 256×256（square）或 1024×1024（mvtec）

**Dataloader 自動處理**：
- ✅ 動態生成缺陷
- ✅ 智慧分配策略
- ✅ Patch 提取與處理
- ✅ 四通道相容

### 測試與驗證

**測試資料結構**：
```
test/
├── good/           # 正常圖片（驗證誤報率）
└── bright_spots/   # 異常圖片（驗證檢測能力）

ground_truth/
└── bright_spots/   # GT masks for evaluation
```

**評估指標**：
- Image-level AUROC：圖片級別異常檢測
- Pixel-level AUROC：像素級別缺陷定位

### 資料擴充建議

**訓練集擴充**：
- 可將更多正常圖片加入 `train/good/`
- 無需預先合成缺陷（動態生成）
- 建議至少 100-500 張正常圖片

**測試集製作**：
- 手動製作以確保品質
- 確保 GT mask 精確
- 平衡正常/異常樣本比例

---

## 與舊版本的差異

### 主要變更

1. **四通道支援**（新）
   - 舊版：只支援 3 通道
   - 新版：支援 3/4 通道，自動處理

2. **缺陷生成策略**
   - 舊版：複雜的線條負樣本實驗
   - 新版：簡潔的點缺陷策略

3. **資料集格式**
   - 舊版：grid/grid_tiff (1024×1024)
   - 新版：grid_stripe_4channel (976×176, 4 channels)

4. **Patch 處理**
   - 舊版：整張圖生成缺陷後裁剪
   - 新版：先裁剪 patch 再生成缺陷（效能優化）

---

## 相關檔案

### 核心程式碼
- `dataloader.py`：資料載入與缺陷生成（支援 3/4 通道）
- `gaussian.py`：高斯缺陷生成函數
- `trainer.py`：訓練主程式（含 validation）

### 文檔
- `development_record.md`：完整開發記錄
- `project_discussion.md`：專案核心概念
