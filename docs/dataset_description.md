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

## 資料集結構

### 目前可用的資料集
專案提供兩種格式的資料集，內容相同但檔案格式不同：

#### 1. grid (PNG 格式)
```
grid/
├── train/
│   └── good/          # 250 張訓練圖片（三通道灰階）
├── test/
│   ├── good/          # 4 張正常測試圖片
│   └── bright_spots/  # 10 張含缺陷測試圖片
└── ground_truth/
    └── bright_spots/  # 10 張對應的 mask
```

#### 2. grid_tiff (TIFF float32 格式)
```
grid_tiff/             # 與 grid 相同結構，但為 float32 TIFF 格式
├── train/
│   └── good/          # 250 張 .tiff 檔案
├── test/
│   ├── good/          # 4 張 .tiff 檔案
│   └── bright_spots/  # 10 張 .tiff 檔案
└── ground_truth/
    └── bright_spots/  # 10 張 .tiff mask 檔案
```

### 資料格式說明
- **三通道定義**：
  - Channel 0 (B)：Target 影像
  - Channel 1 (G)：Ref1 影像  
  - Channel 2 (R)：Ref2 影像
- **影像大小**：1024×1024（訓練時會調整為 256×256）
- **數值範圍**：0-255（TIFF 格式儲存為 float32）

## 合成缺陷設計

### 缺陷參數（2025年1月更新）
- **缺陷數量**：5-15 個（隨機）
- **缺陷類型**：
  - 3×3：標準點狀缺陷，sigma = 1.0（50% 機率）
  - 3×5：略微延展的缺陷，sigma = (1.0, 1.5)（50% 機率）
- **缺陷強度**：[-80, -60, 60, 80]（更新：提高 60% 以增加可見度）
- **二值化閾值**：0.1（用於生成 mask）

### Target-only 缺陷控制（新增）
訓練時的缺陷分配策略：
- **80% 機率**：確保至少有 1-2 個 target-only 缺陷
  - 強制 ref1 和 ref2 的移除集合有交集
- **20% 機率**：完全隨機分配
  - 實際約 5-6% 會產生沒有 target-only 缺陷的困難案例
  - 用於提升模型魯棒性，減少誤報

## 測試資料集製作流程

### bright_spots 測試資料生成
1. 從 train/good 移動 10 張圖片到 test/bright_spots
2. 對每張圖片合成缺陷：
   - 使用與訓練相同的缺陷生成邏輯
   - 確保每張圖片都有 target-only 缺陷（用於評估）
   - 生成對應的 ground truth mask

### 視覺化結果
合成缺陷在 RGB 顯示時會呈現不同顏色：
- **紫色**：B+R 通道（target + ref2）有缺陷
- **綠色**：只有 G 通道（ref1）有缺陷
- **青色**：B+G 通道有缺陷
- **白/黑色**：所有通道都有缺陷

## dataloader.py 實作重點

### 動態缺陷生成流程
```python
# 1. 讀取三通道影像
image = cv2.imread(img_path)  # Shape: (H, W, 3)

# 2. 分離通道（訓練時三通道相同）
target_channel = image[:, :, 0]  # B channel
ref1_channel = image[:, :, 1]    # G channel
ref2_channel = image[:, :, 2]    # R channel

# 3. 生成 5-15 個缺陷
# 4. Target 加入所有缺陷
# 5. Ref1/Ref2 隨機移除部分（80% 確保有交集）
# 6. 生成 mask：只標記 target-only 缺陷
```

### 關鍵改進
1. **缺陷強度提升**：從 ±30-50 提升到 ±60-80
2. **智慧缺陷分配**：80/20 策略平衡訓練效果
3. **困難案例保留**：約 5-6% 無 target-only 缺陷的樣本

## 使用建議

### 訓練
- 使用 grid 或 grid_tiff 的 train/good 資料
- dataloader.py 會動態生成缺陷
- 建議 batch size：16-32

### 測試
- 使用 test/bright_spots 評估缺陷檢測能力
- 使用 test/good 評估誤報率
- ground_truth/bright_spots 提供評估標準

### 資料擴充
- 可將更多正常圖片加入 train/good
- 測試資料建議手動製作以確保品質