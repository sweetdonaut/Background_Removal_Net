# Gaussian Defect Generation 設計文檔

## 概述
本文檔描述了使用高斯分布生成缺陷並融合到背景影像的設計流程。

## 最終設計方案

### 1. 缺陷類型
- **3x3 缺陷**：高 3 像素、寬 3 像素，sigma = 1.0
- **3x5 缺陷**：高 3 像素、寬 5 像素，sigma = (1.0, 1.5)
- **尺寸定義**：統一使用 (height, width) 格式

### 2. 缺陷生成
```python
# 高斯分布生成
defect = create_gaussian_defect(
    center=(x, y),
    size=(height, width),
    sigma=sigma,
    image_shape=(H, W)
)
```
- **輸出格式**：完整影像大小的陣列，缺陷位置為高斯分布值（0-1），其餘為 0
- **高斯特性**：利用高斯分布的自然漸變特性，從中心（≈1）到邊緣（≈0）平滑過渡

### 3. Binary Mask 生成
```python
threshold = 0.1  # 10% 閾值
binary_mask = (defect_image > threshold).astype(np.float32)
```
- **用途**：標記缺陷的精確位置，用於訓練標籤和評估
- **閾值選擇**：0.1 能很好地平衡精確度和覆蓋率

### 4. 缺陷融合方式
```python
# 直接相加法
defect_intensity = np.random.choice([-50, -30, 30, 50])  # 隨機選擇亮暗缺陷
output = background + defect_image * defect_intensity
output = np.clip(output, 0, 255)
```

## 三通道生成策略

### 資料生成流程
1. **Target 通道**：包含所有生成的缺陷（5-15 個）
2. **Ref1 通道**：隨機移除部分缺陷
3. **Ref2 通道**：隨機移除部分缺陷（與 Ref1 不同）
4. **Ground Truth**：只標記「Target 有但 Ref1 和 Ref2 都沒有」的缺陷

### 實作細節
```python
# 缺陷移除邏輯
remove_ref1 = np.random.choice(num_defects, num_remove_ref1, replace=False)
remove_ref2 = np.random.choice(num_defects, num_remove_ref2, replace=False)

# Ground truth 生成
for i in range(num_defects):
    if i in remove_ref1 and i in remove_ref2:
        gt_mask = np.maximum(gt_mask, all_masks[i])
```

## 設計優勢

1. **自然漸變**：高斯分布本身提供平滑的邊緣過渡，無需額外的透明度融合
2. **背景保留**：使用相對變化（加法），保留原始背景紋理
3. **計算簡單**：單一加法運算，效率高
4. **對比學習**：透過三通道差異訓練模型識別獨特缺陷

## 確定的參數

- **缺陷數量**：5-15 個（可調整）
- **缺陷強度**：[-50, -30, 30, 50]（暗/亮缺陷混合）
- **缺陷間距**：最小 20 像素邊界
- **影像大小**：256x256（訓練時）
- **Binary mask 閾值**：0.1

## 實作檔案
- `gaussian.py`：核心缺陷生成功能（已完成）
- `dataloader.py`：整合三通道生成和資料載入（已完成）
- `trainer.py`：訓練主程式（已完成）