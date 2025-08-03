# Inference 優化與改進記錄

## 更新歷史
- **2025-08-03**: 完成 inference 邊界問題修復與視覺化改進

## 概述
本文檔記錄推理（inference）階段的各項優化與改進，特別是針對 strip 類型圖片的邊界合成問題。

## 主要問題與解決方案

### 1. 邊界合成線條問題

#### 問題描述
- Strip 圖片（976×176）在使用滑動窗口推理時出現明顯的邊界線條
- Y 方向：8 個 patches，重疊僅 7 pixels（5.5%）
- X 方向：2 個 patches，重疊高達 80 pixels（62.5%）
- 重疊區域平均造成邊界不連續

#### 解決方案：增加 patches 數量 + 中心區域拼接策略

##### Step 1: 增加 Y 方向 patches 數量
```python
# dataloader.py 和 inference.py
if self.image_type == 'strip':
    # 從 8 個增加到 9 個 patches
    self.y_positions = calculate_positions(img_h, self.patch_size[0], min_patches=9)
```

結果：
- Y 方向：9 個 patches，重疊 22 pixels（17.2%）
- 更多重疊提供更好的融合機會

##### Step 2: 實現中心區域拼接策略
不再使用重疊區域平均，改為智慧裁切拼接：

**Y 方向策略**：
- 第一個 patch (0)：保留頂部，使用 Y[0-117]
- 中間 patches (1-7)：只用中心，各使用 106 pixels
- 最後 patch (8)：保留底部，使用 Y[859-976]

**X 方向策略**：
- 第一個 patch：保留左側，使用 X[0-88]
- 第二個 patch：保留右側，使用 X[88-176]

```python
# inference.py 核心邏輯
if image_type == 'strip' and len(y_positions) > 2:
    # Y 方向裁切
    if y_idx == 0:
        y_start_crop = 0
        y_end_crop = patch_h - 11  # 117
    elif y_idx == len(y_positions) - 1:
        y_start_crop = 11
        y_end_crop = patch_h  # 128
    else:
        y_start_crop = 11
        y_end_crop = patch_h - 11  # 106
    
    # X 方向裁切
    if len(x_positions) > 1:
        x_stride = 48
        x_margin = 40
        if x_idx == 0:
            x_start_crop = 0
            x_end_crop = patch_w - x_margin  # 88
        else:
            x_start_crop = x_margin
            x_end_crop = patch_w  # 128
```

##### 優勢
1. **無縫拼接**：每個像素只來自一個 patch，沒有重疊
2. **保留邊緣**：圖片邊緣信息完整保留
3. **消除平均模糊**：避免重疊區域平均造成的問題
4. **簡單高效**：直接賦值，計算簡單

### 2. 視覺化改進

#### 新增 Double Detection 顯示
在 heatmap 前新增第 6 個子圖，顯示「雙重偵測」結果：

```python
# 取 target-ref1 和 target-ref2 的最小值
double_detection = np.minimum(diff1, diff2)
```

意義：
- 顯示在兩個參考通道都被偵測到的缺陷
- 更保守的缺陷估計，減少誤報
- 幫助識別真實缺陷 vs. 單一參考的雜訊

#### 視覺化布局
現在包含 7 個子圖：
1. Target - 目標通道
2. Ref1 - 參考通道 1  
3. Ref2 - 參考通道 2
4. Target - Ref1 - 差異圖 1
5. Target - Ref2 - 差異圖 2
6. Double Detection - 雙重偵測（新增）
7. Heatmap - 模型預測

### 3. 其他改進

#### 3x3 缺陷 sigma 調整
- 將 3×3 缺陷的 sigma 從 1.0 提升到 1.3
- 解決 binary mask 過小的問題
- 改善小缺陷的可見度

#### 缺陷強度提升
- 缺陷強度從 [-50, -30, 30, 50] 提升到 [-80, -60, 60, 80]
- 增加 60% 強度，提高訓練效果

## 效能影響

### 訓練階段
- 使用 9 個 patches 增加約 12.5% 的訓練數據量
- 提供更多樣的訓練樣本

### 推理階段  
- 計算量增加約 12.5%（9 patches vs 8 patches）
- 但消除了平均運算，實際影響有限
- 視覺品質顯著提升，無邊界線條

## 使用方式

### 訓練
```bash
# train.sh 保持不變，dataloader 自動處理
python trainer.py \
    --image_type strip \
    ...
```

### 推理
```bash
# inference.sh 需要指定 image_type
python inference.py \
    --image_type strip \
    ...
```

## 技術細節

### Patch 位置計算
```
Strip 圖片 (976×176)：
- Y: [0, 106, 212, 318, 424, 530, 636, 742, 848]
- X: [0, 48]

拼接結果：
Y: [0-117] + [117-223] + ... + [859-976] = 976 ✓
X: [0-88] + [88-176] = 176 ✓
```

### 相容性
- 保留原有策略用於 mvtec、square 類型
- 僅對 strip 類型使用新策略
- 向後相容現有模型

## 未來改進方向

1. **自適應裁切邊界**
   - 根據實際重疊計算最佳裁切點
   - 不同圖片類型使用不同策略

2. **邊緣檢測輔助**
   - 在圖片邊緣使用特殊處理
   - 考慮 CNN 的 receptive field

3. **多尺度推理**
   - 結合不同 patch 大小的結果
   - 提高細節和整體的平衡

## 結論

透過增加 patches 數量和實現智慧裁切拼接策略，成功解決了 strip 圖片的邊界合成問題。新方法簡單、高效，且效果顯著。Double Detection 的加入也提供了更多診斷信息，有助於模型分析和改進。