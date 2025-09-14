# 修改記錄：從 DRAEM 到 Background Removal Net

## 一、Model 相關修改

### 1.1 model.py 實作
基於 DRAEM 的 `DiscriminativeSubNetwork` 架構，實作了 `SegmentationNetwork`：

**主要特點**：
- 輸入通道：3 通道（target, ref1, ref2）
- 輸出通道：2 通道（背景和前景的二分類）
- 基礎架構：UNet with skip connections
- Base channels：64

**網路結構**：
1. **Encoder (EncoderSegmentation)**：
   - 6 個 block，每個包含兩個 Conv2d + BatchNorm + ReLU
   - 5 次 MaxPool2d 進行下採樣
   - 通道數變化：3 → 64 → 128 → 256 → 512 → 512 → 512

2. **Decoder (DecoderSegmentation)**：
   - 5 次上採樣，每次 scale_factor=2
   - Skip connections 將 encoder 的特徵圖與 decoder 連接
   - 最終輸出 2 通道（用於二分類）

### 1.2 loss.py 實作
直接採用 DRAEM 的 `FocalLoss` 實作，用於處理類別不平衡問題：

**參數設定**：
- `gamma=2`：聚焦參數，減少易分類樣本的權重
- `alpha=None`：類別平衡參數（使用預設）
- `smooth=1e-5`：標籤平滑參數
- `size_average=True`：對 batch 取平均

## 二、Defect Generation 相關修改

### 2.1 從 point_square_defect.py 到 gaussian.py
原始實作 point_square_defect.py 被重新設計為更通用的 gaussian.py：

**主要改進**：
1. **尺寸定義修正**：統一使用 (height, width) 格式，3x5 表示高 3 寬 5
2. **簡化缺陷類型**：固定使用 3x3 和 3x5 兩種缺陷
3. **融合方式優化**：採用直接相加法，利用高斯自然漸變

**核心函數**：
- `create_gaussian_defect()`：生成單個高斯缺陷
- `create_binary_mask()`：生成二值化遮罩，閾值 0.1
- `apply_defect_to_background()`：將缺陷應用到背景
- `generate_multiple_defects()`：生成多個缺陷（未使用）
- `generate_random_defect_params()`：隨機參數生成（未使用）

### 2.2 dataloader.py 實作
建立統一的資料載入器，整合三通道生成邏輯：

**Dataset 類別特點**：
1. **必須提供背景影像路徑**：不再支援合成背景
2. **動態生成三通道**：
   - Target：包含所有缺陷
   - Ref1、Ref2：隨機移除部分缺陷
3. **Ground Truth 生成**：只標記 target 獨有的缺陷

**參數設定**：
- 缺陷數量：3-8 個（隨機，在每個 patch 上生成）
- 缺陷強度：[-80, -60, 60, 80]（混合亮暗，已提升 60% 亮度）
- 影像大小：可調整，預設 256x256
- 支援格式：PNG 和 TIFF（float32）

## 三、trainer.py 整合修改

### 3.1 資料載入整合
```python
dataset = Dataset(
    training_path=args.training_dataset_path,  # 必要參數
    img_size=args.img_size,
    num_defects_range=args.num_defects_range,
    img_format=args.img_format,  # 'png' 或 'tiff'
    use_mask=args.use_mask  # 是否使用 mask 監督
)
```

### 3.2 命令列參數
**必要參數**：
- `--bs`：批次大小
- `--lr`：學習率
- `--epochs`：訓練輪數
- `--checkpoint_path`：檢查點儲存路徑
- `--training_dataset_path`：訓練資料集路徑

**選擇性參數**：
- `--gpu_id`：GPU ID（預設：0）
- `--img_size`：影像大小 [height, width]（預設：[256, 256]）
- `--num_defects_range`：缺陷數量範圍 [min, max]（預設：[5, 15]）
- `--img_format`：影像格式 'png_jpg' 或 'tiff'（預設：'png_jpg'）
- `--use_mask`：是否使用 mask 訓練（預設：True）

### 3.3 訓練流程驗證
經測試確認：
1. ✅ 缺陷合成正常（生成正確的三通道）
2. ✅ 網路前向傳播正常（3→2 通道）
3. ✅ 損失計算和梯度更新正常（Loss 下降）
4. ✅ 模型儲存正常（包含所需資訊）

## 四、專案當前狀態

### 已完成部分 ✅

1. **核心功能**：
   - `gaussian.py`：高斯缺陷生成
   - `dataloader.py`：資料載入和三通道生成
   - `model.py`：UNet 分割網路
   - `loss.py`：Focal Loss
   - `trainer.py`：訓練主程式

2. **專案整理**：
   - 測試檔案已刪除
   - 文檔移至 docs/ 資料夾
   - MVTec 資料集已連結
   - 建立 grid 和 grid_tiff 兩種格式資料集

3. **功能驗證**：
   - 完整訓練流程已測試
   - 缺陷生成視覺化已確認
   - 三通道差異機制已驗證
   - 支援 PNG 和 TIFF 格式訓練

4. **最新更新（2025年1月）**：
   - 提升缺陷亮度 60%（[-80, -60, 60, 80]）
   - 新增 80/20 target-only 缺陷控制邏輯
   - 移除 samples_per_epoch 參數（使用實際圖片數量）
   - 參數重新命名（background_path → training_dataset_path）
   - 新增 img_format 和 use_mask 參數
   - **重要變更**：缺陷數量從原始設計的 5-15 個（整張圖）改為 1-5 個（每個 patch），因應優化後的 patch-based 處理方式
   - **2025年8月更新**：改進缺陷分配策略，從 1-5 個提升到 3-8 個，並實作智慧分配確保對比學習效果
   - **2025年8-9月更新**：處理條紋偽陽性問題，詳見 [stripe_false_positive_solutions.md](./stripe_false_positive_solutions.md)

### 專案特色

1. **對比學習設計**：利用三通道差異訓練模型識別獨特缺陷
2. **自然缺陷融合**：高斯分布提供自然漸變，無需額外處理
3. **靈活的資料生成**：動態生成訓練資料，無需預先準備
4. **簡潔的實作**：相比原始 DRAEM，移除重建網路，專注分割任務

### 使用範例

```bash
# PNG/JPG 格式訓練範例
python trainer.py \
    --bs 16 \
    --lr 0.001 \
    --epochs 100 \
    --checkpoint_path ./checkpoints \
    --training_dataset_path ./MVTec_AD_dataset/grid/train/good/ \
    --img_size 256 256 \
    --num_defects_range 5 15 \
    --img_format png_jpg \
    --use_mask True

# TIFF 格式訓練範例
python trainer.py \
    --bs 16 \
    --lr 0.001 \
    --epochs 100 \
    --checkpoint_path ./checkpoints \
    --training_dataset_path ./MVTec_AD_dataset/grid_tiff/train/good/ \
    --img_format tiff
```