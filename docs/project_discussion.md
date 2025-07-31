# Background Removal Net 專案討論記錄

## 開發記錄
@development_record.md

## 專案目的說明

### 原始需求描述
這個 project 的目的，是要用一個三通道的影像來訓練去背的任務。影像分別會是 target, ref1, ref2。接著我們會設計 point square defect 貼在三個通道上。target 貼幾顆點，ref1, ref2 會隨機去掉其中幾個點。接著中間會使用 unet 的網路，搭配 focal loss 來計算最後輸出的 mask，mask 的標籤會根據前面三個通道，只保留 target 有，但是 ref1, ref2 都沒有的才會被保留成輸出的 mask。利用這個概念來訓練 unet 學會只保留我們設定的 defect，並且把背景去掉。

### 專案理解整理

**專案目標**：使用三通道影像訓練一個能夠進行智慧去背的 UNet 模型。

**訓練資料設計**：
1. **三個影像通道**：
   - Target：包含完整的 point square defects（點狀方形缺陷）
   - Ref1：從 target 中隨機移除部分 defects
   - Ref2：從 target 中隨機移除部分 defects（與 ref1 移除的不同）

2. **標籤生成邏輯**：
   - 輸出 mask 只保留那些「在 target 中存在，但在 ref1 和 ref2 中都不存在」的 defects
   - 這樣可以確保模型學會識別真正的目標缺陷，而非背景或共同特徵

3. **網路架構**：
   - 使用 UNet 作為主要架構
   - 搭配 focal loss 來處理可能的類別不平衡問題

4. **訓練目標**：
   - 讓模型學會透過對比三個通道的差異，準確識別並保留目標缺陷
   - 有效去除背景和非目標元素

這個設計巧妙地利用了對比學習的概念，透過多參考影像的差異來訓練模型識別特定模式。