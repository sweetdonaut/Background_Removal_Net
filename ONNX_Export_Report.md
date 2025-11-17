# ONNX 模型匯出報告

## 匯出指令

```bash
python export_onnx_fullimage.py \
    --checkpoint_path ./checkpoints/4channel/BgRemoval_lr0.001_ep30_bs16_128x128_strip.pth \
    --output_path ./onnx_models/background_removal_strip_normalized.onnx
```

## 模型架構

```
輸入 (1, 3, 976, 176)
    ↓
輸入正規化 (0-255 → 0-1)
    ↓
Sliding Window (9×2 = 18 patches, 128×128)
    ↓
U-Net Segmentation + Softmax
    ↓
Patch 拼接
    ↓
輸出 (1, 3, 976, 176)
```

## 輸入輸出規格

| 項目 | 規格 |
|------|------|
| 輸入形狀 | `(batch, 3, 976, 176)` |
| 輸入範圍 | 0-255 (uint8 或 float32) |
| 輸入通道 | Channel 0: cur, Channel 1: ref0, Channel 2: ref1 |
| 輸出形狀 | `(batch, 3, 976, 176)` |
| 輸出範圍 | 0-1 (機率值) |
| 輸出通道 | Channel 0: Anomaly Heatmap, Channel 1-2: Zero (placeholder) |

## 關鍵特性

1. **正規化內建**：模型內部自動將 0-255 轉換為 0-1，無需外部預處理
2. **Sliding Window 內建**：18 個 patch 的切割與拼接邏輯已嵌入 ONNX 圖中
3. **單次推論**：一張完整影像只需呼叫模型一次

## 效能指標

- 模型大小：108.79 MB
- 推論時間：43.22 ms/image (RTX 5070 Ti)
- 吞吐量：23.1 images/sec
- GPU 記憶體：< 4 GB

## 使用範例

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('background_removal_strip_normalized.onnx',
                                providers=['CUDAExecutionProvider'])

# 輸入: 3 個灰階通道 (976x176), 數值範圍 0-255
input_tensor = np.stack([cur, ref0, ref1], axis=0).astype(np.float32)
input_tensor = np.expand_dims(input_tensor, 0)  # (1, 3, 976, 176)

# 推論
output = session.run(None, {'input': input_tensor})[0]

# 取得 anomaly heatmap
heatmap = output[0, 0, :, :]  # (976, 176)
```

## 部署需求

- Python 3.8+
- onnxruntime-gpu >= 1.23.0
- CUDA 11.x 或 12.x
- GPU 記憶體 >= 4 GB
