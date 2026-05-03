# PSF Explorer

互動式 PSF（點擴散函數）參數視覺化工具，用來在訓練前挑選想合成的 defect 形態，把調好的參數寫進 `src_core/defects/*.yaml`。

物理模型對應 `src_core/generate_psf.py`：環形光圈 + Zernike 像差 → FFT → |·|² → Poisson + Gaussian 雜訊。

## 環境需求

- Node.js ≥ 18（開發用 v22）

```bash
node --version
```

## 啟動

```bash
cd psf-explorer-app

# 首次 clone / pull 後執行（package-lock 變動時也要重跑）
npm install

# 本機開發（http://localhost:5173/）
npm run dev

# 區網其他裝置也想連線時加 --host
npm run dev -- --host
```

## 其他指令

```bash
npm run build       # 產生 production bundle 到 dist/
npm run preview     # 本機預覽 build 結果
npm run lint        # ESLint
```

## 主要檔案

- `src/App.jsx` — PSFExplorer 主元件（FFT、UI、所有計算）
- `src/index.css` — 最小化 reset
- `src/main.jsx` — React entry

## 對應到 Python 端的參數

UI 上的參數一對一對應到 `src_core/defects/*.yaml` 的 key：

| 網站 UI | Yaml key |
|---------|---------|
| 外圈半徑 | `outer_r` |
| 圓形遮擋 ε | `epsilon` |
| 方形遮擋 ε | `square_eps` |
| 橢圓度 / 角度 | `ellipticity` / `ellip_angle` |
| 水平遮擋條 | `h_stripe_w` |
| 垂直遮擋條 | `v_stripe_w` |
| 上下外側裁切 | `h_outer_crop` |
| 左右外側裁切 | `v_outer_crop` |
| 離焦 / 像散 / 彗差 / 球差 / 三瓣差 | `defocus` / `astig_x,y` / `coma_x,y` / `spherical` / `trefoil_x,y` |
| 亮度 / 背景 / Poisson / Gaussian σ | `brightness` / `background` / `poisson_noise` / `gaussian_sigma` |

Yaml 中 `[a, b]` 形式表示訓練時從區間隨機取樣，scalar 表示固定值。`type3.yaml` 是包含全部新 key 的範本。
