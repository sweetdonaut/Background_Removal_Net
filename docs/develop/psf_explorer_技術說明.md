# PSF Explorer 改進說明（技術版）

## 文件目的

本文記錄 PSF Explorer 工具自最初版本起的兩次擴充。讀者預設熟悉 Fourier 光學、PSF 與 Zernike 像差的基本概念。

兩次改進的目的都是**讓模擬出的 PSF 更貼近真實光學系統**，因為我們把這些 PSF 當作「合成缺陷」用於訓練 AI。模擬與實機如果有 systematic 差距，模型在 deployment 時會 fail。

---

## 第 0 版：原始（純量繞射）模型

原始 Explorer 採用純量 Fraunhofer 繞射：

```
PSF(x, y) = | ℱ { M(u, v) · exp[i · φ(u, v)] } |²
```

其中：
- `M(u, v)` ── 瞳孔形狀，原本只支援環形 mask + 橢圓變形
- `φ(u, v)` ── Zernike 像差相位（離焦、像散、彗差、球差、三瓣差）
- `ℱ{·}` ── 二維 FFT

這在低 NA（NA < 0.5）系統下足夠精確。但忽略：
1. 機構性的瞳孔阻擋（非環形遮擋）
2. 偏振 / 向量電磁場效應
3. 高 NA 物鏡的去偏振（depolarization）與 longitudinal field 生成

---

## 第 1 次擴充：瞳孔遮蔽（pupil obstructions）

### 動機

實際儀器的瞳孔常被機構性結構阻擋：光闌邊框、鏡架 spider arm、stop 內的方形遮罩、限制光路的 slit 等。這些遮擋會在 PSF 上產生特定 signature：

- 細長條阻擋 → PSF 出現垂直於該條方向的 sinc 條紋
- 中央方形阻擋 → 環狀繞射 + 方形對稱調制
- Outer crop（保留中央窄帶） → PSF 在窄帶垂直方向被拉長

### 新增 5 個 mask 修飾符

| 參數 | 公式（座標已中心化）|
|------|------------------|
| `square_eps`     | mask = 0 if `\|x\| ≤ ε·R` and `\|y\| ≤ ε·R` |
| `h_stripe_w`     | mask = 0 if `\|y\| ≤ w·R` |
| `v_stripe_w`     | mask = 0 if `\|x\| ≤ w·R` |
| `h_outer_crop`   | mask = 0 if `\|y\| > (1−c)·R` |
| `v_outer_crop`   | mask = 0 if `\|x\| > (1−c)·R` |

`R` 為 outer pupil radius。實作上是 `M_annular · ∏ M_modifier_i` 的 boolean 乘積。計算成本 negligible（純 mask 乘法，無新的 FFT）。

### 物理上仍然是純量模型

這次擴充純粹是把 `M(u, v)` 從「單一 ring」推廣到「ring 加任意機構遮擋」。Fraunhofer 純量近似不變。

---

## 第 2 次擴充：Richards-Wolf 向量繞射

### 動機

NA > 0.5 時純量近似失效。物理上的 issue：

1. Aplanatic lens 把瞳孔上的橫向 plane wave 折射為對應 angle θ 的 spherical wave
2. 折射後 transverse field (Ex, Ey) 相對於 optical axis 不再純 transverse — 出現 longitudinal Ez
3. 不同偏振對應不同的 (Ex, Ey, Ez) 比例 → PSF 形狀依偏振而異
4. 邊緣光線（θ → π/2）的 obliquity factor 必須考慮（apodization √cos θ 來自 energy conservation）

我們的實際系統 NA ≈ 0.95，這些效應全部 non-negligible。

### Richards-Wolf 積分（連續形式）

對於滿足 Abbe sine condition 的 aplanatic lens，焦點附近的 field：

```
E(r, φ_p, z) = -i / λ ·
    ∫∫ √cos θ · A(θ, φ) · R(θ, φ) · p_in(θ, φ) ·
        exp[i · k · (r · sin θ · cos(φ − φ_p) + z · cos θ)] dΩ
```

其中：
- `√cos θ`：apodization（aplanatic lens 對 power 而非 field 做 conservation）
- `A(θ, φ)`：瞳孔 mask
- `R(θ, φ)`：3×3 rotation matrix，把瞳孔平面的偏振投到焦平面 local 座標
- `p_in`：入射偏振向量

### 離散實作

我們在 (u, v) → (x, y) 的 FFT framework 下實作：

```
sin θ = (ρ_pixel / R) · NA          # ρ_pixel = √(u² + v²), 有效 NA 截斷
cos θ = √(1 − sin²θ)
φ     = atan2(v, u)
apod  = √cos θ
```

對每個瞳孔像素 (u, v) 產生 (Ex, Ey, Ez) 三個 complex amplitudes，分別 FFT，再加平方和：

```
PSF(x, y) = |ℱ{U_x}|² + |ℱ{U_y}|² + |ℱ{U_z}|²
```

旋轉矩陣 R 的非零元素（對 isotropic medium、aplanatic case）：

```
R_xx = cos θ · cos²φ + sin²φ
R_xy = (cos θ − 1) · cos φ · sin φ        (= R_yx, symmetric)
R_yy = cos θ · sin²φ + cos²φ
R_zx = -sin θ · cos φ
R_zy = -sin θ · sin φ
```

`R_zz = cos θ` 不需要，因為瞳孔平面的入射 field 是純 transverse（Ez_in = 0）。

### 偏振選項

| 偏振 | p_in(φ) | 焦點 PSF 形狀（高 NA）|
|------|---------|---------------------|
| linX  | (1, 0)              | 沿 X 壓縮，沿 Y 拉長橢圓（depolarization elongation）|
| linY  | (0, 1)              | 與 linX 正交 |
| lin45 | (1, 1)/√2           | 沿 45° 線壓縮 |
| circR | (1, +i)/√2          | 圓對稱（無 preferred direction）|
| circL | (1, −i)/√2          | 圓對稱 |
| radial    | (cos φ, sin φ)  | 中央 \|Ez\|² 增強，產生 "needle beam"（橫向尺寸更小）|
| azimuthal | (−sin φ, cos φ) | 中央 Ez = 0，PSF 為 doughnut |

注意 radial 與 azimuthal 在 high-NA STED、超解析、orbital angular momentum beam 中都是常見實驗工具。

### Apodization 來源

Aplanatic lens 滿足 Abbe sine 條件，幾何上瞳孔 element `dA = ρ dρ dφ` 對應到 angular element `dΩ = sin θ dθ dφ`。能量守恆要求

```
|E_focal|² · dΩ = |E_pupil|² · dA
∴ |E_focal| = |E_pupil| · √(dA/dΩ) = |E_pupil| · √cos θ
```

這個因子壓抑邊緣光線，使焦點能量分佈更集中。

### 計算成本

| | 純量 | 向量 |
|---|---|---|
| FFT 次數 | 1 | 3 |
| Pupil-plane 計算 | scalar mask × phase | 3 個 complex Float64 array (Ux, Uy, Uz) |
| 256×256 in JS（Web）| ~15 ms | ~50 ms |
| 256×256 in Python (numpy) | 7.3 ms | 12.3 ms（**1.7× slower**，非 3× — FFT 不是唯一瓶頸）|

### 與訓練 pipeline 的整合

向量模式以 yaml 開關啟用：

```yaml
vector_mode: true
na: [0.95, 0.95]              # 可指定 [min, max] 區間隨機取樣
pol_type: linX                # 訓練期間固定一種偏振
```

`PsfDefectPool` 在訓練前預生成（`pool_size` 個 PSF），向量模式 1000 個 PSF 約 12 秒。每個訓練 batch 從 pool 隨機抽取，零生成成本。

### 注意事項：azimuthal 偏振

Azimuthal 產生 doughnut PSF（中央理論為零）。我們的後處理 pipeline `clean_connected_peak` 仍會把整個環抓成單一 connected component，bbox crop 後得到「中央有空洞的方形 defect」。這個 defect shape 對訓練是否合理尚未驗證 ── 程式碼與 yaml 範本都標註「avoid until validated」。

---

## 何時應該用向量模式

| 系統 NA | 純量模型誤差 | 建議 |
|---------|-------------|------|
| < 0.3   | < 1% | 純量足夠 |
| 0.3 – 0.7 | 1% – 10% | 純量可以 quick check，正式請用向量 |
| > 0.7   | 10% 以上 | **必須**用向量 |
| > 1.0（油浸）| 顯著 | **必須**用向量，且需考慮浸油折射率對 sin θ 的修正（目前未實作）|

我們的系統 NA ≈ 0.95，所以正式訓練都應使用向量模式。

---

## 程式碼對應

| 元件 | 檔案位置 |
|------|---------|
| Web 版實作 | `psf-explorer-app/src/App.jsx`（包含純量 + 向量兩個 path）|
| Python 端 mask 修飾符 | `src_core/generate_psf.py:generate_one()` |
| Python 端向量計算 | `src_core/generate_psf.py:_build_vector_pupil()` |
| Yaml 範本（向量）| `src_core/defects/type4_vector.yaml` |

## 參考文獻

- Richards & Wolf (1959), *Electromagnetic Diffraction in Optical Systems II*, Proc. Roy. Soc. A 253:358
- Born & Wolf, *Principles of Optics*, §8.8（Pupil function 與 Fourier 光學）
- Novotny & Hecht, *Principles of Nano-Optics*, Ch. 3（Vector field at high NA focus）
