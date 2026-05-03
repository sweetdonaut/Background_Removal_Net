import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════ FFT ═══════════════════════ */
function fft1d(re, im, invert) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (invert ? -1 : 1);
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cRe = 1, cIm = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = a + half;
        const tRe = re[b] * cRe - im[b] * cIm, tIm = re[b] * cIm + im[b] * cRe;
        re[b] = re[a] - tRe; im[b] = im[a] - tIm;
        re[a] += tRe; im[a] += tIm;
        const nRe = cRe * wRe - cIm * wIm; cIm = cRe * wIm + cIm * wRe; cRe = nRe;
      }
    }
  }
  if (invert) { for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}
function fft2d(re, im, N) {
  const r = new Float64Array(N), ri = new Float64Array(N);
  for (let y = 0; y < N; y++) {
    const o = y * N;
    for (let x = 0; x < N; x++) { r[x] = re[o + x]; ri[x] = im[o + x]; }
    fft1d(r, ri, false);
    for (let x = 0; x < N; x++) { re[o + x] = r[x]; im[o + x] = ri[x]; }
  }
  const c = new Float64Array(N), ci = new Float64Array(N);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) { c[y] = re[y * N + x]; ci[y] = im[y * N + x]; }
    fft1d(c, ci, false);
    for (let y = 0; y < N; y++) { re[y * N + x] = c[y]; im[y * N + x] = ci[y]; }
  }
}
function fftShift(a, N) {
  const o = new Float64Array(N * N), h = N >> 1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    o[y * N + x] = a[((y + h) % N) * N + ((x + h) % N)];
  return o;
}

/* ═══════════════════ COLORMAP (inferno) ═══════════════════ */
const CMAP = [[0,0,4],[10,7,34],[28,16,68],[49,17,96],[72,12,104],[94,10,103],[115,15,97],[136,24,86],[155,37,72],[173,53,56],[189,71,40],[203,93,24],[214,117,10],[222,143,3],[227,170,18],[229,197,50],[228,224,93],[240,249,149],[252,255,164]];
function cmap(val) {
  const t = Math.max(0, Math.min(1, val)) * (CMAP.length - 1);
  const i = Math.floor(t), f = t - i;
  const a = CMAP[Math.min(i, CMAP.length - 1)], b = CMAP[Math.min(i + 1, CMAP.length - 1)];
  return [Math.round(a[0]+(b[0]-a[0])*f), Math.round(a[1]+(b[1]-a[1])*f), Math.round(a[2]+(b[2]-a[2])*f)];
}

/* ═══════════════════ PRNG ═══════════════════ */
function mulberry32(a) {
  return function() { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function boxMullerRng(rng) {
  return function() { let u, v, s; do { u = 2*rng()-1; v = 2*rng()-1; s = u*u+v*v; } while (s >= 1 || s === 0); return u * Math.sqrt(-2 * Math.log(s) / s); };
}
function poissonSample(lam, rng) {
  if (lam < 30) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= rng(); } while (p > L); return k - 1; }
  const g = boxMullerRng(rng); return Math.max(0, Math.round(lam + Math.sqrt(lam) * g()));
}

/* ═══════════════════ RENDER HELPERS ═══════════════════ */
function renderArr(ctx, data, N, useLog) {
  let arr = data;
  if (useLog) { const mx = Math.max(...data), fl = mx*1e-6; arr = data.map(v => Math.log10(Math.max(v, fl))); }
  let mn = Infinity, mx2 = -Infinity;
  for (let i = 0; i < arr.length; i++) { if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx2) mx2 = arr[i]; }
  const rng = mx2 - mn || 1, img = ctx.createImageData(N, N);
  for (let i = 0; i < N*N; i++) {
    const [r, g, b] = cmap((arr[i]-mn)/rng);
    img.data[i*4]=r; img.data[i*4+1]=g; img.data[i*4+2]=b; img.data[i*4+3]=255;
  }
  ctx.putImageData(img, 0, 0);
}
function renderMask(ctx, mask, N) {
  const img = ctx.createImageData(N, N);
  for (let i = 0; i < N*N; i++) {
    const v = mask[i] > 0;
    img.data[i*4] = v ? 220 : 18; img.data[i*4+1] = v ? 220 : 18;
    img.data[i*4+2] = v ? 240 : 24; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function renderPhase(ctx, phase, mask, N) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < N*N; i++) if (mask[i] > 0) { if (phase[i]<mn) mn=phase[i]; if (phase[i]>mx) mx=phase[i]; }
  const rng = mx-mn||1, img = ctx.createImageData(N, N);
  for (let i = 0; i < N*N; i++) {
    if (mask[i] > 0) {
      const t = (phase[i]-mn)/rng; let r,g,b;
      if (t<0.5) { const s=t*2; r=Math.round(30+225*s); g=Math.round(60+195*s); b=Math.round(200+55*s); }
      else { const s=(t-0.5)*2; r=255; g=Math.round(255-200*s); b=Math.round(255-210*s); }
      img.data[i*4]=r; img.data[i*4+1]=g; img.data[i*4+2]=b; img.data[i*4+3]=255;
    } else { img.data[i*4]=18; img.data[i*4+1]=18; img.data[i*4+2]=24; img.data[i*4+3]=255; }
  }
  ctx.putImageData(img, 0, 0);
}
function renderCropped(ctx, data, N, zoom, useLog) {
  const cs = Math.floor(N/zoom), h = cs>>1, ctr = N>>1;
  const crop = new Float64Array(cs*cs);
  for (let y=0;y<cs;y++) for (let x=0;x<cs;x++) crop[y*cs+x]=data[(y+ctr-h)*N+(x+ctr-h)];
  const tmp = document.createElement("canvas"); tmp.width=cs; tmp.height=cs;
  const tc = tmp.getContext("2d"); renderArr(tc, crop, cs, useLog);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, N, N);
}

/* ═══════════════════ UI COMPONENTS ═══════════════════ */
function Slider({label, sub, value, min, max, step, onChange}) {
  return (<div style={{marginBottom:12}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:3}}>
      <div><span style={{color:"#e0e0e0",fontSize:12.5,fontWeight:600}}>{label}</span>
        {sub && <span style={{color:"#555",fontSize:10.5,marginLeft:5}}>{sub}</span>}</div>
      <span style={{color:"#fbbf24",fontSize:12.5,fontFamily:"'DM Mono',monospace",fontWeight:500}}>
        {value.toFixed(step<0.1?2:step<1?1:0)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e=>onChange(parseFloat(e.target.value))}
      style={{width:"100%",accentColor:"#fbbf24",cursor:"pointer",height:4}} />
  </div>);
}
function Section({title, sub, children, open: initOpen = true, color = "#aaa"}) {
  const [open, setOpen] = useState(initOpen);
  return (<div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",paddingBottom:12,marginBottom:12}}>
    <div onClick={()=>setOpen(!open)} style={{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",userSelect:"none",marginBottom:open?10:0}}>
      <div><span style={{color,fontSize:11,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.8px"}}>{title}</span>
        {sub && <span style={{color:"#444",fontSize:10,marginLeft:6}}>{sub}</span>}</div>
      <span style={{color:"#444",fontSize:10,transition:"transform 0.2s",transform:open?"rotate(0)":"rotate(-90deg)"}}>▼</span>
    </div>
    {open && children}
  </div>);
}

/* ═══════════════════ MAIN ═══════════════════ */
const N = 256;

export default function PSFExplorer() {
  // Aperture
  const [outerR, setOuterR] = useState(40);
  const [epsilon, setEpsilon] = useState(0.6);
  const [ellipticity, setEllipticity] = useState(0);
  const [ellipAngle, setEllipAngle] = useState(0);

  // Stripe masks
  const [hStripeW, setHStripeW] = useState(0);
  const [vStripeW, setVStripeW] = useState(0);
  const [squareEps, setSquareEps] = useState(0);
  const [hOuterCrop, setHOuterCrop] = useState(0);
  const [vOuterCrop, setVOuterCrop] = useState(0);

  // Aberrations
  const [defocus, setDefocus] = useState(0);
  const [astigX, setAstigX] = useState(0);
  const [astigY, setAstigY] = useState(0);
  const [comaX, setComaX] = useState(0);
  const [comaY, setComaY] = useState(0);
  const [spherical, setSpherical] = useState(0);
  const [trefoilX, setTrefoilX] = useState(0);
  const [trefoilY, setTrefoilY] = useState(0);

  // Noise
  const [brightness, setBrightness] = useState(5000);
  const [background, setBackground] = useState(5);
  const [poissonOn, setPoissonOn] = useState(false);
  const [gaussNoise, setGaussNoise] = useState(1.5);
  const [noiseSeed, setNoiseSeed] = useState(42);

  // Display
  const [useLog, setUseLog] = useState(false);
  const [zoom, setZoom] = useState(4);

  const apertureRef = useRef(null);
  const phaseRef = useRef(null);
  const psfRef = useRef(null);

  const compute = useCallback(() => {
    const cx = N / 2, cy = N / 2;
    const innerR = outerR * epsilon;

    // Build mask
    const mask = new Float64Array(N * N);
    const cosA = Math.cos(ellipAngle * Math.PI / 180), sinA = Math.sin(ellipAngle * Math.PI / 180);
    const sX = 1 + ellipticity, sY = 1 - ellipticity;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = x - cx, dy = y - cy;
      const rx = (dx * cosA + dy * sinA) / sX, ry = (-dx * sinA + dy * cosA) / sY;
      const r = Math.sqrt(rx * rx + ry * ry);
      mask[y * N + x] = (r <= outerR && r >= innerR) ? 1 : 0;
    }

    // Apply stripe masks
    if (hStripeW > 0) {
      const halfH = outerR * hStripeW;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.abs(y - cy) <= halfH) mask[y * N + x] = 0;
      }
    }
    if (vStripeW > 0) {
      const halfV = outerR * vStripeW;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.abs(x - cx) <= halfV) mask[y * N + x] = 0;
      }
    }
    if (squareEps > 0) {
      const halfSq = outerR * squareEps;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.abs(x - cx) <= halfSq && Math.abs(y - cy) <= halfSq) mask[y * N + x] = 0;
      }
    }
    if (hOuterCrop > 0) {
      const threshold = outerR * (1 - hOuterCrop);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.abs(y - cy) > threshold) mask[y * N + x] = 0;
      }
    }
    if (vOuterCrop > 0) {
      const threshold = outerR * (1 - vOuterCrop);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.abs(x - cx) > threshold) mask[y * N + x] = 0;
      }
    }

    // Build phase
    const phase = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = (x - cx) / outerR, dy = (y - cy) / outerR;
      const rho2 = dx * dx + dy * dy, rho = Math.sqrt(rho2), th = Math.atan2(dy, dx);
      let p = 0;
      p += defocus * (2 * rho2 - 1);
      p += astigX * rho2 * Math.cos(2 * th);
      p += astigY * rho2 * Math.sin(2 * th);
      p += comaX * (3 * rho2 - 2) * rho * Math.cos(th);
      p += comaY * (3 * rho2 - 2) * rho * Math.sin(th);
      p += spherical * (6 * rho2 * rho2 - 6 * rho2 + 1);
      p += trefoilX * rho2 * rho * Math.cos(3 * th);
      p += trefoilY * rho2 * rho * Math.sin(3 * th);
      phase[y * N + x] = p;
    }

    // Pupil function → FFT → |.|²
    const re = new Float64Array(N * N), im = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) if (mask[i] > 0) {
      re[i] = Math.cos(phase[i]); im[i] = Math.sin(phase[i]);
    }
    fft2d(re, im, N);
    const psf = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) psf[i] = re[i] * re[i] + im[i] * im[i];
    const shifted = fftShift(psf, N);

    // Brightness + noise
    let sum = 0;
    for (let i = 0; i < N * N; i++) sum += shifted[i];
    const scale = sum > 0 ? brightness / sum : 1;
    const final = new Float64Array(N * N);
    const rng = mulberry32(noiseSeed);
    const gauss = boxMullerRng(rng);
    for (let i = 0; i < N * N; i++) {
      let v = shifted[i] * scale + background;
      if (poissonOn) v = poissonSample(Math.max(0, v), rng);
      if (gaussNoise > 0) v += gauss() * gaussNoise;
      final[i] = Math.max(0, v);
    }

    // Render
    const apCtx = apertureRef.current?.getContext("2d");
    const phCtx = phaseRef.current?.getContext("2d");
    const psCtx = psfRef.current?.getContext("2d");
    if (apCtx) renderMask(apCtx, mask, N);
    if (phCtx) renderPhase(phCtx, phase, mask, N);
    if (psCtx) renderCropped(psCtx, final, N, zoom, useLog);
  }, [outerR, epsilon, ellipticity, ellipAngle, hStripeW, vStripeW, squareEps, hOuterCrop, vOuterCrop, defocus, astigX, astigY, comaX, comaY, spherical, trefoilX, trefoilY, brightness, background, poissonOn, gaussNoise, noiseSeed, useLog, zoom]);

  useEffect(() => { compute(); }, [compute]);

  const hasAberrations = defocus !== 0 || astigX !== 0 || astigY !== 0 || comaX !== 0 || comaY !== 0 || spherical !== 0 || trefoilX !== 0 || trefoilY !== 0;

  const resetAll = () => {
    setOuterR(40); setEpsilon(0.6); setEllipticity(0); setEllipAngle(0);
    setHStripeW(0); setVStripeW(0); setSquareEps(0);
    setHOuterCrop(0); setVOuterCrop(0);
    setDefocus(0); setAstigX(0); setAstigY(0); setComaX(0); setComaY(0);
    setSpherical(0); setTrefoilX(0); setTrefoilY(0);
    setBrightness(5000); setBackground(5); setPoissonOn(false); setGaussNoise(1.5); setNoiseSeed(42);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(145deg, #0a0a10 0%, #12121c 50%, #0d0d18 100%)",
      color: "#e0e0e0",
      fontFamily: "'DM Sans', 'Noto Sans TC', system-ui, sans-serif",
      padding: "20px 16px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet" />

      <div style={{ textAlign: "center", marginBottom: 22 }}>
        <h1 style={{
          fontSize: 24, fontWeight: 700, margin: 0,
          background: "linear-gradient(90deg, #fbbf24, #f59e0b, #d97706)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>環形光圈 PSF 探索器</h1>
        <p style={{ color: "#555", fontSize: 12, margin: "4px 0 0" }}>
          PSF = | FT{'{'} M · exp(iφ) {'}'} |²
        </p>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 20, maxWidth: 1000, margin: "0 auto", justifyContent: "center" }}>
        {/* ═══ CONTROLS ═══ */}
        <div style={{
          flex: "1 1 290px", maxWidth: 330,
          background: "rgba(255,255,255,0.03)", borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.06)",
          padding: "16px 16px", maxHeight: "calc(100vh - 100px)", overflowY: "auto",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0, color: "#ccc" }}>參數控制</h2>
            <button onClick={resetAll} style={{
              background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.25)",
              color: "#fbbf24", fontSize: 10.5, padding: "3px 10px", borderRadius: 6, cursor: "pointer", fontFamily: "inherit",
            }}>重設全部</button>
          </div>

          {/* Aperture */}
          <Section title="環形光圈" sub="Aperture" color="#8bb4f0">
            <Slider label="外圈半徑" sub="outer R" value={outerR} min={15} max={80} step={1} onChange={setOuterR} />
            <Slider label="圓形遮擋 ε" sub="circle obstruction" value={epsilon} min={0} max={0.95} step={0.01} onChange={setEpsilon} />
            <Slider label="方形遮擋 ε" sub="square obstruction" value={squareEps} min={0} max={0.95} step={0.01} onChange={setSquareEps} />
            <Slider label="橢圓度" sub="ellipticity" value={ellipticity} min={0} max={0.3} step={0.01} onChange={setEllipticity} />
            <Slider label="橢圓角度" sub="degrees" value={ellipAngle} min={0} max={180} step={1} onChange={setEllipAngle} />
            <Slider label="水平遮擋條" sub="horizontal" value={hStripeW} min={0} max={1} step={0.01} onChange={setHStripeW} />
            <Slider label="垂直遮擋條" sub="vertical" value={vStripeW} min={0} max={1} step={0.01} onChange={setVStripeW} />
            <Slider label="上下外側裁切" sub="H outer crop" value={hOuterCrop} min={0} max={1} step={0.01} onChange={setHOuterCrop} />
            <Slider label="左右外側裁切" sub="V outer crop" value={vOuterCrop} min={0} max={1} step={0.01} onChange={setVOuterCrop} />
          </Section>

          {/* Aberrations */}
          <Section title="像差" sub="Aberrations" open={false} color="#a78bfa">
            <Slider label="離焦" sub="defocus" value={defocus} min={-6} max={6} step={0.1} onChange={setDefocus} />
            <Slider label="像散 X" sub="astig cos2θ" value={astigX} min={-5} max={5} step={0.1} onChange={setAstigX} />
            <Slider label="像散 Y" sub="astig sin2θ" value={astigY} min={-5} max={5} step={0.1} onChange={setAstigY} />
            <Slider label="彗差 X" sub="coma cosθ" value={comaX} min={-5} max={5} step={0.1} onChange={setComaX} />
            <Slider label="彗差 Y" sub="coma sinθ" value={comaY} min={-5} max={5} step={0.1} onChange={setComaY} />
            <Slider label="球差" sub="spherical" value={spherical} min={-5} max={5} step={0.1} onChange={setSpherical} />
            <Slider label="三瓣差 X" sub="trefoil cos3θ" value={trefoilX} min={-5} max={5} step={0.1} onChange={setTrefoilX} />
            <Slider label="三瓣差 Y" sub="trefoil sin3θ" value={trefoilY} min={-5} max={5} step={0.1} onChange={setTrefoilY} />
          </Section>

          {/* Noise */}
          <Section title="成像與雜訊" sub="Imaging & Noise" open={false} color="#34d399">
            <Slider label="亮度" sub="photons" value={brightness} min={10} max={10000} step={10} onChange={setBrightness} />
            <Slider label="背景值" sub="background" value={background} min={0} max={200} step={1} onChange={setBackground} />
            <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", marginBottom: 12 }}>
              <input type="checkbox" checked={poissonOn} onChange={e => setPoissonOn(e.target.checked)} style={{ accentColor: "#34d399" }} />
              <span style={{ color: "#e0e0e0", fontSize: 12.5, fontWeight: 600 }}>Poisson noise</span>
              <span style={{ color: "#555", fontSize: 10.5 }}>光子計數雜訊</span>
            </label>
            <Slider label="Gaussian σ" sub="讀取雜訊" value={gaussNoise} min={0} max={50} step={0.5} onChange={setGaussNoise} />
            <button onClick={() => setNoiseSeed(Math.floor(Math.random() * 100000))} style={{
              background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)",
              color: "#34d399", fontSize: 10.5, padding: "4px 12px", borderRadius: 6, cursor: "pointer", fontFamily: "inherit", marginTop: 4,
            }}>🎲 重新取樣雜訊</button>
          </Section>
        </div>

        {/* ═══ VISUALIZATIONS ═══ */}
        <div style={{ flex: "1 1 360px", maxWidth: 580 }}>
          {/* Top: aperture + phase */}
          <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
            <div style={{ flex: 1 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, border: "1px solid rgba(255,255,255,0.06)", padding: 8, textAlign: "center" }}>
                <div style={{ fontSize: 10.5, color: "#666", marginBottom: 5, fontWeight: 500 }}>光圈形狀 Aperture</div>
                <canvas ref={apertureRef} width={N} height={N} style={{ width: "100%", aspectRatio: "1", borderRadius: 6, imageRendering: "pixelated" }} />
              </div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, border: "1px solid rgba(255,255,255,0.06)", padding: 8, textAlign: "center" }}>
                <div style={{ fontSize: 10.5, color: "#666", marginBottom: 5, fontWeight: 500 }}>
                  相位圖 Phase {hasAberrations ? "" : "(無像差)"}
                </div>
                <canvas ref={phaseRef} width={N} height={N} style={{ width: "100%", aspectRatio: "1", borderRadius: 6, imageRendering: "pixelated" }} />
              </div>
            </div>
          </div>

          {/* PSF result */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, border: "1px solid rgba(255,255,255,0.06)", padding: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontSize: 13, color: "#ccc", fontWeight: 600 }}>PSF 結果</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", fontSize: 10.5, color: "#888" }}>
                  <input type="checkbox" checked={useLog} onChange={e => setUseLog(e.target.checked)} style={{ accentColor: "#fbbf24" }} />
                  Log
                </label>
                <select value={zoom} onChange={e => setZoom(Number(e.target.value))} style={{
                  background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                  color: "#ccc", fontSize: 10.5, padding: "2px 5px", borderRadius: 5, cursor: "pointer",
                }}>
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                  <option value={8}>8×</option>
                </select>
              </div>
            </div>
            <canvas ref={psfRef} width={N} height={N} style={{ width: "100%", aspectRatio: "1", borderRadius: 6, imageRendering: "pixelated" }} />
          </div>

          {/* Tips */}
          <div style={{
            marginTop: 10, padding: "10px 14px", borderRadius: 8,
            background: "rgba(139,180,240,0.04)", border: "1px solid rgba(139,180,240,0.1)",
            fontSize: 11.5, color: "#888", lineHeight: 1.6,
          }}>
            <strong style={{ color: "#8bb4f0" }}>使用提示：</strong>
            先把像差歸零，只調光圈參數。接著一次調一種像差觀察效果。最後打開雜訊，比較模擬結果與真實影像。
          </div>
        </div>
      </div>
    </div>
  );
}
