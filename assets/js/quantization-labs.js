/* 博客《LLM 与 AIGC 量化》的交互实验。改编自量化讲义的脚本：只保留通用工具和 10 个交互实验，
 * 样式限定在 .qz-lab 下，配色跟随站点的浅色 / 深色主题。外层 div 同时带 qz-card：站点的 no_defer.js
 * 不会把 class 含 card 的容器里的表格转成 bootstrap-table，这样实验里由脚本生成的表格保持原样。 */
(function () {
  "use strict";
  /* ================= 样式 ================= */
  (function injectStyle() {
    if (document.getElementById("qz-lab-style")) return;
    const st = document.createElement("style");
    st.id = "qz-lab-style";
    st.textContent = `.qz-lab{--paper:var(--global-bg-color);--paper-2:var(--global-card-bg-color);--paper-3:var(--global-code-bg-color);
  --ink:var(--global-text-color);--ink-2:var(--global-text-color-light);--ink-3:#7b8597;--rule:var(--global-divider-color);
  --accent:#2457c5;--accent-2:#e7edfb;--accent-ink:#1a3e8c;--warm:#bf5a0c;--warm-2:#fcefe2;
  --good:#12784a;--good-2:#e2f3e9;--violet:#6340b8;--violet-2:#efe9fa;
  --f-body:inherit;--f-mono:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
  border:1px solid var(--rule);border-radius:8px;background:var(--paper-2);color:var(--ink);margin:26px 0;overflow:hidden;font-size:15px;line-height:1.7}
html[data-theme="dark"] .qz-lab{--ink-3:#8a94a6;--accent:#7ea3ff;--accent-2:#1a2742;--accent-ink:#bcd0ff;--warm:#f2a45e;--warm-2:#2d2016;
  --good:#5fcf8e;--good-2:#13281c;--violet:#b7a0f6;--violet-2:#221b37}
.qz-lab .box-h{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--rule);font-weight:700;font-size:15px}
.qz-lab .tag{font-family:var(--f-mono);font-size:11px;letter-spacing:.1em;padding:2px 8px;border-radius:4px;font-weight:500;background:var(--good-2);color:var(--good)}
.qz-lab .box-b{padding:16px}
.qz-lab .box-b > :last-child{margin-bottom:0}
.qz-lab p{margin:0 0 14px}
.qz-lab ul{padding-left:1.4em;margin:0 0 10px}
.qz-lab li{margin:3px 0}
.qz-lab .small{font-size:13.5px}
.qz-lab .mute{color:var(--ink-2)}
.qz-lab sup,.qz-lab sub{font-size:.72em;line-height:0;position:relative;vertical-align:baseline}
.qz-lab sup{top:-.48em} .qz-lab sub{bottom:-.22em}
.qz-lab .hl-w{color:var(--warm);font-weight:700} .qz-lab .hl-a{color:var(--accent);font-weight:700} .qz-lab .hl-g{color:var(--good);font-weight:700}
.qz-lab .mono{font-family:var(--f-mono);font-size:13px;font-variant-ligatures:none}
.qz-lab .fig{background:var(--paper-2);border:1px solid var(--rule);border-radius:8px;padding:8px;overflow-x:auto;margin:10px 0}
.qz-lab .fig svg{display:block;margin:0 auto;max-width:100%;height:auto}
.qz-lab .fig svg.wide{min-width:620px}
.qz-lab .tw{overflow-x:auto;margin:10px 0}
.qz-lab table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px;width:100%;margin:0;color:var(--ink);background:transparent}
.qz-lab th,.qz-lab td{padding:6px 10px;border:none;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;font-size:14px;background:transparent;color:inherit}
.qz-lab th{font-weight:700;color:var(--ink-2);font-size:13px;border-bottom:1.5px solid var(--ink-3);background:var(--paper-3)}
.qz-lab td.l,.qz-lab th.l{text-align:left;white-space:normal}
.qz-lab td.mono{font-family:var(--f-mono);font-size:13px}
.qz-lab td.hw{background:var(--warm-2);color:var(--warm);font-weight:700}
.qz-lab td.ha{background:var(--accent-2);color:var(--accent-ink);font-weight:700}
.qz-lab td.hg{background:var(--good-2);color:var(--good);font-weight:700}
.qz-lab svg text{font-family:var(--f-mono);font-size:11px;fill:var(--ink-2);font-variant-ligatures:none}
.qz-lab svg .tt{font-family:var(--f-body);font-size:12px;fill:var(--ink)}
.qz-lab svg .tb{font-family:var(--f-body);font-size:13px;font-weight:700;fill:var(--ink)}
.qz-lab svg .ta{fill:var(--accent)} .qz-lab svg .tw{fill:var(--warm)} .qz-lab svg .tg{fill:var(--good)} .qz-lab svg .tv{fill:var(--violet)} .qz-lab svg .t3{fill:var(--ink-3)}
.qz-lab svg .ax{stroke:var(--ink-3);stroke-width:1;fill:none}
.qz-lab svg .gl{stroke:var(--rule);stroke-width:1;fill:none}
.qz-lab svg .ln-a{stroke:var(--accent);stroke-width:2;fill:none}
.qz-lab svg .ln-w{stroke:var(--warm);stroke-width:2;fill:none}
.qz-lab svg .ln-g{stroke:var(--good);stroke-width:2;fill:none}
.qz-lab svg .ln-v{stroke:var(--violet);stroke-width:2;fill:none}
.qz-lab svg .ln-i{stroke:var(--ink-2);stroke-width:1.3;fill:none}
.qz-lab svg .dash{stroke-dasharray:4 4}
.qz-lab svg .bx{fill:var(--paper-2);stroke:var(--ink-3);stroke-width:1.2}
.qz-lab svg .bx-a{fill:var(--accent-2);stroke:var(--accent);stroke-width:1.3}
.qz-lab svg .bx-w{fill:var(--warm-2);stroke:var(--warm);stroke-width:1.3}
.qz-lab svg .bx-g{fill:var(--good-2);stroke:var(--good);stroke-width:1.3}
.qz-lab svg .bx-v{fill:var(--violet-2);stroke:var(--violet);stroke-width:1.3}
.qz-lab svg .bx-3{fill:var(--paper-3);stroke:var(--rule);stroke-width:1}
.qz-lab svg .f-a{fill:var(--accent)} .qz-lab svg .f-w{fill:var(--warm)} .qz-lab svg .f-g{fill:var(--good)} .qz-lab svg .f-v{fill:var(--violet)}
.qz-lab svg .f-3{fill:var(--ink-3)} .qz-lab svg .f-p{fill:var(--paper-2)} .qz-lab svg .f-r{fill:var(--rule)}
.qz-lab .ctrl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 12px;font-size:14px}
.qz-lab .ctrl label{display:inline-flex;align-items:center;gap:8px;color:var(--ink-2);margin:0}
.qz-lab .ctrl input[type=text],.qz-lab .ctrl input[type=number]{font-family:var(--f-mono);font-size:13px;padding:5px 8px;border:1px solid var(--rule);border-radius:6px;background:var(--paper);color:var(--ink);min-width:0}
.qz-lab .ctrl input[type=text]{width:min(100%,430px)}
.qz-lab .ctrl input[type=number]{width:120px}
.qz-lab .ctrl select{font-family:var(--f-body);font-size:13.5px;padding:4px 8px;border:1px solid var(--rule);border-radius:6px;background:var(--paper);color:var(--ink)}
.qz-lab .ctrl input[type=range]{width:160px;accent-color:var(--accent)}
.qz-lab .ctrl output{font-family:var(--f-mono);font-size:13px;color:var(--ink);min-width:3.5em;display:inline-block}
.qz-lab .btn{font-family:var(--f-body);font-size:14px;font-weight:500;line-height:1.5;border:1px solid var(--accent);color:var(--accent);background:transparent;padding:4px 12px;border-radius:6px;cursor:pointer;box-shadow:none}
.qz-lab .btn:hover{background:var(--accent-2);color:var(--accent)}
.qz-lab .btn.pri{background:var(--accent);color:var(--paper-2)}
.qz-lab .btn.pri:hover{filter:brightness(1.08)}
.qz-lab .btn:disabled{opacity:.45;cursor:not-allowed}
.qz-lab .stats{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.qz-lab .stat{display:inline-flex;gap:8px;align-items:baseline;border:1px solid var(--rule);border-radius:6px;padding:3px 10px;background:var(--paper);font-size:13px;color:var(--ink-2)}
.qz-lab .stat b{font-family:var(--f-mono);font-weight:700;color:var(--ink)}
.qz-lab .stat.w b{color:var(--warm)} .qz-lab .stat.g b{color:var(--good)} .qz-lab .stat.a b{color:var(--accent)}
.qz-lab .grid2{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
@media (min-width:720px){.qz-lab .grid2{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}}
.qz-lab .grid2.one{grid-template-columns:minmax(0,1fr)}
.qz-lab #fmt-out td:last-child{min-width:6em}
.qz-lab .panel{min-width:0;border:1px solid var(--rule);border-radius:6px;padding:10px;background:var(--paper)}
.qz-lab .panel h5{margin:0 0 6px;font-size:13px;line-height:1.5;color:var(--ink-2);font-weight:700}
.qz-lab .panel svg{width:100%;height:auto;display:block}
.qz-lab .bits{display:inline-flex;flex-wrap:wrap;gap:2px;font-family:var(--f-mono);font-size:12px}
.qz-lab .bits i{font-style:normal;display:inline-block;width:15px;text-align:center;border-radius:2px;padding:1px 0;line-height:1.5}
.qz-lab .bits i.s{background:var(--warm-2);color:var(--warm)}
.qz-lab .bits i.e{background:var(--accent-2);color:var(--accent-ink)}
.qz-lab .bits i.m{background:var(--good-2);color:var(--good)}
.qz-lab .bits i.n{background:var(--paper-3);color:var(--ink)}`;
    document.head.appendChild(st);
  })();

  /* ================= 通用工具 ================= */
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(n) {
    while (n.firstChild) n.removeChild(n.firstChild);
  }
  // Unicode 上标字符分属不同区块，网页字体常常缺字而回退到系统字体，导致高低不齐；在 SVG 里改用 tspan 排版
  function setSup(t, str) {
    const map = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "−", "⁺": "+" };
    str.split(/([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)/).forEach((p, i) => {
      if (!p) return;
      if (i % 2) {
        el(
          "tspan",
          { dy: -4, "font-size": "0.72em" },
          t,
          p
            .split("")
            .map((c) => map[c])
            .join("")
        );
        el("tspan", { dy: 4 }, t, "\u200b");
      } else el("tspan", {}, t, p);
    });
    return t;
  }
  function rne(x) {
    // round half to even
    const f = Math.floor(x),
      d = x - f;
    if (d === 0.5) return f % 2 === 0 ? f : f + 1;
    return Math.round(x);
  }
  function fmt(x, d) {
    if (d == null) d = 3;
    if (!isFinite(x)) return x > 0 ? "∞" : x < 0 ? "−∞" : "NaN";
    if (Math.abs(x) < 0.5 * Math.pow(10, -d)) x = 0;
    return x.toFixed(d).replace("-", "−");
  }
  function fmtE(x) {
    if (x === 0) return "0";
    const a = Math.abs(x);
    if (a >= 0.01 && a < 1e4) return fmt(x, a < 1 ? 4 : 3);
    return x.toExponential(2).replace("-", "−");
  }
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussFn(rng) {
    return function () {
      let u = 0,
        v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
  }
  function stat(label, value, cls) {
    return '<span class="stat ' + (cls || "") + '">' + label + " <b>" + value + "</b></span>";
  }
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const copyM = (M) => M.map((r) => r.slice());
  function matmul(A, B) {
    // A: r×k, B: k×c
    const r = A.length,
      k = B.length,
      c = B[0].length,
      C = zeros(r, c);
    for (let i = 0; i < r; i++)
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        const Bp = B[p],
          Ci = C[i];
        for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
      }
    return C;
  }
  const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
  function inv(A) {
    const n = A.length,
      M = A.map((r, i) => r.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++)
        if (r !== c) {
          const f = M[r][c];
          if (f !== 0) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
        }
    }
    return M.map((r) => r.slice(n));
  }
  function cholLower(A) {
    const n = A.length,
      L = zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
      }
    return L;
  }
  const sumSq = (M) => M.reduce((a, r) => a + r.reduce((b, v) => b + v * v, 0), 0);
  function sub(A, B) {
    return A.map((r, i) => r.map((v, j) => v - B[i][j]));
  }

  /* 简单的柱状图 / 折线图 / 热力图 */
  function barChart(svg, vals, o) {
    clear(svg);
    const W = o.W || 320,
      H = o.H || 170,
      L = o.L || 34,
      R = 8,
      Tp = 10,
      B = o.B || 22;
    const n = vals.length,
      pw = W - L - R,
      ph = H - Tp - B;
    let ymin = o.ymin != null ? o.ymin : 0,
      ymax = o.ymax != null ? o.ymax : Math.max(...vals.map((v) => Math.abs(v))) || 1;
    const tf = o.log ? (v) => Math.log10(Math.max(v, 1e-12)) : (v) => v;
    const y0 = tf(ymin),
      y1 = tf(ymax);
    const Y = (v) => Tp + ph - ((tf(v) - y0) / (y1 - y0)) * ph;
    el("line", { x1: L, y1: Tp + ph, x2: W - R, y2: Tp + ph, class: "ax" }, svg);
    el("line", { x1: L, y1: Tp, x2: L, y2: Tp + ph, class: "ax" }, svg);
    (o.ticks || []).forEach((t) => {
      const y = Y(t);
      el("line", { x1: L, y1: y, x2: W - R, y2: y, class: "gl" }, svg);
      el("text", { x: L - 4, y: y + 3.5, "text-anchor": "end", "font-size": 9.5 }, svg, o.tickFmt ? o.tickFmt(t) : String(t));
    });
    const bw = pw / n;
    const groups = o.groups || 1;
    vals.forEach((v, i) => {
      const gi = i % groups,
        ci = Math.floor(i / groups);
      const slot = pw / (n / groups),
        sw = (slot * 0.8) / groups;
      const x = L + ci * slot + slot * 0.1 + gi * sw;
      const base = o.signed ? Y(0) : Tp + ph;
      const yv = o.signed ? Y(v) : Y(Math.max(Math.abs(v), ymin));
      const top = Math.min(base, yv),
        h = Math.max(Math.abs(base - yv), 0.6);
      el("rect", { x: x, y: top, width: Math.max(sw - (groups > 1 ? 0.5 : 1), 0.8), height: h, class: o.cls ? o.cls(i, v) : "f-a" }, svg);
    });
    if (o.signed) el("line", { x1: L, y1: Y(0), x2: W - R, y2: Y(0), class: "ax" }, svg);
    if (o.xlabels)
      o.xlabels.forEach(([i, t, c]) => {
        el("text", { x: L + (i + 0.5) * (pw / (n / groups)), y: H - 6, "text-anchor": "middle", "font-size": 9.5, class: c || "" }, svg, t);
      });
    return { Y: Y, L: L, pw: pw };
  }
  function lineChart(svg, series, o) {
    clear(svg);
    const W = o.W || 320,
      H = o.H || 180,
      L = o.L || 46,
      R = 10,
      Tp = 10,
      B = 28;
    const pw = W - L - R,
      ph = H - Tp - B;
    const ty = o.logy ? (v) => Math.log10(Math.max(v, 1e-14)) : (v) => v;
    let lo = Infinity,
      hi = -Infinity;
    series.forEach((s) =>
      s.ys.forEach((v) => {
        if (isFinite(ty(v))) {
          lo = Math.min(lo, ty(v));
          hi = Math.max(hi, ty(v));
        }
      })
    );
    (o.hlines || []).forEach((h) => {
      lo = Math.min(lo, ty(h.y));
      hi = Math.max(hi, ty(h.y));
    });
    if (hi - lo < 1e-9) {
      hi += 0.5;
      lo -= 0.5;
    }
    const padv = (hi - lo) * 0.08;
    lo -= padv;
    hi += padv;
    const X = (x) => L + ((x - o.xmin) / (o.xmax - o.xmin)) * pw;
    const Y = (v) => Tp + ph - ((ty(v) - lo) / (hi - lo)) * ph;
    el("line", { x1: L, y1: Tp + ph, x2: W - R, y2: Tp + ph, class: "ax" }, svg);
    el("line", { x1: L, y1: Tp, x2: L, y2: Tp + ph, class: "ax" }, svg);
    for (let k = 0; k <= 4; k++) {
      const v = lo + ((hi - lo) * k) / 4,
        y = Tp + ph - (k / 4) * ph;
      el("line", { x1: L, y1: y, x2: W - R, y2: y, class: "gl" }, svg);
      const real = o.logy ? Math.pow(10, v) : v;
      el("text", { x: L - 4, y: y + 3.5, "text-anchor": "end", "font-size": 9.5 }, svg, o.yfmt ? o.yfmt(real) : fmtE(real));
    }
    (o.xticks || []).forEach((t) => {
      el("text", { x: X(t), y: H - 12, "text-anchor": "middle", "font-size": 9.5 }, svg, String(t));
    });
    if (o.xlabel) el("text", { x: L + pw / 2, y: H - 1, "text-anchor": "middle", class: "tt", "font-size": 11 }, svg, o.xlabel);
    (o.hlines || []).forEach((h) => {
      el("line", { x1: L, y1: Y(h.y), x2: W - R, y2: Y(h.y), class: (h.cls || "ln-i") + " dash" }, svg);
      if (h.label) el("text", { x: W - R - 2, y: Y(h.y) - 4, "text-anchor": "end", "font-size": 9.5, class: h.tcls || "" }, svg, h.label);
    });
    series.forEach((s) => {
      const d = s.xs.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(s.ys[i]).toFixed(1)).join(" ");
      el("path", { d: d, class: s.cls || "ln-a" }, svg);
    });
    (o.marks || []).forEach((m) => {
      el("circle", { cx: X(m.x), cy: Y(m.y), r: m.r || 4.5, class: m.cls || "f-w" }, svg);
      if (m.label)
        el(
          "text",
          { x: X(m.x) + (m.dx || 6), y: Y(m.y) + (m.dy || -6), "font-size": 10, class: m.tcls || "tw", "text-anchor": m.anchor || "start" },
          svg,
          m.label
        );
    });
    return { X: X, Y: Y };
  }
  function heat(svg, M, o) {
    const rows = M.length,
      cols = M[0].length;
    const cw = o.cw,
      ch = o.ch,
      x0 = o.x0,
      y0 = o.y0,
      vmax = o.vmax || 1;
    el("rect", { x: x0, y: y0, width: cols * cw, height: rows * ch, class: "f-p" }, svg);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        let t = Math.abs(M[i][j]) / vmax;
        if (o.scale === "log") t = Math.log1p(Math.abs(M[i][j]) / (o.v0 || 0.05)) / Math.log1p(vmax / (o.v0 || 0.05));
        else if (o.scale === "sqrt") t = Math.sqrt(t);
        t = Math.min(1, Math.max(0, t));
        el(
          "rect",
          {
            x: x0 + j * cw,
            y: y0 + i * ch,
            width: cw - (o.gap == null ? 0.6 : o.gap),
            height: ch - (o.gap == null ? 0.6 : o.gap),
            class: o.clsFn ? o.clsFn(i, j) : o.cls || "f-a",
            "fill-opacity": (0.03 + 0.97 * t).toFixed(3),
          },
          svg
        );
      }
  }

  /* ================= 实验：数值格式转换器 ================= */
  (function fmtLab() {
    const inp = $("fmt-in"),
      out = $("fmt-out");
    if (!inp) return;
    const F = [
      { name: "FP32", E: 8, M: 23, kind: "ieee" },
      { name: "FP16", E: 5, M: 10, kind: "ieee" },
      { name: "BF16", E: 8, M: 7, kind: "ieee" },
      { name: "FP8 E4M3", E: 4, M: 3, kind: "fn" },
      { name: "FP8 E5M2", E: 5, M: 2, kind: "ieee" },
      { name: "FP4 E2M1", E: 2, M: 1, kind: "none" },
    ];
    function encode(x, f) {
      if (f.name === "FP32") {
        const dv = new DataView(new ArrayBuffer(4));
        dv.setFloat32(0, x);
        const u = dv.getUint32(0);
        return {
          sign: u >>> 31,
          e: (u >>> 23) & 0xff,
          m: u & 0x7fffff,
          val: dv.getFloat32(0),
          sat: !isFinite(dv.getFloat32(0)),
          sub: ((u >>> 23) & 0xff) === 0 && (u & 0x7fffff) !== 0,
        };
      }
      const bias = (1 << (f.E - 1)) - 1,
        sign = x < 0 ? 1 : 0,
        a = Math.abs(x),
        P = Math.pow(2, f.M);
      let maxE, maxM;
      if (f.kind === "ieee") {
        maxE = (1 << f.E) - 2;
        maxM = P - 1;
      } else if (f.kind === "fn") {
        maxE = (1 << f.E) - 1;
        maxM = P - 2;
      } else {
        maxE = (1 << f.E) - 1;
        maxM = P - 1;
      }
      let e = 0,
        m = 0,
        sat = false,
        sub = false;
      if (a === 0) {
        /* zero */
      } else if (a < Math.pow(2, 1 - bias)) {
        m = rne(a / Math.pow(2, 1 - bias - f.M));
        if (m >= P) {
          e = 1;
          m = 0;
        } else sub = m > 0;
      } else {
        let ex = Math.floor(Math.log2(a)),
          fr = a / Math.pow(2, ex);
        if (fr >= 2) {
          ex++;
          fr /= 2;
        }
        if (fr < 1) {
          ex--;
          fr *= 2;
        }
        m = rne((fr - 1) * P);
        if (m === P) {
          m = 0;
          ex++;
        }
        e = ex + bias;
        if (e > maxE || (e === maxE && m > maxM)) {
          e = maxE;
          m = maxM;
          sat = true;
        }
      }
      const v = (e === 0 ? (m / P) * Math.pow(2, 1 - bias) : (1 + m / P) * Math.pow(2, e - bias)) * (sign ? -1 : 1);
      return { sign: sign, e: e, m: m, val: v, sat: sat, sub: sub };
    }
    function bits(n, w) {
      let s = n.toString(2);
      while (s.length < w) s = "0" + s;
      return s;
    }
    function render() {
      const x = parseFloat(inp.value);
      if (!isFinite(x)) {
        out.innerHTML = '<tr><td class="l">请输入一个有效的数。</td></tr>';
        return;
      }
      let h =
        '<thead><tr><th class="l">格式</th><th class="l">比特（符号 | 指数 | 尾数）</th><th>十六进制</th><th>表示的值</th><th>相对误差</th><th class="l">备注</th></tr></thead><tbody>';
      F.forEach((f) => {
        const r = encode(x, f);
        const sb = '<i class="s">' + r.sign + "</i>";
        const eb = bits(r.e, f.E)
          .split("")
          .map((c) => '<i class="e">' + c + "</i>")
          .join("");
        const mb = bits(r.m, f.M)
          .split("")
          .map((c) => '<i class="m">' + c + "</i>")
          .join("");
        const tot = r.sign * Math.pow(2, f.E + f.M) + r.e * Math.pow(2, f.M) + r.m;
        let hex = tot.toString(16).toUpperCase();
        const hw = Math.ceil((1 + f.E + f.M) / 4);
        while (hex.length < hw) hex = "0" + hex;
        const rel = x === 0 ? 0 : Math.abs(r.val - x) / Math.abs(x);
        let note = [];
        if (r.sat) note.push(f.name === "FP32" ? "溢出为 inf" : f.kind === "ieee" ? "溢出（IEEE 语义为 inf，此处显示饱和值）" : "饱和到最大值");
        if (r.sub) note.push("次正规数");
        if (r.val === 0 && x !== 0) note.push("下溢为 0");
        h +=
          '<tr><td class="l">' +
          f.name +
          '</td><td class="l"><span class="bits">' +
          sb +
          eb +
          mb +
          '</span></td><td class="mono">0x' +
          hex +
          '</td><td class="mono">' +
          (isFinite(r.val) ? String(+r.val.toPrecision(9)).replace("-", "−") : "∞") +
          '</td><td class="mono">' +
          (isFinite(r.val) ? (rel * 100).toPrecision(3) + "%" : "—") +
          '</td><td class="l small">' +
          note.join("，") +
          "</td></tr>";
      });
      out.innerHTML = h + "</tbody>";
    }
    inp.addEventListener("input", render);
    document.querySelectorAll("[data-fmt]").forEach((b) =>
      b.addEventListener("click", () => {
        inp.value = b.getAttribute("data-fmt");
        render();
      })
    );
    render();
  })();

  /* ================= 实验：量化计算器 ================= */
  (function qcLab() {
    const vin = $("qc-vals");
    if (!vin) return;
    function run() {
      const vals = vin.value
        .split(/[\s,，;；]+/)
        .filter((s) => s.length)
        .map(Number)
        .filter((v) => isFinite(v));
      const b = +$("qc-bits").value,
        mode = $("qc-mode").value,
        clip = +$("qc-clip").value;
      $("qc-clip-v").textContent = clip.toFixed(2);
      const tb = $("qc-table"),
        svg = $("qc-svg");
      if (!vals.length) {
        tb.innerHTML = '<tr><td class="l">请输入用逗号分隔的数字。</td></tr>';
        $("qc-stats").innerHTML = "";
        clear(svg);
        return;
      }
      let s, z, qmin, qmax;
      if (mode === "sym") {
        qmax = Math.pow(2, b - 1) - 1;
        qmin = -qmax;
        z = 0;
        s = (Math.max(...vals.map(Math.abs)) * clip) / qmax || 1;
      } else {
        qmin = 0;
        qmax = Math.pow(2, b) - 1;
        const mn = Math.min(0, ...vals) * clip,
          mx = Math.max(0, ...vals) * clip;
        s = (mx - mn) / (qmax - qmin) || 1;
        z = Math.min(qmax, Math.max(qmin, rne(qmin - mn / s)));
      }
      let se = 0,
        sx = 0,
        maxe = 0,
        clipped = 0;
      const rows = vals.map((x) => {
        const r = x / s + z,
          rr = rne(r),
          q = Math.min(qmax, Math.max(qmin, rr)),
          xh = s * (q - z),
          e = xh - x;
        se += e * e;
        sx += x * x;
        maxe = Math.max(maxe, Math.abs(e));
        if (q !== rr) clipped++;
        return { x: x, r: r, rr: rr, q: q, xh: xh, e: e, c: q !== rr };
      });
      const mse = se / vals.length,
        sqnr = se > 0 ? 10 * Math.log10(sx / se) : Infinity;
      $("qc-stats").innerHTML =
        stat("scale s", fmt(s, 5), "a") +
        stat("zero-point z", z) +
        stat("码字范围", "[" + qmin + ", " + qmax + "]") +
        stat("MSE", fmtE(mse), "w") +
        stat("最大误差", fmt(maxe, 4)) +
        stat("SQNR", isFinite(sqnr) ? sqnr.toFixed(1) + " dB" : "∞", "g") +
        stat("被截断", clipped + " 个");
      let h =
        '<thead><tr><th class="l">x</th><th>x / s' +
        (z ? " + z" : "") +
        "</th><th>round</th><th>clamp → q</th><th>x̂ = s(q − z)</th><th>误差 x̂ − x</th></tr></thead><tbody>";
      rows.forEach((r) => {
        h +=
          '<tr><td class="l mono">' +
          fmt(r.x, 4) +
          '</td><td class="mono">' +
          fmt(r.r, 3) +
          '</td><td class="mono">' +
          r.rr +
          '</td><td class="mono ' +
          (r.c ? "hw" : "") +
          '">' +
          r.q +
          '</td><td class="mono">' +
          fmt(r.xh, 4) +
          '</td><td class="mono ' +
          (Math.abs(r.e) > 0.75 * s ? "hw" : "") +
          '">' +
          fmt(r.e, 4) +
          "</td></tr>";
      });
      tb.innerHTML = h + "</tbody>";
      // 数轴
      clear(svg);
      const gmin = s * (qmin - z),
        gmax = s * (qmax - z);
      const lo = Math.min(gmin, ...vals),
        hi = Math.max(gmax, ...vals),
        pad = (hi - lo) * 0.04 || 1;
      const X = (v) => 30 + ((v - lo + pad) / (hi - lo + 2 * pad)) * 620;
      el("line", { x1: 30, y1: 64, x2: 650, y2: 64, class: "ax" }, svg);
      el("rect", { x: X(gmin), y: 56, width: Math.max(1, X(gmax) - X(gmin)), height: 16, class: "f-a", "fill-opacity": 0.08 }, svg);
      for (let q = qmin; q <= qmax; q++) {
        const xv = X(s * (q - z));
        el(
          "line",
          {
            x1: xv,
            y1: 56,
            x2: xv,
            y2: 72,
            class: "ln-a",
            "stroke-width": qmax - qmin > 64 ? 0.5 : 1.2,
            "stroke-opacity": qmax - qmin > 64 ? 0.5 : 1,
          },
          svg
        );
      }
      rows.forEach((r) => {
        el("line", { x1: X(r.x), y1: 38, x2: X(r.xh), y2: 64, class: "ln-w", "stroke-width": 1 }, svg);
        el("circle", { cx: X(r.x), cy: 36, r: 3.6, class: "f-w" }, svg);
        el("circle", { cx: X(r.xh), cy: 64, r: 3, class: "f-a" }, svg);
      });
      el("text", { x: 30, y: 16, class: "tw" }, svg, "● 原始值 x");
      el("text", { x: 140, y: 16, class: "ta" }, svg, "｜ 可表示的量化点（共 " + (qmax - qmin + 1) + " 个）");
      el("text", { x: X(gmin), y: 92, "text-anchor": "middle" }, svg, fmt(gmin, 3));
      el("text", { x: X(gmax), y: 92, "text-anchor": "middle" }, svg, fmt(gmax, 3));
      if (gmin < 0 && gmax > 0) el("text", { x: X(0), y: 108, "text-anchor": "middle", class: "t3" }, svg, "0");
    }
    ["qc-vals", "qc-bits", "qc-mode", "qc-clip"].forEach((id) => $(id).addEventListener("input", run));
    run();
  })();

  /* ================= 实验：截断权衡 ================= */
  (function clipLab() {
    if (!$("cl-c")) return;
    const g = gaussFn(mulberry32(11));
    const base = Array.from({ length: 2000 }, () => g());
    const signs = Array.from({ length: 10 }, (_, i) => (i % 2 ? 1 : -1) * (1 + (0.25 * ((i * 7) % 5)) / 5));
    let best = 1;
    function data() {
      const o = +$("cl-out").value;
      const x = base.slice();
      if (o > 0) for (let i = 0; i < 10; i++) x[i * 199] = o * signs[i];
      return x;
    }
    function mseAt(x, r, qmax, am) {
      const c = r * am,
        s = c / qmax;
      let er = 0,
        ec = 0;
      for (const v of x) {
        const q = Math.max(-qmax, Math.min(qmax, rne(v / s))),
          e = q * s - v;
        if (Math.abs(v) <= c) er += e * e;
        else ec += e * e;
      }
      return [er / x.length, ec / x.length];
    }
    function run() {
      const b = +$("cl-bits").value,
        qmax = Math.pow(2, b - 1) - 1,
        r = +$("cl-c").value;
      $("cl-out-v").textContent = (+$("cl-out").value).toFixed(1) + "σ";
      $("cl-c-v").textContent = r.toFixed(2);
      const x = data(),
        am = Math.max(...x.map(Math.abs));
      const rs = [],
        ys = [];
      let bm = Infinity;
      best = 1;
      for (let i = 0; i <= 95; i++) {
        const rr = 0.05 + (0.95 * i) / 95,
          m = mseAt(x, rr, qmax, am),
          t = m[0] + m[1];
        rs.push(rr);
        ys.push(t);
        if (t < bm) {
          bm = t;
          best = rr;
        }
      }
      const cur = mseAt(x, r, qmax, am);
      $("cl-stats").innerHTML =
        stat("max|x|", fmt(am, 2)) +
        stat("阈值 c", fmt(r * am, 3), "a") +
        stat("舍入误差", fmtE(cur[0])) +
        stat("截断误差", fmtE(cur[1]), "w") +
        stat("总 MSE", fmtE(cur[0] + cur[1]), "w") +
        stat("最优比例", best.toFixed(2), "g") +
        stat("最优 MSE", fmtE(bm), "g");
      lineChart($("cl-curve"), [{ xs: rs, ys: ys, cls: "ln-a" }], {
        xmin: 0.05,
        xmax: 1,
        logy: true,
        xticks: [0.2, 0.4, 0.6, 0.8, 1],
        xlabel: "c / max|x|",
        marks: [
          { x: best, y: bm, cls: "f-g", label: "最优", tcls: "tg", dy: 14 },
          { x: r, y: cur[0] + cur[1], cls: "f-w" },
        ],
      });
      // 直方图
      const svg = $("cl-hist");
      clear(svg);
      const nb = 61,
        cnt = new Array(nb).fill(0);
      x.forEach((v) => {
        const k = Math.min(nb - 1, Math.max(0, Math.floor(((v + am) / (2 * am)) * nb)));
        cnt[k]++;
      });
      const mx = Math.log1p(Math.max(...cnt));
      const L = 10,
        W = 300,
        Hh = 140;
      cnt.forEach((c, k) => {
        const h = (Math.log1p(c) / mx) * Hh;
        el(
          "rect",
          {
            x: L + (k * W) / nb,
            y: 10 + Hh - h,
            width: W / nb - 0.8,
            height: h,
            class: Math.abs(-am + ((k + 0.5) * 2 * am) / nb) > r * am ? "f-w" : "f-a",
            "fill-opacity": 0.85,
          },
          svg
        );
      });
      el("line", { x1: L, y1: 10 + Hh, x2: L + W, y2: 10 + Hh, class: "ax" }, svg);
      [-1, 1].forEach((sg) => {
        const xv = L + ((sg * r * am + am) / (2 * am)) * W;
        el("line", { x1: xv, y1: 6, x2: xv, y2: 10 + Hh, class: "ln-w dash" }, svg);
      });
      el("text", { x: L, y: 170, "font-size": 9.5 }, svg, fmt(-am, 1));
      el("text", { x: L + W, y: 170, "font-size": 9.5, "text-anchor": "end" }, svg, fmt(am, 1));
      el("text", { x: L + W / 2, y: 170, "font-size": 9.5, "text-anchor": "middle", class: "t3" }, svg, "纵轴为 log(1+计数)，橙色 = 被截断的区域");
    }
    ["cl-bits", "cl-out", "cl-c"].forEach((id) => $(id).addEventListener("input", run));
    $("cl-best").addEventListener("click", () => {
      $("cl-c").value = best.toFixed(2);
      run();
    });
    run();
  })();

  /* ================= 实验：粒度热力图 ================= */
  (function granLab() {
    if (!$("gr-mode")) return;
    const g = gaussFn(mulberry32(21)),
      rg = mulberry32(22);
    const R = 8,
      C = 16,
      base = zeros(R, C),
      rowF = Array.from({ length: R }, () => 0.6 + 0.8 * rg());
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) base[i][j] = g() * 0.6 * rowF[i];
    function quant(X, mode, qmax) {
      const Q = zeros(R, C);
      const groupOf = (i, j) => (mode === "tensor" ? "a" : mode === "row" ? "r" + i : mode === "col" ? "c" + j : "g" + i + "_" + (j >> 2));
      const am = {};
      for (let i = 0; i < R; i++)
        for (let j = 0; j < C; j++) {
          const k = groupOf(i, j);
          am[k] = Math.max(am[k] || 0, Math.abs(X[i][j]));
        }
      for (let i = 0; i < R; i++)
        for (let j = 0; j < C; j++) {
          const s = (am[groupOf(i, j)] || 1) / qmax;
          Q[i][j] = Math.max(-qmax, Math.min(qmax, rne(X[i][j] / s))) * s;
        }
      return { Q: Q, n: Object.keys(am).length };
    }
    function run() {
      const mode = $("gr-mode").value,
        b = +$("gr-bits").value,
        co = +$("gr-co").value,
        qmax = Math.pow(2, b - 1) - 1;
      $("gr-co-v").textContent = co + "×";
      const X = base.map((r) => r.map((v, j) => (j === 3 || j === 11 ? v * co : v)));
      let emax = 0;
      ["tensor", "row", "col", "group"].forEach((m) => {
        const q = quant(X, m, qmax).Q;
        for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) emax = Math.max(emax, Math.abs(q[i][j] - X[i][j]));
      });
      const res = quant(X, mode, qmax),
        E = sub(res.Q, X);
      let en = 0,
        nn = 0,
        eo = 0,
        no = 0,
        sn = 0;
      for (let i = 0; i < R; i++)
        for (let j = 0; j < C; j++) {
          if (j === 3 || j === 11) {
            eo += E[i][j] * E[i][j];
            no++;
          } else {
            en += E[i][j] * E[i][j];
            sn += X[i][j] * X[i][j];
            nn++;
          }
        }
      const ok = mode === "tensor" || mode === "row";
      $("gr-stats").innerHTML =
        stat("scale 个数", res.n, "a") +
        stat("正常通道相对误差", (Math.sqrt(en / sn) * 100).toFixed(1) + "%", "w") +
        stat("离群通道 MSE", fmtE(eo / no)) +
        stat("可用于激活的整数 GEMM", ok ? "可以" : "不可以", ok ? "g" : "w");
      const am = Math.max(...X.map((r) => Math.max(...r.map(Math.abs))));
      [
        ["gr-a", X, am, "log", "f-a"],
        ["gr-b", E, emax, "sqrt", "f-w"],
      ].forEach(([id, M, vm, sc, cls]) => {
        const svg = $(id);
        clear(svg);
        heat(svg, M, { x0: 22, y0: 8, cw: 19, ch: 18, vmax: vm, scale: sc, v0: 0.3, cls: cls });
        for (let i = 0; i < R; i++) el("text", { x: 18, y: 8 + i * 18 + 12.5, "text-anchor": "end", "font-size": 9 }, svg, "t" + i);
        [3, 11].forEach((j) => el("text", { x: 22 + j * 19 + 9.5, y: 166, "text-anchor": "middle", "font-size": 9.5, class: "tw" }, svg, String(j)));
        if (mode === "group" && id === "gr-b")
          for (let j = 4; j < C; j += 4) el("line", { x1: 22 + j * 19 - 0.3, y1: 8, x2: 22 + j * 19 - 0.3, y2: 8 + R * 18, class: "ax" }, svg);
      });
    }
    ["gr-mode", "gr-bits", "gr-co"].forEach((id) => $(id).addEventListener("input", run));
    run();
  })();

  /* ================= 实验：离群通道吃掉量化等级 ================= */
  (function olLab() {
    if (!$("ol-mag")) return;
    const g = gaussFn(mulberry32(31)),
      rg = mulberry32(32);
    const Tn = 32,
      C = 64,
      OUT = [9, 41];
    const chs = Array.from({ length: C }, () => 0.5 + rg());
    const base = zeros(Tn, C);
    for (let t = 0; t < Tn; t++) for (let c = 0; c < C; c++) base[t][c] = g() * chs[c];
    function run() {
      const mag = +$("ol-mag").value,
        b = +$("ol-bits").value,
        gran = $("ol-gran").value,
        qmax = Math.pow(2, b - 1) - 1;
      $("ol-mag-v").textContent = mag + "×";
      const X = base.map((r) => r.map((v, c) => (OUT.includes(c) ? v * mag : v)));
      const tam = Math.max(...X.map((r) => Math.max(...r.map(Math.abs))));
      const hist = new Map();
      let en = 0,
        sn = 0,
        eo = 0,
        so = 0;
      for (let t = 0; t < Tn; t++) {
        const am = gran === "tensor" ? tam : Math.max(...X[t].map(Math.abs)),
          s = am / qmax;
        for (let c = 0; c < C; c++) {
          const q = Math.max(-qmax, Math.min(qmax, rne(X[t][c] / s))),
            e = q * s - X[t][c];
          if (OUT.includes(c)) {
            eo += e * e;
            so += X[t][c] * X[t][c];
          } else {
            en += e * e;
            sn += X[t][c] * X[t][c];
            hist.set(q, (hist.get(q) || 0) + 1);
          }
        }
      }
      const used = hist.size;
      $("ol-stats").innerHTML =
        stat("正常通道用到的码字数", used + " / " + (2 * qmax + 1), "w") +
        stat("等效位宽", Math.log2(Math.max(used, 1)).toFixed(1) + " bit", "w") +
        stat("正常通道相对误差", (Math.sqrt(en / sn) * 100).toFixed(1) + "%") +
        stat("离群通道相对误差", (Math.sqrt(eo / so) * 100).toFixed(2) + "%", "g");
      const svg = $("ol-hist");
      clear(svg);
      const L = 20,
        W = 600,
        H = 120,
        n = 2 * qmax + 1,
        mx = Math.max(...hist.values());
      el("line", { x1: L, y1: 10 + H, x2: L + W, y2: 10 + H, class: "ax" }, svg);
      for (let q = -qmax; q <= qmax; q++) {
        const c = hist.get(q) || 0;
        if (!c) continue;
        const h = Math.max(1, (c / mx) * H),
          x = L + ((q + qmax) / n) * W;
        el("rect", { x: x, y: 10 + H - h, width: Math.max(W / n - 0.6, 1.2), height: h, class: "f-a" }, svg);
      }
      [-qmax, 0, qmax].forEach((q) =>
        el("text", { x: L + ((q + qmax + 0.5) / n) * W, y: 160, "text-anchor": "middle", "font-size": 10 }, svg, String(q))
      );
    }
    ["ol-mag", "ol-bits", "ol-gran"].forEach((id) => $(id).addEventListener("input", run));
    run();
  })();

  /* ================= 实验：GPTQ 单步执行 ================= */
  (function gptqLab() {
    if (!$("gq-step")) return;
    const EX_W = [
      [-0.96, 0.76, 0.1, 0.38],
      [-0.47, -0.24, -0.23, 0.9],
    ];
    const EX_X = [
      [0.2, -0.7, 2.0, 1.4, 0.0, -0.1],
      [-0.3, -0.2, 1.3, 1.2, 0.0, 0.3],
      [-0.6, 1.6, 0.4, 1.2, -1.0, 1.3],
      [-1.0, 0.8, 0.5, 0.7, -0.5, 1.1],
    ];
    let S = null,
      seed = 100;
    function setup(W, X) {
      const b = +$("gq-bits").value,
        qmax = Math.pow(2, b - 1) - 1,
        qmin = -Math.pow(2, b - 1);
      const n = W[0].length;
      const H = matmul(X, T(X)).map((r) => r.map((v) => 2 * v));
      const md = H.reduce((a, r, i) => a + r[i], 0) / n;
      const Hd = H.map((r, i) => r.map((v, j) => v + (i === j ? 0.01 * md : 0)));
      const U = T(cholLower(inv(Hd)));
      const s = W.map((r) => Math.max(...r.map(Math.abs)) / qmax || 1);
      const qf = (v, i) => Math.max(qmin, Math.min(qmax, rne(v / s[i])));
      const rtn = W.map((r, i) => r.map((v) => qf(v, i) * s[i]));
      S = {
        W0: W,
        X: X,
        Wc: copyM(W),
        Q: zeros(W.length, n),
        codes: zeros(W.length, n),
        U: U,
        s: s,
        qf: qf,
        j: 0,
        n: n,
        rtn: rtn,
        upd: null,
        err: null,
        prev: null,
      };
      render();
    }
    function step() {
      if (!S || S.j >= S.n) return;
      const j = S.j,
        U = S.U;
      S.prev = copyM(S.Wc);
      S.upd = zeros(S.W0.length, S.n);
      S.err = [];
      S.Wc.forEach((row, i) => {
        const w = row[j],
          q = S.qf(w, i),
          qv = q * S.s[i],
          e = (w - qv) / U[j][j];
        S.codes[i][j] = q;
        S.Q[i][j] = qv;
        S.err.push({ w: w, q: q, qv: qv, e: e });
        for (let k = j + 1; k < S.n; k++) {
          row[k] -= e * U[j][k];
          if (Math.abs(e * U[j][k]) > 1e-12) S.upd[i][k] = 1;
        }
      });
      S.j++;
      render();
    }
    function loss(Wh) {
      return sumSq(matmul(sub(S.W0, Wh), S.X));
    }
    function render() {
      const j = S.j,
        n = S.n,
        last = j - 1;
      let h = '<table><thead><tr><th class="l">行</th>';
      for (let k = 0; k < n; k++) h += "<th>列 " + (k + 1) + "</th>";
      h += "</tr></thead><tbody>";
      S.Wc.forEach((row, i) => {
        h += '<tr><td class="l mono">s=' + fmt(S.s[i], 3) + "</td>";
        for (let k = 0; k < n; k++) {
          let st = "",
            txt;
          if (k < j) {
            st = "background:var(--accent-2);color:var(--accent-ink);";
            txt = fmt(S.Q[i][k], 3) + '<br><span class="small">q=' + S.codes[i][k] + "</span>";
          } else {
            txt = fmt(row[k], 4);
            if (S.upd && S.upd[i][k]) st = "color:var(--good);font-weight:700;";
          }
          if (k === last) st += "outline:2px solid var(--warm);outline-offset:-2px;";
          h += '<td class="mono" style="' + st + '">' + txt + "</td>";
        }
        h += "</tr>";
      });
      $("gq-w").innerHTML = h + "</tbody></table>";
      if (last >= 0 && S.err) {
        const U = S.U;
        let u =
          '<p class="mono">U[' +
          (last + 1) +
          "," +
          (last + 1) +
          ":] = [" +
          U[last]
            .slice(last)
            .map((v) => fmt(v, 4))
            .join(", ") +
          "]</p><ul>";
        S.err.forEach((r, i) => {
          u +=
            "<li>第 " +
            (i + 1) +
            " 行：w = " +
            fmt(r.w, 4) +
            " → q = " +
            r.q +
            "（" +
            fmt(r.qv, 3) +
            "），err = (" +
            fmt(r.w, 4) +
            " − " +
            fmt(r.qv, 3) +
            ") / " +
            fmt(U[last][last], 4) +
            " = <b>" +
            fmt(r.e, 4) +
            "</b></li>";
        });
        u += '</ul><p class="mute">右侧每个未量化的列 k：w_k −= err × U[' + (last + 1) + ", k]</p>";
        $("gq-u").innerHTML = u;
      } else
        $("gq-u").innerHTML =
          '<p class="mute">点击“量化下一列”开始。已预先算好 H = 2XX<sup>T</sup>（加 1% 阻尼）、H<sup>−1</sup> 及其上三角 Cholesky 因子 U。</p><p class="mono">U 的对角 = [' +
          S.U.map((r, i) => fmt(r[i], 4)).join(", ") +
          "]</p>";
      const mixed = S.Wc.map((row, i) => row.map((v, k) => (k < j ? S.Q[i][k] : v)));
      const lr = loss(S.rtn),
        lm = loss(mixed);
      let c =
        '<p>RTN 损失：<b class="hl-w">' +
        fmt(lr, 4) +
        '</b>　RTN 码字：<span class="mono">' +
        S.rtn.map((r, i) => "[" + r.map((v) => Math.round(v / S.s[i])).join(", ") + "]").join(" ") +
        "</span></p>";
      c +=
        "<p>" +
        (j < n ? "当前（已量化列取 q、其余列取补偿后的 w）损失：" : "GPTQ 最终损失：") +
        '<b class="hl-g">' +
        fmt(lm, 4) +
        "</b>" +
        (j === n
          ? '　GPTQ 码字：<span class="mono">' +
            S.codes.map((r) => "[" + r.join(", ") + "]").join(" ") +
            "</span>　相对 RTN 降低 <b>" +
            (lr / Math.max(lm, 1e-12)).toFixed(1) +
            "×</b>"
          : "") +
        "</p>";
      c +=
        '<p class="mute">GPTQ 每量化一列，损失只会增加 err<sup>2</sup>/2 的累积量（OBS 增量），所以这条“当前损失”单调上升，最终停在比 RTN 低得多的位置。</p>';
      $("gq-cmp").innerHTML = c;
      $("gq-stats").innerHTML = stat("进度", j + " / " + n + " 列", "a") + stat("阻尼 λ", "1% · mean(diag H)") + stat("校准 token 数", S.X[0].length);
      $("gq-step").disabled = j >= n;
      $("gq-run").disabled = j >= n;
    }
    function randomEx() {
      const rg = mulberry32(seed++),
        g = gaussFn(rg);
      const nf = 8,
        nt = 16,
        fac = zeros(4, nt);
      for (let a = 0; a < 4; a++) for (let t = 0; t < nt; t++) fac[a][t] = g();
      const X = zeros(nf, nt);
      for (let f = 0; f < nf; f++) for (let t = 0; t < nt; t++) X[f][t] = Math.round((fac[f >> 1][t] * (f % 2 ? 0.8 : 1) + 0.4 * g()) * 10) / 10;
      const W = zeros(4, nf).map((r) => r.map(() => Math.round((rg() * 2 - 1) * 100) / 100));
      setup(W, X);
    }
    $("gq-step").addEventListener("click", step);
    $("gq-run").addEventListener("click", () => {
      while (S.j < S.n) step();
    });
    $("gq-reset").addEventListener("click", () => setup(EX_W, EX_X));
    $("gq-rand").addEventListener("click", randomEx);
    $("gq-bits").addEventListener("input", () => setup(S.W0, S.X));
    setup(EX_W, EX_X);
  })();

  /* ================= 实验：AWQ α 搜索 ================= */
  (function awqLab() {
    if (!$("aw-a")) return;
    const K = 32,
      N = 24,
      Tt = 128,
      G = 16,
      SAL = [4, 17, 26];
    const g = gaussFn(mulberry32(41)),
      rg = mulberry32(42);
    const mk = Array.from({ length: K }, () => 0.6 + 0.8 * rg());
    const salJ = SAL.map(() => 0.8 + 0.4 * rg());
    const Xb = zeros(Tt, K);
    for (let t = 0; t < Tt; t++) for (let k = 0; k < K; k++) Xb[t][k] = g() * mk[k];
    const W = zeros(N, K).map((r) => r.map(() => g() * 0.4));
    let cache = null;
    function pq(Wm, b) {
      // 非对称分组伪量化
      const qmax = Math.pow(2, b) - 1;
      return Wm.map((r) => {
        const o = r.slice();
        for (let g0 = 0; g0 < K; g0 += G) {
          const seg = r.slice(g0, g0 + G),
            mx = Math.max(...seg),
            mn = Math.min(...seg);
          const s = Math.max(mx - mn, 1e-8) / qmax,
            z = rne(-mn / s);
          for (let k = g0; k < g0 + G; k++) o[k] = (Math.max(0, Math.min(qmax, rne(r[k] / s) + z)) - z) * s;
        }
        return o;
      });
    }
    function compute() {
      const b = +$("aw-bits").value,
        sal = +$("aw-sal").value;
      const X = Xb.map((r) =>
        r.map((v, k) => {
          const i = SAL.indexOf(k);
          return i >= 0 ? v * sal * salJ[i] : v;
        })
      );
      const Y = matmul(X, T(W));
      const sx = Array.from({ length: K }, (_, k) => X.reduce((a, r) => a + Math.abs(r[k]), 0) / Tt);
      const mse = (Wh) => sumSq(sub(matmul(X, T(Wh)), Y)) / (Tt * N);
      const alphas = [],
        losses = [],
        scales = [];
      for (let i = 0; i <= 20; i++) {
        const a = i / 20;
        let s = sx.map((v) => Math.max(Math.pow(v, a), 1e-4));
        const nm = Math.sqrt(Math.max(...s) * Math.min(...s));
        s = s.map((v) => v / nm);
        const Wq = pq(
          W.map((r) => r.map((v, k) => v * s[k])),
          b
        ).map((r) => r.map((v, k) => v / s[k]));
        alphas.push(a);
        losses.push(mse(Wq));
        scales.push(s);
      }
      const Wr = pq(W, b),
        Wkeep = Wr.map((r, n) => r.map((v, k) => (SAL.includes(k) ? W[n][k] : v)));
      cache = { alphas: alphas, losses: losses, scales: scales, keep: mse(Wkeep), rtn: losses[0] };
    }
    function render() {
      const a = +$("aw-a").value,
        i = Math.round(a * 20);
      $("aw-a-v").textContent = a.toFixed(2);
      $("aw-sal-v").textContent = $("aw-sal").value + "×";
      const c = cache;
      let bi = 0;
      c.losses.forEach((v, k) => {
        if (v < c.losses[bi]) bi = k;
      });
      $("aw-stats").innerHTML =
        stat("RTN（α=0）MSE", fmtE(c.rtn), "w") +
        stat("当前 α MSE", fmtE(c.losses[i]), "a") +
        stat("最优 α", c.alphas[bi].toFixed(2), "g") +
        stat("最优 MSE", fmtE(c.losses[bi]), "g") +
        stat("保留 3 通道 FP16", fmtE(c.keep)) +
        stat("相对 RTN", (c.rtn / c.losses[bi]).toFixed(2) + "× 更好", "g");
      lineChart($("aw-curve"), [{ xs: c.alphas, ys: c.losses, cls: "ln-a" }], {
        xmin: 0,
        xmax: 1,
        logy: true,
        xticks: [0, 0.25, 0.5, 0.75, 1],
        xlabel: "α",
        hlines: [{ y: c.keep, cls: "ln-v", label: "混合精度参考", tcls: "tv" }],
        marks: [
          { x: c.alphas[bi], y: c.losses[bi], cls: "f-g", label: "最优", tcls: "tg", dy: 16 },
          { x: c.alphas[i], y: c.losses[i], cls: "f-w" },
        ],
      });
      const s = c.scales[i];
      barChart($("aw-s"), s, {
        log: true,
        ymin: Math.min(...s) * 0.8,
        ymax: Math.max(...s) * 1.2,
        ticks: [0.25, 0.5, 1, 2, 4].filter((t) => t > Math.min(...s) * 0.8 && t < Math.max(...s) * 1.2),
        tickFmt: (t) => String(t),
        cls: (k) => (SAL.includes(k) ? "f-w" : "f-a"),
        xlabels: SAL.map((k) => [k, String(k), "tw"]).concat([
          [0, "0"],
          [31, "31"],
        ]),
        H: 190,
      });
    }
    ["aw-bits", "aw-sal"].forEach((id) =>
      $(id).addEventListener("input", () => {
        compute();
        render();
      })
    );
    $("aw-a").addEventListener("input", render);
    $("aw-best").addEventListener("click", () => {
      let bi = 0;
      cache.losses.forEach((v, k) => {
        if (v < cache.losses[bi]) bi = k;
      });
      $("aw-a").value = cache.alphas[bi];
      render();
    });
    compute();
    render();
  })();

  /* ================= 实验：SmoothQuant α ================= */
  (function sqLab() {
    if (!$("sq-a")) return;
    const Tt = 64,
      K = 16,
      N = 16,
      OUT = [3, 12];
    const g = gaussFn(mulberry32(51)),
      rg = mulberry32(52);
    const chs = Array.from({ length: K }, () => 0.5 + rg());
    const Xb = zeros(Tt, K);
    for (let t = 0; t < Tt; t++) for (let k = 0; k < K; k++) Xb[t][k] = g() * chs[k];
    const W = zeros(K, N).map((r) => r.map(() => g() * 0.25));
    function qmat(A, b, mode, axis) {
      // axis: 'row' 每行一个 scale，'col' 每列一个
      const qmax = Math.pow(2, b - 1) - 1;
      if (mode === "tensor") {
        const s = Math.max(...A.map((r) => Math.max(...r.map(Math.abs)))) / qmax;
        return A.map((r) => r.map((v) => Math.max(-qmax, Math.min(qmax, rne(v / s))) * s));
      }
      if (axis === "row")
        return A.map((r) => {
          const s = Math.max(...r.map(Math.abs)) / qmax || 1;
          return r.map((v) => Math.max(-qmax, Math.min(qmax, rne(v / s))) * s);
        });
      const sc = A[0].map((_, j) => Math.max(...A.map((r) => Math.abs(r[j]))) / qmax || 1);
      return A.map((r) => r.map((v, j) => Math.max(-qmax, Math.min(qmax, rne(v / sc[j]))) * sc[j]));
    }
    let st = null;
    function compute() {
      const mag = +$("sq-mag").value,
        b = +$("sq-bits").value,
        mode = $("sq-gran").value;
      const X = Xb.map((r) => r.map((v, k) => (OUT.includes(k) ? v * mag : v)));
      const Y = matmul(X, W),
        ny = Math.sqrt(sumSq(Y));
      const ax = Array.from({ length: K }, (_, k) => Math.max(...X.map((r) => Math.abs(r[k]))));
      const aw = W.map((r) => Math.max(...r.map(Math.abs)));
      const relErr = (s) => {
        const Xs = X.map((r) => r.map((v, k) => v / s[k])),
          Ws = W.map((r, k) => r.map((v) => v * s[k]));
        return Math.sqrt(sumSq(sub(matmul(qmat(Xs, b, mode, "row"), qmat(Ws, b, mode, "col")), Y))) / ny;
      };
      const alphas = [],
        errs = [];
      for (let i = 0; i <= 20; i++) {
        const a = i / 20,
          s = ax.map((v, k) => Math.max(Math.pow(v, a) / Math.pow(aw[k], 1 - a), 1e-5));
        alphas.push(a);
        errs.push(relErr(s));
      }
      st = { X: X, ax: ax, aw: aw, alphas: alphas, errs: errs, none: relErr(ax.map(() => 1)) };
    }
    function render() {
      const a = +$("sq-a").value,
        i = Math.round(a * 20);
      $("sq-a-v").textContent = a.toFixed(2);
      $("sq-mag-v").textContent = $("sq-mag").value + "×";
      const s = st.ax.map((v, k) => Math.max(Math.pow(v, a) / Math.pow(st.aw[k], 1 - a), 1e-5));
      const axs = st.ax.map((v, k) => v / s[k]),
        aws = st.aw.map((v, k) => v * s[k]);
      let bi = 0;
      st.errs.forEach((v, k) => {
        if (v < st.errs[bi]) bi = k;
      });
      $("sq-stats").innerHTML =
        stat("不平滑", (st.none * 100).toFixed(2) + "%", "w") +
        stat("当前 α", (st.errs[i] * 100).toFixed(2) + "%", "a") +
        stat("最优 α", st.alphas[bi].toFixed(2), "g") +
        stat("最优误差", (st.errs[bi] * 100).toFixed(2) + "%", "g") +
        stat("激活通道最大/最小比", (Math.max(...st.ax) / Math.min(...st.ax)).toFixed(1) + " → " + (Math.max(...axs) / Math.min(...axs)).toFixed(1));
      const draw = (id, before, after, cls) => {
        const vals = [];
        before.forEach((v, k) => {
          vals.push(v);
          vals.push(after[k]);
        });
        const lo = Math.min(...vals) * 0.7,
          hi = Math.max(...vals) * 1.4;
        const ticks = [];
        for (let p = -4; p <= 4; p++) {
          const t = Math.pow(10, p);
          if (t > lo && t < hi) ticks.push(t);
        }
        barChart($(id), vals, {
          log: true,
          ymin: lo,
          ymax: hi,
          groups: 2,
          ticks: ticks,
          tickFmt: (t) => (t >= 1 ? String(t) : t.toString()),
          cls: (k) => (k % 2 ? cls : "f-r"),
          xlabels: OUT.map((k) => [k, String(k), "tw"]),
        });
      };
      draw("sq-x", st.ax, axs, "f-a");
      draw("sq-w", st.aw, aws, "f-g");
      lineChart($("sq-c"), [{ xs: st.alphas, ys: st.errs.map((v) => v * 100), cls: "ln-a" }], {
        W: 640,
        H: 170,
        xmin: 0,
        xmax: 1,
        logy: true,
        xticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        xlabel: "α（迁移强度）",
        yfmt: (v) => v.toPrecision(2) + "%",
        hlines: [{ y: st.none * 100, cls: "ln-w", label: "不平滑", tcls: "tw" }],
        marks: [
          { x: st.alphas[bi], y: st.errs[bi] * 100, cls: "f-g", label: "最优", tcls: "tg", dy: 16 },
          { x: a, y: st.errs[i] * 100, cls: "f-w" },
        ],
      });
    }
    ["sq-mag", "sq-bits", "sq-gran"].forEach((id) =>
      $(id).addEventListener("input", () => {
        compute();
        render();
      })
    );
    $("sq-a").addEventListener("input", render);
    compute();
    render();
  })();

  /* ================= 实验：旋转 ================= */
  function fwht(v) {
    const a = v.slice(),
      n = a.length;
    for (let h = 1; h < n; h *= 2)
      for (let i = 0; i < n; i += 2 * h)
        for (let j = i; j < i + h; j++) {
          const x = a[j],
            y = a[j + h];
          a[j] = x + y;
          a[j + h] = x - y;
        }
    const r = Math.sqrt(n);
    return a.map((x) => x / r);
  }
  (function rotLab() {
    if (!$("rt-mag")) return;
    let seed = 61;
    let base = null;
    function regen() {
      const rg = mulberry32(seed),
        g = gaussFn(rg);
      base = {
        x: Array.from({ length: 256 }, () => g()),
        w: Array.from({ length: 256 }, () => g() * 0.2),
        d: Array.from({ length: 256 }, () => (rg() < 0.5 ? -1 : 1)),
      };
    }
    function q(v, b) {
      const qmax = Math.pow(2, b - 1) - 1,
        s = Math.max(...v.map(Math.abs)) / qmax || 1;
      return v.map((x) => Math.max(-qmax, Math.min(qmax, rne(x / s))) * s);
    }
    function run() {
      const mag = +$("rt-mag").value,
        n = +$("rt-n").value,
        b = +$("rt-bits").value,
        useD = $("rt-sign").checked;
      $("rt-mag-v").textContent = mag.toFixed(1);
      const oi = 5 % n,
        x = base.x.slice(0, n),
        w = base.w.slice(0, n),
        d = base.d.slice(0, n);
      if (mag > 0) x[oi] = mag;
      const xr = fwht(x.map((v, i) => (useD ? v * d[i] : v))),
        wr = fwht(w.map((v, i) => (useD ? v * d[i] : v)));
      const dot = x.reduce((a, v, i) => a + v * w[i], 0);
      const xq = q(x, b),
        wq = q(w, b),
        xrq = q(xr, b),
        wrq = q(wr, b);
      const d0 = xq.reduce((a, v, i) => a + v * wq[i], 0),
        d1 = xrq.reduce((a, v, i) => a + v * wrq[i], 0);
      const pr = (v) => Math.max(...v.map(Math.abs)) / Math.sqrt(v.reduce((a, t) => a + t * t, 0) / v.length);
      const mse = (A, B) => A.reduce((a, v, i) => a + (v - B[i]) * (v - B[i]), 0) / A.length;
      $("rt-stats").innerHTML =
        stat("峰均比", pr(x).toFixed(2) + " → " + pr(xr).toFixed(2), "a") +
        stat("x 量化 MSE", fmtE(mse(xq, x)) + " → " + fmtE(mse(xrq, xr)), "g") +
        stat("点积 x·w", fmt(dot, 3)) +
        stat("点积误差（不旋转）", ((Math.abs(d0 - dot) / Math.max(Math.abs(dot), 1e-9)) * 100).toFixed(1) + "%", "w") +
        stat("点积误差（旋转）", ((Math.abs(d1 - dot) / Math.max(Math.abs(dot), 1e-9)) * 100).toFixed(1) + "%", "g");
      const ym = Math.max(...x.map(Math.abs), ...xr.map(Math.abs)) * 1.05;
      barChart($("rt-a"), x, {
        signed: true,
        ymin: -ym,
        ymax: ym,
        H: 150,
        cls: (k) => (k === oi && mag > 0 ? "f-w" : "f-a"),
        ticks: [-Math.round(ym), 0, Math.round(ym)],
      });
      barChart($("rt-b"), xr, { signed: true, ymin: -ym, ymax: ym, H: 150, cls: () => "f-v", ticks: [-Math.round(ym), 0, Math.round(ym)] });
    }
    ["rt-mag", "rt-n", "rt-bits", "rt-sign"].forEach((id) => $(id).addEventListener("input", run));
    $("rt-new").addEventListener("click", () => {
      seed++;
      regen();
      run();
    });
    regen();
    run();
  })();

  /* ================= 实验：ConvRot ================= */
  (function convLab() {
    if (!$("cv-row")) return;
    const H4 = [
      [1, 1, 1, -1],
      [1, 1, -1, 1],
      [1, -1, 1, 1],
      [-1, 1, 1, 1],
    ];
    function kron(A, B) {
      const n = A.length,
        m = B.length,
        R = zeros(n * m, n * m);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) for (let k = 0; k < m; k++) for (let l = 0; l < m; l++) R[i * m + k][j * m + l] = A[i][j] * B[k][l];
      return R;
    }
    function syl(n) {
      let H = [[1]];
      while (H.length < n) {
        const m = H.length,
          R = zeros(2 * m, 2 * m);
        for (let i = 0; i < m; i++)
          for (let j = 0; j < m; j++) {
            R[i][j] = H[i][j];
            R[i][j + m] = H[i][j];
            R[i + m][j] = H[i][j];
            R[i + m][j + m] = -H[i][j];
          }
        H = R;
      }
      return H;
    }
    function reg(n) {
      let H = [[1]];
      while (H.length < n) H = kron(H, H4);
      return H;
    }
    const Tn = 8,
      C = 64,
      g = gaussFn(mulberry32(71)),
      rg = mulberry32(72);
    const base = zeros(Tn, C),
      noise = zeros(Tn, C),
      csign = Array.from({ length: Tn }, () => (rg() < 0.5 ? -1 : 1));
    for (let t = 0; t < Tn; t++)
      for (let c = 0; c < C; c++) {
        base[t][c] = g() * 0.5;
        noise[t][c] = g() * 0.1;
      }
    function rot(X, H) {
      const n = H.length,
        r = Math.sqrt(n);
      return X.map((row) => {
        const o = new Array(C).fill(0);
        for (let g0 = 0; g0 < C; g0 += n)
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < n; i++) s += row[g0 + i] * H[i][j];
            o[g0 + j] = s / r;
          }
        return o;
      });
    }
    function relq(row) {
      const s = Math.max(...row.map(Math.abs)) / 7 || 1;
      let e = 0,
        t = 0;
      row.forEach((v) => {
        const q = Math.max(-7, Math.min(7, rne(v / s))) * s;
        e += (q - v) * (q - v);
        t += v * v;
      });
      return e / Math.max(t, 1e-12);
    }
    function run() {
      const rm = +$("cv-row").value,
        cm = +$("cv-col").value,
        n = +$("cv-n").value;
      $("cv-row-v").textContent = rm.toFixed(2);
      $("cv-col-v").textContent = cm.toFixed(1);
      const X = base.map((r, t) =>
        r.map((v, c) => {
          let x = v;
          if (t === 3) x += rm * (1 + noise[t][c]);
          if (c === 5) x += cm * csign[t] * (1 + noise[t][c]);
          return x;
        })
      );
      const Xs = rot(X, syl(n)),
        Xr = rot(X, reg(n));
      const mx = (M) => Math.max(...M.map((r) => Math.max(...r.map(Math.abs))));
      const vmax = Math.max(mx(X), mx(Xs), mx(Xr));
      [
        ["cv-0", X, "f-a"],
        ["cv-1", Xs, "f-w"],
        ["cv-2", Xr, "f-g"],
      ].forEach(([id, M, cls]) => {
        const svg = $(id);
        clear(svg);
        heat(svg, M, { x0: 18, y0: 6, cw: 4.9, ch: 12, vmax: vmax, scale: "sqrt", cls: cls, gap: 0.25 });
        for (let t = 0; t < Tn; t++)
          el("text", { x: 14, y: 6 + t * 12 + 9.5, "text-anchor": "end", "font-size": 8.5, class: t === 3 ? "tw" : "" }, svg, "t" + t);
        el("text", { x: 18, y: 116, "font-size": 9.5, class: "t3" }, svg, "max |·| = " + fmt(mx(M), 2));
      });
      const errs = [];
      for (let t = 0; t < Tn; t++) errs.push(relq(X[t]), relq(Xs[t]), relq(Xr[t]));
      const em = Math.max(...errs) * 1.1;
      barChart($("cv-3"), errs, {
        groups: 3,
        H: 120,
        B: 18,
        ymin: 0,
        ymax: em,
        cls: (k) => ["f-3", "f-w", "f-g"][k % 3],
        ticks: [0, em / 2.2],
        tickFmt: (t) => (t * 100).toFixed(1) + "%",
        xlabels: Array.from({ length: Tn }, (_, t) => [t, "t" + t, t === 3 ? "tw" : ""]),
      });
      $("cv-stats").innerHTML =
        stat("max|X| 原始", fmt(mx(X), 2)) +
        stat("Sylvester", fmt(mx(Xs), 2), "w") +
        stat("正则", fmt(mx(Xr), 2), "g") +
        stat(
          "token 3 相对误差（原始 / Syl / 正则）",
          (errs[9] * 100).toFixed(2) + "% / " + (errs[10] * 100).toFixed(2) + "% / " + (errs[11] * 100).toFixed(2) + "%"
        );
    }
    ["cv-row", "cv-col", "cv-n"].forEach((id) => $(id).addEventListener("input", run));
    run();
  })();
})();
