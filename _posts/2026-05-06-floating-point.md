---
layout: post
title: 揭开浮点数的面纱
date: 2026-05-06 19:42:00
description: 转载一篇浮点数的记录。
tags: computer floating-point 转载
categories: programming
giscus_comments: false
related_posts: false
pretty_table: true

toc:
  beginning: true
---

> 本文是对 Bartosz Ciechanowski 的 [Exposing Floating Point](https://ciechanow.ski/exposing-floating-point/) 的中文译介与整理。原文配套的交互工具 [float.exposed](https://float.exposed/) 很适合拿来检查半精度、单精度和双精度浮点数的真实编码。
> {: .block-tip }

<style>
/* ============================================================
   Floating-point post styles
   - 使用半透明背景 + CSS 变量，适配 light / dark 主题
   - 所有面板 (.fp-panel) 有统一的卡片观感
   ============================================================ */

:root {
  --fp-sign-fg:     #b91c1c;
  --fp-exp-fg:      #1d4ed8;
  --fp-frac-fg:     #15803d;
  --fp-implicit-fg: #b45309;
  --fp-mute-fg:     #475569;
  --fp-rule:        rgba(15, 23, 42, 0.10);
  --fp-card-bg:     rgba(127, 127, 127, 0.04);
  --fp-soft-bg:     rgba(127, 127, 127, 0.08);
  --fp-tick-color:  #475569;
  --fp-tick-tall:   #0f766e;
  --fp-axis-color:  #94a3b8;
}
html[data-theme="dark"] {
  --fp-sign-fg:     #fca5a5;
  --fp-exp-fg:      #93c5fd;
  --fp-frac-fg:     #86efac;
  --fp-implicit-fg: #fcd34d;
  --fp-mute-fg:     #cbd5e1;
  --fp-rule:        rgba(255, 255, 255, 0.14);
  --fp-card-bg:     rgba(255, 255, 255, 0.03);
  --fp-soft-bg:     rgba(255, 255, 255, 0.06);
  --fp-tick-color:  #94a3b8;
  --fp-tick-tall:   #2dd4bf;
  --fp-axis-color:  #64748b;
}

.fp-panel {
  border: 1px solid var(--fp-rule);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin: 1.1rem 0 1.4rem;
  background: var(--fp-card-bg);
  overflow-x: auto;
}

.fp-panel + .fp-panel {
  margin-top: 0.6rem;
}

/* ---- 数位 / 位串 通用 ---- */
.fp-place-row,
.fp-bits,
.fp-equation {
  display: flex;
  align-items: stretch;
  gap: 0.35rem;
  min-width: max-content;
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
}

.fp-place {
  min-width: 3.2rem;
  text-align: center;
  border-radius: 6px;
  border: 1px solid var(--fp-rule);
  padding: 0.4rem 0.5rem;
  background: var(--fp-soft-bg);
}

.fp-place .digit {
  display: block;
  font-size: 1.2rem;
  font-weight: 700;
  line-height: 1.1;
}

.fp-place .power {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: var(--fp-mute-fg);
}

.fp-dot {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0.8rem;
  font-size: 1.45rem;
  font-weight: 700;
}

.fp-number {
  display: inline-flex;
  align-items: baseline;
  gap: 0.2rem;
  padding: 0.4rem 0.65rem;
  border-radius: 6px;
  border: 1px solid var(--fp-rule);
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
  background: var(--fp-soft-bg);
  white-space: nowrap;
}

/* ---- 符号 / 指数 / 尾数 三段着色 ---- */
.fp-sign,
.fp-exp,
.fp-frac,
.fp-implicit {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  padding: 0.5rem 0.7rem;
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.96rem;
  line-height: 1.25;
  letter-spacing: 0;
  white-space: nowrap;
  border: 1px solid var(--fp-rule);
}

.fp-sign     { background: rgba(220, 38, 38, 0.10);  color: var(--fp-sign-fg); }
.fp-exp      { background: rgba( 59,130,246, 0.10);  color: var(--fp-exp-fg); }
.fp-frac     { background: rgba( 34,197, 94, 0.10);  color: var(--fp-frac-fg); }
.fp-implicit { background: rgba(245,158, 11, 0.12);  color: var(--fp-implicit-fg); }

.fp-label {
  margin-top: 0.3rem;
  font-size: 0.7rem;
  letter-spacing: 0.04em;
  color: currentColor;
  opacity: 0.78;
  text-transform: uppercase;
}

.fp-caption {
  margin-top: 0.6rem;
  font-size: 0.86rem;
  color: var(--fp-mute-fg);
}

/* ---- 关键要点 callout ---- */
.fp-key {
  border-left: 3px solid var(--fp-exp-fg);
  background: var(--fp-soft-bg);
  padding: 0.7rem 0.95rem;
  margin: 1rem 0 1.3rem;
  border-radius: 0 8px 8px 0;
  font-size: 0.94rem;
}
.fp-key strong { color: var(--fp-exp-fg); }

/* ---- key/value 速查表 ---- */
.fp-map {
  display: grid;
  grid-template-columns: 8.5rem 1fr;
  gap: 0.45rem 0.85rem;
  align-items: center;
  min-width: 32rem;
}
.fp-map-code {
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
  border-radius: 6px;
  padding: 0.35rem 0.55rem;
  background: var(--fp-soft-bg);
  white-space: nowrap;
}

/* ---- SVG 间距图 ---- */
.fp-spacing-svg {
  display: block;
  width: 100%;
  height: auto;
  max-width: 720px;
  margin: 0 auto;
}
.fp-spacing-svg .axis  { stroke: var(--fp-axis-color); stroke-width: 1; }
.fp-spacing-svg .tick  { stroke: var(--fp-tick-color); stroke-width: 1; }
.fp-spacing-svg .tickB { stroke: var(--fp-tick-tall);  stroke-width: 1.6; }
.fp-spacing-svg text   { fill: var(--fp-mute-fg); font-size: 11px;
                         font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace; }

/* ---- 交互 widget ---- */
.fp-widget {
  display: grid;
  gap: 0.7rem;
}
.fp-widget-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  align-items: center;
}
.fp-widget-controls label {
  font-size: 0.85rem;
  color: var(--fp-mute-fg);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin: 0;
}
.fp-widget input[type="text"],
.fp-widget select {
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
  padding: 0.3rem 0.55rem;
  border-radius: 6px;
  border: 1px solid var(--fp-rule);
  background: var(--fp-soft-bg);
  color: inherit;
}
.fp-widget input[type="text"] { min-width: 12rem; }
.fp-widget-bits {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.fp-widget-bit {
  display: inline-block;
  width: 1.15rem;
  text-align: center;
  font-family: "Comic Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.95rem;
  line-height: 1.6;
  border-radius: 4px;
  border: 1px solid var(--fp-rule);
}
.fp-widget-bit.s { background: rgba(220, 38,  38, 0.14); color: var(--fp-sign-fg); }
.fp-widget-bit.e { background: rgba( 59,130, 246, 0.14); color: var(--fp-exp-fg); }
.fp-widget-bit.f { background: rgba( 34,197,  94, 0.14); color: var(--fp-frac-fg); }
.fp-widget-info {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.3rem 0.9rem;
  font-size: 0.92rem;
}
.fp-widget-info code {
  font-size: 0.92rem;
  word-break: break-all;
}
.fp-widget-info dt { color: var(--fp-mute-fg); margin: 0; }
.fp-widget-info dd { margin: 0; }
.fp-widget-tag {
  display: inline-block;
  padding: 0.05rem 0.45rem;
  border-radius: 999px;
  font-size: 0.75rem;
  background: var(--fp-soft-bg);
  border: 1px solid var(--fp-rule);
  color: var(--fp-mute-fg);
  margin-left: 0.4rem;
}
</style>

浮点数最容易让人产生“玄学感”的地方，是它明明用起来像实数，却又会在某些地方表现得很不像实数：

- `0.1 + 0.2` 不一定等于你眼里的 `0.3`
- 很大的整数之间会突然出现空洞
- `NaN` 甚至不等于它自己
- `0.0` 和 `-0.0` 编码不同，但比较时又相等

如果从 IEEE 754 的二进制科学记数法开始看，这些现象并不神秘。浮点数本质上就是：

<div class="fp-key">
  <strong>一句话总结：</strong>用<em>固定长度</em>的二进制有效数字，配上一个有<em>范围限制</em>的二进制指数，去近似表示实数。所有奇怪的行为，几乎都来自这两个“有限”。
</div>

为了让阅读时方便随手验证，文末的工具栏里也内嵌了一个简化版的 [float.exposed](https://float.exposed/)：你输入一个十进制数，下面会展示它对应的位串和实际存储值。

## 一、从十进制写法开始

我们平时写 `327.849`，其实是在写每一位数字对应的十进制权重：

<div class="fp-panel">
  <div class="fp-place-row">
    <div class="fp-place"><span class="digit">3</span><span class="power">10<sup>2</sup></span></div>
    <div class="fp-place"><span class="digit">2</span><span class="power">10<sup>1</sup></span></div>
    <div class="fp-place"><span class="digit">7</span><span class="power">10<sup>0</sup></span></div>
    <div class="fp-dot">.</div>
    <div class="fp-place"><span class="digit">8</span><span class="power">10<sup>-1</sup></span></div>
    <div class="fp-place"><span class="digit">4</span><span class="power">10<sup>-2</sup></span></div>
    <div class="fp-place"><span class="digit">9</span><span class="power">10<sup>-3</sup></span></div>
  </div>
  <div class="fp-caption"><code>3·10² + 2·10¹ + 7·10⁰ + 8·10⁻¹ + 4·10⁻² + 9·10⁻³ = 327.849</code></div>
</div>

这种写法很自然，但它有几个小麻烦：

- 特别小的数会有一长串前导零，例如 `0.000000000653`
- 特别大的数不容易一眼看出量级，例如 `7298345251`
- 尾部的 `0` 既不省空间，也不方便看精度，例如 `7298000000`

科学记数法把小数点移动到第一个非零数字之后，再用指数记录移动了多少位：

<div class="fp-panel">
  <span class="fp-number">+3.27849 × 10<sup>2</sup></span>
  <div class="fp-caption">符号是 <code>+</code>，有效数字是 <code>3.27849</code>，指数是 <code>2</code>，底数是 <code>10</code>。</div>
</div>

如果只需要保留 4 位有效数字，`327.849` 可以近似为：

<div class="fp-panel">
  <span class="fp-number">+3.278 × 10<sup>2</sup></span>
</div>

所谓“精度”，就是我们愿意保留多少位有效数字。

## 二、二进制也一样

二进制和十进制没有本质区别，只是底数从 `10` 变成了 `2`。例如：

<div class="fp-panel">
  <div class="fp-place-row">
    <div class="fp-place"><span class="digit">1</span><span class="power">2<sup>3</sup></span></div>
    <div class="fp-place"><span class="digit">0</span><span class="power">2<sup>2</sup></span></div>
    <div class="fp-place"><span class="digit">0</span><span class="power">2<sup>1</sup></span></div>
    <div class="fp-place"><span class="digit">1</span><span class="power">2<sup>0</sup></span></div>
    <div class="fp-dot">.</div>
    <div class="fp-place"><span class="digit">0</span><span class="power">2<sup>-1</sup></span></div>
    <div class="fp-place"><span class="digit">1</span><span class="power">2<sup>-2</sup></span></div>
    <div class="fp-place"><span class="digit">0</span><span class="power">2<sup>-3</sup></span></div>
    <div class="fp-place"><span class="digit">1</span><span class="power">2<sup>-4</sup></span></div>
  </div>
  <div class="fp-caption"><code>1001.0101₂ = 8 + 1 + 0.25 + 0.0625 = 9.3125</code></div>
</div>

同样可以写成二进制科学记数法：

<div class="fp-panel">
  <span class="fp-number">+1.0010101 × 2<sup>3</sup></span>
</div>

二进制科学记数法有一个非常重要的特点：**只要数不是 `0`，规格化后最高位一定是 `1`**。因为二进制里非零数字只有 `1`。这个事实让 IEEE 754 偷到了一个很值钱的 bit——既然第一位必然是 `1`，那么这个 `1` 就不需要真的存进编码里。

## 三、浮点数到底是什么

浮点数可以看作有两条限制的二进制科学记数法：

- 有效数字的位数有限
- 指数的范围有限

以 IEEE 754 单精度 `float` 为例：

| 项目         | `float`            |
| :----------- | :----------------- |
| 总长度       | 32 bit             |
| 有效数字精度 | 24 bit（含隐含位） |
| 指数字段     | 8 bit              |
| 尾数字段     | 23 bit             |
| 实际指数范围 | `[-126, +127]`     |
| 偏置值       | `127`              |

“有效数字精度是 24 bit”，但尾数字段只有 23 bit，差出来的那一位就是上文那个一定为 `1` 的最高位——**隐含位（implicit bit）**，不显式存储。

例如下面这个数刚好可以被 `float` 精确表示：

<div class="fp-panel">
  <span class="fp-number">-1.00101100110110001101001 × 2<sup>19</sup></span>
  <div class="fp-caption">小数点后正好 23 位，加上隐含的开头 <code>1</code>，刚好用满 24 位有效数字。</div>
</div>

但不是所有十进制小数都能被二进制有限表示。比如 `0.2` 的二进制有效数字会无限循环：

<div class="fp-panel">
  <span class="fp-number">+1.10011001100110011001100… × 2<sup>-3</sup></span>
  <div class="fp-caption"><code>1001</code> 这段会反复出现，有限长度的 <code>float</code> 只能截断并舍入。</div>
</div>

因此 `0.2f` 实际存下来的并不是数学上精确的 `0.2`，而是非常接近它的：

```text
0.20000000298023223876953125
```

这并不是实现“算错了”，而是有限 bit 装不下无限循环的二进制展开。著名的 `0.1 + 0.2 != 0.3` 也是同样的原因——`0.1`、`0.2`、`0.3` 都是无限循环二进制小数，被分别舍入后再相加，结果落在了 `0.3` 隔壁的另一个 `float` 上。

## 四、把一个数编码成 float

用原文中的例子：`-2343.53125`。它在二进制下是：

<div class="fp-panel">
  <span class="fp-number">-100100100111.10001<sub>2</sub></span>
</div>

规格化成二进制科学记数法：

<div class="fp-panel">
  <span class="fp-number">-1.0010010011110001 × 2<sup>11</sup></span>
</div>

一个 `float` 的 32 bit 被分成三段：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">s<span class="fp-label">sign · 1 bit</span></span>
    <span class="fp-exp">eeeeeeee<span class="fp-label">exponent · 8 bit</span></span>
    <span class="fp-frac">fffffffffffffffffffffff<span class="fp-label">fraction · 23 bit</span></span>
  </div>
</div>

### 1. 符号位

IEEE 754 用 `0` 表示正数，`1` 表示负数。本例的数是负数，符号位为：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">1<span class="fp-label">sign</span></span>
  </div>
</div>

### 2. 尾数字段

规格化形式是 `1.0010010011110001`。开头的 `1` 是隐含位，不写入编码，只存小数点后面的部分，并在末尾补零到 23 bit：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-implicit">1<span class="fp-label">implicit</span></span>
    <span class="fp-frac">00100100111100010000000<span class="fp-label">fraction</span></span>
  </div>
</div>

### 3. 指数字段

真实指数是 `11`，但指数字段不能直接存带符号整数。`float` 使用偏置值 `127` 把指数挪到非负区间：

```text
biased exponent = 11 + 127 = 138 = 10001010₂
```

所以指数字段是：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-exp">10001010<span class="fp-label">exponent</span></span>
  </div>
</div>

### 4. 拼起来

把三段按 `sign | exponent | fraction` 的顺序排好：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">1<span class="fp-label">sign</span></span>
    <span class="fp-exp">10001010<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00100100111100010000000<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption">连起来就是 <code>1100&nbsp;0101&nbsp;0001&nbsp;0010&nbsp;0111&nbsp;1000&nbsp;1000&nbsp;0000</code>，十六进制 <code>0xC512&nbsp;7880</code>。</div>
</div>

把这组 bit pattern 用 C/C++、LLDB 或在线工具反向解释，都能得到同一个 `float`。**浮点数不是“把十进制字符串存起来”，而是把二进制科学记数法拆成这三段存起来。**

### 5. 边敲边看

下面这个小工具可以让你随手验证。换一个数字试试 `0.1`、`0.2`、`16777217`、`1e30`，看看符号 / 指数 / 尾数三段各是什么：

<div class="fp-panel fp-widget" id="fp-widget" data-mode="float32">
  <div class="fp-widget-controls">
    <label>类型
      <select id="fp-widget-type">
        <option value="float32" selected>float (32 bit)</option>
        <option value="float64">double (64 bit)</option>
      </select>
    </label>
    <label>十进制输入
      <input type="text" id="fp-widget-input" value="-2343.53125" spellcheck="false">
    </label>
    <span class="fp-widget-tag" id="fp-widget-tag">normal</span>
  </div>
  <div class="fp-widget-bits" id="fp-widget-bits"></div>
  <dl class="fp-widget-info">
    <dt>真实指数</dt>      <dd id="fp-widget-exp">—</dd>
    <dt>规格化形式</dt>    <dd id="fp-widget-norm">—</dd>
    <dt>实际存储值</dt>    <dd><code id="fp-widget-stored">—</code></dd>
    <dt>十六进制 bits</dt> <dd><code id="fp-widget-hex">—</code></dd>
    <dt>与输入的差</dt>    <dd><code id="fp-widget-err">—</code></dd>
  </dl>
</div>

<script>
(function () {
  if (typeof window === 'undefined') return;
  var input  = document.getElementById('fp-widget-input');
  var typeEl = document.getElementById('fp-widget-type');
  var bitsEl = document.getElementById('fp-widget-bits');
  var hexEl  = document.getElementById('fp-widget-hex');
  var storedEl = document.getElementById('fp-widget-stored');
  var expEl    = document.getElementById('fp-widget-exp');
  var normEl   = document.getElementById('fp-widget-norm');
  var errEl    = document.getElementById('fp-widget-err');
  var tagEl    = document.getElementById('fp-widget-tag');
  if (!input || !typeEl || !bitsEl) return;

  function bitsForFloat32(num) {
    var fa = new Float32Array(1); fa[0] = num;
    var ua = new Uint32Array(fa.buffer);
    var u = ua[0];
    var bits = u.toString(2).padStart(32, '0');
    return { bits: bits, signLen: 1, expLen: 8, fracLen: 23, bias: 127, stored: fa[0], rawHex: '0x' + u.toString(16).padStart(8, '0') };
  }
  function bitsForFloat64(num) {
    var fa = new Float64Array(1); fa[0] = num;
    var ua = new Uint32Array(fa.buffer);
    // little-endian: ua[1] = high, ua[0] = low
    var hi = ua[1], lo = ua[0];
    var bits = hi.toString(2).padStart(32, '0') + lo.toString(2).padStart(32, '0');
    var hex = '0x' + hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
    return { bits: bits, signLen: 1, expLen: 11, fracLen: 52, bias: 1023, stored: fa[0], rawHex: hex };
  }

  function classify(b) {
    var expBits  = b.bits.substr(b.signLen, b.expLen);
    var fracBits = b.bits.substr(b.signLen + b.expLen);
    var allOneExp  = /^1+$/.test(expBits);
    var allZeroExp = /^0+$/.test(expBits);
    var allZeroFrac = /^0+$/.test(fracBits);
    if (allOneExp && allZeroFrac) return 'infinity';
    if (allOneExp && !allZeroFrac) return 'NaN';
    if (allZeroExp && allZeroFrac) return 'zero';
    if (allZeroExp && !allZeroFrac) return 'subnormal';
    return 'normal';
  }

  function fmtNumber(x) {
    if (Number.isNaN(x)) return 'NaN';
    if (!Number.isFinite(x)) return (x < 0 ? '-' : '+') + '∞';
    // 显示足够多有效位以观察舍入差异
    return x.toPrecision(17).replace(/0+e/, 'e').replace(/\.?0+$/, function (s) { return s.indexOf('e') >= 0 ? s : ''; });
  }

  function update() {
    var raw = (input.value || '').trim();
    var num = Number(raw);
    if (raw === '' || Number.isNaN(num) && raw.toLowerCase() !== 'nan') {
      bitsEl.innerHTML = '';
      hexEl.textContent = '—';
      storedEl.textContent = '—';
      expEl.textContent = '—';
      normEl.textContent = '—';
      errEl.textContent = '—';
      tagEl.textContent = 'invalid';
      return;
    }
    var b = (typeEl.value === 'float64') ? bitsForFloat64(num) : bitsForFloat32(num);
    var kind = classify(b);
    tagEl.textContent = kind;

    var html = '';
    for (var i = 0; i < b.bits.length; i++) {
      var cls = (i < b.signLen) ? 's' : (i < b.signLen + b.expLen) ? 'e' : 'f';
      html += '<span class="fp-widget-bit ' + cls + '">' + b.bits[i] + '</span>';
    }
    bitsEl.innerHTML = html;
    hexEl.textContent = b.rawHex;
    storedEl.textContent = fmtNumber(b.stored);

    // 解析真实指数与规格化展示
    var expBits  = b.bits.substr(b.signLen, b.expLen);
    var fracBits = b.bits.substr(b.signLen + b.expLen);
    var rawExp   = parseInt(expBits, 2);
    if (kind === 'normal') {
      var realExp = rawExp - b.bias;
      expEl.textContent = realExp + ' (= ' + rawExp + ' - ' + b.bias + ')';
      normEl.textContent = (b.bits[0] === '1' ? '-' : '+') + '1.' + fracBits + ' × 2^' + realExp;
    } else if (kind === 'subnormal') {
      var realExpS = 1 - b.bias;
      expEl.textContent = realExpS + ' (subnormal, 隐含位为 0)';
      normEl.textContent = (b.bits[0] === '1' ? '-' : '+') + '0.' + fracBits + ' × 2^' + realExpS;
    } else if (kind === 'zero') {
      expEl.textContent = '—';
      normEl.textContent = (b.bits[0] === '1' ? '-' : '+') + '0';
    } else {
      expEl.textContent = '—';
      normEl.textContent = kind;
    }

    // 误差
    if (Number.isFinite(b.stored) && Number.isFinite(num)) {
      var diff = b.stored - num;
      errEl.textContent = (diff === 0) ? '0 (精确表示)' : (diff.toExponential(3));
    } else {
      errEl.textContent = '—';
    }
  }

  input.addEventListener('input', update);
  typeEl.addEventListener('change', update);
  update();
})();
</script>

## 五、特殊值

`float` 的 8 bit 指数字段有 `0..255` 共 256 种编码。普通规格化数只使用其中的 `1..254`。剩下的两个值 `0` 和 `255` 被用来表示特殊情况。

### 1. 一张速查图

<div class="fp-panel">
  <div class="fp-map">
    <strong>普通数</strong>
    <div class="fp-map-code">sign | exponent = 1..254 | fraction = any</div>
    <strong>±0</strong>
    <div class="fp-map-code">sign | 00000000 | 00000000000000000000000</div>
    <strong>次正规数</strong>
    <div class="fp-map-code">sign | 00000000 | fraction != 0</div>
    <strong>±∞</strong>
    <div class="fp-map-code">sign | 11111111 | 00000000000000000000000</div>
    <strong>NaN</strong>
    <div class="fp-map-code">sign | 11111111 | fraction != 0</div>
  </div>
</div>

### 2. 正零和负零

当指数字段全 `0`，尾数字段也全 `0` 时，值是 `+0.0` 或 `-0.0`，由符号位决定：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">_<span class="fp-label">sign</span></span>
    <span class="fp-exp">00000000<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00000000000000000000000<span class="fp-label">fraction</span></span>
  </div>
</div>

`0.0 == -0.0` 为真，但它们的 32 bit 编码不同。负零常常用来保留“从负方向下溢到 0”这类符号信息——例如一个非常小的负数除以一个非常大的正数，可能会得到 `-0.0`。

### 3. 无穷大

当指数字段全 `1`，尾数字段全 `0` 时，值是正负无穷：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">_<span class="fp-label">sign</span></span>
    <span class="fp-exp">11111111<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00000000000000000000000<span class="fp-label">fraction</span></span>
  </div>
</div>

无穷大通常来自上溢，或者来自除以带符号的零：正数除以 `+0.0` 得 `+∞`，除以 `-0.0` 得 `-∞`。有限数和无穷大做加减乘除时，大多数结果都符合直觉——`∞ + 1` 仍然是 `∞`，`∞ × -1` 是 `-∞`。

### 4. NaN

当指数字段全 `1`，尾数字段非零时，值是 NaN（Not a Number）：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">_<span class="fp-label">sign</span></span>
    <span class="fp-exp">11111111<span class="fp-label">exponent</span></span>
    <span class="fp-frac">at least one 1<span class="fp-label">fraction</span></span>
  </div>
</div>

常见会产生 NaN 的操作：

- `0 × ∞`
- `+∞ + (-∞)`
- `0 / 0`
- `∞ / ∞`
- 对负数开平方

NaN 最反直觉的一点是：**它不等于任何东西，包括它自己**。因此 `x != x` 常被用作判断 NaN 的底层技巧（在 C/C++ 里编译器有时会激进地优化它，更稳妥的写法是 `isnan(x)`）。

### 5. 最大值、最小规格化值与次正规数

`float` 的最大有限值用倒数第二大的指数字段（`254`）和全 `1` 的尾数字段：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">0<span class="fp-label">sign</span></span>
    <span class="fp-exp">11111110<span class="fp-label">exponent</span></span>
    <span class="fp-frac">11111111111111111111111<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption">约为 <code>3.40282347 × 10<sup>38</sup></code>。</div>
</div>

最小的正规格化 `float` 是：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">0<span class="fp-label">sign</span></span>
    <span class="fp-exp">00000001<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00000000000000000000000<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption"><code>2<sup>-126</sup></code>，约 <code>1.17549435 × 10<sup>-38</sup></code>，即 <code>FLT_MIN</code>。</div>
</div>

但 `FLT_MIN` 并不是 `float` 能表示的最小正数。指数编码为 `0`、尾数非零时，进入**次正规数（subnormal / denormal）**区域：真实指数仍按 `-126` 解释，但隐含位从 `1` 变成 `0`：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">0<span class="fp-label">sign</span></span>
    <span class="fp-exp">00000000<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00000000000110001101001<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption">含义是 <code>+0.00000000000110001101001<sub>2</sub> × 2<sup>-126</sup></code>，远小于 <code>FLT_MIN</code>。</div>
</div>

次正规数让浮点数可以**渐进下溢**：从最小规格化数继续往 `0` 靠近时，可表示值不会突然断崖式跳到 `0`，而是均匀地铺满最后一段空间。代价是有效精度逐位缩水，所以浮点性能敏感的代码（特别是 SIMD / GPU 内核）有时会把 FTZ（flush-to-zero）打开，把次正规数当作 `0` 处理。

## 六、浮点数不是连续的

因为有效数字位数有限，`float` 不可能表示任意实数。更重要的是，指数会让可表示数的间隔**并不均匀**——在相邻的两个 2 的幂之间间距是固定的，但每跨过一个 2 的幂，间距都要翻倍。

| 区间                                       | 相邻 `float` 的间距（ULP）          |
| :----------------------------------------- | :---------------------------------- |
| `[0.5, 1.0)`                               | `2`<sup>`-24`</sup>                 |
| `[1.0, 2.0)`                               | `2`<sup>`-23`</sup> = `FLT_EPSILON` |
| `[2`<sup>`23`</sup>`, 2`<sup>`24`</sup>`)` | `1`（每一个整数都能表示）           |
| `[2`<sup>`24`</sup>`, 2`<sup>`25`</sup>`)` | `2`（隔一个整数才能表示一个）       |
| `[2`<sup>`n`</sup>`, 2`<sup>`n+1`</sup>`)` | `2`<sup>`n-23`</sup>                |

下面这张图把这种“随指数翻倍”的稀疏感画出来——每一段 `[2`<sup>`n`</sup>`, 2`<sup>`n+1`</sup>`)` 内部 `float` 是均匀的，但段与段之间的密度差了一个 2 倍：

<div class="fp-panel">
  <svg class="fp-spacing-svg" viewBox="0 0 720 140" role="img" aria-label="float spacing diagram">
    <!-- baseline -->
    <line class="axis" x1="20" y1="80" x2="700" y2="80"/>
    <!-- 主刻度: 1, 2, 4, 8, 16, 32 -->
    <g>
      <line class="tickB" x1="40"  y1="60" x2="40"  y2="100"/>
      <line class="tickB" x1="160" y1="60" x2="160" y2="100"/>
      <line class="tickB" x1="280" y1="60" x2="280" y2="100"/>
      <line class="tickB" x1="400" y1="60" x2="400" y2="100"/>
      <line class="tickB" x1="520" y1="60" x2="520" y2="100"/>
      <line class="tickB" x1="640" y1="60" x2="640" y2="100"/>
      <text x="40"  y="120" text-anchor="middle">2⁰ (1)</text>
      <text x="160" y="120" text-anchor="middle">2¹ (2)</text>
      <text x="280" y="120" text-anchor="middle">2²</text>
      <text x="400" y="120" text-anchor="middle">2³</text>
      <text x="520" y="120" text-anchor="middle">2⁴</text>
      <text x="640" y="120" text-anchor="middle">2⁵</text>
    </g>
    <!-- 段内 12 个均匀刻度（示意，密度依次减半才对应真实 float 间距比例） -->
    <!-- [1,2): 每段宽度 120, 12 个间隔 -->
    <g>
      <!-- [1,2) 12 ticks -->
      <g>
        <line class="tick" x1="50"  y1="68" x2="50"  y2="92"/>
        <line class="tick" x1="60"  y1="68" x2="60"  y2="92"/>
        <line class="tick" x1="70"  y1="68" x2="70"  y2="92"/>
        <line class="tick" x1="80"  y1="68" x2="80"  y2="92"/>
        <line class="tick" x1="90"  y1="68" x2="90"  y2="92"/>
        <line class="tick" x1="100" y1="68" x2="100" y2="92"/>
        <line class="tick" x1="110" y1="68" x2="110" y2="92"/>
        <line class="tick" x1="120" y1="68" x2="120" y2="92"/>
        <line class="tick" x1="130" y1="68" x2="130" y2="92"/>
        <line class="tick" x1="140" y1="68" x2="140" y2="92"/>
        <line class="tick" x1="150" y1="68" x2="150" y2="92"/>
      </g>
      <!-- [2,4): 6 ticks -->
      <g>
        <line class="tick" x1="180" y1="68" x2="180" y2="92"/>
        <line class="tick" x1="200" y1="68" x2="200" y2="92"/>
        <line class="tick" x1="220" y1="68" x2="220" y2="92"/>
        <line class="tick" x1="240" y1="68" x2="240" y2="92"/>
        <line class="tick" x1="260" y1="68" x2="260" y2="92"/>
      </g>
      <!-- [4,8): 3 ticks -->
      <g>
        <line class="tick" x1="310" y1="68" x2="310" y2="92"/>
        <line class="tick" x1="340" y1="68" x2="340" y2="92"/>
        <line class="tick" x1="370" y1="68" x2="370" y2="92"/>
      </g>
      <!-- [8,16): 2 ticks -->
      <g>
        <line class="tick" x1="440" y1="68" x2="440" y2="92"/>
        <line class="tick" x1="480" y1="68" x2="480" y2="92"/>
      </g>
      <!-- [16,32): 1 tick -->
      <g>
        <line class="tick" x1="580" y1="68" x2="580" y2="92"/>
      </g>
    </g>
    <text x="360" y="20" text-anchor="middle">每越过一个 2ⁿ，相邻 float 的间距就翻一倍</text>
  </svg>
  <div class="fp-caption">这是示意图：每一段 <code>[2<sup>n</sup>, 2<sup>n+1</sup>)</code> 内部 <code>float</code> 是均匀的，但越往右整体越稀疏。</div>
</div>

`float` 能逐个表示所有整数，一直到 `2`<sup>`24`</sup>` = 16777216` 为止。下一个可表示的 `float` 是 `16777218`：

<div class="fp-panel">
  <span class="fp-number">16777216 → next float = 16777218</span>
  <div class="fp-caption">中间的 <code>16777217</code> 没有任何 <code>float</code> 编码可以精确表示，会被舍入到相邻两个里更近的那个。</div>
</div>

这就是为什么用 `float` 累加大整数会出现“卡住”的现象：当累加到 `2`<sup>`24`</sup> 之后，每次加 `1` 实际上落到了上一个值，循环永远不会推进。

## 七、把 float 当成整数看

对正的 `float` 来说，如果忽略三段结构，直接把 32 bit 当成无符号整数看，有一个很漂亮的性质：**整数编码加 `1`，通常就会得到下一个可表示的 `float`**。

例如：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">0<span class="fp-label">sign</span></span>
    <span class="fp-exp">10010011<span class="fp-label">exponent</span></span>
    <span class="fp-frac">11111111111111111111111<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption">这是 <code>2097151.875</code>。</div>
</div>

把完整 bit 串当无符号整数加 `1`，尾数会全部进位到指数：

```text
  0 10010011 11111111111111111111111
+ 0 00000000 00000000000000000000001
─────────────────────────────────────
  0 10010100 00000000000000000000000
```

再把结果按 `float` 三段解释：

<div class="fp-panel">
  <div class="fp-bits">
    <span class="fp-sign">0<span class="fp-label">sign</span></span>
    <span class="fp-exp">10010100<span class="fp-label">exponent</span></span>
    <span class="fp-frac">00000000000000000000000<span class="fp-label">fraction</span></span>
  </div>
  <div class="fp-caption">得到下一个可表示值 <code>2097152.0</code>。</div>
</div>

这正是 IEEE 754 把字段排成 `sign | exponent | fraction` 这种顺序的妙处：尾数溢出时会自然进位到指数，恰好对应“跨过一个 2 的幂、间距翻倍”的边界。最大有限值再加一步会变成 `+∞`；最小规格化值往下走会进入次正规数；最小次正规数再往下就是 `0`。同样的性质也让 `memcmp` 可以直接比较两个非负 `float` 的大小（处理负数时还要再翻转一下）。

这个技巧有边界：

- 它不能自动在 `+0.0` 和 `-0.0` 之间跳转
- 无穷大继续加可能进入 NaN 区域
- NaN 的整数序也不该被当作普通数值序

## 八、half、float、double

IEEE 754 的常见二进制浮点类型遵循同一套规则，只是总 bit 数、指数字段和尾数字段大小不同。

| 类型                | 总 bit | 指数字段 | 尾数字段 | 有效数字精度 | 实际指数范围     | 偏置值 |
| :------------------ | -----: | -------: | -------: | -----------: | :--------------- | -----: |
| `half` / binary16   |     16 |        5 |       10 |       11 bit | `[-14, +15]`     |     15 |
| `float` / binary32  |     32 |        8 |       23 |       24 bit | `[-126, +127]`   |    127 |
| `double` / binary64 |     64 |       11 |       52 |       53 bit | `[-1022, +1023]` |   1023 |

下列规则对三者都成立：

- 指数全零 + 尾数全零 → `±0.0`
- 指数全一 + 尾数全零 → `±∞`
- 指数全一 + 尾数非零 → NaN
- 指数全零 + 尾数非零 → 次正规数
- 普通规格化数都有一个不显式存储的开头 `1`

`half` 在图形、深度学习和带宽敏感场景中很常见——它非常省空间，但指数范围和有效精度都小很多。`double` 则用更多 bit 换来更大的范围和更高精度，是 JavaScript `Number`、Python `float`、和大多数科学计算默认使用的类型。

## 九、类型转换

从小类型转到大类型时，值可以保持完全一致。例如 `half → float → double`：新的尾数字段在后面补零，指数按新的偏置值重新编码即可。

从大类型转到小类型时就不一定了：

- 如果尾数多出来的 bit 都是 `0`，并且指数落在目标类型的范围内，值可以精确保留
- 如果尾数装不下，需要舍入
- 如果指数超出目标范围，可能变成 `±∞`
- NaN 仍是 NaN，无穷大仍是无穷大

默认舍入方式是 **round-to-nearest, ties-to-even**（最近偶数舍入）：四舍六入，正中间时取偶数末位。这个规则比日常的“`.5` 一律进位”更不容易产生系统性偏差。举个直观例子：如果两个比例分别是 `72.5%` 和 `27.5%`，普通半入法会得到 `73% + 28% = 101%`；最近偶数法则得到 `72% + 28% = 100%`。

IEEE 754 还定义了另外四种舍入模式（向上、向下、向零、向远离零），它们大多在区间运算或可重现性要求高的场景里出现，C 里通过 `<fenv.h>` 切换，平时用得不多。

## 十、打印浮点数

打印浮点数也有坑。`%f` 和 `%e` 默认不打印足够多的数字，所以两个不同的 `float` 完全可能被打印成一样的文本。

如果想保证“打印出来再读回去仍是同一个 `float`”，至少要 `FLT_DECIMAL_DIG = 9` 位有效数字（`double` 是 `DBL_DECIMAL_DIG = 17`）：

```c
printf("%.9g\n",  some_float);   // 至少 9 位
printf("%.17g\n", some_double);  // 至少 17 位
```

缺点是简单的数也会被打印得很长。

更适合精确表达浮点数的是十六进制浮点格式，也就是 `printf` 的 `%a`：

```text
0x1.810682p+1
0x1.810688p+1
```

这里 `0x` 表示十六进制，`p+1` 后面的是 2 的指数。每个 hex digit 刚好对应 4 个 bit，因此能用很短的字符串无损地表达底层 bit。打印调试时如果怀疑某个数被舍入了，把它打成 `%a` 通常立刻就能看出来。

另一个容易混淆的点是：

<div class="fp-key">
不是每个十进制小数都能被浮点数精确表示；但<strong>每一个已经存在的浮点数都有一个精确的十进制展开</strong>。
</div>

原因是有限二进制小数的分母一定是 `2`<sup>`n`</sup>，而 `2`<sup>`n`</sup>` × 5`<sup>`n`</sup>` = 10`<sup>`n`</sup>。例如 `1/16 = 625/10000 = 0.0625`。只是这个精确十进制展开有时会非常长——`double` 表示的最小次正规数有 700 多位十进制小数，长到完全不适合人读。

## 十一、把要点收束一下

理解浮点数时，记住下面这几件事就基本够用了：

- `float` 不是实数，而是一张有限的、间隔不均匀的数值表
- `float` 的本体是 `sign | exponent | fraction`
- 普通数的真实形式是 `(-1)`<sup>`sign`</sup>` × 1.fraction × 2`<sup>`exponent - bias`</sup>
- 次正规数把隐含的 `1` 改成 `0`，用于平滑靠近 `0`
- 指数全 `1` 的区域留给 `±∞` 和 NaN
- 相邻浮点数的间隔随指数变大而变大，变化粒度叫 **ULP**
- 十进制打印和二进制存储不是一回事，调试精确值时 `%a` 很有用

浮点数确实有很多边角，但它不是一堆例外堆出来的怪物。相反，很多看似奇怪的行为都来自同一个朴素设计：**用有限 bit，把二进制科学记数法编码得尽量紧凑、可比较、可渐进下溢，并为错误和溢出留下可传播的特殊值。**

## 参考

- Bartosz Ciechanowski: [Exposing Floating Point](https://ciechanow.ski/exposing-floating-point/)
- 配套工具：[float.exposed](https://float.exposed/)
- David Goldberg: [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- Bruce Dawson: [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
- Exploring Binary: [Floating-Point Articles](https://www.exploringbinary.com/floating-point-articles/)
