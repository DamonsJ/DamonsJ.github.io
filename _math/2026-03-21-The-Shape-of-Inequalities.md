---
layout: post
title: 几个不等式的几何证明
date: 2026-03-21 08:12:00
description: 介绍几个不等式的可视化几何证明方法
tags: 几何代数 杂项 转载
categories: math
---

这篇文章翻译/整理自 Andrei N. Ciobanu 的博文《The Shape of Inequalities》，原文见：[The Shape of Inequalities](https://www.andreinc.net/2026/03/16/the-shape-of-inequalities/)。

## 引言

作者在网上看到一张很有意思的图（见下，Roland H. Eddy，1985），进而尝试把一些常见不等式用几何图形（圆、三角形、正方形、立方体、长方体……）表达出来，做成动画来帮助形成直觉。下面整理其中几条最经典、也最"好画出来"的例子。

作者也坦言：写完之后意识到，大多数代数不等式其实并不天然"几何友好"——一旦超出基础范围，用圆规和直尺来表达复杂代数真相就变得困难。但正是这种"强行装进容器"的过程，让我们得以窥见数学的"物理骨架"，也让我们看到：**对称性不只是对"漂亮形状"的偏好，它本身就意味着极值**。

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/amgmcapsule.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 1. Roland H. Eddy (1985) 的插图（原文引用图）。
</div>


## HM–GM–AM–QM 不等式链

这是学习不等式时几乎一定会碰到的一条链。对正实数 $$a, b, c > 0$$：

$$
\underbrace{\frac{3}{\dfrac{1}{a}+\dfrac{1}{b}+\dfrac{1}{c}}}_{\text{HM}}
\;\le\;
\underbrace{\sqrt[3]{abc}}_{\text{GM}}
\;\le\;
\underbrace{\frac{a+b+c}{3}}_{\text{AM}}
\;\le\;
\underbrace{\sqrt{\dfrac{a^2+b^2+c^2}{3}}}_{\text{QM}}
$$

两变量版本更直观：

$$
\frac{2}{\dfrac{1}{a}+\dfrac{1}{b}}
\;\le\;
\sqrt{ab}
\;\le\;
\frac{a+b}{2}
\;\le\;
\sqrt{\frac{a^2+b^2}{2}}
$$

其中：

- **HM（Harmonic Mean，调和平均）**：比如去程速度 $$v_1$$、回程速度 $$v_2$$，全程的平均速度不是 $$\frac{v_1+v_2}{2}$$，而是 $$\frac{2}{\frac{1}{v_1}+\frac{1}{v_2}}$$。
- **GM（Geometric Mean，几何平均）**：增长因子的"平均"，适用于缩放和复利场景，和调和平均一样，它在自然界和简单的金融计算中都有出现。举个例子：假如你是一位股票投资者，第一年你的投资组合增长了 $$100\%$$，但第二年市场崩盘，跌了 $$50\%$$，那你的平均增长率是多少？

    - 数学不好的投资者会说：$$\frac{100 + (-50)}{2} = 25%$$。
    - 聪明的投资者会看增长因子：第一年因子是 $$2.0$$，第二年是 $$0.5$$，然后这样求平均：$$\text{平均增长因子} = \sqrt{2.0 \times 0.5} = 1.0$$。
    - 这意味着你的平均增长实际上是 $$0$$——你最终回到了原点。说实话，这已经比大多数交易者强了。

- **AM（Arithmetic Mean，算术平均）**：最常见的平均数 $$\frac{a+b}{2}$$。
- **QM（Quadratic Mean，二次平均 / RMS）**：$$\sqrt{\frac{a^2+b^2}{2}}$$，在电工等场景里常出现（比如交流电压标注的 RMS 值）。

    - 欧洲的电压标注为 230V，但这并不是人们通常以为的"实际平均电压"——这个数值其实是用 RMS 算出来的。

下面用"看得见"的方式把这些量放到同一张几何图里。

---

### 两个圆：AM 和 GM 的同台

设大圆直径为 $$a$$，半径 $$R = \frac{a}{2}$$。再放一个小圆，直径为 $$b$$，半径 $$r = \frac{b}{2}$$，让小圆从外部与大圆相切。

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/twocircles.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 2. 两圆外切：AM 与 GM 的几何呈现。
</div>

把小圆圆心 $$O'$$ 投影到过大圆圆心 $$O$$ 的竖直线，得到点 $$P$$。此时三角形 $$OPO'$$ 是直角三角形，各边长度为：

- 斜边 $$OO' = R + r = \dfrac{a+b}{2}$$（这就是 $$a, b$$ 的 **AM**）
- 水平直角边 $$OP = R - r = \dfrac{a-b}{2}$$
- 竖直直角边 $$O'P$$：

$$
(O'P)^2
= \left(\frac{a+b}{2}\right)^2 - \left(\frac{a-b}{2}\right)^2
= ab
\quad\Longrightarrow\quad
O'P = \sqrt{ab}
$$

也就是说，**GM $$= \sqrt{ab}$$** 被"画"成了直角边，而 **AM $$= \frac{a+b}{2}$$** 是斜边。由于直角边 $$\le$$ 斜边，自然得到：

$$
\sqrt{ab} \;\le\; \frac{a+b}{2}
$$

等号当且仅当 $$a = b$$ 时成立（此时两圆等大，直角边退化为零）。

---

### 半圆：把 HM、GM、AM、QM 全装进一张图

取一个半圆，直径长度为 $$a + b$$，圆心为 $$O$$。在半圆弧上取点 $$P$$，向下投影到直径得到垂足 $$P'$$。则 $$\angle OPP' = 90°$$（直径所对圆周角），且：

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/semicircle.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 3. 半圆图1。
</div>

- $$OP  = \dfrac{a+b}{2}$$（半径，即 **AM**）
- $$OP' = \dfrac{|a-b|}{2}$$
- $$PP' = \sqrt{ab}$$（**GM**），由勾股定理：

$$
PP' = \sqrt{\left(\frac{a+b}{2}\right)^2 - \left(\frac{a-b}{2}\right)^2} = \sqrt{ab}
$$

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/semicircle2.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 4. 半圆图2。
</div>

为了引入 **QM**，在圆心处作与直径垂直的半径 $$OM$$，再连线 $$MP'$$。由：

- $$OM = \dfrac{a+b}{2}$$
- $$OP' = \dfrac{|a-b|}{2}$$

再一次勾股：

$$
MP' = \sqrt{\left(\frac{a+b}{2}\right)^2 + \left(\frac{a-b}{2}\right)^2}
= \sqrt{\frac{a^2+b^2}{2}}
$$

这就是 **QM**。

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/semicircle3.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 5. 半圆图3。
</div>

最后，为了让 **HM** 也"显形"，把 $$P'$$ 向 $$OP$$ 上作垂线，垂足为 $$N$$（即直角三角形 $$OPP'$$ 中，从直角顶点 $$P'$$ 向斜边 $$OP$$ 所作的高的垂足）。利用"射影定理"（高与斜边分段的关系）：

$$
PN = \frac{PP'^2}{OP}
= \frac{\left(\sqrt{ab}\right)^2}{\dfrac{a+b}{2}}
= \frac{2ab}{a+b}
= \frac{2}{\dfrac{1}{a}+\dfrac{1}{b}}
$$

这就是 **HM**。

于是同一张半圆图里同时出现了：

$$
\begin{array}{c|l}
\text{线段} & \text{对应量} \\
\hline
PN & \textbf{HM}\ \text{（调和平均）} \\
PP' & \textbf{GM}\ \text{（几何平均）} \\
OP & \textbf{AM}\ \text{（算术平均）} \\
MP' & \textbf{QM}\ \text{（二次平均）}
\end{array}
$$

几何上几乎一眼能看出：除非 $$a = b$$，这些线段严格满足 $$\text{HM} < \text{GM} < \text{AM} < \text{QM}$$。

---

### "容器"视角：AM–GM 的面积直觉

> 这不是严格意义上的证明，但非常直观。

设 $$a + b$$ 固定。考虑：

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/container.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 6. 面积视角：固定周长时，正方形面积最大。
</div>

- **正方形**：边长为 $$\dfrac{a+b}{2}$$，面积为 $$\left(\dfrac{a+b}{2}\right)^2$$
- **长方形**：边长为 $$a$$ 与 $$b$$，面积为 $$ab$$

当 $$a + b$$ 固定时，正方形面积总是不小于任何长方形面积（只有 $$a = b$$ 时取等）：

$$
\left(\frac{a+b}{2}\right)^2 \;\ge\; ab
\quad\Longrightarrow\quad
\frac{a+b}{2} \;\ge\; \sqrt{ab}
$$

也就是在所有周长相同（即 a+b 固定）的长方形里，正方形的面积最大。
也可以这么证明：

$$
\left(\frac{a+b}{2}\right)^2 - ab
= \frac{(a+b)^2}{4} - ab
= \frac{a^2 + 2ab + b^2 - 4ab}{4}
= \frac{(a-b)^2}{4}
\geq 0
$$


---

### 3D 容器：AM–GM 的体积直觉

把上面的想法从面积升级到体积，固定 $$a + b + c$$：

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/cube.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 7. 体积视角：固定棱长之和时，正方体体积最大。
</div>

- **正方体**：边长为 $$\dfrac{a+b+c}{3}$$，体积为 $$\left(\dfrac{a+b+c}{3}\right)^3$$
- **长方体**：边长为 $$a, b, c$$，体积为 $$abc$$

当三边之和固定时，正方体的体积最大：

$$
\left(\frac{a+b+c}{3}\right)^3 \;\ge\; abc
\quad\Longrightarrow\quad
\frac{a+b+c}{3} \;\ge\; \sqrt[3]{abc}
$$

等号当且仅当 $$a = b = c$$ 时成立。

---

## 平方和不等式（Sum of Squares）

另一个常见且"可视化友好"的不等式：

$$
a^2 + b^2 + c^2 \;\ge\; ab + bc + ca
$$

可以参考证明方法： [方法1](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#pivi08)， [方法2](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#pgtm12)， [方法3](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#pmrt01)， [方法4](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#pcbs04) 。

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/sumofsquares.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 8. 平方和不等式的几何直觉：三个正方形与三个长方形。
</div>

几何直觉：

- 取边长分别为 $$a, b, c$$ 的三个正方形，总面积为 $$a^2 + b^2 + c^2$$（不等式左边）。
- 把这三个正方形拼合后，在拼接的图形里画辅助线，可以识别出面积分别为 $$ab, bc, ca$$ 的三个长方形，总面积之和为 $$ab + bc + ca$$（不等式右边）。

当 $$b, c$$ 慢慢增大至与 $$a$$ 相等时，两边的差距逐渐缩小，直到 $$a = b = c$$ 时两边相等，三个正方形完全对齐——不等式取等。

---

## Nesbitt 不等式：用三角形里的距离来"变形"

Nesbitt 不等式是一个经典结论：

$$
\frac{a}{b+c} + \frac{b}{c+a} + \frac{c}{a+b} \;\ge\; \frac{3}{2}
$$

可以参考证明方法： [方法1](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#nesbitts-inequality)， [方法2](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#pcbs12)， [方法3](https://www.andreinc.net/2025/03/17/the-trickonometry-of-math-olympiad-inequalities/#an-interesting-refinement-for-nesbitts-inequality)。

<div class="row mt-3 justify-content-center">
  <div class="col-lg-10 col-md-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/math/inequalities/nesbitt.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Fig. 9. Nesbitt 不等式：通过等边三角形内点到三边距离来变形证明。
</div>

作者给出的思路以 **[Viviani 定理](https://en.wikipedia.org/wiki/Viviani%27s_theorem)**为几何骨架：

- 在等边三角形内取一点 $$Q$$，到三边的距离分别为 $$x, y, z$$；
- Viviani 定理：$$x + y + z = h$$（$$h$$ 为三角形的高，为常数）。

做变量替换：

$$
a = y + z, \quad b = x + z, \quad c = x + y
$$

注意到 $$a + b + c = 2(x + y + z) = 2h$$，从而：

$$
b + c = 2h - a = h + (h - a) = h + x
$$

类似地，$$c + a = h + y$$，$$a + b = h + z$$。

将 Nesbitt 不等式的每一项变形：

$$
\frac{a}{b+c} = \frac{h - x}{h + x}, \quad
\frac{b}{c+a} = \frac{h - y}{h + y}, \quad
\frac{c}{a+b} = \frac{h - z}{h + z}
$$

Nesbitt 不等式等价于：

$$
\frac{h-x}{h+x} + \frac{h-y}{h+y} + \frac{h-z}{h+z} \;\ge\; \frac{3}{2}
$$

几何直觉：
- 当 $$Q$$ 在三角形中心时，$$x = y = z = h/3$$，每一项均为 $$\frac{h - h/3}{h + h/3} = \frac{2/3}{4/3} = \frac{1}{2}$$，左边恰好等于 $$\frac{3}{2}$$，取等。
- 点 $$Q$$ 一旦偏离中心，某些距离变大、某些变小，但总和只会增大——这给出了一个直观的几何解释：**为什么中心对应最小值**。

---

## 总结

把代数不等式硬塞进圆、方形、棱柱这些几何容器里，常常能得到一种"机械式"的直觉：某些长度/面积/体积天然被几何边界约束住，等号对应最对称的形态。

诚然，很多更复杂的不等式未必这么"几何友好"，但这种方法能帮助我们记住：

> **对称性往往意味着极值**——并且通常对应等号成立的情况。

---
