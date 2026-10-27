---
layout: post
title: LLM 与 AIGC 量化：从数值格式到 W4A4
date: 2026-10-27 12:00:00
description: 从比特、舍入和矩阵乘讲起，整理 GPTQ、AWQ、SmoothQuant、QuaRot/SpinQuant、ConvRot、SVDQuant 分别解决什么问题、怎么做、在 GPU 上怎么跑，以及扩散模型、KV Cache、FP8/FP4 的量化
tags: [模型量化, LLM, Diffusion, 推理优化]
categories: [模型优化]
pretty_table: true
toc:
  sidebar: left
---

## 为什么？

前面两篇 AIGC 推理加速，[分布式推理](https://valdrada.cn/AI/2026-08-08-distribute-aigc.html)解决的是"一步怎么在多张卡上算得更快"，[特征缓存](https://valdrada.cn/AI/2026-10-22-diffusion-feature-cache.html)解决的是"能不能少算几步"。还有一个维度一直没有系统地整理过：每一次乘加本身能不能更便宜。这就是量化。

之前写过一篇[量化工具调研](https://valdrada.cn/AI/2026-04-15-quantization_tools_research.html)，那篇主要是工具之间的横向对比，工具里那些名字，GPTQ、AWQ、SmoothQuant、SVDQuant，到底各自在做什么，当时并没有想得很清楚。实际做项目的时候又会碰到一连串的问题：为什么 W4A16 对 LLM 的 decode 很有用，换到 DiT 上几乎不提速？为什么 Hopper 上大家都直接上 FP8，不怎么折腾 SmoothQuant？为什么 LLM 上好用的 QuaRot 搬到 FLUX 上会崩？这些问题背后其实是同一套东西：数值格式、scale 怎么取、矩阵乘里哪些 scale 能提出来、离群值从哪里来。

所以这篇把量化从头到尾理一遍，大概是下面这个顺序：

1. 量化到底省的是什么：显存、带宽还是算力，什么场景下省哪一个；
2. 低比特能表示哪些数，怎么把一组实数装进去，误差从哪里来；
3. 量化之后的矩阵乘在 GPU 上是怎么执行的；
4. LLM 量化难在哪里，以及 GPTQ、AWQ、SmoothQuant、QuaRot/SpinQuant、ConvRot、SVDQuant 这几个经典算法分别怎么解决；
5. 扩散模型、KV Cache、FP8/FP4 这些落地时绕不开的问题。

先约定一下记号。线性层统一写成 PyTorch 的形式 $$Y = XW^\top$$，激活 $$X\in\mathbb{R}^{T\times K}$$（T 个 token，K 个输入通道），权重 $$W\in\mathbb{R}^{N\times K}$$（N 个输出通道，就是 `nn.Linear.weight` 的形状）。有些论文用 $$W\in\mathbb{R}^{C_{in}\times C_{out}}$$，遇到的时候会单独说明。`W4A16` 表示权重 4 bit、激活 16 bit，`W8A8` 表示两者都是 8 bit，`KV4` 表示 KV Cache 4 bit，`g128` 表示每 128 个元素共享一组 scale。文中的手算例题都可以用 NumPy 复现，舍入方式是 round-half-to-even。交互实验在浏览器里实时计算，用的是固定随机种子的合成数据，用来演示原理，不代表真实模型的数值。

## 量化能省什么：显存、带宽与算力

量化（Quantization）就是用更少的比特来表示权重、激活或者 KV Cache。要知道它能带来什么收益，得先知道推理的开销花在了哪里。

### 推理的成本从哪里来

一次前向会同时消耗 GPU 的三种资源，哪一种不够都会成为瓶颈：

- **显存容量**：权重、KV Cache、中间激活必须放得下，放不下就只能多卡切分或者 offload；
- **显存带宽**：计算之前要把数据从 HBM 搬到计算单元，搬的字节越多越慢；
- **算力**：Tensor Core 每秒能做多少次乘加，和数据精度直接相关，INT8/FP8 的峰值是 FP16 的 2 倍，INT4/FP4 是 4 倍。

哪种资源先耗尽，取决于**一次前向处理多少个 token**：

| 负载                               | 一次前向处理的 token 数           | 主要瓶颈                        |
| :--------------------------------- | :-------------------------------- | :------------------------------ |
| LLM prefill（处理输入的 prompt）   | 整段 prompt，几百到几千个         | 算力                            |
| LLM decode（逐个生成新 token）     | 每个序列 1 个，共 batch 个        | 显存带宽（读权重、读 KV Cache） |
| 扩散模型的每个去噪步（DiT / UNet） | 全部图像/视频 token，几千到几万个 | 算力                            |

### 显存：权重和 KV Cache 的账本

权重显存就是参数量乘以每个参数的字节数。INT4 分组量化（g128）还要额外存 scale 和 zero，等效大约 4.25 bit：

| 模型 | FP16 / BF16 | INT8 / FP8 | INT4 g128（≈4.25 bit） |
| :--- | :---------- | :--------- | :--------------------- |
| 7B   | 14.0 GB     | 7.0 GB     | 3.7 GB                 |
| 13B  | 26.0 GB     | 13.0 GB    | 6.9 GB                 |
| 70B  | 140 GB      | 70 GB      | 37.2 GB                |

LLM 还要存 KV Cache：每个 token 在每一层的 K 和 V 都要缓存下来给后面的 token 做 attention，它随序列长度和 batch 线性增长：

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times L_{\text{seq}} \times B \times \text{bytes/elem}
$$

以 Llama-2-7B 为例：32 层、32 个 KV head（MHA）、每个 head 128 维，序列长度 4096、batch=1、FP16：

$$
2\times 32\times 32\times 128\times 4096\times 1\times 2\ \text{B} = 2\ \text{GiB}
$$

batch 开到 32 就是 64 GiB，远远超过 13.5 GB 的权重本身。换成 GQA 的 Llama-3-8B（8 个 KV head），同样的配置每条序列是 512 MiB。所以长上下文、高并发的场景下，KV Cache 的量化和权重量化一样重要。

### 带宽与算力：Roofline

放得下之后，速度由带宽和算力一起决定。衡量一个算子更依赖哪一个，用的是**算术强度**（Arithmetic Intensity，AI），也就是每读写 1 字节显存做多少次浮点运算：

$$
\text{AI}=\frac{\text{FLOPs}}{\text{Bytes}},\qquad \text{可达性能}=\min\big(\text{峰值算力},\ \text{带宽}\times\text{AI}\big)
$$

这就是 Roofline 模型。两条"屋顶"的交点叫拐点（峰值算力 ÷ 带宽）：AI 低于拐点时，计算单元在等数据，是带宽受限；高于拐点时，数据供得上但算不过来，是算力受限。以 A100-80GB 为例，FP16 Tensor Core 峰值 312 TFLOPS，HBM 带宽约 2.0 TB/s，拐点约为 156 FLOP/Byte。

放到线性层 $$Y=XW^\top$$ 上，$$X\in\mathbb{R}^{b\times K}$$，b 是这次前向处理的 token 数：

$$
\text{FLOPs} = 2bKN,\qquad \text{Bytes}\approx \underbrace{2KN}_{\text{FP16 权重}} + \underbrace{2bK + 2bN}_{\text{激活，b 小时可忽略}}\quad\Rightarrow\quad \text{AI}\approx b
$$

线性层的算术强度大约就等于 token 数。对照上面的三类负载：

- **decode**：b 就是 batch 数，通常远小于 156。batch=1 时 AI ≈ 1，只能用到峰值算力的 1% 左右，时间几乎都花在从显存读权重上；
- **prefill 和扩散模型**：b 是几千甚至几万，远超拐点，属于算力受限，读权重的时间可以被计算掩盖。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-01-roofline.svg" alt="A100 的 Roofline：decode 的点落在带宽受限的斜线上，prefill 落在算力受限的水平段" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. A100 的 Roofline。decode 的点在斜线上，权重从 FP16 换成 INT4，读同样的权重只要 1/4 的字节，相当于 AI 变成 4 倍，点沿斜线右移；prefill 在水平段，读得再少也没用，只有换成更低比特的 Tensor Core 把"屋顶"抬高才能继续加速。
</div>

batch=1 的 decode，每生成一个 token 都要把全部权重从 HBM 读一遍，所以速度上限大约是带宽 ÷ 权重字节数：

| 配置                  | 权重字节 | A100（2.0 TB/s）     | RTX 4090（1.0 TB/s） |
| :-------------------- | :------- | :------------------- | :------------------- |
| Llama-2-7B FP16       | 13.5 GB  | 6.7 ms → ≤ 148 tok/s | 13.4 ms → ≤ 75 tok/s |
| Llama-2-7B W4A16 g128 | ≈ 3.9 GB | 1.9 ms → ≤ 513 tok/s | 3.9 ms → ≤ 258 tok/s |

实际速度会低于这个上限（kernel 效率、读 KV、调度开销），但量级关系是成立的。W4A16 的 3.9 GB 里包含了仍然是 FP16 的 embedding 和 lm_head。

### 不同方案省的是什么

把显存和 Roofline 合起来，就能看清每种量化方案省的是哪种资源、在哪个阶段生效：

| 方案                        | 省什么                         | 在哪个阶段加速                                       | 代表方法                             | 硬件要点                                                              |
| :-------------------------- | :----------------------------- | :--------------------------------------------------- | :----------------------------------- | :-------------------------------------------------------------------- |
| W4A16 / W8A16（只量化权重） | 权重显存与带宽                 | decode（小 batch）明显加速；prefill 基本不变甚至略慢 | GPTQ、AWQ、NF4、GGUF                 | 任意 GPU；kernel 内部把 INT4 反量化成 FP16 再算                       |
| W8A8（INT8 / FP8）          | 权重减半 + 2× Tensor Core 算力 | prefill、大 batch decode、扩散模型                   | SmoothQuant、FP8 per-token           | INT8：Turing 起；FP8：Ada / Hopper 起                                 |
| W4A4（INT4 / FP4）          | 权重 1/4 + 4× 算力             | 算力受限的场景（prefill、扩散模型）                  | QuaRot、SpinQuant、SVDQuant、ConvRot | INT4 MMA：Ampere / Ada 有，Hopper 没有；FP4（NVFP4/MXFP4）：Blackwell |
| KV Cache 量化               | 长上下文的显存与带宽           | 长序列 decode、高并发                                | FP8 KV、KIVI、QuaRot 的 KV4          | 需要 attention kernel 支持                                            |

这里有一个很容易想当然的地方："4 bit 一定比 8 bit 快"。其实不一定。W4A16 的计算仍然是 FP16 GEMM，只是读权重时少读了 3/4。大 batch 的 prefill 本来就受算力限制，这时 W4A16 反而可能比 FP16 慢（多了反量化指令），而 W8A8 用上了 INT8 Tensor Core 会更快。扩散模型一次前向要处理几千个图像 token，更像 prefill，所以 **AIGC 场景更需要 W8A8 / W4A4，而不是只量化权重**。这一点后面会反复用到。

## 是什么：把实数装进低比特

量化的终点是某种低比特格式，也就是"装数的容器"。先看有哪些容器，再看怎么把一组实数装进去。

### 整数：等间隔的码点

b 位整数有 $$2^b$$ 个码字，相邻码字的间隔都是 1：

| 格式         | 码字数 | 有符号（补码）范围 | 无符号范围 |
| :----------- | :----- | :----------------- | :--------- |
| INT8 / UINT8 | 256    | −128 … 127         | 0 … 255    |
| INT4 / UINT4 | 16     | −8 … 7             | 0 … 15     |
| INT2 / UINT2 | 4      | −2 … 1             | 0 … 3      |

有符号整数用补码表示：INT4 里 3 是 `0011`，−3 是 `1101`，−8 是 `1000`；同样的 `1101` 按无符号读就是 13。整数本身只能表示整数，要表示 0.37、−2.8 这样的实数，还得配一个缩放因子 scale（实数 ≈ scale × 整数），这是后面量化公式要做的事。整数格式硬件简单、乘加快，缺点是码点等间隔，表示跨越好几个数量级的数据时很吃力。

### 浮点：指数管范围，尾数管精度

浮点数由符号位 S、指数位 E、尾数位 M 组成：

$$
\begin{aligned}
x &= (-1)^S\times 2^{E-\text{bias}}\times\Big(1+\frac{M}{2^{m}}\Big) && (\text{规格化})\\
x &= (-1)^S\times 2^{1-\text{bias}}\times\frac{M}{2^{m}} && (E=0,\ \text{次正规})
\end{aligned}
$$

指数位决定动态范围，尾数位决定相对精度：每个 $$[2^e, 2^{e+1})$$ 区间里有 $$2^m$$ 个等间隔的码点，越往外间隔越大，所以浮点的码点是"近密远疏"的。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-02-bit-layout.svg" alt="FP32、FP16、BF16、FP8 E4M3、FP8 E5M2、FP4 E2M1、INT8、INT4 的比特分配" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. 各种格式的比特分配。橙色是符号位，蓝色是指数位，绿色是尾数位，灰色是整数的补码位。BF16 和 FP32 的指数位一样（范围一样），只是尾数短；FP16 是用更少的指数位换了更多的尾数位。
</div>

| 格式     | S/E/M  | bias | 最大有限值 | 最小正规数   | 1 附近的间隔 | 典型用途             |
| :------- | :----- | :--- | :--------- | :----------- | :----------- | :------------------- |
| FP32     | 1/8/23 | 127  | 3.4×10³⁸   | 1.2×10⁻³⁸    | 1.2×10⁻⁷     | 主权重、累加器       |
| FP16     | 1/5/10 | 15   | 65504      | 6.1×10⁻⁵     | 9.8×10⁻⁴     | 推理默认精度         |
| BF16     | 1/8/7  | 127  | 3.4×10³⁸   | 1.2×10⁻³⁸    | 7.8×10⁻³     | 训练、推理           |
| FP8 E4M3 | 1/4/3  | 7    | 448        | 2⁻⁶ = 0.0156 | 0.125        | 前向的权重与激活     |
| FP8 E5M2 | 1/5/2  | 15   | 57344      | 6.1×10⁻⁵     | 0.25         | 梯度                 |
| FP4 E2M1 | 1/2/1  | 1    | 6          | 1.0          | 0.5          | NVFP4 / MXFP4 的元素 |

E4M3 是 "FN" 变体，没有 ±∞，只有 `S.1111.111` 表示 NaN，所以最大值能扩展到 1.75×2⁸ = 448。FP4 E2M1 只能表示 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}。可以看到，比特越少，浮点的范围和精度都急剧缩小。

几个编码的例子能更直观地感受这些差别：

- **1/3 → FP16 与 BF16**：1/3 = 1.333…×2⁻²。FP16 尾数 10 位，0.333…×1024 = 341.33 → 341，得到 0.333252；BF16 尾数 7 位，0.333…×128 = 42.67 → 43，得到 0.333984，相对误差大约是 FP16 的 8 倍。
- **70000 → FP16 与 BF16**：FP16 最大只有 65504，直接溢出成 inf；BF16 的指数范围和 FP32 一样，得到 70144。这就是训练偏爱 BF16 的原因：宁可精度粗一点，也不能溢出。
- **0.3 → FP8 E4M3**：0.3 = 1.2×2⁻²，指数字段 −2+7 = 5，尾数 0.2×8 = 1.6 → 2，编码为 `0 0101 010`，表示 1.25×2⁻² = 0.3125，相对误差 4.2%。500 超过了 448，按饱和转换得到 448。

其他的数可以直接在下面试，输入任意实数，看它在各种格式里的编码和误差：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>数值格式转换器</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="fmt-in">输入一个实数</label>
      <input type="number" id="fmt-in" value="0.3" step="any">
      <button class="btn" type="button" data-fmt="3.14159">π</button>
      <button class="btn" type="button" data-fmt="70000">70000</button>
      <button class="btn" type="button" data-fmt="0.001">0.001</button>
      <button class="btn" type="button" data-fmt="-447.5">−447.5</button>
    </div>
    <div class="tw"><table id="fmt-out"></table></div>
    <p class="small mute">舍入方式为“就近舍入、平局取偶”（round-to-nearest-even）。超出范围时显示饱和后的值：FP16/BF16/E5M2 在 IEEE 语义下会变成 inf，E4M3 与 E2M1 没有 inf，转换时通常饱和到最大值。</p>
  </div>
</div>

### 码点放在哪里：均匀与非均匀

整数等间隔，浮点近密远疏。8 bit 有 256 个码字，怎么放差别不大；到了 4 bit 只剩 16 个码字，放在哪里就直接决定了误差。

神经网络的权重大多是钟形分布，大部分值在 0 附近，直觉上应该在数据密的地方多放码点。QLoRA 提出的 **NF4**（NormalFloat4）就是这么做的：把 16 个点放在标准正态分布的分位点上，让每个区间覆盖的概率质量大致相等。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-03-codebooks.svg" alt="INT4、FP4 E2M1、NF4 三种 4 bit 码本在 -1 到 1 上的码点分布，叠加正态分布密度" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. 三种 4 bit 码本（都归一化到 [−1, 1]），灰色曲线是正态密度。INT4 完全均匀，FP4 呈对数式分布，NF4 在密度高的地方更密。
</div>

拿 8 个权重 `[0.12, −0.85, 2.40, 0.33, −1.10, 0.05, −0.47, 1.75]` 试一下（真实的 block 是 64 个）：除以 absmax 2.4 之后找最近的 NF4 码点，反量化得到 `[0.191, −0.948, 2.400, 0.386, −0.948, 0.000, −0.444, 1.735]`，MSE = 0.00554；同一组数据用对称 INT4（s = 2.4/7）的 MSE 是 0.00832。QLoRA 还把每个 block 的 FP32 absmax 再量化成 8 bit（Double Quantization），scale 的开销从 0.5 bit/参数降到了约 0.127 bit/参数。

代价是 NF4 这种非均匀码本没有对应的硬件乘法器，计算前必须查表还原成 FP16/BF16，所以只用来压缩存储权重（QLoRA 微调、bitsandbytes 加载）。整数和浮点格式则可以直接送进 Tensor Core，后面的算法也都是基于均匀的整数或浮点格式。

### 块缩放格式：每一小块共享一个 scale

前面留了一个问题：低比特浮点的动态范围太小。FP4 只能表示 0.5～6，FP8 E4M3 也只有约 2⁻⁹～448，而真实张量里的值可能是 10⁻³ 量级，也可能上百。解决办法是**让每一小块元素共享一个缩放因子**：元素用低比特存"相对大小"，共享的 scale 负责把整块缩放到真实的范围，$$x_i\approx \text{scale}_{\text{block}}\times \text{elem}_i$$。这其实就是后面要讲的量化 scale，只不过块大小和 scale 的格式被写进了硬件标准，Tensor Core 可以直接消费，不需要单独的反量化步骤。

| 格式   | 元素                                    | 块大小 | 共享 scale                    | 等效位宽 | 说明                                            |
| :----- | :-------------------------------------- | :----- | :---------------------------- | :------- | :---------------------------------------------- |
| MXFP8  | FP8 E4M3 或 E5M2                        | 32     | E8M0（2 的幂）                | 8.25 bit | 8 bit 训练与推理；Blackwell 原生支持            |
| MXINT8 | 8 bit 定点（补码整数 × 2⁻⁶，范围约 ±2） | 32     | E8M0                          | 8.25 bit | 整数元素的块缩放版本                            |
| MXFP6  | E2M3 或 E3M2                            | 32     | E8M0                          | 6.25 bit | 介于 8 bit 和 4 bit 之间                        |
| MXFP4  | FP4 E2M1                                | 32     | E8M0                          | 4.25 bit | gpt-oss 的 MoE 权重用的就是这个格式             |
| NVFP4  | FP4 E2M1                                | 16     | E4M3 + 张量级 FP32 二级 scale | 4.5 bit  | 块更小、scale 不限于 2 的幂，精度通常优于 MXFP4 |

MX 格式的共享 scale 是 E8M0，只能取 2 的幂：$$\text{scale}=2^{\lfloor\log_2 \text{absmax}\rfloor-e_{\max}}$$，其中 $$e_{\max}$$ 是元素格式最大值的指数（E4M3 为 8，E2M1 为 2，MXINT8 为 0）。

8 bit 上的块缩放也不是 MX 才有的想法。DeepSeek-V3 的 FP8 训练所有张量都用 E4M3，激活每 1×128 个元素（每个 token 的每 128 个通道）一个 scale，权重每 128×128 块一个 scale。Hopper 的 FP8 Tensor Core 只支持张量级（或按行/列）的 scale，这种分块缩放需要软件在累加时补乘；Blackwell 的 Tensor Core 则直接支持 MX 的块 scale。

用一个 16 元素的 block 看一下 NVFP4 是怎么量化的：

```text
block = [0.12, -0.85, 2.40, 0.33, -1.10, 0.05, -0.47, 1.75,
         0.90, -0.20, 0.61, -2.05, 0.28, 1.30, -0.66, 0.02]
```

1. block scale = absmax / 6 = 2.4 / 6 = 0.4，存成 E4M3：0.4 = 1.6×2⁻²，尾数 0.6×8 = 4.8 → 5，实际的 scale 是 1.625×0.25 = 0.40625（这里省略了 FP32 二级 scale，相当于它等于 1）；
2. 每个元素除以 0.40625，再找最近的 E2M1 值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}，得到 `[0.5, −2, 6, 1, −3, 0, −1, 4, 2, −0.5, 1.5, −6, 0.5, 3, −1.5, 0]`；
3. 反量化之后 MSE = 0.0141。误差最大的是 −2.05：除以 scale 是 −5.05，FP4 在 4 和 6 之间没有码点，只能落到 −6，还原成 −2.44。

同一个 block 换成 MXFP4，scale = 2^(⌊log₂2.4⌋−2) = 0.5，最大值 2.4/0.5 = 4.8 只能落到 4，还原成 2.0，误差 0.4，2 的幂 scale 让最大值附近的损失更大。有意思的是，这组数据接近均匀分布，用对称 INT4（s = 2.4/7）MSE 反而只有 0.0075。8 bit 上也有类似的现象：前面那 8 个数用 MXINT8 的 MSE 是 6.1×10⁻⁵，比 MXFP8 的 1.43×10⁻³ 小一个数量级，块内数值量级接近时等间隔的整数码点反而更精细。FP4 / FP8 的优势在于重尾、钟形的分布，以及硬件原生的 block scale，**格式之间没有绝对的优劣，要看数据分布**。

### 线性量化：一把刻度尺

有了容器，接下来是怎么把一组实数装进去。假设一组数都落在 [−1.5, 3.0] 之间，要用 4 bit（0～15）来存。最自然的做法是做一把刻度尺：把 [−1.5, 3.0] 等分成 15 段，每段宽 4.5/15 = **0.3**，16 条刻度线编号 0～15，编号 0 对应 −1.5，编号 5 对应 0，编号 15 对应 3.0。存的时候找离它最近的刻度线，只记编号；取的时候用编号换算回实数：

| 原始值 x | x / 0.3 | 取整 | 刻度编号（+5） | 取回的值 | 说明                               |
| :------- | :------ | :--- | :------------- | :------- | :--------------------------------- |
| −1.5     | −5.00   | −5   | 0              | −1.5     | 正好在刻度上，没有误差             |
| −0.7     | −2.33   | −2   | 3              | −0.6     | 误差 0.1                           |
| 0        | 0       | 0    | 5              | 0        | 0 能被精确表示                     |
| 1.1      | 3.67    | 4    | 9              | 1.2      | 误差 0.1                           |
| 3.4      | 11.33   | 11   | 16 → 15        | 3.0      | 超出尺子，只能记最后一格，误差 0.4 |

这把尺子只由两个数决定：刻度间隔 0.3，和"0 落在第几格"的 5。前者叫 **scale**，后者叫 **zero-point**。表里也已经出现了量化误差的两种来源：落在两格之间要舍入，超出尺子要截断。

写成一般的公式，把实数区间 $$[x_{\min}, x_{\max}]$$ 映射到整数区间 $$[q_{\min}, q_{\max}]$$：

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\qquad z=\operatorname{round}\Big(q_{\min}-\frac{x_{\min}}{s}\Big)
$$

$$
q=\operatorname{clamp}\Big(\operatorname{round}\big(\tfrac{x}{s}\big)+z,\;q_{\min},\;q_{\max}\Big),\qquad \hat{x}=s\,(q-z)
$$

第一行由区间算出 scale 和 zero-point；q 那一步叫量化（quantize），$$\hat x$$ 那一步叫反量化（dequantize），$$x$$ 和 $$\hat x$$ 的差就是量化误差。z 取整数是为了让实数 0 一定落在某条刻度上，能被精确表示（padding、ReLU 之后的 0 很常见）。因为刻度是等间隔的，这种量化叫均匀量化或者线性量化，是 LLM 量化的主流。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-04-linear-quant.svg" alt="实数轴到 4 比特整数轴的仿射映射，s=0.3，z=5" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. 线性量化的几何含义，就是上面那把刻度尺。实数轴上每隔 s 有一个可表示点，每个点对应一个整数，实数被"吸附"到最近的点上：区间内的误差是舍入误差，区间外的是截断误差。
</div>

### 对称与非对称

实践中这套公式有两种常用的写法，区别在于要不要 zero-point：

- **对称量化**：令 $$z=0$$，刻度尺以 0 为中心，$$s=\max\lvert x\rvert/q_{\max}$$，$$q_{\max}=2^{b-1}-1$$，码字是有符号整数（比如 −7…7）。反量化只要一次乘法，矩阵乘里也没有零点交叉项，**激活和权重都要量化的 W8A8 / W4A4 几乎都用对称量化**。代价是数据偏向一边的时候会浪费一半的码字。
- **非对称量化**：保留 z，刻度尺贴合 $$[x_{\min},x_{\max}]$$，码字通常用无符号整数（0…15）。GPTQ / AWQ 的 W4A16 一般用它，因为 4 bit 下每个码字都很宝贵，而只量化权重时反量化的开销可以忽略。

同一个向量 $$x$$ = `[0.52, −1.37, 2.10, −0.08, 0.95, −2.80, 1.63, 0.31]` 用几种方式量化：

| 方式                | scale / zero                 | 码字 q                              | 反量化                                                     | MSE       |
| :------------------ | :--------------------------- | :---------------------------------- | :--------------------------------------------------------- | :-------- |
| 对称 INT4           | s = 2.80/7 = 0.4             | [1, −3, 5, 0, 2, −7, 4, 1]          | [0.40, −1.20, 2.00, 0.00, 0.80, −2.80, 1.60, 0.40]         | 0.0114    |
| 对称 INT8           | s = 2.80/127 = 0.022047      | [24, −62, 95, −4, 43, −127, 74, 14] | 和 x 相差都不到 0.01                                       | 2.48×10⁻⁵ |
| 非对称 INT4（0…15） | s = 4.90/15 = 0.32667，z = 9 | [11, 5, 15, 9, 12, 0, 14, 10]       | [0.653, −1.307, 1.960, 0.000, 0.980, −2.940, 1.633, 0.327] | 0.00857   |

有几个值得注意的地方：

- 对称 INT4 的码字 −8 没用上，对称量化通常只用 −7…7，保证正负对称；
- 从 INT4 到 INT8，s 缩小到大约 1/18，MSE 缩小了约 460 倍；
- 非对称比对称 INT4 好 25%，但 −2.80 被还原成了 −2.94：z 取整之后整个网格平移了 0.43s，端点不再能精确表示，而 0 可以；
- 非对称这里用的是无符号 4 bit，这是 GPTQ / AWQ 存非对称权重的惯例。换成有符号的 −8…7 也完全可以，只是整体平移 8（z 变成 1），反量化的结果一模一样。

数据偏向一侧的时候差别更明显。GELU 之后的激活 `[0, 0.3, 1.2, 2.5, 0.8, 3.1, 0.05, 1.7]` 全是非负数，对称 INT4 的 s = 3.1/7，负半轴的码字全部浪费，MSE = 0.00957；非对称 INT4 的 s = 3.1/15、z = 0，MSE = 0.00201，好了 4.8 倍。

下面的计算器把上面公式的每一步都列了出来。可以换成自己的数据，切换位宽和对称 / 非对称，也可以先拖一下截断比例，感受一下下一节要讲的截断：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>量化计算器：每一步都算给你看</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="qc-vals">数据</label>
      <input type="text" id="qc-vals" value="0.52, -1.37, 2.10, -0.08, 0.95, -2.80, 1.63, 0.31">
    </div>
    <div class="ctrl">
      <label for="qc-bits">位宽
        <select id="qc-bits"><option>2</option><option>3</option><option selected>4</option><option>6</option><option>8</option></select>
      </label>
      <label for="qc-mode">方式
        <select id="qc-mode"><option value="sym">对称</option><option value="asym">非对称</option></select>
      </label>
      <label for="qc-clip">截断比例 <input type="range" id="qc-clip" min="0.3" max="1" step="0.01" value="1"><output id="qc-clip-v">1.00</output></label>
    </div>
    <div class="stats" id="qc-stats"></div>
    <div class="fig" style="padding:8px"><svg id="qc-svg" viewBox="0 0 680 120" class="wide" role="img" aria-label="量化网格与数据点"></svg></div>
    <div class="tw"><table id="qc-table"></table></div>
    <p class="small mute">截断比例 &lt; 1 时，量化区间缩小为原来的该比例（对称：±c·max|x|；非对称：两端同比收缩）。可以观察到 s 变小、舍入误差变小，但超出区间的值误差突然变大。</p>
  </div>
</div>

### 误差从哪来：舍入与截断

先看区间内的舍入误差。如果数据相对刻度是"随机"分布的，舍入误差近似服从 $$[-s/2, s/2]$$ 上的均匀分布：

$$
\mathbb{E}[e^2]=\frac{1}{s}\int_{-s/2}^{s/2}e^2\,de=\frac{s^2}{12}
$$

位宽每多 1 bit，刻度数翻倍、s 减半，MSE 降为 1/4，信号量化噪声比（SQNR）提高约 6.02 dB。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-05-rounding-error.svg" alt="3 比特量化的阶梯函数，以及锯齿状的舍入误差" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. 3 bit 量化的阶梯函数和锯齿状的舍入误差，误差始终落在 [−s/2, s/2] 之内。
</div>

这个式子还说明，**位宽固定时误差只取决于 s**，而 s 由量化区间的宽度决定。所以控制误差的核心问题就是：区间取多宽？

直接取 $$\max\lvert x\rvert$$ 当端点可以保证没有截断误差，但只要数据里有个别特别大的值，s 就会被拉得很大。另一种选择是主动截断：取一个小于 $$\max\lvert x\rvert$$ 的阈值 $$c$$，令 $$s=c/q_{\max}$$，超出 $$[-c, c]$$ 的值 clamp 到边界。总误差变成两部分：

$$
\text{MSE}(c)\approx\underbrace{\frac{(c/q_{\max})^2}{12}\cdot P(\lvert x\rvert\le c)}_{\text{舍入误差，随 } c \text{ 增大}}+\underbrace{\mathbb{E}\big[(\lvert x\rvert-c)^2\cdot\mathbb{1}_{\lvert x\rvert>c}\big]}_{\text{截断误差，随 } c \text{ 减小}}
$$

下面的实验用 2000 个 N(0,1) 样本加上 0.5% 的离群值，可以拖动截断比例，看两部分误差怎么此消彼长。默认 INT4、离群值 8σ 时，不截断的 MSE 是 0.158；点"跳到最优"会看到截断比例取 0.69 时 MSE 降到 0.0997：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>截断阈值的权衡</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="cl-bits">位宽 <select id="cl-bits"><option value="3">INT3</option><option value="4" selected>INT4</option><option value="8">INT8</option></select></label>
      <label for="cl-out">离群值幅度 <input type="range" id="cl-out" min="0" max="20" step="0.5" value="8"><output id="cl-out-v">8.0σ</output></label>
      <label for="cl-c">截断比例 c / max|x| <input type="range" id="cl-c" min="0.05" max="1" step="0.01" value="1"><output id="cl-c-v">1.00</output></label>
      <button class="btn" type="button" id="cl-best">跳到最优</button>
    </div>
    <div class="stats" id="cl-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>数据直方图（2000 个 N(0,1) 样本 + 0.5% 离群值）与截断线</h5><svg id="cl-hist" viewBox="0 0 320 180"></svg></div>
      <div class="panel"><h5>MSE 随截断比例变化（对数纵轴）</h5><svg id="cl-curve" viewBox="0 0 320 180"></svg></div>
    </div>
  </div>
</div>

最优的 c 在两者的平衡点上。数据越重尾、离群值越多，这个平衡就越难找：保留离群值，s 很大，绝大多数正常值被粗暴地舍入；截掉离群值，少数关键值的误差又会很大。**后面 LLM 量化的各种算法，很大程度上都在处理这个矛盾。**

### 校准：静态与动态

权重在推理前就已知，可以直接从权重本身确定区间。激活随输入变化，推理前不知道它的范围，有两种办法：

- **静态量化**：先用一小批校准数据（比如 128 条 2048 token 的文本）跑一遍前向，统计激活分布确定 scale，推理时固定不变。没有额外开销，但遇到分布外的输入可能溢出。
- **动态量化**：推理时对每个 token（或每个张量）实时算 absmax 得到 scale。要多一次归约，但精度高得多。LLM 的 W8A8 / W4A4 绝大多数用 **per-token 动态量化**，这个归约通常融合到前一个算子（比如 RMSNorm）的 kernel 里。

静态量化需要从校准统计里选出区间的端点，常见的策略有这几种：

| 方法                        | 做法                                            | 特点                                             |
| :-------------------------- | :---------------------------------------------- | :----------------------------------------------- |
| Min-Max                     | 取观察到的最大最小值                            | 不截断，对离群值最敏感                           |
| Percentile                  | 取 99.9% / 99.99% 分位数                        | 简单地截掉极端值                                 |
| MSE 搜索                    | 枚举 c，选使 $$\lVert x-Q_c(x)\rVert^2$$ 最小的 | 直接优化重建误差，上面实验里的"最优"就是这么找的 |
| KL 散度（TensorRT entropy） | 2048 bin 的直方图，选使 KL(P‖Q) 最小的阈值      | 保持分布形状，CNN 时代的标准做法                 |

### 粒度：多少个元素共享一个 scale

到目前为止都默认整组数据共用一把刻度尺。一个张量可以只用一把（per-tensor），也可以按行、按列或者按小组各用一把，"多少个元素共享一个 scale"叫做量化的**粒度**。粒度越细，每组的数据范围越小、误差越小，但 scale 的存储和计算开销也越大。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-06-granularity.svg" alt="per-tensor、per-channel、per-input-channel、per-group 四种量化粒度示意" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6. 四种粒度，同一颜色的格子共享一个 scale。以 N×K 的权重为例：per-tensor 只有 1 个 scale；per-channel 每个输出通道（行）一个，共 N 个；per-group 沿输入维 K 每 g 个元素一个，共 N·K/g 个。对激活来说，"每行一个"叫 per-token。
</div>

粒度的影响可以非常大。权重 $$W=\begin{bmatrix}0.10 & -0.20 & 0.15 & 0.05\\ 3.00 & -2.40 & 1.80 & 2.70\end{bmatrix}$$，第 2 个输出通道的幅值是第 1 个的 15 倍，用对称 INT4：

- **per-tensor**：s = 3.00/7 = 0.4286，第 1 行除以 s 之后都在 ±0.47 以内，全部舍入成 0，第 1 行的信息全丢了，MSE = 0.0188；
- **per-channel**：第 1 行单独用 s₁ = 0.20/7，q = [4, −7, 5, 2]，还原成 [0.114, −0.200, 0.143, 0.057]，MSE = 0.0001。

只多存了 1 个 scale，第 1 行的误差就降低了两个数量级。

下面的热力图可以切换几种粒度对比一下，激活里第 3、11 列是离群通道：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>粒度对比热力图</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="gr-mode">粒度
        <select id="gr-mode">
          <option value="tensor">per-tensor</option>
          <option value="row" selected>per-row（per-token / per-out-channel）</option>
          <option value="col">per-column（per-input-channel）</option>
          <option value="group">per-group（行内每 4 个）</option>
        </select>
      </label>
      <label for="gr-bits">位宽 <select id="gr-bits"><option value="3">INT3</option><option value="4" selected>INT4</option><option value="8">INT8</option></select></label>
      <label for="gr-co">离群通道倍数 <input type="range" id="gr-co" min="1" max="30" step="1" value="12"><output id="gr-co-v">12×</output></label>
    </div>
    <div class="stats" id="gr-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>|X|：8 个 token × 16 个通道（第 3、11 列为离群通道）</h5><svg id="gr-a" viewBox="0 0 330 170"></svg></div>
      <div class="panel"><h5>|量化误差|（颜色越深误差越大，统一色标）</h5><svg id="gr-b" viewBox="0 0 330 170"></svg></div>
    </div>
    <p class="small mute">切换到 per-column 时误差最小，因为离群值正好按列分布；但统计栏提示这种粒度“不能用于激活的整数矩阵乘”，原因在后面"哪些 scale 能提到求和号外面"一节。per-row 无法隔离离群列：每一行都含有离群值，所以每行的 scale 都被拉大。</p>
  </div>
</div>

per-group 把每一行沿 K 维切成长度为 g 的小组，每组有自己的 scale（和 zero），等效的比特数是：

$$
b_{\text{eff}}=b+\frac{b_{\text{scale}}+b_{\text{zero}}}{g}
$$

INT4、g=128、FP16 scale + FP16 zero 是 $$4+32/128=4.25$$ bit，zero 用 4 bit 打包存储时约 4.16 bit。g 越小越准，但 g=32 时开销已经到 5 bit 了。g=128 是 LLM 的事实标准，扩散模型常用 g=64。前面讲的块缩放格式，本质上就是硬件规定了 g（16 或 32）和 scale 格式的分组量化。

### PTQ 与 QAT

上面这些步骤可以在两种流程里完成：

- **训练后量化（PTQ）**：模型训练好之后，只用少量校准数据确定量化参数（有时也会调整权重，比如 GPTQ），不做完整的反向传播。后面要讲的 GPTQ、AWQ、SmoothQuant、QuaRot、ConvRot、SVDQuant 都是 PTQ。
- **量化感知训练（QAT）**：在训练图里插入伪量化节点 $$\hat x = s\cdot\operatorname{clamp}(\operatorname{round}(x/s))$$，让模型在训练中适应量化噪声。round 的导数几乎处处为 0，所以用直通估计器（STE）来近似：

$$
\frac{\partial \hat x}{\partial x}\approx\begin{cases}1, & -c \le x \le c\\ 0, & \text{otherwise}\end{cases}
$$

LSQ 这类方法还把 s 也当成可学习的参数。QAT 精度最好但成本高，LLM 上多用于 2～4 bit 的极低比特，或者厂商发布的官方量化版本。后面 SpinQuant 学习旋转矩阵，也借用了 QAT 的思路。

## 量化之后的矩阵乘在 GPU 上怎么跑

前面量化的都是"一组数"，模型里真正耗时的是线性层的矩阵乘。把量化放进矩阵乘之后会发现，它对粒度有硬性的限制。

### 哪些 scale 能提到求和号外面

线性层的一个输出元素是 $$Y_{ij}=\sum_{k=1}^{K}X_{ik}W_{jk}$$。如果激活**按行（per-token）**量化 $$X_{ik}\approx s^x_i\,q^x_{ik}$$，权重**按输出通道**量化 $$W_{jk}\approx s^w_j\,q^w_{jk}$$，两个 scale 都和求和下标 k 无关，可以提出来：

$$
Y_{ij}\approx s^x_i\,s^w_j\underbrace{\sum_{k}q^x_{ik}\,q^w_{jk}}_{\text{纯整数点积，INT32 累加}}
$$

求和全在整数域里完成，最后乘一次 $$s^x_i s^w_j$$ 就行。但如果激活**按输入通道**量化 $$X_{ik}\approx s_k\,q^x_{ik}$$：

$$
Y_{ij}\approx s^w_j\sum_{k}\underbrace{s_k}_{\text{随 }k\text{ 变化}}\,q^x_{ik}\,q^w_{jk}
$$

$$s_k$$ 随求和下标变化，只能留在求和号里面，每一项都要乘一个不同的浮点数，整数 Tensor Core 就用不上了。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-07-scale-factoring.svg" alt="per-token 与 per-output-channel 的 scale 可以提到求和号外，per-input-channel 的 scale 不行" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 7. scale 的"可提取性"。上：沿非归约维度（T 和 N）的 scale 都能提到外面，这就是 W8A8 的标准配置"激活 per-token + 权重 per-channel"；下：沿归约维度 K 的 scale 不行。
</div>

由此可以得到粒度选择的几条规律：

- **W8A8 / W4A4（整数或 FP8 矩阵乘）**：激活 per-token（动态）或 per-tensor，权重 per-channel，scale 必须沿 T 或 N 维；
- **W4A16（只量化权重）**：权重在 kernel 里先反量化成 FP16 再乘，所以沿 K 维的 per-group（g=128）也可以，这是 GPTQ / AWQ 的标准配置；
- **例外**：沿 K 分块的 scale 也能用，代价是分段累加、每段乘各自的 scale。W4A8 kernel（比如 QServe）用软件这么做，Blackwell 的 MX / NVFP4 则由 Tensor Core 在硬件里完成每块的缩放。

这里先埋个伏笔：后面会看到，LLM 激活的离群值恰好集中在少数**输入通道**上。最需要 per-input-channel scale 的维度，正好是这里不能用的维度。

### W8A8：整数矩阵乘 + 反量化 epilogue

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-08-w8a8-dataflow.svg" alt="W8A8 推理数据流：RMSNorm 融合激活量化，INT8 Tensor Core 计算，epilogue 中反量化" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 8. W8A8 的数据流。激活量化融合在前一个算子里，省掉一次显存往返；反量化融合在 GEMM 的 epilogue 里，INT32 的结果不落显存。整条路径上只有 X_q 和 Y 进出显存。
</div>

用一个很小的例子走一遍：

$$
X=\begin{bmatrix}0.8 & -1.2 & 0.5\\ 2.0 & 0.4 & -0.6\end{bmatrix},\quad W=\begin{bmatrix}0.3 & -0.7 & 1.1\\ -0.5 & 0.9 & 0.2\end{bmatrix},\quad Y_{\text{FP}}=XW^\top=\begin{bmatrix}1.63 & -1.38\\ -0.34 & -0.76\end{bmatrix}
$$

**离线，权重 per-channel 对称 INT8**。第 1 行 absmax 是 1.1，$$s^w_1=1.1/127=0.008661$$；第 2 行 $$s^w_2=0.9/127=0.007087$$，得到 $$W_q=\begin{bmatrix}35 & -81 & 127\\ -71 & 127 & 28\end{bmatrix}$$。

**在线，激活 per-token 动态量化**。$$s^x_1=1.2/127=0.009449$$，$$s^x_2=2.0/127=0.015748$$，得到 $$X_q=\begin{bmatrix}85 & -127 & 53\\ 127 & 25 & -38\end{bmatrix}$$。

**Tensor Core，INT8 乘、INT32 累加**：

| 输出  | 逐项乘积                                             | INT32 累加 |
| :---- | :--------------------------------------------------- | :--------- |
| acc₁₁ | 85·35 + (−127)·(−81) + 53·127 = 2975 + 10287 + 6731  | 19993      |
| acc₁₂ | 85·(−71) + (−127)·127 + 53·28 = −6035 − 16129 + 1484 | −20680     |
| acc₂₁ | 127·35 + 25·(−81) + (−38)·127 = 4445 − 2025 − 4826   | −2406      |
| acc₂₂ | 127·(−71) + 25·127 + (−38)·28 = −9017 + 3175 − 1064  | −6906      |

单个乘积最大是 127² = 16129，K=4096 时累加和能到 6.6×10⁷，所以累加器必须是 INT32。

**Epilogue，乘两个 scale 的外积**。$$Y_{11}=19993\times0.009449\times0.008661=1.6362$$，同理 $$Y_{12}=-1.3847$$、$$Y_{21}=-0.3282$$、$$Y_{22}=-0.7707$$，和 FP 的结果相差都在 0.012 以内。

如果激活用非对称量化 $$X_{ik}\approx s^x_i(q^x_{ik}-z_i)$$，展开之后：

$$
\sum_k (q^x_{ik}-z_i)\,q^w_{jk}=\underbrace{\sum_k q^x_{ik}q^w_{jk}}_{\text{整数 GEMM}}-z_i\underbrace{\sum_k q^w_{jk}}_{\text{离线预计算的行和}}
$$

修正项只要 $$O(T+N)$$ 的额外计算。如果权重也有零点，还会多出 $$-z^w_j\sum_k q^x_{ik}$$ 和 $$+K z_i z^w_j$$ 两项。没有交叉项、kernel 最简单，这就是 W8A8 里普遍用对称量化的原因。

### W4A16：权重在寄存器里"现场解压"

W4A16 的计算单元还是 FP16 Tensor Core，INT4 只是一种压缩存储格式，在 kernel 里被解包、反量化成 FP16，再参与矩阵乘，速度收益全部来自"少读 3/4 的字节"。

**打包**。GPU 没有 4 bit 的数据类型，量化后的权重用 `int32` 存，每个 int32 装 8 个 4 bit 值。GPTQ / AWQ 一般存 0…15 的无符号码字加一个零点（对称时零点为 8）。比如码字 `[11, 6, 15, 0, 3, 9, 12, 7]` 依次放到第 0、4、8…28 位，packed = Σ qᵢ ≪ 4i，从高位到低位读是 `0x7C930F6B`，解包就是右移 4i 位再和 `0xF` 做按位与。实际的 kernel 往往用交错的顺序（AutoAWQ 的 GEMM 格式用 [0, 2, 4, 6, 1, 3, 5, 7]，Marlin 有自己的 tile 重排），让一条指令解出的两个值正好落在 Tensor Core 需要的寄存器位置上，所以不同格式的 checkpoint 不能直接混用。

**INT4 → FP16 的魔数**。逐个做 int → float 的转换太慢了。FasterTransformer / Marlin 用了一个位运算的技巧：FP16 的 `0x6400` 表示 1024.0，这时尾数最低位的权重正好是 1，把 4 bit 码字直接按位或进尾数的低位，就得到了 1024 + q：

$$
\texttt{half}(\texttt{0x6400}\ \vert\ q)=1024+q,\qquad (1024+q)-(1024+z)=q-z
$$

比如 q = 11、z = 8、s = 0.02：`0x6400` 和 `0x000B` 按位或得到 `0x640B`，FP16 值是 1035.0，减去 1032（1024 + z）得 3.0，乘 s 得 0.06。实际实现中用一条 `lop3` 同时处理两个 nibble 得到一个 `half2`，再用一条 `fma` 完成 $$h\cdot s-(1024+z)\cdot s$$（后一项预先算好），每个权重摊下来一两条指令。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-09-w4a16-kernel.svg" alt="W4A16 GEMM kernel 的数据流：HBM 到共享内存，寄存器内解包反量化，再送入 FP16 Tensor Core" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 9. W4A16 kernel 的一次 tile 迭代。小 batch 时 kernel 受带宽限制，ALU 大量空闲，解包和反量化的指令几乎是免费的，所以能接近 4× 的加速。
</div>

batch 变大之后 kernel 逐渐变成算力受限，反量化指令开始和 MMA 抢算力，收益就缩小了。Marlin 在 batch ≤ 16～32 时接近理想加速，有些实现在 token 多的时候干脆先整体反量化成 FP16，再调 cuBLAS。一个 W4A16 GEMM 大概是这样的：

```python
# 每个 threadblock 负责输出 Y 的一个 [BM, BN] tile
acc = zeros(BM, BN, fp32)
for k0 in range(0, K, BK):                       # 沿归约维 K 推进
    cp_async(smem_x, X[m0:m0+BM, k0:k0+BK])      # FP16 激活
    cp_async(smem_w, Wpacked[k0//8 : (k0+BK)//8, n0:n0+BN])   # int32, 每个装 8 个 nibble
    g = k0 // group_size                          # 这一段属于第几组
    cp_async(smem_s, scales[g, n0:n0+BN]); cp_async(smem_z, zeros[g, n0:n0+BN])
    wait_and_sync()
    for kk in range(0, BK, 16):                   # Tensor Core 的 MMA 粒度
        a = load_frag(smem_x, kk)                 # FP16 片段
        p = load_frag(smem_w, kk)                 # 打包的 int32
        b = dequant(p, smem_s, smem_z)            # lop3 + fma -> half2，寄存器内完成
        acc = mma_fp16(a, b, acc)                 # HMMA，FP32 累加
store(Y[m0:m0+BM, n0:n0+BN], acc.to(fp16))
```

### W4A4、FP8 与块缩放格式

其余几种低比特矩阵乘，结构上都是 W8A8 那条路径的变体：

- **W4A4（INT4）**：激活在线做 per-token INT4 量化（两个 INT4 打包成一个字节），权重离线做 per-channel INT4，送进 INT4 Tensor Core（Ampere / Ada 的 `mma` 支持 s4，Hopper 不支持），INT32 累加后在 epilogue 乘 $$s^x_i s^w_j$$。难点不在 kernel 而在精度：16 个码字很难容纳激活的离群值，这正是后面旋转和低秩方法要解决的问题；
- **FP8 W8A8**（Ada / Hopper）：把 $$x/s$$ 转成 E4M3 而不是取整成 INT8，FP8 Tensor Core 相乘后 FP32 累加，epilogue 同样乘两个 scale。scale 常用 per-tensor，或者激活 per-token、权重 per-channel；
- **块缩放格式**（Blackwell 的 MXFP8 / MXFP4 / NVFP4）：激活量化的 kernel 输出"低比特元素 + 每 16/32 个元素一个 scale"，Tensor Core 指令同时读元素和块 scale，在累加过程中逐块乘上去。这就是前面那条规则的硬件例外，沿 K 分块的 scale 由硬件处理，不需要单独的反量化步骤。

### 伪量化与真量化

读论文和开源代码的时候要分清两种"量化"：

- **伪量化（模拟量化）**：把权重量化之后立刻反量化回 FP16，仍然用 FP16 kernel 计算。用来研究精度、做搜索（AWQ 的 scale 搜索、GPTQ 的评测都是这样），**没有任何加速**；
- **真量化**：权重以打包的低比特格式存储，配合专用的 kernel（Marlin、ExLlamaV2、CUTLASS INT8/FP8、QuaRot 的 INT4 kernel、Nunchaku 等）执行，才有显存和速度上的收益。

一个模型"能量化到 W4A4 且精度不掉"和"W4A4 跑得快"是两件事，后者还要求激活量化、旋转这些在线操作的开销足够小。这也是 QuaRot、ConvRot、SVDQuant 这些论文都要花大量篇幅讨论 kernel 融合的原因。

## LLM 量化难在哪里

把前面的方法直接用到 LLM 上：每组算 absmax 当 scale，然后四舍五入，这叫 **RTN**（Round-To-Nearest），是所有算法的基线。

### RTN 能走多远

| 配置                 | RTN 的表现                          | 主要原因                                               | 对应算法                              |
| :------------------- | :---------------------------------- | :----------------------------------------------------- | :------------------------------------ |
| W8A16（per-channel） | 几乎无损                            | 权重分布温和，256 个码字足够                           | RTN 即可                              |
| W4A16（g128）        | 大模型尚可；小模型和 3 bit 明显掉点 | 16 个码字太粗，误差沿 K 维累积；少数重要权重被粗暴舍入 | GPTQ、AWQ                             |
| W8A8（per-token）    | 约 6.7B 以上的模型开始明显掉点      | 激活中出现系统性的离群值                               | LLM.int8()、SmoothQuant               |
| W4A4                 | 基本不可用                          | 离群值 + 只有 16 个码字                                | QuaRot / SpinQuant、ConvRot、SVDQuant |

表里的问题可以归成两类：权重在 4 bit 以下舍入误差太大，激活因为离群值而难以量化。

### 难点一：低比特权重的舍入误差

一个输出元素的误差，是所有输入通道误差的加权和：

$$
\Delta Y_{ij}=\sum_{k=1}^{K}\big(\hat W_{jk}-W_{jk}\big)\,X_{ik}
$$

RTN 让每个 $$\lvert\hat W_{jk}-W_{jk}\rvert$$ 各自最小，但它没考虑两件事：

- **误差可以相互抵消**。如果不同输入通道的激活是相关的，一个权重向上舍入的误差可以由另一个权重向下舍入来抵消，联合地决定舍入方向比各自就近舍入更好。这是 GPTQ 的出发点。
- **误差会被激活放大**。同样的权重误差，乘上大的激活就是大的输出误差。比如一组权重 $$w=[0.84,-0.40,0.47,-0.61]$$，对应的激活 $$x=[0.5,\ 8.0,\ 0.4,\ 0.6]$$，INT3 RTN 之后四个权重的误差分别是 [0, 0.12, 0.09, 0.05]，乘上激活之后对输出的贡献是 [0, **0.96**, 0.036, 0.030]，激活大的第 2 个通道几乎贡献了全部的误差。保护这类通道是 AWQ 的出发点。

### 难点二：激活里的离群值

CNN 时代激活的 INT8 量化几乎是标配，到了 LLM 却突然变难了，原因是 Transformer 的激活里有系统性的**离群值**（outliers）：

- **按通道出现**：LLM.int8() 发现，参数量超过约 6.7B 之后，某些固定的隐藏维度在几乎所有 token、大多数层上都会出现大幅值，比其他通道大几十到上百倍；
- **按 token 出现的超大激活**（massive activations）：少数 token（BOS、句号、换行符）在个别维度上出现比中位数大上千倍的值，起到类似 attention sink、固定偏置的作用；
- **权重很温和**：权重近似钟形分布，没有系统性的离群值，所以 INT8 权重几乎无损；
- **DiT 还会出现按行（token）的离群值**，以及随时间步变化的分布，后面讲扩散模型时再说。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-10-outliers.svg" alt="激活与权重的幅值热力图：激活有贯穿所有 token 的离群通道，权重没有明显结构" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 10. 激活与权重的幅值热力图（合成数据，对数色标）。激活里第 9、33、50 号通道是贯穿所有 token 的竖条，第 0 个 token 在 33 号通道上还有一个超大激活；权重没有明显的结构。
</div>

把它和前面的两个结论放在一起，困难的根源就清楚了：

- 离群值会把 scale 拉大，同组的正常值只剩下寥寥几个量化等级；截掉离群值又会丢掉关键的信息；
- 离群值**沿输入通道分布**，而能把它们隔离开的 per-input-channel 粒度恰好不能用于矩阵乘。per-token 量化只能给每一行选一个 scale，而每一行都含有离群值。

这个影响有多大？下面这个实验里 64 个通道中有 2 个离群通道。默认幅值是正常通道的 60 倍，做 INT8 per-token 量化，正常通道只能用到 255 个码字里的 44 个，等效位宽 5.5 bit，相对误差 19%：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>离群通道如何“吃掉”量化等级</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="ol-mag">离群通道幅值倍数 <input type="range" id="ol-mag" min="1" max="150" step="1" value="60"><output id="ol-mag-v">60×</output></label>
      <label for="ol-bits">位宽 <select id="ol-bits"><option value="4">INT4</option><option value="8" selected>INT8</option></select></label>
      <label for="ol-gran">激活粒度 <select id="ol-gran"><option value="tensor">per-tensor</option><option value="token" selected>per-token</option></select></label>
    </div>
    <div class="stats" id="ol-stats"></div>
    <div class="panel"><h5>正常通道的值被量化到哪些整数（直方图，横轴为整数码字）</h5><svg id="ol-hist" viewBox="0 0 640 170"></svg></div>
    <p class="small mute">64 个通道中有 2 个离群通道。把倍数调到 100 以上，INT8 per-token 下正常通道只剩十来个码字可用，相当于有效位宽不到 4 bit。</p>
  </div>
</div>

### 最早的解法：LLM.int8()

LLM.int8()（Dettmers et al., 2022）的思路最直接：把离群通道单独拿出来。激活里任何元素绝对值超过阈值 τ=6 的列记为离群列集合 O，这些列和权重里对应的输入通道走 FP16 矩阵乘，其余部分走 INT8（激活 per-token、权重 per-output-channel）：

$$
XW^\top=\underbrace{X_{:,O}\,W_{:,O}^\top}_{\text{FP16，约 0.1}\%\text{ 的维度}}+\underbrace{s^x\,\big(Q(X_{:,\bar O})\,Q(W_{:,\bar O})^\top\big)\,s^{w\top}}_{\text{INT8}}
$$

举个例子，$$X=\begin{bmatrix}0.5 & -1.1 & 8.3 & 0.7\\ -0.9 & 0.4 & -7.6 & 1.2\end{bmatrix}$$ 的第 3 列超过了阈值，单独拿出来走 FP16，剩下 3 列走 INT8，两部分相加之后最大误差是 0.005；不分解、直接 per-token INT8 的最大误差是 0.022，而且正常值 0.5 只被量化成了码字 8（scale = 8.3/127 = 0.065），有效精度非常低。

LLM.int8() 几乎无损，但要在运行时检测离群列、拆分矩阵、做两次 GEMM，**实际推理比 FP16 还慢**。它的价值在于把问题的本质揭示了出来，后面的算法都在找"既能处理离群值、又能保持规整计算"的办法。

### 方法全景

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-11-method-map.svg" alt="应对量化难点的五类方法：隔离、误差补偿、缩放迁移、正交旋转、低秩吸收" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 11. 应对量化难点的五类思路。③④⑤ 都利用了"计算不变性"：插入一对互逆的变换 A⁻¹A，让两侧都更容易量化。③ 的 A 是对角阵，④ 的 A 是正交阵，⑤ 则把权重拆成低秩部分和残差。② 和 ③④ 可以叠加使用。
</div>

| 算法               | 解决的难点                  | 手段              | 典型配置      |
| :----------------- | :-------------------------- | :---------------- | :------------ |
| GPTQ               | 低比特权重的舍入误差        | ② 误差补偿        | W4A16 / W3A16 |
| AWQ                | 少数重要权重主导误差        | ③ 对角缩放        | W4A16         |
| SmoothQuant        | 激活离群值（8 bit）         | ③ 对角缩放迁移    | W8A8          |
| QuaRot / SpinQuant | 激活与权重的离群值（4 bit） | ④ 正交旋转        | W4A4KV4       |
| ConvRot            | DiT 的行、列离群值          | ④ 分组正则旋转    | DiT W4A4      |
| SVDQuant           | DiT 的离群值                | ③ + ⑤ 平滑 + 低秩 | DiT W4A4      |

下面一个个看。和上篇读缓存方法一样，每个算法都按固定的几个问题去读：它解决什么问题、核心操作是什么、需要什么数据、推理时有没有额外开销。

## 怎么做：几种经典量化算法

### GPTQ：把误差"摊"给还没量化的权重

GPTQ（Frantar et al., ICLR 2023）是第一个能在几个 GPU 小时内把 175B 模型量化到 3～4 bit、精度损失又很小的方法。它只改权重（W4A16 / W3A16），核心思想是：**每量化一列权重，就立刻调整其余还没量化的权重，抵消这一列引入的输出误差**，调整的方向由输入的二阶统计量 $$H=2XX^\top$$ 决定。

| GPTQ       |                                                                     |
| :--------- | :------------------------------------------------------------------ |
| 解决的问题 | 4 bit 及以下的权重量化，RTN 逐个就近舍入，误差在输出端累积          |
| 核心操作   | 逐列量化，每量化一列就用 H⁻¹ 把这一列的误差补偿到还没量化的列       |
| 需要的数据 | 约 128 条校准样本，统计每层输入的 H = 2XXᵀ                          |
| 推理开销   | 无，产出标准的分组 INT4 权重                                        |
| 主要实现   | GPTQModel、llm-compressor；推理用 Marlin、ExLlamaV2 等 W4A16 kernel |

#### 问题定义：逐层重建

这一节沿用 GPTQ 论文的记号：权重 $$W\in\mathbb{R}^{d_{row}\times d_{col}}$$（行是输出通道，列是输入通道），校准输入 $$X\in\mathbb{R}^{d_{col}\times n}$$（每一列是一个 token 的输入向量）。目标是找到量化权重 $$\hat W$$，让这一层的输出变化最小：

$$
\hat W^\star=\arg\min_{\hat W}\ \big\lVert WX-\hat WX\big\rVert_F^2
$$

各行之间互不影响，可以逐行处理。对某一行 $$w$$，令扰动 $$\delta=\hat w-w$$：

$$
\lVert (w-\hat w)X\rVert^2=\delta\,XX^\top\delta^\top=\tfrac12\,\delta H\delta^\top,\qquad H=2XX^\top\in\mathbb{R}^{d_{col}\times d_{col}}
$$

有两个关键的观察：损失是 $$\delta$$ 的**精确**二次型，不需要泰勒近似；$$H$$ 只和输入有关，**所有行共享同一个 H**。

#### 前身：OBS / OBQ 的闭式解

假设决定把第 q 个权重量化成 $$\operatorname{quant}(w_q)$$，其余权重可以自由调整来补偿，这是一个带等式约束的二次优化：

$$
\min_{\delta}\ \tfrac12\,\delta H\delta^\top\quad\text{s.t.}\quad \delta_q=\operatorname{quant}(w_q)-w_q
$$

用拉格朗日乘子法：记 $$c=\operatorname{quant}(w_q)-w_q$$，$$\mathcal{L}=\tfrac12\delta H\delta^\top+\lambda(\delta e_q-c)$$，对 $$\delta$$ 求导置零得到 $$\delta^\top=-\lambda H^{-1}e_q$$，代入约束得 $$\lambda=-c/[H^{-1}]_{qq}$$，于是：

$$
\delta=-\frac{w_q-\operatorname{quant}(w_q)}{[H^{-1}]_{qq}}\cdot\big(H^{-1}\big)_{:,q},\qquad \Delta\mathcal{L}=\frac{\big(w_q-\operatorname{quant}(w_q)\big)^2}{2\,[H^{-1}]_{qq}}
$$

$$\delta$$ 的第 q 个分量正好等于 c，其余分量就是对其他权重的"补偿量"。$$H^{-1}$$ 的第 q 列告诉我们：第 q 个输入通道的误差，应该由哪些和它相关的通道、以多大的比例来弥补。量化完第 q 个权重之后把它从问题里移除，剩下权重的逆 Hessian 可以用一步高斯消元直接更新，不用重新求逆：

$$
H^{-1}_{-q}=\Big(H^{-1}-\frac{1}{[H^{-1}]_{qq}}\,H^{-1}_{:,q}H^{-1}_{q,:}\Big)_{\text{删去第 }q\text{ 行与列}}
$$

OBQ（Optimal Brain Quantization）对每一行贪心地选 $$\Delta\mathcal{L}$$ 最小的权重先量化，重复 $$d_{col}$$ 次。由于每行的量化顺序不同，每行都要维护自己的 $$H^{-1}$$，总复杂度是 $$O(d_{row}\cdot d_{col}^3)$$，对 4096×4096 的矩阵已经不可行了。

#### GPTQ 的三个改进

**所有行按相同的列顺序量化**。作者发现，对大模型来说贪心顺序相对固定顺序（从左到右）的优势很小。顺序固定之后，所有行共享同一串 $$H^{-1}$$ 的更新，只需要算一次，复杂度从 $$O(d_{row}d_{col}^3)$$ 降到 $$O(\max\{d_{row}d_{col}^2,\ d_{col}^3\})$$。具体做法是整列一起处理：量化第 j 列（所有行），再用同一个向量更新所有行的剩余列。

**Lazy batch 更新**。逐列更新整个剩余矩阵是"读写多、计算少"的操作，GPU 利用率很低。GPTQ 把列分成 B=128 的块，块内逐列量化、只更新块内的列；整块处理完之后，再把累积的误差一次性用矩阵乘更新到右侧的所有列：$$W_{:,\,i+B:}\mathrel{-}=E\cdot (H^{-1})_{i:i+B,\ i+B:}$$。数学上完全等价，但是把大量的向量操作合成了一次 GEMM。

**Cholesky 重构**。大模型上反复做消元会累积数值误差，让 $$H^{-1}$$ 失去正定性。作者注意到，从第 j 列开始消元之后的逆矩阵，它的第 j 行（除以对角元的平方根）恰好等于 $$H^{-1}$$ 的上三角 Cholesky 因子 U（$$H^{-1}=U^\top U$$）的第 j 行。所以只要一次性算出 U，更新公式就变成：

$$
\text{err}_j=\frac{w_j-\operatorname{quant}(w_j)}{U_{jj}},\qquad w_{j+1:}\leftarrow w_{j+1:}-\text{err}_j\cdot U_{j,\,j+1:}
$$

再加上阻尼项 $$H\leftarrow H+\lambda I$$，$$\lambda=1\%\times\operatorname{mean}(\operatorname{diag}H)$$，数值就很稳定了。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-12-gptq-order.svg" alt="GPTQ 的逐列量化与分块延迟更新" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 12. GPTQ 的执行顺序。从左到右逐列量化，第 j 列的误差 err_j 立即补偿到当前块里右侧的列；整块结束后，块内所有误差组成矩阵 E，用一次矩阵乘补偿到右侧的全部列。
</div>

#### 手算一遍：2 行 × 4 列

用一个能完整手算的例子走一遍。4 个输入通道、6 个校准 token，通道 1、2 强相关，通道 3、4 强相关（这正是 GPTQ 能发挥作用的地方）：

$$
X=\begin{bmatrix}0.2 & -0.7 & 2.0 & 1.4 & 0.0 & -0.1\\ -0.3 & -0.2 & 1.3 & 1.2 & 0.0 & 0.3\\ -0.6 & 1.6 & 0.4 & 1.2 & -1.0 & 1.3\\ -1.0 & 0.8 & 0.5 & 0.7 & -0.5 & 1.1\end{bmatrix},\qquad W=\begin{bmatrix}-0.96 & 0.76 & 0.10 & 0.38\\ -0.47 & -0.24 & -0.23 & 0.90\end{bmatrix}
$$

量化规则是 INT3 对称、每行一个 scale：$$s=\max\lvert w\rvert/3$$，$$q=\operatorname{clamp}(\operatorname{round}(w/s),-4,3)$$，第 1 行 $$s_1=0.32$$，第 2 行 $$s_2=0.30$$。GPTQ 在开始前就确定好 scale，之后只决定每个权重舍入到哪个格点。

**第 1 步，Hessian 与阻尼**：

$$
H=2XX^\top=\begin{bmatrix}13.00 & 8.66 & 2.22 & 2.22\\ 8.66 & 6.70 & 4.42 & 3.92\\ 2.22 & 4.42 & 14.42 & 9.70\\ 2.22 & 3.92 & 9.70 & 7.68\end{bmatrix}
$$

对角均值 10.45，阻尼 λ = 0.01×10.45 = 0.1045，加到对角线上得到 $$H_d$$。

**第 2 步，求逆再做 Cholesky 分解**：

$$
H_d^{-1}=\begin{bmatrix}1.4658 & -2.2991 & -0.1097 & 0.8764\\ -2.2991 & 3.8171 & 0.2123 & -1.5310\\ -0.1097 & 0.2123 & 0.4261 & -0.6066\\ 0.8764 & -1.5310 & -0.6066 & 1.4053\end{bmatrix}
$$

$$
U=\begin{bmatrix}1.2107 & -1.8990 & -0.0906 & 0.7239\\ 0 & 0.4592 & 0.0875 & -0.3403\\ 0 & 0 & 0.6405 & -0.7981\\ 0 & 0 & 0 & 0.3584\end{bmatrix}
$$

验证一下第一行：$$U_{11}=\sqrt{1.4658}=1.2107$$，$$U_{12}=-2.2991/1.2107=-1.8990$$。

**第 3 步，第 1 行逐列量化（s = 0.32）**：

| 列 j | 当前 $$w_j$$ | $$w_j/s$$ | q   | 量化值 | err = (w − 量化值) / U_jj | 更新右侧各列                                                              |
| :--- | :----------- | :-------- | :-- | :----- | :------------------------ | :------------------------------------------------------------------------ |
| 1    | −0.9600      | −3.000    | −3  | −0.96  | 0                         | 误差为 0，不更新                                                          |
| 2    | 0.7600       | 2.375     | 2   | 0.64   | 0.12/0.4592 = 0.2613      | w₃ = 0.10 − 0.2613×0.0875 = 0.0771；w₄ = 0.38 − 0.2613×(−0.3403) = 0.4689 |
| 3    | 0.0771       | 0.241     | 0   | 0.00   | 0.0771/0.6405 = 0.1204    | w₄ = 0.4689 − 0.1204×(−0.7981) = 0.5650                                   |
| 4    | 0.5650       | 1.766     | 2   | 0.64   | −0.2092                   | 最后一列，不用更新                                                        |

RTN 在第 4 列会得到 0.38/0.32 = 1.19 → **1**；GPTQ 因为前面两列都向下舍入了（输出偏小），把 w₄ 推高到了 0.565，最后取 **2**。单看这个权重，误差从 0.06 变成了 0.26，但是输出的误差变小了。

**第 4 步，第 2 行（s = 0.30），用同一个 U**：

| 列 j | 当前 $$w_j$$ | $$w_j/s$$ | q   | err                  | 更新                                    |
| :--- | :----------- | :-------- | :-- | :------------------- | :-------------------------------------- |
| 1    | −0.4700      | −1.567    | −2  | 0.13/1.2107 = 0.1074 | w₂ = −0.0361；w₃ = −0.2203；w₄ = 0.8223 |
| 2    | −0.0361      | −0.120    | 0   | −0.0786              | w₃ = −0.2134；w₄ = 0.7955               |
| 3    | −0.2134      | −0.711    | −1  | 0.1352               | w₄ = 0.9034                             |
| 4    | 0.9034       | 3.011     | 3   | —                    | —                                       |

**第 5 步，对比**：

| 方法 | 第 1 行码字       | 第 2 行码字        | 行 1 损失 | 行 2 损失 | 总损失 ‖(W−Ŵ)X‖² |
| :--- | :---------------- | :----------------- | :-------- | :-------- | :--------------- |
| RTN  | [−3, 2, 0, 1]     | [−2, −1, −1, 3]    | 0.2736    | 0.2636    | 0.5372           |
| GPTQ | [−3, 2, 0, **2**] | [−2, **0**, −1, 3] | 0.0585    | 0.0139    | 0.0724           |

总损失降低了 **7.4 倍**。这个例子只有 6 个 token，而且相关性很强，补偿的效果被放大了；在真实的 4096 维的层上，GPTQ 相对 RTN 的改进幅度会小一些，但在 3～4 bit 时依然很显著。

顺便可以验证一下 Cholesky 那个结论：按 OBQ 的方式，量化第 1 列之后对 $$H_d^{-1}$$ 做一步消元、删去第 1 行和第 1 列，剩下 3×3 矩阵的第一行除以对角元的平方根，得到 [0.4592, 0.0875, −0.3403]，和 U 的第 2 行完全一样；再消元一次得到 [0.6405, −0.7981]，和第 3 行一样。所以 GPTQ 只要做一次 Cholesky 分解，就拿到了所有消元步骤需要的行向量。

下面可以把这个例子一列一列地跑一遍，看每一步用到的 Cholesky 行，以及误差怎么补偿到右边的列；也可以换位宽，或者随机生成一个 4×8 的新例子：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>单步执行 GPTQ</div>
  <div class="box-b">
    <div class="ctrl">
      <button class="btn pri" type="button" id="gq-step">量化下一列</button>
      <button class="btn" type="button" id="gq-run">一次跑完</button>
      <button class="btn" type="button" id="gq-reset">重置为例题</button>
      <button class="btn" type="button" id="gq-rand">随机新例子（4×8）</button>
      <label for="gq-bits">位宽 <select id="gq-bits"><option value="2">INT2</option><option value="3" selected>INT3</option><option value="4">INT4</option></select></label>
    </div>
    <div class="stats" id="gq-stats"></div>
    <div class="grid2 one">
      <div class="panel"><h5>当前权重（蓝底＝已量化，橙框＝刚处理的列，绿色数字＝刚被补偿修改的值）</h5><div class="tw" id="gq-w"></div></div>
      <div class="panel"><h5>本步使用的 Cholesky 行 U[j, j:] 与误差</h5><div id="gq-u" class="small"></div></div>
    </div>
    <div class="panel" style="margin-top:12px"><h5>输出误差 ‖(W − Ŵ)X‖<sup>2</sup>：RTN 与 GPTQ</h5><div id="gq-cmp" class="small"></div></div>
  </div>
</div>

#### 参考实现

下面是和官方实现结构一致的精简版，`quant()` 用预先确定的 scale/zero 把一整列量化之后再反量化：

```python
import torch

def accumulate_hessian(H, n_seen, x):
    # x: [tokens, d_col]，本批校准输入；H 是流式均值，等价于 2/n * X^T X
    t = x.shape[0]
    H *= n_seen / (n_seen + t)
    n_seen += t
    x = (2.0 / n_seen) ** 0.5 * x.float()
    H += x.T @ x
    return H, n_seen

def gptq(W, H, quant, blocksize=128, percdamp=0.01):
    W = W.clone().float()                      # [d_row, d_col]
    d_col = W.shape[1]
    dead = torch.diag(H) == 0                  # 校准中从未激活的输入通道
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(d_col, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)           # H^-1
    U = torch.linalg.cholesky(Hinv, upper=True)  # H^-1 = U^T U
    Q = torch.zeros_like(W)
    for i1 in range(0, d_col, blocksize):
        i2 = min(i1 + blocksize, d_col)
        W1, U1 = W[:, i1:i2].clone(), U[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for j in range(i2 - i1):
            w = W1[:, j]
            q = quant(w)                       # 整列（所有行）一起量化
            Q[:, i1 + j] = q
            err = (w - q) / U1[j, j]
            W1[:, j:] -= err.unsqueeze(1) @ U1[j, j:].unsqueeze(0)   # 块内补偿
            Err1[:, j] = err
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]       # lazy batch：块外一次性补偿
    return Q
```

#### 工程细节

- **校准数据**：通常是从 C4 / WikiText 里随机抽的 128 段 × 2048 token。数据量不大，但要覆盖目标领域，代码或者多语言模型用通用文本校准会有偏差。
- **逐层顺序处理**：一次只把一个 Transformer block 放进显存。量化第 l 个 block 时，它的输入取**已经量化**的前 l−1 个 block 的输出，这样后面的层能"看到"并补偿前面层的误差。block 内部也按 q/k/v → o → gate/up → down 的顺序进行（true-sequential）。
- **分组 + GPTQ**：处理到某组（比如 128 列）的第一列时，用**当前已经被补偿修改过的**权重来计算这组的 scale / zero。
- **act-order（desc_act）**：按 $$\operatorname{diag}(H)$$（也就是该输入通道的激活能量）从大到小的顺序量化。重要的列先量化，这时还有很多列可以吸收它的误差，3 bit 下改进明显。代价是分组和列的顺序不再一致，需要 `g_idx` 记录每个输入通道属于哪一组，推理时读 scale 变成了不连续的访问；`static_groups` 选项先按原顺序算好各组的参数，可以避免这个问题。
- **阻尼**：`damp_percent` 默认 0.01，Cholesky 失败（H 非正定）时调到 0.1 左右。
- **成本**：OPT-175B / BLOOM-176B 在单张 A100 上约 4 个 GPU 小时，7B 的模型几分钟到十几分钟。
- **推理**：GPTQ 的输出就是普通的分组 INT4 权重（scale、zero、packed int32），和 RTN 的格式一样，用同样的 W4A16 kernel（ExLlamaV2、Marlin，vLLM 里是 `gptq_marlin`）。GPTQ 也经常作为"权重量化器"和其他方法叠加：SmoothQuant + GPTQ 做 W8A8，QuaRot / SpinQuant + GPTQ 做 W4A4。

### AWQ：按激活挑出重要权重，用缩放保护它们

AWQ（Activation-aware Weight Quantization，Lin et al., MLSys 2024 最佳论文）同样做 W4A16，但思路和 GPTQ 完全不同：它不做回归补偿，而是给每个输入通道乘一个缩放因子，让"重要"的权重在量化时相对误差更小。整个过程只需要激活的平均幅值和一个一维的网格搜索。

| AWQ        |                                                                      |
| :--------- | :------------------------------------------------------------------- |
| 解决的问题 | 4 bit 权重量化中，少数和大激活相乘的"重要"权重主导了输出误差         |
| 核心操作   | 按激活幅值对输入通道做等价缩放，放大重要权重，降低它们的相对量化误差 |
| 需要的数据 | 校准样本上每个输入通道的平均激活幅值；网格搜索一个指数 α             |
| 推理开销   | 无，缩放已经融合进前一个算子                                         |
| 主要实现   | llm-compressor（AutoAWQ 已停止维护）；推理用 Marlin 等 W4A16 kernel  |

#### 观察：1% 的权重决定了大部分误差

作者在 OPT 上做了个实验：INT3 量化时，如果把 0.1%～1% 的权重通道保留成 FP16，困惑度几乎能回到 FP16 的水平。关键在于**怎么挑这 1%**：按权重幅值挑几乎没有改善，按激活幅值挑（对应输入通道 k 的 $$\overline{\lvert X_{:,k}\rvert}$$ 大）改善显著。直觉很简单，输出误差是 $$\sum_k (\hat W_{jk}-W_{jk})X_{ik}$$，激活大的通道会把这个通道上权重的量化误差放大。但混合精度（同一个矩阵里一部分 FP16、一部分 INT4）对硬件很不友好，于是 AWQ 想找一种不用混合精度也能"保护"这些通道的办法。

#### 缩放为什么能减小误差

考虑一组共享 scale 的权重里的一个权重 w 和它对应的输入 x，按组内 absmax 量化：$$Q(w)=\Delta\cdot\operatorname{Round}(w/\Delta)$$，$$\Delta=\max\lvert w_{\text{group}}\rvert/q_{\max}$$。量化误差对输出的贡献是：

$$
\operatorname{Err}\big(Q(w)\,x\big)=\Delta\cdot\operatorname{RoundErr}\big(\tfrac{w}{\Delta}\big)\cdot x
$$

现在把这个输入通道的权重乘 $$s>1$$、激活除以 s（数学上等价，$$wx=(ws)(x/s)$$）：

$$
\operatorname{Err}\big(Q(ws)\cdot\tfrac{x}{s}\big)=\Delta'\cdot\operatorname{RoundErr}\big(\tfrac{ws}{\Delta'}\big)\cdot x\cdot\frac1s
$$

RoundErr 大致服从 [−0.5, 0.5] 上的均匀分布，平均幅度和 s 无关，所以两者的误差之比约为 $$\dfrac{\Delta'}{\Delta}\cdot\dfrac{1}{s}$$：

- 只放大一个通道、而且没改变组内最大值时，$$\Delta'=\Delta$$，**这个通道的误差缩小为 1/s**；
- s 过大，被放大的权重成了组内新的最大值，$$\Delta'$$ 变大，**组内其他通道的误差都会变大**。

所以存在一个最优的 s。被缩小的激活 x/s 也不需要真的在运行时去除，这个 1/s 会被融合进前一个算子。

还是用前面那组权重 $$w=[0.84,-0.40,0.47,-0.61]$$、输入 $$x=[0.5,\ 8.0,\ 0.4,\ 0.6]$$，INT3 对称（q ∈ [−4, 3]，Δ 等于组内绝对值最大值除以 3），精确输出 wx = −2.958，只对第 2 个通道乘 s：

| s   | w·s                        | Δ     | 码字 q         | ŵ = Δq / s                    | 逐通道误差 abs(ŵ − w)·x      | 输出误差 Σ(ŵ−w)x |
| :-- | :------------------------- | :---- | :------------- | :---------------------------- | :--------------------------- | :--------------- |
| 1   | [0.84, −0.40, 0.47, −0.61] | 0.280 | [3, −1, 2, −2] | [0.84, −0.28, 0.56, −0.56]    | [0, **0.960**, 0.036, 0.030] | 1.026            |
| 1.5 | [0.84, −0.60, 0.47, −0.61] | 0.280 | [3, −2, 2, −2] | [0.84, −0.373, 0.56, −0.56]   | [0, 0.213, 0.036, 0.030]     | 0.279            |
| 2   | [0.84, −0.80, 0.47, −0.61] | 0.280 | [3, −3, 2, −2] | [0.84, −0.42, 0.56, −0.56]    | [0, 0.160, 0.036, 0.030]     | −0.094           |
| 3   | [0.84, −1.20, 0.47, −0.61] | 0.400 | [2, −3, 1, −2] | [0.80, −0.40, 0.40, −0.80]    | [0.020, 0, 0.028, **0.114**] | −0.162           |
| 4   | [0.84, −1.60, 0.47, −0.61] | 0.533 | [2, −3, 1, −1] | [1.067, −0.40, 0.533, −0.533] | [**0.113**, 0, 0.025, 0.046] | 0.185            |

- s = 1：重要通道的权重误差只有 0.12，但是被 x = 8 放大成了 0.96，几乎贡献了全部的输出误差；
- s = 1.5、2：Δ 不变（组内最大值还是 0.84），重要通道的误差被压缩，输出误差降到原来的 1/4 到 1/10；
- s = 3、4：−0.40 × s 的绝对值超过了 0.84，成为新的组内最大值，Δ 增大到 0.40、0.53，其他三个通道的误差明显变大（比如第 4 个通道从 0.03 涨到了 0.114）。

单个例子里误差有正有负，会相互抵消，所以不是严格单调的。真实的层里有成千上万个输出和 token，AWQ 直接拿整层输出的 MSE 当搜索目标，自然地平衡了这两种效应。

#### 搜索目标：一个参数 α

给每个输入通道单独搜索 s，空间太大了。AWQ 用激活的平均幅值 $$s_X$$（每个输入通道的 $$\overline{\lvert X_{:,k}\rvert}$$）构造一族候选，只搜一个指数 α：

$$
s=s_X^{\ \alpha},\qquad \alpha^\star=\arg\min_{\alpha\in\{0,\frac{1}{20},\dots,\frac{19}{20}\}}\Big\lVert \big(X\operatorname{diag}(s)^{-1}\big)\,Q\big(W\operatorname{diag}(s)\big)^{\top}-XW^\top\Big\rVert^2
$$

α = 0 退化成普通的 RTN；α 越大，越偏向保护激活大的通道。实现里还会把 s 归一化为 $$s/\sqrt{\max(s)\min(s)}$$，避免整体缩放带来的无意义变化。搜完之后再做一次**截断搜索**：每组权重尝试把 absmax 缩小到 (1 − i/20) 倍（最多缩小一半），同样以输出 MSE 为目标选最优的截断。

为什么不用梯度优化？Q 不可导，而且只搜一个标量，对校准集的依赖很弱，不容易过拟合。论文的实验也显示 AWQ 对校准数据分布的变化比 GPTQ 更稳健。

下面是一个玩具层上的 α 搜索，可以拖动 α 看输出误差怎么变化。默认 INT3 下 RTN（α=0）的 MSE 是 1.668，最优的 α=0.35 时降到 0.544：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>AWQ 的 α 搜索</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="aw-a">α <input type="range" id="aw-a" min="0" max="1" step="0.05" value="0.5"><output id="aw-a-v">0.50</output></label>
      <label for="aw-bits">位宽 <select id="aw-bits"><option value="3" selected>INT3</option><option value="4">INT4</option></select></label>
      <label for="aw-sal">重要通道激活倍数 <input type="range" id="aw-sal" min="1" max="30" step="1" value="12"><output id="aw-sal-v">12×</output></label>
      <button class="btn" type="button" id="aw-best">跳到最优 α</button>
    </div>
    <div class="stats" id="aw-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>输出 MSE 随 α 变化</h5><svg id="aw-curve" viewBox="0 0 320 190"></svg></div>
      <div class="panel"><h5>当前 α 下每个输入通道的缩放因子 s（重要通道为橙色）</h5><svg id="aw-s" viewBox="0 0 320 190"></svg></div>
    </div>
    <p class="small mute">玩具层：32 个输入通道（其中 3 个是重要通道）、24 个输出通道、128 个 token，非对称量化，组大小 16。虚线表示“把 3 个重要通道保留为 FP16、其余 RTN”的混合精度结果，可以看作这一思路的理想上限（但对硬件不友好）。AWQ 只用纯低比特格式，就把 RTN 与这个上限之间的差距缩小了一大截。</p>
  </div>
</div>

#### 缩放融合到哪里

x/s 里的除法必须离线消化掉，否则每次推理都要多一个逐元素的算子。AWQ 把它融合进产生 x 的那个算子：RMSNorm 的 γ，或者前一个线性层的输出行。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-13-awq-fusion.svg" alt="Llama 解码层中 AWQ 缩放的四个融合位置" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 13. AWQ 在 Llama 解码层中的四组缩放（AutoAWQ 的实现）。s1、s3 融进 RMSNorm 的权重；s2 融进 v_proj 的输出行（attention 对 V 是按 token 加权求和，不混合通道，所以按通道的缩放可以穿过它）；s4 融进 up_proj 的输出行（SiLU(g)⊙u 对 u 是逐元素线性的）。GQA 模型里 v_proj 和 o_proj 的维度对不上，通常跳过 s2。
</div>

这里有两个我一开始理解错的地方。

**不是只缩放一列**。上面的例子为了看清机制只放大了一个通道，实际算法是给**每个输入通道** k 都算一个 $$s_k$$（$$s=s_X^{\alpha}$$ 再归一化），也就是权重（形状 [out, in]）的每一列乘以各自的 $$s_k$$：激活大的通道 $$s_k>1$$，激活小的通道 $$s_k<1$$。同一列的所有输出行共用同一个 $$s_k$$，因为补偿要在它们共享的输入 $$x_k$$ 上做；共享输入的几个线性层（比如 q/k/v）也共用同一组 s。

**推理时没有"恢复"这一步**。由 $$XW^\top=(X\operatorname{diag}(s)^{-1})(W\operatorname{diag}(s))^\top$$，s 被同时藏进了两个离线保存的张量里：

- 权重侧：保存的是 $$W\operatorname{diag}(s)$$ 量化后的 INT4 码字，以及分组的量化 scale / zero；
- 激活侧：$$\operatorname{diag}(s)^{-1}$$ 已经融合进了上游的算子（RMSNorm 的 γ 除以 s，或者前一个线性层的输出行除以 s），上游输出的已经是 X/s。

kernel 只做普通的 W4A16：用分组的 scale / zero 把 INT4 反量化成 $$W\operatorname{diag}(s)$$ 的近似值，再和输入相乘，两个 s 在乘积里自然抵消。要区分两种 scale：AWQ 的 $$s_k$$ 是等价变换，离线融合之后就"消失"了；反量化用的是量化 scale（每 128 个元素一组），随 INT4 权重一起保存。

用上面的例题（第 2 个通道 s = 2）验证一下：保存的权重是 w·s = [0.84, −0.80, 0.47, −0.61]，INT3 量化后是 [0.84, −0.84, 0.56, −0.56]；上游 RMSNorm 的 γ 已经除过 s，送来的激活是 x/s = [0.5, 4.0, 0.4, 0.6]。直接相乘：0.42 − 3.36 + 0.224 − 0.336 = −3.052，和精确值 −2.958 只差 0.094；不缩放的 RTN 结果是 −1.932，差了 1.026。

#### 离线流程与精简实现

1. 用约 128 条校准样本跑模型，逐个 block 收集每组线性层的输入 x；
2. 对每组（比如共享输入的 q/k/v）计算 $$s_X$$，网格搜索 α，选出 s，比较的是整个子模块（比如整个 self_attn）的输出 MSE；
3. 把 s 融合进前一个算子和当前的权重（真正修改模型参数）；
4. 对每组权重搜索最优的截断比例（q/k 通常跳过截断）；
5. 对修改后的权重做普通的分组非对称 INT4 量化（g128），打包成 int32 存储。

下面代码里的 `pseudo_quant(w * s) / s` 只出现在搜索阶段：它把 1/s 临时乘回权重，这样直接喂原始输入 x 就能得到等价的输出，方便比较误差。搜索结束后由 `apply_scale` 做真正的融合。

```python
@torch.no_grad()
def pseudo_quant(w, n_bit=4, group=128):          # 非对称分组量化 → 立即反量化
    shape = w.shape
    w = w.reshape(-1, group)
    mx, mn = w.amax(1, keepdim=True), w.amin(1, keepdim=True)
    qmax = 2 ** n_bit - 1
    s = (mx - mn).clamp(min=1e-5) / qmax
    z = (-mn / s).round()
    q = (w / s).round().add(z).clamp(0, qmax)
    return ((q - z) * s).reshape(shape)

@torch.no_grad()
def awq_search(module, linears, x, n_grid=20):
    """module: 要比较输出的子模块；linears: 以 x 为输入的线性层"""
    s_x = x.abs().reshape(-1, x.shape[-1]).float().mean(0)   # 每个输入通道的平均幅值
    y_ref = module(x)
    orig = [fc.weight.data.clone() for fc in linears]
    best_err, best_s = float("inf"), None
    for i in range(n_grid):
        alpha = i / n_grid
        s = s_x.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for fc, w in zip(linears, orig):
            fc.weight.data = pseudo_quant(w * s) / s           # Q(W·diag(s))·diag(s)^-1
        err = (module(x) - y_ref).float().pow(2).mean().item()
        if err < best_err:
            best_err, best_s = err, s.clone()
    for fc, w in zip(linears, orig):
        fc.weight.data = w
    return best_s

@torch.no_grad()
def apply_scale(prev_op, linears, s):
    if prev_op.weight.dim() == 1:                  # RMSNorm / LayerNorm
        prev_op.weight.div_(s)
        if getattr(prev_op, "bias", None) is not None: prev_op.bias.div_(s)
    else:                                          # 前一个是 Linear：缩放它的输出行
        prev_op.weight.div_(s.view(-1, 1))
        if prev_op.bias is not None: prev_op.bias.div_(s)
    for fc in linears:
        fc.weight.mul_(s.view(1, -1))              # 当前层的输入列
```

AWQ 的产物和 GPTQ 一样是标准的 W4A16 分组权重（通常 g128、带零点），用同类的 kernel 执行，vLLM 里 AWQ 模型在 Ampere 及以上的 GPU 默认走 `awq_marlin`。两者放在一起对比：

| 维度         | GPTQ                                                  | AWQ                                  |
| :----------- | :---------------------------------------------------- | :----------------------------------- |
| 使用的信息   | 二阶统计 $$H=2XX^\top$$（通道间的相关性）             | 一阶统计：每个通道的平均幅值 $$s_X$$ |
| 如何改动权重 | 逐列量化，用 H⁻¹ 修改剩余权重做误差补偿               | 按输入通道等价缩放 + 截断，不做回归  |
| 优化目标     | 逐层的 $$\lVert WX-\hat WX\rVert^2$$                  | 子模块输出的 MSE，网格搜索 α         |
| 对校准集     | 拟合能力强，有过拟合的风险                            | 参数极少，更稳健                     |
| 可组合性     | 可以先做 AWQ 缩放再做 GPTQ，产出格式相同，kernel 通用 | 同左                                 |

### SmoothQuant：把激活的难度"搬"给权重

SmoothQuant（Xiao et al., ICML 2023）的目标是 **W8A8**：权重和激活都用 INT8，从而用上 INT8 Tensor Core。前面说过，激活的离群值集中在少数输入通道上，而 per-input-channel 的 scale 又不能用于整数矩阵乘。SmoothQuant 的解法是：**离线地把每个输入通道的 scale 乘进权重**，让激活本身变得平滑。

| SmoothQuant |                                                                  |
| :---------- | :--------------------------------------------------------------- |
| 解决的问题  | 激活离群值让 W8A8 的激活量化误差过大                             |
| 核心操作    | 用对角缩放把激活的离群值"迁移"到权重上，让两边都容易量化         |
| 需要的数据  | 校准得到的每个输入通道的激活最大值，以及权重每个输入通道的最大值 |
| 推理开销    | 无，缩放已融合进 LayerNorm / RMSNorm                             |
| 主要实现    | llm-compressor、TensorRT Model Optimizer；推理用 INT8 GEMM       |

#### 平滑变换

这一节沿用论文的记号 $$Y=XW$$，$$X\in\mathbb{R}^{T\times C_{in}}$$，$$W\in\mathbb{R}^{C_{in}\times C_{out}}$$。对任意正向量 $$s\in\mathbb{R}^{C_{in}}$$：

$$
Y=XW=\big(X\operatorname{diag}(s)^{-1}\big)\cdot\big(\operatorname{diag}(s)\,W\big)=\hat X\hat W
$$

第 j 个输入通道的激活除以 $$s_j$$，权重的第 j 行乘以 $$s_j$$。s 的选择要在两边之间取一个平衡：

$$
s_j=\frac{\max(\lvert X_j\rvert)^{\alpha}}{\max(\lvert W_j\rvert)^{1-\alpha}}
$$

α 叫**迁移强度**。α = 0 时 $$s_j=1/\max\lvert W_j\rvert$$（把权重压平，难度全给激活）；α = 1 时 $$s_j=\max\lvert X_j\rvert$$（把激活压平，难度全给权重）。论文里 OPT、BLOOM 取 0.5，离群值更严重的 GLM-130B 取 0.75；Llama 系列实践中常取 0.8 左右，最好用校准集搜一下。

α = 0.5 有一个很漂亮的性质：平滑之后 $$\max\lvert\hat X_j\rvert=\max\lvert X_j\rvert/s_j=\sqrt{\max\lvert X_j\rvert\max\lvert W_j\rvert}$$，$$\max\lvert\hat W_j\rvert=\max\lvert W_j\rvert\,s_j=\sqrt{\max\lvert X_j\rvert\max\lvert W_j\rvert}$$，**每个通道上激活与权重的最大值变得相等**，两边的量化难度被均分了。

用一个 3 个 token × 4 个通道的例子看一下，第 2 个通道是离群通道：

$$
X=\begin{bmatrix}0.6 & -45.0 & 1.1 & -0.8\\ -1.2 & 62.0 & 0.5 & 0.9\\ 0.4 & -38.0 & -1.4 & 1.3\end{bmatrix},\quad W=\begin{bmatrix}0.40 & -0.25 & 0.10\\ 0.05 & 0.08 & -0.06\\ -0.30 & 0.55 & 0.20\\ 0.15 & -0.20 & 0.45\end{bmatrix}
$$

**平滑前**，per-tensor 对称 INT8 的 $$s_X=62/127=0.4882$$，正常通道的值都在 ±1.4 以内，除以 0.488 之后只剩 −3…3：

$$
X_q=\begin{bmatrix}1 & -92 & 2 & -2\\ -2 & 127 & 1 & 2\\ 1 & -78 & -3 & 3\end{bmatrix}
$$

三个正常通道一共只用到 7 个码字，相当于 3 bit，输出的相对误差 $$\lVert\hat Y-Y\rVert/\lVert Y\rVert$$ = 3.81%。

**计算平滑因子（α = 0.5）**：

| 通道 j                                                       | 1      | 2       | 3      | 4      |
| :----------------------------------------------------------- | :----- | :------ | :----- | :----- |
| $$\max\lvert X_j\rvert$$                                     | 1.2    | 62.0    | 1.4    | 1.3    |
| $$\max\lvert W_j\rvert$$（第 j 行）                          | 0.40   | 0.08    | 0.55   | 0.45   |
| $$s_j$$                                                      | 1.7321 | 27.8388 | 1.5954 | 1.6997 |
| 平滑后 $$\max\lvert\hat X_j\rvert=\max\lvert\hat W_j\rvert$$ | 0.6928 | 2.2271  | 0.8775 | 0.7649 |

比如 $$s_2=\sqrt{62}/\sqrt{0.08}=7.874/0.2828=27.84$$。平滑之后：

$$
\hat X=\begin{bmatrix}0.346 & -1.616 & 0.690 & -0.471\\ -0.693 & 2.227 & 0.313 & 0.530\\ 0.231 & -1.365 & -0.878 & 0.765\end{bmatrix},\qquad \hat W=\begin{bmatrix}0.693 & -0.433 & 0.173\\ 1.392 & 2.227 & -1.670\\ -0.479 & 0.878 & 0.319\\ 0.255 & -0.340 & 0.765\end{bmatrix}
$$

$$\hat X\hat W=Y$$ 是精确相等的。激活的通道间差距从 62/1.2 ≈ 52 倍缩小到 2.23/0.69 ≈ 3.2 倍；权重的第 2 行被放大了，但权重本来就好量化。

**平滑后**，per-tensor 对称 INT8 的 $$s_{\hat X}=2.2271/127=0.01754$$，正常通道现在能用到 ±50 的码字：

$$
\hat X_q=\begin{bmatrix}20 & -92 & 39 & -27\\ -40 & 127 & 18 & 30\\ 13 & -78 & -50 & 44\end{bmatrix},\qquad \hat W_q=\begin{bmatrix}40 & -25 & 10\\ 79 & 127 & -95\\ -27 & 50 & 18\\ 15 & -19 & 44\end{bmatrix}
$$

输出的相对误差降到 0.45%，是平滑前的 1/8.4。

把这个例子推广一下：下面的玩具层有 2 个离群通道，可以拖动 α，也可以调离群通道的倍数、位宽和粒度。默认 W8A8 per-tensor 下，不平滑的输出误差是 1.68%，最优的 α = 0.45 时降到 0.83%；α 太大时难度全部压到权重上，误差又会回升。把位宽换成 W4A4 再试一下：不平滑是 15.77%，最优 α 下也还有 14.01%，怎么调 α 都救不回来，这就是后面旋转类方法要解决的问题：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>拖动迁移强度 α</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="sq-a">α <input type="range" id="sq-a" min="0" max="1" step="0.05" value="0.5"><output id="sq-a-v">0.50</output></label>
      <label for="sq-mag">离群通道倍数 <input type="range" id="sq-mag" min="1" max="100" step="1" value="40"><output id="sq-mag-v">40×</output></label>
      <label for="sq-bits">位宽 <select id="sq-bits"><option value="8" selected>W8A8</option><option value="6">W6A6</option><option value="4">W4A4</option></select></label>
      <label for="sq-gran">粒度 <select id="sq-gran"><option value="tensor" selected>per-tensor</option><option value="token">激活 per-token + 权重 per-channel</option></select></label>
    </div>
    <div class="stats" id="sq-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>每个输入通道的 max|X|（灰＝平滑前，蓝＝平滑后，对数纵轴）</h5><svg id="sq-x" viewBox="0 0 320 170"></svg></div>
      <div class="panel"><h5>每个输入通道的 max|W|（灰＝平滑前，绿＝平滑后，对数纵轴）</h5><svg id="sq-w" viewBox="0 0 320 170"></svg></div>
    </div>
    <div class="panel" style="margin-top:12px"><h5>输出相对误差随 α 变化（虚线＝不做平滑）</h5><svg id="sq-c" viewBox="0 0 640 170"></svg></div>
  </div>
</div>

#### s 融合进哪里

和 AWQ 一样，$$\operatorname{diag}(s)^{-1}$$ 融合进前面的 LayerNorm / RMSNorm：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sigma}+\beta\quad\Rightarrow\quad \gamma'=\gamma/s,\ \ \beta'=\beta/s;\qquad W'_{j,:}=s_j\,W_{j,:}
$$

PyTorch 里 `nn.Linear.weight` 的形状是 [out, in]，对应的操作是 `weight[:, j] *= s[j]`。原始实现只对**输入来自 LayerNorm 的线性层**做平滑：attention 的 QKV 投影和 FFN 的第一层（Llama 里是 gate/up）。o_proj 和 down_proj 的输入离群值比较轻，直接量化成 INT8；也可以像 AWQ 那样把 s 融进 v_proj / up_proj 的输出行来处理它们。

#### 推理配置

| 配置             | 权重        | 激活            | 说明                                           |
| :--------------- | :---------- | :-------------- | :--------------------------------------------- |
| SmoothQuant-O1   | per-tensor  | per-token 动态  | 精度最高                                       |
| SmoothQuant-O2   | per-tensor  | per-tensor 动态 | 折中                                           |
| SmoothQuant-O3   | per-tensor  | per-tensor 静态 | 最快，scale 全部离线确定                       |
| 现代推理框架常用 | per-channel | per-token 动态  | scale 外积在 epilogue 里完成，几乎没有额外开销 |

执行路径就是前面 W8A8 那张图：RMSNorm kernel 输出时直接做 per-token 量化 → CUTLASS INT8 GEMM（INT32 累加）→ epilogue 乘 $$s^x_i s^w_j$$ 并转回 FP16。论文还把 attention 里的两个批量矩阵乘（$$QK^\top$$ 和 $$PV$$）也做成了 INT8，softmax、LayerNorm 保持 FP16，报告的收益是最高约 1.56× 加速、2× 显存节省，精度几乎无损。

校准统计和平滑的代码（和官方实现结构一致）：

```python
@torch.no_grad()
def collect_act_scales(model, calib_batches):
    act_max = {}
    def hook(name):
        def f(mod, inp, out):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).float()
            act_max[name] = torch.maximum(act_max[name], x) if name in act_max else x
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear)]
    for batch in calib_batches:            # 例如 512 条样本
        model(batch)
    for h in hs: h.remove()
    return act_max                         # 每个线性层输入的逐通道 max|X_j|

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    # fcs 共享同一个输入（如 q/k/v），取它们在每个输入通道上的最大权重
    w_max = torch.cat([fc.weight.abs().amax(0, keepdim=True) for fc in fcs]).amax(0).clamp(min=1e-5)
    s = (act_scales.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
    ln.weight.div_(s)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s)
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))       # weight: [out, in]，乘在输入列上
```

#### 和 AWQ 的缩放有什么不同

SmoothQuant 和 AWQ 第一次看很容易混在一起：两者用的是同一个等价变换，s 都是每个输入通道一个，都融合进上游算子，推理时都没有额外开销，连方向都一样（激活越大的通道 s 越大，激活被缩小、权重被放大）。真正的区别在于**为什么要缩放**和**怎么选 s**：

|                   | SmoothQuant                                    | AWQ                                                           |
| :---------------- | :--------------------------------------------- | :------------------------------------------------------------ |
| 目标配置          | W8A8：激活和权重都量化                         | W4A16：只量化权重，激活保持 FP16                              |
| 要解决的问题      | 激活离群值让激活的 scale 过大                  | 少数和大激活相乘的权重主导了输出误差                          |
| 缩放的作用        | 缩小激活的离群通道，把难度转移给权重           | 放大重要权重，降低它们的相对量化误差；激活变小只是副作用      |
| s 的公式          | 激活最大值的 α 次方除以权重最大值的 1−α 次方   | 激活平均幅值的 α 次方，再归一化，不看权重                     |
| α 怎么定          | 每个模型一个超参数（常取 0.5～0.85），闭式计算 | 每组层网格搜索 20 个点，以子模块输出 MSE 为目标，另加截断搜索 |
| 误差为什么变小    | 激活的 scale 变小，正常通道分到更多码字        | 组内最大值不变时，重要通道的误差缩小为 1/s                    |
| s 过大的代价      | 权重的离群通道被放大，权重的量化误差变大       | 被放大的权重成为组内新的最大值，同组其他权重的误差变大        |
| 融合位置（Llama） | QKV 与 gate/up 的输入（LayerNorm 之后）        | 另外加上 v_proj→o_proj、up_proj→down_proj，共 4 处            |

一句话：SmoothQuant 是在激活和权重**之间**分摊难度，因为两边都要量化；AWQ 是在权重**内部**分配精度，因为只有权重要量化。

### QuaRot 与 SpinQuant：用旋转把离群值抹平

SmoothQuant 用对角矩阵在激活和权重之间"搬运"难度，到了 W4A4 就不够用了：4 bit 只有 16 个码字，搬到哪一边都放不下。旋转类方法换了一个思路：**用正交矩阵把离群值的能量均匀地分散到所有通道**，让激活和权重同时变得"没有离群值"。

| QuaRot / SpinQuant |                                                                                 |
| :----------------- | :------------------------------------------------------------------------------ |
| 解决的问题         | W4A4 时对角缩放不够用，激活、权重、KV Cache 必须同时没有离群值                  |
| 核心操作           | 正交（Hadamard）旋转把离群值的能量摊到所有通道；SpinQuant 进一步学习旋转矩阵    |
| 需要的数据         | QuaRot 本身不需要（权重用 GPTQ 时需要校准集）；SpinQuant 需要数百条样本训练旋转 |
| 推理开销           | 少量在线 Hadamard 变换（FWHT），其余旋转离线融进权重                            |
| 主要实现           | QuaRot / SpinQuant 官方代码；推理用 INT4 GEMM（Ampere / Ada）                   |

#### 计算不变性

这一节用数学记号 $$Y=XW$$，$$W\in\mathbb{R}^{C_{in}\times C_{out}}$$。对任意正交矩阵 $$Q$$（$$QQ^\top=I$$）：

$$
Y=XW=(XQ)(Q^\top W)
$$

和对角缩放不同，正交变换保持向量的长度：$$\lVert xQ\rVert=\lVert x\rVert$$。如果 x 有一个幅值为 a 的尖峰，经过一个"足够随机"的正交变换之后，能量 $$\lVert x\rVert^2$$ 会近似均匀地分到 C 个坐标上，每个坐标大约是 $$\pm\lVert x\rVert/\sqrt C$$。离群值没有消失，而是被"抹平"了。衡量这一点的指标是峰均比 $$\max\lvert x\rvert/\operatorname{rms}(x)$$，旋转之后会接近高斯分布的水平。

看一个 8 维的例子，一个 token 的激活和权重矩阵的某一列：

```text
x = [0.3, -0.5, 9.0, 0.2, -0.4, 0.6, -0.1, 0.4]
w = [0.21, -0.35, 0.12, 0.40, -0.18, 0.27, -0.30, 0.15]
x·w = 1.722
```

**不旋转，直接对称 INT4**。$$s_x=9.0/7=1.286$$，除了 9.0，其余 7 个值除以 1.286 都在 ±0.47 以内，全部变成 0，$$\hat x=[0,0,9,0,0,0,0,0]$$；$$s_w=0.40/7$$，$$w_q=[4,-6,2,7,-3,5,-5,3]$$。点积是 1.029，误差 **−0.693（−40%）**。

**先乘归一化的 Hadamard 矩阵 $$H_8/\sqrt8$$，再 INT4**：

```text
xH = [3.359, 2.864, -3.359, -3.005, 3.005, 3.924, -3.288, -2.652]   峰均比 2.81 → 1.23
wH = [0.113, -0.219, -0.149, 0.297, 0.156, 0.417, -0.318, 0.297]
(xH)·(wH) = 1.722
```

$$s_x=3.924/7=0.5606$$，$$x_q=[6,5,-6,-5,5,7,-6,-5]$$；$$s_w=0.4172/7=0.0596$$，$$w_q=[2,-4,-2,5,3,7,-5,5]$$。整数点积乘两个 scale 得到 1.604，误差 **−0.118（−6.9%）**，降低了约 6 倍。

注意 w 本来就没有离群值，旋转之后仍然没有，**旋转对"已经平坦"的向量基本无害**。这是它比对角缩放更适合 W4A4 的原因，不需要在两边之间取舍。

下面可以调离群值的幅度、维度和位宽，也可以关掉随机符号看看区别。默认 64 维、离群值 12、INT4 时，峰均比从 6.90 降到 2.10，和一个普通权重向量做点积的误差从 63.2% 降到 5.1%：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>旋转如何抹平离群值</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="rt-mag">离群值幅度 <input type="range" id="rt-mag" min="0" max="40" step="0.5" value="12"><output id="rt-mag-v">12.0</output></label>
      <label for="rt-n">维度 <select id="rt-n"><option value="16">16</option><option value="64" selected>64</option><option value="256">256</option></select></label>
      <label for="rt-bits">位宽 <select id="rt-bits"><option value="4" selected>INT4</option><option value="8">INT8</option></select></label>
      <label for="rt-sign"><input type="checkbox" id="rt-sign" checked> 随机符号 diag(±1)</label>
      <button class="btn" type="button" id="rt-new">换一组数据</button>
    </div>
    <div class="stats" id="rt-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>旋转前的 x（橙色为离群通道）</h5><svg id="rt-a" viewBox="0 0 320 150"></svg></div>
      <div class="panel"><h5>旋转后的 xQ（同一纵轴尺度）</h5><svg id="rt-b" viewBox="0 0 320 150"></svg></div>
    </div>
  </div>
</div>

#### Hadamard 矩阵与快速变换

任意正交矩阵都满足计算不变性，但随机稠密正交矩阵的乘法是 $$O(C^2)$$，在线用太贵了。**Hadamard 矩阵**的元素全是 ±1，归一化之后正交，而且有 $$O(C\log C)$$ 的快速算法。Sylvester 构造是：

$$
H_1=[1],\qquad H_{2n}=\begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix},\qquad \tfrac{1}{\sqrt n}H_n\ \text{是正交矩阵}
$$

QuaRot 用的是**随机化 Hadamard** $$Q=H\cdot\operatorname{diag}(\pm1)$$（随机翻转符号），在保持快速算法的同时得到接近随机正交矩阵的"打散"效果。快速 Walsh-Hadamard 变换（FWHT）是 $$\log_2 n$$ 级蝶形运算，每一级把距离为 h 的两个元素做 $$(a,b)\to(a+b,\ a-b)$$：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-14-fwht.svg" alt="8 点快速 Walsh-Hadamard 变换的三级蝶形图" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 14. 8 点 FWHT，输入就是上面例子里的 x。三级蝶形（h = 1, 2, 4），每级 8 次加减法，最后除以 √8，结果和矩阵乘 xH₈/√8 完全一致。n 维一共需要 n·log₂n 次加减法，4096 维只要约 4.9 万次，而稠密矩阵乘需要约 1677 万次乘加。
</div>

#### QuaRot：把整个残差流旋转掉

QuaRot（Ashkboos et al., NeurIPS 2024）实现了端到端的 W4A4KV4：所有线性层的权重、激活和 KV Cache 都是 4 bit。它的旋转分两类：**能离线融合进权重的**（零开销），和**必须在线执行的**（用快速 Hadamard kernel）。

**第一步，让 RMSNorm 和旋转可以交换**。把 RMSNorm 的缩放 γ 融合进后面线性层的权重（$$W\leftarrow\operatorname{diag}(\gamma)W$$），RMSNorm 就变成了纯粹的 $$x/\lVert x\rVert\cdot\sqrt d$$。由于旋转不改变范数：

$$
\operatorname{RMSNorm}(xQ)=\frac{xQ}{\lVert xQ\rVert}\sqrt d=\operatorname{RMSNorm}(x)\,Q
$$

这意味着整个残差流可以"住"在旋转后的坐标系里。LayerNorm 的模型（比如 OPT）需要先像 SliceGPT 那样把均值减法吸收进前面的线性层，转换成 RMSNorm。

**第二步，离线融合 R1（残差流旋转）**：

- Embedding：$$E\leftarrow EQ$$，残差流从一开始就是 $$XQ$$；
- 所有"读残差流"的输入投影（q/k/v、gate/up、lm*head）：$$W*{in}\leftarrow Q^\top\operatorname{diag}(\gamma)W\_{in}$$；
- 所有"写残差流"的输出投影（o*proj、down_proj）：$$W*{out}\leftarrow W\_{out}Q$$，输出也在旋转后的坐标系里，可以直接加到残差上。

这些全部在加载之前完成，推理时没有任何额外的计算，q/k/v、gate/up 的输入激活也因此不再有离群通道了。

**第三步，在线 Hadamard 处理剩下的激活**：

- **down_proj 的输入**（SiLU(g)⊙u 之后）不在残差流里，R1 管不到。QuaRot 在它前面插入一个在线 Hadamard，同时把 $$H^\top$$ 融进 $$W_{down}$$。维度不是 2 的幂时（比如 Llama-2-7B 的 11008），用小阶 Hadamard 和 2 的幂阶做 Kronecker 积；
- **o_proj 的输入**：每个 head 内部的 Hadamard $$H_{d_h}$$ 可以离线融进 $$W_v$$（输出侧）和 $$W_o$$（输入侧）；跨 head 的 $$H_{n_h}$$ 在线执行，两者合起来就是 $$H_{n_h}\otimes H_{d_h}$$；
- **KV Cache**：在 RoPE 之后给 Q 和 K 都乘同一个 $$H_{d_h}$$，因为 $$(QH)(KH)^\top=QK^\top$$，注意力分数不变，而缓存下来的 K 已经被抹平了，可以量化成 4 bit。V 已经被 $$W_v$$ 里融合的 Hadamard 旋转过了。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-15-quarot.svg" alt="QuaRot 在 Llama 解码层中的旋转位置：离线融合的旋转与在线 Hadamard" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 15. QuaRot 的旋转布置。SpinQuant 沿用了同样的位置并重新命名：R1 是残差流旋转，R2 是每个 head 的 V/O 旋转（都可以离线融合），R3 是 Q/K 上的在线 Hadamard，R4 是 down_proj 前的在线 Hadamard。
</div>

一个线性层的 W4A4 前向大概是这样的：

1. （仅 down_proj、o_proj）**在线 Hadamard**：FWHT kernel 对激活的每一行做变换；
2. **per-token 对称 INT4 量化**：每行求 absmax（QuaRot 对激活用 0.9 的截断比例），除以 scale、取整，两个 INT4 打包成一个字节，这一步通常和上一步或者 RMSNorm 融合；
3. **INT4 GEMM**：CUTLASS INT4 Tensor Core（Ampere / Ada），INT32 累加，权重是离线用 GPTQ 量化的 per-channel 对称 INT4；
4. **反量化**：乘 $$s^x_i s^w_j$$，转回 FP16；
5. **KV Cache**：K（已旋转）和 V 按每个 head 分组做非对称 INT4 量化后写进缓存，attention kernel 读取时再反量化。

论文报告 Llama-2-70B 在 W4A4KV4 下 WikiText-2 困惑度最多增加 0.47，保留了 99% 的零样本精度。权重这一侧仍然需要 GPTQ：旋转让 RTN 也能用，但 GPTQ 明显更好。

#### SpinQuant：学出来的旋转

SpinQuant（Liu et al., ICLR 2025）发现，不同的随机旋转之间精度差别很大，零样本推理任务上能差十几个点。既然 R1、R2 可以离线融合，那就可以把它们当成参数来**优化**：在网络里插入伪量化，最小化量化后网络的任务损失。约束是 R 必须保持正交，所以在 Stiefel 流形上用 **Cayley SGD** 更新：

$$
\hat G=GR^\top,\quad Y=\hat G-\hat G^\top\ (\text{反对称}),\quad R'=\Big(I-\tfrac{\eta}{2}Y\Big)^{-1}\Big(I+\tfrac{\eta}{2}Y\Big)R
$$

Cayley 变换把反对称矩阵映射成正交矩阵，所以 $$R'$$ 始终是严格正交的。只需要几百条校准样本，7B 模型上训练不到一个小时。R3、R4 仍然用在线 Hadamard。论文报告 W4A4KV4 的 Llama-2-7B 在零样本推理任务上和全精度的差距缩小到了 2.9 个点。

旋转这个家族还有几个值得知道的成员：

- **QuIP / QuIP#**：从理论上提出"非相干处理"（incoherence processing），对权重两侧做随机 Hadamard 变换，再配合 E8 格点码本，做到 2 bit 的权重量化；
- **DuQuant**：旋转 + 之字形的通道置换，把离群值更均匀地分配到各个块；
- **FlatQuant** 等：学习一般的可逆仿射变换，不限于正交矩阵；
- **ConvRot**：针对扩散 Transformer，用分组的正则 Hadamard 解决 Sylvester Hadamard 放大"行离群值"的问题，下面单独讲。

### ConvRot：DiT 上即插即用的分组旋转

ConvRot（Huang et al., arXiv 2512.03673，2025 年 12 月）把旋转量化带到了扩散 Transformer（FLUX.1 这类 DiT）上，实现了不需要重新训练的 W4A4 推理。它要解决的是把 QuaRot 直接搬到 DiT 上时遇到的两个问题。

| ConvRot    |                                                                             |
| :--------- | :-------------------------------------------------------------------------- |
| 解决的问题 | DiT 上的 W4A4：LLM 的旋转方法开销大，而且会放大 DiT 特有的行（token）离群值 |
| 核心操作   | 分组（N₀ = 256）的正则 Hadamard 旋转，用 reshape + matmul 实现              |
| 需要的数据 | 不需要校准，也不需要训练                                                    |
| 推理开销   | 每个线性层一次分组旋转，复杂度 O(K)，在 Tensor Core 上执行                  |
| 主要实现   | 论文代码（复用 QuaRot 的 kernel）                                           |

#### 为什么不能直接用 QuaRot

- **开销**。DiT 没有 LLM 那样干净的"残差流 + RMSNorm"结构（adaLN 调制、文本和图像 token 的联合注意力等），QuaRot 的离线融合策略很难迁移过来。如果每个线性层都做在线的全局旋转，复杂度是 $$O(K^2)$$（稠密）或者至少 $$O(K\log K)$$（FWHT），开销不小。
- **行离群值**。LLM 的离群值主要按通道（列）分布；DiT 的部分层还存在按 token（行）分布的离群值：某些 token 整行的激活都很大，而且同号。论文发现 Sylvester 型的 Hadamard 会**放大**这类离群值。

#### Sylvester Hadamard 为什么会放大行离群值

旋转后第 j 个输出通道是 $$(xH)_j=\frac{1}{\sqrt n}\sum_i x_iH_{ij}$$，也就是 x 和 H 第 j 列的内积。Sylvester 矩阵的**第一列全是 1**，于是：

$$
(xH)_1=\frac{1}{\sqrt n}\sum_{i=1}^n x_i
$$

如果某个 token 的整行都是大的同号值（约等于 c），这一项就是 $$\sqrt n\cdot c$$，**放大了 √n 倍**。论文用"列偏差"（column discrepancy）来刻画这一点，也就是 H 的最大列和：

- 任意 n 阶 Hadamard 矩阵都满足 $$\sum_j\big(\sum_i H_{ij}\big)^2=\mathbf 1^\top HH^\top\mathbf 1=n^2$$，所以最大列和至少是 $$\sqrt n$$；
- Sylvester 矩阵取到上界 n（第一列和为 n，其余列和为 0），对常数行的放大最严重；
- **正则（regular）Hadamard 矩阵**每行每列的和都是 $$\pm\sqrt n$$，取到下界：常数行 c 旋转之后每个通道都是 $$\pm c$$，**完全不放大**。

正则 Hadamard 可以从 4 阶出发用 Kronecker 积递归构造，阶数是 $$4^k$$（列和按 $$(\pm2^k)(\pm2)=\pm2^{k+1}$$ 相乘，保持正则）：

$$
H_4=\begin{bmatrix}1 & 1 & 1 & -1\\ 1 & 1 & -1 & 1\\ 1 & -1 & 1 & 1\\ -1 & 1 & 1 & 1\end{bmatrix},\qquad H_{4^{k+1}}=H_{4^k}\otimes H_4
$$

用一个 4 维的例子对比一下。某个 token 在一组 4 个通道上的激活是 $$x=[5.0,\ 5.2,\ 4.8,\ 5.1]$$（整行偏大而且同号），两种矩阵都除以 $$\sqrt4=2$$ 归一化：

| 矩阵           | 列和         | xH / 2                         | 最大绝对值 |
| :------------- | :----------- | :----------------------------- | :--------- |
| 原始（不旋转） | —            | [5.00, 5.20, 4.80, 5.10]       | 5.20       |
| Sylvester H₄   | [4, 0, 0, 0] | [**10.05**, −0.25, 0.15, 0.05] | 10.05      |
| 正则 H₄        | [2, 2, 2, 2] | [4.95, 5.25, 4.85, 5.05]       | 5.25       |

以正则 H₄ 的第一列 [1, 1, 1, −1] 为例：(5.0 + 5.2 + 4.8 − 5.1)/2 = 4.95。Sylvester 把能量全部堆到了第一个通道上，per-token 量化的 scale 翻倍，其余通道的分辨率随之减半。组大小为 16 时，同样的现象会放大 4 倍：一行约 3.0 的值经过 Sylvester H₁₆ 之后最大值是 11.65，经过正则 H₁₆ 之后是 3.45。论文里 FLUX.1-dev 的真实层 `single_transformer_blocks.37.proj_out`：原始激活最大值是 14.48，标准 Hadamard 旋转后变成了 106.19，分组正则 Hadamard 旋转后是 9.26。

下面的实验同时放了行离群（token 3 整行偏移）和列离群（通道 5），可以分别调大，看两种 Hadamard 的区别。默认参数下 Sylvester 把最大值从 7.28 放大到 17.11，正则 Hadamard 是 5.96；token 3 的 INT4 相对误差在 Sylvester 下是 2.49%，正则下只有 0.34%：

<div class="qz-lab qz-card">
  <div class="box-h"><span class="tag">交互实验</span>Sylvester vs 正则 Hadamard：行离群与列离群</div>
  <div class="box-b">
    <div class="ctrl">
      <label for="cv-row">行离群强度（token 3 整行偏移） <input type="range" id="cv-row" min="0" max="8" step="0.25" value="4"><output id="cv-row-v">4.00</output></label>
      <label for="cv-col">列离群强度（通道 5） <input type="range" id="cv-col" min="0" max="20" step="0.5" value="6"><output id="cv-col-v">6.0</output></label>
      <label for="cv-n">组大小 N<sub>0</sub> <select id="cv-n"><option value="4">4</option><option value="16" selected>16</option><option value="64">64</option></select></label>
    </div>
    <div class="stats" id="cv-stats"></div>
    <div class="grid2">
      <div class="panel"><h5>原始 |X|（8 个 token × 64 个通道）</h5><svg id="cv-0" viewBox="0 0 330 120"></svg></div>
      <div class="panel"><h5>Sylvester Hadamard 分组旋转后</h5><svg id="cv-1" viewBox="0 0 330 120"></svg></div>
      <div class="panel"><h5>正则 Hadamard 分组旋转后</h5><svg id="cv-2" viewBox="0 0 330 120"></svg></div>
      <div class="panel"><h5>每个 token 的 INT4 per-token 量化误差（相对 MSE）</h5><svg id="cv-3" viewBox="0 0 330 120"></svg></div>
    </div>
    <p class="small mute">三张热力图使用同一色标。把“行离群强度”调大：Sylvester 在每组的第一个通道上出现一条亮竖线（√N<sub>0</sub> 倍放大），token 3 的量化误差飙升；正则 Hadamard 不放大。只调“列离群强度”时，两者都能把离群列打散。右下的柱状图里，灰色是原始激活，橙色是 Sylvester，绿色是正则 Hadamard。</p>
  </div>
</div>

#### 分组旋转：从 O(K²) 到 O(K)

ConvRot 不做全局旋转，而是把输入维 K 切成大小为 $$N_0$$ 的组（默认 256），每组独立乘同一个 $$N_0$$ 阶的正则 Hadamard。设 $$X\in\mathbb{R}^{M\times K}$$，$$W\in\mathbb{R}^{N\times K}$$，按列切块 $$X=[X_1,\dots,X_G]$$、$$W=[W_1,\dots,W_G]$$：

$$
Y=XW^\top=\sum_{i=1}^{G}X_iW_i^\top=\sum_{i=1}^{G}\operatorname{RHT}(X_i)\,\operatorname{RHT}(W_i)^\top,\qquad \operatorname{RHT}(A)=A\,\tfrac{1}{\sqrt{N_0}}H_{N_0}
$$

这等价于用块对角矩阵 $$\operatorname{diag}(H_{N_0},\dots,H_{N_0})$$ 做旋转，它仍然是正交的，所以计算不变性成立。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-16-convrot-group.svg" alt="ConvRot 分组旋转等价于块对角正交矩阵，用 reshape 加矩阵乘实现" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 16. 分组旋转。不用蝶形 FWHT（跑在 CUDA core 上，还需要单独的归一化步骤），而是把张量 reshape 之后和 H 做一次矩阵乘，直接利用 Tensor Core。
</div>

| 旋转方式            | 复杂度（每个 token） | 说明                      |
| :------------------ | :------------------- | :------------------------ |
| 全局稠密旋转        | O(K²)                | 任意正交阵                |
| 全局 FWHT           | O(K log K)           | QuaRot 的在线 Hadamard    |
| 分组旋转（N₀ 固定） | O(K·N₀) = O(K)       | ConvRot，reshape + matmul |

论文在 A100 上测了一下：N₀ = 64 时 RHT 的 GPU 占用率只有 13%，延迟反而偏高；N₀ = 256 时延迟最低（0.209 ms）；N₀ = 1024 时占用率 100%，略慢于 FWHT。默认 N₀ = 256，离群值特别大的层用 1024。

#### ConvLinear4bit：一个即插即用的 W4A4 线性层

1. **离线**：权重按组做 RHT，再做 per-output-channel 对称 INT4 量化并打包；
2. **在线 RHT**：激活 reshape 之后乘 $$H_{N_0}$$（Tensor Core）；
3. **在线量化**：per-token 对称 INT4；
4. **INT4 GEMM**：INT4 Tensor Core，INT32 累加；
5. **反量化**：乘 $$s^x_i s^w_j$$ 回到 BF16 / FP16，量化、GEMM、反量化的 kernel 复用 QuaRot 的实现。

```python
import torch

H4 = torch.tensor([[ 1,  1,  1, -1],
                   [ 1,  1, -1,  1],
                   [ 1, -1,  1,  1],
                   [-1,  1,  1,  1]], dtype=torch.float32)

def regular_hadamard(n):                      # n 必须是 4 的幂
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.kron(H, H4)                 # 列和保持 ±sqrt(n)
    return H / n ** 0.5                       # 归一化为正交矩阵

def group_rht(x, H):                          # x: [..., K]，K 为 N0 的整数倍
    n0, shp = H.shape[0], x.shape
    return (x.reshape(-1, shp[-1] // n0, n0) @ H).reshape(shp)

class ConvLinear4bit(torch.nn.Module):
    def __init__(self, linear, n0=256):
        super().__init__()
        self.register_buffer("H", regular_hadamard(n0).to(linear.weight))
        w = group_rht(linear.weight.data.float(), self.H)        # 离线旋转权重
        self.w_q, self.s_w = quant_int4_per_channel(w)            # 离线 INT4（打包）
        self.bias = linear.bias
    def forward(self, x):
        x = group_rht(x, self.H)                                  # 在线分组旋转
        x_q, s_x = quant_int4_per_token(x)                        # 在线 per-token INT4
        y = int4_gemm(x_q, self.w_q)                              # INT32 累加
        y = (y * s_x * self.s_w.view(1, -1)).to(x.dtype)          # 反量化
        return y if self.bias is None else y + self.bias
# quant_int4_* 与 int4_gemm 对应 QuaRot 提供的 CUDA kernel
```

它可以直接替换 `nn.Linear`，不需要重新训练，也不需要专门的推理引擎。为了在质量和速度之间取舍，论文给了两种配置：一种是除了最后 3 个敏感层（比如 `single_transformer_blocks.37.proj_out`）保持高精度，其余 Transformer 线性层全部 W4A4；另一种是约 20% 的层（各 block 的 `attn.to_out.0`、单流 block 的 `attn.to_v` 等）用 W8A8，其余 W4A4。

| FLUX.1-dev（MJHQ，论文数据） | FID ↓ | ImageReward ↑ | LPIPS ↓ | PSNR ↑ | 显存 (GiB) | 延迟（s，50 步，RTX 4090） |
| :--------------------------- | :---- | :------------ | :------ | :----- | :--------- | :------------------------- |
| BF16                         | 10.07 | 0.99          | —       | —      | 22.7       | 54.6                       |
| NF4（W4A16）                 | 12.10 | 0.96          | 0.23    | 19.23  | 6.9        | 38.6                       |
| SVDQuant（W4A4）             | 10.01 | 0.97          | 0.18    | 20.49  | 6.5        | 14.9                       |
| ConvRot W4A4                 | 12.32 | 0.84          | 0.22    | 19.43  | 5.6        | 23.2                       |
| ConvRot 混合精度             | 10.03 | 0.97          | 0.18    | 20.73  | 7.0        | 28.3                       |

论文的标题级结论是相对 BF16 约 2.26× 加速、4.05× 显存节省。消融实验里，QuaRot 式的全局标准 Hadamard 在 W4A4 下直接崩溃（FID 98.04），256 分组的标准 Hadamard 是 12.55，1024 分组的正则 Hadamard 是 12.30。SVDQuant 依赖 Nunchaku 这样深度融合的专用引擎，速度更快；ConvRot 的卖点是即插即用。

### SVDQuant：用 16 bit 低秩支路吸收离群值

SVDQuant（Li et al., ICLR 2025，MIT Han Lab）面向 FLUX.1、PixArt-Σ、SDXL 这些扩散模型的 W4A4。它把 SmoothQuant 的平滑和一个新想法结合了起来：平滑之后，离群值从激活转移到了权重上，再用一条 16 bit 的低秩支路把权重里的离群部分"吸收"掉。

| SVDQuant   |                                                                                               |
| :--------- | :-------------------------------------------------------------------------------------------- |
| 解决的问题 | 扩散模型 W4A4 中，激活和权重都有离群值，4 bit 两边都放不下                                    |
| 核心操作   | 先平滑，把激活的离群值全部迁到权重；再用 16 bit 低秩支路吸收权重的离群部分，残差做 4 bit 量化 |
| 需要的数据 | 校准样本上每个输入通道的激活幅值；每个权重矩阵做一次 SVD                                      |
| 推理开销   | 一条秩 32 的 16 bit 支路，需要 kernel 融合才能把开销压下来                                    |
| 主要实现   | DeepCompressor（离线量化）+ Nunchaku（推理引擎）                                              |

#### 为什么平滑在 4 bit 下不够

SmoothQuant 在 W8A8 下有效，是因为激活和权重各有 256 个码字，难度分到两边都放得下。到了 W4A4，两边都只有 16 个码字：α 偏向激活，权重的输入通道就被放大得很厉害；偏向权重，激活又放不下。QuaRot、ConvRot 用正交旋转同时抹平两边，SVDQuant 换了一个思路：干脆让激活彻底变平（离群值全部迁给权重），再专门处理变得"难看"的权重。

#### 低秩分解：把离群部分留在 16 bit

先像 SmoothQuant 一样做 $$\hat X=X\operatorname{diag}(\lambda)^{-1}$$、$$\hat W=W\operatorname{diag}(\lambda)$$（W 是 [out, in]，λ 乘在输入列上）。平滑后的权重里，被放大的少数输入通道表现为几列特别大的值，这种结构的能量集中在少数几个奇异方向上。对 $$\hat W$$ 做 SVD，取前 r 个奇异方向（r = 32）作为低秩部分 $$L_1L_2$$，剩下的残差 $$R=\hat W-L_1L_2$$ 幅值小、离群值少，再做 4 bit 量化：

$$
XW^\top=\hat X\hat W^\top=\underbrace{\hat X\,(L_1L_2)^\top}_{\text{16 bit 低秩支路}}+\underbrace{\hat X R^\top}_{\approx\,Q(\hat X)\,Q(R)^\top\text{，W4A4 主支路}}
$$

误差只来自第二项。记 $$A=\hat X$$，$$B=R^\top$$，量化后是 $$a=Q(A)$$、$$b=Q(B)$$，由恒等式 $$AB-ab=(A-a)B+A(B-b)-(A-a)(B-b)$$ 加上三角不等式得到：

$$
\begin{aligned}
\lVert \hat XR^\top-Q(\hat X)Q(R)^\top\rVert\le{} & \lVert \hat X-Q(\hat X)\rVert\,\lVert R\rVert+\lVert \hat X\rVert\,\lVert R-Q(R)\rVert\\
&+\lVert \hat X-Q(\hat X)\rVert\,\lVert R-Q(R)\rVert
\end{aligned}
$$

右边每一项都是"一侧的量化误差 × 另一侧的大小"。平滑让 $$\hat X$$ 没有离群值，量化误差小；低秩分解让 R 没有离群值，量化误差和幅值都小。两边同时变好，上界就小了。r 相对隐藏维度（FLUX 是 3072）很小，低秩支路的计算量大约是主支路的 r/d ≈ 1%。

看一个秩 1 的例子。平滑后的权重（4 个输出 × 6 个输入），第 3 个输入通道被放大了：

$$
\hat W=\begin{bmatrix}0.00 & 0.09 & 3.10 & -0.27 & -0.14 & -0.30\\ 0.02 & 0.40 & -2.60 & -0.19 & 0.15 & 0.11\\ 0.03 & -0.28 & 2.90 & 0.21 & -0.40 & -0.14\\ -0.57 & -0.39 & -3.30 & -0.07 & -0.38 & 0.08\end{bmatrix}
$$

奇异值是 [5.99, 0.93, 0.48, 0.19]，第一个奇异方向几乎就是那一列。取秩 1 的部分 $$L_1L_2=\sigma_1u_1v_1^\top$$ 之后，残差每一行的最大值从 [3.1, 2.6, 2.9, 3.3] 降到了 [0.31, 0.38, 0.34, 0.45]。per-row 对称 INT4 的重建误差（Frobenius 范数）：直接量化 $$\hat W$$ 是 0.513，量化 R 再加回 16 bit 的 $$L_1L_2$$ 是 **0.075**，降低了约 7 倍。代价是多了一个 6×1 和一个 1×4 的小矩阵乘。

#### 推理：低秩支路的 kernel 融合

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/quantization/fig-17-svdquant.svg" alt="SVDQuant 的双支路结构，以及 Nunchaku 把低秩下投影与量化融合、上投影与 4 bit GEMM 融合" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 17. SVDQuant 与 Nunchaku 推理引擎。低秩支路的 FLOPs 很少，但如果单独调用 kernel，会多读写一遍激活，延迟开销很明显。Nunchaku 把下投影和激活量化融合（两者读同一个输入），把上投影和 4 bit GEMM 融合（两者写同一个输出）。
</div>

离线准备和前向的示意代码：

```python
import torch

@torch.no_grad()
def svdquant_prepare(W, act_absmax, rank=32, alpha=0.5):
    """W: [out, in]；act_absmax: 校准得到的每个输入通道 max|X_j|"""
    # 1) 平滑：激活除以 lam，权重的输入列乘以 lam（同 SmoothQuant）
    lam = (act_absmax.pow(alpha) / W.abs().amax(0).pow(1 - alpha)).clamp(min=1e-5)
    W_hat = W.float() * lam.view(1, -1)
    # 2) 低秩分解：前 rank 个奇异方向留在 16 bit
    U, S, Vh = torch.linalg.svd(W_hat, full_matrices=False)
    L_up = (U[:, :rank] * S[:rank]).half()            # [out, r]
    L_down = Vh[:rank, :].half()                       # [r, in]
    R = W_hat - L_up.float() @ L_down.float()          # 残差：幅值小、没有离群列
    R_q, s_R = quant_int4_per_group(R, group=64)       # 4 bit 主支路，离线完成
    return lam, L_down, L_up, R_q, s_R

def svdquant_forward(x, lam, L_down, L_up, R_q, s_R):
    x_hat = x / lam                                    # 实际实现中融合进前一个算子
    x_q, s_x = quant_int4_per_group(x_hat, group=64)   # Nunchaku：与下投影在同一 kernel
    low = (x_hat @ L_down.T) @ L_up.T                  # 16 bit 低秩支路
    main = int4_gemm(x_q, s_x, R_q, s_R)               # W4A4 主支路；上投影在同一 kernel 中累加
    return main + low
# quant_int4_per_group 与 int4_gemm 对应 Nunchaku 中融合后的 CUDA kernel
```

SVDQuant 论文报告：12B 的 FLUX.1 显存降低约 3.5 倍，在 16GB 的笔记本 4090 上相对 4 bit weight-only（NF4）基线约 3 倍加速，图像质量和 BF16 接近。上面 ConvRot 论文的对比表里，SVDQuant 在 FLUX.1-dev 上 FID 10.01（BF16 是 10.07），50 步延迟 14.9 s（BF16 是 54.6 s），是表里最快的方案。和 ConvRot 相比，它精度和速度都更好，但依赖校准和专用引擎；ConvRot 胜在即插即用。

### 放在一起看

把这几个算法放到一张表里：

| 方法        | 典型配置                | 核心操作                                 | 需要的统计 / 优化           | 推理时额外开销          | 一句话                                |
| :---------- | :---------------------- | :--------------------------------------- | :-------------------------- | :---------------------- | :------------------------------------ |
| RTN         | 任意                    | 直接四舍五入                             | 无                          | 无                      | 基线；INT8 权重够用，4 bit 以下误差大 |
| LLM.int8()  | W8A8 + FP16 离群列      | 混合精度分解                             | 运行时阈值检测              | 大（两次 GEMM、拆分）   | 几乎无损但慢，揭示了离群值问题        |
| GPTQ        | W4A16 / W3A16 g128      | 逐列量化 + H⁻¹ 误差补偿                  | H = 2XXᵀ，Cholesky          | 无（标准分组 INT4）     | 二阶信息驱动的"边量化边修正"          |
| AWQ         | W4A16 g128              | 按激活幅值的通道缩放 + 截断              | 激活平均幅值，网格搜索 α    | 无（缩放已融合）        | 保护激活大的通道，参数极少、很稳健    |
| SmoothQuant | W8A8                    | diag(s) 把激活离群值迁移到权重           | 激活与权重的逐通道最大值，α | 无（融合进 LN）         | 让激活也能 INT8                       |
| QuaRot      | W4A4KV4                 | 随机 Hadamard 旋转，离线融合 + 在线 FWHT | 无需优化（权重用 GPTQ）     | 小（少量在线 Hadamard） | 把离群值"抹平"，端到端 4 bit          |
| SpinQuant   | W4A4KV4                 | 学习 R1、R2 正交旋转                     | Cayley SGD，数百条样本      | 同 QuaRot               | 旋转也可以学，比随机的更好            |
| ConvRot     | DiT W4A4（可混合 W8A8） | 分组正则 Hadamard（N₀ = 256）            | 无需校准与训练              | 小（reshape + matmul）  | 解决行离群值放大，即插即用            |
| SVDQuant    | DiT W4A4 / NVFP4        | 平滑 + 16 bit 低秩支路 + 4 bit 残差      | SVD、平滑因子               | 小（Nunchaku 融合后）   | 低秩支路吸收离群值                    |
| FP8 动态    | W8A8 FP8                | 格式自带动态范围                         | 几乎不需要                  | 无                      | Hopper 上最省心                       |
| NVFP4       | W4A4 FP4                | 16 元素 block + E4M3 scale               | 校准，可配合旋转与 QAT      | 无（硬件原生）          | Blackwell 的 4 bit 方案               |

按时间顺序读下来，会发现几乎所有方法都在做同一件事：在 $$XW^\top$$ 中间插入一对互逆的变换 $$A^{-1}A$$，或者调整舍入的方式，让量化误差在**输出**上最小：

- A 是**对角阵**：SmoothQuant（让激活平坦）、AWQ（保护重要权重）；
- A 是**正交阵**：QuaRot（随机 Hadamard）、SpinQuant（学出来的）、ConvRot（分组正则 Hadamard）；
- 把 W **拆开**：LLM.int8()（按列拆出 FP16 部分）、SVDQuant（拆出低秩部分）；
- 调整**舍入**：GPTQ（用 H⁻¹ 补偿）、AdaRound / BRECQ（学习舍入方向）；
- 换**格式**：FP8、NF4、NVFP4 / MXFP4。

这些手段作用在不同的地方，经常叠加使用。原则是：变换类的方法（缩放、旋转、低秩）先改变数据分布，GPTQ 最后负责权重的舍入。

| 组合                      | 目标配置 | 分工                                                              |
| :------------------------ | :------- | :---------------------------------------------------------------- |
| SmoothQuant + GPTQ        | W8A8     | SmoothQuant 把激活的离群值迁到权重；GPTQ 再量化这些被放大过的权重 |
| AWQ 缩放 + GPTQ           | W4A16    | AWQ 先做通道缩放保护重要权重；GPTQ 再做误差补偿式的舍入           |
| QuaRot / SpinQuant + GPTQ | W4A4KV4  | 旋转抹平激活、权重和 KV Cache 的离群值；GPTQ 量化旋转后的权重     |
| SVDQuant                  | DiT W4A4 | 本身就是"平滑 + 低秩"两种手段的组合                               |
| 旋转 + NVFP4 / MXFP4      | FP4 W4A4 | 旋转消除块内的离群值，块缩放格式处理剩下的局部范围差异            |

按目标配置来选：

- **W4A16**（显存或 decode 速度优先）：AWQ 或 GPTQ，两者精度接近；对校准数据的分布不放心时优先 AWQ，追求 3 bit 时用 GPTQ（act-order）或者两者组合；
- **W8A8 INT8**（Ampere 上的高吞吐）：SmoothQuant + GPTQ；
- **W8A8 FP8**（Ada / Hopper / Blackwell）：通常直接做 FP8 动态量化就够了，FP8 自带的动态范围让它对离群值不敏感；
- **LLM 的 W4A4**：QuaRot / SpinQuant + GPTQ，需要 INT4 Tensor Core（Ampere / Ada）；Blackwell 上考虑 NVFP4；
- **DiT 的 W4A4**：追求极致的速度和质量用 SVDQuant（Nunchaku）；希望不校准、即插即用就用 ConvRot。

## 扩散模型量化有什么不同

文生图、文生视频的模型（SD / SDXL 的 UNet，FLUX、SD3、HunyuanVideo、Wan 这些 DiT）同样能用前面的方法量化，但有几个 LLM 没有的特点，直接决定了方法的选择：

| 特点       | LLM（decode）             | 扩散模型                                                                      | 对量化的影响                                                                   |
| :--------- | :------------------------ | :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| 计算特征   | 每步 1 个 token，带宽受限 | 每步几千到几万个图像/视频 token，跑 20～50 步，算力受限                       | weight-only 只省显存不提速；要加速必须 W8A8 / W4A4 / FP8 / FP4，以及注意力量化 |
| 激活分布   | 相对稳定                  | 随去噪时间步 t 剧烈变化                                                       | 校准数据要覆盖所有时间步；有的方法使用随时间步变化的量化参数                   |
| 误差传播   | 逐 token 累积             | 沿去噪轨迹逐步累积                                                            | 单步误差小也可能导致最终图像的构图、纹理偏移                                   |
| 结构特殊性 | 残差流 + RMSNorm 很规整   | UNet 跳连拼接、adaLN 调制、文本与图像 token 联合注意力                        | QuaRot 式的离线融合很难直接迁移                                                |
| 离群值形态 | 以按通道为主              | 按通道 + 按 token（行）                                                       | 需要能同时处理两种离群值的变换                                                 |
| 评价指标   | 困惑度、下游任务          | FID、CLIP Score、ImageReward；与 FP16 同 seed 同 prompt 输出比较 LPIPS / PSNR | FID 好不代表和原模型一致，两类指标都要看                                       |

这些年扩散模型的量化大概经历了这几个阶段：

- **UNet 时代：Q-Diffusion**。一是**时间步感知的校准**，在整个去噪过程中均匀采样时间步来收集校准激活，而不是只用某一步，否则 scale 只适配部分时间步；二是**跳连拆分量化**，UNet 的解码器把深层特征和浅层跳连特征拼接之后送进卷积，两部分的数值范围差别很大，拼接后呈双峰分布，拆成两部分分别量化，再各自和对应的权重切片相乘。在此基础上用 BRECQ / AdaRound 式的块重建优化权重的舍入，实现 W4A8。
- **DiT 的 W8A8 / W4A8：PTQ4DiT、ViDiT-Q**。PTQ4DiT（NeurIPS 2024）观察到 DiT 里激活和权重都存在"显著通道"，而且激活的显著通道随时间步变化，于是提出了通道显著性平衡（类似 SmoothQuant 的对角缩放，但同时考虑激活和权重的显著性）和基于 Spearman 相关系数的跨时间步校准。ViDiT-Q 面向视频 DiT，用细粒度的动态激活量化，加上静态–动态结合的通道平衡，处理视频模型里按 token 和按时间步变化的分布。
- **注意力量化**。视频 DiT 的注意力序列极长，经常是瓶颈。SageAttention 系列把 $$QK^\top$$ 用 INT8、$$PV$$ 用 FP8 来算（并对 K 做均值平滑），属于激活–激活矩阵乘的量化，和线性层的量化是正交的，可以叠加。
- **DiT 的 W4A4：ConvRot 与 SVDQuant**。扩散模型受算力限制，W4A4 才能带来大幅加速，前面已经讲过两者的区别。

落地时有几条我觉得比较实用的经验：

- **哪些部分先保留高精度**：输入的 patch embedding、时间步和文本条件的嵌入 MLP、adaLN 调制的线性层、最后的输出层（比如 FLUX 的 `proj_out`、`norm_out`）通常对量化很敏感，先保持 BF16，再逐层尝试量化。ConvRot 的消融也显示少数最后的层需要保留高精度。
- **周边模块单独处理**：T5 / CLIP 文本编码器只跑一次，做 weight-only（FP8 / INT8 / NF4）主要是为了省显存；VAE 解码器对颜色和纹理敏感，一般保留 BF16。
- **校准**：用一批覆盖不同主题和风格的 prompt，在所有时间步上采样激活，CFG 的条件和无条件分支都要覆盖到。
- **先简单后复杂**：Hopper / Ada 上 FP8 W8A8（动态 per-token）通常几乎无损，是第一选择；需要更低的显存或者更高的速度时，再上 SVDQuant（Nunchaku）、ConvRot 这类 W4A4，Blackwell 上考虑 NVFP4。另外量化和[特征缓存](https://valdrada.cn/AI/2026-10-22-diffusion-feature-cache.html)的误差会相互影响，叠加使用时缓存的阈值要在量化后的模型上重新调。

## KV Cache 与 FP8 / FP4

前面讲的都是线性层的权重和激活。落地的时候还有两件事绕不开：LLM 的 KV Cache 要不要量化、怎么量化；以及 FP8、FP4 这些新格式和前面的整数量化相比到底好在哪里。

### KV Cache 量化

前面算过，长上下文或者大 batch 时 KV Cache 会超过权重本身。K 和 V 的统计特性不同，量化方式也不一样：

- **Key 有固定的离群通道**（和激活类似），所以应该 **per-channel** 量化，也就是沿 token 方向分组、每个通道一个 scale。KIVI 的做法是 K 按通道、V 按 token 量化到 2 bit，并把最近的一小段 token 保留为 FP16 的残差窗口（凑满一组之后再量化）。
- **RoPE 的影响**：RoPE 按位置旋转 K 的通道对，会打乱通道结构。KVQuant 选择在 RoPE **之前**量化 K；QuaRot 则在 RoPE 之后乘 Hadamard 把 K 抹平。
- **FP8 KV Cache**：最简单、最常用的方案。vLLM、TensorRT-LLM、SGLang 都支持（比如 vLLM 的 `--kv-cache-dtype fp8`），显存减半，精度损失通常很小。

### FP8：动态范围带来的鲁棒性

FP8 E4M3 的相对精度在整个范围内近似恒定：3 位尾数意味着在每个 2 的幂区间内，间隔是区间起点的 1/8，就近舍入的最大相对误差约 6%。INT8 则是绝对间隔恒定，小值处的相对误差很大。所以有离群值的时候，FP8 对小值友好得多。拿 SmoothQuant 例子里的一行激活 $$x=[0.6,\ -45.0,\ 1.1,\ -0.8]$$ 做 per-tensor 量化：

| 格式     | scale           | x / scale                  | 编码后            | 反量化                      | 相对误差                |
| :------- | :-------------- | :------------------------- | :---------------- | :-------------------------- | :---------------------- |
| INT8     | 45/127 = 0.3543 | [1.69, −127, 3.10, −2.26]  | [2, −127, 3, −2]  | [0.709, −45, 1.063, −0.709] | [18.1%, 0, 3.4%, 11.4%] |
| FP8 E4M3 | 45/448 = 0.1004 | [5.97, −448, 10.95, −7.96] | [6, −448, 11, −8] | [0.603, −45, 1.105, −0.804] | [0.4%, 0, 0.4%, 0.4%]   |

E4M3 在 [4, 8) 区间的间隔是 0.5，在 [8, 16) 是 1，所以 5.97 → 6、10.95 → 11，相对误差都很小。代价是大值附近的间隔很粗（[256, 448] 的间隔是 32）。

因此在 Hopper / Ada / Blackwell 上，**FP8 W8A8**（权重 per-channel 或 per-tensor，激活 per-token 动态）往往不需要 SmoothQuant 这类预处理就几乎无损，是生产环境里最省心的方案。这也回答了开头的那个问题：Hopper 上大家不怎么折腾 SmoothQuant，是因为 FP8 的格式本身就把离群值的问题消化掉了大半。一些工具还提供不需要校准数据的"FP8 动态"方案：权重离线转换，激活在运行时求 scale。

### FP4：Blackwell 时代的 W4A4

NVFP4 / MXFP4 通过 16 或 32 个元素的小 block 共享 scale，在局部范围内处理离群值，硬件直接支持 block scale 的乘法，没有额外的反量化开销。但 4 bit 依然只有 16 个码字，激活里的强离群值仍然会明显影响精度，所以常常和旋转、平滑或者 QAT 结合使用。实际的例子有：OpenAI 的 gpt-oss 以 MXFP4 格式发布 MoE 层的权重；NVIDIA 的 TensorRT Model Optimizer、vLLM 的 llm-compressor 都提供了 NVFP4 的 PTQ 流程。

## 工程上要注意的几件事

论文里比的是困惑度、FID 和 kernel 的加速比，真正落到推理系统里，还要考虑用什么工具产出量化模型、按硬件和场景怎么选、以及怎么确认量化真的生效了。

### 用什么工具

| 工具                                        | 支持的方法                                                         | 输出 / 推理后端                        |
| :------------------------------------------ | :----------------------------------------------------------------- | :------------------------------------- |
| llm-compressor（vLLM 项目）                 | GPTQ、AWQ、SmoothQuant、FP8、INT8、NVFP4、W4A16 等，用 recipe 组合 | compressed-tensors 格式，vLLM 直接加载 |
| NVIDIA TensorRT Model Optimizer             | FP8、INT8 SmoothQuant、INT4 AWQ、NVFP4，以及 QAT                   | TensorRT-LLM、vLLM、SGLang             |
| GPTQModel（AutoGPTQ 的后继）                | GPTQ 及其变体                                                      | vLLM、SGLang、Transformers             |
| AutoAWQ                                     | AWQ（已停止维护，功能并入 llm-compressor）                         | vLLM、Transformers                     |
| bitsandbytes                                | NF4、LLM.int8()                                                    | Transformers 加载即量化，常用于 QLoRA  |
| llama.cpp（GGUF）                           | k-quants（Q4_K_M、Q5_K_M 等）、imatrix 重要性校准                  | CPU、Apple Silicon、消费级 GPU         |
| Nunchaku                                    | SVDQuant（W4A4 / NVFP4）                                           | FLUX、SANA 等扩散模型，ComfyUI 插件    |
| diffusers + torchao / bitsandbytes / quanto | weight-only INT8 / INT4 / NF4、FP8                                 | diffusers 管线                         |

各个工具更细的对比可以看之前那篇[量化工具调研](https://valdrada.cn/AI/2026-04-15-quantization_tools_research.html)。拿 llm-compressor 举个例子，SmoothQuant + GPTQ 做 W8A8 大概是这样（API 以 2025 年的版本为准，不同版本的导入路径和参数名可能有变化）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(model_id)

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),                        # SmoothQuant 的 α = 0.8
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),  # GPTQ 量化权重
]
oneshot(model=model, dataset="open_platypus", recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)

model.save_pretrained("Llama-3-8B-W8A8", save_compressed=True)
tok.save_pretrained("Llama-3-8B-W8A8")

# 其他常见 recipe：
#   W4A16（GPTQ）：GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
#   W4A16（AWQ） ：AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])
#   FP8 动态    ：QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
#                 （不需要校准数据：oneshot(model=model, recipe=recipe)）
```

```bash
# 推理：vLLM 会从 config 中识别量化格式
vllm serve ./Llama-3-8B-W8A8
# 不做离线量化，直接在线 FP8（权重加载时转换，激活动态量化），并量化 KV Cache
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --quantization fp8 --kv-cache-dtype fp8
```

### 怎么选

| 场景                            | 推荐                                                     | 理由                                                              |
| :------------------------------ | :------------------------------------------------------- | :---------------------------------------------------------------- |
| 单卡 / 端侧部署，低并发聊天     | W4A16 g128（AWQ 或 GPTQ），Marlin kernel                 | decode 受带宽限制，权重越小越快；精度损失小                       |
| Hopper / Ada 上的高并发在线服务 | FP8 W8A8（动态 per-token）+ FP8 KV Cache                 | prefill 和大 batch 受算力限制；FP8 几乎无损，不需要复杂校准       |
| Ampere（A100 / A10）高吞吐      | INT8 W8A8：SmoothQuant + GPTQ                            | Ampere 没有 FP8 Tensor Core                                       |
| Blackwell                       | NVFP4（必要时配合旋转或 QAT）                            | 原生 FP4 Tensor Core 和 block scale                               |
| 追求极限吞吐的 W4A4             | QuaRot / SpinQuant + GPTQ                                | INT4 Tensor Core 只在 Ampere / Ada 上有；需要在线 Hadamard kernel |
| 消费级 GPU 跑 FLUX 等 DiT       | SVDQuant（Nunchaku）W4A4；或 ConvRot；保守选择 FP8 / NF4 | 扩散模型受算力限制，W4A4 才有大幅加速                             |
| CPU / Mac 本地运行              | GGUF Q4_K_M / Q5_K_M（配合 imatrix）                     | llama.cpp 生态，CPU 和 Metal kernel 成熟                          |
| 2～3 bit 极限压缩               | QuIP#、AQLM 等向量量化，或者 QAT                         | 标量 PTQ 在 3 bit 以下精度下降明显                                |

有时候量化首先是"能不能跑"的问题，然后才是"跑多快"。比如在 RTX 4090（24 GB）上跑 Llama-2-13B，FP16 权重约 26 GB，根本放不下；W4A16 约 7.3 GB（含 FP16 的 embedding / lm_head），batch=1 时 decode 上限约 1.0 TB/s ÷ 7.3 GB ≈ 137 tok/s。

### 怎么评估

- **LLM**：WikiText-2 / C4 的困惑度（敏感但不完整）；MMLU、GSM8K、HumanEval 等下游任务；长上下文任务（KV 量化尤其要测）；和原模型逐 token 对比的一致率。
- **扩散模型**：同 prompt、同 seed 下和 BF16 输出比较 LPIPS / PSNR（衡量"像不像原模型"）；FID、CLIP Score、ImageReward（衡量质量和对齐）；再人工抽查文字渲染、手部、细纹理这些难点。线上业务通常希望"量化前后用户看到的结果尽量一致"，所以和特征缓存一样，我觉得保真度类的指标更重要一些。
- **性能**：分别测 prefill（TTFT）和 decode（TPOT）的延迟、不同 batch 下的吞吐、峰值显存。还要确认真正调用了低比特的 kernel，而不是走了伪量化的路径，这一点很容易被忽略：模型能加载、结果也对，但速度没有任何变化。

## 小结

写到这里，回头看这些方法，其实都在围绕两个问题打转。

第一个问题是**省什么**。decode 受带宽限制，只量化权重就有效，所以 W4A16 的 GPTQ、AWQ 在 LLM 部署里这么流行；prefill 和扩散模型受算力限制，必须连激活一起量化，用上更低比特的 Tensor Core 才能加速，这就是为什么 AIGC 场景绕不开 W8A8 和 W4A4。

第二个问题是**离群值放在哪**。权重量化的难点在舍入，GPTQ 用二阶信息把误差摊给还没量化的权重，AWQ 用缩放保护和大激活相乘的那些权重。激活量化的难点在离群值，而离群值恰好沿着矩阵乘里不能单独给 scale 的输入通道分布。SmoothQuant 用对角缩放把难度搬到权重上，8 bit 下够用；到了 4 bit 两边都放不下，QuaRot 用正交旋转把离群值抹平，SpinQuant 把旋转学出来，ConvRot 换成正则 Hadamard 处理 DiT 的行离群，SVDQuant 则把离群部分剥离到一条 16 bit 的低秩支路上。FP8、NVFP4 这些格式又从另一个方向解决问题：让数值格式本身带上动态范围，或者把分组 scale 直接做进硬件。

所以它们大多不是互相排斥的，而是可以组合的：变换类的方法先改变数据分布，GPTQ 最后负责舍入，格式负责剩下的局部范围。以后再看到新的量化方法，我会先问三件事：它插入的是什么样的变换（对角、正交、低秩，还是换格式），误差最后落在了哪一侧，推理时多出来的在线操作能不能被 kernel 融合掉。

至于选哪个，论文里的精度对比只是起点。目标硬件有没有对应的 Tensor Core、推理框架有没有成熟的 kernel、要不要校准数据、和缓存这些其他加速手段叠加之后误差会不会放大，往往比论文里零点几的困惑度差距更能决定最后用哪一种。

## 参考

1. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
2. [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning（OBQ）](https://arxiv.org/abs/2208.11580)
3. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
4. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
5. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
6. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
7. [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
8. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
9. [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers](https://arxiv.org/abs/2512.03673)
10. [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007)
11. [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304)
12. [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005)
13. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
14. [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)
15. [QLoRA: Efficient Finetuning of Quantized LLMs（NF4）](https://arxiv.org/abs/2305.14314)
16. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
17. [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743)
18. [Microscaling Data Formats for Deep Learning（MX 格式）](https://arxiv.org/abs/2310.10537)
19. [DeepSeek-V3 Technical Report（细粒度 FP8 训练）](https://arxiv.org/abs/2412.19437)
20. [llm-compressor](https://github.com/vllm-project/llm-compressor)

<script defer src="{{ '/assets/js/quantization-labs.js' | relative_url }}"></script>
