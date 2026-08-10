---
layout: post
title: AIGC 分布式推理：从 Wan2.2 理解 DP/TP/CP/SP/PP
date: 2026-08-08 12:00:00
description: 以 Wan2.2 TI2V 5B 为例，记录分布式推理中各种并行方式切什么、怎么通信、代价有多大
tags: [分布式推理, Diffusion, 视频生成, 推理优化]
categories: [生成模型]
pretty_table: true
toc:
  sidebar: left
---

## 为什么？

最近做了一个项目，是图生视频的项目，项目要求实时性，也就是说如果是要生成 1s 的视频，那么必须要在 1s 内生成完成。根据计算量来衡量，单卡是无论如何不能达到需求的，因此考虑多卡来实现，那么最简单的就是 CP 了，将整个上下文分在多卡上推理来提高整体的吞吐。这个过程中对分布式推理产生了兴趣，以前大概是知道分布式推理中 TP/CP/PP/DP 等是怎么回事，也就是停留在知道原理上，借着这个机会把这些分布式推理的原理彻底搞明白，也算是知其所以然了。

为了介绍如何进行分布式推理，我们以 Wan2.2 TI2V 5B 这个模型来进行说明，先看一下 Wan2.2 的模型结构。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/wan22-p1.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/wan22-p2.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. Wan2.2 模型结构图（分两部分展示）.
</div>

其中参数如下：

```text
B=1, N=27,280, H=3,072
Heads=24, HeadDim=128
FFN=14,336
Layers=30, Steps=50, dtype=BF16
```

这几个数字后面会反复出现，值得先解释一下它们是怎么来的：

- **N=27280**：目标视频是 704×1280、121 帧。先过 VAE，stride 是 (4,16,16)，得到 latent `[1,48,31,44,80]`；再过 Conv3d patch embedding，kernel=stride=(1,2,2)，得到 `[1,3072,31,22,40]`。展平后 N = 31×22×40 = 27280。这 27280 个 video token 就是整篇文章要反复切来切去的东西。
- **H=3072、24 heads × 128**：每个 WanBlock 的隐藏维，24×128 = 3072。
- **FFN=14336**：FFN 中间层，3072 → 14336 → 3072。
- **Layers=30、Steps=50**：30 个 WanBlock 堆叠，扩散去噪循环 50 步。也就是说，一次完整的推理要执行 1500 次 WanBlock。

最后这个数字是理解整件事的关键：**任何一个"每层一次"的通信操作，在一条视频里都会被放大 1500 倍。** 这就是为什么下文会反复纠结"这次 collective 到底有多贵"。

## 是什么：分布式推理到底在切什么？

分布式推理并不是简单地把同一个模型放到多张 GPU 上运行，而是把原本由一张 GPU 完成的计算，拆成多个 rank 上的局部计算，再通过集合通信恢复与单卡等价的结果。分布式计算可以从下面几个维度进行切分：

| 切分对象            | 对应方式 | Wan2.2 中的实际对象                     | 主要目的                  |
| :------------------ | :------- | :-------------------------------------- | :------------------------ |
| Batch / Request     | DP       | `B` 或多个生成请求                      | 提高吞吐量                |
| Token Sequence      | CP       | `N=27280` 个视频 token                  | 降低单卡 Attention 计算量 |
| Token Sequence      | SP       | LayerNorm、Residual、FFN 等区域的 token | 降低激活显存              |
| Hidden / Head / FFN | TP       | `24 heads`、`H=3072`、`FFN=14336`       | 拆分权重和矩阵乘法        |
| Transformer Layers  | PP       | `30` 个 WanBlock                        | 降低单卡模型显存          |

把这张表画到那个真实的张量上，就是下面这张图。同一个 `[1, 27280, 3072]`，DP 切 B、CP/SP 切 N、TP 切 D，而 PP 切的是"这张图重复 30 次"的那个方向：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-01-parallel-axes.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. 五条正交的切分轴.
</div>

从这个角度看，DP、TP、CP、PP 的区别，并不只是使用了不同通信算子，而是它们选择了不同的切分轴。常见 world-size 关系是：

$$
W = DP \times PP \times TP \times CP
$$

SP 通常复用 TP group，而不是额外占一个独立 world dimension（这里先不考虑 CFG 并行）。

后面每一节的图，都建议按同样的四个问题去读：

1. **权重切了吗？** —— DP/CP 复制权重，TP/PP 真正切开权重。
2. **激活切了吗？** —— 切的是哪一维：样本、token，还是 hidden 通道。
3. **什么时候通信？** —— 在哪个算子的前面或后面。
4. **通信完谁拥有什么？** —— All-Reduce 后人人有完整和；All-Gather 后人人拼出完整张量；All-to-All 后仍是分片，只是换了一维。

## 怎么做：Wan2.2 如何在多张 GPU 上切分、计算与通信？

### DP：按请求与 Batch 切分

DP 是最简单的并行方式，它直接切的是 Batch 或者请求的维度。以图生图为例，假设你要生成 8 个视频，需要使用 8 张图，总共有 8 张卡，那么 DP 就是每张卡用一个图生成一个视频，每个卡都是全部的模型和全部的 sequence 在跑。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-02-dp.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. DP 切样本，不切权重.
</div>

这张图值得注意的是"**没有发生什么**"：不切 head、不切 FFN 通道、不切 token、不切层。每张卡都持有完整的 30 个 WanBlock（BF16 约 10 GB），各自跑完整的 50 步去噪。

因此 **推理时 DP rank 之间是零通信的** —— 算完按 request id 把结果返回就行。只有训练才需要在 backward 之后 All-Reduce 梯度 `dW = (dW_A + dW_B)/2`，否则两张卡的权重会在 optimizer step 之后逐渐漂移。

对我这个项目来说，DP 的结论很直接：**它提高吞吐，但一点也不降低单条视频的时延。** 8 张卡跑 8 个请求，每个请求的耗时和单卡完全一样。所以在"1 秒视频必须 1 秒内生成"的约束下，DP 是优先级最低的那条轴。

### TP：按 Attention Head 和 FFN 通道切分

#### 什么是 TP

TP 是唯一真正把**权重矩阵本身**切开的并行方式（PP 切的是"哪些层归谁"，单个矩阵仍然是完整的）。

理解 TP 的全部内容，其实就是理解一件事：一个 `Y = X · W` 的矩阵乘法，有两条可以切的轴，选哪条决定了要不要通信、通信什么。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-03-gemm-split-axes.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. Column Parallel 与 Row Parallel.
</div>

先约定记号，不然很容易被 PyTorch 绕进去：数学形状写作 `X[m,k] · W[k,n] = Y[m,n]`，而 `nn.Linear` 内部把权重存成 `weight[n,k]`（转置存放）。所以下文说"沿 n 切"时，代码里对应的是 `narrow(dim=0)`。

- **Column Parallel（切 W 的输出维 n）**：每个 rank 算出 Y 的一段通道，直接拼起来就是完整的 Y。**本身不产生任何通信。**
- **Row Parallel（切 W 的输入维 k）**：每个 rank 算出的是形状完整、但数值不完整的 partial。**必须 All-Reduce SUM 才能得到正确结果。**

而这两者必须成对出现：Column 的输出恰好就是 Row 的输入所需的形状。于是在 `Column → 逐元素算子 → Row` 这一段里，中间的激活全程都是切开的，只在最后合一次。这就是 TP 的基本骨架：

```text
Q/K/V (Column) → Attention → Wo (Row) → All-Reduce
ffn_0 (Column) → GELU      → ffn_2 (Row) → All-Reduce
```

#### TP 切的什么

**Column Parallel：沿输出通道切。**

以 Wan2.2 的 Q 投影为例，`Wq_math = [3072, 3072]`，TP=2 时沿 n 切成两半：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-04-column-parallel.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. Column Parallel 的权重切分.
</div>

rank 0 只持有 `Wq₀`（n = 0…1535），只算出 `Q₀`；rank 1 同理。**没有任何一个 rank 需要另一半的结果才能算完自己那段**，所以这一步零通信。

这里的 1536 不是随便切的：1536 = 12 heads × 128。也就是说，"沿 hidden 维切成两半"和"把 24 个 head 分给两张卡"是同一件事。这一点很重要，因为 **head 是 Attention 的天然独立单元** —— head i 的 softmax 只用 head i 的 Q/K/V，所以按 head 切完全不损失数学等价性。

由此得到 TP 的第一条硬约束：**heads 必须能被 TP 整除**。24 % TP == 0，所以 TP ∈ {1,2,3,4,6,8,12,24}。

**Row Parallel：沿输入通道切。**

输出投影 `Wo` 走的是另一条路。它的输入 `A` 已经是被切开的（因为上一步是 Column），所以 `Wo` 必须沿 k 切，才能和 A 的分片对上：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-05-row-parallel.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6. Row Parallel 与 All-Reduce SUM.
</div>

为什么必须是 SUM 而不是拼接？直接看分块矩阵乘法：

$$
W_o = \begin{bmatrix} W_{o0} \\ W_{o1} \end{bmatrix}, \quad
A = \begin{bmatrix} A_0 \mid A_1 \end{bmatrix}
$$

$$
A \cdot W_o = A_0 \cdot W_{o0} + A_1 \cdot W_{o1}
$$

沿 k 切开之后，每个 rank 算出的是完整 `[m,n]` 矩阵中"**一部分被加项的和**"。少加一项，结果就是错的。所以只能逐元素求和，不能拼接。

> **最容易踩的坑：bias 会被加 TP 次。**
> 如果在每个 rank 的 partial 上先加 bias，再 All-Reduce，bias 就被重复加了 TP 次。正确顺序是：各 rank 算无 bias 的 partial → All-Reduce SUM → 只加一次 bias。

#### Wan2.2 中如何进行切分

把上面两条规则套到 Wan2.2 的 Self-Attention 上，完整链路是这样的：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-06-tp-attention-wan22.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 7. TP 应用到 Wan2.2 Self-Attention.
</div>

从输入到输出，两个 rank 各自的形状变化：

| 步骤           | rank 0                     | rank 1                       | 通信               |
| :------------- | :------------------------- | :--------------------------- | :----------------- |
| 输入 X         | `[1,27280,3072]`           | `[1,27280,3072]`（完整复制） | 无                 |
| Q/K/V Column   | `Wq₀ = weight[0:1536, :]`  | `Wq₁ = weight[1536:3072, :]` | 无                 |
| Q/K/V 输出     | `[1,27280,12,128]`         | `[1,27280,12,128]`           | 无                 |
| 本地 Attention | 12 heads，全部 27280 token | 12 heads，全部 27280 token   | 无                 |
| Wo Row         | `Wo₀ = weight[:, 0:1536]`  | `Wo₁ = weight[:, 1536:3072]` | 无                 |
| Wo 输出        | `[1,27280,3072]` partial   | `[1,27280,3072]` partial     | **All-Reduce SUM** |
| 最终 residual  | `[1,27280,3072]`           | 与 rank 0 逐比特相同         | —                  |

有几点需要留意：

- **TP 不切 token。** 每个 rank 仍然要算 27280 × 27280 的 attention 分数矩阵，只是 head 数从 24 降到 12。TP 省的是显存和 GEMM 宽度，**不是序列长度**。
- **QK RMSNorm 需要额外通信。** RMSNorm 是跨 head 归一化的，而 head 已经被切开了，所以需要先 All-Reduce 局部的 sumsq。
- **3D RoPE 不受影响。** RoPE 是按 token 位置作用的，和 head 怎么分无关。
- **Cross-Attention 完全同构**：video Q 是 Column，text K/V 是 Column，输出 O 是 Row。所以每个 block 里光是 Attention 部分就有 2 次 All-Reduce。

FFN 部分的逻辑一样，只是数字更大：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-07-tp-ffn-wan22.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 8. TP 应用到 Wan2.2 FFN.
</div>

`ffn_0` 是 Column（`[14336,3072]` → 2×`[7168,3072]`），中间过一个逐元素的 GELU（不需要通信），`ffn_2` 是 Row（All-Reduce）。**14336 维的中间激活全程都是切开的**，这是 TP 在 FFN 上省显存的主要来源。

这里还有一个实现上的细节值得说清楚：**权重是加载时切一次，不是每个 denoise step 重切。**

1. checkpoint 里是完整的 `ffn_0.weight = [14336,3072]`；
2. 构造模块时先确定 target shape：`local_out = 14336 / TP = 7168`；
3. 流式加载时直接 `narrow`：Column 用 `narrow(0, rank*7168, 7168)`，Row 用 `narrow(1, rank*7168, 7168)`；
4. 之后 50 步去噪全程不再切，每个 tp rank 只常驻自己那片。

一张 TP 切分速查表：

| 模块                            | 切法                       | 通信           |
| :------------------------------ | :------------------------- | :------------- |
| self/cross 的 Q、K、V           | weight `dim 0`（输出通道） | 无             |
| self/cross 的 O                 | weight `dim 1`（输入通道） | All-Reduce SUM |
| ffn_0                           | weight `dim 0`             | 无             |
| ffn_2                           | weight `dim 1`             | All-Reduce SUM |
| LayerNorm、modulation、residual | 不切，完整复制             | 无             |

#### 通信之后如何恢复

前面反复说"All-Reduce 之后就恢复了"，这里把这个"恢复"拆开看一眼：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-08-tp-allreduce-restore.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 9. All-Reduce 如何把 partial 还原成与单卡等价的结果.
</div>

四个 rank 各自持有一个"形状对、数值不对"的 partial —— 每个都少加了 3/4 的项。All-Reduce SUM 把它们逐元素相加，再让每个 rank 都拿到这个和。此时四张卡的 residual 逐比特一致，可以继续进入下一个子层，**与单卡结果完全等价**。

另外，NCCL 实际上并不会真的"先全部汇总再广播"，而是拆成两步执行：

- **Reduce-Scatter**：每个 rank 只对 1/p 的通道求和，结束后 rank i 持有第 i 段的正确值；
- **All-Gather**：这 p 段正确值互相交换拼接，每个 rank 拿到完整张量。

总链路流量约为 `2 × (p-1)/p × 张量大小`，而不是 p 倍。顺带一提，这个分解也直接解释了 Sequence Parallel 的思路 —— 既然中间本来就会经过一个"按维度切开"的状态，那不如就停在那里，别再 All-Gather 回去了。

最后算一下这次通信到底有多贵（BF16、B=1）：

| 场景                      | 逻辑张量                 | 大小        |
| :------------------------ | :----------------------- | :---------- |
| TP Row All-Reduce（CP=1） | `[1,27280,3072]`         | 159.84 MiB  |
| TP Row All-Reduce（CP=2） | `[1,13640,3072]`         | 79.92 MiB   |
| 每个 WanBlock             | self.o + cross.o + ffn_2 | 3 次        |
| 每条视频                  | 3 × 30 层 × 50 步        | **4500 次** |

4500 次、每次上百 MiB。这就是 **TP group 必须待在同一台机器的 NVLink 域内**、几乎不能跨机的原因。

### CP：按视频 Token 序列切分

TP 把 head 切开了，但每个 rank 仍然要面对完整的 27280 个 token。而 Self-Attention 的复杂度是 O(N²) —— 对视频生成来说，N 才是真正的瓶颈。CP 就是冲着这个来的。

CP 的第一步很直白：进入 30 层之前，把 token 序列切开，而且**只切这一次**。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-09-cp-token-split.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 10. CP 切 token，以及马上遇到的问题.
</div>

注意 RoPE 的 cos/sin 必须按同一个 token 范围同步切片，否则 position 和 token 就对不上了。

然后马上就撞上一个问题：**切完之后 Self-Attention 就算不对了。** rank 0 手里只有前一半的 K/V，但 token 0 完全可能需要 attend 到 token 20000 —— 而那个 token 的 K/V 在 rank 1 手上。直接算，结果就是错的。

有两条路：

- **方案 A：All-Gather K/V**，把完整的 K/V 复制到每张卡。正确，但激活显存一点没省下来，在 27280 这种长度上代价过高。
- **方案 B：Ulysses —— 换一个切法。** 既然"按 token 切"在 Attention 内部看不到全局，那就临时换成"按 head 切"。

Ulysses 的核心就是这一步。把 `[token, head]` 看成一张二维所有权表，All-to-All 做的事情本质上是"转置所有权"：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-10-cp-a2a-ownership.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 11. All-to-All 把「按 token 切」换成「按 head 切」.
</div>

- **A2A 之前**：每个 rank 持有 **半个序列 × 全部 24 heads**，即 `[13640 tokens, 24 heads]`。
- **A2A 之后**：每个 rank 持有 **完整序列 × 一半 heads**，即 `[27280 tokens, 12 heads]`。

两边的数据量完全一样，变的只是"它拥有哪一块"。这也是为什么说 A2A 是**重分片，不是复制** —— 它和 All-Gather 有本质区别，每个 rank 最终都不会拥有全部 24 个 head。

为什么这样换就对了？因为 Attention 的独立性有两个方向：

- **head 之间互相独立** —— head i 的 softmax 只用 head i 的 Q/K/V；
- **token 之间不独立** —— 每个 query token 必须看到全部 27280 个 key。

所以"切 head"是安全的，"切 token"在 Attention 内部是不安全的。**Ulysses 就是在进入 softmax 之前，把不安全的切法临时换成安全的，算完再换回来。** 代价是两次 All-to-All，收益是整个 block 的其余部分都只需处理 N/CP 个 token。

完整的数据流：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-11-cp-ulysses-flow.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 12. Ulysses CP 的完整数据流.
</div>

所以每个 Self-Attention 的通信总账是：**1 次 fused QKV All-to-All + 1 次 output All-to-All**。

还有一个容易忽略但很划算的工程细节：**30 层跑完之后的那次 All-Gather，要放在 output head 之后。**

先过 output head 把 hidden 从 3072 投影到 192（`192 = 48×1×2×2`），再 gather，然后 unpatchify。本地 shard 在 192 维时约 5.00 MiB，在 3072 维时约 79.92 MiB —— **通信量差 16 倍**。这是一条通用经验：collective 尽量放在张量最窄的地方。

CP 到底省了什么，一张表说清楚：

| 维度                  | 变化                 | 说明         |
| :-------------------- | :------------------- | :----------- |
| Self-Attention 计算量 | O(N²) → O((N/CP)·N)  | CP=2 时减半  |
| block 内激活显存      | 27280 → 13640 tokens | 线性下降     |
| 模型权重              | 完整复制             | **一分没省** |
| 新增通信              | 每层 2 次 All-to-All | 换来的代价   |

**CP 是"时延优化"，而不是"显存装得下"的手段。** 这正是实时视频生成需要它的原因 —— 也是我这个项目最开始就选它的原因。

另外，Cross-Attention 不需要这次 A2A：video token 做 Query，text 的 K/V 只有 512 个，且在每张卡上都是完整复制的。

### SP：按非 Attention 区域的 Token 切分

SP 和 CP 都切 token，名字也很像，很多框架甚至混用这两个词。但它们的边界完全不同，区别就在于：**Attention 期间，谁拥有完整序列。**

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-12-sp-vs-cp.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 13. SP 与 CP 的边界差异.
</div>

**Megatron 风格的 SP**，是 TP 的一个补丁。TP 切了 Attention 和 FFN 的权重，但 LayerNorm、dropout、residual 这些 token-local 算子的激活在每个 TP rank 上还是完整复制的。SP 的做法是：在 block 边界保持 sequence shard，只在进入 TP Column Linear 之前 All-Gather 恢复完整序列，在 TP Row Linear 之后用 Reduce-Scatter 重新切回去。

注意这正好利用了前面提到的分解：`All-Reduce = Reduce-Scatter + All-Gather`。SP 只是把这两半拆开用，所以**它几乎不增加通信量** —— 只是把原本 All-Reduce 的那次通信换了个形式。

**Ulysses CP** 则是整个 block 全程保持 sequence shard，只在 Attention 内部用 head shard 临时换取完整上下文。

| 对比项           | Megatron-SP                         | Ulysses CP                               |
| :--------------- | :---------------------------------- | :--------------------------------------- |
| 主要目的         | 省掉 TP 没覆盖的激活（LN/residual） | 降低长序列 Attention 的计算与激活        |
| Attention 内部   | 不保留 sequence shard               | 不保留 sequence shard（换成 head shard） |
| 标志性算子       | All-Gather + Reduce-Scatter         | **sequence ↔ head All-to-All**          |
| 是否解决长上下文 | 否                                  | 是                                       |
| 是否与 TP 耦合   | 紧耦合                              | 正交，可独立使用                         |

> **术语提醒**：判断一份实现到底是哪一种，不要看它叫什么名字，看 Attention 前后出现的是什么算子。出现 All-Gather / Reduce-Scatter 的是 Megatron-SP；出现 sequence ↔ head All-to-All 的是 Ulysses CP。

### PP：按 WanBlock 层切分

PP 切的是深度：把 30 个 WanBlock 分成几段，每段放一张卡。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-13-pp-split-timeline.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 14. PP 按层切分与流水线气泡.
</div>

PP 传的既不是权重也不是梯度，而是**层与层之间的那一个激活张量** —— 形状就是 residual 本身 `[B, N, 3072]`。而且用的是 P2P send/recv，不是 collective，所以单次通信其实很便宜。

问题在于**气泡**。stage 0 在算 micro-batch 0 的时候，stage 1/2/3 都在空转；要等流水线填满才能全速运行。4 个 micro-batch、4 个 stage 时，气泡占比是 `(p-1)/(m+p-1) = 3/7 ≈ 43%`。

但对我这个场景来说，PP 有两个更根本的问题：

1. **扩散推理是 50 步串行的循环**，每一步都依赖上一步的完整 latent。单条视频天然没有 micro-batch 可以拿来填气泡。只有并发多个请求时，PP 才能把流水线填满 —— 而那时候你其实更应该直接用 DP。
2. **PP 的价值场景是"模型单卡装不下"**。Wan2.2-TI2V-5B 在 BF16 下只有约 10 GB，单张 A100/H100 绰绰有余。

所以后面的组合方案里不启用 PP。它在 100B+ 的大模型上是刚需，但在 5B 的视频扩散模型上基本没有用武之地。

### TP × CP：同时切 Head 与 Token

现在把 TP 和 CP 叠起来。TP=2 × CP=2 = 4 张卡，每张卡的身份是一对坐标 `(cp, tp)`，rank 编号是 `rank = cp × TP + tp`。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-14-tp-cp-mesh.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 15. TP × CP 的 4 卡网格与两个 process group.
</div>

关键在于**两个 process group 在 rank 网格上是正交的** —— 一横一竖：

- **TP groups（固定 cp）**：`[0,1]` 和 `[2,3]`。Row Linear 的 partial 在这里 All-Reduce SUM。
- **CP groups（固定 tp）**：`[0,2]` 和 `[1,3]`。Self-Attention 前后在这里 All-to-All。

读这个网格有个口诀：

- rank 0 与 rank 1：**token 相同，head 权重不同**（同一个 CP row，不同 TP 列）；
- rank 0 与 rank 2：**head 权重相同，token 不同**（同一个 TP 列，不同 CP row）；
- 同一个 tp 坐标的 TP shard，在不同 cp 坐标上是复制的。

这里有一条很容易挂掉的约束：head 要被切两次。先被 TP 切成 `24/TP = 12`，再被 CP 切成 `12/CP = 6`。所以必须**同时**满足：

```text
24 % TP == 0   且   (24 / TP) % CP == 0
```

把一个 WanBlock 内部展开，rank 0 看到的形状变化是这样九步：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-15-tp-cp-flow.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 16. TP × CP 在一个 WanBlock 内的九步数据流.
</div>

| 步  | 操作              | 形状                      | 通信          |
| :-- | :---------------- | :------------------------ | :------------ |
| 1   | CP 切 token       | `[1,13640,3072]`          | 入口切一次    |
| 2   | TP Column Q/K/V   | `[1,13640,12,128]`        | 无            |
| 3   | CP A2A(seq→head)  | `→ [1,27280,6,128]`       | group `[0,2]` |
| 4   | 本地 Attention    | 完整 27280 token，6 heads | 无            |
| 5   | CP A2A(head→seq)  | `→ [1,13640,12,128]`      | group `[0,2]` |
| 6   | reshape           | `[1,13640,1536]`          | 无            |
| 7   | TP Row Wo         | `[1,13640,3072]` partial  | 无            |
| 8   | TP All-Reduce SUM | `[1,13640,3072]`          | group `[0,1]` |
| 9   | 本地 residual     | `[1,13640,3072]`          | 无            |

FFN 部分紧接着走 `ffn_0 TP Column [3072→7168] → 本地 GELU → ffn_2 TP Row [7168→3072] → All-Reduce`。**CP 在 FFN 里不新增任何 collective**，它唯一的作用是把 N 从 27280 变成 13640 —— 于是计算量和 All-Reduce 的字节数同时减半（159.84 MiB → 79.92 MiB）。

一句话总结这两条轴的分工：

> **CP 通信改变 token / head 的归属；TP 通信把 hidden-channel 的 partial sum 合成完整 residual。**

顺序不能反：A2A 换的是 head 归属，必须发生在 Attention 两侧；All-Reduce 合的是通道 partial，必须发生在 Row Linear 之后。

### DP × TP × CP：多维并行如何组合

再加一层 DP，就是 8 张卡。rank 编号公式：

```text
rank = ((dp × CP) + cp) × TP + tp
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-16-dp-tp-cp.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 17. DP × TP × CP 的 8 卡组合.
</div>

结构其实很简单：**每个 DP plane 内部就是一份完整的 TP2×CP2 拓扑**，两个 plane 同构，只是输入不同的样本。

- plane 0：TP groups `[0,1] [2,3]`，CP groups `[0,2] [1,3]`；
- plane 1：TP groups `[4,5] [6,7]`，CP groups `[4,6] [5,7]`。

推理前向时，**所有 collective 都被关在 plane 内部**，两个 plane 之间零通信。这是 DP 扩展性最好的原因：加卡几乎线性加吞吐。代价也同样明显 —— 单条视频的时延一点没变。

如果是训练，需要同步的是"同一个 tp 坐标上那份权重"的梯度。比如 `Wq` 的 tp0 shard 存在于 ranks `[0,2,4,6]`：其中 cp ranks 处理同一个样本的不同 token（要先 SUM），dp planes 处理不同样本（再求平均）。工程上通常把这两步合并成一个 DP×CP reduction group。推理没有 dW，这一步完全不需要。

### CFG × TP × CP：AIGC 特有的第五条轴

上面五种并行都是从 LLM 那边继承来的。但扩散模型有一条自己的轴：**CFG（Classifier-Free Guidance）**。

CFG 要求每个去噪步同时算两条分支 —— conditional（用正向 prompt）和 unconditional（用负向或空 prompt），然后组合：

$$
\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
$$

这两条分支输入同一个 latent，互不依赖，天然可以并行。于是就有了实际跑起来的 8 卡拓扑：**CFG=2 × TP=2 × CP=2**。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-17-cfg-tp-cp.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 18. CFG × TP × CP 的 8 卡拓扑.
</div>

rank 公式和坐标反解：

```text
rank = ((cfg × CP + cp) × TP + tp)
tp   = rank % TP
cp   = (rank // TP) % CP
cfg  = rank // (TP × CP)
```

于是有三组正交的 process group：

| group             | 成员                      | 用途                  |
| :---------------- | :------------------------ | :-------------------- |
| TP（固定 cfg,cp） | `[0,1] [2,3] [4,5] [6,7]` | Row Linear All-Reduce |
| CP（固定 cfg,tp） | `[0,2] [1,3] [4,6] [5,7]` | Self-Attention A2A    |
| CFG（固定 cp,tp） | `[0,4] [1,5] [2,6] [3,7]` | 每步末尾交换分支预测  |

每个 denoise step 的完整顺序是：

1. 所有 rank 从同一个 scheduler latent 和 timestep 出发（可以用 rank 0 broadcast 保证逐比特相同）；
2. cfg=0 的 plane 选 `ctx_cond`，cfg=1 的 plane 选 `ctx_uncond`；
3. 每个 CFG plane 内独立跑完整的 TP2×CP2 WanModel；
4. 每个 rank 拿到本分支的完整预测 `[1,48,31,44,80]`；
5. **CFG group All-Gather**，`[0,4] [1,5] [2,6] [3,7]` 各自交换 cond/uncond；
6. 每个 rank 本地 combine + UniPC scheduler step，更新后的 latent 进入下一步。

**CFG 并行和 DP 看起来像，但本质完全不同：**

| 对比项              | DP         | CFG 并行                              |
| :------------------ | :--------- | :------------------------------------ |
| 两个 plane 的输入   | 不同的样本 | **同一个 latent**，不同的 context     |
| 推理时 plane 间通信 | 0          | **每个 denoise step 一次 All-Gather** |
| 目的                | 提高吞吐   | **降低单条视频的时延**                |

最后一行才是重点：CFG 并行是少数几个**真正能降低单条视频时延**的手段之一 —— 本来要串行算两遍的分支，现在并行算了。这对实时场景很关键。

## 怎么选：回到最开始那个问题

把所有维度放在一起对比：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-19-decision.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 19. 各种并行的收益、代价与放置顺序.
</div>

|     | 单卡权重显存 | 单卡激活显存        | 单卡计算量 | 新增通信             |
| :-- | :----------- | :------------------ | :--------- | :------------------- |
| DP  | 不变         | 不变                | 不变       | 推理 0；训练梯度 AR  |
| TP  | ÷ TP         | ÷ TP（切开的部分）  | ÷ TP       | 每层 3 次 All-Reduce |
| CP  | 不变         | ÷ CP                | ÷ CP       | 每层 2 次 All-to-All |
| SP  | 随 TP        | ÷ TP（LN/residual） | 略降       | AR 拆成 RS + AG      |
| PP  | ÷ PP         | ÷ PP                | ÷ PP       | stage 间 P2P + 气泡  |

回到项目最开始的约束 —— **1 秒的视频必须在 1 秒内生成**，推导过程其实很直接：

1. **目标是时延，不是吞吐。** DP 增加吞吐但不降低单条时延，所以在实时场景里优先级最低。
2. **5B 模型单卡装得下。** BF16 约 10 GB，PP 的主要理由（模型放不下）不成立；加上扩散推理天然没有 micro-batch 填气泡，PP 直接出局。
3. **N=27280 才是瓶颈。** Attention 是 O(N²)，CP 直接把它砍成 1/CP。这就是当初第一反应选 CP 的原因，事后看是对的。
4. **还要更低时延就叠 TP。** 把 GEMM 宽度也切开，但要盯住每层 3 次 All-Reduce 的开销。
5. **CFG 是白送的一倍。** 两条分支本来就要算，并行起来几乎没有额外代价，每步只多一次 All-Gather。

还有一条很容易被忽略、但影响可能比选型本身更大的经验 —— **放置顺序：把通信最频繁的轴放在最快的链路上。**

| 层级   | 轴       | 通信频率             | 放置要求         |
| :----- | :------- | :------------------- | :--------------- |
| 最内层 | TP       | 每层 3 次 All-Reduce | 必须 NVLink 同机 |
| 次内层 | CP       | 每层 2 次 A2A        | 同机优先         |
| 再外层 | PP / CFG | 每步 1 次            | 可以跨机         |
| 最外层 | DP       | 推理零通信           | 随便跨机         |

反过来放（TP 跨机、DP 同机）通常直接慢一个量级。先按"谁通信最频繁"排序，再往拓扑上贴，基本不会错。

## 附：通信算子速查

最后附一张速查表，方便回头查。看到 shape 就知道该用哪个算子：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/distribute-aigc/fig-18-collective-cheatsheet.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 20. 通信算子与 Wan2.2 上的真实字节数.
</div>

选算子的判断法：

| 你想要的效果                | 用哪个         |
| :-------------------------- | :------------- |
| 每个人都拿到相同的完整和    | All-Reduce     |
| 拼接 shard 并复制成完整张量 | All-Gather     |
| 从按 token 切改成按 head 切 | All-to-All     |
| 求和之后仍然保持 shard      | Reduce-Scatter |

Wan2.2 上的真实字节数（BF16、B=1）：

| 场景                      | 逻辑张量               | 大小       |
| :------------------------ | :--------------------- | :--------- |
| TP Row All-Reduce（CP=1） | `[1,27280,3072]`       | 159.84 MiB |
| TP Row All-Reduce（CP=2） | `[1,13640,3072]`       | 79.92 MiB  |
| CP=2 fused QKV A2A        | `3 × [1,13640,24,128]` | 239.77 MiB |
| CP=2 output A2A           | `[1,27280,12,128]`     | 79.92 MiB  |
| CP final gather（192 维） | `[1,13640,192]`        | 5.00 MiB   |

需要说明的是，**逻辑张量大小不等于实际链路流量** —— 后者取决于 collective 的具体算法（ring / tree / NVLS）和拓扑。但作为量级判断，这张表已经够用了：它能告诉你哪次通信值得优化，以及为什么最后那次 gather 一定要放在 output head 之后。

## 小结

写到这里，回头看最开始那个"停留在知道原理上"的状态，差别其实在于三件事：

1. **每种并行选的是哪条切分轴** —— DP 切 B，TP 切 D，CP/SP 切 N，PP 切层。搞清楚这个，剩下的都是推论。
2. **通信发生在哪、为什么必须发生** —— Row Parallel 之后必须 SUM，是因为分块矩阵乘法的定义；Attention 前后必须 A2A，是因为 token 之间不独立而 head 之间独立。这些都不是约定，是数学上绕不过去的。
3. **代价有多大** —— 4500 次 All-Reduce、每次上百 MiB，这个数量级决定了 TP 只能待在 NVLink 域内。脱离具体数字谈并行策略，基本都是纸上谈兵。

选型本身反而是最后一步，而且一旦前三件事清楚了，选型几乎是自动的。
