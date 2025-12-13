---
layout: post
title: 对CUTE中的Thread-Value Layout的理解
date: 2025-12-10 12:42:00
description: 记录一下学习cute过程中的对Thread-Value Layout的理解
tags: cute cuda
categories: AI
pretty_table: true
toc:
  beginning: true
---


## 问题由来

最近在研究 [flash-attention2](https://github.com/Dao-AILab/flash-attention)的源码，算法层面的流程基本已经理解了{%sidenote 'One' '参考[Attention记录](https://valdrada.site/AI/2025-04-26-rethinking-attention-2.html)'%}，但实际代码里是怎么把这些计算落实到硬件上的，这部分一直没有完全想清楚。所以我开始沿着源码，把底层实现细节再看一遍。

在阅读源码时，我先去理解了 MMA（Matrix Multiply-Accumulate）的基本知识。参考了[CuTe Tiled MMA](https://leimao.github.io/blog/CuTe-Tiled-MMA/)这篇博客，大致搞明白了 MMA 是如何分块和运算的。不过，对于其中 cute::SM80_16x8x16_F16F16F16F16_TN 这个具体的 MMA 原子操作，还是有一个地方没弄懂——它打印出来的 Thread-Value Layout 图到底表示什么？这个“Thread-Value Layout”究竟是如何画出来的，又代表怎样的数据分布？

```
mma_atom
MMA_Atom
  ThrID:      _32:_1
  Shape_MNK:  (_16,_8,_16)
  LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

图片展示如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p1.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. MMA Atom Thread Value Layout.
</div>

为了搞懂这一点，我又花了一些时间，终于理清了 Thread-Value Layout 的含义与表示方式。

## 如何理解Thread-Value Layout

首先强烈推荐下面这个视频，是cute的创建者讲的，非常的清晰透彻：

<div class="row mt-3 justify-content-center">
    <div class="col-lg-10 col-md-12 mt-3 mt-md-0 text-center" >
        {% include video.liquid path="https://www.youtube.com/embed/ufa4pmBOBT8?si=o1xZU0JXJYQ5D5yx" class="rounded z-depth-1" width="100%" height="500" %}
    </div>
</div>
<div class="caption">
    Video. 1. Cute Explaination.
</div>

什么是thread value layout？ {%sidenote 'Two' '参考[博客](https://leimao.github.io/blog/CuTe-Thread-Value-Layout/)'%} 它是一种二维布局，它描述了线程组中的每个线程以及该线程将访问的每个值如何映射到任何布局的目标数据的一维坐标。Thread Value layout有两个mode，第一个mode代表的是thread的layout，第二个mode代表的是value的layout，意思是某个线程要负责哪几个数据。

举个实际的例子：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. Thread Value Layout Example 1.
</div>

如上所示，假设有一个一维的内存数据，总共有24个，每个线程要读的数据以不同的色块表示，也就是说每个线程读6个数据，以第0个线程为例，它读的数据就是第0，1，4，5，8，9个数据。这样分，总共需要24/6=4个线程，也就是4个线程读24个数据，每个线程读6个数据，每个线程读的数据是有分布的,线程本身也是有分布的，也就是如下所示的样子。
{%sidenote 'Three' '感觉这个layout 应该是((2,2),(3,2):(12,2),(4,1))才对，不过就是为了方便理解而已'%}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. Thread Value Layout Example 2.
</div>

更一般的，对于任意的内存数据，我都可以用这样的thread value layout来描述线程和一维的数据坐标怎么映射，实际上就是layout的composition而已。 这样做的好处是，线程要去读数据的时候直接从这个layout和真实数据composition之后的tensor中slice就可以了，也就是取第几个线程进行slice，得到这个线程对应的真实的数据坐标。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. Thread Value Layout Example 3.
</div>

## MMA Atom中的Thread Value(TV) Layout

理解了Thread Value Layout之后，我们看看具体的MMA Atom中的这个layout是怎么排布的，图中的线程和值又是怎么对应的？

先看A的TV是什么？

```
LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
```

就像之前说的TV有两个mode，一个是thread，一个是value，那么thread和value的layout分别是

```
Thread: (_4,_8):(_32,_1)
Value:  (_2,_2,_2):(_16,_8,_128)
```

对Thread的layout来说表示的是，总共有4*8=32个线程，shape是(4,8),stride是(32，1)，
对Value的layout来说表示的是，总共有2\*2\*2=8个线程，shape是(2,2,2),stride是(16,8,128)，
也就是对于MMA的A矩阵总共有32个线程负责读数据，每个线程读8个数据,总共32*8=256个数据，这和A的矩阵的shape 16\*16 = 256正好相等。把这个layout通过一下代码可视化一下：

```
auto la = cute::make_layout(cute::make_shape (cute::make_shape (4,8),cute::make_shape (2,2,2)),cute::make_stride(cute::make_stride(32,1),cute::make_stride(16,8,128)));
cute::print_latex(la);
```

得到：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. A's layout.
</div>

图中左侧行方向32个数字表示的线程id，上侧列方向的数字表示的value的坐标。 以thread 0为例, thread 0 负责的数据的坐标是 0, 16, 8, 24, 128, 144, 136, 152. 这些坐标都数据铺平后的一维坐标，如何对应到16X16的矩阵中呢？ 那么假设A矩阵是列主序的，也就是A矩阵的内存是列主序排列，那么 0, 16, 8, 24, 128, 144, 136, 152 这些坐标对应矩阵的哪一行哪一列呢？ 其实就很好计算了

```
假设 坐标是idx
那么 ：
row = idx % 16
col = idx / 16
```

那么带入计算一下就能得到对应的坐标是(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9),这和图中的T0这个线程的位置就对应上了，同理可以计算其他线程的位置。

那么问题是为什么A是列主序的？实际上可以看一下打印的代码：

```

template <class... Args, class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex(TiledMMA<Args...> const& mma,
            TikzColorFn color = {})             // lambda(thr_idx,val_idx) -> tikz color string
{
  auto tile_mnk = tile_shape(mma);

  Tensor refC = make_identity_tensor(select<0,1>(tile_mnk));
  Tensor tensorC_TV = composition(refC, mma.get_layoutC_TV());

  Tensor refA = make_identity_tensor(select<0,2>(tile_mnk));
  Tensor tensorA_TV = composition(refA, mma.get_layoutA_TV());

  Tensor refB = make_identity_tensor(select<1,2>(tile_mnk));
  Tensor tensorB_TV = composition(refB, mma.get_layoutB_TV());

  detail::print_latex_mma(tensorC_TV, tensorA_TV, tensorB_TV, tile_mnk, color);
}

```

其中每个逻辑Tensor都是通过make_identity_tensor创建的，这里面的默认排列方式就是列主序。

这里有个不一样的地方就是矩阵B，上面的代码中B的shape是nxk， 不是我们直观意义上kxn，这是为什么？

这里就需要仔细看看CUTE的[官方解释](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html),大概意思是说：

1. **逻辑意义**  
   在 MMA 的数学公式里，矩阵乘法写作：

   $$
   \begin{aligned}
   C[m,n] = \sum_k A[m,k] \cdot B[n,k]
   \end{aligned}
   $$

   可以看到，B 的第二个维度是 K，用于与 A 做内积。逻辑上，B 的 shape 就是 `(N,K)`，而不是 `(K,N)`，即使它在画图或概念上看起来是“转置了”的。

2. **转置的体现**  
   所谓 TN/NT 只是描述 MMA 指令如何**遍历和加载**数据，而不是改变逻辑 tensor 的 shape。B 的逻辑坐标 `(n,k)` 永远固定不变，真正的“转置”通过 layout（thread/value mapping）以及寄存器访问顺序实现。

3. **执行层面**  
   最终执行 MMA 时会调用 PTX 指令，例如：
    ```
    mma.sync.aligned.m16n8k16.row.col.f16
    ```
逻辑 tensor `(N,K)` 被映射到寄存器组，每个线程负责不同的 N slice，沿 K 连续访问。虽然逻辑上是 `(N,K)`，但在物理执行和绘图上，B 看起来像是 “K×N”，这是为了和 A 的 M×K 对齐，方便可视化和寄存器映射理解。

## 最后

看到SM80_16x8x16_F16F16F16F16_TN的定义

```
    struct SM80_16x8x16_F16F16F16F16_TN
    {
      using DRegisters = uint32_t[2];
      using ARegisters = uint32_t[4];
      using BRegisters = uint32_t[2];
      using CRegisters = uint32_t[2];
    
      CUTE_HOST_DEVICE static void
      fma(uint32_t      & d0, uint32_t      & d1,
          uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
          uint32_t const& b0, uint32_t const& b1,
          uint32_t const& c0, uint32_t const& c1)
      {
    #if defined(CUTE_ARCH_MMA_SM80_ENABLED)
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{\%0,  \%1},"
          "{\%2,  \%3,  \%4,  \%5},"
          "{\%6,  \%7},"
          "{\%8,  \%9};\n"
          : "=r"(d0), "=r"(d1)
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
             "r"(b0),  "r"(b1),
             "r"(c0),  "r"(c1));
    #else
        CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
    #endif
      }
    };
```

其中调用的PTX指令为

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

和之前解释的A和B的矩阵排布不一样， .row 表示A是行主序，.col表示B是列主序，为什么？

这个地方要明白PTX指令操作的是寄存器不是全局内存，这个图中展示的内存最终会按照约定的方式被拷贝到寄存器中。