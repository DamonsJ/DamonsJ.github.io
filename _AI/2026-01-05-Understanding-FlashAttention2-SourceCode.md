---
layout: post
title: 解析FlashAttention2源码
date: 2026-01-05 18:48:00
description: 记录一下学习FlashAttention2源码过程中对一些概念的理解
tags: [FlashAttention, CuTe, CUDA, 源码解读]
categories: [GPU 与高性能计算]
pretty_table: true
toc:
  beginning: true
---

## 为什么？

随着优化工作的深入，需要对底层的具体执行代码要有深入的理解。之前算法层面的流程基本已经理解了{%sidenote 'One' '参考[Attention记录](https://valdrada.site/AI/2025-04-26-rethinking-attention-2.html)'%}，但实际代码执行和算法理解上也不是一个层面的知识。所以决定沿着源码，把底层实现细节再看一遍，同时也是熟悉一下CUTE和Cutlass的一些知识，为以后的算子层面的优化做铺垫。

## Flash Attention2 源码

为了方便，我们以headdim=128，sm80为例子来阅读源码，具体的代码路径参考[flash_bwd_hdim128_bf16_sm80.cu](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu), 仅为参考所以不考虑attention中的特殊的参数，也就是设置causal=False，dropout=False, window 不设置，Has_alibi = False, Is_local = False, 也就是Full Attention的情况。

实际上FA2的源码虽然看上去很多，但是核心的执行部分在[flash_fwd_kernel.h](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)中，也就是其中的`compute_attn_1rowblock`函数。

根据[Attention记录](https://valdrada.site/AI/2025-04-26-rethinking-attention-2.html)中的算法流程，我主要记录显存搬运，MMA以及softmax三个大的部分的详细内容，代码基本上是按照算法流程进行的。

### 基本模版参数及说明

根据以上假设，有以下模版参数, 这些参数的设置位于[kernel_traits.h](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h)中

```
ElementAccum    = float
Element         = bfloat16
kHeadDim_       = 128
kBlockM_        = 128
kBlockN_        = 32
kNWarps_        =  4
kNThreads       = kNWarps * 32 = 128
Share_Q_K_smem  = false
Is_Q_in_regs_   = false
Has_cp_async    = true
kBlockKSmem     = 64
kBlockKGmem     = 128
kSwizzle        =  3

kGmemElemsPerLoad         = sizeof(uint128) / sizeof(Element)= 16/2 = 8
kGmemThreadsPerRow        = kBlockKSmem / kGmemElemsPerLoad = 8
Shape<kBlockM, kHeadDim>  = (128, 128)
sizeof(Element)           = 2 bytes（half）
kSmemQSize                = 128 × 128 × 2 = 32768 bytes = 32 KB

Shape<kBlockN, kHeadDim>  = (32,128)
kSmemK                    = 32 × 128 × 2 = 8192 bytes
kSmemV                    = 8192 bytes
kSmemKVSize               = kSmemK + kSmemV = 16384 bytes = 16 KB

kSmemSize                 = kSmemQSize + kSmemKVSize
                          = 32768 + 16384
                          = 49152 bytes = 48 KB

```

这里可以看到`kBlockM_ = 128` 也就是`Q`在M方向上按照`kBlockM_ = 128`进行分块，在[flash_fwd_launch_template.h](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h)中可以看到，在启动kernel的时候设置的grid为：

```
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
```

也就是对于shape为`b x h x s x d`的`Q`矩阵，一个block只计算部分seqlen长度，在上述的基本参数下， 1个block计算`kBlockM_ = 128` 这么长的seqlen，因为`kHeadDim_ = 128`，所以Q矩阵的大小就是`128x128`, 因为`kBlockN_ = 32`，所以K矩阵的大小是`32x128`， 也就是一个block计算的attention 分块的大小是 `Q: 128x128 K: 32x128` ,总共启动了`num_m_block x params.b x params.h`这么多个block进行全部的attention运算。

### 显存搬运

#### 显存视图

因为我们只考虑full attention这种用况，函数 `compute_attn_1rowblock`的前半部分的代码基本上是处理一些特殊情况，因此直接来到：

```
const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
    + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                      + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_q, params.h, params.d),
                        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                      + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                      + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                        make_stride(params.v_row_stride, params.v_head_stride, _1{}));
Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                        Shape<Int<kBlockM>, Int<kBlockN>>{},
                        make_stride(params.seqlen_k_rounded, _1{}));

Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                        typename Kernel_traits::SmemLayoutQ{});
// Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                        typename Kernel_traits::SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
```

这部分负责global memory到shared memory的显存搬运，但是这个地方的操作只是对内存进行视图划分并没有进行真正的global memory到shared memory的显存搬运。

张量`mQ`表示的是`Q`的分块显存张量，`gQ`表示的是当前分块的global memory的划分，也就是当前block内操作的`gQ`显存，总的`Q`的shape是`b x h x s x d`，`gQ`是第`bidb`个batch，第`bidh`个head和第`m_block`个Q的划分，大小就是`kBlockM x kHeadDim = 128 x 128`， 也就是说`gQ`表示了当前处理的Q的分块的global memory张量，同时`gK`和 `gV`分别表示的是K和V的分块的global memory张量，K和V的张量shape是`kBlockN x kHeadDim = 32 x 128`。

然后是规定了QKV的shared memory的layout

```
Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
// Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                        typename Kernel_traits::SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
```

主要看一下这里的Q的shared memory的layout，`sQ`表示的是一个tensor，指针是shared memory的地址，layout是`SmemLayoutQ`。`SmemLayoutQ`的定义如下：

```
using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
// SmemLayoutAtomQ是一个原子布局，意思是对于Q的共享内存，使用SmemLayoutAtomQ作为原子布局，然后tile_to_shape将原子布局转换为形状布局。
// 为的是将计算和访存并行
using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
```

这里主要是两个点，一个是`SmemLayoutAtomQ`这个原子布局（包括swizzle），一个是如何将这个布局铺成最终的shape。

##### 1. SmemLayoutAtomQ

它其实是swizzle和一个layout的复合layout

首先看看layout：

```
Layout<Shape<_8, Int<kBlockKSmem> >,Stride<Int<kBlockKSmem>, _1> >{}
即：
Layout<Shape<_8, Int<64>>,Stride<Int<64>, _1>>{}
```

它表示的是`8x64`的矩阵，stride是`64x1`，也就是行主序的一个矩阵。

在看看swizzle。 swizzle的定义如下：

```
template <uint32_t S0, uint32_t F0, auto S1>
CUTE_HOST_DEVICE constexpr
auto
shiftr(MixedBits<S0,F0> const& m, C<S1> s)
{
  if constexpr (S1 >= 0) {
    return m >> s;
  } else {
    return m << -s;
  }
}

template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0,             "MBase must be positive.");
  static_assert(num_bits >= 0,             "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk::value | zzz_msk::value);

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }

  template <int B, int M, int S>
  CUTE_HOST_DEVICE constexpr
  auto
  operator==(Swizzle<B,M,S> const&) const
  {
    return B == BBits && M == MBase && S == SShift;
  }
};
```

对于`Swizzle<3,3,3>`,展开如下:

```
uint32_t apply(uint32_t offset) {
  constexpr uint32_t bit_mask = (1u << 3) - 1u;          // 0b111
  constexpr uint32_t yyy_mask = bit_mask << (3 + 3);     // 0b111000000 (选择 offset 的第 6~8 位)
  constexpr int      shift    = 3;                       // 把 Y 区块右移 3 位
  uint32_t y_part = offset & yyy_mask;                   // 提取 Y 区域
  uint32_t shifted_y = y_part >> shift;                  // 等价于 shiftr(..., C<3>{})
  return offset ^ shifted_y;                             // 把移位后的 Y 区与原 offset 做异或
}

swizzled_offset = (i << 6) + ((j >> 3) ^ i) << 3 + (j & 0b111)。
直观理解：把每行 64 个元素按 8 个元素一组共有 8 组（j_hi = j >> 3），Swizzle 会把组号变成 j_hi ^ i。
同一行内部元素顺序（每组的 8 个 j_lo = j & 0b111）保持不变。
示例（行号 i、列号 j、原地址、Swizzle 后地址）：
i=0: 组号不变，(0,0) → 0, (0,15) → 15。
i=1: 组号按 ^1 互换：(1,0..7) → 72..79（原先是 64..71），(1,8..15) → 64..71。
i=2: 组号按 ^2：(2,0..7) → 144..151（对应第 18 列块），(2,16..23) → 128..135。
i=7: 组号按 ^7，即把 j_hi 与 111b 异或，实现更复杂的块置换。
```

那么：

```
composition(Swizzle<kSwizzle, 3, 3>{},Layout<Shape<_8, Int<kBlockKSmem>>,Stride<Int<kBlockKSmem>, _1>>{}));
```

也就是把

```
Layout<Shape<_8, Int<kBlockKSmem>>,Stride<Int<kBlockKSmem>, _1>>{}
```

计算之后的offset按照上边的Swizzle计算方式重新计算offset，矩阵行列只是根据这个offset计算的逻辑坐标，Swizzle实际操作的是offset。

用图来展示一下这个变化如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/before-swizzle.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. 原始矩阵视图.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/after-swizzle.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. swizzle变化之后的矩阵视图
</div>

为什么需要swizzle？{%sidenote 'eight' '参考[文章](https://mp.weixin.qq.com/s/IpGhpQlQBoE3RVBZqx2o-w)介绍，其中对Bank Conflict详细解释并附有例子。 Bank Conflict发生在同一warp（32个线程）中，两个或更多线程在同一时钟周期内访问同一bank的不同地址。GPU的Shared Memory划分为32个独立bank，每个bank每个时钟周期只能服务一个内存请求。当多个线程同时请求同一bank时，请求被序列化执行，导致性能下降。'%}

GPU的smem 被分为固定数量的 bank（Ampere起通常是32 个），每个 bank 服务固定“地址模”的请求，每个bank是4byte 也就是2个float16。
在一次指令内，如果多个线程访问映射到同一个 bank 的不同地址，就发生冲突（串行化），吞吐骤降。
FlashAttention 的常见访问模式：
Q/K/V tile 在 K 维（head_dim）方向连续布局；
行内线程做 128-bit（16B）向量化搬运（fp16/bf16 时每次 8 个元素），天然倾向“同列段”的相邻访问；
不经处理时，行主序布局会让线程访问周期性地落到相同 bank，产生多路冲突。

以上述的矩阵作为说明例子，

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/before-bank.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. 原始矩阵bank视图.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/after-bank.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. swizzle变化之后的bank矩阵视图
</div>

##### 2. tile_to_shape

`sQ`的大小 `128x128`，`SmemLayoutAtomQ`的shape是`8x64`，怎么才能用原子布局构造`128x128`的布局？ 那其实就是多次重复，如何重复？ 就是这个tile_to_shape的操作了，它将原子布局在矩阵维度上重复，以便构成整个`sQ`。 `sQ`的大小是`128x128`，那么直观上理解应该需要`128/8 x 128/64`次也就是`16x2`次拷贝。

写如下程序来打印一下`sQ`的相关信息

```
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include "cute/tensor.hpp"
#include <thrust/host_vector.h>

using namespace cute;
int main(int argc, const char** argv)
{
    using Element = cutlass::half_t;
    auto const size_a{128 * 128};
    auto h_Q = thrust::host_vector<Element>(size_a);
    auto s_Q = thrust::host_vector<Element>(size_a);

    auto const smem_shape_A{cute::make_shape(128, 128)};
    auto const smem_stride_A{cute::make_stride(cute::Int<1>{}, 128)};
    auto const smem_layout_A{cute::make_layout(smem_shape_A, smem_stride_A)};


    using SmemLayoutAtomQ = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Int<64>>, Stride<Int<64>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<128>, Int<128>>{}));

    Tensor gQ = cute::make_tensor(h_Q.data(), smem_layout_A);
    Tensor sQ = cute::make_tensor(s_Q.data(), SmemLayoutQ{});
    std::cout<<"gQ layout " << std::endl;
    cute::print(gQ.layout());
    std::cout<<std::endl;
    std::cout<<"sQ layout " << std::endl;
    cute::print(sQ.layout());
    std::cout<<std::endl;

    // cute::print_latex(sQ.layout());

    return 0;
}
```

得到如下结果：

```
gQ layout
(128,128):(_1,128)
sQ layout
Sw<3,3,3> o _0 o ((_8,_16),(_64,_2)):((_64,_512),(_1,_8192))
```

这里我们假设了`gQ`的输入是`128x128`的列主序矩阵，得到的`sQ`的layout是

```
Sw<3,3,3> o _0 o ((_8,_16),(_64,_2)):((_64,_512),(_1,_8192))
```

和上面类似，这表示一个复合layout，`Sw<3,3,3>`就是之前的swizzle，

```
((_8,_16),(_64,_2)):((_64,_512),(_1,_8192))
```

表示的tile_to_shape之后的layout， 这个layout的意思是说：

```
这是 CUTE 打印的 sQ 的“分解后布局”：(形状):(步长)。可按分块理解 M×K=(128×128) 被拆成 (8×16) 和 (64×2)。
形状 ((8,_16), (_64,_2)):
M 维被分解成 M0=8、M1=16 → 8×16=128 行
K 维被分解成 K0=64、K1=2 → 64×2=128 列
步长 ((64, 512), (1, 8192)) 表示每个子维增 1 时地址前进的元素数：
M0 步长 64：在同一“8×64 面板”里行内移动，行步长=64（即 kBlockKSmem）
M1 步长 512：跨面板行块移动，= 64×8
K0 步长 1：列内连续存放
K1 步长 8192：跨面板列块移动，= 128×64（整张 128 行 × 每面板列宽 64）
把逻辑坐标 (m,k) 映射到物理偏移的方法（以元素为单位）：
m0 = m % 8, m1 = m / 8
k0 = k % 64, k1 = k / 64
offset = 64xm0 + 512xm1 + 1xk0 + 8192xk1
含义总结：
最内层列（K0）连续，保证向量化/ldmatrix 取数；
行内步长=64 对应 kBlockKSmem；
(8×64) 原子面板按 (16×2) 平铺成 128×128；

(8×64) 原子面板按 (16×2) 平铺成 128×128,为什么是((_8,_16),(_64,_2))
而不是((_16,_2),(_8,_64))
原因在于 CUTE 的约定：tile_to_shape 会把“内层原子面板维度”放在前、“外层平铺次数”放在后。对 sQ：
原子面板 SmemLayoutAtomQ 的形状是 (8, 64) → 内层 M0=8、K0=64
平铺到总形状 (128,128) 时，外层平铺次数是 M1=128/8=16、K1=128/64=2
因此打印为 ((8,_16), (_64,_2))，即 (M0,M1), (K0,K1)
如果写成 ((16,_2), (_8,_64)) 就把“内层/外层”弄反了：会改变步长含义，
破坏 K0 连续（stride=1）和 M0 的行步长=64（=kBlockKSmem）的设计，
从而影响向量化和 bank 冲突优化。

```

还可以将`sQ`的latex打印出来，得到矩阵视图如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/sQNoSwizzle.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. 原始矩阵sQ视图.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/understanding-fa2/sQ.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6. swizzle变化之后的sQ矩阵视图
</div>

对于`K`和`V`的视图来说是同样的道理，就不一一解释了。

#### 显存搬运

我们知道CUDA中的矩阵乘是在MMA中计算的，这个指令读取的都是寄存器的数据，也就是CUDA的显存搬运的路径是：

```
global memory -> shared memory -> register
```

其中 `shared memory -> register`在下一节中继续补充，先看`global memory -> shared memory`部分。

继续看代码，来到了如下的部分：
{%sidenote 'Two' '先忽略TileMMA的部分，下一小结讲。'%}

```
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)


// Repeat the partitioning with identity layouts
Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

// Allocate predicate tensors for k
Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));


// Prologue

// We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                   binfo.actual_seqlen_q - m_block * kBlockM);
if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }
```

其中

```
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
```

的定义如下：

```
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
// kGmemElemsPerLoad         = sizeof(uint128) / sizeof(Element)= 16/2 = 8
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
// kGmemThreadsPerRow = 64 / 8 = 8
static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
// 这是 全局内存加载 Q/K/V 时的线程到元素映射布局，也就是 每个线程应该从 global memory 中加载哪些元素。
using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>,Int<kGmemThreadsPerRow>>,Stride<Int<kGmemThreadsPerRow>, _1>>;
// GmemLayoutAtom = Layout<Shape<16,8>,Stride<8,1>>
// We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
// from the same address by the same threadblock. This is slightly faster.
using Gmem_copy_struct = std::conditional_t<
    Has_cp_async, // true
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
// make_tiled_copy 会生成一个 tiled copy object，描述如何把 global memory tile 拷贝到 shared memory tile（或者相反）
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
```

##### 1. 确定每次加载的数据量

```
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
// 结果: 16 bytes / 2 bytes = 8 个元素
```

使用 128-bit(16字节)的加载指令,这是 GPU 全局内存访问的最优粒度,对于 half/bfloat16 类型(2字节),一次可加载 8 个元素.

##### 2. 计算每行需要的线程数

```
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
// 结果: 64 / 8 = 8 个线程
```

kBlockKSmem 是共享内存中 K 维度的大小(也就是64),每行需要 8 个线程协作,每个线程负责加载 8 个连续元素，8个线程 × 8个元素, 完整覆盖64维的一行数据。

##### 3. 定义线程到数据的映射布局

```
using GmemLayoutAtom = Layout
    Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
    Stride<Int<kGmemThreadsPerRow>, _1>
>;
// 实例化: Layout<Shape<16, 8>, Stride<8, 1>>
```

`Shape<16, 8>`: 16行×8列的线程网格,16行表示同时处理16行数据,8列对应每行的8个线程,`Stride<8, 1>`: 内存访问步长,行方向步长为8(跳过同行的其他线程),列方向步长为1(连续访问)

##### 4. 选择拷贝指令

```
using Gmem_copy_struct = std::conditional_t
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
```

优先使用 cp.async 异步拷贝指令(Ampere架构及以上).
{%sidenote 'Three' '这里选择 CACHEGLOBAL 而不是 CACHEALWAYS，原因是：
Q / K / V 数据 不会被同一个 thread block 重复读取,
没有必要污染 L1 cache,
使用 CACHEGLOBAL 可以减少 cache 管理开销，提升整体带宽利用率
'%}

##### 5. 组合

make_tiled_copy：把一切“组合”起来，创建了Tiled Copy对象

```
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(
        Copy_Atom<Gmem_copy_struct, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{}  // 值布局: 每次读取8个值
    )
);
```

make_tiled_copy 将三件事组合成一个可执行的拷贝策略对象：

- 拷贝指令（cp.async + 128-bit）
- 线程布局,定义多个线程如何协作（GmemLayoutAtom）
- 值布局,每个线程每次读取8个连续值

也就是说tile copy中的thread layout，把一个线程块的拷贝线程组织成二维网格，确保同一行的相邻线程访问相邻地址，从而实现 128-bit 向量化/cp.async 合并访问。thread layout 只定义“线程如何排布”，拷贝多少数据，由 CopyAtom 决定。一个拷贝指令读取128-bit也就是8个值，thread layout的shape是`16x8`，因为设置中总共threads的数量就是128，也就是4个warps， 128个线程布局成了二维网格，也就是`16x8`， 那么这128个线程总共读取的数据就是`16x8x8 = 16x64`个数据， 总共数据大小是`128x128`，需要`128/16 x 128/64 = 8x2`次才能读取完成。

`gmem_thr_copy_QKV`实际上是一个thread-value的layout{%sidenote 'Four' '参考[CUTE中的Thread-Value Layout](https://valdrada.site/AI/2025-12-10-CUTE-Thread-Value-Layout.html)'%}

```
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
```

- get_thread_slice(tidx): 为当前线程(线程ID为tidx)提取其专属的拷贝切片,之后每个线程获得一个"视图",明确知道自己负责处理哪些数据
- partition_S: 分区源(Source)张量,即global memory中的数据
- partition_D: 分区目标(Destination)张量,即shared memory中的位置

也就是说，对于每个线程都可以获取到这个线程我要读取哪几个（8个）元素，除了这个信息，还要知道，为了完成全部的加载需要重复几次，也就是上面提到的`8x2`次。 为了直观的感受，写程序打印部分信息如下：

```
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include "cute/tensor.hpp"
#include <thrust/host_vector.h>

using namespace cute;
int main(int argc, const char** argv)
{
constexpr bool Has_cp_async = true;
using Element = cutlass::half_t;
auto const size_a{128 * 128};
auto h_Q = thrust::host_vector<Element>(size_a);
auto s_Q = thrust::host_vector<Element>(size_a);

auto const smem_shape_A{cute::make_shape(128, 128)};
auto const smem_stride_A{cute::make_stride(cute::Int<1>{}, 128)};
auto const smem_layout_A{cute::make_layout(smem_shape_A, smem_stride_A)};

using GmemLayoutAtom = Layout<Shape <_16, _8>, Stride<_8, _1>>;

using Gmem_copy_struct = std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
// make_tiled_copy 会生成一个 tiled copy object，描述如何把 global memory tile 拷贝到 shared memory tile（或者相反）
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));

using SmemLayoutAtomQ = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Int<64>>, Stride<Int<64>, _1>>{}));
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<128>, Int<128>>{}));

Tensor gQ = cute::make_tensor(h_Q.data(), smem_layout_A);
Tensor sQ = cute::make_tensor(s_Q.data(), SmemLayoutQ{});
std::cout<<"gQ layout " << std::endl;
cute::print(gQ.layout());
std::cout<<std::endl;
std::cout<<"sQ layout " << std::endl;
cute::print(sQ.layout());
std::cout<<std::endl;

// cute::print_latex(sQ.layout());


GmemTiledCopyQKV gmem_tiled_copy_QKV;
cute::print_latex(gmem_tiled_copy_QKV);

auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(10);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);


// 打印 tQgQ 张量信息
std::cout << "tQgQ 张量信息：" << std::endl;
std::cout << "======================" << std::endl;

// 1. 打印形状
std::cout << "形状 (Shape): " << tQgQ.layout() << std::endl;

// 2. 打印步幅
std::cout << "步幅 (Stride): " << tQgQ.layout().stride() << std::endl;

// 3. 打印张量大小
std::cout << "元素数量 (Size): " << tQgQ.size() << std::endl;

// 4. 打印布局模式
std::cout << "布局 (Layout): " << std::endl;
std::cout << "  " << tQgQ.layout() << std::endl;

// 5. 打印张量维度信息
std::cout << "维度数 (Rank): " << rank(tQgQ) << std::endl;
std::cout << "  维度 " << 0 << ": 大小 = " << size<0>(tQgQ) << std::endl;
std::cout << "  维度 " << 1 << ": 大小 = " << size<1>(tQgQ) << std::endl;
std::cout << "  维度 " << 2 << ": 大小 = " << size<2>(tQgQ) << std::endl;
// 6. 打印张量类型信息
std::cout << "数据类型: " << typeid(decltype(tQgQ)::value_type).name() << std::endl;
cute::print_tensor(tQgQ);

// 打印 tQgQ 张量信息
std::cout << "tQsQ 张量信息：" << std::endl;
std::cout << "======================" << std::endl;

// 1. 打印形状
std::cout << "形状 (Shape): " << tQsQ.layout() << std::endl;

// 2. 打印步幅
std::cout << "步幅 (Stride): " << tQsQ.layout().stride() << std::endl;

// 3. 打印张量大小
std::cout << "元素数量 (Size): " << tQsQ.size() << std::endl;

// 4. 打印布局模式
std::cout << "布局 (Layout): " << std::endl;
std::cout << "  " << tQsQ.layout() << std::endl;

// 5. 打印张量维度信息
std::cout << "维度数 (Rank): " << rank(tQsQ) << std::endl;

std::cout << "  维度 " << 0 << ": 大小 = " << size<0>(tQsQ) << std::endl;
std::cout << "  维度 " << 1 << ": 大小 = " << size<1>(tQsQ) << std::endl;
std::cout << "  维度 " << 2 << ": 大小 = " << size<2>(tQsQ) << std::endl;
cute::print_tensor(tQsQ);


// 8. 可选：使用 cute 的 print 函数
std::cout << "使用 cute::print_tensor() 输出:" << std::endl;
cute::print(tQgQ.layout());
std::cout<<std::endl;
cute::print(tQsQ);
std::cout<<std::endl;
auto ppp = tQgQ(_,0,1);
cute::print(ppp);
std::cout<<std::endl;
return 0;
}

```

得到的结果为：

```
tQgQ 张量信息：
======================
形状 (Shape): ((_8,_1),8,2):((128,_0),_16,8192)
步幅 (Stride): ((128,_0),_16,8192)
元素数量 (Size): 128
布局 (Layout):
  ((_8,_1),8,2):((128,_0),_16,8192)
维度数 (Rank): _3
  维度 0: 大小 = _8
  维度 1: 大小 = 8
  维度 2: 大小 = 2
数据类型: N7cutlass6half_tE
tQsQ 张量信息：
======================
形状 (Shape): ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
步幅 (Stride): ((_1,_0),_1024,_8192)
元素数量 (Size): _128
布局 (Layout):
  ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
维度数 (Rank): _3
  维度 0: 大小 = _8
  维度 1: 大小 = _8
  维度 2: 大小 = _2
```

得到这些视图之后，真正的拷贝发生在：

```
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                   binfo.actual_seqlen_q - m_block * kBlockM);

函数定义为：

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
```

可以看到，在`size<1>(S)` 和 `size<2>(S)` 这个维度对拷贝进行了循环操作，也就是上面说的，循环了`8x2`次，来完成整个的`128x128`的矩阵从global memory到shared memory的拷贝工作。

这里面还有一些没有提到的是`cQ`的这个张量，这个的目的是构建 Identity Tensor {%sidenote 'Five' 'Identity Tensor是一种特殊的坐标张量,其值就是自身的索引，例如: cQ(i, j) = (i, j),表示位置 (i, j) 的坐标就是 (i, j)'%}
来判断当前线程是否越界(超出实际序列长度)以及是否需要执行拷贝操作(通过谓词控制)，这对应着这`copy`函数中的`predictate_K`及其他的判断操作。

对于`K` 和 `V`也是同样的理解，不再一一解释。

### MMA

继续看代码，来到了TileMMA部分，这个地方可以参考[博客](https://leimao.github.io/blog/CuTe-Tiled-MMA/)来更深入的理解

```
typename Kernel_traits::TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

Tensor tSgS  = thr_mma.partition_C(gP);

Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

//
// Copy Atom retiling
//

auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
// if (cute::thread0()) {smem_thr_copy_Q.print_all();}
Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
// if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
Tensor tSsK = smem_thr_copy_K.partition_S(sK);

auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
```

#### TiledMMA

这里查看TiledMMa的定义如下：

```

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;

```

##### 1. 最小计算单元：MMA Atom

其中 `MMA_Atom_Arch`是定义了一个原子的矩阵乘法运算，如果元素是`float16`类型则会选择`MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>`， 查看这个原子矩阵乘法的定义为：

```
// MMA 16x8x16 TN
struct SM80_16x8x16_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{\%0,  \%1,  \%2,  \%3},"
      "{\%4,  \%5,  \%6,  \%7},"
      "{\%8,  \%9},"
      "{\%10, \%11, \%12, \%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x16_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

```

这个原子操作实际上是对PTX {%sidenote 'Six' '参考PTX[官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)'%}指令的一个封装，意思是说计算一个`m16n8k16` 也就是`16x16 和 16x8 `的矩阵乘法，`f32.f16.f16.f32`分别表示`D=A*B + C`对应的数据类型，也就是D是`f32`，A和B分别是`f16`， C累加是`f32`, `.row` 表示A是行主序，`.col`表示B是列主序{%sidenote 'Seven' '参考[CUTE中的Thread-Value Layout](https://valdrada.site/AI/2025-12-10-CUTE-Thread-Value-Layout.html)'%}
。

需要注意的是这个PTX是**warp**级别的指令，也就是在warp中完成这个矩阵的乘法，它操作的是**寄存器**数据，**不是shared memory**，所以为了执行这个指令，需要将shared memory数据搬运到寄存器中，然后在执行这个指令。

##### 2. 线程块布局：TiledMMA 的定义

```
using TiledMma = TiledMMA<
    typename Base::MMA_Atom_Arch,          // 基础 Atom: 16x8x16
    Layout<Shape<Int<4>,_1,_1>>,           // Warp 布局 (Thread Block Layout)
    Tile<Int<16 * 4>, _16, _16>>;          // 逻辑 Tile 形状
```

- Atom (16x8x16)：表示单次硬件指令的“颗粒度”。

- Warp Layout (kNWarps x 1 x 1)：这决定了 Warp 是如何排列的。这里 `Shape<kNWarps, _1, _1>` 表示所有 Warp 都在 M 方向（行方向）纵向排列。如果有 4 个 Warp，它们会堆叠成一个更“高”的矩阵块。

- Target Tile (64x16x16)：
  这是该 TiledMMA 期望完成的总任务量（其中 kNWarps = 4）。

##### 3. 映射逻辑：如何从 16x8x16 变成 64x16x16？

- M 方向：
  空间并行计算：$16 \text{ (Atom M)} \times 4 \text{ (Warps)} = 64$。4 个 Warp 在 M 方向并行，每个 Warp 负责自己那 16 行，互不干扰。这对应了 `Layout<Shape<Int<kNWarps>,_1,_1>>`。

- N 方向：时间序列（重复累加）计算：$8 \text{ (Atom N)} \rightarrow 16 \text{ (Tile N)}$。由于我们在 Layout 的 N 方向定义的是 \_1（即 N 方向没有并行的 Warp），但 Tile 的 N 方向是 16。结果：同一个 Warp 需要在 N 方向执行两次运算。CuTe 会自动生成代码，让每个 Warp 先算前 8 列，再算后 8 列，最终合成 16 列的结果。

- K 方向：内积循环计算：$16 \text{ (Atom K)} \rightarrow 16 \text{ (Tile K)}$。这里 K 方向匹配，意味着在一个内层循环迭代中，刚好消耗掉 $16$ 宽度的 $Q$ 和 $K$。

总体上来说：

一个warp 运算一个 `16×8×16` 的 MMA tile（NVIDIA Tensor Core 的硬件 tile,也就是atom的定义）
一个block 有 kNWarps 个 warp，可以并行计算一个更大的 tile，block 级别计算的 `tile = (16*kNWarps) × 16 × 16 = 64x16x16`，也就是有一个矩阵块是64×16×16。我需要用TiledMma 去计算完成
这个TiledMma计算的时候的Warp的分布是`Layout<Shape<Int<kNWarps>,_1,_1>> `，也就是M方向分了4份，N方向1份，K方向1份，但 Atom 只能算 N=8，N=16 是通过两个 MMA Atom 在 N 方向累加完成的，每份都是`16×16×16`，意思是1个warp去计算`16×16×16` ，在N方向需要两次atom操作 。

为了方便线程和值之间的对应，也可以用`cute::print_latex()`打印TiledMMA。参考之前我的[文章](https://valdrada.site/AI/2025-12-10-CUTE-Thread-Value-Layout.html)查看具体的操作过程，对于这个MMA来说结果如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/thread-value-layout/thread-value-p1.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 7. MMA Atom Thread Value Layout.
</div>

#### 寄存器视图

定义好了TiledMMA之后，就可以根据TiledMMA的定义获取当前线程需要的寄存器的视图，也就是：

```
auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);   // (MMA,MMA_M,MMA_K)
Tensor tSrK  = thr_mma.partition_fragment_B(sK);   // (MMA,MMA_N,MMA_K)
Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)
Tensor tSgS  = thr_mma.partition_C(gP);
Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
```

这里的`partition_fragment_A`就是获取矩阵乘中A的寄存器视图，`partition_fragment_B`就是获取矩阵乘中B的寄存器视图.

获取到寄存器视图之后，就需要把显存从 `shared memory` 搬运到 `register`， 那么同样的需要当前线程搬运的`shared memory`视图：

```
auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
// if (cute::thread0()) {smem_thr_copy_Q.print_all();}
Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
// if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
Tensor tSsK = smem_thr_copy_K.partition_S(sK);

auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
```

`make_tiled_copy_A`就是为 MMA 的 A 操作数（这里是 Q）生成从共享内存到寄存器的分片拷贝算子，匹配 tiled_mma 的片段几何与对齐要求。同样的对 B 操作数也是这样操作，得到对应的源`shared memory`视图。

同样的，这里只是划分的各自的视图，真正的拷贝发生在三个地方：

- 显式把 Q 预加载到寄存器（共享 Q/K 的 smem 时）

```
if (Kernel_traits::Share_Q_K_smem) {
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
}
```

- 常规在 GEMM 内按 K-stage 从 smem 加载到寄存器

```
FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );


template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    //这里发生真正的拷贝
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

```

- P·V 这步（gemm_rs）只从 smem 取 B（V），A（P）已在寄存器

```
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    //这里发生拷贝
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
```

#### V的相关解释

代码中有两个`V`相关的视图，一个layout是`SmemLayoutVtransposed`，一个layout是`SmemLayoutVtransposedNoSwizzle`，也就是一个是带Swizzle来防止bank冲突的，一个不带。

```
Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
```

`sVt` 用于真实的物理搬运（从内存读写），而 `sVtNoSwizzle` 用于逻辑上的计算坐标映射。

- 什么时候用的`sVt`？

执行从共享内存到寄存器的搬运 (Smem to Reg Copy)

```
auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
Tensor tOsVt = smem_thr_copy_V.partition_S(sVt); // 注意这里用的是带 Swizzle 的 sVt
```

当调用 `cute::copy(smem_tiled_copy_V, tOsVt, tOrVt) `时，代码会生成 LDSM（Load Shared Memory）或普通的共享内存加载指令。此时，程序必须知道数据在共享内存里的真实物理排布。因为我们在将数据写入共享内存时（Global $\to$ Smem）为了效率使用了 Swizzle，那么在读取时（Smem $\to$ Reg）必须使用同样的 Swizzle 函数来反向定位，才能读到正确的数据，同时享受无 Bank Conflict 的加载。

- 什么时候用 `sVtNoSwizzle`？

```
Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
```

`partition_fragment_B` 的目的是为了在寄存器中分配空间。寄存器是线程私有的，根本不存在物理 Bank 的概念。
CuTe 需要知道的是：对于一个 $16 \times 8 \times 16$ 的 MMA 指令，当前线程逻辑上应该负责矩阵 $V$ 中的哪些坐标（索引）。
如果在这里传入带 Swizzle 的布局，CuTe 的布局推导系统会把地址混淆函数也代入寄存器的索引计算中，这会导致寄存器布局变得极其复杂且错误。因此，在分配寄存器（Fragment）或建立逻辑映射时，必须使用 NoSwizzle。

- 为什么 $V$ 需要 Transposed？

在 FlashAttention 的第二个 GEMM（$O = P \times V$）中：$P$ 是 Score 矩阵（在寄存器中），形状为 $(BlockM, BlockN)$。$V$ 存储在共享内存中，形状通常是 $(BlockN, HeadDim)$。对于 TiledMMA 来说，它期望的 $B$ 矩阵（右矩阵）是按列存储或者符合特定的 Tensor Core 读取顺序。FlashAttention-2 为了优化，通常将 $V$ 在共享内存中以 转置布局（Transposed） 组织。这样在第二个 GEMM 计算时，Tensor Core 可以更高效地读取 $V$ 的数据。

#### Q和K的拷贝

代码中的这两个地方进行了`Q`和`K`的真正的拷贝

```
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);

FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
cute::cp_async_fence();
```

这里需要注意的是[`cute::cp_async_fence()`](https://docs.nvidia.com/cuda/parallel-thread-execution/#:~:text=Data%20Movement%20and%20Conversion%20Instructions%3A%20cp.async.commit_group),这个地方并不会等待拷贝结束。
在 NVIDIA Ampere (SM80) 及之后的架构中，从 Global Memory 到 Shared Memory 的异步拷贝分为三个必经步骤：

- 发出指令 (cp.async)：

  代码：`cute::copy(gmem_tiled_copy, tQgQ, tQsQ);`

  作用：告诉硬件“请帮我把这块数据搬走”，指令发出后立即返回，线程可以继续干别的。

- 划定边界 (cp_async_fence)：

  代码：`cute::cp_async_fence();`

  作用：将之前所有发出的异步拷贝指令“打包”成一个批次（Batch）并提交给硬件调度器。

  核心理解：它就像是在订单列表下面画了一道横线。硬件只有看到这道横线，才知道刚才那些订单是一组的。

- 真正等待 (`cp_async_wait<N>`)：

  代码：`FLASH_NAMESPACE::cp_async_wait<0>();`

  作用：这才是真正的阻塞操作。它告诉线程：“请在这里停下，直到剩下的异步拷贝批次只剩下 N 个为止。” (Wait<0> 表示等待所有批次完成)。

### softmax

#### 循环计算

在 FlashAttention2 的源码中，将 n_block 的循环拆分为 “Masking 阶段”（第一个循环）和 “Standard 阶段”（第二个循环）,第一个循环处理“边缘/边界”块，主要是一些mask的块和序列长度不能被整除的边界块，第二个循环处理“内部/全量”块。这两个部分相差不多，主要是边界和特殊情况的处理，为了方便我们只看第二个部分，也就是全量计算的块。

```
for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );
    if constexpr (Is_softcap){
        FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
    }

    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
        // This cp_async_fence needs to be in the if block, otherwise the synchronization
        // isn't right and we get race conditions.
        cute::cp_async_fence();
    }

    mask.template apply_mask</*Causal_mask=*/false>(
        acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
    );

    softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
    int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    int block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
        Tensor rP_drop = make_fragment_like(rP);
        cute::copy(rP, rP_drop);
        dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
            rP_drop, block_row_idx, block_col_idx, kNWarps
        );
        cute::copy(rP_drop, tSgS);
        tSgS.data() = tSgS.data() + (-kBlockN);
    }
    if (Is_dropout) {
        dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
    }

    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
    FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
}
```

这些代码对应着算法中的这些部分：

$$
\begin{array}{ll} \hline
& &\text { Algorithm } 1 \text { FlashAttention-2 forward pass }\\ \hline

&6: &\quad \text { for } 1 \leq j \leq T_c \text { do }\\
&7: &\qquad \text { Load } \mathbf{K}_j, \mathbf{V}_j \text { from HBM to on-chip SRAM. }\\
&8: &\qquad \text { On chip, compute } \mathbf{S}_i^{(j)}=\mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c} \text {. }\\
&9: &\qquad \text { On chip, compute } m_i^{(j)}=\max \left(m_i^{(j-1)}, \operatorname{rowmax}\left(\mathbf{S}_i^{(j)}\right)\right) \in \mathbb{R}^{B_r}, \tilde{\mathbf{P}}_i^{(j)}=\exp \left(\mathbf{S}_i^{(j)}-m_i^{(j)}\right) \in \mathbb{R}^{B_r \times B_c} \\
& &\qquad \text { (pointwise), } \ell_i^{(j)}=e^{m_i^{j-1}-m_i^{(j)}} \ell_i^{(j-1)}+\operatorname{rowsum}\left(\tilde{\mathbf{P}}_i^{(j)}\right) \in \mathbb{R}^{B_r} \text {. }\\
&10: &\qquad \text { On chip, compute } \mathbf{O}_i^{(j)}=\operatorname{diag}\left(e^{m_i^{(j-1)}-m_i^{(j)}}\right) \mathbf{O}_i^{(j-1)}+\tilde{\mathbf{P}}_i^{(j)} \mathbf{V}_j \text {. }\\
&11: &\quad  \text { end for }\\
\hline
\end{array}
$$

对应关系如下：

| 算法步骤 (Algorithm 2)                                   | 源码中的代码段 (FlashAttention-2)                   | 功能说明                                                                                         |
| :------------------------------------------------------- | :-------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **Line 6:** `for 1 <= j <= Tc`                           | `for (; n_block >= n_block_min; --n_block)`         | **分块迭代**：遍历 $K, V$ 的 Tile。源码采用从后往前的反向遍历以优化寄存器使用。                  |
| **Line 8:** $S_i^{(j)} = Q_i K_j^T$                      | `FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, ...)`     | **第一步 GEMM**：计算当前 Block 的注意力原始分数，结果存入寄存器 `acc_s`。                       |
| **Line 9:** 更新 $m_i^{(j)}, \ell_i^{(j)}$               | `softmax.template softmax_rescale_o(...)`           | **核心逻辑**：执行 Online Softmax。更新最大值 $m$ 和累加和 $\ell$，并对旧的 $O$ 进行指数重缩放。 |
| **Line 10:** $O_i^{(j)} = \dots + \tilde{P}_i^{(j)} V_j$ | `FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, ...)` | **第二步 GEMM**：将当前块计算出的概率 $P$ 与 $V$ 相乘，累加到已缩放的 $O$ (`acc_o`) 上。         |

这里需要注意的是先加载`V`，然后计算`Q、K`，然后在加载下一个`K`，这是流水线的并行，将拷贝和计算并行。

#### softmax计算过程

下面的函数完成了softmax的计算并完成了对输出矩阵`O`的累加。

```
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) {
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
        FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
        FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_max_cur = !Check_inf
                ? row_max(mi)
                : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            row_sum(mi) *= scores_scale;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
        // We don't do the reduce across threads here since we don't need to use the row_sum.
        // We do that reduce at the end when we need to normalize the softmax.
        FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
};
```

这里实际上是[online softmax](https://valdrada.site/AI/2025-04-26-rethinking-attention-2.html)的计算过程，流程解释如下：

- 寄存器布局重排 (Data Re-layout)

  `acc_s` 是 Tensor Core 算出来的寄存器片段，其布局（Layout）是为矩阵乘法优化的。为了方便按“行”计算 Max 和 Sum，这里将其重映射为逻辑上的`(row, col)`布局。

- Online Softmax 动态更新流程

  1. 缓存旧的`row_max`

  ```
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
  ```

  2. 更新全局最大值 计算当前块的行最大值，并与旧最大值比较，更新得到全域最大值

  ```
      reduce_max(scores, row_max);
  ```

  3. 核心重缩放 (Rescale) 这是 Online Softmax 的灵魂。如果出现了更大的 $max$，之前累加在 row*sum 和 acc_o 中的结果（它们是基于 $m*{old}$ 计算的）就需要通过缩放因子进行修正：

  ```
      // 计算缩放因子：exp(m_old - m_new)
      scores_scale = exp2((scores_max_prev - row_max) * softmax_scale_log2);
      row_sum *= scores_scale; // 修正旧的分母
      acc_o *= scores_scale;   // 修正旧的分子累加和 (Output)
  ```

  4. 处理当前块并累加 计算当前块的指数项，并累加到分母中。

  ```
      scale_apply_exp2(scores, row_max, softmax_scale_log2); // 对当前分块计算 exp(S - m_new)
      reduce_sum(scores, row_sum); // 将当前块的 sum 加进总分母 row_sum
  ```

- 硬件优化：

  为什么使用 $2^x$ 而非 $e^x$？在源码中，所有的指数运算都指向了 exp2（即 $2^x$），这是基于硬件性能的深度考量：

  换底公式的数学支撑：利用对数恒等式 $e = 2^{\log_2 e}$，我们可以推导出：$$e^{x \cdot \text{scale}} = (2^{\log_2 e})^{x \cdot \text{scale}} = 2^{x \cdot (\text{scale} \cdot \log_2 e)}$$

  硬件指令加速：NVIDIA GPU 的 SFU (Special Function Unit) 对 $2^x$ 指令（如 MUFU.EX2）有原生硬件支持，执行效率远高于通用的 $e^x$。

  预计算优化：FlashAttention 在 Host 端预先计算好 $softmax\\_scale\\_log2 = softmax\\_scale \cdot \log_2 e$，并将其传入 Kernel。这样在 Kernel 内部只需一次乘法和一次 exp2 即可完成等效于自然指数的运算，极大提升了指令吞吐。

- 最终的计算

  之前的步骤只更新了online softmax中的分子部分，没有除上分母。分母部分的计算融合在了函数：

  ```
  template<bool Is_dropout=false, bool Split=false, typename Tensor0>
  __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
      SumOp<float> sum_op;
      quad_allreduce_(row_sum, row_sum, sum_op);
      TensorT lse = make_fragment_like(row_sum);
      Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
      static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
      #pragma unroll
      for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
          float sum = row_sum(mi);
          float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
          lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
          float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
          #pragma unroll
          for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
      }
      return lse;
  };
  ```

### 最终回写

整个计算流程基本上结束了，现在需要把`O`矩阵写回全局内存。

- 数据转换

  主循环中的计算为了保证精度使用的是 FP32。但在写回显存之前，需要将其转换为模型定义的精度（如 FP16 或 BF16），以节省带宽和空间。

  ```
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  ```

- 寄存器到共享显存（R2S）：布局重组

  为了实现高效的合并写回（Coalesced Write），通常不直接从寄存器写到全局显存，而是先写到共享显存（Smem）做中转。

  ```
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  // ... 布局重排 (Retile & Partition)
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
  ```

  注意 sO 使用了 sQ.data()。因为计算结束了，$Q$ 已经没用了，所以直接复用 $Q$ 的共享显存空间来存 $O$，完全不占额外空间。通过 `smem_tiled_copy_O `将寄存器中碎片的布局重新排列成 Smem 中连续的布局。

- 共享显存到寄存器（S2R）：向量化对齐

  ```
  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)
  Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  ```

  这里又将共享内存拷贝到了寄存器，为什么？

  全局显存写回最高效的方式是使用 128-bit 向量化指令（STG.128）。这要求每个线程持有的寄存器数据在物理地址上必须是绝对连续且对齐的。

  但是写回 gmem 的拷贝原子（GmemTiledCopyO）期望的是寄存器片段布局，而 sO 的共享内存布局是为 MMA/访存优化的（可能带 swizzle/不同步长）。直接从 sO 写 gmem 不能保证：

  布局匹配：gmem 写回要求连续/对齐的寄存器向量；
  向量化与对齐：每线程 128‑bit 写出需要寄存器中的连续向量；
  谓词与边界处理：FLASH_NAMESPACE::copy 使用寄存器片段和谓词来屏蔽越界。

  因此先把 O 写入 sO（布局重排/暂存），再把 sO “装回寄存器”形成 tOrO，最后用 gmem 拷贝原子高效写回。这样实现“计算布局”和“写回布局”的解耦。

  上一步寄存器到共享内存搬运的意义是做一次共享内存“重排/落地”作为写回缓冲：
  taccOrO 是 MMA 寄存器片段布局，不适合直接按 gmem 线性/向量化方式写出。
  taccOsO 的 SmemLayoutO 是为写回准备的布局（对齐、连续性更好），先把寄存器结果“按正确布局”排到 taccOsO。

  随后再从 taccOsO 装回寄存器形成 tOrO，配合 GmemTiledCopyO 做 128‑bit 合并写入并处理边界谓词。

- 将输出 $O$ 写回全局显存 (Writing O to Gmem)

  ```
  // 处理 K 维度的 Padding
  if (!Is_even_K) {
      #pragma unroll
      for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
  }

  // 最终写回
  FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
  );
  ```

### 总结

整体上整个FlashAttention2的源码到这就解读完了，其实还有其他的分支情况，比如各种mask和各种特殊情况，但是理解了full attention之后其他的部分应该就比较清晰了。

最主要的是如果有任何一个地方不理解，打印出来直观看看数据是如何排布应该就会清晰了。
