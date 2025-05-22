---
layout: post
title: Attention系列整理Part2-FlashAttention
date: 2025-04-26 12:42:00
description: 整理FlashAttention系列以及其计算原理
tags: attention
categories: AI
pretty_table: true
toc:
  beginning: true
---

接着[上篇](<[AI/2025-04-02-rethinking-attention-1.html](https://valdrada.site/AI/2025-04-02-rethinking-attention-1.html)>)对Attention的基础分析之后，继续来整理一下FlashAttention系列。

# FlashAttention1

## 为什么？

要介绍为什么，也就是解释FlashAttention1解决了什么问题？

我们知道Attention的计算如下：

$$
\mathbf{S}=\mathbf{Q} \mathbf{K}^{\top} \in \mathbb{R}^{N \times N}, \quad \mathbf{P}=\operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O}=\mathbf{P V} \in \mathbb{R}^{N \times d},
$$

{% marginfigure 'mf-id-1' 'assets/img/AI/attention2/gpu.png' 'GPU中显存结构和带宽示意图'%}

对于中间计算矩阵$$\mathbf{S}$$， 它的维度是$${N \times N}$$，这个矩阵维度很大，存储在HBM中需要$$\mathbf{O}(N^2)$$的显存，例如对GPT2来说$$N = 1024，d = 64$$，假设是float16计算，需要的显存是$$1024 \times 1024 \times sizeof(float16) = 2 \mathbf{M}$$，如果$$N = 102400$$ 也就是100K的长度，占用的显存就是$$102400 \times 102400 \times sizeof(float16) = 20 \mathbf{G}$$。这在长序列中显然带来显存上的瓶颈。

原始的标准Attention的计算方式如下：

$$
\begin{array}{l}
\hline
&\text { Algorithm } 0 \text { Standard Attention Implementation }\\ \hline
&\text { Require: Matrices } \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d} \text { in } \mathrm{HBM} \text {. }\\ \hline
&\text { Load } \mathbf{Q}, \mathbf{K} \text { by blocks from HBM, compute } \mathbf{S}=\mathbf{Q} \mathbf{K}^{\top} \text {, write } \mathbf{S} \text { to HBM. }\\ \hline
&\text { Read } \mathbf{S} \text { from HBM, compute } \mathbf{P}=\operatorname{softmax}(\mathbf{S}) \text {, write } \mathbf{P} \text { to HBM. }\\ \hline
&\text { Load } \mathbf{P} \text { and } \mathbf{V} \text { by blocks from HBM, compute } \mathbf{O}=\mathbf{P V} \text {, write } \mathbf{O} \text { to HBM. }\\ \hline
&\text { Return } 0 . \\ \hline
\end{array}
$$

从算法中可以看出，原始的Attention算法需要将完整的Q和K矩阵从HBM中分块读取到SM上进行矩阵乘法。计算得到的中间结果S被完整写回到HBM，再次从HBM读取S矩阵，对每一行应用softmax操作得到P矩阵，然后再次将P写回HBM。从HBM中读取P和V，进行矩阵乘法，得到最终的输出O。O也是写回HBM。由于每一步都需要从HBM读入矩阵数据、计算后再写回，造成了大量的内存带宽消耗。这在实际硬件上往往成为Attention推理或训练的性能瓶颈。

正是为了缓解上述两个问题，FlashAttention1提出了一种改进的Attention计算方式，通过块状处理（tiling），避免显存中存储完整的S或P矩阵，大幅降低了内存占用和带宽压力，从而实现了更高效的Attention计算。

## 是什么？

$$
\begin{array}{ll}
\hline
& &\text { Algorithm } 1 \text { FlashAttention }\\ \hline
& &\text { Require: Matrices } \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d} \text { in HBM, on-chip SRAM of size } M \text {. }\\
&1: &\text { Set block sizes } B_c=\left\lceil\frac{M}{4 d}\right\rceil, B_r=\min \left(\left\lceil\frac{M}{4 d}\right\rceil, d\right) \text {. }\\
&2: &\text { Initialize } \mathbf{O}=(0)_{N \times d} \in \mathbb{R}^{N \times d}, \ell=(0)_N \in \mathbb{R}^N, m=(-\infty)_N \in \mathbb{R}^N \text { in HBM. }\\
&3: &\text { Divide } \mathbf{Q} \text { into } T_r=\left\lceil\frac{N}{B_r}\right\rceil \text { blocks } \mathbf{Q}_1, \ldots, \mathbf{Q}_{T_r} \text { of size } B_r \times d \text { each, and divide } \mathbf{K}, \mathbf{V} \text { in to } T_c=\left\lceil\frac{N}{B_c}\right\rceil \text { blocks }\\
& & \mathbf{K}_1, \ldots, \mathbf{K}_{T_c} \text { and } \mathbf{V}_1, \ldots, \mathbf{V}_{T_c} \text {, of size } B_c \times d \text { each. }\\
&4: &\text { Divide } \mathbf{O} \text { into } T_r \text { blocks } \mathbf{O}_i, \ldots, \mathbf{O}_{T_r} \text { of size } B_r \times d \text { each, divide } \ell \text { into } T_r \text { blocks } \ell_i, \ldots, \ell_{T_r} \text { of size } B_r \text { each, }\\
& & \text { divide } m \text { into } T_r \text { blocks } m_1, \ldots, m_{T_r} \text { of size } B_r \text { each. }\\
&5: &\text { for } 1 \leq j \leq T_c \text { do }\\
&6: &\quad \text { Load } \mathbf{K}_j, \mathbf{V}_j \text { from HBM to on-chip SRAM. }\\
&7: &\quad \text { for } 1 \leq i \leq T_r \text { do }\\
&8: &\qquad\text { Load } \mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i \text { from HBM to on-chip SRAM. }\\
&9: &\qquad\text { On chip, compute } \mathbf{S}_{i j}=\mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c} \text {. }\\
&10: &\qquad\text { On chip, compute } \tilde{m}_{i j}=\operatorname{rowmax}\left(\mathbf{S}_{i j}\right) \in \mathbb{R}^{\boldsymbol{B}_r}, \tilde{\mathbf{P}}_{i j}=\exp \left(\mathbf{S}_{i j}-\tilde{m}_{i j}\right) \in \mathbb{R}^{\boldsymbol{B}_r \times \boldsymbol{B}_{\boldsymbol{C}}} \text { (pointwise), } \tilde{\ell}_{i j}= \text { rowsum }\left(\tilde{\mathbf{P}}_{i j}\right) \in \mathbb{R}^{B_r} \text {. }\\
&11: &\qquad\text { On chip, compute } m_i^{\text {new }}=\max \left(m_i, \tilde{m}_{i j}\right) \in \mathbb{R}^{B_r}, \ell_i^{\text {new }}=e^{m_i-m_i^{\text {new }}} \ell_i+e^{\tilde{m}_{i j}-m_i^{\text {new }}} \tilde{\ell}_{i j} \in \mathbb{R}^{B_r} \text {. }\\
&12: &\qquad\text { Write } \mathbf{O}_i \leftarrow \operatorname{diag}\left(\ell_i^{\text {new }}\right)^{-1}\left(\operatorname{diag}\left(\ell_i\right) e^{m_i-m_i^{\text {new }}} \mathbf{O}_i+e^{\tilde{m}_{i j}-m_i^{\text {new }}} \tilde{\mathbf{P}}_{i j} \mathbf{V}_j\right) \text { to HBM. }\\
&13: &\qquad\text { Write } \ell_i \leftarrow \ell_i^{\text {new }}, m_i \leftarrow m_i^{\text {new }} \text { to HBM. }\\
&14: &\quad\text { end for }\\
&15: &\text { end for }\\
&16: &\text { Return } \mathbf{0} \text {. }\\
\hline
\end{array}
$$

我在一开始看这个算法的时候，感觉不是很直观，大概知道它在做一些事情，但是不能够完全明白。实际上在原论文中把这部分的贡献叫做Kernel Fusion，也就是算子融合，核心是通过矩阵分块和算子融合，来减少显存和带宽的压力。其中有两个部分的关键点，一个是online softmax， 一个是矩阵的tiling，下面我们会一一详细解释。

## 怎么做？

### online softmax

如果采用矩阵分块，首先会想到，Attention的softmax是如何操作，因为矩阵分块之后，得到的$$\mathbf{S}=\mathbf{Q} \mathbf{K}^{\top}$$ 应该也是分块的，而softmax是按行进行操作的，如何正确的处理softmax操作，这个就是online softmax{%sidenote 'One' 'see [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf/)'%}算法解决的问题。

先看softmax的操作：

对于向量 $$\mathbf{x} = (x_0,x_1,\ldots, x_n) $$, 令$$\mathbf{m} = max(x_0,x_1,\ldots, x_n) $$, 那么:

$$softmax(\mathbf{x}) = \frac{(e^{x_0-m},e^{x_1-m},\ldots, e^{x_n-m})}{ \sum{(e^{x_0-m},e^{x_1-m},\ldots, e^{x_n-m})} }$$

采用论文中的标记就是:

$$
m(x):=\max _i \quad x_i, \quad f(x):=\left[\begin{array}{lll}
e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}
\end{array}\right], \quad \ell(x):=\sum_i f(x)_i, \quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)} .
$$

对于softmax过程中，影响分块的其实是分母上的和，它需要计算整个向量，假设对于$$\ell = \sum_{j=1}^{N} { e^{x_j - m} } $$,
那么有如下递推关系：

$$
\begin{array}{l}
\ell_i = \sum_{j=1}^{i} { e^{x_j - m_i} } \\

\ell_{i+1} = \sum_{j=1}^{i+1} { e^{x_j - m_{i+1}} } \\

\ell_{i+1} = \sum_{j=1}^{i} { e^{x_j - m_{i+1}} } + e^{x_{i+1} - m_{i+1}} \\

\ell_{i+1} = \sum_{j=1}^{i} { e^{x_j - m_{i} + m_{i} - m_{i+1}} } + e^{x_{i+1} - m_{i+1}} \\

\ell_{i+1} = e^{m_{i} - m_{i+1}} \sum_{j=1}^{i} { e^{x_j - m_{i} } } + e^{x_{i+1} - m_{i+1}} \\

\ell_{i+1} = e^{m_{i} - m_{i+1}} \ell_{i} +   e^{x_{i+1} - m_{i+1}} \\

\end{array}
$$

根据这个递推关系可以知道，最终的分母上的和可以由开始的时候递推获取，不需要等到所有的数据准备好在计算。根据这个递推关系，很容易知道，对于分块的向量：

$$

\begin{array}{l}
& m(x) = m\left(\left[x^{(1)} \ x^{(2)}\right]\right) = \max(m(x^{(1)}), m(x^{(2)})) \\
& f(x) = \left[e^{m(x^{(1)}) - m(x)} f(x^{(1)}) \quad e^{m(x^{(2)}) - m(x)} f(x^{(2)})\right] \\
& \ell(x) = \ell\left([x^{(1)} \ x^{(2)}]\right) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)}) \\
& \text{softmax}(x) = \frac{f(x)}{\ell(x)}.
\end{array}


$$

因此对于分块的矩阵或分块向量，只需要记录和更新对应的$$(m{(x)},\ell{(x)})$$,就能正确的计算最终的结果。

### tiling

理解了online softmax之后，就可以详细的解释flash attention的算法了。
首先对QKV三个矩阵，沿着序列的方向进行分块，外循环是KV矩阵，内循环是Q矩阵，也就是先load KV，然后在当前的KV下，遍历所有的Q去计算。此时在内循环的一个计算中，会计算QK的乘法，然后对这个块矩阵进行online softmax，记录局部$$m$$和局部$$\ell$$, 还要计算O，需要注意的是此时计算的O是其中的一部分，每次内循环都需要把之前的O矩阵读入，然后累加，在内循环结束之后一个O的块才被全部计算完成。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/attention2/flashattention1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. flash-attention1.
</div>

### 复杂度

论文中详细的提到了计算的复杂度的几个计算，下面按照论文中的方式记录一下：

- FlashAttention1的FLOPs计算

  先看一个循环内，一个循环计算了$$Q$$和$$K$$的乘法，还有和$$V$$的乘法，以及$$softmax$$操作. 其中$$Q_i$$的维度是$$\left[B_r,d\right]$$, $$K_{i}^{\top}$$的维度是$$\left[B_c,d\right]$$, 因此$$Q_i \times K_{i}^{\top}$$的FLOPs是$$\mathbf{O}(B_rB_cd)$$,同理$$V_{i}$$的维度是$$\left[B_c,d\right]$$，$$P_{ij}$$的维度是$$\left[B_r,B_c\right]$$,因此$$P_{ij} \times V_{i}$$的FLOPs也是$$\mathbf{O}(B_rB_cd)$$。 忽略$$softmax$$的情况下，一个循环内的Flops就是$$\mathbf{O}(B_rB_cd)$$，总共循环的次数是$$T_rT_c=\left\lceil\frac{N}{B_r}\right\rceil \left\lceil\frac{N}{B_c}\right\rceil $$,所以总的FLOPs就是：

  $$
  \mathbf{O}(\frac{N^2}{B_rB_c} B_rB_cd) = \mathbf{O}(N^2d)
  $$

- 原始的HBM操作次数

  1. 对于$$Q,K,V \in \mathbf{R}^{N \times d}$$， 计算$$\mathbf{S} = Q K^{\top}$$，会把$$Q$$ 和 $$K$$ 从HBM载入，把$$S \in \mathbf{R}^{N \times N}$$写出, 因此总共需要$$\mathbf{O}{(Nd+N^2)}$$次HBM读写

  2. 对于$$P = softmax(S)$$, 需要把$$S$$载入，然后把$$P$$写出，总共需要$$\mathbf{O}{(N^2)}$$次HBM读写

  3. 对于$$\mathbf{O} = PV$$,需要把$$PV$$载入，把$$O$$写出，总共需要$$\mathbf{O}{(Nd+N^2)}$$次HBM读写

  所以整体的HBM的读写次数是$$\mathbf{O}{(Nd+N^2)}$$

- FlashAttention1的HBM操作次数

  1. 对内循环一次读取1个$$Q_i$$和1个$$O_i$$，同时写出1次$$O_i$$， 其中$$Q_i$$和$$O_i$$的大小都是$$B_r \times d$$,因此一次
     内循环读写次数是$$3*B_r \times d$$, 总共的内循环此时是$$T_r$$，也就是一次外循环的读写次数是$$3*B_r \times d \times T_r$$ = $$ \mathbf{O}(N \times d)$$

  2. 每次外循环需要载入$$K_i$$和1个$$V_i$$,其中$$K_i$$和$$V_i$$的大小都是$$B_c \times d$$,总共循环$$T_c$$次，因此总共的读
     写次数是$$B_c \times d \times T_c$$ = $$N \times d$$

  3. 总共外循环$$T_c$$次，总共的读写次数是 外循环的$$N \times d$$次$$KV$$读写加上 $$T_c \times N \times d$$次内循环的读
     写，总共$$\mathbf{O}{(N \times d + T_c \times N \times d)}$$, 也就是$$\mathbf{O}{(T_c \times N \times d)}$$次

  4. $$T_c$$怎么计算？ 对于SRAM的大小$$M$$，需要满足 $$B_c d = \mathbf{O}(M) $$,也就是$$B_c = \mathbf{O}(\frac{M}{d})
     $$ , $$T_c = \frac{N}{B_c} = \frac{Nd}{M}$$

  5. 因此FlashAttention1的HBM读写次数是 $$\mathbf{O}{(T_c \times N \times d)}$$ = $$\mathbf{O}{(\frac{Nd}{M} \times N
    \times d)}$$ = $$\mathbf{O}{(\frac{N^2d^2}{M})}$$

### questions？

在读这个论文的时候有个疑问，就是算法中的分块大小$$B_c=\left\lceil\frac{M}{4 d}\right\rceil, B_r=\min \left(\left\lceil\frac{M}{4 d}\right\rceil, d\right)$$ 是怎么推导来的？

一开始的想法是kernel中同时在shared memory存在$$Q,K,V,O,S$$这几个矩阵，假设不考虑$$S$$，那么一共4个矩阵，假设分块大小是$$B$$,那么 $$4 \times B \times d \leq M $$就能得到$$B \leq \frac{M}{4d}$$， 但这样有点勉强，因为$$M$$是shared memory的大小，是字节数，公式应该是$$4 \times B \times d \times sizeof(dtype) \leq M $$,假设是float16推理，那么$$sizeof(float16) = 2$$,得到的结果应该是$$B \leq \frac{M}{8d}$$。 但无论怎么说都是不精确的，感觉是个经验值，问了deepseek也是说是一个经验值，不是一个严格的数学推导。

# FlashAttention2

## 为什么？

那么FlashAttention2又是解决了什么问题呢？

简单的说就是改进了FlashAttention1能够使Attention计算更快。论文中提到了三个点，最终使Attention快了2到3倍。这三个改进点分别是：

- 改进FlashAttention1算法，降低FlashAttention1中的非矩阵乘法的计算量
- 给出如何在不同的线程块上并行计算以充分使用GPU资源
- 描述了如何将一个线程块中的负载拆分在不同warp之上，以减少共享内存访问的量

下面我们分别介绍。

## 是什么？

$$
\begin{array}{ll} \hline
& &\text { Algorithm } 1 \text { FlashAttention-2 forward pass }\\ \hline
& &\text { Require: Matrices } \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d} \text { in HBM, block sizes } B_c, B_r \text {. }\\
&1: &\text { Divide } \mathbf{Q} \text { into } T_r=\left\lceil\frac{N}{B_r}\right\rceil \text { blocks } \mathbf{Q}_1, \ldots, \mathbf{Q}_{T_r} \text { of size } B_r \times d \text { each, and divide } \mathbf{K}, \mathbf{V} \text { in to } T_c=\left\lceil\frac{N}{B_c}\right\rceil \text { blocks } \\
& & \mathbf{K}_1, \ldots, \mathbf{K}_{T_c} \text { and } \mathbf{V}_1, \ldots, \mathbf{V}_{T_c} \text {, of size } B_c \times d \text { each. }\\
&2: &\text { Divide the output } \mathbf{O} \in \mathbb{R}^{N \times d} \text { into } T_r \text { blocks } \mathbf{O}_i, \ldots, \mathbf{O}_{T_r} \text { of size } B_r \times d \text { each, and divide the logsumexp } L \\
& & \text { into } T_r \text { blocks } L_i, \ldots, L_{T_r} \text { of size } B_r \text { each. }\\
&3: &\text { for } 1 \leq i \leq T_r \text { do }\\
&4: & \quad \text { Load } \mathbf{Q}_i \text { from HBM to on-chip SRAM. }\\
&5: & \quad \text { On chip, initialize } \mathbf{O}_i^{(0)}=(0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}, \ell_i^{(0)}=(0)_{B_r} \in \mathbb{R}^{B_r}, m_i^{(0)}=(-\infty)_{B_r} \in \mathbb{R}^{B_r} \text {. }\\
&6: &\quad \text { for } 1 \leq j \leq T_c \text { do }\\
&7: &\qquad \text { Load } \mathbf{K}_j, \mathbf{V}_j \text { from HBM to on-chip SRAM. }\\
&8: &\qquad \text { On chip, compute } \mathbf{S}_i^{(j)}=\mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c} \text {. }\\
&9: &\qquad \text { On chip, compute } m_i^{(j)}=\max \left(m_i^{(j-1)}, \operatorname{rowmax}\left(\mathbf{S}_i^{(j)}\right)\right) \in \mathbb{R}^{B_r}, \tilde{\mathbf{P}}_i^{(j)}=\exp \left(\mathbf{S}_i^{(j)}-m_i^{(j)}\right) \in \mathbb{R}^{B_r \times B_c} \\
& &\qquad \text { (pointwise), } \ell_i^{(j)}=e^{m_i^{j-1}-m_i^{(j)}} \ell_i^{(j-1)}+\operatorname{rowsum}\left(\tilde{\mathbf{P}}_i^{(j)}\right) \in \mathbb{R}^{B_r} \text {. }\\
&10: &\qquad \text { On chip, compute } \mathbf{O}_i^{(j)}=\operatorname{diag}\left(e^{m_i^{(j-1)}-m_i^{(j)}}\right) \mathbf{O}_i^{(j-1)}+\tilde{\mathbf{P}}_i^{(j)} \mathbf{V}_j \text {. }\\
&11: &\quad  \text { end for }\\
&12: &\quad \text { On chip, compute } \mathbf{O}_i=\operatorname{diag}\left(\ell_i^{\left(T_c\right)}\right)^{-1} \mathbf{O}_i^{\left(T_c\right)} \text {. }\\
&13: &\quad \text { On chip, compute } L_i=m_i^{\left(T_c\right)}+\log \left(\ell_i^{\left(T_c\right)}\right) \text {. }\\
&14: &\quad \text { Write } \mathbf{O}_i \text { to HBM as the } i \text {-th block of } \mathbf{O} \text {. }\\
&15: &\quad \text { Write } L_i \text { to HBM as the } i \text {-th block of } L \text {. }\\
&16: &\text { end for }\\
&17: &\text { Return the output } \mathbf{O} \text { and the logsumexp } L \text {. } \\ \hline
\end{array}
$$

这个就是FlashAttention2的算法部分(我们这里只关注前向)，我们下面详细介绍一下具体的改进。

## 怎么做？

### 对FlashAttention1算法的改进

FlashAttention2对FlashAttention1的改进主要在非矩阵乘法的部分，意思是不需要存储临时的$$(m{(x)},\ell{(x)})$$。 根据之前推导过的online softmax{%sidenote 'Two' 'see [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf/)'%}, 我们可以知道：

$$
\begin{aligned}
m^{(1)} & =\operatorname{rowmax}\left(\mathbf{S}^{(1)}\right) \in \mathbb{R}^{B_r} \\
\ell^{(1)} & =\operatorname{rowsum}\left(e^{\mathbf{S}^{(1)}-m^{(1)}}\right) \in \mathbb{R}^{B_r} \\
\tilde{\mathbf{O}^{(1)}} & =e^{\mathbf{S}^{(1)}-m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d} \\
m^{(2)} & =\max \left(m^{(1)}, \operatorname{rowmax}\left(\mathbf{S}^{(2)}\right)\right)=m \\
\ell^{(2)} & =e^{m^{(1)}-m^{(2)}} \ell^{(1)}+\operatorname{rowsum}\left(e^{\mathbf{S}^{(2)}-m^{(2)}}\right)=\operatorname{rowsum}\left(e^{\mathbf{S}^{(1)}-m}\right)+\operatorname{rowsum}\left(e^{\mathbf{S}^{(2)}-m}\right)=\ell \\
\tilde{\mathbf{P}}^{(2)} & =\operatorname{diag}\left(\ell^{(2)}\right)^{-1} e^{\mathbf{S}^{(2)}-m^{(2)}} \\
\tilde{\mathbf{O}}^{(2)} & =\operatorname{diag}\left(e^{m^{(1)}-m^{(2)}}\right) \tilde{\mathbf{O}}^{(1)}+e^{\mathbf{S}^{(2)}-m^{(2)}} \mathbf{V}^{(2)}=e^{s^{(1)}-m} \mathbf{V}^{(1)}+e^{s^{(2)}-m} \mathbf{V}^{(2)} \\
\mathbf{O}^{(2)} & =\operatorname{diag}\left(\ell^{(2)}\right)^{-1} \tilde{\mathbf{O}}^{(2)}=\mathbf{O} .
\end{aligned}
$$

也就是说根本不需要记录$$(m{(x)},\ell{(x)})$$， 在一次循环中就可以解决，如果要这样做，就需要循环的时候按输出矩阵的行计算，也就是，相比FlashAttention1来说，由KV作为外循环变成由Q变成外循环，这样的更改的意思是，在一次循环$$Q$$的时候，就可以计算一行输出矩阵，这样也同时减少了对输出矩阵$$\mathbf{O}$$的多次读写。流程图如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/attention2/flashattention2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. flash-attention2.
</div>

也就是说对于FlashAttention2的前向计算来说，主要的更改就是更改了$$Q$$和$$K,V$$的循环顺序，这样的更改可以更好的利用online softmax的特性，不需要存储$$(m{(x)},\ell{(x)})$$，也不需要多次读写$$\mathbf{O}$$矩阵。

### 并行

这个更改点是指在FlashAttention1上并行的时候是对batch 和 num_heads维度进行并行，但这样对SM的利用率不高，FlashAttention2中改成了对batch 和 num_heads维度还有seq维度进行并行，增加了SM利用率。

### Work Partitioning Between Warps

FlashAttention2将Q分成4个warp，同时使所有warp都可以访问K和V。在每个warp执行矩阵乘以获得$$QK$$的矩阵之后，他们只需要乘以共享$$V$$即可获得相应的输出对应的块。warp之间不需要通信，因此会减少共享内存读取/写入进而产生加速。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/attention2/warps.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. warps between flash attention1 and flash attention2.
</div>

这个部分对我来说，一开始是完全不明白的。后来明白，对于这个其实就是对应的你是如何去抛thread block的，把$$QK$$分成4个warp也就是每个warp都只加载需要计算的那部分的$$QK$$块，$$KV$$则是对所有warp都共享的，实际上我理解可以对应算法中的内外循环的不同。

# FlashAttention3

## 为什么？

## 是什么？

## 怎么做？
