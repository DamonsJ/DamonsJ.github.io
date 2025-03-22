---
layout: post
title: ByteTransformer源码解读
date: 2023-09-14 22:21:00
description: 记录学习ByteTransformer过程中的代码释疑
tags: 源码解读
categories: AI
pretty_table: true
toc:
  sidebar: left
---

# 一、 简介

ByteTransformer库是字节开源的针对Bert模型的推理库，其中没有读入模型的部分，只有完整的执行逻辑。花了一些时间看懂了ByteTransformer中的非cutlass部分的源码，记录一下整个[ByteTransformer](https://github.com/bytedance/ByteTransformer)的整个执行流程。建议先读一遍[ByteTransformer论文](https://arxiv.org/abs/2210.03052) 会对理解代码有很大的帮助。

# 二、 Bert模型的结构

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/bert.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

上图是Bert整个模型的结构，对于B个batch的输入，假设max seq len是S，那么B$\times$S个输入词，经过embedding之后，得到了一个$B\times S \times H$个矩阵，其中$H = head\_num \times size\_per\_head$是Hidden Dim。这个输入矩阵首先经过一个线性层会得到QKV三个矩阵，然后就是进入self-attention块，之后经过残差层和layernorm层，然后进入FFN块， 之后再经过残差层和layer norm得到结果，得到的结果矩阵和输入是同样的大小，都是$B\times S \times H$的矩阵。关于从原始输入句子输入如何经过embedding层得到输入数据，参考下图（假设H=768）：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/embedding.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    embedding
</div>

# 三、ByteTransformer的执行逻辑及代码分析

## 1. ByteTransformer的代码整体流程

如图:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/bytetransformer.process.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

ByteTransformer代码的整体执行流程还是比较清晰的，主流程函数是`bert_infer()`函数，这个函数包含了整个bert推理的各个大的流程，主要分成8个主要的步骤，这几个步骤可以对应在下图中，具体可以参考[ByteTransformer论文](https://arxiv.org/abs/2210.03052)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/bert_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- step1. 如果设置了`remove_padding`标志,处理输入，去掉padding的操作。（`if (is_remove_padding_) {...}`）
- step2. 线性层。 得到QKV矩阵，也就是第一个GEMM操作。（`dense_layer_kernel_launcher`）
- step3. attention层。这个地方计算整个attention块。（`attention_layer_->infer(attention_infer_param)`）
- step4.线性层。attention之后的线性层。（`dense_layer_kernel_launcher`）
- step5.残差和layernorm层。（`add_bias_input_layernorm_kernel_launcher`）
- step6.FFN中的升维层和激活层，融合了GEMM和激活层。（`gemm_bias_gelu`）
- step7.FFN中的降维层，也是个GEMM。（`dense_layer_kernel_launcher`）
- step8.残差和layernorm层。（`if (is_remove_padding_) {...}`）

## 2. 关于remove padding的解释（step 1）

### 1. 为什么会有padding？

​ 对于输入，我们每次给模型的数据有个最大的长度也就是max seq len，也就是输入的一句话中最多有多少个单词，但是每次实际的输入并不是都是最长的，不够的怎么办？一般就会pad成最长，这就是padding的由来。 假设max seq len设置成了5，我们输入了一句话也就是一个batch是： I love you 。那么经过分词每个词都会表达成一个H(hidden dim)维的向量，那么这句话就会表示成3$\times$H个向量（假设是简单的空格分词），因为max seq len是5，也就是需要5$\times$H个向量，我们现在才有3个，剩下的2个需要pad，也就是最终我们会有5$\times$H个向量，最后的2个是pad的向量，一样参与计算但是不会有用就是了。下图展示一个例子，其中Batch是3,根据实际的输入每个batch pad的数据多少都不一样。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/padding.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### 2. removing padding

​ 需要注意的是ByteTransformer输入的矩阵默认就是经过padding之后的，也就是$B \times S \times H$的矩阵。显然padding之后的数据是没有什么用的，但是一样参与了计算，为了减少计算量需要remove padding。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/padding-process.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

​ 怎么样才能remove padding？我们只需要记录目标矩阵和原矩阵之间的行的对应关系就行了，我们需要记录每个batch的实际的seq的长度，以及每个batch中的序列标号的偏移量就行了（也就是prefix sum），参考下图可以进一步理解：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/remove.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

对于给出的例子，有3个batch，也就是B=3，max seq len是5 也就是S=5，第一个batch的有效长度是3，第二个有效长度是2，第三个有效长度是4，在ByteTransformer中分别记录了途中batch seq offset 和 wordidx这两个数据，通过batch seq offset 可以得到每个batch的实际的有效长度，通过wordidx可以知道目标的矩阵的行和原始输入矩阵的行的对应关系。计算这个数据的kernel是`build_sequence_length_padding_offset_kernelLauncher`,我们来看一下具体的实现。这里需要有个前提知识是ByteTransformer计算有效行数，是通过attention mask来计算的。attention mask的大小是$B \times S \times S$，其中每个batch的维度都是$S \times S$，这个矩阵是个mask矩阵，矩阵中的元素是0或者1，值代表当前word对其他word的可见性。如果输入的某一行是pad的也就是无效的，那么对应在mask 矩阵中那一行的数值应该都是0，因为这一行对其他行都是无效的。第一行对其他的有效行总是可见的，所以第一行的非0的值的个数就是这个batch有效的行数。所以ByteTransformer统计每个batch的有效行数是通过计算attention mask的第一行的非0值的个数来计算的，下图的mask只是一个示例：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/mask.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

具体的代码为：

```c++
template <typename T>
void build_sequence_length_padding_offset_kernelLauncher(const T *atten_mask, int *batch_idx,
                                                         int *word_idx, int *valid_word_num,
                                                         const int batch_size,
                                                         const int max_seq_len,
                                                         cudaStream_t stream) {
  // 这里抛线程的逻辑是1个warp计算1个batch的有效行数
  // 因为线程数有1024的限制，如果batch_size * 32超过了1024，那么1个warp可能就处理多个batch
  // 所以总共需要batch个warp，1个warp总共32个线程，所以线程数就是batch*32
  dim3 block(batch_size * 32);  // one warp per sequence
  if (block.x > 1024)
    block.x = 1024;

  //这里的shared memory的大小是(2 * batch_size + 1)个int
  //是因为存储offset 需要 batch+1个（prefix sum）， 中间过程存储有效行数是batch个
  parallel_prefix<<<1, block, (2 * batch_size + 1) * sizeof(int), stream>>>(
      atten_mask, batch_idx, word_idx, batch_size, max_seq_len);
  cudaMemcpyAsync(valid_word_num, batch_idx + batch_size, sizeof(int), cudaMemcpyDeviceToHost,
                  stream);
}
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/prefixsum.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```c++
template <typename T>
__global__ void parallel_prefix(const T *atten_mask, int *batch_idx, int *word_idx,
                                const int batch_size, const int max_seq_len) {
  const int tid = threadIdx.x;
  const int warp_count = blockDim.x >> 5;
  int warp_id = tid >> 5; // 这个是warp的id
  int warp_tid = tid & 0x1F;//这个是线程的id 这个类似对32求余操作

  extern __shared__ int base[];

  int *seq_len = base;
  int *seq_offset = base + batch_size;
  // 因为线程数有1024的限制，如果batch_size * 32超过了1024，那么1个warp可能就处理多个batch
  // 所以需要加上一个循环处理一下这种情况
  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    //同理这个地方是1个warp内的操作，1个warp总共32个线程，1个线程处理1行中的1个元素
    //但是如果这行超过了32个元素，那么1个线程就多处理几个就可以了
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < max_seq_len ? atten_mask[wid * max_seq_len * max_seq_len + i] : (T)0.0f;
      // unsigned __ballot_sync(unsigned mask, int predicate);
      // 意思是返回一个 32 位无符号整数，代表了该warp内变量 predicate 的非零值分布情况
      //（即线程 predicate 为零的该函数返回值该位为 0，线程 predicate 非零的该函数返回值该位为 1 ）
      // 也就是返回一个32位整数 其中如果线程N的predicate是true，那么返回值第N位是1，否则返回值第N位是0
      // __popc这个原语的意思是统计二进制位中1的个数
      // __ballot_sync这个操作是同步的，也就是统计这个warp中32个线程的mask >= (T)0.5f的个数
      //需要注意的是 每次for循环__ballot_sync这个就会重新统计,也就是假设warp_tid=0的时候__ballot_sync统计的是
      //0~31个元素中的mask >= (T)0.5f的个数，warp_tid=32的时候统计的是32~63个元素中的mask >= (T)0.5f的个数
      //count是寄存器的值，每个线程都有个count
      count += __popc(__ballot_sync(0xFFFFFFFF, mask >= (T)0.5f));
    }
    //只在线程0写出就可以了 不用重复写
    if (warp_tid == 0)
      seq_len[wid] = count;
  }
  // 这个地方需要同步，意思是把所有warp计算完成，也就是每个batch的有效word数就统计完成了
  __syncthreads();

  if (warp_id == 0) {
    // 这个是通过seq_len计算prefix sum
    // 假设 seq_len 有3个值 分别是 2，4，5
    // 那么seq_offset分别是0,2,6,11
    // 因为只需要对seq_len计算，所以只需要1个warp就可以了
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? seq_len[i] : 0;
      // 这个是个warp sum操作，需要注意的是，warpPrefixSum这个函数中只统计线程id比warp_tid小的数的和
      // 也就是warp_tid如果是2 那么只累加0 和 1的值就可以了,就是prefix sum
      temp = warpPrefixSum(warp_tid, offset + len);
      if (i < batch_size)
        seq_offset[i] = temp - len;
      // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
      // 表示被 mask 指定的线程返回标号为 srcLane 的线程中的变量 var 的值，其余线程返回0 。类似 broadcast，mask 是参与的线程掩码，如0xffffffff
      // var 是待广播的值，srclane 是被广播的 laneid，warpsize 是参与 warp 大小；
      // 计算完前32个值之后，需要将prefix sum的结果广播到其他的线程，第二次循环的时候会用到这个temp的值
      temp = __shfl_sync(0xffffffff, temp, 31);
    }
    //最后的temp就是总的有效word数
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  __syncthreads();

  // 这个就是把seq len offset 写出
  for (int i = tid; i <= batch_size; i += blockDim.x)
    batch_idx[i] = seq_offset[i];
  // 这个就是把原始矩阵和去掉pad之后的矩阵的seq的id的对应关系 写出
  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    for (int i = warp_tid; i < seq_len[wid]; i += 32)
      word_idx[offset + i] = wid * max_seq_len + i;
  }
}
```

之后的`compressBertInput_kernelLauncher`kernel就简单了，就是把pad的数据给去掉，需要注意的是这个地方其实就是个数据的拷贝，一次写4个float也就是8个half，所以抛线程的时候是H'/4,需要注意这个H'是hidden_dim/2。

## 3. 关于第一个线性层的解释（step 2）

第一个线性层是计算$Input \times QKV_{kernel}$，代码中其实是调用了cublas的gemm，也就是调用了函数[cublasGemmEx](https://docs.nvidia.com/cuda/archive/10.1/cublas/index.html#cublas-GemmEx)，在输入给函数参数中其实是计算了$QKV_{kernel} \times Input$，为什么呢？下图是解释，这里需要注意的是cublas的矩阵是**列主序**的，其实不需要这么算，只要记住内存一定，列主序就是行主序的转置就行了。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/mat_mul.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## 4.关于Attention层的解释（step 3）

attention层是主要的代码部分，也是主要的优化的代码，就像之前说的，attention层有两个主要的优化方法，一部分是cutlass的部分，这部分暂时不看，另一部分就是自己写的kernel，叫做FMA(fused multi attention)，attention层的整体逻辑是:

```c++
virtual void infer(AttentionInferParam infer_param) {
    if (use_fused_attention_) {
      if (infer_param.seq_len <= 80) {
        if (is_remove_padding_)
          fused_rm_infer(infer_param);
        else
          fused_infer(infer_param);
      } else {
        if (is_remove_padding_)
          fused_long_rm_infer(infer_param);
        else
          fused_long_infer(infer_param);
      }
    } else
      nofused_infer(infer_param);
  }

```

这里分成两个部分，一个是fused的kernel，一个是no fused的kernel， fused的就是做了优化的，将attention的几个计算步骤融合到了一个kernel，no fused就是一步步计算。我们先从no fused的开始以便熟悉计算流程。

### 1. no fused kernel源码分析

这部分的代码主要的计算步骤如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/nofused.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### 0.add bias

​ 这一步的kernel是`add_QKV_bias`或者`add_QKV_bias_padding`， 主要的目的有两个： 将bias加上和矩阵数据的转置，如果是`is_remove_padding_`为true，那么`add_QKV_bias_padding`还会把之前去掉的padding的内容还原回来。

​ 这一步的输入是上一步经过GEMM之后的QKV矩阵，输入的矩阵的数据排布如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/qkv.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

为了后续的GEMM的计算，这一步将数据重新排布，排布之后的矩阵数据是 $head\_{num} \times B\times S \times size\_per\_head$, 这个地方抛线程是如下：

```c++
dim3 grid, block;
grid.x = seq_len, grid.y = batch_size;//这个block的方式是按照输出矩阵的维度去安排的，也就是一个block去排布head_num_ *size_per_head这么多数据
block.x = head_num_ *(size_per_head_ / 2);  //这个地方除2是为了一次写出2个数据
```

```c++
//add_QKV_bias： 这个kernel主要理解好src_id和trt_id就可以了，也就是从QKV数据矩阵到转置之后的head_numXBXSXsize_per_head矩阵
int batch_id = blockIdx.y;
int seq_id = blockIdx.x;
int head_id = threadIdx.x / half_size_per_head; // 这个就是head id
int id = threadIdx.x % half_size_per_head;//这个相当于size_per_head的维度的id
int src_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
// batch_id * S * 3H + seq_id * 3H + threadIdx.x 注意threadIdx.x是head_num*size_per_head这个方向的id，看上图的数据排布
int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
// 目标数据的排布是head_numXBXSXsize_per_head
```

这个地方只需要注意一次写2个数据，所以抛线程除了2；这个`add_QKV_bias_padding`的kernel会把去掉的填充数据还原。

#### 1. GEMM $Q \times K^T$

这一步很简单，直接调用了cublas的Batch GEMM, `cublas_Gemm_Strided_Batched`这个函数，注意这个函数是列主序，而且计算的是$Q \times K^T$，所以要去注意函数传入的参数。

#### 2. softmax

这个步骤涉及到三个kernel：

```c++
// 优化分支，如果seq_len是偶数并且是half才进入，要求seq_len是偶数的原因是因为一次要处理2个half数据
if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
    if (seq_len <= 1024) {
     //SOFTMAX_HALF2_REG(xxx) //这个kernel是要利用寄存器来计算和存储中间结果
     //......
    } else {
      if (shmem_size > 48 * 1024)
        cudaFuncSetAttribute(softmax_kernel_warp_half2<half2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
      //这个kernel是要利用shared memory来计算和存储中间结果，其他思路是一样的
      //但是这个分支进不了，因为bytetransformer最大的seq len就是1024，可以对照的学习一下
      softmax_kernel_warp_half2<half2><<<grid, block, shmem_size, stream>>>(
          (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num,
          seq_len);
    }
  } else {
    if (shmem_size > 48 * 1024)
      cudaFuncSetAttribute(softmax_kernel_warp<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           64 * 1024);
    //这个分支是如果seq_len是奇数或者数据是float的时候进入，基本思路和SOFTMAX_HALF2_REG相差不多，对照一起看
    softmax_kernel_warp<T><<<grid, block, shmem_size, stream>>>(qk_buf, atten_bias, atten_mask,
                                                                batch_size, head_num, seq_len);
  }
```

这里我们主要看一下SOFTMAX_HALF2_REG这个kernel就可以了，其他的kernel思路是一样的。

首先是线程抛出的逻辑：

```c++
dim3 grid(batch_size * seq_len), block(32, head_num);
```

也就是总共有batch_size \* seq_len个block，每个block抛出head_num个warp（1个warp32个线程）。需要注意的是这个地方的输入是$Q \times K^T$之后的矩阵，尺寸是$head\_num \times B\times S\times S$,也就是1个warp处理一行seq_len的数据。

然后就是计算了(seq_len + 63) / 64，这个值代表了一行数据有多少个64，为什么是64？，因为1个warp是32个线程，1个线程处理2个half，所以是64，将seq_len行方向按照64分一下。

```c++
//这里的REG_COUNT就是2*多少个64，为什么是乘2，因为一次处理2个half,需要存储这个两个half相应的最大值
//按照是否是64的倍数分开，只有传参true和false的区别，具体原因参考下图
#define SOFTMAX_HALF2_REG(REG_COUNT)                                                            \
  if (seq_len % 64 == 0)                                                                        \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, false>                                 \
        <<<grid, block, 0, stream>>>((half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, \
                                     batch_size, head_num, seq_len);                            \
  else                                                                                          \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, true><<<grid, block, 0, stream>>>(     \
        (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num, seq_len)
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/softmax.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```c++

template <typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register(half2 *qk_buf, const half2 *atten_bias,
                                                   const half2 *atten_mask, const int batch_size,
                                                   const int head_num, const int seq_len) {
  // dim3 grid(batch_size * seq_len)
  // block(32, head_num);
  int word_id = blockIdx.x; // 这个就grid的维度
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;//这个是线程id
  int head_id = threadIdx.y;//head维度
  int half2_seq_len = seq_len / 2;
  //索引矩阵数据 维度是head_numXBXSXS
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  //这个是上图中的存储最大值的数据 注意这个是寄存器数据，每个线程都有
  float s_qk_buf[count];
  if (need_padding) // true和false的区别是 是否是64的倍数
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          __hadd2(qk, __ldg(&atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]));
    half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
  }
  //到这里 每个线程都有s_qk_buf 也就是每个线程都计算了一行的一部分数据
  //然后1个warp中的数据计算完 就用warpsum去计算这一行的所有数据的最大值
  for (int i = 0; i < count; i++)
    max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);
  //这里是计算指数值
  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  //同理 这个是个warpsum 求一行的指数和
  exp_sum = warpReduceSum(exp_sum);
  //最后计算完然后写出数据
  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] =
        __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}
```

关于warpReduceSum可以参考:

https://zhuanlan.zhihu.com/p/572820783

https://zhuanlan.zhihu.com/p/572901115

#### 3. GEMM $M_{softmax} \times V$

这一步也很简单，也是直接调用了cublas的Batch GEMM, `cublas_Gemm_Strided_Batched`这个函数，要去注意函数传入的参数。

#### 4. transpose

做完Attention之后，需要把数据重新变换回来，由 $head\_{num} \times B\times S \times size\_per\_head$还原回$B\times S\times H$,以便后续计算。这个就是第0步的逆过程，可以结合着一起看，需要注意如果是`is_remove_padding_`为true，会再次将pad的数据去掉。

### 2. fused kernel源码分析

进入到fused kernel这个分支之后，可以看到按照seq len是否小于等于80做了特化，如果小于等于80，那么执行`fused_infer`或者`fused_rm_infer`kernel，如果是在80到352之间那么执行`fused_long_infer`或者`fused_long_rm_infer`kernel，至于80和352这两个数字是怎么来的，应该是根据不同卡上的shared memory的大小得到的。

```c++
 if (use_fused_attention_) {
      if (infer_param.seq_len <= 80) {
        if (is_remove_padding_)
          fused_rm_infer(infer_param);
        else
          fused_infer(infer_param);
      } else {
        if (is_remove_padding_)
          fused_long_rm_infer(infer_param);
        else
          fused_long_infer(infer_param);
      }
    } else
      nofused_infer(infer_param);
```

#### 1. fused_infer源码

这个kernel执行的思想是， 将self attention这个步骤的几个kernel融合，减少IO带来的时间消耗。由于seq len小于等于80，占用的shared memory 较小，计算的时候将Q和K的矩阵全部加载的shared memory，然后开始计算Q和K的乘法，矩阵乘用到了WMMA，也就是Tensor Core。乘法做完之后在去做softmax，之后再去做和V的矩阵乘法，同样也是将V全部load到shared memory，最后写出结果。也就是说这个kernel需要同步四次。在做矩阵乘法的时候用到了WMMA，分块的小矩阵设置成16，也就是将矩阵分成16x16的块就做wmma乘法，因此在抛线程的时候除了16，也就是按照16x16的块去分。

```c++
void Attention<OpType>::fused_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  //抛线程是按照head_numXBXSXsize_per_head这个方向去抛的
  //一共是head_numXB个block，每个block处理SXsize_per_head这个矩阵
  dim3 grid(head_num_, batch_size), block;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;
    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;
    // 1个warp计算1个16x16的矩阵，因为QK的乘完之后的结果是SxS,所以才有了 max(((seq_len + 15) / 16), size_per_head_ / 16)
    // 和V相乘的结果是Sxsize_per_head
    block.x = 32 * ((seq_len + 15) / 16) * max(((seq_len + 15) / 16), size_per_head_ / 16);
    if (size_per_head_ == 64) {
      //seq_len 按照16 pad，因为分块矩阵的大小是16
      if (seq_len <= 16)
        WMMA_ATTENTION(16, 64);
      else if (seq_len <= 32)
        WMMA_ATTENTION(32, 64);
      else if (seq_len <= 48)
        WMMA_ATTENTION(48, 64);
      else if (seq_len <= 64)
        WMMA_ATTENTION(64, 64);
      else if (seq_len <= 80)
        WMMA_ATTENTION(80, 64);
    } else if (size_per_head_ == 16) {
      //这个分支走不到
      if (seq_len <= 48)
        WMMA_ATTENTION_16(48, 16);
    }
  }
}
```

需要注意的是attention的输入的矩阵尺寸是$B \times S \times H$, 但是矩阵计算是按照$head\_num \times B \times S \times size\_per\_head$计算的，线程也是按照这个尺寸去抛的。

代码第一部分：

```c++
template <const int max_seq_len, const int size_per_head>
__global__
    void
    wmma_attention_kernel(const half2 *qkv, const half2 *qkv_bias, const __half *attention_mask,
                          __half *attention_output, const int seq_len, const float scale) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  // 分配shared memory，加上SKEW_HALF是为了防止bank conflict
  // 关于bank conflict 参考: https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf
  // 但这里处理了half的数据且size_per_head是64 也就是一行放了128个byte，正好填满整个bank，所以对s_kv
  // 和s_query来说并不存在bank conflict
  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = blockIdx.y * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query & Key
  // 将QK全部load进来，并且加上之前步骤的bias
  // 注意这个输入矩阵的尺寸是BS3H 因为是经过了GEMM了之后的矩阵，QKV在一起
  // 即:(B,S,H) X (H,3H) = (B,S,3H)
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();//这个地方等待所有的数据load完成
```

之后一部分是进行$Q \times K^T$的乘法:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/kernel_QK.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```c++
 // const int from_size = max_seq_len / 16;
 // const int to_size = max_seq_len / 16;
// 这个地方QK的乘法结果尺寸是seq_len X seq_len
// 按照结果的尺寸进行计算，每次计算是16X16的块，所以总共的warp数是 max_seq_len / 16 X max_seq_len / 16
if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;// 注意是乘K的转置，所以是列主序
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    //这个地方为什么是4，因为size_per_head是64,64按照16分正好是分4部分
    //这个是矩阵乘法的内循环，只不过是按照块的乘法
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();//数据同步 等待QK计算完成

```

之后进行softmax的计算，这个步骤中融合了load V矩阵的过程，之前计算之后的k矩阵的shared memory已经用完不再使用，所以复用s_kv

```c++
 // softmax
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float mask =
            (float)__ldg(&attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]]);
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);

    // loading Value
    int pos = (batch_seq_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[from_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }
```

之后是一部清零的操作，因为输入的max_seq_len（16的倍数）和真实seq len之间有差别，也就是多出来的数据是随机的（上上步计算QK的时候也是，有些数据是随机的），为了计算V的矩阵乘法的正确性，将V的最后几行设置为0，计算的时候就可以抹去随机值带来的错误。

```c++
// K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/kernel_zero.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

最后一部分是V矩阵的乘法：

```c++
//* V
// 这个地方为什么左移2，也就是乘4呢？ 和之前QK矩阵乘法类似，在做V矩阵乘法的时候，是SXS的矩阵和SXsize_per_head的矩阵相乘，结果尺寸还是SXsize_per_head
// 其中size_per_head=64，也就是将SXsize_per_head分成16X16的块，
// 所以应该是 (max_seq_len/16) * (size_per_head/16) = from_size*4 = from_size << 2 个warp
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;//warp id到矩阵数据的映射id
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    //这个地方和QK的乘法是类似,内循环变成了max_seq_len/16大小的块矩阵乘法
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();//同步

// 写出结果
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}
```

#### 2. fused_long_infer源码

当seq len较大的时候，将QK全部load到shared memory就不合适了，毕竟shared memory是有限的。执行这个分支的seq len 是大于等于80 小于等于352，这些参数在不同的平台上可能不同，因为ByteTransformer主要运行在A100上，使用在别的平台的时候参数可能需要按照需要调整一下。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/bytetransformer/fused_long.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

这部分的代码的逻辑是，既然QK不能同时加载进shared memory，那就将Q拆分成块，每个小块加载进shared memory，但是K需要全部加载的，因为QK的乘法是需要全部的K的，之后在做softmax然后在和V相乘得到结果，其中将Q拆分成块之后的计算kernel，和fused_infer里面的kernel极其相似，只不过load Q的时候有差别。

```c++
template <OperationType OpType>
void Attention<OpType>::fused_long_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;

    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    dim3 grid, block;
    int shared_memory_size = 0;
    //这个地方其实之前说过，就是按照16去划分块
    const int split_count = (seq_len + 15) / 16;
    switch (split_count) {
        //......
        //WMMA_ATTENTION_LONG
    }
  }
}

#define WMMA_ATTENTION_LONG(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                    \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    cudaFuncSetAttribute(wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>,           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);                 \
//因为将Q拆分了多块，所以在抛线程的时候，就多抛了一个维度，就是Q的分块的维度
grid.x = head_num_, grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid.z = batch_size,        \
// block的逻辑和之前的一样，就是几个warp来计算Q的一小块
  block.x = 32 * (SPLIT_LEN / 16 * split_count);                                                  \
  wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>                                   \
      <<<grid, block, shared_memory_size, infer_param.stream>>>(                                  \
          qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, seq_len,       \
          scale)


//这里shared memory的计算是这样的：
// __shared__ __half     s_kv  [max_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half  s_query[split_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half s_logits[split_seq_len][max_seq_len   + SKEW_HALF];
// 因为Qload的一块 所以大小是split_seq_lenXsize_per_head  K是全部load 所以是 max_seq_lenXsize_per_head
// s_logits是QK的结果所以是split_seq_lenXmax_seq_len

```

想明白了这些，在去看`wmma_attention_long_kernel`这个kernel就清楚很多了，对照着`wmma_attention_kernel`看就可以了，基本上是一样的。

## 5.关于其它层的解释（step 4 ~ step 8）

剩下的层就是FFN和layernorm相关的kernel了，和GEMM相关的是用了cublas， layernorm相关的kernel也相对比较简单，唯一复杂的是`gemm_bias_gelu`这个kernel，是将GEMM和激活层融合到了一起，使用的是cutlass的自定义实现，这部分也是没有看，后续可以继续补充。
