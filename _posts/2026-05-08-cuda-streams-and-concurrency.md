---
layout: post
title: 揭开CUDA Stream的面纱
date: 2026-05-08 10:00:00
description: 从基础概念到 CUDA Graphs / Stream-Ordered Allocator / Hopper 异步特性，系统梳理 CUDA Stream 的过去、现在与今天的最佳实践。
tags: cuda gpu computer
categories: programming
giscus_comments: false
related_posts: false
pretty_table: true

toc:
  sidebar: left
---

> 本文最早整理自 NVIDIA 的 [CUDA C/C++ Streams and Concurrency Webinar](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)（Steve Rennich，NVIDIA）。
> 那份材料发布于 Kepler 时代，许多结论需要按当下（CUDA 12/13、Hopper、Ada、Blackwell）重新审视。本文在原文基础上补充了 **CUDA Graphs**、**Stream-Ordered Memory Allocator**、**Hopper 上的 TMA / async barrier / Programmatic Dependent Launch**、以及 **MPS / MIG** 等现代特性的内容。
> {: .block-tip }

## 一、为什么我们需要 Stream

很多教程在讲 CUDA 时，第一句话都是：

> CUDA 是单指令多线程（SIMT）的并行编程模型。
> {: .block-tip }

但是这里的“并行”，更多说的是**一个 kernel 内部的线程并行**——成百上千的线程在 SM 上同时跑同一段代码。

而真实的 GPU 程序，往往不是“只有一个 kernel”这么简单。你需要：

- 把数据从 host 拷贝到 device（H2D）
- 启动一个或多个 kernel
- 把结果从 device 拷回 host（D2H）

如果这些操作完全顺序进行，那 CPU 在等 GPU、GPU 又在等 PCIe，一片资源就这样空着。

CUDA Stream 解决的就是这种“**任务级别**”的并行：让 H2D、kernel、D2H 这些异构操作可以**重叠执行**，让 GPU 上多个互相独立的 kernel 也能**同时跑**起来。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/concept-overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. CUDA Stream 在概念上是一个 FIFO 队列：host 把异步操作 push 进去，GPU 按顺序执行。
</div>

## 二、什么是 CUDA Stream？

### 1. 一句话定义

> A CUDA stream is a sequence of CUDA operations that execute on the device **in the order in which they are issued by the host**. Operations in different streams may run **concurrently** or be **interleaved**.

把这句话拆开来理解：

- **顺序**：同一个 stream 内的操作是严格按 issue 顺序串行执行的。
- **并发**：不同 stream 之间没有强制顺序，硬件资源允许时它们可以并行。
- **异步**：把操作丢进 stream 这个动作本身是 non-blocking 的，host 立刻返回。

所以 stream 本质上是一个**软件抽象**，它告诉 GPU：“这一串操作是有依赖的，必须按顺序来；其他 stream 跟我无关，你看着办。”

### 2. 默认 stream（Legacy / Per-thread）

如果你没指定 stream，所有调用都会落在**默认 stream**（也叫 NULL stream，stream 0）。

默认 stream 的语义有两种，由编译器开关决定：

| 模式                      | 编译选项                                                                | 行为                                                                                  |
| :------------------------ | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| Legacy default stream     | 默认 / `--default-stream legacy`                                        | 默认 stream 是**全局同步**的，它和任何其他 stream 上的操作都互相阻塞。                |
| Per-thread default stream | `--default-stream per-thread` 或宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM` | 每个 host 线程都有自己独立的默认 stream，行为与显式创建的 stream 一致，不会互相阻塞。 |

CUDA 7 之后官方推荐 per-thread 模式，因为它让多线程程序天然地获得并发。

> 一条很值得记的经验法则：**只要你想要并发，就不要把任何操作放在 legacy 默认 stream 上**——它会一刀切地把所有 stream 都同步住。
> {: .block-warning }

### 3. 创建和使用一个 stream

最常见的用法长这样：

```cpp
// 1) 创建 stream
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// 2) 用 pinned memory，async copy 才能真正异步
float *h_in;
cudaMallocHost(&h_in, N * sizeof(float));   // page-locked

float *d_in;
cudaMalloc(&d_in, N * sizeof(float));

// 3) issue 异步操作到 stream
cudaMemcpyAsync(d_in, h_in, N * sizeof(float),
                cudaMemcpyHostToDevice, s1);
my_kernel<<<grid, block, 0, s1>>>(d_in, N);
cudaMemcpyAsync(h_in, d_in, N * sizeof(float),
                cudaMemcpyDeviceToHost, s1);

// 4) host 在这里其实并没有等 GPU，它会立刻往下走
//    需要时再显式等待：
cudaStreamSynchronize(s1);

// 5) 用完销毁
cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```

几个非常容易踩的小坑：

- `cudaMemcpyAsync` 必须配合 **pinned (page-locked) memory** 才能真正异步执行；用普通 `malloc` 出来的内存，运行时会偷偷走同步路径，看起来异步其实根本没并行。
- kernel launch 的第四个模板参数才是 stream，第三个是 dynamic shared memory size，写错很常见。
- `cudaStreamCreate` 的对偶是 `cudaStreamDestroy`，不释放就泄漏。

## 三、Stream 让什么变快了：异构引擎的重叠

要理解 stream 为什么能加速，需要先看一眼 GPU 上有几台“发动机”在并行工作。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/gpu-engines.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. 一块 GPU 上至少有 1 个 Compute Engine 和 1 个 Copy Engine。Tesla / 数据中心卡通常有 2 个 Copy Engine（H2D 和 D2H 各一个），这意味着拷贝方向不同的两个 memcpy 也能并行。
</div>

也就是说，GPU 的硬件层面本来就有多个**互相独立**的执行单元：

- Compute Engine：执行 kernel
- Copy Engine：通过 PCIe 做 DMA 拷贝
- 一些更新一点的卡：H2D 和 D2H 是分开的两个 copy engine

只要你**告诉**驱动，"这些操作之间没有依赖"，它们就能被分配到不同引擎上并行跑。而把"没有依赖"这件事告诉驱动的标准方式，就是把它们丢到不同的 stream 里。

下面这张图直观地说明了使用 stream 之前和之后的差别：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/serial-vs-stream.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. 上：单 stream / 同步拷贝时 H2D、Kernel、D2H 完全串行；下：把数据切成 4 个 chunk，每个 chunk 走一个独立 stream，三种引擎被同时利用，墙钟时间几乎对半减少。
</div>

这就是经典的"**流水线并行 (pipelining)**"：

1. Stream 1 在做 Kernel 时，Stream 2 已经可以用 Copy Engine 把自己的数据拷上来；
2. Stream 1 拷结果回去时（D2H），Stream 2 在跑 kernel，Stream 3 又开始 H2D；
3. 三种引擎一直处于忙碌状态。

## 四、何时使用多 stream 才有效果？

不是所有任务都能从 stream 拿到加速，下面是几个非常实际的判断标准：

**(1) 你的程序里有明显的"传输 + 计算"循环**

像深度学习推理、图像批量处理、信号处理 chunk by chunk 这种典型 pipeline，最适合 stream，几乎一定能拿到 1.5x ~ 2x 的提升。

**(2) 单个 kernel 没把 GPU 占满**

如果一个 kernel 已经把 SM、register、shared memory 用满了，再开几个 stream 也只是排队，并不能真的并发跑。  
反过来，如果你的 kernel 很小（比如每个 batch 只发射几十个 block），那 GPU 是吃不饱的，多开几个 stream 才能"塞满"它。

**(3) 你有多个互相独立的 kernel**

经典场景：模型并行里多个分支可以同时算，或者多用户/多请求复用同一块 GPU。

**(4) 你愿意用 pinned memory**

这一点经常被忽视。`cudaMemcpyAsync` 不用 pinned memory 是没有真异步的，所有"重叠"全是错觉。

**反过来，下面这些情况开 stream 没有意义：**

- 单个 kernel 已经占满 SM；
- 工作量太小，kernel 启动延迟（几微秒）就把节省的时间吃掉了；
- 中间穿插了同步 API（如 `cudaDeviceSynchronize`、阻塞版 `cudaMemcpy`）；
- 数据有强依赖，第二步必须等第一步完成。

## 五、Stream（软件） vs SM（硬件）：别再混淆了

这一节是面试里最容易栽跟头的点。**Stream 和 SM 完全是两个层级的概念**：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/stream-vs-sm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. Stream 是 host 端的逻辑队列；SM 是 GPU 上的物理执行单元。Stream 描述"任务的依赖关系"，SM 描述"代码在哪里跑"。
</div>

具体来说：

- **Stream = 软件抽象**。它是一个 host 端的命令队列，里面装的是对 GPU 的请求（memcpy、kernel launch、event 等）。Stream 的数量你想开多少都可以（实际由驱动管理）。
- **SM (Streaming Multiprocessor) = 硬件单元**。一个 SM 内部包含许多 CUDA Core / Tensor Core、Warp Scheduler、Shared Memory、寄存器堆等。一个 GPU 由几十到上百个 SM 组成，整个 grid 就是分散到所有 SM 上去跑的。

它们之间的联系是：

> Stream 决定 GPU **要不要**执行这个 kernel；当多个 stream 上有 kernel 同时被允许执行时，调度器会把它们的 thread block 分配到**空闲的 SM** 上去。
> {: .block-tip }

也就是说：

- **SM 是真正干活的人**；
- **Stream 是工头**，负责告诉 SM 哪些活可以并行，哪些必须排队；
- 没有空闲 SM 时，再多的 stream 也只能等。

## 六、多个 kernel 用多个 stream 提交，什么时候会真的并行？

这是本文最核心的问题。把所有现实约束列在一起就是下面这张图：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/parallel-conditions.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. 不同 stream 上的 kernel 真正并行的六大条件。
</div>

下面逐条解释。

### 1. 不能落在 legacy 默认 stream 上

`kernel<<<grid, block>>>(...)` 没指定 stream 时，等价于 `kernel<<<grid, block, 0, 0>>>(...)`，落在 stream 0。在 legacy 模式下，stream 0 上的操作会和**所有其他 stream 同步**，并发立刻消失。

### 2. SM 资源够用

每个 thread block 会占用 SM 的若干 register 和 shared memory，每个 SM 同时能驻留的 warp 也有上限。如果第一个 kernel 已经把这些资源全用了，第二个 kernel 哪怕在不同 stream 也只能等。

判断一个 kernel 用了多少资源，可以用 `nvcc -Xptxas -v` 编译看输出，或者用 Nsight Compute 的 occupancy 视图。

### 3. 没有 false dependency（伪依赖）

这是历史遗留问题，需要展开讲一下。

**Fermi 时代**，整个 GPU 只有 **1 条** 硬件工作队列（hardware work queue）。多个 stream 全部通过这一条队列被串行送进 GPU。结果就是：

- Stream 1 的 K1 → K2 → K3
- Stream 2 的 K4 → K5 → K6

驱动看到队列里是 K1, K4, K2, K5, K3, K6（按 issue 顺序），但调度器在硬件层面**只能按队列顺序看依赖**，所以 K4 就被 K2 莫名其妙地"挡"住了，明明它们处于不同 stream，没有任何真实依赖。这就是 false dependency。

**Kepler GK110** 引入了 **Hyper-Q** 和 **Grid Management Unit (GMU)**，把硬件队列从 1 条扩展到 **32 条**：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/hyperq.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6. 左：Fermi 时代所有 stream 共用 1 条硬件队列，存在 false dependency。右：Kepler+ 的 Hyper-Q 提供最多 32 条独立硬件连接，CUDA 驱动会自动把不同 stream 映射到不同 HW queue。
</div>

从 Kepler 开始（CC ≥ 3.5），如果你用的 stream 数 ≤ 32，每个 stream 都有自己独立的硬件队列，false dependency 基本不存在；超过 32 个 stream 时，多个 stream 会复用同一条硬件队列，又会出现一些复用上的开销。

**今天（Hopper / Ada / Blackwell）你应该怎么想这件事？**

- 所有还在卖的 NVIDIA GPU 都早已支持 Hyper-Q，**false dependency 已经几乎不是问题**，再去为它纠结 issue 顺序只会浪费时间。
- 数据中心的 Volta / Ampere / Hopper / Blackwell 还有 **MPS（Multi-Process Service）**，让多个进程也能走同一个 GPU context、同样享受并发。
- 真正影响并发的，已经从"硬件队列够不够"转移到了"**SM 资源够不够**"和"**kernel 启动延迟有没有摊薄**"。后者就是 CUDA Graphs 解决的事，本文后半段会专门展开。

把这条演化路径放在一起看更清楚：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/arch-timeline.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6.5 从 Fermi 到 Blackwell 的架构演进与 stream 相关特性。今天主流的卡都已经在紫色括号涵盖的"现代时代"——Hyper-Q 是缺省、stream-ordered 显存分配和 CUDA Graphs 已成新默认。
</div>

> 所以一条更新过的工程经验是：**stream 数量大致 8~16 就足够了**，再加上 CUDA Graphs 来摊薄启动延迟，比单纯堆 stream 数更有效。
> {: .block-tip}

### 4. 中间不能穿插 host 端的同步

下面这些 API 都会把整个设备 flush 一遍，所有 stream 立刻被同步：

- `cudaDeviceSynchronize()`
- 阻塞版 `cudaMemcpy`（注意没有 Async 后缀）
- `cudaFree`、`cudaMalloc`（在某些版本上是同步的）
- `cudaMallocHost` 这种页锁内存分配

如果你在两个 kernel launch 之间不小心加了一个同步操作，所有努力都白费。

### 5. Async copy 必须用 pinned memory

这点前面提过了，再强调一次：`cudaMemcpyAsync` + `malloc` 的内存 = 假异步。

### 6. issue 顺序：breadth-first 优于 depth-first

这一点比较微妙，专门用一节来讲。

## 七、Issue 顺序：Breadth-first vs Depth-first

考虑 4 个 stream，每个都做 H2D → Kernel → D2H。代码可以这样写：

**Depth-first（每个 stream 一次性 issue 完）**：

```cpp
for (int i = 0; i < N_STREAMS; ++i) {
    cudaMemcpyAsync(d_in[i], h_in[i], bytes, H2D, s[i]);
    kernel<<<g, b, 0, s[i]>>>(d_in[i], d_out[i]);
    cudaMemcpyAsync(h_out[i], d_out[i], bytes, D2H, s[i]);
}
```

**Breadth-first（按操作类型分批 issue）**：

```cpp
for (int i = 0; i < N_STREAMS; ++i)
    cudaMemcpyAsync(d_in[i], h_in[i], bytes, H2D, s[i]);

for (int i = 0; i < N_STREAMS; ++i)
    kernel<<<g, b, 0, s[i]>>>(d_in[i], d_out[i]);

for (int i = 0; i < N_STREAMS; ++i)
    cudaMemcpyAsync(h_out[i], d_out[i], bytes, D2H, s[i]);
```

在 Fermi 这种**单硬件队列**的设备上，两者表现差距巨大：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/breadth-vs-depth.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 7. 上：depth-first 在 Fermi 时几乎完全串行；下：breadth-first 让 H2D 引擎、Compute 引擎、D2H 引擎的流水线全部跑起来。Hyper-Q 时代两者差距已经显著缩小，但 breadth-first 仍然是更安全的写法。
</div>

在 Kepler+ Hyper-Q 之后，这个差异变小了——硬件已经能识别真实依赖。但**让不同类型的操作分批 issue**仍然是更稳的写法，特别是对老一些的 GPU 或者跨平台代码。

## 八、跨 stream 同步：events 和 cudaStreamWaitEvent

有时候 stream 之间确实有依赖：例如 stream B 需要用 stream A 算出来的中间结果。强行用 `cudaDeviceSynchronize` 会把所有 stream 都停下来，太重了。正确的做法是用 **event**：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/sync-events.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 8. 用 cudaEventRecord 在 stream A 上"打个标记"，再用 cudaStreamWaitEvent 让 stream B 等到这个标记被达到，不影响其他 stream。
</div>

代码骨架：

```cpp
cudaEvent_t e;
cudaEventCreate(&e);

// stream A
cudaMemcpyAsync(..., A);
kernel_a1<<<..., A>>>(...);
cudaEventRecord(e, A);          // 在 A 上记录 event
kernel_a2<<<..., A>>>(...);     // 这个不阻塞 A 自己

// stream B
cudaMemcpyAsync(..., B);
cudaStreamWaitEvent(B, e, 0);   // 让 B 等 e 被记录后再继续
kernel_b<<<..., B>>>(...);      // 现在它一定看到了 kernel_a1 的结果
```

`cudaStreamWaitEvent` 的关键特性：

- **只阻塞 stream B**，不影响 host，也不影响其他 stream；
- B 上**这次调用之前**已经 issue 的命令不会被它阻塞；
- 它影响的是**这次调用之后**入队的命令。

另外，event 还可以拿来计时：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);

cudaEventRecord(start, 0);
// ... your stream operations
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

比 `clock()` 或 `chrono` 准确得多，因为它直接量的是 GPU 上 event 之间的时间。

## 九、Stream-Ordered Memory Allocator：现代异步显存

CUDA 11.2（2020 年底）引入了一组非常重要、但很多人还没用上的 API：`cudaMallocAsync` / `cudaFreeAsync` / `cudaMallocFromPoolAsync`。它们让显存分配/释放从一个**全局阻塞的、要 syscall 的操作**变成了一个**stream 序的、走内存池的廉价操作**。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/stream-mempool.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 9. Stream-ordered allocator：分配/释放都按 stream 顺序排队进入内存池，块在池内被回收复用，绝大多数情况都不再触发设备级同步或 OS 调用。NVIDIA 在大数据 benchmark 上观察到 2~5x 的端到端加速。
</div>

它解决的痛点：

- 老的 `cudaMalloc` / `cudaFree` 是**同步**的——任何调用都会让整个设备 flush 一遍，前文一直强调"不要在 stream 中间穿插同步"，但显存分配本身就是个隐藏的同步点。
- 工程上常见的回避做法是"自己写一个内存池"，但那意味着每个项目都重复造轮子。`cudaMallocAsync` 让 CUDA 驱动直接帮你管这块。

典型用法：

```cpp
cudaStream_t s;
cudaStreamCreate(&s);

float *d_buf;
cudaMallocAsync(&d_buf, N * sizeof(float), s);   // stream 序分配
my_kernel<<<g, b, 0, s>>>(d_buf, N);
cudaFreeAsync(d_buf, s);                          // stream 序释放

cudaStreamSynchronize(s);
```

几个值得知道的小细节：

- `cudaMallocAsync` 的"释放"并不会立刻把内存还给操作系统，而是回到**当前设备的内存池**里供后续 `cudaMallocAsync` 复用，所以反复 alloc/free 的代价非常低。
- 你可以用 `cudaDeviceGetDefaultMemPool` 查询/调整默认内存池的水位线（如 `cudaMemPoolAttrReleaseThreshold`），决定多少内存在空闲时还回 OS。
- 不同 stream 之间的内存如果有跨 stream 引用，需要用 `cudaStreamSynchronize` 或 event 来保证可见性。
- PyTorch 的 caching allocator 几年前就在做类似的事，从 2.0 之后也提供了基于 `cudaMallocAsync` 的 backend（`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`）。

> 一条新经验法则：**只要你在 CUDA 11.2+ 上写新代码，就该把 cudaMallocAsync 当默认选项。** 它不仅快，更重要的是它让显存生命周期和 stream 流水线天然对齐，写出来的代码更难出 bug。
> {: .block-tip }

## 十、CUDA Graphs：把 stream 模式榨到极致

写到这里你可能注意到一个问题：每次 kernel launch，host 都要走 driver、做参数打包、入队……一次大概 5~10 微秒。对于跑得很快的 kernel（比如几十微秒级别），这个 launch overhead 占比惊人。

CUDA Graphs（CUDA 10 引入，CUDA 12 已经成为主流）把"一连串 kernel/copy/event"打包成一个**有向无环图**，host 端"录"一次，之后 GPU 端可以**反复 replay**：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/cuda-graphs.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 10. 上：传统 stream 模式下 CPU 必须为每个 kernel 付一次 launch overhead；下：CUDA Graph 在第一次"capture"时把所有依赖关系记录下来，之后每次迭代只需一次 cudaGraphLaunch，CPU 几乎闲到无事可做。
</div>

在迭代式负载（训练循环、推理服务、仿真步进、信号处理）上，CUDA Graphs 通常能拿到 **1.1x ~ 1.7x** 的整体加速，而单看 graph 内部的执行段，可以是 **5x** 级别的提升（数据来自 NVIDIA 公开 benchmark）。

### 1. 两种构建方式

**(a) Stream Capture（推荐入门）**：把已有的 stream 代码"录"成 graph。

```cpp
cudaGraph_t graph;
cudaGraphExec_t exec;
cudaStream_t s;
cudaStreamCreate(&s);

// 1) Begin capture
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);

// 2) 像往常一样发射 kernel / memcpy
kernel_a<<<g, b, 0, s>>>(...);
cudaMemcpyAsync(..., s);
kernel_b<<<g, b, 0, s>>>(...);

// 3) End capture, instantiate
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// 4) Replay 任意多次
for (int i = 0; i < 1000; ++i) {
    cudaGraphLaunch(exec, s);
}
cudaStreamSynchronize(s);
```

**(b) 手工构建 graph 节点**：用 `cudaGraphAddKernelNode` 等 API 显式建图，能力最强，写起来最累，一般框架开发者才会用。

### 2. PyTorch 里的 CUDA Graphs

PyTorch 直接把它包成了三层 API，由易到难：

```python
# (i) 最省事 —— torch.compile 自动做
m = torch.compile(model, mode="reduce-overhead")

# (ii) make_graphed_callables ——专门给训练循环
graphed_model, graphed_optim = torch.cuda.make_graphed_callables(
    (model, optimizer), sample_args)

# (iii) 完全手动控制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)
# 之后每步只需要把数据填进 static_in 然后：
g.replay()
```

### 3. 使用 CUDA Graphs 的硬性约束

CUDA Graph 的强大来自"录一次、跑很多次"，所以它要求每次 replay 都**完全一致**：

- **shape / scalar / 控制流要静态**：if/else 选择不同的 kernel，shape 不固定，都没法 capture。动态 shape 现在的常用对策是 **padding 到桶（bucket）**或者用 `torch.compile(dynamic=True)`。
- **指针要稳定**：graph 记的是地址，不是 tensor 对象。用框架的"static input/output"占位 tensor，不要每步重新分配。
- **CPU 端的 Python 副作用不会被 replay**：在 graph 内部 `print`、append 到 list 都不会被记录到 replay 里，初学者经常被这个 bug 坑。

> 经验法则：**stream 解决"任务级并行"，CUDA Graphs 解决"启动开销"——两者是互补的，不是二选一。** 现代高性能推理代码一般会同时用 stream + graph。
> {: .block-tip }

## 十一、Hopper 之后：从"task 级"到"指令级"的异步化

更新一点的 GPU（H100、B100/B200）在 stream 之外又开了一层异步：**kernel 内部的指令也可以异步**。这部分对一般应用代码不是必须的，但理解它对读 CUTLASS、Triton、FlashAttention v3 这种最新 kernel 很有帮助。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/hopper-async.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 11. Hopper 引入的两类新异步：左侧是 kernel 内部的 producer/consumer warp + TMA + cuda::barrier，让数据搬运和 Tensor Core 计算重叠；右侧是 Programmatic Dependent Launch (PDL)，让同一 stream 中前后两个 kernel 的尾巴和头部也能重叠。
</div>

### 1. TMA (Tensor Memory Accelerator) 与 async barrier

H100 引入了 **TMA**——一个专门的硬件单元，由**单个线程**触发，就能把整块多维 tensor 从 global memory 异步搬到 shared memory（或反向）。配合 `cuda::barrier`，可以写出真正的"**warp-specialized**"kernel：

- 一组**生产者 warp**只负责发起 TMA 拷贝；
- 另一组**消费者 warp**在 barrier 上等数据，等到了就立刻做 Tensor Core 矩阵乘。

```cuda
// 伪代码，参考 CUTLASS / FlashAttention v3
if (warp_id == PRODUCER_WARP) {
    if (lane == 0) {
        cp_async_bulk_tensor(smem_buf, tma_desc, /*coords*/);
        bar.arrive();
    }
} else {
    bar.wait();
    wgmma(/*...*/);   // warp-group MMA, 异步 Tensor Core
}
```

它和 stream 是不同层级的概念：stream 是**多 kernel 之间**的并行抽象，TMA 和 async barrier 是**单 kernel 内部**的并行抽象。两者叠加，才能榨干 H100 的算力。

### 2. Programmatic Dependent Launch (PDL)

在 Hopper / Blackwell 上，同一个 stream 里**前一个 kernel 还没结束**，后一个 kernel 已经可以**开始装载、做准备工作**了，甚至在前一个 kernel "准备好被替换"时直接跑：

- `cudaTriggerProgrammaticLaunchCompletion()` 让 kernel A 主动告诉系统"我后面 SM 已经空了，B 可以开始"；
- B 在 launch 时附带 `cudaLaunchAttributeProgrammaticStreamSerialization`，就会等这个信号。

CUTLASS 的 pipelined GEMM 大量用这个特性，让 epilogue 阶段和下一个 GEMM 的 prologue 阶段重叠。

### 3. Thread Block Cluster

H100 还把 grid → block 这两层增加了一层 **cluster**：一组 block 被保证**同时调度到相邻 SM 上**，并且它们之间可以直接读写对方的 shared memory（**Distributed Shared Memory, DSMEM**）。这同样是 kernel 内部的事，但它让设计更大、依赖更复杂的并行 kernel 成为可能。

> 用一句话归纳现代 GPU 的异步层级：
>
> **stream（任务）→ kernel（block / cluster）→ warp（producer / consumer）→ TMA / async copy（指令级）。**
>
> 每层都有自己的"重叠"机制，越底层的层级越接近硬件、越能压榨极限性能。
> {: .block-tip }

## 十二、Stream 之外：MPS 与 MIG 怎么共享一块 GPU

到这里我们一直在讨论"**一个进程内**怎么用多个 stream"。但生产环境经常有这种需求：

- 多个用户、多个推理服务想共享一块 H100；
- 训练在跑，同时还要给一个 dashboard 留点 GPU 算力。

CUDA 给了三种层次的共享方式：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/cuda-streams/gpu-sharing.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 12. Stream / MPS / MIG 的对比。Stream 解决"一个进程内"的并行；MPS 让多个进程也能时间分片地共享同一个 context；MIG 把整块 GPU 物理切成多个互相隔离的小 GPU。
</div>

|             | 隔离强度           | 适用场景                     | 代价                                       |
| :---------- | :----------------- | :--------------------------- | :----------------------------------------- |
| **Streams** | 无（同一进程）     | 单应用内部并行               | 0                                          |
| **MPS**     | 弱（同一 context） | 多进程共享，希望最大化利用率 | 多进程时一个挂会拖累其他                   |
| **MIG**     | 强（硬件分区）     | 多租户、SLA 隔离、机密计算   | 资源被切成几个 fixed-size 实例，分配粒度粗 |

### 1. MPS（Multi-Process Service）

MPS 的核心思路是：**所有客户端进程共享同一个 CUDA context**，由一个 MPS daemon 居中调度。这样不同进程发到 GPU 上的 kernel 就能像同一进程的 stream 一样并行；不再需要每次 context 切换。

启用方式很简单（Linux）：

```bash
nvidia-cuda-mps-control -d
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# 之后启动你的 CUDA 应用即可
```

Volta 之后 MPS 经过了一次重写，每个 client 有独立的 page table 和地址空间，安全性比早期版本好很多；Blackwell 上还提供了 **MLOPart** 选项（Memory Locality Optimized Partitions），把 MPS client 绑定到更亲和的 SM/L2 区域。

### 2. MIG（Multi-Instance GPU）

MIG 是从 A100 开始的硬件特性：把一个 GPU 物理切成最多 **7 个 instance**，每个 instance 拥有独立的 SM 子集、独立的 L2 切片、独立的显存通道。从应用看就是几张小 GPU，互相之间**完全不会争抢资源**。

启用方式（管理员权限）：

```bash
nvidia-smi -i 0 -mig 1                       # 打开 MIG
nvidia-smi mig -cgi 9,14,14 -C               # 创建 3 个分区（H100 上的 GI profile）
```

Hopper / Blackwell 还把 MIG 升级到了第二代，支持机密计算（Confidential Computing），适合云上多租户。

> 选型一句话总结：
>
> - 一个应用内部 → **stream + CUDA Graphs**；
> - 同主机多个应用想 share GPU → **MPS**；
> - 多租户、安全隔离、SLA 严格 → **MIG**。
>   {: .block-tip }

## 十三、PyTorch 里的 stream

PyTorch 的 `torch.cuda.Stream` 直接封装了 CUDA stream，用法和 C++ 几乎一一对应：

```python
import torch

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

x = torch.randn(N, device='cuda')

with torch.cuda.stream(s1):
    y1 = compute_branch_a(x)

with torch.cuda.stream(s2):
    y2 = compute_branch_b(x)

# wait for both
torch.cuda.synchronize()
out = y1 + y2
```

几个值得注意的点：

- 默认情况下 PyTorch 把所有 op 丢到 current stream，多线程时是 per-thread 的；
- 跨 stream 用 tensor 时，PyTorch 内部会插入 `record_stream` 来防止 caching allocator 把还在用的内存提前回收；
- 用 `non_blocking=True` 配合 `pin_memory()` 才能真正让 H2D 异步。

## 十四、调试和 profile

光看代码很难判断 stream 到底有没有并行起来。推荐工具链：

- **Nsight Systems**（`nsys`）：能画出每个 stream 的时间线，清楚看到三种引擎是不是在重叠；从 2023 起也直接可视化 CUDA Graph 节点。这是一切的基础。
- **Nsight Compute**（`ncu`）：分析单个 kernel 的占用率、寄存器、shared memory，回答"是不是 SM 资源把并发限制住了"。Hopper 之后还能看 TMA / async barrier 的事件。
- **CUPTI / NVTX**：在代码里 `nvtxRangePush/Pop` 给关键阶段打 tag，nsys 报告里看得清清楚楚。
- 环境变量 `CUDA_LAUNCH_BLOCKING=1`：强制所有 launch 同步，方便定位"到底是哪个 kernel 出错"。注意它会破坏并发，仅用于 debug。
- PyTorch：`torch.profiler` 可以直接导出 Chrome trace，看 kernel/stream/CUDA Graph 的时间线。

经验上，**在引入多 stream 或 CUDA Graphs 之前，先用 Nsight Systems 跑一遍，看 GPU 利用率到底是被什么挡住的**。

- 如果是 PCIe 带宽瓶颈 → stream 流水线一定有效；
- 如果是 launch 开销占比高（小 kernel 多） → CUDA Graphs 是首选；
- 如果是 kernel 本身没占满 SM → 试试更大的 block / cluster、或者考虑 warp specialization；
- 如果是 kernel 已经把 SM 占满 → stream 帮不上忙，得去优化 kernel 本身。

## 十五、小结

文章很长，最后用一组分层的 takeaway 捋清楚：

**关于 stream 本身**

- Stream 是**软件抽象**，是一个 FIFO 队列，用来告诉 GPU 哪些操作有依赖、哪些没有。
- Stream ≠ SM。SM 是真正干活的硬件单元；stream 只是工头。
- Stream 的价值在于**让 H2D / Compute / D2H 三种异构引擎重叠**，以及**让多个独立 kernel 在 SM 资源足够时同时跑**。
- 真正能看到并发的硬性条件：不在 legacy 默认 stream 上、SM 资源够、没有 host 同步穿插、用 pinned memory、单个工作量不要太小。

**关于 Hyper-Q / 历史包袱**

- Fermi 那种 false dependency 在今天**已经不需要再担心**——所有数据中心 GPU 自 Kepler 起就是 Hyper-Q。
- 工程上 stream 数 8~16 通常就够了，再多收益递减。
- breadth-first issue order 仍是好习惯，但不再是性能的关键。

**关于现代特性（CUDA 11.2+ / Hopper / Blackwell）**

- **`cudaMallocAsync` + 内存池**应当是新代码的默认显存分配方式。
- **CUDA Graphs** 是迭代式负载（训练 / 推理 / 仿真）摊薄启动延迟的标准做法。和 stream 不是二选一，而是叠加使用。
- **Hopper 的 TMA + async barrier + PDL** 把异步从 task 级推到了指令级，主要影响 kernel 实现者（CUTLASS / Triton 这一层）。
- 一块 GPU 想被多个进程或租户共享：**MPS** 适合"协作型"多进程，**MIG** 适合"隔离型"多租户。

**最后一条经验法则**

CUDA stream 理论上看一遍就懂；但工程上能不能用对、用满，是要靠 profiler 一次次打脸出来的。和别的并行编程模型一样，**最好的并行，是先把单线程版本写到极致**。在那之上，再按"消除 host 同步 → 加 stream → 加 CUDA Graphs → 调 kernel 内部异步"的顺序逐层优化。

## Sources

**经典材料（理解原理）**

- [CUDA C/C++ Streams and Concurrency Webinar (Steve Rennich, NVIDIA)](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
- [CUDA Streams: Best Practices and Common Pitfalls (Justin Luitjens, NVIDIA)](https://www.hpcadmintech.com/wp-content/uploads/2016/03/Carlo_Nardone_presentation.pdf)
- [GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [CUDA Stream — Lei Mao's Log Book](https://leimao.github.io/blog/CUDA-Stream/) · [CUDA Default Stream](https://leimao.github.io/blog/CUDA-Default-Stream/)

**官方文档（最新版本）**

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) · [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) · [Multi-Instance GPU (MIG) User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html)

**现代特性（深入阅读）**

- [Using the NVIDIA CUDA Stream-Ordered Memory Allocator (Part 1)](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/) · [Part 2](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/)
- [Getting Started with CUDA Graphs (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/cuda-graphs/) · [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [Accelerating PyTorch with CUDA Graphs (PyTorch Blog)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) · [PyTorch CUDA Graph Best Practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) · [Deep Dive on the Hopper TMA Unit (PyTorch Blog)](https://pytorch.org/blog/hopper-tma-unit/)
- [CUTLASS: Dependent Kernel Launches](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) · [CUTLASS Tutorial: Mastering TMA (Colfax Research)](https://research.colfax-intl.com/tutorial-hopper-tma/)

**历史背景**

- [Hyper-Q Example (Thomas Bradley, NVIDIA)](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/simpleHyperQ/doc/HyperQ.pdf)
- [NVIDIA Kepler GK110 Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)
- [CUDA Series: Streams and Synchronization (Dmitrij Tichonov)](https://medium.com/@dmitrijtichonov/cuda-series-streams-and-synchronization-873a3d6c22f4)
