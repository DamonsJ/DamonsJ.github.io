---
layout: post
title: 关于SM Occupancy的一些解释说明
date: 2025-03-21 11:46:00
description: 解释SM Occupancy的计算和意义
tags: GPU
categories: AI
pretty_table: true
---

## 什么是SM Occupancy
SM Occupancy 也叫 Achieved Occupancy{% sidenote 'One' 'See [Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)' %}、 SM Warp Occupancy。官方解释是:

The achieved occupancy for each SM. The values reported are the average across all warp schedulers for the duration of the kernel execution.

也就是说SM Occupancy表示的是当前kernel的执行时间内参与计算的warp和理论的warp的占比。其中占用率定义为 SM 上的活动 warp 与 SM 支持的最大活动 warp 数之比。占用率会随着 warp 的开始和结束而变化，并且每个 SM 的占用率可能不同。

低占用率会导致指令发出效率低下，因为没有足够的合格 warp 来隐藏相关指令之间的延迟。当占用率达到足以隐藏延迟的水平时，进一步增加占用率可能会降低性能，因为每个线程的资源减少。内核性能分析的早期步骤应该是检查占用率并观察在不同占用率水平下运行时对内核执行时间的影响。

需要注意的是活动 Warp 存在上限，因此占用率也存在上限，该上限可从启动配置、内核的编译选项和设备功能中得出。内核启动的每个块都会分发到其中一个 SM 进行执行。从块的 Warp 开始执行到块中的所有 Warp 都退出内核为止，该块被视为处于活动状态。SM 上可并发执行的块数受以下列出的因素限制。活动 Warp 的上限是活动块的上限与每个块的 Warp 数的乘积。因此，可以通过增加每个块的 Warp 数（由块尺寸定义）或更改限制 SM 上可容纳多少块的因素以允许更多活动块来提高活动 Warp 的上限。限制因素包括：每个SM的warp数，每个SM的block数，每个SM的寄存器数和每个SM的shared memory数。也就是说你写一个kernel用了多少资源，如何设置了启动参数等等都会影响这个占用率。这个就是占用率上线就是Theoretical Occupancy。 当你使用Nsys进行性能profile的时候，鼠标移动到执行的kernel处你会看到显示的Theoretical Occupancy值。


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/1-theoretical-roccupancy.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


## 实际案例
我们以T4为目标机器实际计算一下SM的Occupancy来说明这个值的计算过程。下面是T4机器的一些信息：
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/2-t4-deviceinfo.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

| Items | Values |
| :----------- | :------------: |
|Shared Memory Per Block | 49152 bytes|
|Register Per Block | 65536|
|Max Threads Per SM | 1024|
|Max Threads Per Block | 1024|
|Warp size | 32|

<p></p>

下面是一个kernel的实际例子：
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/3-kernel.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

其中运行关键参数是：

| Items | Values |
| :----------- | :------------: |
|grid  | 5，20，1 |
|block | 256,1,1|
|Dynamic shared memory | 24576 bytes|
|Register per thread  | 158|
|Shared memory | 32768 bytes|

<p></p>

总共抛了5x20x1 = 100个block， 每个block有256个线程，也就是256/32 = 8个warp。
每个SM有1024个线程，理论可以抛1024/256 = 4个block
每个线程占用register是158， 那么每个block256个线程占用register是158x256=40448， 每个SM最大的register是65536， 那么65536/40448 = 1.62，也就是最多允许1个block运行
每个SM的shared memory是49152，目前用到32768， 49152/32768=1.5 ，也是最多允许1个block运行

1个SM理论可以运行4个block也就是32个warp，但是根据实际资源使用情况，只能允许1个block也就是8个warp，所以当前执行的理论最大占用率是8/32 = 25%，所以统计的时候最大的占用率也就25%。


## 参考链接

```
https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm
https://xmartlabs.github.io/cuda-calculator/
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
https://docs.nvidia.com/nsight-systems/UserGuide/index.html
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#
https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/issueefficiency.htm
```