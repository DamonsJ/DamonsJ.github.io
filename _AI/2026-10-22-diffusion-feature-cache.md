---
layout: post
title: AIGC 推理加速：扩散模型的特征缓存
date: 2026-10-22 12:00:00
description: 整理 FBCache、TeaCache、TaylorSeer、SeaCache、DPCache、SoftCache 等特征缓存方法，记录它们缓存什么、什么时候重算、跳过的步怎么补
tags: [特征缓存, Diffusion, 推理优化, 视频生成]
categories: [生成模型]
pretty_table: true
toc:
  sidebar: left
---

## 为什么？

上一篇[分布式推理](https://valdrada.cn/AI/2026-08-08-distribute-aigc.html)写的是"一步怎么算得更快"：把 27280 个 token、24 个 head 切到多张卡上，把单步的时延压下去。但是扩散模型还有另外一个维度，就是步数本身。还是以 Wan2.2 为例，30 个 WanBlock × 50 步 = 1500 次 WanBlock，不管卡怎么切，这 1500 次是一次都没有少的。

减少步数最直接的办法是蒸馏，但蒸馏要训练，换一个模型、换一个 LoRA 可能都要重新来一遍。另一条路是训练无关（training-free）的特征缓存：相邻两步的输入只差一点点，模型的中间特征和输出也很接近，那有些步其实没必要完整地算一遍，直接用上一步的结果，或者根据前几步的结果"猜"一个出来就行了。

特征缓存是 AIGC 推理优化里绕不开的一块，diffusers、xDiT、cache-dit、lightx2v 这些框架基本都集成了一种或几种。需要把这块系统地整理一遍：每种方法到底缓存的是什么、什么时候重新计算、跳过的那些步是怎么补上的，代码大概长什么样子。

这里的缓存和 LLM 的 KV Cache 不是一回事。因果 Attention 里，前缀的 K/V 不会因为后面追加了 token 而改变，缓存下来是精确复用；扩散模型每一步的 latent 和 timestep 都在变，上一步的特征其实已经过期了，拿来用是一种近似。所以特征缓存本质上是**用一点误差换掉一部分昂贵的计算**，速度和结果的一致性要放在一起看。

后面的代码都是只保留核心逻辑的伪代码，默认单请求、单个 CFG 分支。文中 step 指实际执行的第几步（从 0 递增），t 或 σ 是 scheduler 的时间坐标，"锚点"指最近一次完整计算并更新了缓存的那一步。

## 是什么：特征缓存到底在缓存什么？

### 冗余从哪里来

一个 DiT 的前向大概是这样的：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-00-dit-forward.svg" alt="DiT 一次前向的数据流：patch_embed 得到 s_t，经过 L 个 block 得到 h_t，再由输出头和 scheduler 得到下一步的 latent；timestep embedding 通过 AdaLN 调制每个 block 和输出头" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. DiT 的一次前向。xₜ 经 patch_embed 变成 token 序列 sₜ，过 L 个 block 得到 hₜ，再由输出头和 scheduler 得到下一步的 latent；timestep 通过 AdaLN 调制每个 block 和输出头。
</div>

计算量主要在中间那 L 个 block 里。patch embedding、timestep embedding 和最后的输出投影都比较便宜，不过当大部分 block 被跳过之后，它们的占比就上来了，后面算加速比的时候会看到这一点。而相邻两步之间：

- 输入 $$x_t$$ 和 $$x_{t-1}$$ 只差一次很小的 scheduler 更新；
- timestep 是相邻的，调制特征之间也有很强的相关性。

所以很多步的中间特征 $$h_t$$ 和 $$h_{t-1}$$ 都比较接近，这就是缓存方法利用的冗余。不过这个冗余不是均匀分布的，常见的情况是两头变化大、中间一大段比较平稳，具体还和模型、采样器以及观察的是哪一层有关。后面会看到，几乎所有方法都在想办法刻画"这一步到底变了多少"。

### 缓存残差，而不是输出

先看 TeaCache、SeaCache 这类跳过整个 block 堆栈的方法。它们缓存的不是堆栈的输出 $$h$$，而是残差：

$$
r_t = \mathcal{B}(s_t, e_t) - s_t
$$

跳过的时候，用当前步新算出来的 $$s_{t'}$$ 加上旧的残差：$$\hat h_{t'} = s_{t'} + r_t$$。原因很简单，$$s_{t'}$$ 是从当前的 $$x_{t'}$$ 做 patch embedding 得到的，它本身就是新的，过期的只是 transformer 堆栈贡献的那部分"增量"。直接复用整个 $$h_t$$ 会把输入的变化也一起丢掉，缓存残差则把这部分更新保留了下来。同理，最后的输出头也要用当前步的 $$e_{t'}$$ 来调制。

举个只有一个数的例子：锚点时 $$s_a=10, h_a=13$$，缓存的就是 $$r_a=3$$。下一步输入变成了 $$s_n=11$$，我们用的是 $$11+3=14$$，而不是把 13 原样搬过来。当然，如果残差本身变化很大，这样补也不准，这时就得重算，后面各种判据要解决的正是"什么时候该重算"。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-01-residual-flow.svg" alt="完整计算和缓存命中的数据流：当前输入与输出头仍然计算，只复用 block 堆栈的残差" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. 缓存命中省掉的是中间的 block 计算，当前步的输入、输出头和 scheduler 仍然执行。FBCache 则把缓存边界放在首个 block 之后。
</div>

把这个流程写成代码，大概是下面的样子。不同方法的缓存边界不一样，FBCache 要把首个 block 挪到判断之前，TaylorSeer 原版则把缓存放进了每个 block 内部，但三个接口的含义是相通的：

```python
class CachedDiT(nn.Module):
    def forward(self, x_t, t, ctx, step):
        s = self.patch_embed(x_t)                      # [B, N, D]，很便宜
        e = self.time_embed(t)
        if self.policy.should_compute(s, e, step):     # ① 什么时候重算
            h = s
            for blk in self.blocks:                    # 绝大部分计算量在这里
                h = blk(h, e, ctx)
            r = h - s
            self.policy.update(r, step)                # ② 缓存什么
        else:
            r = self.policy.predict(step)              # ③ 跳过的步怎么补
        return self.head(s + r, e)                     # 输出头用当前步的 s 和 e
```

### 读每种方法时问的几个问题

和上篇读各种并行方式一样，这里也按固定的几个问题去读每种方法：

1. **缓存什么？** 整个 transformer 的残差、某几个 block 的残差、每个 block 里 attention/FFN 的输出，还是只缓存部分 token。粒度越细越灵活，显存也越大。
2. **什么时候重算？** 固定间隔（静态），根据某个指标在线判断（动态），还是离线标定好一张表（全局规划）。
3. **跳过的步怎么补？** 直接复用（零阶保持）、泰勒展开外推，还是线性外推。
4. **步长变不变？** 前面几种方法都是在 scheduler 固定的网格上一个点一个点地走，到了 SoftCache，跳过的步开始直接跨过好几个网格点。

用一张示意图来看几种调度的区别：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-01b-schedule-compare.svg" alt="四种调度在网格点上的对比：固定间隔、阈值动态、离线规划都逐个访问 15 个网格点，只是完整计算的位置不同；SoftCache 跨过部分网格点，只访问 9 个节点" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 3. 前三种调度都逐个访问 15 个网格点，区别只在哪些点完整计算；SoftCache 跨过了一部分网格点，只访问 9 个节点，跳完之后的落点也不再在原网格上。
</div>

把这几篇放在一张表里：

| 方法       | 缓存什么                              | 什么时候重算                               | 跳过的步怎么补        | 离线标定           |
| :--------- | :------------------------------------ | :----------------------------------------- | :-------------------- | :----------------- |
| FBCache    | 第 2~L 个 block 的残差                | 首个 block 残差的相对 L1 超过阈值          | 直接复用              | 不需要             |
| TeaCache   | 整个 transformer 的残差               | 调制输入的相对 L1 经多项式校正后累计超阈值 | 直接复用              | 每个模型拟合多项式 |
| TaylorSeer | 每个 block 的 attn/FFN 输出及各阶差分 | 固定间隔                                   | 泰勒展开外推          | 不需要             |
| SeaCache   | 整个 transformer 的残差               | 频域滤波后的相对 L1 累计超阈值             | 直接复用              | 不需要             |
| DPCache    | 末层特征及各阶差分（官方 FLUX 实现）  | 离线动态规划选出的关键步                   | 泰勒展开外推          | 约 10 条样本       |
| SoftCache  | 最近两次完整计算的残差                | 同 SeaCache                                | 线性外推 + 跨网格跳步 | 不需要             |

按时间顺序读下来，会发现每篇基本只换掉了骨架里的一个零件：FBCache 先算一层来判断后面能不能省；TeaCache 觉得这一层也贵，换成几乎不花钱的调制输入；TaylorSeer 不满足于原样复用，把补值换成了外推；SeaCache 回过头来换掉了衡量"变了多少"的空间；DPCache 把在线判断搬到了离线规划；SoftCache 则让判断的结果去决定下一步走多远。后面就按这个顺序一篇篇看。

### 能省多少：先算一笔账

还是用 Wan2.2 的数字：L=30，T=50。假设 50 步里只有 K 步做完整计算：

- 判断指标几乎不花钱的方法（TeaCache、SeaCache）：代价约为 $$K \cdot L$$ 个 block；
- FBCache 每一步都要先算第一个 block 才能判断：代价是 $$K \cdot L + (T-K) \cdot 1$$。

| K（完整计算的步数） | TeaCache 类 | FBCache        | 加速比（Tea / FB） |
| :------------------ | :---------- | :------------- | :----------------- |
| 50（不缓存）        | 1500        | 1500           | 1.00 / 1.00        |
| 25                  | 750         | 750 + 25 = 775 | 2.00 / 1.94        |
| 20                  | 600         | 600 + 30 = 630 | 2.50 / 2.38        |
| 10                  | 300         | 300 + 40 = 340 | 5.00 / 4.41        |

这张表算的只是 **block 计算量的压缩比**，端到端的时延还要把缓存步本身的开销算进去。记完整一步耗时为 $$C_{\rm full}$$，缓存命中的一步为 $$C_{\rm hit}$$，text encoder、VAE 这些固定耗时为 $$C_{\rm fixed}$$，实际的加速比更接近：

$$
S = \frac{C_{\rm fixed} + T C_{\rm full}}{C_{\rm fixed} + K C_{\rm full} + (T-K) C_{\rm hit}}
$$

比如完整一步 100 ms，缓存步 10 ms，T=50、K=10，即使不算固定开销也只有 $$5000/(1000+400)\approx3.57$$ 倍，而不是 5 倍。判断、读缓存、外推、输出头都要花时间，缓存的比例越高，这部分越显眼。后面的 SoftCache 还会减少实际访问的节点数，那时要另算。

所以在 $$C_{\rm hit}$$ 差不多的前提下，缓存方法之间主要在比：**同样的 K，谁挑的那 K 步更好，谁补的那 T−K 步更准。**

## 怎么做：几种缓存方法

### FBCache：用第一个 block 的残差做探针

FBCache（First Block Cache）最早出现在 [ParaAttention](https://github.com/chengzeyi/ParaAttention) 里，后来 diffusers 也加了 `FirstBlockCacheConfig`。思路很直接：不知道这一步整个网络的输出会变多少，那就先算第一个 block，看第一个 block 的残差变了多少。如果变化很小，就认为后面的 block 变化也很小，剩下的 L−1 个 block 全部跳过，直接加上上次缓存的残差。

判断用的是首个 block 残差的相对 L1：

$$
\delta_t = \frac{\operatorname{mean}\lvert r^{(1)}_t - r^{(1)}_{a} \rvert}{\operatorname{mean}\lvert r^{(1)}_{a} \rvert}
$$

其中 $$a$$ 是锚点，$$\delta_t$$ 小于阈值就跳过。

参照 ParaAttention 的实现简化后的代码（去掉了 FLUX 双流里 encoder_hidden_states 的处理）：

```python
def are_similar(prev, cur, threshold, cp_group=None):
    # 用 sum 再相除，CP 分片长度不同时也能按元素数正确加权
    stats = torch.stack([(cur.float() - prev.float()).abs().sum(),
                         prev.float().abs().sum()])
    if cp_group is not None:                  # 合成全局分子、分母，一次 collective
        dist.all_reduce(stats, group=cp_group)
    return (stats[0] / stats[1].clamp_min(1e-6)).item() < threshold

class FBCacheBlocks(nn.Module):
    def __init__(self, blocks, threshold=0.1):
        super().__init__()
        self.blocks = blocks
        self.threshold = threshold
        self.prev_r1 = None    # 上一次完整计算时，首个 block 的残差
        self.r_rest = None     # 上一次完整计算时，剩下 L-1 个 block 贡献的残差

    def forward(self, h, e, ctx, cp_group=None):
        h_in = h
        h = self.blocks[0](h, e, ctx)                 # 首个 block 每一步都要算
        r1 = h - h_in
        if self.prev_r1 is not None and are_similar(self.prev_r1, r1, self.threshold, cp_group):
            return h + self.r_rest                    # 命中：后面的 block 全部跳过
        self.prev_r1 = r1                             # 注意：只有完整计算时才更新
        h1 = h
        for blk in self.blocks[1:]:
            h = blk(h, e, ctx)
        self.r_rest = h - h1
        return h
```

有几个细节值得注意：

- **`prev_r1` 只在完整计算时更新。** 也就是说每次都是和锚点比，而不是和上一步比。连续跳过的时候，偏离锚点的程度会越来越大，最终超过阈值触发重算，相当于隐式地做了累计。ParaAttention 源码里 `set_first_hidden_states_residual` 只出现在 `else` 分支。
- **FLUX 这种双流结构要缓存两份残差。** image token 和 text token 各有一份 `hidden_states_residual` 和 `encoder_hidden_states_residual`，命中时两份都要加回去。
- **代价是每步固定多一个 block，外加一次同步。** 一个 block 对 30 层的 Wan2.2 来说是 1/30。更麻烦的是判断结果要 `.item()` 拿回 CPU 才能决定走哪个分支，这是一次强制同步，整步前向也没法 capture 成一张 CUDA Graph。ParaAttention 在判断的前后都显式地调用了 `torch._dynamo.graph_break()`。
- **被跳过的区域里有 collective 时，所有 rank 必须做出同样的决定。** 普通 TP 在 All-Reduce 之后，每个 rank 拿到的是同一份完整的 residual，算出来的 $$\delta$$ 自然相同；但 CP 或者 TP+SP 下每个 rank 只有一部分 token，局部的 $$\delta$$ 各不相同。一旦有的 rank 跳过、有的 rank 计算，计算的那个就会卡死在下一次 collective 上。所以要先把分子、分母归约成全局的统计量再比较，ParaAttention 的并行路径就是这么做的。上面的示例把两个 sum 拼成一个张量，只做一次 all-reduce，统计用 FP32；如果有 padding，还要把无效的 token 排除掉。

阈值一般在 0.05~0.2 这个量级，按模型调。cache-dit 在这个基础上做了 DBCache，把"首个 block"推广成前 Fn 个 block 用来做判断、后 Bn 个 block 始终计算用来修正，FBCache 可以看成 F1B0 的情况。

### TeaCache：用 timestep 调制后的输入做探针

TeaCache（Timestep Embedding Aware Cache）出自 CVPR 2025 的 [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108)。它想做的是：FBCache 毕竟还要算一个 block，能不能连这个 block 都省掉，只看输入就做判断？

论文的分析过程大概是这样的：

1. 真正想知道的是相邻两步**输出**的差异，但输出要算完才知道；
2. 直接用输入 $$x_t$$（加噪的 latent）的差异，和输出差异的相关性很差，因为 $$x_t$$ 相邻两步几乎不变；
3. DiT 里 timestep 是通过 AdaLN 调制进去的，第一个 block 的 `norm1(x) * (1 + scale(t)) + shift(t)` 同时包含了输入和 timestep 的信息，用它的差异就和输出差异的趋势接近多了；
4. 趋势接近，但数值上还有偏差，于是再离线拟合一个多项式，把输入差异映射成输出差异。

所以 TeaCache 每一步做的事情是：

$$
\Delta_t = \frac{\lVert I_t - I_{t-1} \rVert_1}{\lVert I_{t-1} \rVert_1}, \quad
A_t = \sum_{s=a+1}^{t} f(\Delta_s)
$$

其中 $$I_t$$ 是第一个 block 的调制输入，$$f$$ 是拟合出来的 4 次多项式，$$a$$ 是锚点。$$A_t$$ 小于阈值就复用整个 transformer 的残差，否则重算并把 $$A_t$$ 清零。

参照 TeaCache4FLUX 简化后的代码：

```python
class TeaCachePolicy:
    def __init__(self, first_block, coeffs, thresh, num_steps):
        self.norm1 = first_block.norm1          # 只借用第一个 block 的 AdaLN
        self.rescale = np.poly1d(coeffs)        # 离线拟合，每个模型一套系数
        self.thresh = thresh
        self.num_steps = num_steps
        self.acc = 0.0
        self.prev_inp = None
        self.residual = None

    def should_compute(self, s, e, step):
        inp = self.norm1(s, emb=e)[0]           # timestep 调制后的输入，几乎不花时间
        if step == 0 or step == self.num_steps - 1:
            calc, self.acc = True, 0.0          # 第一步和最后一步强制计算
        else:
            rel = (inp - self.prev_inp).abs().mean() / self.prev_inp.abs().mean()
            self.acc += self.rescale(rel.item())
            calc = self.acc >= self.thresh
            if calc:
                self.acc = 0.0
        self.prev_inp = inp                     # 注意：每一步都更新
        return calc

    def update(self, r, step):
        self.residual = r

    def predict(self, step):
        return self.residual
```

和 FBCache 对比一下：FBCache 是和锚点比、不做累加；TeaCache 是和上一步比，然后把每一步的增量加起来。两种写法都是在估计"离上次完整计算已经漂了多远"。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-02-probe-anchor.svg" alt="FBCache 固定与完整计算的锚点比较，TeaCache 每步更新探针并累计变化；两者都只在重算时更新残差" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 4. 探针和残差是两套状态。TeaCache 每步更新的是探针，不是拿来补 block 输出的残差；这一点很容易混。
</div>

累计量 $$A_t$$ 只是一个触发重算的信号，并不是输出误差的上界：分母一直在变，多项式也是拟合出来的，$$A_t<\tau$$ 并不意味着输出误差一定小于 τ。另外在拟合区间之外，多项式可能给出负值，实现时最好夹一下，保证累计量单调。

多项式是离线拟合的，过程也很简单：在一些 prompt 上跑完整推理，把每一步输入侧和输出侧的相对 L1 记下来做回归：

```python
xs, ys = [], []
for prompt in calib_prompts:
    prev_inp = prev_out = None
    for step, t in enumerate(timesteps):
        inp, out = run_full_step(prompt, t)     # inp: 调制输入, out: 模型输出
        if prev_inp is not None:
            xs.append(rel_l1(inp, prev_inp))
            ys.append(rel_l1(out, prev_out))
        prev_inp, prev_out = inp, out
coeffs = np.polyfit(xs, ys, deg=4)
```

TeaCache4FLUX 的代码注释里给了阈值和加速比的对应关系：`0.25 → 1.5x，0.4 → 1.8x，0.6 → 2.0x，0.8 → 2.25x`。

视频模型的版本里还有两个值得看的细节：

- **CFG 的两条分支要各自维护一套状态。** Wan2.1 推理时 cond、uncond 是交替调用同一个模型的，TeaCache4Wan2.1 用 `cnt % 2` 区分奇偶，`accumulated_rel_l1_distance`、`previous_e0`、`previous_residual` 都有 even/odd 两份。如果共用一份，就会拿 cond 的输入去和 uncond 的输入比，判断完全是错的。ParaAttention 里的 `enable_alter_cache` 做的也是同一件事。如果像上篇那样做了 CFG 并行，两条分支本来就在不同的 plane 上，缓存状态天然就是分开的。
- **Wan2.1 版本的探针其实只和 timestep 有关。** [官方实现](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/teacache_generate.py)用的是 `modulated_inp = e0 if self.use_ref_steps else e`，`e` 和 `e0` 都只是 timestep 的函数。也就是说步数、shift 和阈值给定之后，哪些步算、哪些步跳就已经定死了，和 prompt、输入图都没有关系，完全可以预先算成一张静态的调度表。它另外还有 `ret_steps`（开头保留计算）和 `cutoff_steps`，这两个计数是按 forward 调用算的，cond、uncond 各算一次，所以是 scheduler 步数的两倍。

TeaCache 的问题也比较明显：多项式要按模型拟合，换模型、换分辨率、换步数都可能要重新拟合；跳过的步直接复用整块残差，跳得越远误差越大。后面几篇基本就是围绕这两点在改。

### TaylorSeer：从"复用"到"预测"

[TaylorSeer](https://arxiv.org/abs/2503.06923)（ICCV 2025）换了一个角度。前面的方法都在想"什么时候可以复用"，但复用本身是一个零阶近似 $$F(t+k) \approx F(t)$$，跳得越远误差越大。既然特征沿着时间步是平滑变化的，就不应该原样复用，而是用前几次完整计算的结果把变化趋势外推出来。

用的工具是泰勒展开，导数用有限差分来近似。论文里时间是递减的，符号容易绕，这里统一按 step 递增来写。最近一次完整计算在 a，上一次在 b，$$a>b$$：

$$
D_a^{(0)}=F_a, \quad D_a^{(i)} \approx \frac{D_a^{(i-1)}-D_b^{(i-1)}}{a-b}
$$

往后 k 步的特征预测为：

$$
\hat F_{a+k}=\sum_{i=0}^{m}\frac{D_a^{(i)}}{i!}k^i
$$

固定间隔时 $$a-b=N$$，缩放已经包含在 $$D_a^{(i)}$$ 里了，预测时不用再除一次 N。m=0 就退化成直接复用，m=1 是线性外推，一般 m 取 1~2。这里的高阶"导数"都是递归差分出来的近似值，并不是对历史点做精确的多项式插值。

比如 step 2 完整算得 F=10，step 4 完整算得 F=14，那么每步的变化估计为 $$(14-10)/(4-2)=2$$。到了 step 5，直接复用得到 14，一阶外推得到 $$14+2\times1=16$$。先用这个例子理解，再看高阶项就容易多了。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-03-taylor-predict.svg" alt="用 step 2 和 step 4 的完整计算估计每步变化，在 step 5 对比零阶复用和一阶外推" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 5. TaylorSeer 用真实计算更新差分，缓存步只读取差分做外推。图中的数是讲解用的标量，不是模型测量结果。
</div>

实现上，每个缓存位置（某一层的某个模块）维护一个"泰勒因子"列表，第 i 个元素是 i 阶导的估计。下面的写法和源码里 `derivative_approximation`、`taylor_formula` 的逻辑一致：

```python
class TaylorCache:
    """某一个 (layer, module) 位置的特征及其 0~m 阶差分"""
    def __init__(self, max_order=2):
        self.max_order = max_order
        self.factors = []          # factors[i] ≈ F 对 step 的 i 阶导
        self.last_step = None

    def update(self, feature, step):       # 完整计算时调用
        new = [feature]
        if self.last_step is not None:
            dist = step - self.last_step   # 两次完整计算之间隔了几步
            for i in range(min(self.max_order, len(self.factors))):
                new.append((new[i] - self.factors[i]) / dist)
        self.factors, self.last_step = new, step

    def predict(self, step):               # 跳过的步调用
        k = step - self.last_step
        return sum(f * k ** i / math.factorial(i) for i, f in enumerate(self.factors))
```

可以看到阶数是一点点"长"出来的：第一次完整计算只有 0 阶，第二次有了 1 阶，第三次才有 2 阶。调度是静态的，前几步强制完整计算（源码里叫 `first_enhance`），之后每隔 N 步（`fresh_threshold`）完整计算一次：

```python
def is_full_step(step, warmup=3, interval=5):
    return step < warmup or (step - (warmup - 1)) % interval == 0
```

缓存的粒度比前面细得多。在 FLUX 的实现里，每个 block 的 attention 输出和 MLP 输出是分别缓存的（双流 block 里 image、text 各一份）。跳过的步里 block 还是要"走一遍"，只是把 attention 和 MLP 换成预测值，modulation、gate、残差相加这些便宜的操作照常算：

```python
def block_forward(self, x, e, ctx, step, full):
    shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(e)   # 便宜，每步都算
    if full:
        attn = self.attn(modulate(self.norm1(x), shift1, scale1), ctx)
        self.attn_cache.update(attn, step)
    else:
        attn = self.attn_cache.predict(step)
    x = x + gate1 * attn
    if full:
        mlp = self.mlp(modulate(self.norm2(x), shift2, scale2))
        self.mlp_cache.update(mlp, step)
    else:
        mlp = self.mlp_cache.predict(step)
    return x + gate2 * mlp
```

[官方仓库](https://github.com/Shenyi-Z/TaylorSeer)给的数字是：FLUX 上计算量压缩 4.99x，对应的实际时延加速是 3.53x；HunyuanVideo 上分别是 5.00x 和 4.65x。计算量和时延之间的这段差距，就是前面说的缓存步本身的开销。

但这个粒度的代价是显存。按 Wan2.2 算一下：一个 `[1, 27280, 3072]` 的 BF16 张量是 159.84 MiB，如果每个 block 缓存 self-attention 和 FFN 两个输出，2 阶要存 3 个张量：

$$
30 \times 2 \times 3 \times 159.84\ \text{MiB} \approx 28\ \text{GiB}
$$

再算上 CFG 的两条分支就是 56 GiB，是模型权重（约 10 GB）的好几倍。所以在视频模型上，更实际的做法是把泰勒预测用在更粗的粒度上，比如只对整个 transformer 的残差做预测，这样一共只要 3 个张量。ParaAttention 里 FBCache + TaylorSeer 的组合就是这么做的：开启 TaylorSeer 之后，`set_hidden_states_residual` 变成 `taylorseer.update(...)`，命中缓存时加回去的不再是原样的残差，而是 `taylorseer.approximate_value()` 外推出来的残差。

还有三个容易忽略的点：

- **阶数不是越高越好。** 高阶差分会把特征里的噪声不断放大，间隔 N 大的时候泰勒余项也大，实践中 1~2 阶就够了。
- **源码里的"距离"用的是 step 下标，不是真实的 t。** `difference_distance = activated_steps[-1] - activated_steps[-2]`，下面还有一行注释掉的用 `activated_times` 的版本。在固定网格上这没有问题，相当于把特征看成 step 的函数，即使 scheduler 加了 shift 也一样；但如果要在网格之外预测（比如后面的 SoftCache），就得换成连续的时间坐标。
- **预测值不要写回去当作真实观测。** 差分只在完整计算的步更新，否则下一次外推就建立在上一次猜出来的结果上，误差会不断反馈进差分里。

### SeaCache：在频域里衡量"变了多少"

[SeaCache](https://arxiv.org/abs/2602.18993)（Spectral-Evolution-Aware Cache）关注的是指标本身。TeaCache 这一类方法都是直接在原始特征上算相对 L1，但扩散过程中的特征其实混着两样东西：和最终内容对齐的信号，以及 latent 里还没去掉的噪声分量。在原始特征上算距离，两者的变化搅在一起，于是可能出现"内容没怎么变，距离却很大"，或者反过来的情况。

它借助的先验是扩散模型的谱演化（spectral evolution）：早期的步先生成低频结构，后期才去补高频细节。论文先做了一个 oracle 实验：直接用相邻两步**输出**的距离来决定缓存（这样就不存在"用输入估计输出"的误差），对比原始输出距离和滤波后输出距离两种判据。在同样的重算比例下，滤波后的判据 PSNR 一直更高。这说明问题不只是"用输入估计输出不准"，而是"原始特征空间本身就不是衡量冗余的好空间"。

那用什么滤波器呢？论文从最优线性去噪器推了一个出来。前向过程是 $$x_t = a_t x_0 + b_t \varepsilon$$，找一个线性滤波器 $$h_t$$ 让 $$\lVert h_t * x_t - x_0 \rVert^2$$ 最小。到频域里每个频率是独立的，求导就得到 Wiener 滤波的形式：

$$
G_t(f) = \frac{a_t S_x(f)}{a_t^2 S_x(f) + b_t^2}
$$

其中 $$S_x(f)$$ 是干净数据的功率谱，噪声是白噪声，功率谱为常数 1。再假设自然图像的功率谱服从幂律 $$S_x(f) \approx A \lvert f \rvert^{-\beta}$$（论文取 A=1，图像 β=2，视频 β=3），这个滤波器就只和 $$a_t, b_t$$ 有关了：

- 早期 $$a_t \to 0$$，只有最低频能通过；
- 越往后 $$a_t$$ 越大，通带越宽，高频逐渐被放进来。

这正好就是谱演化。DDPM 里 $$a_t=\sqrt{\bar\alpha_t}, b_t=\sqrt{1-\bar\alpha_t}$$，rectified flow 里 $$a_t = 1-\sigma_t, b_t = \sigma_t$$。

还要做一步归一化：不同 t 的 $$G_t$$ 整体增益不一样，直接用的话不同时间步之间的距离没法比，所以按径向频率分 bin，把平均增益归一化成 1：

$$
G_t^{\text{norm}}(f) = \nu_t G_t(f), \quad \nu_t = \Big(\frac{1}{L}\sum_{\ell=0}^{L-1} G_t(f_\ell)\Big)^{-1}
$$

最后的指标就是滤波之后的相对 L1，累计和阈值的规则和 TeaCache 完全一样，**而且不再需要多项式拟合**：

$$
\tilde\Delta_t = \mathrm{L1}_{\mathrm{rel}}\big(\mathcal{P}(G_t^{\text{norm}}, I_t),\ \mathcal{P}(G_{t+1}^{\text{norm}}, I_{t+1})\big), \quad
\mathcal{P}(G, I) = \mathrm{iFFT}\big(G \odot \mathrm{FFT}(I)\big)
$$

要注意，这里滤的是**用来做判断的探针**，不是把缓存的残差滤一遍再送回模型。可以理解成换了一副眼镜去看特征有没有变，模型的 Attention、FFN 和权重都没有动。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-04-sea-probe.svg" alt="SeaCache 的探针路径：调制输入还原空间网格，FFT 后乘时间相关滤波器，iFFT 后计算距离并累计判定" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 6. SEA 只改变判定特征。相邻节点各用各自时间的滤波器，再比较投影后的距离。
</div>

实现时 $$I_t$$ 是 token 序列 `[B, N, C]`，要按 patch 的真实顺序还原成空间网格（视频是 `F × H × W`）再做 FFT。FFT 是沿空间轴做的，不是沿 hidden 通道；text token 和 padding 也要先剔掉，不能混进网格里。

论文里写的是径向分 bin 的归一化，而[官方 `util_seacache.py`](https://github.com/jiwoogit/SeaCache/blob/main/FLUX/util_seacache.py)用的是另一种离散实现：每根轴单独构造一维响应，再乘起来得到一个可分离的滤波器，然后按整个频谱网格的均值归一化。下面按官方代码的思路来写，前面的径向公式用来理解就好：

```python
def sea_filter(a_t, b_t, grid, beta=2.0, device="cuda"):
    """grid = (H, W) 或 (F, H, W)，返回完整 FFT 网格上的可分离响应"""
    a, b = max(float(a_t), 1e-6), float(b_t)     # 避免纯噪声端点 a=0，归一化变成 0/0
    G = torch.ones(grid, device=device, dtype=torch.float32)
    for axis, n in enumerate(grid):
        f = torch.fft.fftfreq(n, device=device).abs()
        S = 1.0 / (f ** beta + 1e-16)           # DC 处也用 eps 保护
        g = a * S / (a * a * S + b * b + 1e-16)
        shape = [1] * len(grid)
        shape[axis] = n
        G = G * g.reshape(shape)
    return G / G.mean().clamp_min(1e-16)

def sea_project(I, G, grid):
    B, N, C = I.shape                                # token 序列还原成空间网格
    x = I.transpose(1, 2).reshape(B, C, *grid).float()
    dims = tuple(range(2, 2 + len(grid)))
    return torch.fft.ifftn(torch.fft.fftn(x, dim=dims) * G, dim=dims).real

class SeaCachePolicy:
    def __init__(self, grid, filters, delta):
        self.grid, self.filters, self.delta = grid, filters, delta
        self.acc, self.prev, self.residual = 0.0, None, None

    def should_compute(self, I, step, force=False):
        P = sea_project(I, self.filters[step], self.grid)  # 固定网格的滤波器预先生成
        if self.prev is None or force:
            calc = True
        else:
            self.acc += ((P - self.prev).abs().mean() / (self.prev.abs().mean() + 1e-6)).item()
            calc = self.acc >= self.delta
        if calc:
            self.acc = 0.0
        self.prev = P                                # 每步都更新，各自用自己那一步的滤波器
        return calc
```

几个实现上的小细节：示例里投影后的特征保留了 FP32，官方代码会转回输入的 dtype，要精确复现得对齐；想用 `rfftn` 省一半频域存储也可以，但归一化时要把半谱的对称权重补上；纯噪声端点 $$a_t=0$$ 会让归一化变成 0/0，官方是在 scheduler 那边把 σ 夹住，示例里直接在 a 上加了保护，解决的是同一个问题。

分辨率和时间表固定时，所有步的滤波器都可以在初始化时预先算好，在线只剩 FFT、逐元素乘、iFFT 和一次距离归约。论文附录测的开销是：FLUX 上 0.058 s / 9.4 s（约 0.6%），HunyuanVideo 上 0.362 s / 90.8 s（约 0.4%），基本可以忽略。

几个我觉得有意思的结论：

- **开头保留几步不太需要手调了。** 滤波后的距离在早期天然就大，重算会自然集中到前面。不过[公开的 FLUX 脚本](https://github.com/jiwoogit/SeaCache/blob/main/FLUX/seacache_generate.py)还是强制计算了首尾两步，边界保护并没有完全去掉。
- **时间相关性是关键。** 消融里换成固定的低通滤波（只保留最低 30% 的频率）效果明显变差，去掉归一化也会变差。
- **即插即用。** 它只替换了距离的计算，论文里把 SEA 滤波套到 DiCache 的 block 级特征上同样有提升。

论文自己也提到了一个局限：推导是在像素空间、按自然图像的幂律谱做的，而现在的模型都跑在 VAE 的 latent 空间里，latent 的功率谱不一定满足这个假设。

还有一个论文没有讨论、但落地时要想清楚的问题：CP 下每个 rank 只拿到一段 token（视频一般是按帧切的），而 3D FFT 需要完整的网格。要么把探针 gather 起来再做 FFT（Wan2.2 上一次就是上百 MiB 的通信），要么每个 rank 只对本地那段做 FFT，再把分子分母 all-reduce，后者是一种近似。

### DPCache：把挑关键步当成路径规划

[DPCache](https://arxiv.org/abs/2602.22654)（Denoising as Path Planning）关注的是"什么时候重算"。作者把已有方法分成两类：

- 固定调度（比如 TaylorSeer 每 N 步算一次）：不管轨迹在哪里变化剧烈，一视同仁；
- 局部自适应（TeaCache、FBCache 这类阈值判断）：每一步只看眼前，本质是贪心，一旦在关键的时间步跳过了，后面的偏差就回不来了。

DPCache 的出发点是之前工作里的一个观察：**去噪轨迹的形状主要取决于模型本身，和生成的具体内容关系不大。** 那就可以在少量样本上离线把"跳哪些步代价最小"算清楚，推理时直接查表。

#### 路径相关的代价张量

最直观的做法是一个二维代价矩阵 `C[j, k]`，表示从关键步 j 直接跳到下一个关键步 k 的误差。但如果跳过的步是用泰勒外推来预测的，预测值不只取决于 j，还取决于 j 前面的那个关键步 i（1 阶差分就是用 i、j 两次完整计算的结果算出来的）。所以代价是路径相关的，要用一个三维张量，论文叫 PACT（Path-Aware Cost Tensor）：

$$
\mathcal{C}[i,j,k] = \sum_{\tau=k}^{j-1} \big\lVert h_\tau^{L} - h_{\text{pred},\tau}^{L}(i, j) \big\rVert_1, \quad i > j > k
$$

也就是：前一个关键步是 i、当前关键步是 j 时，一路预测到下一个关键步 k，中间所有步的误差之和。为了省显存，只用最后一层的特征 $$h^L$$ 来算。

这里沿用论文的倒序编号 $$T\to\cdots\to1$$，0 是终点哨兵，并不对应一次真实的模型调用。另外公式把 k 点自身的预测误差也算进了段代价，所以 PACT 更像是一个用来选表的代理目标，和线上真实的缓存误差并不完全相等。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-05-dp-path.svg" alt="DPCache 离线用完整轨迹构造三元代价 C[i,j,k]，选出关键步表，在线保留全部 scheduler 更新" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 7. 从同一个 j 出发，前驱 i 不同，预测趋势也不同。离线选的是完整计算的位置，线上其余位置仍做预测和 scheduler 更新。
</div>

构造的时候，对固定的 (i, j)，往后每一步的预测误差只需要算一次，C 就是它的前缀和。下面先用一阶、两个锚点的版本说明这个过程：

```python
# h[T] ... h[1]: 完整标定轨迹的末层特征
# h[0]: 事先构造的终点代理特征，不是一次真实 forward；多样本的 C 再取平均
C = np.full((T + 1, T + 1, T + 1), np.inf)
for i in range(T, 1, -1):
    for j in range(i - 1, 0, -1):
        slope = (h[j] - h[i]) / (j - i)
        acc = 0.0
        for k in range(j - 1, -1, -1):
            pred = h[j] + (k - j) * slope
            acc += (h[k] - pred).abs().mean().item()
            C[i, j, k] = acc
```

论文实验用的是 2 阶预测。[公开的标定代码](https://github.com/argsss/DPCache/blob/main/dpcache/cali_utils.py)还借用了前驱附近连续几步的完整特征来估计高阶差分，并给哨兵构造了代理目标。复现的时候，这些约定和预测器的阶数、dtype 都得对齐。

#### 动态规划

有了 PACT，选 K 个关键步 $$\{t_1=T > t_2 > \dots > t_K > 0\}$$ 就是最小化整条路径的代价：

$$
\min_{\mathcal{T}} \sum_{m=2}^{K} \mathcal{C}[t_{m-1}, t_m, t_{m+1}], \quad t_{K+1}=0
$$

用两张表：`D[m][k]` 是恰好用 m 个关键步走到 k 的最小代价，`P[m][k]` 记录 k 的前驱。前 M 步固定为关键步（论文取 M=3），递推式为：

$$
D[m,k] = \min_{j>k} \Big( D[m-1,j] + \mathcal{C}\big[P[m-1,j],\ j,\ k\big] \Big)
$$

```python
def dp_schedule(C, T, K, M=3):
    assert 2 <= M <= K <= T
    INF = float("inf")
    D = [[INF] * (T + 1) for _ in range(K + 2)]
    P = [[None] * (T + 1) for _ in range(K + 2)]
    for m in range(1, M + 1):                   # 前 M 步固定：T, T-1, ..., T-M+1
        k = T - m + 1
        D[m][k], P[m][k] = 0.0, k + 1
    for m in range(M + 1, K + 2):
        for k in range(T - m + 1, -1, -1):
            for j in range(k + 1, T + 1):
                if D[m - 1][j] == INF:
                    continue
                cost = D[m - 1][j] + C[P[m - 1][j], j, k]
                if cost < D[m][k]:
                    D[m][k], P[m][k] = cost, j
    if D[K + 1][0] == INF:
        raise ValueError("给定预算与代价张量下没有可用路径")
    sched, k = [], 0                            # 从哨兵 t_{K+1} = 0 往回找
    for m in range(K + 1, 1, -1):
        k = P[m][k]
        sched.append(k)
    return sched[::-1]                          # [t_1 = T, t_2, ..., t_K]
```

DP 本身的复杂度是 $$O(KT^2)$$，构造代价张量有 $$O(T^3)$$ 项，每项还要算一次特征误差。不过真正花时间的是收集完整的标定轨迹，查这张小表几乎不花时间。推理时只在选出的关键步完整计算，其余步做泰勒预测，scheduler 照常逐点更新。论文用大约 10 条样本标定，在 FLUX 上 4.87x 时 ImageReward 比对比方法里最好的高 0.031，3.54x 时甚至比不加速的基线还高 0.028。

缓存的粒度也要和 TaylorSeer 原版区分开。[官方 FLUX 实现](https://github.com/argsss/DPCache/blob/main/dpcache/cache_utils.py)只在各个 stream 的最后一层保存输出和差分，命中时直接在末层填入预测特征，并没有给每个 attention/FFN 都存一套。可见"用泰勒预测"和"缓存放在哪里"是两个独立的选择。

有两点是我自己的理解：

- **这个压缩状态的递推，对一般的路径相关代价不保证精确最优。** 举个例子：有两条路径都到达 j，已有代价分别是 1 和 2，前驱不同；从 j 去 k 的新增代价分别是 100 和 0。递推只保留已有代价为 1 的那条，最后得到 101，错过了总代价只有 2 的那条。这是对递推结构的分析，不是在论文真实 PACT 上测出来的退化。要精确优化给定的三元代价，状态得保留最后两个关键步，递推变成 $$E[m,j,k]=\min_{i>j}\{E[m-1,i,j]+\mathcal C[i,j,k]\}$$，复杂度 $$O(KT^3)$$，T=50 时也完全能接受。当然，即使这样求出的也只是 PACT 这个代理目标的最优。
- **静态调度更容易放进推理系统。** 调度表在推理前就定好了，不需要逐步做探针判断，也就没有对应的 CPU 同步，组织 CUDA Graph 和多卡执行都简单得多。不过公开实现里还留着用来计数的 `.item()`，接入时要顺手清掉。代价是放弃了逐样本的自适应，"轨迹和内容无关"这个假设在一些极端 prompt 上不一定成立。另外 PACT 和预测器、步数、数值精度都是绑定的，SoftCache 那篇论文复现 DPCache 时，就把用 fp16 Taylor 状态标定的 FLUX 代价张量按 bf16 重新标定了一遍。

### SoftCache：不只决定跳不跳，还决定跳多远

[SoftCache](https://arxiv.org/abs/2610.22369) 是这几篇里最新的，它发现动态缓存的累计信号里有一部分信息被浪费了。

以 TeaCache/SeaCache 为例，累计量 $$A_i$$ 只被用来和阈值 τ 比一下：小于 τ 就复用，否则重算。刚刚重算完（$$A_i$$ 很小）和马上就要重算（$$A_i$$ 接近 τ）的两个节点被当成完全一样的情况处理，而且都只往前走 scheduler 规定的那一步。论文在不加缓存的 FLUX 轨迹上测了复用误差 $$E_i = \lVert r_i - r_a \rVert / \lVert r_a \rVert$$，发现它随 $$A_i$$ 单调变大，刚重算完和快要重算的节点，中位数误差能差 2.2 倍。

于是定义一个余量（headroom）：

$$
H_i = 1 - \frac{A_i}{\tau} \in [0, 1]
$$

余量大说明缓存还很可靠，这一步就可以走得更远。SoftCache 用 flow matching 的 ODE 形式来描述：t 从 0（噪声）走到 1（数据），Euler 一步是 $$x_{i+1} = x_i + \Delta_i v_\theta(x_i, t_i)$$。缓存步的步长设为：

$$
\bar\Delta_i =
\begin{cases}
\Delta_i, & A_i \ge \tau \ \text{（重算）} \\
\min\{1 - t_i,\ \Delta_i + (J-1) H_i \Delta_i\}, & A_i < \tau \ \text{（复用）}
\end{cases}
$$

J 是一步最多能跨几个原始步长（论文取 J=8）。余量越大步子越大，快要重算的时候步长缩回原来的 $$\Delta_i$$。

步子一大就带来了两个新问题：

1. **落点不在原来的网格上，而且离锚点更远，零阶复用的误差更大。** 所以复用时不再原样用 $$r_a$$，而是沿着最近两次完整计算的残差做线性外推：

   $$
   \hat r_i = r_a + \beta \lambda_i (r_a - r_{a-1}), \quad \lambda_i = \frac{t_i - t_a}{t_a - t_{a-1}}
   $$

   $$\lambda_i$$ 是用真实的时间间隔算的，所以在网格外也有定义（对比一下 TaylorSeer 用的是 step 下标）。β 在 4x 以下取 1，4x 以上取 0.5，只多存一个张量 $$r_{a-1}$$。

2. **后面的网格要重排。** 跳完之后，落点 $$t_{i+1}$$ 和下一个旧网格点之间可能只差零点几步，继续按旧网格走就浪费了一次计算。做法是数一下旧网格里 $$t_{i+1}$$ 之后还剩 $$m_i$$ 个点，把这 $$m_i$$ 个点（$$t_{i+1}$$ 算第一个）均匀地铺到 $$[t_{i+1}, 1]$$ 上，被跨过的那些网格点就真的从轨迹里去掉了。

另外还有两个保护：t < 0.1 的开头阶段不跳；当前节点的探针变化本身很大（$$\rho_i > 0.18$$）时也不跳。这里的"不跳"只是把步长恢复成原来的 $$\Delta_i$$，要不要重算 block 还是看累计量。探针直接用的是 SeaCache 的 SEA 滤波特征。

还要说明一点，headroom 和复用误差之间的关系是论文观察到的相关性，步长规则是据此设计出来的，并不是 ODE 意义上的局部截断误差估计，也没有"跨这么远一定满足某个误差容限"的保证。

用一组数就能看清楚：$$t_i=0.2,\ \Delta_i=0.1,\ A_i/\tau=0.5,\ J=8$$，那么 $$H_i=0.5$$，这一步走 $$0.1+7\times0.5\times0.1=0.45$$，落到 0.65。旧网格里的 0.3、0.4、0.5、0.6 都不用再访问了；之后的旧点是 0.7、0.8、0.9、1.0，共 4 个，重排成 0.65、约 0.767、约 0.883、1.0，落点也算在这 4 个里面。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI/diffusion-feature-cache/fig-06-softcache-grid.svg" alt="SoftCache 从 0.2 跳到 0.65，跨过四个旧节点，再把后续四个节点均匀重排到 0.65 至 1.0" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 8. 普通缓存仍要到每一个旧网格点，只省 block；SoftCache 连被跨过节点的探针、输出头和 scheduler 更新也省掉了。
</div>

整理成伪代码（对应论文里的 Alg. A.1.1）：

```python
def softcache_sample(x, grid, model, tau, J=8, beta=1.0):
    # grid: 0 = t_0 < t_1 < ... < t_N = 1（本文采样坐标，0 是噪声，1 是数据）
    # model.embed/head 已适配这个坐标；ctx 固定并封装在 model 内
    grid = list(grid)
    i, A = 0, 0.0
    r_a = t_a = r_prev = t_prev = None
    prev_probe = None
    while grid[i] < 1.0:
        t, dt = grid[i], grid[i + 1] - grid[i]
        s, e = model.embed(x, t)                     # patch embedding + timestep embedding
        probe = model.sea_probe(s, e, t)             # SeaCache 的滤波探针
        rho = 0.0 if prev_probe is None else rel_l1(probe, prev_probe)
        prev_probe = probe
        if i > 0:
            A += rho
        if i == 0 or A >= tau:                       # 重算：跑一遍所有 block
            r_prev, t_prev = r_a, t_a
            r_a, t_a = model.blocks(s, e) - s, t
            r_hat, step, A = r_a, dt, 0.0
        else:                                        # 复用：外推残差，拉长步长
            if r_prev is not None:
                r_hat = r_a + beta * (t - t_a) / (t_a - t_prev) * (r_a - r_prev)
            else:
                r_hat = r_a
            H = 1.0 - A / tau
            step = min(1.0 - t, dt + (J - 1) * H * dt)
            if t < 0.1 or rho > 0.18:                # 两个保护：开头不跳，变化大不跳
                step = dt
        x = x + step * model.head(s + r_hat, e)      # Euler
        t_next = t + step
        if step > dt:                                # 跨过了网格点：重排后面的节点
            m = sum(1 for g in grid if g > t_next)   # 包含终点 1；落点也占用一个剩余节点
            tail = np.linspace(t_next, 1.0, m).tolist() if m >= 2 else [t_next, 1.0]
            grid = grid[: i + 1] + (tail if t_next < 1.0 else [1.0])
        i += 1
    return x
```

SoftCache 本质上是把"缓存"和"自适应步长"合到了一起，加速不只来自跳过 block，还来自少走了网格点。论文报告的加速在 3.5x~6.9x 之间，在测的几个速度档位上四项保真度指标都更好，其中视频实验对比的是 SeaCache 和 DPCache。τ 是主要用来调速度的参数，J、β 和两个保护阈值在论文里是固定的，β 会随速度区间切换。

工程上的代价是时间网格在推理过程中会变，scheduler 必须支持在任意 t 上走一步。Euler 没有问题，但像 UniPC、DPM-Solver++ 这种多步求解器依赖历史步的间隔，就需要额外处理，论文里跑 Wan2.1 时也是把原本的 UniPC 换成了 Euler。另外节点数是动态的，那些按 step 下标写的逻辑（比如只在某几步开 CFG、guidance 随步数变化）都要改成按 t 来写。

接入 diffusers 时还要注意坐标方向。论文里 t 从噪声 0 走到数据 1，而 diffusers 的 flow scheduler 大多是 σ 从 1 走到 0，$$t=1-\sigma$$、送进模型的 timestep 缩放、速度的符号都要核对一遍，上面的 `x += step * v` 没法原样贴进一个按 σ 更新的循环里。对比质量的时候，不加缓存的 baseline 也要换成同样的采样器，否则比的就不只是缓存了。

### 其他值得知道的

还有几个没有展开的，简单记一下核心思路：

- **DeepCache**（CVPR 2024）：UNet 时代的方法。利用 UNet 的 skip connection，缓存深层分支的高层特征，跳过的步只更新浅层分支。DiT 上的缓存很多都是从这个思路延伸出来的。
- **PAB**（Pyramid Attention Broadcast）：发现 attention 输出的差异沿时间步呈 U 型，中间一大段很稳定，于是把 attention 的输出广播给后续几步。而且不同 attention 的稳定程度不同，cross-attention 最稳定、广播范围最大，temporal 次之，spatial 最小，所以叫"金字塔"。
- **ToCa / DuCa**：token 粒度的缓存，不同 token 对缓存的敏感程度差别很大，每步只重算一部分 token。ToCa 的部分打分依赖 attention map，而 FlashAttention 不会把 attention map 写出来；DuCa 发现所谓的重要 token 选得并不比随机好，干脆随机选。
- **FasterCache**：除了时间步之间的冗余，还利用了同一步里 CFG 两条分支之间的冗余（CFG-Cache），复用 cond 和 uncond 输出之间的关系。
- **MagCache**：观察到相邻步残差的模长比在大部分步里稳定、单调地下降，最后几步才快速下降，而且这个规律跨模型、跨 prompt 都成立，于是用模长比来估计误差、决定跳过，只需要 1 条样本标定。

## 工程上要注意的几件事

论文里比的是 PSNR、LPIPS 和加速比，落到推理系统里还有不少要考虑的东西。

**1. 同步和 CUDA Graph。** 动态调度常见的写法是用 `.item()` 把探针标量拿回 CPU 再决定分支，每步至少有一次设备到主机的同步。这个开销大不大要用 profiler 看：大分辨率的视频模型一步几百毫秒，通常不明显；小模型、小分辨率下 kernel launch 本来就占比高，再加上整步没法 capture，损失就明显了。常见的做法是把完整路径和缓存路径分别 capture 成图，判断放在图外。CUDA Graph 的 conditional node 可以在设备侧做条件执行，但要看框架支不支持，分支里有 collective 时也要单独验证。静态调度没有和数据相关的分支，不过缓存状态张量的地址、原地更新和形状一样要满足 capture 的要求。

**2. 分布式下的决策一致性。** 关键看缓存边界处张量的布局，以及被跳过的区域里有哪些 process group。普通 TP 下探针是完整复制的，各 rank 的决策天然一致；CP、TP+SP 下探针是分片的，要先合成全局统计量，或者由一个 rank 判定之后把结果 broadcast 出去。CFG 并行时，两个 plane 可以各自决定要不要重算，只要各自内部的 TP/CP ranks 一致就行。SoftCache 是个例外：两条分支的落点和步长必须同步，否则 guidance 组合的就不是同一个 latent、同一个 t 上的预测了。

**3. 显存。** 还是按 Wan2.2 算，一个 `[1,27280,3072]` 的 BF16 张量是 159.84 MiB。FBCache 的探针加剩余残差大约两份；TeaCache4FLUX 的全尺寸探针加残差也是这个量级，而 Wan 版本的探针只是 timestep embedding，几乎不占显存。整网残差上做 2 阶泰勒预测是三份同尺寸的张量，用 FP32 存还要翻倍；逐模块缓存则可能到几十 GiB。SeaCache 另外还有 FFT 的复数临时张量。CP 分片能降低每个 rank 的缓存显存，但 gather、FFT workspace、差分更新时新旧张量同时存在，这些都要算进峰值。

**4. 和其他加速手段叠加。** 4~8 步的蒸馏模型本来就没几个节点可省，步间变化又大，缓存的收益通常很有限。量化、稀疏 attention 和缓存的误差会相互影响，阈值要在组合之后的系统上重新调，论文里的值只能当起点。泰勒的高阶差分在 BF16/FP16 下尤其要注意相消和噪声放大。

**5. 怎么评估。** 一般看两类指标：和不加速的结果比的保真度（PSNR、SSIM、LPIPS），以及生成质量本身（ImageReward、CLIP Score、VBench）。这两者不完全一致，比如 DPCache 在 3.54x 时 ImageReward 比基线还高，但它和基线的结果肯定不是逐像素一样的。线上业务通常希望"加速前后用户看到的结果尽量一致"，所以我觉得保真度更重要一些。

评测时固定 prompt、seed、分辨率、采样器、CFG 和 dtype，预热之后同时记录 denoiser 时延和端到端时延、完整计算的次数、峰值显存，以及质量的分布而不只是均值。多看几个 seed 和一些异常的 prompt，视频还要看时间一致性。只报一个平均 PSNR 和"少算了多少步"，很难判断上线后到底有没有收益。

**6. 缓存的生命周期。** 缓存应该绑定到 `(request_id, cfg_branch, model/LoRA, shape)` 这样的身份上，每条新轨迹开始前先 reset，而不是在模型对象上挂一份全局的 `previous_residual`。动态 batching 时，batch 的第 0 行下一步可能已经换了请求，要么按请求 gather/scatter 缓存，要么整批共用一个保守的决策。中途换权重、换 LoRA、换条件或者切换模型分段时，旧缓存也要作废。很多"偶发的质量下降"，其实是状态串到了别的请求上。

## 几个容易被追问的问题

**为什么 TeaCache 这类方法缓存残差？** 因为跳过的步里，patch embedding 之后的 $$s_t$$ 是新算的，保留当前输入、只近似 transformer 的增量，就把输入的更新利用上了。当然这不是唯一的选择，DPCache 的公开 FLUX 实现预测的就是 stream 末层的输出特征。

**为什么常用相对 L1，而不是 L2 或者 cosine？** 除以参考特征的幅值，可以抵消不同时间步之间的尺度变化，让一个阈值在整条轨迹上都能用；L1 比平方 L2 对个别离群值没那么敏感，算起来也便宜。cosine 只看方向，对纯幅值的漂移不敏感，可能会漏掉这种变化。不过相对 L1 也要加分母保护，阈值也得按模型验证。

**为什么要累计？** 单看每一步的变化都不大，但连续跳过时误差是累积的。不累计的话，一条缓慢漂移的轨迹可以一直不触发重算。FBCache 和锚点比、TeaCache 把增量加起来，都是在处理这件事。

**静态调度和动态调度怎么选？** 静态调度没有逐步的探针判断，也就没有对应的同步，更容易组织 CUDA Graph 和多卡执行；动态调度能随样本自适应，但多了探针和决策的开销。哪个质量更好，要在同样的实际时延下去测。如果请求的配置比较固定，我会先试静态表，再看在线自适应带来的收益能不能覆盖它的额外开销。

**TaylorSeer 为什么阶数不能太高？** 有限差分的阶数越高，对噪声越敏感；而且间隔 N 大的时候泰勒余项本身就大，高阶项不一定是在修正误差，也可能是在放大误差。

**SoftCache 和前面的方法有什么区别？** 前几种方法决定的都是"这个节点要不要完整算 block"，scheduler 的节点还是一个一个地访问；SoftCache 还让缓存信号决定走多远，后面的网格跟着变，实际访问的节点也少了。

## 小结

写到这里，回头看这六篇，其实都是在同一个骨架上换零件。

FBCache 和 TeaCache 解决的是"拿什么来判断"：一个付一层 block 的代价换来更贴近网络的探针，一个用调制输入把判断做到几乎免费，代价是要按模型拟合多项式。TaylorSeer 换的是补值方式，从原样复用变成了外推，代价是显存。SeaCache 又回到判据上，不改探针本身，而是换了一个衡量距离的空间。DPCache 把在线的逐步判断搬到了离线，用整条标定轨迹去规划关键步。SoftCache 则第一次让判断的结果去影响采样器，连时间网格也改了。

所以它们并不是六套互相排斥的系统，很多是能组合的：FBCache 的判据可以配上 TaylorSeer 的预测器，ParaAttention 里就是这么做的；SEA 滤波可以套到别的判据上；SoftCache 本身就建立在 SeaCache 的探针之上。以后再看到新的缓存方法，我会先问三件事：它换掉了骨架里的哪个零件，误差是从哪里引入的，省下的计算又付出了什么代价。

至于选哪个，不同论文的模型、采样器、速度口径和质量指标都不完全一样，很难直接按加速比排高低。放到推理系统里，静态还是动态、要不要同步、多卡能不能保持一致、显存够不够，往往比论文里零点几 dB 的 PSNR 差距更能决定最后用哪一种。

## 参考

1. [ParaAttention（FBCache）](https://github.com/chengzeyi/ParaAttention)
2. [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model（TeaCache）](https://arxiv.org/abs/2411.19108)，[代码](https://github.com/ali-vilab/TeaCache)
3. [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/abs/2503.06923)，[代码](https://github.com/Shenyi-Z/TaylorSeer)
4. [SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](https://arxiv.org/abs/2602.18993)，[代码](https://github.com/jiwoogit/SeaCache)
5. [Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](https://arxiv.org/abs/2602.22654)，[代码](https://github.com/argsss/DPCache)
6. [SoftCache: Adaptive Step Size Caching for Diffusion Transformers](https://arxiv.org/abs/2610.22369)
7. [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858)
8. [Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588)
9. [Accelerating Diffusion Transformers with Token-wise Feature Caching（ToCa）](https://arxiv.org/abs/2410.05317)
10. [Rethinking Token-wise Feature Caching: Accelerating Diffusion Transformers with Dual Feature Caching（DuCa）](https://arxiv.org/abs/2412.18911)
11. [FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/abs/2410.19355)
12. [MagCache: Fast Video Generation with Magnitude-Aware Cache](https://arxiv.org/abs/2506.09045)
13. [cache-dit](https://github.com/vipshop/cache-dit)
