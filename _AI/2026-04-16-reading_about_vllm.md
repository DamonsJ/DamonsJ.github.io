---
layout: post
title: vLLM V1 源码阅读笔记
date: 2026-04-16 12:00:00
description: 阅读 vLLM V1 源码后的架构理解与核心流程记录
tags: vllm inference cuda
categories: AI
pretty_table: true
toc:
  sidebar: left
---

> 本文基于 vLLM V1 (v0.18.x) 源码阅读整理，记录了从请求进入到 token 输出的完整链路。
> 完整的架构图可下载 [vllm-architecture.drawio](/assets/img/AI/reading-vllm-v1/vllm-architecture.drawio) 在 draw.io 中查看。

## 整体架构概览

vLLM V1 采用**多进程架构**，核心由三类进程组成：

1. **主进程 (Main Process)** — API Server + 请求预处理 + 输出后处理
2. **EngineCore 子进程** — Scheduler 调度 + KV Cache 管理 + 执行器编排
3. **Worker 进程** — 每个 GPU 一个进程，负责模型推理（Tensor Parallel）

进程间通过 **ZMQ** (input/output socket) 进行异步通信，EngineCore 与 Worker 之间则使用基于 `multiprocessing.shared_memory` 的 **shm_broadcast** 环形缓冲 + RPC 机制。

## 一、主进程 (Main Process)

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/01-main-process.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### 1.1 启动入口

主进程运行 `vllm.entrypoints.openai.api_server`，由 Uvicorn + FastAPI 驱动。启动流程：

```
api_server
  └─ lifespan: build_async_engine_client (协程, contextmanager)
       └─ build_async_engine_client_from_engine_args
            └─ AsyncLLM.from_vllm_config(vllm_config)
                 └─ AsyncLLM.__init__()
```

### 1.2 AsyncLLM 初始化

`AsyncLLM.__init__` 创建以下核心组件：

| 组件                          | 职责                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Renderer** (`BaseRenderer`) | 聊天模板渲染、异步 tokenize（`AsyncMicrobatchTokenizer`）、异步多模态数据加载 → 生成 `ProcessorInputs` |
| **InputProcessor**            | 将 `ProcessorInputs` 封装为 `EngineCoreRequest`（校验参数、处理多模态 hash/placeholder 等）            |
| **OutputProcessor**           | detokenize `EngineCoreOutputs` → `RequestOutput`，检查 stop conditions                                 |
| **IOProcessor** _(可选插件)_  | 通过 `io_processor_plugin` 配置加载，用于 Pooling 等特殊场景的自定义 I/O 预处理/后处理，默认 `None`    |
| **EngineCoreClient**          | 与 EngineCore 子进程的 ZMQ 通信客户端                                                                  |

```python
# async_llm.py __init__ 核心初始化
self.renderer = renderer = renderer_from_config(self.vllm_config)
self.io_processor = get_io_processor(...)          # 可选插件，多数模型为 None
self.input_processor = InputProcessor(self.vllm_config, renderer)
self.output_processor = OutputProcessor(renderer.tokenizer, ...)
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

**Renderer** 是 V1 新引入的关键抽象层，位于 `vllm/renderers/`。不同 tokenizer 类型对应不同实现：

- `HfRenderer`：HuggingFace tokenizer（大多数模型）
- `MistralRenderer`：Mistral tokenizer
- `QwenVLRenderer` 等：特定模型变体

Renderer 内部持有 `AsyncMicrobatchTokenizer`，将阻塞的 tokenizer 调用卸载到 `ThreadPoolExecutor`，并支持微批量（micro-batch）合并以提升吞吐。

`EngineCoreClient` 根据 `data_parallel_size` 决定使用：

- 单 DP：`AsyncMPClient`
- 多 DP：`DPAsyncMPClient` / `DPLBAsyncMPClient`

### 1.3 请求处理流程（以 OpenAI Chat 请求为例）

一个 Chat Completion 请求的完整链路如下：

```
HTTP Request
  → OpenAIServingChat.create_chat_completion()
    → render_chat_request()
      → renderer.render_chat_async()          ← ① 异步渲染
        → parse_chat_messages_async()         ← 异步解析消息 + 多模态加载
          → AsyncMultiModalItemTracker        ← fetch_video_async() / fetch_image_async()
            → MediaConnector.load_from_url_async()
              → run_in_executor(global_thread_pool, load_bytes)  ← 线程池解码
        → tokenize_prompts_async()            ← ② 异步 tokenize
          → AsyncMicrobatchTokenizer.encode()
            → run_in_executor(ThreadPoolExecutor, tokenizer)     ← 线程池 tokenize
      → 返回 ProcessorInputs
    → engine_client.generate(engine_prompt, sampling_params)
      → AsyncLLM.add_request()
        → InputProcessor.process_inputs()     ← ③ 封装 EngineCoreRequest
        → engine_core.add_request_async()     ← ④ ZMQ 发送到 EngineCore
      → output_handler 推送结果到 queue       ← ⑤ 异步输出
      → async for chunk in generate(): yield  ← ⑥ SSE 流式返回
```

**① Renderer.render_chat_async() — 异步渲染**

- `parse_chat_messages_async()` 解析 OpenAI messages 格式，提取文本和多模态 URL
- 多模态数据（图像/视频/音频）通过 `fetch_*_async()` → `run_in_executor(global_thread_pool, ...)` 在**媒体线程池**中异步加载和解码
- 多个媒体文件通过 `asyncio.gather()` **并发**加载
- 渲染聊天模板（`apply_chat_template`）

**② tokenize_prompts_async() — 异步 Tokenize**

- 使用 `AsyncMicrobatchTokenizer`，将阻塞的 tokenizer 调用通过 `run_in_executor(ThreadPoolExecutor(max_workers=1), ...)` 卸载到**单独的 worker 线程**
- 支持微批量（micro-batch）：在 `batch_wait_timeout_s`（默认 2ms）内收集多个请求合并 tokenize，减少调用开销

**③ InputProcessor.process_inputs() — 封装请求**

- 接收已处理好的 `ProcessorInputs`（已包含 `prompt_token_ids`、`mm_kwargs`、`mm_placeholders` 等）
- 校验参数、计算 `max_tokens`、构造 `EngineCoreRequest`

**④ engine_core.add_request_async() — 发送到 EngineCore**

- 通过 ZMQ input_socket 将 `EngineCoreRequest` (msgpack 序列化) 发送到 EngineCore 子进程

**⑤ output_handler (后台 asyncio Task)**

- 持续从 ZMQ output socket 拉取 `EngineCoreOutputs`
- 调用 `OutputProcessor.process_outputs()` 进行 detokenize
- 将输出按 chunk 分批处理（`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`），每批之间 `await asyncio.sleep(0)` 让出事件循环，避免长时间阻塞
- 生成 `RequestOutput` 推送到对应请求的 `RequestOutputCollector` 队列

> **关于 GIL**：主进程中 tokenizer 和视频解码等 CPU 密集操作不会阻塞 event loop 接收新请求，这依靠**两重机制**协同工作：
>
> 1. **`run_in_executor` + 线程池**：所有阻塞操作都被卸载到工作线程，event loop 线程通过 `await` 等待结果，期间可以处理其他协程
> 2. **C/Rust 扩展自动释放 GIL**：HuggingFace tokenizers（Rust 实现）、OpenCV（C++ 实现）、NumPy（C 实现）在执行计算时会**主动释放 GIL**，工作线程不会阻塞 event loop 线程获取 GIL
>
> 即使存在少量纯 Python 代码持有 GIL，CPython 3.2+ 的 GIL 调度机制也会每 **5ms**（`sys.getswitchinterval()`）强制切换，确保 event loop 线程有机会运行。

## 二、EngineCore 子进程

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/02-engine-core.png" class="img-fluid rounded z-depth-1" zoomable=true %}

每个 DP rank 对应一个 `EngineCoreProc` 进程，这是 vLLM 的调度核心。

### 2.1 初始化

```
EngineCoreProc.__init__()
  ├─ 连接 ZMQ socket (input/output address)，与主进程握手建立通信
  └─ EngineCore.__init__() — 创建核心组件
       ├─ MultiprocExecutor (model_executor)
       ├─ Scheduler
       └─ StructuredOutputManager
```

### 2.2 IO 线程与主循环

EngineCore 进程内部有 **3 个线程**协同工作，实现 ZMQ 通信与 GPU 计算的重叠：

```
┌─ EngineCoreProc 进程 ───────────────────────────────────────────────┐
│                                                                     │
│  Input IO Thread ──► input_queue ──► Main Thread ──► output_queue ──► Output IO Thread
│  (ZMQ recv,                         (busy loop)                      (ZMQ send,
│   msgpack 反序列化,                                                    msgpack 序列化,
│   释放 GIL)                                                           释放 GIL)
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

源码注释写道：_"Background Threads and Queues for IO. These enable us to overlap ZMQ socket IO with GPU since they release the GIL, and to overlap some serialization/deserialization with the model forward pass."_

**Input IO Thread** (`process_input_sockets`):

- 从 ZMQ input socket 接收消息并 msgpack 反序列化
- 将 `(request_type, data)` 放入 `input_queue`（Python `queue.Queue`）

**Output IO Thread** (`process_output_sockets`):

- 从 `output_queue` 取出 `EngineCoreOutputs`
- msgpack 序列化后通过 ZMQ output socket 发回主进程

**Main Thread** 运行 `run_busy_loop()`：

```python
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()    # ① 取请求
        self._process_engine_step()    # ② 调度 + 执行 + 更新 + 输出
```

**① `_process_input_queue()`** — 从 `input_queue` 取出新请求

- 无工作时阻塞等待（`queue.get(block=True)`）
- 有工作时非阻塞排空队列（`queue.get(block=False)`）
- `_handle_client_request()` 根据请求类型分派：`ADD` → `preprocess_request()` → 加入 `Scheduler.waiting`

**② `_process_engine_step()`** — 单次引擎步骤

- `scheduler.schedule()` → `SchedulerOutput`（本次 step 跑哪些请求、token budget、block 分配）
- `model_executor.execute_model(scheduler_output)` → `ModelRunnerOutput`（含 sampled tokens）
- `scheduler.update_from_output()` — 更新请求状态，释放已完成请求的 KV blocks
- 构造 `EngineCoreOutputs` 放入 `output_queue` → Output IO Thread 异步发送

### 2.3 Scheduler 与 KV Cache 管理

`schedule()` 每个 step 调用：

1. **优先调度 running 队列**（decode 请求）— 计算 token budget，`allocate_slots`
2. **调度 waiting 队列**（prefill 请求）— `get_computed_blocks`（prefix cache lookup）→ `allocate_slots` → 移入 running

**KVCacheManager**（PagedAttention 核心）：

- `free_block_queue`（双向链表）管理空闲 block
- `req_to_blocks` dict 记录每个请求占用的 block
- `allocate_slots` / `free` / `cache_blocks` 操作
- **Prefix Caching**: `hash_request_tokens` + `find_longest_cache_hit`

## 三、Worker 进程

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/03-gpu-worker.png" class="img-fluid rounded z-depth-1" zoomable=true %}

每个 GPU 一个 Worker 进程，通过 Tensor Parallel 协同工作。

### 3.1 启动与初始化

```
MultiprocExecutor (EngineCore 进程中)
  └─ spawn WorkerProc × local_world_size (= TP × PP × PCP)
       └─ Worker init (每个 GPU)
            ├─ init_device() — 设置 CUDA device, 分布式通信组 (NCCL)
            ├─ model_runner.load_model() — 实例化模型架构, 加载权重, torch.compile()
            └─ initialize_cache() — profiling forward pass → 分配 KV cache tensors → warmup CUDA graphs
```

Worker 进程进入 `worker_busy_loop`，通过 `rpc_broadcast_mq.dequeue` 等待来自 MultiprocExecutor 的消息（func + 参数），执行后 `handle_output`。

> **shm_broadcast**: 基于 `multiprocessing.shared_memory` 的多槽环形缓冲 + 内存栅栏/每读端完成位 + SpinCondition 唤醒。pickle 后小对象整块进环，大对象打 overflow 标并用本机 ZMQ IPC 传 multipart；多机时再配合 TCP XPUB/SUB。

### 3.2 GPUModelRunner.execute_model()

每次 step 的执行流程：

**① \_update_states()**

- prune 已结束请求，更新 InputBatch metadata
- 更新 block table（paged KV 索引）

**② \_prepare_inputs()**

- CPU → GPU: `input_ids`, `positions`, `slot_mapping`
- 构造 attention metadata（FlashAttention / FlashInfer）

**③ model.forward()**

- eager 模式或 CUDA Graph replay
- 所有序列拼接为超长序列（Continuous Batching）
- PagedAttention kernel: 读写 KV cache blocks

**④ gather last-token hidden states → compute logits**

**⑤ Sampler: 从 logits 采样 token**

- 支持 greedy / temperature / top-p / top-k / guided decoding

返回 `SamplerOutput` → `EngineCoreOutputs`

## 四、多模态输入预处理

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/04-multimodal.png" class="img-fluid rounded z-depth-1" zoomable=true %}

V1 将多模态预处理**全部异步化**，通过线程池避免阻塞 event loop 和 GPU Worker。

### 4.1 请求解析（异步路径）

HTTP 请求进入 `OpenAIServingChat` 后，走**异步**解析链路：

**① `parse_chat_messages_async()`** — 解析 OpenAI messages 格式

遍历消息内容，遇到多模态 URL 时不立即加载，而是向 `AsyncMultiModalItemTracker` 注册一个**协程**（coroutine）：

```python
# chat_utils.py — AsyncMultiModalContentParser
async def _video_with_uuid_async(self, video_url, uuid):
    video = await self._connector.fetch_video_async(video_url) if video_url else None
    return video, uuid

def parse_video(self, video_url, uuid=None):
    coro = self._video_with_uuid_async(video_url, uuid)  # 注册协程，不立即执行
    placeholder = self._tracker.add("video", coro)
```

**② `AsyncMultiModalItemTracker.resolve_items()`** — 并发加载所有多模态数据

```python
resolved_items_by_modality = {
    modality: await asyncio.gather(*coros)  # 所有 image/video/audio 并发加载
    for modality, coros in self._items_by_modality.items()
}
```

**③ 媒体加载线程池** — 真正的 IO 和解码在线程池中执行

```python
# connector.py
global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT  # 可配置
)

async def load_from_url_async(self, url, media_io, ...):
    # HTTP 下载使用 aiohttp（纯异步，不占线程）
    data = await connection.async_get_bytes(url)
    # 解码在线程池中执行（OpenCV/NumPy 释放 GIL）
    return await loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
```

| 操作                          | 执行方式                                 | GIL 行为          |
| ----------------------------- | ---------------------------------------- | ----------------- |
| HTTP 下载视频/图片            | `aiohttp` 异步 I/O                       | 不持有 GIL        |
| 视频解码 (`cv2.VideoCapture`) | `run_in_executor(global_thread_pool)`    | C++ 扩展释放 GIL  |
| 图片解码 (`PIL.Image.open`)   | `run_in_executor(global_thread_pool)`    | C 扩展释放 GIL    |
| NumPy 数组操作 (resize 等)    | 同上线程池                               | C 扩展释放 GIL    |
| Tokenize                      | `run_in_executor(ThreadPoolExecutor(1))` | Rust FFI 释放 GIL |

**④ Renderer.render_chat_async()** — tokenize + 组装

tokenize 通过 `AsyncMicrobatchTokenizer` 在独立线程中完成后，与多模态数据一起组装为 `ProcessorInputs`。

### 4.2 多模态缓存

**mm_processor_cache**（主进程缓存）:

- key = `hash(mm_data)` ← 同一图/视频跨请求复用
- 避免重复 CPU 解码 / resize

### 4.3 构造 EngineCoreRequest

```
prompt_token_ids: list[int]          ← 已含 mm placeholder token
mm_features: list[MultiModalFeature] ← 每个 mm 对象的 pixel_values / audio features
mm_hashes: list[str]                 ← 用于 encoder_cache 查找
mm_placeholders: list[PlaceholderRange] ← 在 token 序列中的位置信息
```

经 **msgpack 序列化** → ZMQ input_socket → EngineCore 进程。

### 4.4 Encoder Cache (GPU 侧)

在 EngineCore 进程中，`preprocess_request()` 检查 `mm_hash` 是否在 `encoder_cache` 中：

- **命中** → 直接取已有 GPU embedding，跳过视觉编码
- **未命中** → 将 `pixel_values` 等打入 `EngineCoreRequest`，在 Worker 中由 Vision Encoder 处理

### 4.5 model.forward() 执行阶段

```
Vision Encoder:  pixel_values → vision embeddings (e.g. 4096 dims/image)
Projection Layer: vision_embeddings → LLM hidden_size
Merge: text token hidden states + vision hidden states → 统一 hidden sequence
```

合并后进入标准 Transformer decoder 流程。

## 五、Continuous Batching + Chunked Prefill

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/05-continuous-batch.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### 5.1 核心设计

每个 Engine Step 都有一个固定 **Token Budget** (`max_num_batched_tokens`, 默认 2048+)。Scheduler 在 budget 内打一个混合 batch：先塞满所有 decode 请求（每个占 1 token），剩余 budget 分给 prefill（可 chunk）。

| Step   | 内容                                                                                   |
| ------ | -------------------------------------------------------------------------------------- |
| Step 1 | R1 Prefill chunk①(512 tokens) + R2 Decode(1 token) + 空余 budget                       |
| Step 2 | R1 Prefill chunk②(512 tokens) + R2 Decode(1 token) + R3 Prefill chunk①(300 tokens)     |
| Step 3 | R1 Decode(1 token, prefill done✓) + R2 Decode(1 token) + R3 Prefill chunk②(200 tokens) |
| Step 4 | R1 Decode(1 tok) + R4 新请求 Prefill(100 tokens)                                       |

### 5.2 序列 Flatten

Forward Pass 内部，所有序列被 flatten 拼接为一条超长序列（无 padding）。以 Step 2 为例：

```
input_ids: [R1_tok[512..1023], R2_tok[n], R3_tok[0..299]]
```

attention mask 保证每个序列只 attend 自身。`slot_mapping` 记录每个 token 对应的 KV cache 物理 block slot（PagedAttention 寻址）。

### 5.3 KV Cache 管理

- Step 1: R1 分配 2 blocks, R2 无新分配
- Step 2: R1 +2 blocks, R2 无新, R3 +2 blocks
- Step 3: R1 decode +1tok, R2 +1tok, R3 +2 blocks, R4 +1 block

### 5.4 抢占 (Preemption)

KV Cache 不足时，V1 默认 **RECOMPUTE** 模式（V0 支持 SWAP）：将低优先级 running 请求踢回 waiting 队列，释放其 KV blocks。下次调度时重新从头 prefill（开销低于 swap to CPU）。

### 5.5 Continuous Batching 核心特性

- 序列在 forward pass 中被 flatten 拼接（无 padding）
- 每个 step 结束后，已完成的请求立即被移出，新请求可在下个 step 加入
- V1 取消了 prefill/decode 的人为分界，统一用 `{req_id: num_tokens}` 表示调度决策
- KV Cache 用 PagedAttention 管理，物理块按需分配/回收

## 六、GPUModelRunner V1 vs Model Runner V2 (MRV2)

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/06-v1-vs-mrv2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

MRV2 (`VLLM_USE_V2_MODEL_RUNNER=1` 开启) 目前仍为 experimental；V1 为当前默认。两者共用同一套 Scheduler / Executor / KV Cache 架构，仅 ModelRunner 层不同。

### 6.1 整体设计哲学

| V1 (GPUModelRunner)                             | MRV2 (ModelRunner V2)                                 |
| ----------------------------------------------- | ----------------------------------------------------- |
| 单文件 monolith: `gpu_model_runner.py` ~6283 行 | 拆分为 ~40 个子模块，最大文件 <1300 行                |
| 所有逻辑（通用 + 模型专属）耦合在一起           | 三大原则：Be Modular / Be GPU-native / Be Async-first |
| 特性逐步 bolt-on，随时间积累技术债              | ModelState 抽象：模型专属逻辑与公共路径隔离           |
| 难以扩展新模型，新贡献者学习曲线陡峭            | 从头重写，不继承 V1 复杂性                            |

### 6.2 Persistent Batch 设计

**V1 的问题**：

persistent state tensor 直接用作 model input tensor（都在 CPU），block table 布局与请求顺序强耦合。这带来几个严重问题：

- 添加/删除请求 → 需要整张 tensor 重排（complex reorder）
- 需要 `CachedRequestState` 作冗余备份
- async scheduling 下容易引发 race condition
- 所有 CPU 操作必须在 async barrier 内，灵活性低

**MRV2 的解法**：

将持久状态与输入解耦为两部分：

- **stable state table** (CPU, fixed 1024 rows) — 每个 req 在生命周期内固定一行
- **per-step input tensor** (GPU) — Triton gather kernel 按当前顺序组装

```python
# MRV2: 持久 state 与 input 解耦
self.states[req_idx] = new_req.data       # 写持久 state (非 pinned)
tmp_states = self.states.pin_memory()      # 每步拷贝到临时 pinned
states = tmp_states.to('cuda', non_blocking=True)  # H2D
input_tensors = triton_gather(states, schedule_indices)  # GPU gather
```

解耦好处：

- 添加/删除请求只改 state table 一行，O(1)
- 消除 `CachedRequestState` 冗余备份
- `StagedWriteTensor`: CPU 写 state, GPU 读 tmp copy → 无 race
- `input_ids` / `positions` / `seq_lens` / `block_table` 全在 GPU 组装
- 不再需要 async barrier

### 6.3 Input Preparation

| V1                               | MRV2                            |
| -------------------------------- | ------------------------------- |
| CPU (Python/NumPy) → copy to GPU | GPU-native Triton gather kernel |

### 6.4 Async Scheduling

| V1                       | MRV2                            |
| ------------------------ | ------------------------------- |
| Retrofit + async barrier | Design-first，zero CPU-GPU sync |

### 6.5 Sampler

| V1                            | MRV2                        |
| ----------------------------- | --------------------------- |
| PyTorch softmax + multinomial | Triton Gumbel-Max，内存更省 |

### 6.6 ModelState 抽象 (MRV2 核心创新)

```python
class ModelState(ABC):
    def add_request(self, req: EngineCoreRequest): ...
    def remove_request(self, req_id: str): ...
    def get_mm_embeddings(self, req_id: str) -> Tensor: ...
    def prepare_inputs(self, scheduler_output) -> dict: ...
```

模型扩展方式对比：

- V1: override 方法，无统一抽象
- MRV2: ModelState ABC，职责清晰

Spec Decoding + Async 支持：

- V1: 难以同时支持，需要 hack
- MRV2: 天然支持，GPU prep 直接消费 rejection output

### 6.7 性能对比 (官方 benchmark)

**Qwen3-0.6B · 1×GB200**（小模型大 GPU → host overhead 占比大）

| 指标       | V1         | MRV2                  |
| ---------- | ---------- | --------------------- |
| 输出吞吐量 | ~16K tok/s | ~25K tok/s (+56.2% ↑) |

**GLM-4.7-FP8 · 4×GB200 · MTP spec decoding**

MRV2 实现 -6.3% TPOT（zero-sync spec decoding）。

### 6.8 当前功能覆盖 (v0.18.0)

| V1 (完整生产可用)              | MRV2 (Experimental)                   |
| ------------------------------ | ------------------------------------- |
| ✅ 所有 decoder-only 模型      | ✅ 标准 decoder-only 模型推理         |
| ✅ MoE (Mixtral / DeepSeek)    | ✅ Eagle / Eagle3 / MTP spec decoding |
| ✅ 多模态 VLM (图像/视频/音频) | ✅ Async scheduling (zero sync)       |
| ✅ Full spec decoding          | ✅ 基础多模态                         |
| ✅ LoRA / EPLB / DBO           | ❌ Linear attention (Qwen3.5)         |
| ✅ Structured output           | ❌ MTP 以外的 spec decoding           |

### 6.9 核心差异一览表

| 维度                  | V1 (GPUModelRunner)              | MRV2 (ModelRunner V2)             |
| --------------------- | -------------------------------- | --------------------------------- |
| 代码规模              | 单文件 ~6283 行                  | ~40 模块，最大 <1300 行           |
| Persistent Batch      | state 直接 = input，耦合强       | stable state table + gather，解耦 |
| Input Preparation     | CPU (Python/NumPy) → copy to GPU | GPU-native Triton gather kernel   |
| Async Scheduling      | Retrofit + async barrier         | Design-first，zero CPU-GPU sync   |
| Sampler               | PyTorch softmax + multinomial    | Triton Gumbel-Max，内存更省       |
| 模型扩展              | override 方法，无统一抽象        | ModelState ABC，职责清晰          |
| Spec Decoding + Async | 难以同时支持，需 hack            | 天然支持                          |
| 状态                  | ✅ 默认，生产就绪                | ⚠ Experimental，部分功能缺失     |

## 七、torch.compile Backend：默认 Inductor vs VllmBackend

{% include figure.liquid loading="eager" path="assets/img/AI/reading-vllm-v1/07-torch-compile.png" class="img-fluid rounded z-depth-1" zoomable=true %}

vLLM 对 `torch.compile` 进行了深度定制，实现了 `VllmBackend` 以替代默认的 Inductor backend。

### 7.1 触发入口

| 默认 Inductor                     | VllmBackend（vLLM 定制）                                        |
| --------------------------------- | --------------------------------------------------------------- |
| `torch.compile(model)`            | `torch.compile(model, backend=VllmBackend(...))`                |
| Dynamo 追踪字节码，生成完整 FX 图 | Dynamo 追踪后调用 `VllmBackend.__call__(graph, example_inputs)` |
| 直接将整图交给 Inductor 处理      | vLLM 接管整个编译流程                                           |
| 用户无需关心内部切分逻辑          | 支持 prefix / model_tag 区分多模型部件                          |

### 7.2 图切分策略 (Graph Splitting)

| 默认 Inductor                                | VllmBackend                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| 无主动切分，整图交给 Inductor 自动 op fusion | 主动 `split_graph()` 切分                                |
| 切分完全由 Inductor 自动决定                 | 按 `splitting_ops`（如 all_reduce、flash_attention）切割 |
| 不感知 all_reduce / attention 等通信算子边界 | 相邻 splitting op 合并为同一子图，避免碎片化             |
| 无法针对 tensor parallel 做精细控制          | 纯空分配节点（empty）合并到前一分区，防止空 cudagraph    |

### 7.3 CUDA Graph 捕获方式

| 默认 Inductor                                              | VllmBackend                                                           |
| ---------------------------------------------------------- | --------------------------------------------------------------------- |
| 整图静态捕获（FULL 模式）                                  | 分段 **PIECEWISE** 捕获                                               |
| `torch.compile(mode="reduce-overhead")` 触发整图 cudagraph | 每个非 splitting 子图单独包进 `CUDAGraphWrapper`                      |
| 要求整图输入形状完全静态                                   | splitting 子图（all_reduce 等）不被捕获，允许主机同步                 |
| 动态 batch size 场景需要 padding 或 bucketing              | `sym_tensor_indices` 追踪动态维度，用 `copy_and_call` 注入静态 buffer |

### 7.4 编译缓存机制

| 默认 Inductor                             | VllmBackend                                           |
| ----------------------------------------- | ----------------------------------------------------- |
| `torch._inductor.codecache` 内置          | `compilation_config.cache_dir` 或 `VLLM_CACHE_ROOT`   |
| 以 FX 图 hash 作 key                      | 以 FX 图 hash + model_tag 作 key                      |
| 仅缓存 Inductor 产物（Triton code / .so） | 缓存 split 后每段 Inductor 产物 + CUDA Graph metadata |

### 7.5 Post-grad 自定义 Pass

| 默认 Inductor                  | VllmBackend                                 |
| ------------------------------ | ------------------------------------------- |
| 只跑 Inductor 内置 pass        | 注入 `fix_functionalization` 等自定义 pass  |
| 无法干预 attention kernel 选择 | 可替换 FlashAttention 后端 / 注入 custom op |

### 7.6 动态形状 (Dynamic Shape) 处理

| 默认 Inductor                       | VllmBackend                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| SymInt 符号追踪，需要 guard 验证    | `PiecewiseCompileIntermediate` 记录每段输入张量索引                   |
| 动态 batch → recompile 或 bucketing | `sym_tensor_indices` 标记动态维度 → cudagraph 捕获时只替换动态 tensor |

### 7.7 多模型组件支持

| 默认 Inductor                     | VllmBackend                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| 单一 compile 调用，不区分模型部件 | `prefix_str` / `model_tag` 区分 draft model、vision encoder、LLM backbone |
| 所有子图共享同一编译配置          | 每个部件可独立控制是否 cudagraph、是否 compile                            |

### 7.8 返回值与序列化

| 默认 Inductor                          | VllmBackend                                         |
| -------------------------------------- | --------------------------------------------------- |
| 返回编译后的 callable（Triton code）   | 返回 `VllmSerializableFunction`                     |
| `torch.save` / `torch.load` 标准序列化 | 支持 `pickle` 序列化已编译函数 → 跨进程共享编译产物 |

## 八、进程拓扑与数据并行 (DP)

### 8.1 进程层次模型

vLLM V1 的进程架构是**三层嵌套**的：

```
API 进程 (AsyncLLM)
  │
  ├── EngineCore 进程 #0  (DP rank 0)
  │     └── MultiprocExecutor
  │           ├── Worker 进程 #0  (GPU 0) ← TP rank 0
  │           ├── Worker 进程 #1  (GPU 1) ← TP rank 1
  │           ├── Worker 进程 #2  (GPU 2) ← TP rank 2
  │           └── Worker 进程 #3  (GPU 3) ← TP rank 3
  │
  ├── EngineCore 进程 #1  (DP rank 1)
  │     └── MultiprocExecutor
  │           ├── Worker 进程 #4  (GPU 4) ← TP rank 0
  │           └── ...
  │
  └── (可选) DPCoordinator 进程
        └── 负载均衡统计、MoE wave 协调
```

关键的配置公式：

| 参数                 | 含义                                                |
| -------------------- | --------------------------------------------------- |
| `world_size`         | = `TP × PP × PCP`，**每个 DP 副本**内的 Worker 数量 |
| `data_parallel_size` | DP 副本数，即 EngineCore 进程数                     |
| 总 GPU 数            | = `world_size × data_parallel_size`                 |
| 总 Worker 进程数     | = 总 GPU 数（一 GPU 一 Worker）                     |

```python
# config/parallel.py
self.world_size = (
    self.pipeline_parallel_size
    * self.tensor_parallel_size
    * self.prefill_context_parallel_size
)
```

### 8.2 典型部署拓扑举例

**场景 A：单卡部署（最简）**

`TP=1, PP=1, DP=1` → 1 个 EngineCore + 1 个 Worker

```
API 进程 ──ZMQ──► EngineCore ──shm──► Worker (GPU 0)
```

**场景 B：4 卡张量并行**

`TP=4, PP=1, DP=1` → 1 个 EngineCore + 4 个 Worker

```
API 进程 ──ZMQ──► EngineCore ──shm──► Worker 0 (GPU 0)
                                  ├──► Worker 1 (GPU 1)
                                  ├──► Worker 2 (GPU 2)
                                  └──► Worker 3 (GPU 3)
                                       ↕ NCCL AllReduce
```

**场景 C：8 卡数据并行**

`TP=1, PP=1, DP=8` → 8 个 EngineCore，各带 1 个 Worker

```
API 进程 ──ZMQ──► EngineCore 0 ──► Worker (GPU 0)
           ├───► EngineCore 1 ──► Worker (GPU 1)
           ├───► ...
           └───► EngineCore 7 ──► Worker (GPU 7)
                 ↕
           DPCoordinator (负载均衡)
```

**场景 D：DP + TP 混合（8 卡，DP=2, TP=4）**

```
API 进程 ──ZMQ──► EngineCore 0 (DP rank 0) ──► 4 Workers (GPU 0-3, NCCL)
           └───► EngineCore 1 (DP rank 1) ──► 4 Workers (GPU 4-7, NCCL)
```

### 8.3 DP 下 EngineCore 与 Worker 的关系

每个 EngineCore 进程**独立管理自己副本内的 Workers**，不同 DP rank 之间的 Worker 完全隔离：

```python
# multiproc_executor.py — 每个 EngineCore 内部
for local_rank in range(self.local_world_size):  # local_world_size = TP × PP
    WorkerProc.make_worker_process(
        local_rank=local_rank,
        rank=global_rank,
        ...
    )
```

跨 DP rank 的协调由 **DPCoordinator** 进程完成（仅在 `data_parallel_size > 1` 时启动）：

- 收集各 EngineCore 的**队列长度统计**，发布给前端做负载均衡
- MoE 模型的 **wave coordination**（确保所有 DP rank 同步执行 all-to-all）

### 8.4 Worker 启动流程

```
MultiprocExecutor._init_executor()
  ├─ 创建 MessageQueue (shm_broadcast 环形缓冲)
  ├─ for local_rank in range(local_world_size):
  │     WorkerProc.make_worker_process()
  │       └─ multiprocessing.Process(target=WorkerProc.__init__)
  │            ├─ WorkerWrapperBase → 实例化 Worker (gpu_worker.py)
  │            ├─ Worker.init_device()  → CUDA device 设置, NCCL 通信组
  │            └─ Worker.load_model()   → 加载权重, torch.compile
  └─ WorkerProc.wait_for_ready()  → 等待所有 Worker 就绪
```

Worker 就绪后进入 `worker_busy_loop`，通过 `rpc_broadcast_mq.dequeue` 等待 EngineCore 派发 RPC 调用（如 `execute_model`）。

### 8.5 通信机制总结

| 通信对                   | 机制                                 | 特点                      |
| ------------------------ | ------------------------------------ | ------------------------- |
| API 进程 ↔ EngineCore   | **ZMQ** (IPC/TCP) + msgpack          | 跨进程，支持多节点        |
| EngineCore ↔ Worker     | **shm_broadcast** (共享内存环形缓冲) | 同机低延迟，pickle 序列化 |
| Worker ↔ Worker (TP/PP) | **NCCL**                             | GPU 直连，高带宽          |
| DP rank 协调             | **DPCoordinator** (ZMQ)              | 统计发布 + wave 同步      |
