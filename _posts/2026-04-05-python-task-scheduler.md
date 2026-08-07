---
layout: post
title: python中多进程实现任务调度
date: 2026-04-05 18:07:00
description: 记录python中实现多进程任务调度的工程实现,其中的进程调度架构类似vllm中的实现。
tags: [Python, 多进程, 任务调度, 系统设计]
categories: [分布式与并发]
giscus_comments: false
related_posts: false
pretty_table: true

toc:
  beginning: true
---

## 背景

工作中遇到这样的场景，同一套服务同时承担在线请求与离线任务。在线服务通常面向实时用户请求，对时延和可用性要求较高；而离线服务多为批处理或异步任务，对实时性要求相对较低。当前业务中通常将同样的服务单独部署在在线集群和离线集群中，这需要同时维护两个同样的服务，并且当前系统缺乏对在线与离线服务的明确区分和统一调度机制，导致在资源紧张或负载较高的情况下，离线任务可能占用关键资源，从而影响在线服务的响应能力和稳定性。有必要设计并实现一套服务框架，对同一服务的在线与离线执行进行统一管理和调度，并支持在线服务对离线服务的抢占能力，以满足业务对实时性和稳定性的要求。这样同样的服务可以支持在线和离线任务，在线的任务优先级最高，任务会被优先调度运行，离线任务优先级低，只有在没有在线任务的时候才被调度。

## 设计目标

- 在线 / 离线服务统一建模
  对同一业务服务，支持以"在线模式"和"离线模式"两种运行形态对外提供能力。
  在调度和执行层面明确区分两类服务，而非通过约定或人工控制。

- 在线服务抢占式执行
  在线服务请求具备最高执行优先级。
  当系统资源不足且存在正在运行的离线服务时，在线服务可触发抢占机制，中断或终止离线服务，确保在线请求及时执行。

- 离线服务可控中断
  当前版本离线服务中断模式且返回明确错误码。
  当前版本离线任务的中断不支持回滚及重试，返回错误码后由用户决定下一步动作。

## 整体架构

{% include figure.liquid loading="eager" path="assets/img/posts/scheduler/architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}

整个系统由四层组成，从外到内分别是：**HTTP 接入层**、**ZMQ 通信层**、**调度层**和**执行层**。

| 进程index | 层级        | 组件                                           | 职责                                            |
| :-------- | :---------- | :--------------------------------------------- | :---------------------------------------------- |
| 进程1     | HTTP 接入层 | Gunicorn + Flask                               | 接收 HTTP 请求，解析参数，调用 Processor        |
| 进程1     | 通信层      | SchedulerClient (DEALER) ↔ Scheduler (ROUTER) | 进程间 IPC 通信，请求/结果传递                  |
| 进程2     | 调度层      | Scheduler + ProcessManager                     | 任务入队、优先级调度、抢占、Worker 生命周期管理 |
| 进程3     | 执行层      | WorkerProcess (active / standby)               | 加载模型、执行推理、返回结果                    |

### 实现说明

这套系统的核心挑战是实现抢占。
所谓抢占，是指当在线任务到来时，如果所有 Worker 都已被离线任务占用，调度器需要主动中断某个正在运行的离线任务，腾出 Worker 来立即执行在线任务。由于离线任务本身对时延不敏感，这种中断是可以接受的，被中断的客户端会收到 preempted 状态码，由业务层决定是否重试。

- 抢占后的冷启动问题
  抢占的执行很直接——kill 掉进程即可。真正的难点在于：kill 掉一个 Worker 之后，需要立刻有一个已经加载好模型的进程顶上来，而不是从零开始重新初始化。GPU 模型的加载通常需要数秒到数十秒，如果每次抢占都要等模型重新加载，在线任务的时延优势就荡然无存了。

- 用 Standby 进程解决冷启动
  为此，系统引入了 Standby Worker 的概念：在正常运行期间，后台始终维护一批预热好的备用进程。这些进程以 device=cpu 启动，提前完成 predictor 的初始化并将模型权重加载到内存，但不占用 GPU 资源，也不参与任务调度。
  当抢占发生时，Standby Worker 被直接顶替到被杀 Worker 的位置，并在接到第一个任务时将模型从 CPU 迁移到 GPU（load_model("cuda")），整个替换过程的额外耗时仅是 CPU→GPU 的数据搬运，远比从零加载模型要快。

- Standby 的补建
  Standby 进程是一次性消耗品——每次抢占都会消耗一个 Standby。为了维持随时可抢占的能力，调度器在完成一次抢占后会立即通知 ProcessManager 异步补建一个新的 Standby Worker。ProcessManager 在独立线程中完成新进程的启动和模型预热，完成后通过结果队列通知调度器，整个过程对调度主循环完全无阻塞。
  这样，整个抢占链路形成了一个自我修复的闭环：抢占 → 消耗 Standby → 异步补建 → 恢复 Standby 储备，系统始终保持随时可响应在线任务的能力。

### 关键设计决策

**为什么用 ZMQ 而不是 multiprocessing.Queue？**

`multiprocessing.Queue` 只能在父子进程间使用，且模式单一。ZMQ 的 ROUTER/DEALER 模式天然支持：

- 多客户端（Gunicorn 多线程）到单调度器的多对一通信
- 调度器到多 Worker 的一对多派发
- 基于 identity 的精确路由回包

**为什么 Scheduler 运行在独立进程？**

调度器持有所有 Worker 状态，是全局唯一的。把它放在独立进程里可以避免 Gunicorn fork 导致的状态不一致，也让调度逻辑与 HTTP 请求处理完全解耦。Gunicorn 可以有多个 Worker 进程，每个进程里的多个线程共享同一个 `SchedulerClient`，通过 ZMQ IPC 与 Scheduler 通信。

**为什么用 spawn 而不是 fork？**

CUDA 运行时不支持在 `fork` 出的子进程中重新初始化。由于 Worker 进程需要加载 GPU 模型，必须使用 `spawn` 方式创建子进程，确保每个 Worker 拥有独立的 CUDA 上下文。

## 工程实现

### 1. Scheduler 调度器

Scheduler 是整个系统的核心，运行在独立进程中，负责：

- 接收来自 Gunicorn 各线程的任务请求
- 按优先级将任务放入 `high_queue` 或 `low_queue`
- 驱动调度循环，分配任务给空闲 Worker
- 在无空闲 Worker 时触发抢占
- 接收 Worker 返回的结果，路由回对应的客户端

#### 启动流程

Scheduler 通过 `_scheduler_entry()` 函数作为 `spawn` 进程的入口。在子进程内部，先兜底设置 `set_start_method("spawn")`，确保 Scheduler 内部再创建的 WorkerProcess 也使用 spawn：

```python
def _scheduler_entry(worker_num, predictor_config):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    Scheduler(worker_num=worker_num, predictor_config=predictor_config).run()

def start_scheduler(worker_num=2, predictor_config=None):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_scheduler_entry, args=(worker_num, predictor_config), name="Scheduler")
    p.daemon = False
    p.start()
    return SchedulerHandle(process=p)
```

注意这里不能写成 `Process(target=scheduler.run)`——因为 spawn 会尝试 pickle `Scheduler` 实例，而它内部包含 `queue.Queue`（不可 pickle），所以必须在子进程入口函数里再构造。

#### ZMQ 双 ROUTER 架构

Scheduler 持有两个 ZMQ ROUTER socket：

```python
def _init_zmq_sockets(self):
    self.ctx = zmq.Context()
    # 面向 Gunicorn 客户端
    self.router = self.ctx.socket(zmq.ROUTER)
    self.router.bind("ipc:///tmp/scheduler_router.sock")
    # 面向 Worker 进程
    self.worker_router = self.ctx.socket(zmq.ROUTER)
    self.worker_router.bind("ipc:///tmp/worker_router.sock")

    self.poller = zmq.Poller()
    self.poller.register(self.router, zmq.POLLIN)
    self.poller.register(self.worker_router, zmq.POLLIN)
```

- `router`：接收 `SchedulerClient`（DEALER）发来的任务请求，ROUTER 会自动记录每个 DEALER 的 identity，回包时按 identity 路由。
- `worker_router`：向 `WorkerProcess`（DEALER）派发任务和接收结果。

使用 IPC 而非 TCP，是因为所有组件在同一台机器上运行，IPC 比 TCP 少了网络协议栈的开销。

#### 主循环

```python
def run(self):
    self._start_workers(self.worker_num)
    self._start_process_manager()
    self._init_zmq_sockets()

    while self._is_running:
        events = dict(self.poller.poll(timeout=100))
        if self.worker_router in events:
            self._handle_worker_result()
        if self.router in events:
            self._handle_client_request()
        self._schedule()
```

主循环做三件事：

1. 收 Worker 的结果（有结果就路由回客户端）
2. 收客户端的新请求（按优先级入队）
3. 执行调度（分配任务 / 抢占）

`poller.poll(timeout=100)` 是事件驱动的核心：100ms 超时确保即使没有事件也能周期性执行 `_schedule()`，不会死等。

#### 调度策略

`_schedule()` 分三个阶段：

```python
def _schedule(self):
    self._check_background_workers()

    # 阶段一：高优任务分配给空闲 Worker
    while not self.high_queue.empty():
        worker_id = self._get_idle_worker()
        if worker_id is not None:
            self._assign(worker_id, self.high_queue.get())
        else:
            break

    # 阶段二：高优任务仍有剩余，抢占低优 Worker
    while not self.high_queue.empty():
        killed_id = self._preempt_low_priority()
        if killed_id is not None:
            self._handle_kill_low_priority(killed_id)
            if len(self.standby_workers) >= 1:
                self._arrange_background_workers(killed_id)
                self._assign(killed_id, self.high_queue.get())
            # 通知 ProcessManager 异步补建 standby
            self.process_manager_task_queue.put({"type": "start", "worker_id": killed_id})
        else:
            break

    # 阶段三：没有高优任务时，分配低优任务
    while not self.low_queue.empty():
        worker_id = self._get_idle_worker()
        if worker_id is None:
            break
        self._assign(worker_id, self.low_queue.get())
```

### 2. 抢占机制

抢占是这套调度系统最关键的能力。当高优任务到达但所有 Worker 都在忙时，调度器会：

1. **选择一个正在执行低优任务的 Worker**
2. **杀掉该 Worker 进程**（先 SIGTERM，等待短暂时间后 SIGKILL）
3. **向被抢占任务的客户端返回 preempted 状态**
4. **用预热好的 standby Worker 替换被杀的 Worker**
5. **将高优任务分配给新 Worker**
6. **异步通知 ProcessManager 补建新的 standby Worker**

```python
def _kill_worker(self, worker_id):
    w = self.active_workers[worker_id]
    p = w["process"]
    try:
        p.stop()
        os.kill(p.pid, signal.SIGTERM)
        for _ in range(5):
            res, _ = os.waitpid(p.pid, os.WNOHANG)
            if res != 0:
                break
            time.sleep(0.001)
        # 优雅关闭失败，强制杀死
        os.kill(p.pid, signal.SIGKILL)
        for _ in range(5):
            pid, _ = os.waitpid(p.pid, os.WNOHANG)
            if pid == p.pid:
                return
            time.sleep(0.001)
    except Exception as e:
        os.kill(p.pid, signal.SIGKILL)
```

被抢占的客户端会收到明确的状态码：

```python
msg = {
    "task_id": task_id,
    "status": "preempted",
    "result": None,
    "error_msg": None,
}
```

### 3. Worker 进程

WorkerProcess 是实际执行推理的进程。每个 Worker 在启动时：

1. 根据 `predictor_config` 在子进程内构造 predictor 实例
2. 调用 `predictor.load_model(device)` 加载模型到指定设备
3. 初始化 ZMQ DEALER socket 并连接到 Scheduler 的 worker_router
4. 进入事件循环，等待任务、执行、回包

```python
class WorkerProcess(Process):
    def run(self):
        self._init_predictor()       # 子进程内构造 predictor
        self._init_zmq_sockets()     # 子进程内初始化 ZMQ

        if self.predictor is not None:
            self.predictor.load_model(self.device)

        if self.ready_event:
            self.ready_event.set()   # 通知 ProcessManager 已就绪

        while self._running:
            socks = dict(self.poller.poll(self.timeout * 1000))
            if self.worker_socket in socks:
                _, reply = self.worker_socket.recv_multipart()
                task = json.loads(reply.decode())
                try:
                    result = self.run_task(task)
                    self.worker_socket.send_multipart([b"", json.dumps(result).encode()])
                except Exception as e:
                    error_result = {"status": "failed", "error_msg": str(e), ...}
                    self.worker_socket.send_multipart([b"", json.dumps(error_result).encode()])
```

#### Active vs Standby

系统启动时会同时创建两组 Worker：

| 类型           | 设备          | 用途                                                 |
| :------------- | :------------ | :--------------------------------------------------- |
| Active Worker  | `device=cuda` | 正常接收并执行任务                                   |
| Standby Worker | `device=cpu`  | 预热 predictor（仅 CPU），不参与调度，等待抢占时切换 |

Standby Worker 在 CPU 上预先加载了模型，当发生抢占替换时，只需要将模型从 CPU 搬到 GPU（`load_model("cuda")`），比从零启动一个新进程快得多。

### 4. ProcessManager

ProcessManager 是一个后台线程，负责异步补建 standby Worker。当 Scheduler 抢占消耗了一个 standby Worker 后，会通过 `process_manager_task_queue` 发送补建请求：

```python
class ProcessManager(Thread):
    def run(self):
        while self._running:
            task = self.task_queue.get()
            if task.get("type") == "start":
                worker_id = task.get("worker_id")
                ready_event = Event()
                p = WorkerProcess(worker_id=worker_id, device="cpu",
                                  ready_event=ready_event,
                                  predictor_config=self.predictor_config)
                p.start()
                if ready_event.wait(timeout=120):
                    self.result_queue.put({
                        "type": "worker_ready",
                        "worker_id": worker_id,
                        "process": p
                    })
```

ProcessManager 使用 `Event` 机制等待新 Worker 加载完成。Scheduler 在每轮 `_schedule()` 开头调用 `_check_background_workers()` 非阻塞地检查是否有新 standby 就绪。

### 5. SchedulerClient

SchedulerClient 运行在 Gunicorn Worker 进程中，负责将业务请求通过 ZMQ 发送给 Scheduler，并同步等待结果。

```python
class SchedulerClient:
    _instance = None
    _lock = threading.Lock()
    _thread_local = threading.local()

    @property
    def socket(self):
        if not hasattr(self._thread_local, 'socket'):
            self._thread_local.socket = self.ctx.socket(zmq.DEALER)
            identity = str(uuid.uuid4()).encode()
            self._thread_local.socket.setsockopt(zmq.IDENTITY, identity)
            self._thread_local.socket.connect(SCHEDULER_ROUTER_ADDR)
        return self._thread_local.socket

    def submit_task(self, task_id=None, payload=None, priority="high", timeout=60.0):
        task_msg = {
            "task_id": task_id,
            "priority": priority,
            "payload": payload,
            "submit_ts": time.time(),
            "timeout": timeout
        }
        self.socket.send_multipart([b"", json.dumps(task_msg).encode()])
        socks = dict(self.poller.poll(timeout * 1000))
        if self.socket in socks:
            _, reply = self.socket.recv_multipart()
            return json.loads(reply)
        else:
            raise TimeoutError("Scheduler response timeout")
```

关键设计点：

- **单例模式**：每个 Gunicorn Worker 进程只有一个 `SchedulerClient` 实例，共享同一个 `zmq.Context`。
- **线程局部 Socket**：Gunicorn 的 gthread worker 是多线程的，ZMQ Socket 不是线程安全的，所以每个线程持有自己的 DEALER socket。`zmq.Context` 是线程安全的，可以跨线程共享。
- **DEALER + 空帧**：DEALER 发消息时需要加一个空帧 `b""`，以兼容 ROUTER 端期望的 `[identity, empty, payload]` 三帧结构。

## 写在最后

为什么要记录这个任务的实现，因为最近在vllm的整体架构，我发现vllm的整体的进程架构和我自己实现的这个功能很类似，在实现之前我还没开始看vllm的整体架构，看完却发现似曾相识，这样理解起来vllm的架构相对来说就轻松一些了。

{% include figure.liquid loading="eager" path="assets/img/posts/scheduler/vllm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
