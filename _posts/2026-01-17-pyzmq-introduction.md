---
layout: post
title: PyZMQ的简介
date: 2026-01-17 17:10:00
description: 使用PyZMQ过程中的记录。
tags: computer communication
categories: programming
giscus_comments: false
related_posts: false
pretty_table: true

toc:
  sidebar: left
---

## 一、从 Socket 谈起：为什么要进化？

### 1. 什么是socket？

我最早接触 ZMQ，是在 sglang 的代码里看到它被用来做进程间通信。当时的理解很模糊：

> ZMQ 好像是个用来进程通信的库。
{: .block-tip }

真正去调研之后才发现，ZMQ 并不是“绕过” Socket，而是构建在 Socket 之上：
它可以在进程内、进程间，甚至跨机器（TCP）进行消息传递，而底层依然是我们熟悉的网络通信机制。

于是问题就回到了一个老问题上：
Socket 到底是什么？

简单来说，Socket（套接字）是应用程序访问 TCP/IP 协议栈的统一接口。
它不是协议本身，而是操作系统提供的一层抽象 API，用来屏蔽底层复杂的网络实现。

你可以把它理解为一个“插口”：

- 一端连着应用程序

- 另一端连着 TCP / UDP 协议栈

- 应用只需要通过 Socket 读写数据，而不需要关心包是怎么在网络中传输的

从设计角度看，Socket 本质上是一个门面（Facade）：
它把复杂的 TCP/IP 细节隐藏在一组统一的接口之后，让应用层可以用相对简单的方式进行网络通信。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/pyzmq/socket.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 1. socket层级图.
</div>

### 2. 有什么痛点？

如果只做简单的点对点通信，原生 Socket 完全够用。
但一旦系统开始变复杂，问题就会迅速暴露出来。

#### 1. 消息边界问题（粘包 / 半包）

TCP 是面向字节流的协议，它并不知道什么是“一条完整的业务消息”。
你发送的一条"完整消息"可能被拆成多个 TCP 包，或者多条消息粘在一起被一次性接收。这意味着你必须手动实现消息帧的分割与重组逻辑，例如添加长度前缀或特殊分隔符。

#### 2. 连接与重连逻辑

Socket 本身是脆弱的，如果服务端重启或网络抖动，客户端的 Socket 连接会断开。你需要手写自动重连、超时重传、心跳检测等一整套容错机制，代码复杂且容易出错。而这些代码，和业务逻辑几乎没有任何关系。

#### 3. 并发与扩展性成本高

传统的同步阻塞 Socket（一个连接一个线程）在高并发场景下性能低下。改用非阻塞 I/O 或多路复用（如 select/epoll）又会让代码复杂度急剧上升，需要精心设计事件循环和状态机。
这时你会发现： 通信本身反而成了系统中最难维护的部分。

#### 4. 缺乏高层消息模式

Socket 只提供点对点的双向通道，如果要实现发布-订阅、请求-响应、任务分发等常见模式,你需要从零开始设计协议和拓扑结构。

## 二、ZMQ：不仅仅是 Socket

### 1. ZMQ是什么？

官方定义为：

> ZeroMQ (also known as ØMQ, 0MQ, or zmq) looks like an embeddable networking library but acts like a concurrency framework. It gives you sockets that carry atomic messages across various transports like in-process, inter-process, TCP, and multicast. You can connect sockets N-to-N with patterns like fan-out, pub-sub, task distribution, and request-reply. It's fast enough to be the fabric for clustered products. Its asynchronous I/O model gives you scalable multicore applications, built as asynchronous message-processing tasks. It has a score of language APIs and runs on most operating systems.
{: .block-tip }

也就是说[ZeroMQ](https://zeromq.org/)可嵌入的网络库，但实际上更像是一个并发框架。它提供套接字，支持跨进程内、进程间、TCP 和多播等多种传输方式传递原子消息。您可以利用扇出、发布/订阅、任务分发和请求/响应等模式，实现多对多套接字连接。它的速度足以支撑集群产品的构建。其异步 I/O 模型支持可扩展的多核应用程序，这些应用程序以异步消息处理任务的形式构建。它拥有丰富的语言 API，并且可以在大多数操作系统上运行。

ZMQ 并不取代 Socket，它是建立在 Socket 之上的高层抽象，把很多重复、容易出错的通信逻辑封装起来，同时提供了**“消息模式”**来约束和指导你的通信结构。

### 2. ZMQ解决了哪些问题？

针对我们在原生 Socket 中遇到的痛点，ZMQ 提供了如下的解决方案：

#### 1. 解决消息边界：从“字节流”到“离散消息”

ZMQ 彻底终结了“粘包”噩梦。在 ZMQ 中，你发送的是原子级的消息块（Message Frames）。
在 TCP 之上，ZMQ 把消息拆成“帧”，保证你发送的消息在对端完整接收。你不必再自己设计长度字段、分隔符、缓冲区状态机。

#### 2. 解决可靠性：静默重连与拓扑无关

自动重连：如果你的 Worker 进程崩溃重启，Scheduler 甚至不需要感知。ZMQ 会在后台自动排队消息并不断尝试重连，直到链路恢复。

启动顺序无关：你可以先启动 Client，再启动 Server。Client 会静默等待直到连接建立，不会抛出任何报错。这让分布式系统的启动脚本变得无比简单。

#### 3. 解决并发难题：全异步的后台 IO

原生 Socket 的并发往往意味着复杂的 select/epoll。

什么是select？ 意思是对所有的socket轮询，哪个有数据就通知哪个。

```
ready_sockets = select(read_list)
for s in ready_sockets:
    s.recv()
```

什么是epoll？ epoll 是 Linux 为解决高并发而设计的进阶工具（也是现在高性能服务器如 Nginx、Redis 的底层基石）。把 socket 注册到 epoll，epoll 在内核中维护监听集合，返回有事件的socket，监听到了之后通知.

```
events = epoll.wait()
for fd in events:
    handle(fd)
```

而 ZMQ 内部维护了一个专用的 IO 线程。

零阻塞：当你的业务代码调用 send() 时，数据其实是先进入了一个内存缓冲区。IO 线程会在后台处理真正的网络传输。

无锁并发：ZMQ 的核心设计遵循“不要通过共享内存来通信，而要通过通信来共享内存”，极大地减少了多线程编程中的锁冲突。你只和“本地队列”打交道，真正的网络 IO 由后台 IO 线程统一处理。每个 Context 内部会启动 1 个或多个 IO 线程，这些线程统一管理所有网络 socket，负责连接、重连、读写、发送缓冲。

#### 4. 跨平台、跨语言

ZMQ 提供 Python、C、C++、Go 等语言接口。
进程间、主机间通信接口一致，移植成本极低。

## 三、灵魂：ZMQ中的消息模式

### 1. 消息

[ZeroMQ](https://zeromq.cn/) 消息是在应用程序之间或同一应用程序组件之间传递的离散数据单元。从 ZeroMQ 本身的角度来看，消息被视为不透明的二进制数据。

在线路上传输时，ZeroMQ 消息是大小从零开始、适合内存的任意二进制块。你可以使用 protocol buffers、msgpack、JSON 或应用程序需要通信的任何其他方式进行自己的序列化。选择一种可移植的数据表示形式是明智的，但你可以自行决定权衡。

最简单的 ZeroMQ 消息由一个帧（也称为消息部分）组成。帧是 ZeroMQ 消息的基本线路格式。帧是一个长度指定的数据块。长度可以从零开始。ZeroMQ 保证要么传递消息的所有部分（一个或多个），要么都不传递。这允许你将帧列表作为单个在线路上传输的消息发送或接收。

消息（单部分或多部分）必须适合内存。如果你想发送任意大小的文件，应该将其分成多块，并将每块作为单独的单部分消息发送。使用多部分数据不会减少内存消耗

### 2. 消息模式介绍

ZeroMQ 的 Socket API 仅仅是冰山一角，藏在朴实接口之下的，是一个丰富多彩的**消息模式（Messaging Patterns）**世界{%sidenote 'One' '参考[ZMQ中文文档](https://zeromq.cn/socket-api/) 这里面写的比较详细，易懂'%}。

在 ZMQ 中，Socket 不是孤立存在的，它们必须成对出现（Socket Pairs）。不同的 Socket 类型组合在一起，构成了特定的拓扑结构，决定了消息是如何路由、排队以及在异常状况下如何处理的。

#### 1. 请求-应答模式 (Request-Reply)

**场景**：远程过程调用 (RPC)、任务分发。

**定义**：[RFC 28/REQREP](https://rfc.zeromq.org/spec/28/)

这是最常见的服务导向架构模式。它有两种主要形式：**同步**（严谨的乒乓交互）和**异步**（灵活的流量控制）。

##### A. 同步双雄：REQ 与 REP

这是最简单的形式，类似于“打电话”。

- **REQ (客户端)**

  - **行为**：严格遵守“发送 -> 接收 -> 发送”的顺序。
  - **路由**：
    - **出站**：轮询 (Round-Robin)。如果连接了多个服务，请求会轮流发给它们。
    - **入站**：匹配最后发出的请求。
  - **阻塞**：如果没有可用服务，发送操作会**阻塞**，直到至少有一个服务上线。它非常执着，**不会丢弃消息**。

- **REP (服务端)**
  - **行为**：严格遵守“接收 -> 发送 -> 接收”的顺序。
  - **路由**：
    - **入站**：公平队列 (Fair-Queuing)。从所有连接的客户端中公平地接收请求。
    - **出站**：总是路由回发送最后那个请求的客户端。
  - **异常**：如果原来的请求者断线了，应答会被**静默丢弃**。

##### B. 异步双雄：DEALER 与 ROUTER

这是 ZMQ 最强大的组合，常用于构建自定义调度器、代理或高并发服务。

- **DEALER (异步客户端/工头)**

  - **定位**：REQ 的异步版本。它不用傻等回复，可以像机关枪一样连续发送。
  - **行为**：**全双工**。发送和接收没有任何顺序限制。
  - **路由**：
    - **出站**：轮询 (Round-Robin) 分发给后端。
    - **入站**：公平队列接收结果。
  - **静默状态**：如果缓冲区满了（高水位）或没有对等方，它会阻塞发送，但**绝不丢弃消息**。
  - **注意**：当连接到 REP 时，DEALER 必须在消息头部手动加一个空帧作为分隔符。

- **ROUTER (异步服务端/调度中心)**
  - **定位**：REP 的异步版本。它是网络拓扑中的“大脑”，具有**显式寻址**能力。
  - **身份魔法 (Identity)**：
    - **接收时**：ZMQ 会自动在消息最前面插入一帧，包含发送者的 **路由 ID (Routing ID)**。应用层通过读取这一帧知道消息是谁发的。
    - **发送时**：应用层必须把目标的 **路由 ID** 放在消息第一帧。ZMQ 读到这一帧，就把消息路由给对应的连接，并剥离该帧。
  - **静默状态**：最无情的一个 Socket。如果目标 ID 不存在，或者目标缓冲区已满，消息会被**直接丢弃 (Drop)**。它不会阻塞调用者。

##### C. 混合使用

同步和异步可以混合使用，比如客户端可以是`REQ` 而 服务端是`ROUTER`， 意思是客户端是傻瓜式的、同步的；而服务端是精细的、异步的，这种组合允许你保留客户端代码的简单性（写个循环 send/recv 即可），同时在服务端获得极高的并发处理能力。

想象一个银行柜台的场景：

- 客户端 (REQ)：是普通储户。

  - 逻辑很简单：去窗口 -> 递交存折 -> 等着（阻塞） -> 拿回回执 -> 离开。

  - 储户不需要复杂的并发能力，一次只办一件事。

- 服务端 (ROUTER)：是那个超级业务员（或者业务分发系统）。

  - 逻辑很复杂：它同时面对成千上万个储户。

  - 它不能因为张三存钱动作慢，就让李四一直排队。它必须能同时接收所有人的单子，处理完一个就给对应的那个储户扔回去。

---

#### 2. 发布-订阅模式 (Publish-Subscribe)

**场景**：数据分发、实时行情、日志收集。

**定义**：[RFC 29/PUBSUB](https://rfc.zeromq.org/spec/29/)

这是一种一对多的广播模式，类似于“收音机”。

- **PUB (发布者)**

  - **行为**：只管发，不收。
  - **路由**：**扇出 (Fan-out)**。一份消息会复制分发给所有当前的订阅者。
  - **静默状态**：**即发即弃**。如果订阅者处理不过来（达到高水位），或者根本没有订阅者，消息会被**丢弃**。发送永远不会阻塞。
  - **主题**：必须在消息第一帧指定“主题 (Topic)”。

- **SUB (订阅者)**

  - **行为**：只管收，不发。
  - **过滤机制**：
    - SUB Socket 默认是**聋子**（不接收任何消息）。
    - 必须通过 `ZMQ_SUBSCRIBE` 设置订阅主题。
    - **前缀匹配**：订阅 "topic" 会收到 "topic/A"，但收不到 "topi"。
    - 订阅空字符串 `""` 表示接收所有。

- **XPUB & XSUB (代理扩展)**
  - 用于在中间构建 Proxy。它们允许“订阅信息”像数据一样在网络中传递，从而让上游的发布者知道下游到底想要什么数据。

---

#### 3. 流水线模式 (Pipeline)

**场景**：并行计算、MapReduce、多阶段任务处理。

**定义**：[RFC 30/PIPELINE](rfc.zeromq.cn/spec:30)

这是一个单向的推拉结构，像工厂的流水线一样。它主要关注吞吐量和负载均衡。

- **PUSH (推/分发者)**

  - **行为**：只发不收。
  - **路由**：**轮询 (Round-Robin)**。这也是一种天然的**负载均衡器**。
  - **静默状态**：如果下游节点都满了，它会**阻塞**，直到有空位。它保证任务不丢失。

- **PULL (拉/工入)**
  - **行为**：只收不发。
  - **路由**：**公平队列 (Fair-Queuing)**。它不会让某个来源饿死。

---

#### 4. 独占对模式 (Exclusive Pair)

**场景**：进程内 (inproc) 线程间通信。

**定义**：[RFC 31/EXPAIR](https://rfc.zeromq.org/spec/31/)

- **PAIR**
  - **特点**：**1 对 1 绑定**。它不像其他 Socket 那样支持 N:M 连接。
  - **警告**：不要用于 TCP！因为它不支持自动重连。如果连接断开，它就废了。它就是为了替代传统的线程锁（Mutex）而生的。

---

#### 5. 草稿模式 (Draft Patterns)

_注：这些模式较新，可能需要特定的编译选项支持。_

- **客户端-服务器 (Client-Server)**

  - **目的**：为了简化 ROUTER/DEALER 的复杂性。
  - **SERVER**：类似于 ROUTER，但它不使用“消息帧”来存 ID，而是返回一个 32 位的整数 `routing_id`。发送时也是通过 API 指定 ID，而不是拼接消息帧。它是线程安全的且不会丢弃消息。
  - **CLIENT**：线程安全的客户端。

- **Radio-Dish**
  - **目的**：更简单的发布订阅。
  - **特点**：使用 **"Group" (组)** 的概念代替 "Topic"。它是**精确匹配**，而不是前缀匹配。长度限制 255 字节。

---

#### 6. 核心模式全景对比表

| Socket 类型 | 典型搭档    | 模式 / 角色        | 发送/接收  | 出站路由 (Outbound)   | 入站路由 (Inbound) | 静默/异常行为          |
| :---------- | :---------- | :----------------- | :--------- | :-------------------- | :----------------- | :--------------------- |
| **REQ**     | REP, ROUTER | 请求-应答 / 客户端 | 严格交替   | 轮询 (Round-Robin)    | 匹配最后请求       | **阻塞** (直到有服务)  |
| **REP**     | REQ, DEALER | 请求-应答 / 服务端 | 严格交替   | 回复给请求者          | 公平队列           | 若客户端消失则**丢弃** |
| **DEALER**  | ROUTER, REP | 异步 / 客户端      | 双向无限制 | 轮询 (Round-Robin)    | 公平队列           | **阻塞** (不丢消息)    |
| **ROUTER**  | DEALER, REQ | 异步 / 核心路由    | 双向无限制 | **显式寻址** (根据ID) | 公平队列 + 注入ID  | 无目标或满载时**丢弃** |
| **PUB**     | SUB, XSUB   | 发布-订阅 / 发送方 | 仅发送     | **扇出 (Fan-out)**    | N/A                | 满载或无订阅则**丢弃** |
| **SUB**     | PUB, XPUB   | 发布-订阅 / 接收方 | 仅接收     | N/A                   | 公平队列           | **过滤** (默认全丢)    |
| **PUSH**    | PULL        | 流水线 / 任务源    | 仅发送     | 轮询 (负载均衡)       | N/A                | **阻塞** (等待下游)    |
| **PULL**    | PUSH        | 流水线 / 任务工    | 仅接收     | N/A                   | 公平队列           | **阻塞** (等待数据)    |
| **PAIR**    | PAIR        | 独占 / 线程通信    | 双向       | 1对1 独占             | 1对1 独占          | **阻塞** (不丢消息)    |

## 四、实战：用pyzmq构建一个异步客户端服务端消息流程

因为是异步的消息流程，因此我们将使用 DEALER 作为客户端，ROUTER 作为服务端，客户端可以连续“投递”任务，服务端可以并行处理，并根据任务完成的先后顺序异步“回邮”。


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/pyzmq/router-dealer.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig. 2. Router-Dealer结构图.
</div>


### 客户端：异步的“任务提交者” (DEALER)

客户端使用 DEALER，它不会像 REQ 那样因为没收到回复而锁定 send() 操作。

```
import zmq
import uuid
import json
import time
import threading
from aipinfer import logger
from .utils import config_socket,SCHEDULER_ROUTER_ADDR

# 全局的 Context，ZMQ Context 是线程安全的，可以在进程内共享
# 但必须确保它是在 Worker 进程启动后创建的，或者在 fork 后重建
# 为了安全，我们也把它放在 lazy load 里
import zmq
import json
import uuid

def run_client():
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)

    # 设置身份，方便服务端回邮
    client_id = b"Client-A"
    socket.setsockopt(zmq.IDENTITY, client_id)
    socket.connect("ipc:///tmp/scheduler.sock")

    # 构建任务
    task_msg = {
        "task_id": str(uuid.uuid4()),
        "payload": "Calculate AI Inference",
        "priority": "high"
    }

    print(f"发送任务: {task_msg['task_id']}")

    # 【核心点】发送多帧消息：空帧 + 序列化后的数据
    # DEALER 模拟 REQ 行为时，必须加一个空帧（Delimiter）
    socket.send_multipart([b"", json.dumps(task_msg).encode()])

    # 使用 Poller 等待结果（不阻塞整个进程）
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    if poller.poll(timeout=5000):  # 等待 5 秒
        # 接收时也要注意：第一帧是空帧，第二帧才是数据
        _, reply = socket.recv_multipart()
        print(f"收到回复: {reply.decode()}")
    else:
        print("任务超时")

    socket.close()
    context.term()
```

### 服务端：精明的“任务调度员” (ROUTER)

服务端采用 ROUTER 模式，它是整个调度器的“大脑”。与简单的 REP 不同，ROUTER 能同时“记住”成千上万个客户端的身份。

- 核心逻辑：
  身份剥离：
  当 recv_multipart() 被调用时，ZMQ 会自动在消息头部注入客户端的 identity。这就像信封上的回信地址，调度器处理完任务后，必须靠它寄回结果。

```
def _init_zmq_sockets(self):
    self.ctx = zmq.Context()
    # gunicorn worker <-> scheduler
    self.router = self.ctx.socket(zmq.ROUTER)
    self.router.bind("ipc:///tmp/scheduler_router.sock")

    self.poller = zmq.Poller()
    self.poller.register(self.router, zmq.POLLIN)

def run(self):
    self._init_zmq_sockets()

    while self._is_running:
        events = dict(self.poller.poll(timeout=100))

        if self.router in events:
            identity, _, msg = self.router.recv_multipart()
            req = json.loads(msg.decode())
            # 记住identity，回复的时候只回复给identity对应的client
            # do your job
```
