---
layout: post
title: AIGC 模型量化工具调研报告
date: 2026-04-15 12:00:00
description: 对主流量化工具的调研与对比
tags: quantization
categories: AI
pretty_table: true
toc:
  sidebar: left
---

## 1. 调研背景与目标

### 1.0 说明

文档全是由claude生成，基本没有改动

### 1.1 背景

随着 AIGC（AI Generated Content）技术的快速发展，大语言模型（LLM）、视觉语言模型（VLM）、扩散模型（Diffusion Model）等各类生成式 AI 模型在文本生成、图像生成、视频生成、语音合成等场景中得到了广泛应用。然而，这些模型的参数规模日益庞大（从数十亿到数千亿参数），对计算资源和显存的需求极高，严重制约了模型的部署效率和推理成本。

模型量化（Quantization）作为一种关键的模型压缩技术，通过将模型权重和/或激活值从高精度浮点格式（如 FP32/FP16/BF16）映射到低精度数据格式（如 INT8、INT4、FP8、FP4 等），可以有效降低模型的存储需求和计算开销，同时在可接受的精度损失范围内大幅提升推理速度。

当前，开源社区和各大厂商已推出多种量化工具和框架，各具特色且持续迭代。为了在 AIGC 业务场景中选择最合适的量化方案，有必要对主流量化工具进行系统性调研和对比分析。

### 1.2 调研目标

本次调研的核心目标包括：

1. 全面梳理：系统梳理 7 款主流模型量化工具的功能特性、技术架构和生态系统
2. 深度对比：从量化方法、量化粒度、模型支持、部署集成等多个维度进行横向对比
3. 场景匹配：结合 AIGC 业务场景（LLM 推理、图像生成、多模态理解等）评估各工具的适用性
4. 选型建议：基于调研结果，为不同 AIGC 场景提供量化工具选型建议

### 1.3 调研范围

本次调研覆盖以下 7 款量化工具：

| 序号 | 工具名称                 | 开发方                 | GitHub 仓库                                    |
| :--- | :----------------------- | :--------------------- | :--------------------------------------------- |
| 1    | llm-compressor           | vLLM 项目组            | https://github.com/vllm-project/llm-compressor |
| 2    | LightCompress (LLMC)     | 商汤 ModelTC           | https://github.com/ModelTC/LightCompress       |
| 3    | AngelSlim                | 腾讯                   | https://github.com/tencent/AngelSlim           |
| 4    | DeepCompressor           | MIT HAN Lab / Nunchaku | https://github.com/nunchaku-ai/deepcompressor  |
| 5    | TorchAO Quantization     | PyTorch 官方           | https://github.com/pytorch/ao                  |
| 6    | TensorRT Model Optimizer | NVIDIA                 | https://nvidia.github.io/Model-Optimizer/      |
| 7    | Optimum-Quanto           | Hugging Face           | https://github.com/huggingface/optimum-quanto  |

## 2\. 量化技术基础

### 2.1 量化基本概念

模型量化是将浮点数值映射到低精度整数或浮点数的过程。核心公式为：

对称量化：Q(x) = round(x / s), 其中 s 为缩放因子（scale）

非对称量化：Q(x) = round(x / s) + z, 其中 z 为零点（zero-point）

### 2.2 量化方法分类

#### 按量化阶段分类

| 方法                | 说明                                 | 优势               | 劣势                     |
| :------------------ | :----------------------------------- | :----------------- | :----------------------- |
| PTQ（训练后量化）   | 模型训练完成后直接量化，无需重新训练 | 速度快、资源消耗低 | 低精度下精度损失可能较大 |
| QAT（量化感知训练） | 在训练过程中模拟量化效果             | 精度更高           | 需要训练数据和计算资源   |

#### 按量化对象分类

| 方法                               | 说明                              | 典型格式         |
| :--------------------------------- | :-------------------------------- | :--------------- |
| Weight-Only（仅权重量化）          | 仅量化模型权重                    | W4A16, W8A16     |
| Weight-Activation（权重+激活量化） | 同时量化权重和激活值              | W8A8, W4A8, W4A4 |
| KV Cache 量化                      | 量化注意力机制中的 Key-Value 缓存 | FP8 KV, INT4 KV  |

#### 按核心算法分类

| 算法策略                | 代表方法                | 原理                                            |
| :---------------------- | :---------------------- | :---------------------------------------------- |
| Round-to-Nearest (RTN)  | 直接四舍五入            | 最简单的 PTQ 方法，直接将权重映射到最近的量化值 |
| 变换类 (Transformation) | SmoothQuant, AWQ, OS+   | 通过等价变换减少激活/权重中的异常值             |
| 裁剪类 (Clipping)       | AWQ (含裁剪), OmniQuant | 限制权重范围以减少量化误差                      |
| 重构类 (Reconstruction) | GPTQ, AutoRound         | 逐列/逐组更新未量化权重以补偿量化误差           |
| 旋转类 (Rotation)       | QuaRot, SpinQuant, QuIP | 通过正交矩阵变换消除异常值                      |
| SVD 分解类              | SVDQuant                | 通过 SVD 低秩分解吸收异常值                     |

### 2.3 量化粒度

| 粒度                  | 说明                                               | 精度 | 效率 |
| :-------------------- | :------------------------------------------------- | :--- | :--- |
| Per-Tensor（每张量）  | 整个张量共享一个 scale/zp                          | 低   | 高   |
| Per-Channel（每通道） | 每个输出通道独立的 scale/zp                        | 中   | 中   |
| Per-Group（每组）     | 将通道分组，每组独立 scale/zp（如 group_size=128） | 高   | 中   |
| Per-Token（每token）  | 激活值按 token 维度独立量化                        | 中   | 中   |
| Per-Block（块级）     | 将张量分块（如 128×128），每块独立 scale           | 高   | 中   |
| Per-Head（每头）      | KV Cache 按注意力头独立量化                        | 高   | 中   |

### 2.4 数据格式

| 格式            | 位宽  | 说明                               |
| :-------------- | :---- | :--------------------------------- |
| INT8            | 8 bit | 经典整数量化，硬件支持广泛         |
| INT4            | 4 bit | 极低精度整数量化，通常需要分组量化 |
| INT2            | 2 bit | 超低精度，实验性质                 |
| FP8 (E4M3/E5M2) | 8 bit | 浮点量化，对长尾分布更友好         |
| FP4 (E2M1)      | 4 bit | NVIDIA NVFP4 格式                  |
| MXFP8           | 8 bit | 微缩浮点格式，块级缩放             |
| MXFP4           | 4 bit | 微缩浮点格式，块级缩放             |

## 3\. 调研工具概览

| 特性     | llm-compressor      | LightCompress       | AngelSlim      | DeepCompressor    | TorchAO          | TensorRT ModelOpt | Optimum-Quanto  |
| :------- | :------------------ | :------------------ | :------------- | :---------------- | :--------------- | :---------------- | :-------------- |
| 开发方   | vLLM 项目组         | 商汤 ModelTC        | 腾讯           | MIT HAN Lab       | PyTorch 官方     | NVIDIA            | Hugging Face    |
| 开源协议 | Apache 2.0          | Apache 2.0          | —              | MIT               | BSD-3            | Apache 2.0/MIT    | Apache 2.0      |
| Stars    | ~5k                 | ~700                | ~500           | ~774              | ~8k              | ~2k               | ~1k             |
| 活跃度   | 非常活跃            | 活跃                | 活跃           | 中等活跃          | 非常活跃         | 活跃              | 维护模式        |
| 主要语言 | Python              | Python              | Python         | Python            | Python/C++/CUDA  | Python/C++        | Python/CUDA     |
| 核心定位 | LLM PTQ + vLLM 部署 | AIGC 全模态压缩基准 | 全链路模型压缩 | 扩散模型+LLM 压缩 | PyTorch 原生量化 | NVIDIA 硬件优化   | HF 生态量化后端 |

## 4\. 各工具详细分析

### 4.1 llm-compressor

项目地址: https://github.com/vllm-project/llm-compressor

#### 4.1.1 项目概述

llm-compressor 是 vLLM 项目组开发的大语言模型压缩工具，专为 vLLM 推理引擎设计，是 vLLM 生态中模型量化的官方推荐工具。项目以 compressed-tensors 作为底层序列化格式，确保量化模型可以无缝加载到 vLLM 进行高效推理。

#### 4.1.2 支持的量化方法

| 量化方法    | 类型       | 说明                                                                                                     |
| :---------- | :--------- | :------------------------------------------------------------------------------------------------------- |
| RTN         | 直接量化   | 通过 QuantizationModifier 实现 Round-to-Nearest PTQ                                                      |
| GPTQ        | 重构类     | 逐列更新权重补偿量化误差，适用于 weight-only 量化                                                        |
| AWQ         | 变换+裁剪  | 激活感知权重量化，自动搜索最优缩放因子                                                                   |
| AutoRound   | 自动优化   | 自动选择最优量化参数                                                                                     |
| SmoothQuant | 变换类     | 通过数学等价变换平滑激活异常值，适用于 W8A8                                                              |
| SpinQuant   | 旋转类     | 基于旋转矩阵的变换方法（实验性）                                                                         |
| QuIP        | 旋转类     | 通过正交变换分散权重中的异常值（outliers），从而降低低比特量化误差。目前主要用于研究探索，工业部署较少。 |
| IMatrix     | 重要性评估 | 信息矩阵引导量化                                                                                         |

#### 4.1.3 支持的量化格式

| 格式        | 权重精度 | 激活精度                | 说明                                                   |
| :---------- | :------- | :---------------------- | :----------------------------------------------------- |
| FP8_DYNAMIC | FP8      | FP8 动态 per-token      | 权重 per-channel/per-tensor                            |
| FP8_BLOCK   | FP8 块级 | FP8 动态 per-group(128) | 权重按 128×128 块量化                                  |
| W8A8 (INT8) | INT8     | INT8                    | 支持对称/非对称，per-channel/per-group                 |
| W4A16       | INT4     | FP16                    | 权重 per-group/per-channel，保留激活精度               |
| W8A16       | INT8     | FP16                    | 权重 per-group/per-channel                             |
| NVFP4       | FP4      | FP8/FP4                 | 全局 scale + 每组 16 元素 local scale（Blackwell GPU） |
| MXFP8       | MXFP8    | MXFP8                   | 微缩浮点，实验性                                       |
| MXFP4       | MXFP4    | —                       | 微缩浮点，实验性                                       |
| W4AFP8      | INT4     | FP8                     | 混合精度                                               |
| W4AINT8     | INT4     | INT8                    | 混合精度                                               |

#### 4.1.4 量化粒度支持

| 粒度        | 权重 | 激活 | 说明                             |
| :---------- | :--- | :--- | :------------------------------- |
| Per-Tensor  | ✅   | ✅   | 全张量共享 scale                 |
| Per-Channel | ✅   | —    | 沿输出维度独立 scale             |
| Per-Group   | ✅   | ✅   | 列方向分组，支持 group_size 配置 |
| Per-Token   | —    | ✅   | 激活按 token 维度动态量化        |
| Per-Block   | ✅   | ✅   | 128×128 块级量化（FP8_BLOCK）    |
| Per-Head    | —    | ✅   | KV Cache 按注意力头量化          |

#### 4.1.5 模型支持

- 文本 LLM: Llama 全系列、Qwen2/2.5/3/3.5、Mistral Large、DeepSeek-V3/R1、GLM-4、Gemma、Granite、Phi、Kimi-K2
- 视觉语言模型 (VLM): Qwen2-VL/2.5-VL/3-VL、LLaVA、InternVL、Pixtral、Gemma 3/4 多模态、Llama 4 Scout
- 音频模型: Qwen2-Audio、Whisper
- MoE 模型: Mixtral、DeepSeek-V3/R1 MoE
- ❌ 不支持: 扩散模型（Stable Diffusion、FLUX 等）

#### 4.1.6 关键特性

- vLLM 原生集成: 量化后的模型可直接通过 vllm serve 或 from vllm import LLM 加载
- 大模型支持: 通过 accelerate 和 disk/offload 机制支持超大模型量化
- 分布式支持: 支持 DDP 并行量化
- Oneshot 和 Model-Free PTQ: 两种入口模式，后者无需完整 HF 模型定义
- 校准数据集: 内置 C4、WikiText、UltraChat、GSM8K 等多种数据集
- 多种 Observer: 支持 min/max、MSE 等 scale 估计方法
- 稀疏化: 代码中保留 SparseGPT、Wanda 等剪枝算法，但 2:4 稀疏部署已不再支持

#### 4.1.7 硬件适配

文档明确将量化格式与 NVIDIA 计算能力对应：

| GPU 架构         | 推荐格式     |
| :--------------- | :----------- |
| Turing (T4)      | INT8, W4A16  |
| Ampere (A100)    | INT8, W4A16  |
| Hopper (H100)    | FP8          |
| Blackwell (B200) | NVFP4, MXFP4 |

### 4.2 LightCompress (LLMC)

项目地址: https://github.com/ModelTC/LightCompress

#### 4.2.1 项目概述

LightCompress（前身为 LLMC）是商汤科技 ModelTC 团队开发的 AIGC 模型压缩工具包，学术论文发表于 EMNLP 2024 和 AAAI 2026。该工具定位为"强大的大模型压缩基准和工具包"，整合了多种压缩算法，覆盖 LLM、VLM 和视频生成模型的量化与压缩。

#### 4.2.2 支持的量化方法

LightCompress 集成了 16+ 种量化算法，覆盖三大策略类别：

**变换类 (Transformation)**:

| 方法            | 说明                                                                                 |
| :-------------- | :----------------------------------------------------------------------------------- |
| **SmoothQuant** | 激活/权重间的等价缩放变换                                                            |
| **AWQ**         | 激活感知的权重缩放变换                                                               |
| **OS+**         | 离群值抑制+（Outlier Suppression Plus）                                              |
| **OmniQuant**   | 可学习的等价变换（Learnable Weight Clipping \+ Learnable Equivalent Transformation） |
| **QuaRot**      | 基于旋转矩阵的异常值消除                                                             |

**重构类 (Reconstruction)**:

| 方法     | 说明                            |
| :------- | :------------------------------ |
| **GPTQ** | 基于 Hessian 信息的逐列权重重构 |

**组合方法**:

| 方法                 | 说明                                                         |
| :------------------- | :----------------------------------------------------------- |
| **AWQ \+ OmniQuant** | AWQ 搜索参数作为 OmniQuant 的初始化，训练轮次从 20-40 降至 5 |
| **AWQ \+ GPTQ**      | 缩放变换 \+ 重构                                             |
| **QuaRot \+ GPTQ**   | 旋转变换 \+ 重构（效果最佳组合之一）                         |

#### 4.2.3 支持的量化格式

| 格式        | 说明                           |
| :---------- | :----------------------------- |
| INT8        | 权重和激活 8-bit 整数量化      |
| INT4        | 权重 4-bit 量化（weight-only） |
| INT3/INT2   | 超低精度量化（实验性）         |
| FP8         | 浮点 8-bit 量化                |
| FP 浮点量化 | 支持 FP 格式用于长尾分布场景   |
| 混合精度    | W4A8, W4A16, W6A6, W3A16 等    |

#### 4.2.4 量化粒度支持

| 粒度        | 权重 | 激活 | 说明                                 |
| :---------- | :--- | :--- | :----------------------------------- |
| Per-Tensor  | ✅   | ✅   | 对称/非对称量化                      |
| Per-Channel | ✅   | —    | 沿输出通道独立 scale                 |
| Per-Token   | —    | ✅   | 激活按 token 维度量化                |
| Per-Group   | ✅   | —    | 支持 group_size 配置（如 g128, g64） |

#### 4.2.5 模型支持

- 大语言模型 (LLM): LLaMA 全系列、Qwen 系列、DeepSeek-V3/R1/R1-Zero、ChatGLM、Mixtral MoE
- 视觉语言模型 (VLM): LLaVA-1.5 等，针对 VLM 支持 20+ 种算法（含 token 裁剪与量化）
- 视频生成模型: Wan2.1 系列（INT8/FP8 量化导出）
- MoE 模型: Mixtral、DeepSeek 等 MoE 架构

#### 4.2.6 关键特性

- 超低成本量化: 单块 40GB A100 即可校准和评估 OPT-175B（约 350GB 权重）
- 多后端兼容: 支持导出到 vLLM、SGLang、TRT-LLM、MLC-LLM、AutoAWQ、lightx2v 等
- 全面的基准评测: 覆盖 PPL、MMLU、ARC-e、BoolQ、HellaSwag、PIQA 等多种评测指标
- 高可扩展性: 模块化设计，可灵活扩展新算法和模型
- 学术系统性: 论文提供了校准数据、算法、数据格式三维度的系统性基准分析

### 4.3 AngelSlim

项目地址: https://github.com/tencent/AngelSlim

#### 4.3.1 项目概述

AngelSlim 是腾讯推出的大语言模型压缩工具包，定位为"增强易用性、全面性和高效性的模型压缩工具"。该工具紧密围绕腾讯混元系列模型生态，同时支持主流开源模型。提供 YAML 一键执行、多 GPU 并行、低显存模式等企业级功能。

#### 4.3.2 支持的量化方法

| 量化方法                       | 类型        | 说明                                      |
| :----------------------------- | :---------- | :---------------------------------------- |
| FP8 (静态/动态)                | 标准量化    | 支持静态和动态两种模式                    |
| INT8 动态                      | 标准量化    | 动态 INT8 量化                            |
| GPTQ                           | 重构类      | INT4 权重量化                             |
| AWQ                            | 变换类      | INT4 激活感知权重量化                     |
| GAPTQ                          | 重构类      | GPTQ 的改进版本                           |
| NVFP4                          | NVIDIA 专用 | Blackwell GPU 优化的 FP4 格式             |
| LeptoQuant                     | 自研        | 腾讯自研量化算法                          |
| Tequila                        | 自研        | 三值化量化算法                            |
| DAQ (Delta-Aware Quantization) | 自研        | 2025年3月发布，保留后训练参数更新中的知识 |
| Sherry                         | 自研        | 硬件高效的 1.25-bit 量化算法              |

#### 4.3.3 量化粒度支持

| 粒度        | 权重 | 激活 | 说明                            |
| :---------- | :--- | :--- | :------------------------------ |
| Per-Tensor  | ✅   | ✅   | 扩散模型动态量化                |
| Per-Channel | ✅   | —    | 沿通道维度独立 scale            |
| Per-Group   | ✅   | —    | INT4-AWQ 支持 group_size=128/64 |
| Per-Token   | —    | ✅   | 扩散模型动态量化                |
| Per-Block   | ✅   | —    | FP8 块级量化，可配置 block_size |

#### 4.3.4 模型支持

AngelSlim 的模型支持非常广泛，特别是对腾讯混元系列模型有原生支持：

文本生成 (LLM):

- 腾讯 Hunyuan-Dense、Hunyuan-MoE
- Qwen3、Qwen2.5
- DeepSeek-V3/R1
- GLM-4.6

视觉语言 (VLM):

- Hunyuan-VL、HunyuanOCR
- Qwen3-VL、Qwen2.5-VL

文生图/视频/3D:

- Hunyuan-Image、Hunyuan-Video、Hunyuan-3D
- Qwen-Image
- FLUX、SDXL

语音 (TTS/ASR):

- Qwen3-Omni、Qwen2-Audio
- Fun-CosyVoice3

#### 4.3.5 关键特性

- 一键执行: YAML 配置文件驱动的全流程自动化
- 多 GPU 并行: 支持多卡并行量化处理
- 低显存模式: GPU 资源受限时的内存优化策略
- 端到端工作流: 从压缩到部署的完整流程
- vLLM 集成: 量化后模型可直接部署到 vLLM
- Eagle3 推测解码: 支持全规模模型的投机解码加速
- SpecExit: 推理早退算法
- HY-1.8B-2Bit: 端侧语言模型部署方案

### 4.4 DeepCompressor

项目地址: https://github.com/nunchaku-ai/deepcompressor

#### 4.4.1 项目概述

DeepCompressor 是 MIT HAN Lab（韩松团队）开发的模型压缩工具箱，专注于大语言模型和扩散模型的量化压缩。其核心创新是 SVDQuant 算法——通过 SVD 低秩分解吸收异常值，实现 4-bit 权重和激活的同时量化（W4A4），该工作被 ICLR 2025 接收为 Spotlight 论文。搭配 Nunchaku 推理引擎，在扩散模型推理场景中展现了卓越的性能。

#### 4.4.2 支持的量化方法

**LLM 量化方法**:

| 方法            | 格式    | 说明                                              |
| :-------------- | :------ | :------------------------------------------------ |
| **AWQ**         | W4A16   | 激活感知权重量化                                  |
| **GPTQ**        | W4A16   | 基于 Hessian 的权重重构                           |
| **SmoothQuant** | W8A8    | 激活平滑变换                                      |
| **QoQ**         | W4A8KV4 | 权重-激活-KV Cache 联合量化（QServe，MLSys 2025） |

**扩散模型量化方法**:

| 方法         | 格式 | 说明                                             |
| :----------- | :--- | :----------------------------------------------- |
| **SVDQuant** | W4A4 | 通过 SVD 低秩分支吸收异常值，ICLR 2025 Spotlight |

#### 4.4.3 量化数据类型

| 数据类型   | 位宽  | 说明     |
| :--------- | :---- | :------- |
| INT8       | 8 bit | 整数量化 |
| INT4       | 4 bit | 整数量化 |
| FP4 (E2M1) | 4 bit | 浮点量化 |

#### 4.4.4 模型支持

扩散模型（核心优势）:

- SDXL
- PixArt-Σ
- FLUX.1（12B 模型实现 3.6× 内存降低，3.5× 加速）

大语言模型:

- Llama-2-7B、Llama-3-8B
- Qwen1.5-72B
- Mixtral-8x7b

#### 4.4.5 关键特性

- SVDQuant 创新: 首个扩散模型 W4A4 量化方案，不同于传统 smoothing 方法，通过 SVD 低秩分支吸收异常值
- Nunchaku 推理引擎: 专为 SVDQuant 设计的推理系统，融合低秩和低精度分支的 kernel，消除冗余显存访问
- LoRA 无缝支持: 量化后的模型支持 LoRA 适配器加载，无需重新量化
- QServe 系统: 针对 LLM 的 W4A8KV4 推理系统，已集成到 NVIDIA TensorRT-LLM（2024年12月）

#### 4.4.6 性能亮点

| 模型         | 指标              | 数值                         |
| :----------- | :---------------- | :--------------------------- |
| FLUX.1 (12B) | 内存降低          | 3.6×                         |
| FLUX.1 (12B) | 对比 W4-only 加速 | 3.5× (RTX-4090)              |
| FLUX.1 (12B) | 图像质量          | 与 FP16 基线视觉上无明显差异 |

### 4.5 TorchAO Quantization

项目地址: https://github.com/pytorch/ao

#### 4.5.1 项目概述

TorchAO（PyTorch Architecture Optimization）是 PyTorch 官方推出的架构优化库，提供原生的量化、稀疏化和其他优化功能。作为 PyTorch 生态的一部分，TorchAO 深度集成了 torch.compile，可在不修改用户代码的情况下自动应用量化和 kernel 优化。截至 2026年4月，最新版本为 v0.17.0，拥有 200+ 贡献者。

#### 4.5.2 支持的量化方法

| 量化类型         | 具体方法                         | 说明                                 |
| :--------------- | :------------------------------- | :----------------------------------- |
| Weight-Only INT4 | Int4WeightOnlyQuantizer          | INT4 权重量化，支持多种 packing 格式 |
| Weight-Only INT8 | Int8WeightOnlyQuantizer          | INT8 权重量化                        |
| Dynamic W8A8     | Int8DynActInt4WeightQuantizer    | 动态激活量化 + INT4 权重             |
| Static W8A8      | 静态量化 API                     | 需要校准数据确定 scale               |
| FP8 (Float8)     | Float8Linear, Float8Tensor       | E4M3 浮点量化                        |
| FPx              | 灵活浮点格式                     | float3~float8 自定义浮点             |
| UINTx            | 无符号整数                       | uint1~uint7 灵活位宽                 |
| QAT              | intx_quantization_aware_training | 量化感知训练                         |

#### 4.5.3 量化粒度支持

| 粒度                       | 权重 | 激活 | 说明               |
| :------------------------- | :--- | :--- | :----------------- |
| Per-Tensor (PerTensor)     | ✅   | ✅   | 全张量共享 scale   |
| Per-Channel/Axis (PerAxis) | ✅   | —    | 沿指定轴独立 scale |
| Per-Group (PerGroup)       | ✅   | —    | 分组量化           |
| Per-Row (PerRow)           | ✅   | ✅   | FP8 常用模式       |

#### 4.5.4 模型支持

TorchAO 通过 Hugging Face Diffusers 和 Transformers 集成，支持广泛的模型类型：

- LLM: Llama 全系列、Gemma 系列（支持 QAT 精度恢复）
- 扩散模型: 通过 Diffusers 集成支持 FLUX.1-dev、CogVideoX 等
- 通用支持: 任何包含 torch.nn.Linear 层的 PyTorch 模型

#### 4.5.5 关键特性

- PyTorch 原生: 深度集成 torch.compile，无需额外推理引擎
- 灵活的位宽: 从 uint1 到 float8，支持几乎任意精度
- QAT 支持: 可通过量化感知训练恢复精度（如 Gemma3-4B 恢复 67% 量化精度退化）
- FP8 训练: 支持 float8 训练，Llama-3.1-70B 预训练速度提升 1.5×
- 生态完整: 与 Transformers、Diffusers、vLLM 等上层框架无缝集成
- Autoquant: 自动选择最优量化策略

#### 4.5.6 性能亮点

| 场景                   | 效果                     |
| :--------------------- | :----------------------- |
| Llama-3-8B INT4 推理   | 1.89× 加速，58% 内存减少 |
| Llama-3.1-70B FP8 训练 | 1.5× 训练加速            |
| Gemma3-4B QAT          | 恢复 67% 量化精度退化    |

### 4.6 TensorRT Model Optimizer

项目地址: https://github.com/NVIDIA/Model-Optimizer  
文档: https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html

#### 4.6.1 项目概述

TensorRT Model Optimizer（简称 ModelOpt）是 NVIDIA 官方推出的模型优化工具，专为 NVIDIA GPU 和TensorRT/TensorRT-LLM 推理引擎优化。支持 PyTorch 和 ONNX 两种框架的量化，提供 PTQ 和 QAT 两种量化模式，并针对不同的 NVIDIA GPU 架构提供最优的量化策略建议。

#### 4.6.2 支持的量化方法

**PTQ 方法**:

| 方法                 | 格式       | 压缩率 | 精度损失 |
| :------------------- | :--------- | :----- | :------- |
| **FP8**              | W8A8       | 50%    | 非常低   |
| **INT8 SmoothQuant** | W8A8       | 50%    | 低       |
| **INT4 AWQ**         | W4A16      | 25%    | 低       |
| **INT4-FP8 AWQ**     | W4A8       | 25%    | 低       |
| **NVFP4**            | W4A4/W4A16 | \~18%  | 中       |
| **AutoQuantize**     | 混合       | 自适应 | 自动优化 |

**QAT 方法**:

| 方法                     | 说明                                                  |
| :----------------------- | :---------------------------------------------------- |
| **QAT**                  | 通过模拟量化的反向传播进行微调，在 PTQ 精度不足时使用 |
| **NeMo Megatron Bridge** | 与 NeMo 框架集成的 QAT 工作流                         |

#### 4.6.3 量化粒度支持

| 粒度            | 权重 | 激活 | 应用场景              |
| :-------------- | :--- | :--- | :-------------------- |
| Per-Tensor      | ✅   | ✅   | FP8 默认模式          |
| Per-Channel     | ✅   | —    | INT8 SmoothQuant 权重 |
| Per-Group/Block | ✅   | ✅   | INT4 AWQ, NVFP4       |
| Block-wise      | ✅   | —    | NVFP4 块级缩放        |

#### 4.6.4 模型支持

LLM 模型:

- LLaMA 全系列（含 LLaMA 4）
- Qwen-2/2.5/3
- Mixtral
- DeepSeek-R1
- Nemotron-3-Super（提供 FP8/NVFP4 预量化 checkpoint）

扩散模型:

- Stable Diffusion 2.1, SDXL 1.0, SDXL Turbo
- Stable Diffusion 3 / 3.5 (Large & Medium)
- FLUX.1-dev（FP8 量化 2.4× 加速）

ONNX 模型:

- 支持 Linux 和 Windows 上的 ONNX PTQ

#### 4.6.5 关键特性

- NVIDIA 硬件原生优化: 针对 Turing/Ampere/Hopper/Blackwell 架构提供最优量化方案
- PTQ + QAT 双模式: 提供完整的量化精度恢复路径
- AutoQuantize: 自动搜索最优量化策略
- TensorRT-LLM 原生集成: 量化模型直接部署到 TensorRT-LLM
- NeMo 集成: 与 NeMo Megatron Bridge 集成支持大规模模型的 QAT
- ONNX 支持: 跨平台 ONNX 量化能力
- 推测解码: 支持投机解码加速推理

#### 4.6.6 选型指南

NVIDIA 官方提供的量化方法选择建议：

- 小 batch 推理（batch ≤ 4）: 优先 INT4 AWQ (weight-only)
- 大 batch 推理（batch ≥ 16）: 优先 FP8，如不满足则 INT4-FP8 AWQ
- 精度敏感场景: FP8 PTQ → QAT 渐进式精度恢复

### 4.7 Optimum-Quanto

项目地址: https://github.com/huggingface/optimum-quanto

#### 4.7.1 项目概述

Optimum-Quanto 是 Hugging Face 推出的 PyTorch 量化后端，专为 Optimum 生态设计。以通用性和简洁性为核心设计目标，支持 eager mode（适用于不可追踪的模型），可在 CUDA 和 MPS 等多种设备上运行。  
⚠️ 重要提示: 该项目目前处于维护模式，仅接受小 bug 修复和文档改进。官方建议生产环境使用 bitsandbytes 或 TorchAO。

#### 4.7.2 支持的量化方法

| 数据类型 | 权重 | 激活 | 说明         |
| :------- | :--- | :--- | :----------- |
| INT2     | ✅   | —    | 超低精度     |
| INT4     | ✅   | —    | 低精度       |
| INT8     | ✅   | ✅   | 经典整数量化 |
| Float8   | ✅   | ✅   | 浮点量化     |

支持的量化流程:

1. 量化 (Quantize): 动态量化权重
2. 校准 (Calibrate): 通过 Calibration 上下文管理器记录激活范围
3. 微调 (QAT): 量化感知训练恢复精度
4. 冻结 (Freeze): 将浮点权重替换为量化整数权重
5. 序列化 (Serialize): 保存为 pickle 或 safetensors 格式

#### 4.7.3 量化粒度支持

| 粒度               | 权重           | 激活      | 说明                                      |
| :----------------- | :------------- | :-------- | :---------------------------------------- |
| Per-Tensor         | ✅ (INT8/FP8)  | ✅ (固定) | 对称量化                                  |
| Per-Channel        | ✅ (INT8/FP8)  | —         | 沿第一维（输出特征）量化                  |
| Per-Group (Affine) | ✅ (低于 8bit) | —         | 仿射量化（含 zero-point），用于 INT4/INT2 |

#### 4.7.4 模型支持

- LLM: Meta-Llama-3.1-8B 等 AutoModelForCausalLM 模型
- 扩散模型: 通过 Diffusers 集成支持 PixArt、Stable Diffusion 等子模块量化
- 通用支持: QLinear, QConv2D, LayerNorm 模块

**支持的量化模块**:

| 模块             | 权重量化 | 激活量化      | 偏置   |
| :--------------- | :------- | :------------ | :----- |
| Linear (QLinear) | ✅       | 输入/输出可选 | 不量化 |
| Conv2d (QConv2D) | ✅       | 输入/输出可选 | 不量化 |
| LayerNorm        | ❌       | 输出可选      | 不量化 |

#### 4.7.5 关键特性

- Eager Mode: 支持不可追踪的模型，通用性强
- 多设备支持: CUDA 和 MPS
- 加速矩阵乘法: CUDA 上 int8-int8, fp16-int4, bf16-int8, bf16-int4 优化 kernel
- Safetensors 兼容: 支持 HF 标准序列化格式
- Diffusers 集成: 可量化 diffusers pipeline 中的子模型并重新组装

#### 4.7.6 局限性

- 维护模式: 不再接受新功能，建议迁移至 TorchAO 或 bitsandbytes
- 未实现功能: 动态激活平滑、全设备优化 kernel、torch.compile 兼容

## 5\. 量化方法对比分析

### 5.1 量化算法覆盖度对比

| 量化算法     | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :----------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| RTN          |       ✅       |      ✅       |    ✅     |       —        |   ✅    |         —         |       ✅       |
| GPTQ         |       ✅       |      ✅       |    ✅     |       ✅       |    —    |         —         |       —        |
| AWQ          |       ✅       |      ✅       |    ✅     |       ✅       |    —    |        ✅         |       —        |
| SmoothQuant  |       ✅       |      ✅       |     —     |       ✅       |    —    |        ✅         |       —        |
| AutoRound    |       ✅       |       —       |     —     |       —        |    —    |         —         |       —        |
| OmniQuant    |       —        |      ✅       |     —     |       —        |    —    |         —         |       —        |
| QuaRot       |       —        |      ✅       |     —     |       —        |    —    |         —         |       —        |
| SpinQuant    |   ✅ (实验)    |       —       |     —     |       —        |    —    |         —         |       —        |
| QuIP         |   ✅ (实验)    |       —       |     —     |       —        |    —    |         —         |       —        |
| SVDQuant     |       —        |       —       |     —     |       ✅       |    —    |         —         |       —        |
| QoQ          |       —        |       —       |     —     |       ✅       |    —    |         —         |       —        |
| DAQ          |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |
| Tequila      |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |
| Sherry       |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |
| LeptoQuant   |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |
| AutoQuantize |       —        |       —       |     —     |       —        |   ✅    |        ✅         |       —        |
| QAT          |       —        |       —       |     —     |       —        |   ✅    |        ✅         |       ✅       |

### 5.2 量化格式覆盖度对比

| 量化格式   | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :--------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| W8A8 INT8  |       ✅       |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| W8A8 FP8   |       ✅       |      ✅       |    ✅     |       —        |   ✅    |        ✅         |       ✅       |
| W4A16      |       ✅       |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| W8A16      |       ✅       |      ✅       |     —     |       —        |   ✅    |         —         |       ✅       |
| W4A8       |       ✅       |      ✅       |     —     |       ✅       |   ✅    |        ✅         |       —        |
| W4A4       |       —        |      ✅       |     —     |       ✅       |    —    |         —         |       —        |
| NVFP4      |       ✅       |       —       |    ✅     |       —        |    —    |        ✅         |       —        |
| MXFP8      |   ✅ (实验)    |       —       |     —     |       —        |    —    |         —         |       —        |
| MXFP4      |   ✅ (实验)    |       —       |     —     |       —        |    —    |         —         |       —        |
| FP8 Block  |       ✅       |       —       |    ✅     |       —        |    —    |         —         |       —        |
| INT2       |       —        |   ✅ (实验)   |    ✅     |       —        |   ✅    |         —         |       ✅       |
| FPx 自定义 |       —        |       —       |     —     |       ✅       |   ✅    |         —         |       —        |

## 6\. 量化粒度对比分析

| 量化粒度      | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :------------ | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| Per-Tensor    |       ✅       |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| Per-Channel   |       ✅       |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| Per-Group     |       ✅       |      ✅       |    ✅     |       —        |   ✅    |        ✅         |       ✅       |
| Per-Token     |       ✅       |      ✅       |    ✅     |       —        |    —    |         —         |       —        |
| Per-Row       |       —        |       —       |     —     |       —        |   ✅    |         —         |       —        |
| Per-Block     |       ✅       |       —       |    ✅     |       —        |    —    |        ✅         |       —        |
| Per-Head (KV) |       ✅       |       —       |     —     |    ✅ (QoQ)    |    —    |         —         |       —        |

粒度细节说明：

- llm-compressor: 粒度体系最完整，通过 compressed_tensors 的 QuantizationStrategy 定义 TENSOR/CHANNEL/GROUP/TOKEN/BLOCK 策略，并支持 KV Cache 的 per-head 量化
- LightCompress: 主要覆盖标准粒度（per-tensor/channel/group/token），对低精度场景（W3/W2）支持 per-group 分组量化
- AngelSlim: 独特支持 FP8 Block-wise 量化和扩散模型的 per-tensor/per-token 动态量化
- TorchAO: 提供 PerRow 粒度，FP8 场景常用，是较独特的选项
- TensorRT ModelOpt: 针对 NVFP4 提供块级缩放（global scale + local scale per 16 elements）

## 7\. 模型支持范围对比

### 7.1 模型类型覆盖度

| 模型类型     | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :----------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| 文本 LLM     |     ✅✅✅     |    ✅✅✅     |  ✅✅✅   |      ✅✅      |  ✅✅   |      ✅✅✅       |      ✅✅      |
| 视觉语言 VLM |     ✅✅✅     |     ✅✅      |  ✅✅✅   |       —        |   ✅    |        ✅         |       ✅       |
| 扩散模型     |       ❌       |   ✅ (视频)   |  ✅✅✅   |     ✅✅✅     |  ✅✅   |       ✅✅        |       ✅       |
| 音频模型     |       ✅       |       —       |   ✅✅    |       —        |    —    |         —         |       —        |
| MoE 模型     |     ✅✅✅     |     ✅✅      |   ✅✅    |       ✅       |    —    |        ✅         |       —        |
| 3D 生成      |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |

注：✅✅✅ = 强力支持（官方示例、专用优化）；✅✅ = 良好支持；✅ = 基础支持；— = 不支持

### 7.2 具体 LLM 模型支持

| 模型           | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :------------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| Llama 系列     |       ✅       |      ✅       |     —     |       ✅       |   ✅    |        ✅         |       ✅       |
| Qwen 系列      |       ✅       |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| DeepSeek V3/R1 |       ✅       |      ✅       |    ✅     |       —        |    —    |        ✅         |       —        |
| Mixtral MoE    |       ✅       |      ✅       |     —     |       ✅       |    —    |        ✅         |       —        |
| 混元系列       |       —        |       —       |    ✅     |       —        |    —    |         —         |       —        |
| GLM 系列       |       ✅       |      ✅       |    ✅     |       —        |    —    |         —         |       —        |
| Gemma 系列     |       ✅       |       —       |     —     |       —        |   ✅    |         —         |       —        |
| Phi 系列       |       ✅       |       —       |     —     |       —        |    —    |         —         |       —        |

### 7.3 扩散模型支持

| 模型             | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :--------------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| Stable Diffusion |       ❌       |       —       |    ✅     |       ✅       |   ✅    |        ✅         |       —        |
| SDXL             |       ❌       |       —       |    ✅     |       ✅       |    —    |        ✅         |       —        |
| SD 3/3.5         |       ❌       |       —       |     —     |       —        |    —    |        ✅         |       —        |
| FLUX             |       ❌       |       —       |    ✅     |       ✅       |   ✅    |        ✅         |       —        |
| PixArt           |       ❌       |       —       |     —     |       ✅       |    —    |         —         |       ✅       |
| Hunyuan 系列     |       ❌       |       —       |    ✅     |       —        |    —    |         —         |       —        |
| Wan2.1 视频      |       ❌       |      ✅       |     —     |       —        |    —    |         —         |       —        |
| CogVideoX        |       ❌       |       —       |     —     |       —        |   ✅    |         —         |       —        |

## 8\. 部署生态对比

### 8.1 推理引擎兼容性

| 推理引擎        | llm-compressor | LightCompress | AngelSlim | DeepCompressor |  TorchAO  | TensorRT ModelOpt | Optimum-Quanto |
| :-------------- | :------------: | :-----------: | :-------: | :------------: | :-------: | :---------------: | :------------: |
| vLLM            |   ✅ (原生)    |      ✅       |    ✅     |       —        |    ✅     |        ✅         |       —        |
| TensorRT-LLM    |       —        |      ✅       |     —     |  ✅ (QServe)   |     —     |     ✅ (原生)     |       —        |
| SGLang          |   ✅ (有限)    |      ✅       |     —     |       —        |     —     |         —         |       —        |
| MLC-LLM         |       —        |      ✅       |     —     |       —        |     —     |         —         |       —        |
| torch.compile   |       —        |       —       |     —     |       —        | ✅ (原生) |         —         |   — (计划中)   |
| Nunchaku        |       —        |       —       |     —     |   ✅ (原生)    |     —     |         —         |       —        |
| llama.cpp       |       —        |      ✅       |     —     |       —        |     —     |         —         |       —        |
| HF Transformers |       ✅       |      ✅       |    ✅     |       —        |    ✅     |        ✅         |       ✅       |
| HF Diffusers    |       —        |       —       |    ✅     |       —        |    ✅     |        ✅         |       ✅       |

### 8.2 序列化格式

| 格式               | llm-compressor | LightCompress | AngelSlim | DeepCompressor | TorchAO | TensorRT ModelOpt | Optimum-Quanto |
| :----------------- | :------------: | :-----------: | :-------: | :------------: | :-----: | :---------------: | :------------: |
| safetensors        |       ✅       |      ✅       |    ✅     |       —        |   ✅    |        ✅         |       ✅       |
| compressed-tensors |   ✅ (原生)    |       —       |     —     |       —        |    —    |         —         |       —        |
| GGUF               |       —        |      ✅       |     —     |       —        |    —    |         —         |       —        |
| PyTorch pickle     |       —        |      ✅       |    ✅     |       ✅       |   ✅    |        ✅         |       ✅       |
| ONNX               |       —        |       —       |     —     |       —        |    —    |        ✅         |       —        |

## 9\. AIGC 场景适用性分析

### 9.1 场景一：LLM 在线推理服务（文本生成/对话）

核心需求: 低延迟、高吞吐、高精度、支持大规模并发

| 工具              | 适用度     | 推荐理由                                             |
| :---------------- | :--------- | :--------------------------------------------------- |
| llm-compressor    | ⭐⭐⭐⭐⭐ | vLLM 原生集成，量化方法最全面，文档和社区成熟度最高  |
| TensorRT ModelOpt | ⭐⭐⭐⭐⭐ | NVIDIA 原生优化，TRT-LLM 性能最优，提供 PTQ+QAT 路径 |
| LightCompress     | ⭐⭐⭐⭐   | 算法种类丰富，多后端兼容，学术系统性强               |
| AngelSlim         | ⭐⭐⭐⭐   | 一键执行，企业级易用性强，自研算法有差异化优势       |
| TorchAO           | ⭐⭐⭐     | PyTorch 原生、灵活，但缺乏高级 PTQ 算法              |
| DeepCompressor    | ⭐⭐       | LLM 支持有限，主要方法针对扩散模型                   |
| Optimum-Quanto    | ⭐⭐       | 维护模式，不建议新项目采用                           |

### 9.2 场景二：文生图/视频推理服务（Stable Diffusion/FLUX/视频生成）

核心需求: 显存优化、推理加速、图像/视频质量保持、LoRA 兼容

| 工具              | 适用度     | 推荐理由                                                             |
| :---------------- | :--------- | :------------------------------------------------------------------- |
| DeepCompressor    | ⭐⭐⭐⭐⭐ | SVDQuant W4A4 专为扩散模型设计，Nunchaku 引擎性能卓越，LoRA 无缝支持 |
| AngelSlim         | ⭐⭐⭐⭐⭐ | 支持 Hunyuan 全系列图/视频/3D 模型，扩散模型覆盖最广                 |
| TensorRT ModelOpt | ⭐⭐⭐⭐   | SD/SDXL/FLUX FP8 优化成熟，与 TensorRT 深度集成                      |
| TorchAO           | ⭐⭐⭐     | 通过 Diffusers 集成支持，灵活但缺乏专用优化                          |
| LightCompress     | ⭐⭐⭐     | 支持 Wan2.1 视频生成模型 INT8/FP8 导出                               |
| Optimum-Quanto    | ⭐⭐       | 可用于 Diffusers 子模型量化，但功能有限                              |
| llm-compressor    | ⭐         | 不支持扩散模型                                                       |

### 9.3 场景三：多模态理解与生成（VLM）

核心需求: 支持图文/音视频多模态输入，多种模态处理模块的协同量化

| 工具              | 适用度     | 推荐理由                                                   |
| :---------------- | :--------- | :--------------------------------------------------------- |
| llm-compressor    | ⭐⭐⭐⭐⭐ | VLM 支持最全面（Qwen-VL、LLaVA、InternVL、Gemma 多模态等） |
| AngelSlim         | ⭐⭐⭐⭐⭐ | 混元 VL 原生支持，Qwen VL 全系列覆盖                       |
| LightCompress     | ⭐⭐⭐⭐   | VLM 专项支持 20+ 算法，含 token 裁剪和量化                 |
| TensorRT ModelOpt | ⭐⭐⭐     | 基础 VLM 支持                                              |
| TorchAO           | ⭐⭐       | 通用框架，需自行适配                                       |
| DeepCompressor    | ⭐         | VLM 支持有限                                               |
| Optimum-Quanto    | ⭐         | 基础 HF 模型支持                                           |

### 9.4 场景四：端侧/边缘部署

核心需求: 极致压缩比、低功耗、跨平台兼容

| 工具              | 适用度     | 推荐理由                                               |
| :---------------- | :--------- | :----------------------------------------------------- |
| AngelSlim         | ⭐⭐⭐⭐⭐ | HY-1.8B-2Bit 端侧模型、Sherry 1.25-bit 硬件高效量化    |
| TorchAO           | ⭐⭐⭐⭐   | PyTorch 原生，torch.compile 优化，灵活位宽 uint1~uint7 |
| LightCompress     | ⭐⭐⭐     | 超低精度（W2/W3）和 MLC-LLM 导出支持                   |
| TensorRT ModelOpt | ⭐⭐⭐     | NVIDIA Jetson 等嵌入式 GPU 支持                        |
| llm-compressor    | ⭐⭐       | 主要面向服务器 GPU 场景                                |
| DeepCompressor    | ⭐⭐       | RTX 消费级 GPU 有优化                                  |
| Optimum-Quanto    | ⭐         | MPS 支持但整体能力有限                                 |

## 10\. 综合评估与结论

### 10.1 核心结论

#### 结论一：没有"万能"工具，选型需结合具体场景

每款工具都有其核心优势领域：

- LLM 推理: llm-compressor（vLLM 生态）和 TensorRT ModelOpt（TRT-LLM 生态）是两个最成熟的选择
- 扩散模型: DeepCompressor（SVDQuant）和 AngelSlim 在图像/视频生成场景有独特优势
- 全模态 AIGC: AngelSlim 模型覆盖最广泛，是唯一同时强力支持 LLM、VLM、扩散模型、音频和 3D 的工具

#### 结论二：量化格式趋势明确

- FP8 成为主流: 几乎所有工具都支持 FP8 量化，精度损失极低，Hopper+ GPU 硬件原生支持
  - W4A16 是权重压缩标准: GPTQ/AWQ 方法成熟，vLLM 支持完善
  - NVFP4/MXFP4 是新兴方向: Blackwell 架构专用，仅 llm-compressor、AngelSlim、TensorRT ModelOpt 支持
  - W4A4 在扩散模型中有突破: DeepCompressor 的 SVDQuant 展示了 W4A4 在扩散模型上的可行性

#### 结论三：部署生态是关键考量

量化工具与推理引擎的集成深度直接影响最终的部署效率：

- vLLM 生态: llm-compressor 是最佳选择，LightCompress 和 AngelSlim 也有良好支持
- TRT-LLM 生态: TensorRT ModelOpt 是唯一的一等公民选择
- PyTorch 原生: TorchAO + torch.compile 提供最灵活的方案
- 多后端需求: LightCompress 后端兼容性最强（6+ 推理引擎）

#### 结论四：自研算法带来差异化竞争力

- AngelSlim 的 DAQ/Tequila/Sherry: 针对后训练场景和极致压缩的自研方案
  - DeepCompressor 的 SVDQuant/QoQ: 学术创新驱动，扩散模型量化的技术领先者
  - LightCompress 的系统性 Benchmark: 论文级的公平对比框架，为算法选择提供科学依据

#### 结论五：Optimum-Quanto 已进入维护模式

Hugging Face 官方明确建议迁移到 bitsandbytes 或 TorchAO。新项目不建议采用 Optimum-Quanto。

## 11\. 选型建议

### 11.1 按场景推荐

| AIGC 场景              | 首选工具          | 备选工具                     | 说明                 |
| :--------------------- | :---------------- | :--------------------------- | :------------------- |
| LLM 在线服务 (vLLM)    | llm-compressor    | LightCompress                | 原生集成，方法全面   |
| LLM 在线服务 (TRT-LLM) | TensorRT ModelOpt | —                            | NVIDIA 原生优化      |
| 文生图 (SD/FLUX)       | DeepCompressor    | AngelSlim, TensorRT ModelOpt | SVDQuant 性能领先    |
| 文生视频               | AngelSlim         | LightCompress                | 混元视频/Wan2.1 支持 |
| 多模态 VLM             | llm-compressor    | AngelSlim                    | VLM 支持最全面       |
| 音频/语音              | AngelSlim         | llm-compressor               | 混元+Qwen 音频支持   |
| 端侧部署               | AngelSlim         | TorchAO                      | 极致压缩 + 灵活位宽  |
| 研究/实验              | LightCompress     | TorchAO                      | 学术基准 + 灵活接口  |

### 11.2 按技术栈推荐

| 技术栈                         | 推荐工具                 |
| :----------------------------- | :----------------------- |
| vLLM + HuggingFace             | llm-compressor           |
| TensorRT-LLM + NVIDIA          | TensorRT Model Optimizer |
| PyTorch + torch.compile        | TorchAO                  |
| 多后端（vLLM + TRT-LLM + MLC） | LightCompress            |
| 腾讯混元生态                   | AngelSlim                |
| 扩散模型 + LoRA                | DeepCompressor           |

### 11.3 综合 AIGC 场景建议

对于需要覆盖 AIGC 全场景（文本生成、图像生成、视频生成、多模态理解）的团队，建议采用 组合方案：

1. LLM/VLM 量化: 使用 llm-compressor（若部署在 vLLM）或 TensorRT Model Optimizer（若部署在 TRT-LLM），两者都有成熟的 FP8/INT4 方案
2. 扩散模型量化: 使用 DeepCompressor（若追求极致压缩）或 AngelSlim（若追求全面的模型覆盖）
3. 统一方案: 若希望用单一工具覆盖所有场景，AngelSlim 是当前模型类型覆盖面最广的选择

## 12\. 参考资料

### 12.1 项目仓库

1. [llm-compressor](https://github.com/vllm-project/llm-compressor)
2. [LightCompress](https://github.com/ModelTC/LightCompress)
3. [AngelSlim](https://github.com/tencent/AngelSlim)
4. [DeepCompressor](https://github.com/nunchaku-ai/deepcompressor)
5. [TorchAO](https://github.com/pytorch/ao)
6. [TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
7. [Optimum-Quanto](https://github.com/huggingface/optimum-quanto)

### 12.2 学术论文

1. [LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit (EMNLP 2024)](https://arxiv.org/abs/2405.06001)
2. [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models (ICLR 2025 Spotlight)](https://arxiv.org/abs/2411.05007)
3. [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving (MLSys 2025)](https://arxiv.org/abs/2405.04532)
4. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
5. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
6. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

### 12.3 官方文档

1. [TensorRT Model Optimizer 量化指南](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)
2. [TorchAO 量化 API 文档](https://docs.pytorch.org/ao/main/)
3. [Optimum-Quanto HuggingFace 文档](https://huggingface.co/docs/transformers/quantization/quanto)
4. [AngelSlim 文档](https://angelslim.readthedocs.io/)
5. [llm-compressor 文档](https://github.com/vllm-project/llm-compressor/tree/main/docs)
