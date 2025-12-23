---
sidebar_position: 3
title: 经典论文
description: LLM 领域必读的核心论文
---

# 经典论文

## 必读基础（奠定认知框架）

**Transformer 架构：**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)  
  Transformer 开山之作，理解 Self-Attention 机制的必读论文

**预训练语言模型：**

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (2018)  
  双向预训练范式，理解 Masked Language Modeling

- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)  
  自回归语言模型的规模化探索

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020)  
  In-context Learning 的涌现，规模化的力量

---

## 推理系统优化（课程核心）

**注意力机制优化：**

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (2022)  
  IO-aware 的注意力优化，理解 GPU 内存层次结构

- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (2023)  
  进一步优化并行度和工作分区

**KV Cache 管理：**

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, 2023)  
  虚拟内存思想应用于 KV cache，解决内存碎片问题

**推理加速：**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023)  
  用小模型加速大模型推理，理解推测解码

- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) (2024)  
  多头并行解码，进一步加速

**模型压缩：**

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022)  
  理解量化技术，降低推理成本

- [AWQ: Activation-aware Weight Quantization for LLM Compression](https://arxiv.org/abs/2306.00978) (2023)  
  激活感知的权重量化

**长文本处理：**

- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (2023)  
  位置插值扩展上下文长度

---

## 训练与对齐

**规模化规律：**

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020)  
  理解模型规模、数据量、计算量之间的关系

**指令微调：**

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (FLAN, 2021)  
  指令微调的开山之作，证明指令微调能提升零样本泛化能力

**RLHF 与对齐：**

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (2017)  
  RLHF 的早期探索，从人类偏好学习奖励函数

- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) (2020)  
  OpenAI 在摘要任务上应用 RLHF 的实践

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)  
  RLHF 的标准范式：SFT → 奖励模型 → PPO 优化

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (DPO, 2023)  
  无需训练奖励模型和 RL，直接从偏好数据优化，更简单高效

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (2022)  
  用 AI 反馈替代人类反馈，可扩展的对齐方法

- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) (2023)  
  用 AI 标注替代人类标注，降低 RLHF 成本

**高效微调：**

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)  
  低秩适配，高效微调大模型

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023)  
  量化 + LoRA，在消费级 GPU 上微调大模型

---

## 应用范式

**推理能力：**

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (2022)  
  思维链提示，激发模型推理能力

**知识增强：**

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020)  
  RAG 开山之作，检索增强生成

**多模态：**

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (LLaVA, 2023)  
  视觉指令微调，理解多模态对齐

---

## 系统设计（可选进阶）

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022)  
  分布式推理系统设计

- [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665) (2023)  
  模型并行与统计复用

