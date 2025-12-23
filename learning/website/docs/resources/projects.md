---
sidebar_position: 4
title: 开源项目
description: LLM 领域核心开源项目推荐
---

# 开源项目

## 推理引擎（课程核心）

**生产级推理引擎：**

- [**vLLM**](https://github.com/vllm-project/vllm) - 最流行的高性能推理引擎，PagedAttention 的实现，必读
- [**SGLang**](https://github.com/sgl-project/sglang) - mini-sglang 的完整版，RadixAttention + 结构化生成
- [**TGI (Text Generation Inference)**](https://github.com/huggingface/text-generation-inference) - HuggingFace 官方推理引擎，Rust 实现
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) - C++ 实现，CPU 友好，量化支持好
- [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 官方，极致性能优化
- [**LMDeploy**](https://github.com/InternLM/lmdeploy) - 上海 AI Lab 出品，支持多种推理后端

**教学项目：**

- [**mini-sglang**](https://github.com/sgl-project/mini-sglang) - 从零学习 LLM系统，代码简洁易懂

---

## 训练框架

**大规模训练：**

- [**DeepSpeed**](https://github.com/microsoft/DeepSpeed) - 微软出品，ZeRO 优化，分布式训练必备
- [**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM) - NVIDIA 官方，张量并行、流水线并行
- [**Colossal-AI**](https://github.com/hpcaitech/ColossalAI) - 潞晶科技出品，易用的大规模训练框架

**微调框架：**

- [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory) - 中文友好，支持多种微调方法，WebUI 易用
- [**Axolotl**](https://github.com/OpenAccess-AI-Collective/axolotl) - 配置驱动的微调框架，支持多种技术
- [**Unsloth**](https://github.com/unslothai/unsloth) - 超快的微调框架，内存优化极致

---

## RLHF 框架

**强化学习对齐：**

- [**trl (Transformer Reinforcement Learning)**](https://github.com/huggingface/trl) - HuggingFace 官方 RLHF 库，支持 PPO、DPO
- [**DeepSpeed-Chat**](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) - 微软出品，完整的 RLHF 训练流程
- [**OpenRLHF**](https://github.com/OpenLLMAI/OpenRLHF) - 开源 RLHF 框架，支持多种算法
- [**RLAIF**](https://github.com/anthropics/hh-rlhf) - Anthropic 的 AI 反馈强化学习

---

## 应用框架

**RAG & Agent：**

- [**LangChain**](https://github.com/langchain-ai/langchain) - 最流行的 LLM 应用框架，生态丰富
- [**LlamaIndex**](https://github.com/run-llama/llama_index) - RAG 专用框架，数据连接器丰富
- [**Dify**](https://github.com/langgenius/dify) - 可视化 LLM 应用开发平台，国产优秀项目
- [**AutoGPT**](https://github.com/Significant-Gravitas/AutoGPT) - 自主 Agent 框架，探索 AGI 边界

---

## 开源模型

**基础模型：**

- [**LLaMA**](https://github.com/meta-llama/llama) - Meta 开源，最流行的基础模型系列
- [**Qwen**](https://github.com/QwenLM/Qwen) - 阿里通义千问，中文能力强，生态完善
- [**ChatGLM**](https://github.com/THUDM/ChatGLM-6B) - 清华 KEG 实验室，中文对话模型
- [**Mistral**](https://github.com/mistralai/mistral-src) - Mistral AI 出品，性能优异的开源模型

---

## 工具库

**实用工具：**

- [**Transformers**](https://github.com/huggingface/transformers) - HuggingFace 核心库，模型加载必备
- [**Ollama**](https://github.com/ollama/ollama) - 本地运行 LLM 的最简单方式，类似 Docker
- [**LM Studio**](https://lmstudio.ai/) - 图形化界面运行本地模型

---

## 学习路径建议

### 推理引擎学习路径：
1. **mini-sglang** - 理解核心概念（教学项目）
2. **vLLM** - 学习生产级实现（PagedAttention）
3. **SGLang** - 了解高级特性（RadixAttention）
4. **llama.cpp** - 理解 CPU 推理和量化

### 训练到部署完整链路：
1. **LLaMA-Factory** - 微调模型
2. **vLLM** - 部署推理
3. **LangChain** - 构建应用

### 本地实验推荐：
1. **Ollama** - 快速启动本地模型
2. **llama.cpp** - CPU 运行小模型
3. **LLaMA-Factory** - 微调实验

