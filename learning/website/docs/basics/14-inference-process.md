# LLM 推理揭秘：Prefill 与 Decode 的二重奏

## 核心目标
理解 LLM 推理的两个截然不同的阶段，以及它们对计算资源的不同需求。

## 1. 自回归生成 (Autoregressive Generation)
- 什么是自回归？（预测下一个 token，并将结果作为输入预测下下个）。
- 伪代码演示。

## 2. 第一阶段：Prefill (预填充)
- **任务**: 处理输入的 Prompt，生成第一个 token。
- **特点**: 
    - 并行计算（所有 token 同时进 Attention）。
    - **Compute Bound** (计算密集型)。
    - 延迟敏感 (Time to First Token, TTFT)。

## 3. 第二阶段：Decode (解码)
- **任务**: 逐个生成后续 token。
- **特点**:
    - 串行计算（依赖前一个 token）。
    - **Memory Bound** (访存密集型) - 每次都要加载全部权重。
    - 吞吐量敏感。

## 4. 性能瓶颈分析
- 为什么 Batch Size 对 Decode 阶段如此重要？
- 算术强度 (Arithmetic Intensity) 分析。
