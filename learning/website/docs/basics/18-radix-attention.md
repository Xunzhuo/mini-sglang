# 前缀复用的极致：Radix Attention 与 SGLang

## 核心目标
理解 SGLang 的核心创新点 Radix Attention，以及它如何加速多轮对话和复杂 Prompt 场景。

## 1. 场景分析
- **多轮对话**: 每一轮都包含之前的历史记录。
- **Few-shot Learning**: 多个请求共享相同的示例前缀。
- **思维链 (CoT)**: 多个推理路径共享相同的起始步骤。

## 2. 传统方法的局限
- 每次请求都视为独立，重复计算相同的前缀 KV Cache。

## 3. Radix Attention 原理
- **Radix Tree (基数树)**: 将 Token 序列组织成树状结构。
- **自动复用**: 
    - 请求进来时，在树中进行最长前缀匹配。
    - 命中节点直接复用 KV Cache，无需重新计算。
- **LRU 驱逐**: 显存不足时，优先删除最久未访问的叶子节点。

## 4. 实际收益
- 显著降低 TTFT (Time to First Token)。
- 提高吞吐量。
