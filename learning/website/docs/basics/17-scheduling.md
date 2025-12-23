# 吞吐量的飞跃：Continuous Batching 原理

## 核心目标
理解 Continuous Batching 如何打破静态 Batching 的限制，最大化 GPU 利用率。

## 1. 静态 Batching 的缺陷
- **短板效应**: 一个 Batch 的处理时间取决于最长的那个请求。
- **Padding 浪费**: 短请求必须 Padding 到长请求的长度，进行无效计算。

## 2. Continuous Batching (Orca)
- **核心思想**: 迭代级调度 (Iteration-level scheduling)。
- **机制**:
    - 一个请求生成完 token 后立即移出 Batch。
    - 新的请求可以随时加入 Batch。
- **状态管理**: Running, Waiting, Finished。

## 3. 结合 PagedAttention
- PagedAttention 使得 Continuous Batching 更加高效（无需担心显存碎片）。

## 4. 调度策略
- FCFS (First Come First Serve)
- 优先级调度
