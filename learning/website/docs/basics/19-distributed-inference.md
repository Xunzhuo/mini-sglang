# 打破单卡限制：张量并行 (Tensor Parallelism) 基础

## 核心目标
理解当模型大到单卡放不下时，如何通过 Tensor Parallelism 进行分布式推理。

## 1. 为什么需要分布式推理？
- 显存限制 (70B 模型 FP16 需要 140GB+ 显存)。
- 计算限制 (单卡算力不足以满足低延迟要求)。

## 2. 并行策略概览
- Pipeline Parallelism (流水线并行)
- Tensor Parallelism (张量并行) - 推理中最常用。

## 3. Tensor Parallelism 详解
- **Column Parallel**: 权重矩阵按列切分。
- **Row Parallel**: 权重矩阵按行切分。
- **MLP 层切分示例**: A 矩阵列切 -> B 矩阵行切 -> 结果相加。
- **Attention 层切分示例**: 多头注意力天然适合切分。

## 4. 通信原语
- **All-Reduce**: 聚合所有卡的结果并分发。
- 通信开销对推理性能的影响。
