# 推理加速的基石：KV Cache 详解

## 核心目标
理解 KV Cache 的原理、作用以及它带来的显存挑战。

## 1. 为什么需要 KV Cache？
- **重复计算问题**: 在 Decode 阶段，如果不存 KV，每次都要重新计算 Prompt 和之前生成 token 的 KV 值。
- **解决方案**: 缓存每一层的 K 和 V 矩阵。

## 2. KV Cache 的工作原理
- 图解：随着序列长度增加，KV Cache 如何增长。
- 显存占用估算公式：
    - $Size = 2 \times Batch \times Layers \times Heads \times HeadDim \times SeqLen \times DtypeSize$

## 3. KV Cache 带来的挑战
- **显存爆炸**: 长文本场景下，KV Cache 可能比模型权重还大。
- **显存碎片**: 动态增长的特性导致难以预分配连续内存。

## 4. 优化方向预览
- PagedAttention (下一章)
- Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)
