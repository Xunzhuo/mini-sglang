# Transformer 架构演进：为什么是 Decoder-only？

## 核心目标
理解现代 LLM 为何普遍采用 Decoder-only 架构，以及 Self-Attention 的计算本质。

## 1. 架构演进史
- **Encoder-Decoder (T5, BART)**: 适用于翻译、摘要。
- **Encoder-only (BERT)**: 适用于理解、分类。
- **Decoder-only (GPT系列, LLaMA)**: 适用于生成。
- **分析**: 为什么生成任务选择了 Decoder-only？（因果掩码、训练效率、涌现能力）。

## 2. Self-Attention 机制详解
- **Q, K, V 的物理含义**: 查询、键、值。
- **计算公式**: `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`
- **Multi-Head Attention**: 多头注意力的作用。

## 3. 位置编码 (Positional Encoding)
- 绝对位置编码 (Sinusoidal, Learned)
- 相对位置编码 (RoPE - Rotary Positional Embeddings) **[重点]**
    - 为什么 RoPE 在长文本中表现更好？

## 4. 思考题
- 如果让你设计一个推理引擎，Attention 层计算中最耗时的部分是什么？
