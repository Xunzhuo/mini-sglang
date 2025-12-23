---
sidebar_position: 4
---

# 位置编码：让模型理解顺序

Transformer 架构使用自注意力机制并行处理所有位置，但这也带来一个问题：模型无法区分 "猫追狗" 和 "狗追猫"。位置编码 (Positional Encoding) 正是为了解决这个问题。

## 为什么需要位置编码？

### RNN 的隐式位置信息

RNN 按顺序处理输入，位置信息自然地蕴含在计算顺序中：

```
h₁ = f(x₁)
h₂ = f(x₂, h₁)  # h₂ "知道"自己在 x₁ 之后
h₃ = f(x₃, h₂)  # h₃ "知道"自己在 x₂ 之后
```

### Transformer 的并行困境

Transformer 同时处理所有位置：

```
Attention(Q, K, V) 不依赖位置顺序
```

对于 Transformer 来说，"我爱你" 和 "你爱我" 的 Self-Attention 计算结果完全相同！

## 绝对位置编码

### 正弦余弦编码 (Sinusoidal PE)

原始 Transformer 论文提出的方案：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

其中：
- `pos`：位置索引 (0, 1, 2, ...)
- `i`：维度索引
- `d`：模型维度

**特点**：
- 每个位置有唯一的编码
- 相对位置可以通过线性变换表示
- 理论上可以外推到更长序列

```python
import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

### 可学习位置编码 (Learned PE)

让模型自己学习位置表示：

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)
```

**使用模型**：BERT、GPT-1/2

**局限性**：
- 无法处理超过训练长度的序列
- 需要额外参数

## 相对位置编码

绝对位置编码的问题：位置 5 和位置 6 的关系，与位置 100 和位置 101 的关系，编码完全不同。

相对位置编码关注的是 token 之间的**相对距离**。

### Relative Position Bias

在注意力分数中加入相对位置偏置：

```
Attention(Q, K) = softmax(QK^T / √d + B)
```

其中 `B[i,j]` 取决于 `i-j`（相对距离）。

**使用模型**：T5、DeBERTa

### ALiBi (Attention with Linear Biases)

更简单的方案：直接在注意力分数上减去与距离成正比的惩罚：

```
Attention(Q, K) = softmax(QK^T / √d - m · |i-j|)
```

其中 `m` 是每个头不同的斜率。

**优点**：
- 无需额外参数
- 外推能力强
- 实现简单

**使用模型**：BLOOM、MPT

## 旋转位置编码 (RoPE)

### 核心思想

RoPE (Rotary Position Embedding) 是目前最流行的位置编码方案，被 LLaMA、Qwen、Mistral 等主流模型采用。

**关键洞察**：将位置信息编码为向量的旋转。

```
q_m = R(m) · q  # 位置 m 的 query
k_n = R(n) · k  # 位置 n 的 key

q_m · k_n = q · R(m-n) · k  # 内积只依赖相对位置 m-n
```

### 旋转矩阵

将向量的相邻两个维度视为一个 2D 平面，应用旋转：

```
[cos(mθ)  -sin(mθ)] [q₀]
[sin(mθ)   cos(mθ)] [q₁]
```

对于 d 维向量，进行 d/2 次旋转，每次使用不同的 θ：

```python
def apply_rope(x, cos, sin):
    # x: (batch, seq_len, heads, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # 旋转
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)
    
    return x_rotated
```

### RoPE 的优势

| 特性 | RoPE |
|------|------|
| 相对位置 | ✅ 内积自然编码相对位置 |
| 外推能力 | ✅ 较好（可通过技巧增强） |
| 实现效率 | ✅ 高效（只需逐元素操作） |
| 额外参数 | ✅ 无 |

## 长度外推技术

训练长度有限，如何在推理时处理更长序列？

### 位置插值 (Position Interpolation)

将位置索引线性缩放到训练范围内：

```
原始: pos = 0, 1, 2, ..., 8191 (训练 4096)
插值: pos' = pos * (4096/8192) = 0, 0.5, 1, ..., 4095.5
```

**Meta 论文**：Extending Context Window of Large Language Models via Positional Interpolation

### NTK-Aware Scaling

调整 RoPE 的基频而非位置：

```
原始: θ = 10000^(-2i/d)
NTK:  θ' = (10000 * α)^(-2i/d)
```

### YaRN

结合多种技术的混合方案：
- 低频分量：不修改
- 高频分量：应用 NTK 缩放
- 中频分量：平滑过渡

## 位置编码对比

| 方法 | 代表模型 | 外推能力 | 额外参数 |
|------|----------|----------|----------|
| Sinusoidal | Transformer | 中 | 无 |
| Learned | BERT, GPT-2 | 差 | 有 |
| ALiBi | BLOOM, MPT | 强 | 无 |
| RoPE | LLaMA, Qwen | 中→强 | 无 |
| RoPE + YaRN | 各种微调模型 | 强 | 无 |

## 本章小结

- 位置编码让 Transformer 理解序列顺序
- 从绝对位置到相对位置是重要演进
- RoPE 是当前最流行的方案
- 长度外推技术突破训练长度限制

## 延伸阅读

- RoFormer: Enhanced Transformer with Rotary Position Embedding
- ALiBi: Train Short, Test Long
- YaRN: Efficient Context Window Extension

---

*下一篇：[Transformer 架构演进](./05-transformer-architecture.md)*
