---
sidebar_position: 4
---

# 位置编码：让模型理解顺序

Transformer 架构使用自注意力机制并行处理所有位置，但这也带来一个问题：模型无法区分 "猫追狗" 和 "狗追猫"。位置编码 (Positional Encoding) 正是为了解决这个问题。

## 1. 为什么需要位置编码？

### 1.1 RNN 的隐式位置信息

RNN 按顺序处理输入，位置信息自然地蕴含在计算顺序中：

```mermaid
graph LR
    X1[x₁] --> H1[h₁]
    H1 --> H2[h₂]
    X2[x₂] --> H2
    H2 --> H3[h₃]
    X3[x₃] --> H3
    
    H1 -.- |"h₂知道自己在x₁之后"| H2
```

### 1.2 Transformer 的并行困境

Transformer 同时处理所有位置：

```mermaid
graph TB
    subgraph 位置无关
        X1[我] --> A[Self-Attention]
        X2[爱] --> A
        X3[你] --> A
    end
    
    subgraph 问题
        P["'我爱你' 和 '你爱我'<br/>计算结果相同!"]
    end
```

对于 Transformer 来说，**"我爱你" 和 "你爱我" 的 Self-Attention 计算结果完全相同**！这显然不合理。

**解决方案**：显式地将位置信息注入模型。

## 2. 位置编码的发展历程

```mermaid
timeline
    title 位置编码演进
    section 绝对位置
        2017 : Sinusoidal PE
             : Transformer原论文
        2018 : Learned PE
             : BERT/GPT
    section 相对位置
        2019 : Relative Position
             : Transformer-XL
        2021 : ALiBi
             : 线性偏置
        2021 : RoPE
             : 旋转位置编码
    section 长度外推
        2023 : Position Interpolation
             : NTK-Aware
        2024 : LongRoPE/YaRN
             : 200万token上下文
```

## 3. 绝对位置编码

### 3.1 正弦余弦编码 (Sinusoidal PE)

原始 Transformer 论文提出的方案，使用不同频率的正弦和余弦函数生成位置编码。

```mermaid
graph LR
    subgraph 正弦余弦编码
        P[位置 pos] --> S["偶数维度用正弦"]
        P --> C["奇数维度用余弦"]
        S --> E[位置编码向量]
        C --> E
    end
```

**核心思想**：
- 对于每个位置，生成一个与模型维度相同大小的向量
- 偶数维度使用正弦函数，奇数维度使用余弦函数
- 不同维度使用不同频率（从低频到高频）
- 这样设计的好处是：相对位置可以通过线性变换表示

**使用方式**：将位置编码向量直接加到输入的词嵌入上。

**特点**：
- 每个位置有唯一的编码
- 理论上可以外推到更长序列
- 无需学习额外参数
- 实际外推能力有限

### 3.2 可学习位置编码 (Learned PE)

另一种简单直接的方案：创建一个位置嵌入表，让模型自己学习每个位置的表示。

**工作原理**：
- 预设最大序列长度（如 512 或 2048）
- 为每个位置学习一个嵌入向量
- 将位置嵌入加到词嵌入上

**使用模型**：BERT、GPT-1/2

**局限性**：
- 无法处理超过预设最大长度的序列
- 需要额外的参数量

## 4. 相对位置编码

绝对位置编码的问题：

```mermaid
graph TB
    subgraph 绝对位置编码问题
        A["位置5和位置6的关系"]
        B["位置100和位置101的关系"]
        A -.- |"编码完全不同!"| B
    end
```

直觉上，位置 5 和 6 的相邻关系应该与位置 100 和 101 的相邻关系相似，但绝对位置编码无法表达这一点。相对位置编码关注的是 token 之间的**相对距离**。

### 4.1 Relative Position Bias

在注意力分数中加入相对位置偏置：

```mermaid
graph LR
    QK["Query和Key的内积"] --> ADD["+"]
    B["位置偏置矩阵"] --> ADD
    ADD --> SM[Softmax]
```

**工作原理**：在计算注意力分数时，根据 Query 和 Key 的相对位置距离，加上一个偏置值。这个偏置值可以是学习得到的，也可以是预定义的。

**使用模型**：T5、DeBERTa

### 4.2 ALiBi (Attention with Linear Biases)

更简单的方案：直接在注意力分数上减去与距离成正比的惩罚。

```mermaid
graph TB
    subgraph ALiBi
        A["注意力分数"] --> S["减去 m × 距离"]
        S --> R["距离越远，惩罚越大"]
    end
```

**核心思想**：
- 每个注意力头有一个固定的斜率 m（超参数，不需要学习）
- 对于相对距离为 d 的两个位置，惩罚值为 m 乘以 d
- 距离越远，惩罚越大，注意力权重越低

**优点**：
- 无需额外参数
- 外推能力强（训练 1K，推理可达 100K+）
- 实现极其简单

**使用模型**：BLOOM、MPT、Falcon

## 5. 旋转位置编码 (RoPE)

### 5.1 核心思想

RoPE (Rotary Position Embedding) 是目前**最流行**的位置编码方案，被 LLaMA、Qwen、Mistral、DeepSeek 等主流模型采用。

**关键洞察**：将位置信息编码为向量的**旋转**。

```mermaid
graph LR
    subgraph RoPE核心
        Q["Query 向量"] --> R1["旋转 m 次"]
        R1 --> QM["位置 m 的 Query"]
        
        K["Key 向量"] --> R2["旋转 n 次"]
        R2 --> KN["位置 n 的 Key"]
        
        QM --> DOT["计算内积"]
        KN --> DOT
        DOT --> RE["结果只依赖相对位置 m-n"]
    end
```

**工作原理**：
- 将向量的每两个相邻维度视为一个二维平面上的点
- 根据位置索引，对这个点进行旋转
- 位置 m 旋转 m 个角度，位置 n 旋转 n 个角度
- 神奇的是，两个旋转后向量的内积只取决于相对位置 (m-n)

### 5.2 旋转矩阵

将向量的相邻两个维度视为一个 2D 平面，应用旋转变换：

```mermaid
graph TB
    subgraph 2D旋转
        V["原始向量 2个维度"] --> R["应用旋转矩阵"]
        R --> V2["旋转后的向量"]
    end
```

对于 d 维向量，进行 d/2 次独立的旋转，每次使用不同的旋转角度。旋转角度由位置索引和维度索引共同决定：低维度旋转快（高频），高维度旋转慢（低频）。

### 5.3 RoPE 的优势

```mermaid
graph TB
    subgraph RoPE优势
        R1[相对位置感知] --> R["RoPE"]
        R2[高效实现] --> R
        R3[无额外参数] --> R
        R4[可扩展性好] --> R
    end
```

| 特性 | RoPE | 说明 |
|------|------|------|
| **相对位置** | 支持 | 内积自然编码相对位置 |
| **外推能力** | 较好 | 可通过技巧增强 |
| **实现效率** | 高 | 只需逐元素操作 |
| **额外参数** | 无 | 完全由公式定义 |
| **长度外推** | 支持 | 可结合 PI/NTK/YaRN |

## 6. 长度外推技术

训练长度有限（如 4K），如何在推理时处理更长序列（如 32K 甚至 2M）？

```mermaid
graph LR
    T[训练 4K tokens] --> P[推理更长序列?]
    P --> |方法1| PI[位置插值]
    P --> |方法2| NTK[NTK缩放]
    P --> |方法3| YaRN[YaRN混合]
    P --> |方法4| LR[LongRoPE]
```

### 6.1 位置插值 (Position Interpolation)

**核心思想**：将超出范围的位置索引线性缩放到训练范围内。

假设模型训练时最大长度是 4096，现在想处理 8192 长度的序列。位置插值的做法是：将原本的位置 0 到 8191 线性映射到 0 到 4095.5。这样所有位置都落在模型"见过"的范围内。

**Meta 论文**：*Extending Context Window of Large Language Models via Positional Interpolation*

### 6.2 NTK-Aware Scaling

**核心思想**：调整 RoPE 的**基频**而非位置。

与直接缩放位置不同，NTK 方法通过调整旋转的频率基数来实现扩展。直觉上，相当于让旋转变得更"缓慢"，这样即使位置变大，旋转角度也不会超出训练时的范围。

### 6.3 YaRN (Yet another RoPE extensioN)

结合多种技术的混合方案：

```mermaid
graph TB
    subgraph YaRN策略
        L[低频分量] --> |不修改| O[输出]
        M[中频分量] --> |平滑过渡| O
        H[高频分量] --> |NTK缩放| O
    end
```

**核心思想**：
- **低频分量**：负责长距离依赖，不修改（外推）
- **高频分量**：负责局部信息，应用 NTK 缩放（插值）
- **中频分量**：平滑过渡

**效果**：训练 4K，推理可达 128K+

### 6.4 LongRoPE (2024)

微软研究院提出的最新技术，将上下文窗口扩展到**200 万 tokens**。

```mermaid
graph TB
    subgraph LongRoPE核心创新
        A[非均匀插值] --> L[LongRoPE]
        B[渐进式扩展] --> L
        C[短上下文恢复] --> L
    end
```

**关键创新**：
- **非均匀插值**：使用进化算法搜索每个维度的最优缩放因子，而不是统一缩放
- **渐进式扩展**：先微调到 256K，再进一步扩展到 2M，只需约 1000 步微调
- **短上下文恢复**：通过调整缩放因子，保持短序列（如 8K）的性能

**效果**：在 LLaMA2-7B 和 Mistral-7B 上达到 SOTA，同时保持原始短序列性能。

## 7. 位置编码对比总结

```mermaid
graph TB
    PE[位置编码] --> ABS[绝对位置]
    PE --> REL[相对位置]
    
    ABS --> SIN[Sinusoidal]
    ABS --> LEA[Learned]
    
    REL --> ALI[ALiBi]
    REL --> ROP[RoPE]
    
    SIN --> |Transformer| M1[原始论文]
    LEA --> |BERT/GPT-2| M2[早期模型]
    ALI --> |BLOOM/MPT| M3[长序列模型]
    ROP --> |LLaMA/Qwen/Mistral| M4[主流LLM]
```

| 方法 | 代表模型 | 外推能力 | 额外参数 | 实现复杂度 |
|------|----------|----------|----------|------------|
| **Sinusoidal** | Transformer | 中 | 无 | 低 |
| **Learned** | BERT, GPT-2 | 差 | 有 | 低 |
| **ALiBi** | BLOOM, MPT, Falcon | 强 | 无 | 低 |
| **RoPE** | LLaMA, Qwen, Mistral | 中 | 无 | 中 |
| **RoPE + YaRN** | 微调模型 | 强 | 无 | 中 |
| **LongRoPE** | 最新研究 | 极强 | 无 | 高 |

**2024 年主流选择**：
- **短序列 (< 8K)**：RoPE 即可
- **长序列 (8K-128K)**：RoPE + NTK/YaRN
- **超长序列 (128K-2M)**：LongRoPE 或优化的 RoPE 变体

## 8. 本章小结

```mermaid
mindmap
  root((位置编码))
    为什么需要
      Transformer并行处理
      无法区分顺序
    绝对位置
      Sinusoidal
      Learned
    相对位置
      ALiBi
      RoPE-主流
    长度外推
      Position Interpolation
      NTK
      YaRN
      LongRoPE
```

**核心要点**：
- 位置编码让 Transformer 理解序列顺序
- 从绝对位置到相对位置是重要演进
- RoPE 是当前最流行的方案（LLaMA、Qwen、Mistral）
- 长度外推技术（YaRN、LongRoPE 等）突破训练长度限制
- 2024 年已经实现了 200 万 token 的上下文窗口

## 延伸阅读

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)

---

*下一篇：[Transformer 架构演进](./05-transformer-architecture.md)*
