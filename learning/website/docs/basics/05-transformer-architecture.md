---
sidebar_position: 5
---

# Transformer 架构演进：从 Encoder-Decoder 到 Decoder-only

2017 年，Google 发表了划时代的论文 "Attention Is All You Need"，提出了 Transformer 架构。此后，这一架构衍生出三大变体，而 Decoder-only 架构最终成为大语言模型的主流选择。本文将深入分析这一演进过程。

## 1. Transformer 原始架构

原始 Transformer 采用 Encoder-Decoder 结构，设计用于机器翻译任务。

```mermaid
graph TB
    subgraph 输入处理
        I[源语言输入] --> IE[输入嵌入]
        IE --> PE1[位置编码]
    end
    
    subgraph Encoder编码器
        PE1 --> E1[Self-Attention]
        E1 --> EN1[Add & Norm]
        EN1 --> EF[Feed Forward]
        EF --> EN2[Add & Norm]
        EN2 --> |×N层| EOUT[编码器输出]
    end
    
    subgraph 输出处理
        O[目标语言] --> OE[输出嵌入]
        OE --> PE2[位置编码]
    end
    
    subgraph Decoder解码器
        PE2 --> D1[Masked Self-Attention]
        D1 --> DN1[Add & Norm]
        DN1 --> D2[Cross-Attention]
        EOUT --> D2
        D2 --> DN2[Add & Norm]
        DN2 --> DF[Feed Forward]
        DF --> DN3[Add & Norm]
        DN3 --> |×N层| DOUT[解码器输出]
    end
    
    DOUT --> Linear[线性层]
    Linear --> Softmax[Softmax]
    Softmax --> OUT[输出概率]
```

### 1.1 核心组件详解

**编码器 (Encoder)**
- 处理输入序列，生成上下文表示
- 使用**双向自注意力**，每个位置可以看到所有其他位置
- 堆叠 N 层（原论文 N=6）

**解码器 (Decoder)**
- 自回归生成输出序列
- 使用**因果自注意力**（Masked Self-Attention），只能看到前面的位置
- **交叉注意力**（Cross-Attention）连接编码器输出
- 同样堆叠 N 层

```mermaid
graph LR
    subgraph Encoder["双向注意力 Encoder"]
        A1["我"] <--> A2["爱"] <--> A3["你"]
    end

    subgraph Decoder["因果注意力 Decoder"]
        B1["I"] --> B2["love"]
        B2 --> B3["you"]
    end
```

## 2. 三大架构变体

根据组件的取舍，Transformer 演化出三种主要架构：

```mermaid
graph TB
    T[原始 Transformer<br>Encoder-Decoder] --> E[Encoder-only<br>BERT 系列]
    T --> D[Decoder-only<br>GPT 系列]
    T --> ED[Encoder-Decoder<br>T5/BART]
    
    E --> E1[双向理解<br>分类/NER/问答]
    D --> D1[自回归生成<br>对话/写作/代码]
    ED --> ED1[序列转换<br>翻译/摘要]
```

### 2.1 Encoder-only：BERT 家族

**代表模型**：BERT, RoBERTa, ALBERT, DistilBERT

**核心特点**：
- 只保留 Encoder 部分
- 双向注意力，每个 token 可以看到所有其他 token
- 擅长**理解任务**

**预训练目标**：
```mermaid
graph LR
    subgraph MLM["MLM掩码语言建模"]
        M1["今天"] --> M2["[MASK]"] --> M3["真好"]
        M2 --> P["预测: 天气"]
    end
```

- **MLM（Masked Language Modeling）**：完形填空，随机遮住 15% 的词让模型预测
- **NSP（Next Sentence Prediction）**：句子关系预测

**适用场景**：文本分类、命名实体识别、句子相似度、抽取式问答

### 2.2 Decoder-only：GPT 家族

**代表模型**：GPT-1/2/3/4, LLaMA, Qwen, Mistral, Claude, DeepSeek

**核心特点**：
- 只保留 Decoder 部分（去掉 Cross-Attention）
- 因果注意力，每个 token 只能看到它之前的 token
- 擅长**生成任务**

**预训练目标**：
```mermaid
graph LR
    subgraph CLM["CLM因果语言建模"]
        C1["今"] --> C2["天"]
        C2 --> C3["天"]
        C3 --> C4["气"]
        C4 --> C5["真"]
        C5 --> C6["好"]
    end
```

- **CLM（Causal Language Modeling）**：预测下一个 token，每个位置都产生 loss

**适用场景**：文本生成、对话系统、代码生成，以及翻译、摘要等所有任务！

### 2.3 Encoder-Decoder：T5/BART 家族

**代表模型**：T5, BART, mT5, FLAN-T5

**核心特点**：
- 保留完整的 Encoder-Decoder 结构
- 编码器处理输入，解码器生成输出
- 输入输出可以有不同长度

```mermaid
graph LR
    subgraph Encoder
        E[translate English to German: Hello world]
    end
    
    subgraph Decoder
        D[Hallo Welt]
    end
    
    E --> |Cross-Attention| D
```

**适用场景**：机器翻译、文本摘要、生成式问答

## 3. 架构对比分析

| 特性 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|--------------|--------------|-----------------|
| **代表模型** | BERT | GPT, LLaMA | T5, BART |
| **注意力模式** | 双向 | 因果（单向） | 双向 + 因果 |
| **预训练目标** | MLM (15%) | CLM (100%) | Span/Denoising |
| **擅长任务** | 理解 | 生成 | Seq2Seq |
| **训练效率** | 较低 | **高** | 中 |
| **扩展性** | 中 | **极强** | 中 |
| **涌现能力** | 弱 | **强** | 中 |

## 4. 为什么 Decoder-only 成为主流？

现代 LLM（GPT-4、Claude、LLaMA、Qwen、DeepSeek 等）几乎都采用 Decoder-only 架构。这并非偶然：

### 4.1 训练效率更高

```mermaid
graph TB
    subgraph DecoderOnly["Decoder-only 100%利用率"]
        D1["今天天气真好"]
        D2["Loss = L(今→天) + L(天→天) + L(天→气) + L(气→真) + L(真→好)"]
        D1 --> D2
        D2 --> D3["每个token都贡献梯度!"]
    end

    subgraph EncoderOnly["Encoder-only 仅15%"]
        E1["今天[MASK]真好"]
        E2["Loss = L([MASK]→天气)"]
        E1 --> E2
        E2 --> E3["只有15%的token贡献梯度"]
    end
```

### 4.2 统一的生成范式

Decoder-only 可以统一处理各种任务：

```python
# 分类任务
prompt = "这部电影太棒了！情感是："
output = "正面"

# 翻译任务
prompt = "翻译成英文：今天天气真好"
output = "The weather is nice today"

# 代码生成
prompt = "写一个Python函数计算斐波那契数列："
output = "def fibonacci(n):..."

# 问答任务
prompt = "问题：地球到月球的距离是多少？答案："
output = "约38万公里"
```

**一个模型，所有任务**——这就是 In-Context Learning 的魔力。

### 4.3 涌现能力 (Emergent Abilities)

```mermaid
graph LR
    S[模型规模] --> |增大| E[涌现能力]
    
    E --> E1[10B+<br/>Few-shot Learning]
    E --> E2[100B+<br/>Chain-of-Thought]
    E --> E3[复杂推理<br/>代码生成]
```

当模型规模足够大时，展现出惊人的涌现能力，这些能力在 Encoder-only 模型中很难观察到。

### 4.4 扩展性更好 (Scaling Law)

Decoder-only 架构的性能随规模增长更加平滑可预测：

| 规模 | 参数量 | 层数 | 注意力 | 位置编码 | FFN |
|------|--------|------|--------|----------|-----|
| **LLaMA-2-7B** | 7B | 32 | GQA | RoPE | SwiGLU |
| **LLaMA-2-70B** | 70B | 80 | GQA | RoPE | SwiGLU |
| **Mistral-7B** | 7B | 32 | GQA | RoPE | SwiGLU |
| **Qwen-7B** | 7B | 32 | MHA | RoPE | SwiGLU |
| **DeepSeek-V2** | 236B(21B) | - | MLA | RoPE | MoE |

## 5. 因果掩码 (Causal Mask)

Decoder-only 架构的核心是**因果掩码**，确保自回归特性：

```mermaid
graph TB
    subgraph Matrix["因果掩码矩阵"]
        M["今 天 气 好<br/>今 1 0 0 0<br/>天 1 1 0 0<br/>气 1 1 1 0<br/>好 1 1 1 1"]
    end
    M --> R["1 = 可以attend<br/>0 = 不可以attend<br/>下三角矩阵"]
```

```python
def causal_attention_mask(seq_len):
    """生成因果注意力掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 下三角矩阵

# 应用掩码
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
attention_weights = softmax(attention_scores)  # -inf → 0
```

## 6. Self-Attention 核心计算

### 6.1 单头注意力

```mermaid
graph LR
    X[输入 X] --> Q[Query = X·Wq]
    X --> K[Key = X·Wk]
    X --> V[Value = X·Wv]
    
    Q --> A[Attention Score<br/>Q·K^T / √d]
    K --> A
    A --> S[Softmax]
    S --> O[Output = Attn·V]
    V --> O
```

```python
def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_model)
    """
    d_k = K.shape[-1]
    
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 6.2 多头注意力 (Multi-Head Attention)

多头注意力让模型从**不同的表示子空间**学习关系：

```mermaid
graph TB
    X[输入] --> H1[Head 1<br/>语法关系]
    X --> H2[Head 2<br/>语义关系]
    X --> H3[Head 3<br/>位置关系]
    X --> H4[Head 4<br/>...]
    
    H1 --> C[Concat]
    H2 --> C
    H3 --> C
    H4 --> C
    
    C --> W[Linear Projection]
    W --> O[输出]
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 线性投影
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分头: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        
        # 加权求和并合并多头
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(output)
```

## 7. 完整的 Decoder Block

现代 LLM 的基本构建单元：

```mermaid
graph TB
    X[输入] --> N1[RMSNorm]
    N1 --> A[Self-Attention]
    A --> ADD1[Add]
    X --> ADD1
    
    ADD1 --> N2[RMSNorm]
    N2 --> F[FFN/SwiGLU]
    F --> ADD2[Add]
    ADD1 --> ADD2
    
    ADD2 --> OUT[输出]
```

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = RMSNorm(d_model)  # 现代LLM使用RMSNorm
        self.ffn = SwiGLU(d_model, d_ff)  # 现代LLM使用SwiGLU
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm 结构 (现代 LLM 标配)
        # Self-Attention
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-Forward Network
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x
```

### Pre-Norm vs Post-Norm

```mermaid
graph LR
    subgraph Post-Norm原始
        A1[x] --> A2[Attention] --> A3[x + Attn] --> A4[LayerNorm]
    end
    
    subgraph Pre-Norm现代
        B1[x] --> B2[Norm] --> B3[Attention] --> B4[x + Attn]
    end
```

**Pre-Norm 优势**：训练更稳定，尤其对于深层网络（100+ 层）。

## 8. 现代 LLM 架构改进

相比原始 Transformer，2024 年的 LLM 有众多改进：

```mermaid
graph TB
    subgraph 原始Transformer
        O1[LayerNorm]
        O2[Sinusoidal位置编码]
        O3[ReLU激活]
        O4[MHA多头注意力]
        O5[2层FFN]
    end
    
    subgraph 现代LLM
        N1[RMSNorm]
        N2[RoPE旋转位置编码]
        N3[SwiGLU激活]
        N4[GQA分组查询注意力]
        N5[GLU变体FFN]
    end
    
    O1 --> |更高效| N1
    O2 --> |更好外推| N2
    O3 --> |更好性能| N3
    O4 --> |更省显存| N4
    O5 --> |更强表达| N5
```

### 8.1 RMSNorm (Root Mean Square Normalization)

比 LayerNorm 更高效，去掉了均值中心化：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 只计算均方根，不减均值
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**优势**：减少约 10% 计算量，效果相当。

### 8.2 RoPE (Rotary Position Embedding)

旋转位置编码，相比绝对位置编码有更好的外推性：

```mermaid
graph LR
    subgraph Sinusoidal
        S1[位置信息加到输入]
        S2[绝对位置]
    end
    
    subgraph RoPE
        R1[位置信息融入QK计算]
        R2[相对位置]
        R3[更好的长度外推]
    end
```

**核心思想**：通过旋转 Q、K 向量来编码位置信息。

```python
def apply_rope(x, cos, sin):
    """应用旋转位置编码"""
    # x: (batch, seq, heads, dim)
    x1 = x[..., ::2]   # 偶数维度
    x2 = x[..., 1::2]  # 奇数维度
    
    # 旋转操作
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rotated
```

### 8.3 GQA (Grouped Query Attention)

分组查询注意力，在 MHA 和 MQA 之间取得平衡：

```mermaid
graph TB
    subgraph MHA 标准多头
        Q1[Q1] --> KV1[K1,V1]
        Q2[Q2] --> KV2[K2,V2]
        Q3[Q3] --> KV3[K3,V3]
        Q4[Q4] --> KV4[K4,V4]
    end
    
    subgraph GQA 分组查询
        GQ1[Q1] --> GKV1[K1,V1]
        GQ2[Q2] --> GKV1
        GQ3[Q3] --> GKV2[K2,V2]
        GQ4[Q4] --> GKV2
    end
    
    subgraph MQA 多查询
        MQ1[Q1] --> MKV[K,V]
        MQ2[Q2] --> MKV
        MQ3[Q3] --> MKV
        MQ4[Q4] --> MKV
    end
```

| 注意力类型 | KV头数 | 显存占用 | 推理速度 | 效果 |
|-----------|--------|----------|----------|------|
| **MHA** | =Q头数 | 高 | 慢 | 最好 |
| **GQA** | Q头数/g | 中 | 较快 | 接近MHA |
| **MQA** | 1 | 低 | 最快 | 有损失 |

**GQA 是现代 LLM 的首选**（LLaMA-2/3、Mistral 等）。

### 8.4 SwiGLU 激活

结合 Swish 和 GLU 的优势：

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
    
    def forward(self, x):
        # SwiGLU = Swish(xW1) ⊙ (xW3)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**优势**：比 ReLU/GELU 有更好的性能，是 LLaMA、PaLM 等模型的选择。

### 8.5 FlashAttention

优化注意力计算的内存访问模式：

```mermaid
graph LR
    subgraph Standard["标准Attention"]
        S1["O(n²)显存"]
        S2["多次HBM访问"]
    end

    subgraph Flash["FlashAttention"]
        F1["O(n)显存"]
        F2["分块计算"]
        F3["最少HBM访问"]
    end

    S1 --> |优化| F1
    S2 --> |优化| F2
```

**核心思想**：利用 GPU 内存层次结构，通过分块（tiling）减少 HBM 访问。

## 9. 计算复杂度分析

```mermaid
graph TB
    subgraph Self-Attention复杂度
        T["时间: O(n² × d)"]
        S["空间: O(n²) 注意力矩阵"]
    end
    
    subgraph 序列长度影响
        L1["n=1K → 1M操作"]
        L2["n=10K → 100M操作"]
        L3["n=100K → 10B操作"]
    end
```

这就是为什么长序列处理如此具有挑战性——注意力矩阵随序列长度**平方增长**。

## 10. 主流模型架构对比

| 模型 | 参数量 | 层数 | 注意力 | 位置编码 | FFN |
|------|--------|------|--------|----------|-----|
| **LLaMA-2-7B** | 7B | 32 | GQA | RoPE | SwiGLU |
| **LLaMA-2-70B** | 70B | 80 | GQA | RoPE | SwiGLU |
| **Mistral-7B** | 7B | 32 | GQA | RoPE | SwiGLU |
| **Qwen-7B** | 7B | 32 | MHA | RoPE | SwiGLU |
| **DeepSeek-V2** | 236B(21B) | - | MLA | RoPE | MoE |

## 11. 本章小结

```mermaid
mindmap
  root((Transformer演进))
    三大架构
      Encoder-only BERT
      Decoder-only GPT
      Encoder-Decoder T5
    为何Decoder-only
      训练效率高
      统一生成范式
      涌现能力强
      扩展性好
    核心机制
      因果掩码
      多头注意力
      残差连接
    现代改进
      RMSNorm
      RoPE
      GQA
      SwiGLU
      FlashAttention
```

**核心要点**：
- ✅ Transformer 衍生出三大架构，Decoder-only 成为 LLM 主流
- ✅ 因果掩码是 Decoder-only 架构的核心
- ✅ 现代 LLM 采用 RMSNorm、RoPE、GQA、SwiGLU 等改进
- ✅ FlashAttention 通过优化内存访问大幅提升效率

## 思考题

1. 如果让你设计一个推理引擎，Attention 层计算中最耗时的部分是什么？
2. 为什么 BERT 风格的模型难以用于文本生成？
3. GQA 是如何在效率和效果之间取得平衡的？

## 延伸阅读

- 📄 [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- 📄 [LLaMA: Open and Efficient Foundation Language Models (2023)](https://arxiv.org/abs/2302.13971)
- 📄 [FlashAttention: Fast and Memory-Efficient Exact Attention (2022)](https://arxiv.org/abs/2205.14135)
- 📄 [RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)

---

*下一篇：[注意力机制深度解析](./06-attention-mechanism.md) - 深入理解 Self-Attention 的数学原理*
