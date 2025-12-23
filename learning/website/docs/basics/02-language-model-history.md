---
sidebar_position: 2
---

# 语言模型简史：从 N-gram 到 GPT

语言模型 (Language Model, LM) 是自然语言处理的核心技术。本文将带你回顾语言模型的发展历程，理解为什么 Transformer 架构能够取得突破性成功。

## 1. 什么是语言模型？

语言模型的本质是**预测下一个词的概率分布**：

```
P(w_t | w_1, w_2, ..., w_{t-1})
```

给定前面的词序列，模型预测下一个词最可能是什么。这个看似简单的任务，实际上需要模型理解语法、语义，甚至世界知识。

```mermaid
graph LR
    subgraph 语言模型的任务
        A[我 爱 吃] --> LM[语言模型]
        LM --> B[苹果 0.3<br/>西瓜 0.2<br/>米饭 0.15<br/>...]
    end
```

## 2. 语言模型发展时间线

```mermaid
timeline
    title 语言模型发展历程
    section 统计时代
        1980s-2000s : N-gram模型
                    : 马尔可夫假设
    section 神经网络时代
        2003 : 神经概率语言模型
             : Bengio词向量
        2013 : Word2Vec
             : 高效词向量训练
    section RNN时代
        1997 : LSTM
             : 门控机制
        2014 : Seq2Seq
             : 机器翻译突破
        2015 : 注意力机制
             : 动态聚焦
    section Transformer时代
        2017 : Transformer
             : Attention Is All You Need
        2018 : GPT-1, BERT
             : 预训练范式
        2020 : GPT-3
             : 涌现能力
        2023-2024 : GPT-4, Claude, Gemini
                  : 多模态大模型
```

## 3. 第一阶段：统计语言模型 (1980s-2000s)

### 3.1 N-gram 模型

N-gram 是最早的语言模型方法，基于**马尔可夫假设**：当前词只依赖于前 N-1 个词。

```mermaid
graph LR
    subgraph B["Bigram (N=2)"]
        W1[我] --> W2[爱]
        W2 --> W3[吃]
    end
    
    subgraph T["Trigram (N=3)"]
        T1[我 爱] --> T2[吃]
        T2[爱 吃] --> T3[苹果]
    end
```

**计算示例**：预测 "我爱吃___"

```
统计语料库中 "我爱吃" 后面出现的词频:
- "苹果" 出现 100 次
- "西瓜" 出现 50 次  
- "米饭" 出现 30 次

P(苹果|我爱吃) = 100 / (100+50+30) ≈ 0.56
```

**N-gram 的局限性**：

| 问题 | 说明 |
|------|------|
| **无法捕捉长距离依赖** | 只看前几个词，忽略更远的上下文 |
| **数据稀疏** | 很多 N-gram 组合从未在语料中出现 |
| **存储爆炸** | 词表大小为 V，N-gram 组合数为 V^N |
| **缺乏泛化** | 无法理解语义相似性 |

## 4. 第二阶段：神经网络语言模型 (2003-2013)

### 4.1 词向量的诞生 (2003)

Bengio 等人提出**神经概率语言模型**，开创性地引入词向量概念：

```mermaid
graph TB
    subgraph 传统方法
        O1[One-hot编码]
        O1 --> |维度=词表大小| O2[稀疏向量]
    end
    
    subgraph 词向量方法
        W1[词] --> |查表| E[Embedding层]
        E --> |维度=几百| W2[稠密向量]
    end
```

**核心思想**：
- 将离散的词映射到**连续的向量空间**
- 语义相似的词，向量距离也相近
- 解决了数据稀疏问题

### 4.2 Word2Vec (2013)

Mikolov 提出 Word2Vec，让词向量训练变得高效实用：

```mermaid
graph TB
    subgraph CBOW
        C1[上文] --> M1[模型]
        C2[下文] --> M1
        M1 --> P1[预测中心词]
    end
    
    subgraph Skip-gram
        W[中心词] --> M2[模型]
        M2 --> P2[预测上下文]
    end
```

**著名的词向量算术**：
```
King - Man + Woman ≈ Queen
Paris - France + Italy ≈ Rome
```

这说明词向量捕捉到了语义关系！

## 5. 第三阶段：循环神经网络 (1986-2016)

### 5.1 RNN 基础

循环神经网络通过**隐藏状态**传递历史信息：

```mermaid
graph LR
    subgraph RNN展开
        X1[x₁] --> H1[h₁]
        H1 --> H2[h₂]
        X2[x₂] --> H2
        H2 --> H3[h₃]
        X3[x₃] --> H3
        H3 --> Y[输出]
    end
```

```
h_t = f(W_h · h_{t-1} + W_x · x_t + b)
y_t = g(h_t)
```

**问题**：梯度在反向传播时会指数级衰减（消失）或爆炸，难以学习长距离依赖。

### 5.2 LSTM (1997)

长短期记忆网络引入**门控机制**，像"阀门"一样控制信息流：

```mermaid
graph TB
    subgraph LSTM单元
        F[遗忘门<br/>决定丢弃什么]
        I[输入门<br/>决定存储什么]
        O[输出门<br/>决定输出什么]
        C[记忆单元<br/>Cell State]
        
        F --> C
        I --> C
        C --> O
    end
```

**三个门的作用**：
- **遗忘门 (Forget Gate)**：决定从记忆中丢弃哪些信息
- **输入门 (Input Gate)**：决定保存哪些新信息到记忆中
- **输出门 (Output Gate)**：决定输出记忆中的哪些信息

LSTM 成功缓解了梯度消失问题，成为 2010s 的主流架构。

### 5.3 GRU (2014)

门控循环单元是 LSTM 的简化版本：
- 将遗忘门和输入门合并为**更新门**
- 参数更少，训练更快
- 效果与 LSTM 相当

## 6. 第四阶段：Seq2Seq 与注意力机制 (2014-2017)

### 6.1 Seq2Seq 模型

序列到序列模型为机器翻译带来突破：

```mermaid
graph LR
    subgraph Encoder 编码器
        E1[I] --> E2[love]
        E2 --> E3[you]
        E3 --> C[Context<br/>上下文向量]
    end
    
    subgraph Decoder 解码器
        C --> D1[我]
        D1 --> D2[爱]
        D2 --> D3[你]
    end
```

**问题**：整个输入序列压缩到一个固定长度向量，信息瓶颈严重！

### 6.2 注意力机制 (2015)

Bahdanau 等人提出注意力机制，允许 Decoder 在每一步"关注"输入序列的不同位置：

```mermaid
graph TB
    subgraph 注意力机制
        H1[h₁] --> |α₁| C[Context]
        H2[h₂] --> |α₂| C
        H3[h₃] --> |α₃| C
        
        C --> D[Decoder输出]
    end
    
    subgraph 注意力权重
        A["翻译'爱'时:<br/>α₁=0.1, α₂=0.8, α₃=0.1"]
    end
```

**核心思想**：不再压缩成固定向量，而是**动态选择**相关信息。

## 7. 第五阶段：Transformer 革命 (2017)

### 7.1 "Attention is All You Need"

Google 提出 Transformer 架构，**完全抛弃循环结构**：

```mermaid
graph TB
    subgraph Transformer架构
        Input[输入] --> PE[位置编码]
        PE --> SA[自注意力层]
        SA --> FF[前馈网络]
        FF --> |重复N次| Output[输出]
    end
```

**核心创新**：

| 创新点 | 说明 |
|--------|------|
| **自注意力 (Self-Attention)** | 序列内部的元素相互关注 |
| **多头注意力 (Multi-Head)** | 从多个角度学习关系 |
| **位置编码** | 注入位置信息（因为没有顺序处理） |
| **并行计算** | 摆脱 RNN 的顺序依赖 |

### 7.2 为什么 Transformer 成功？

```mermaid
graph LR
    subgraph RNN处理方式
        R1[词1] --> R2[词2] --> R3[词3] --> R4[词4]
    end
    
    subgraph Transformer处理方式
        T1[词1] <--> T2[词2]
        T1 <--> T3[词3]
        T1 <--> T4[词4]
        T2 <--> T3
        T2 <--> T4
        T3 <--> T4
    end
```

| 对比维度 | RNN/LSTM | Transformer |
|----------|----------|-------------|
| **并行能力** | 差（必须顺序处理） | 强（完全并行） |
| **长距离依赖** | 信息逐步衰减 | 直接连接（路径长度=1） |
| **训练速度** | 慢 | 快 |
| **可扩展性** | 有限 | 极强（可堆叠很多层） |

## 8. 第六阶段：预训练大模型时代 (2018-至今)

### 8.1 GPT 系列 (OpenAI)

```mermaid
graph TB
    subgraph GPT发展历程
        G1[GPT-1 2018<br/>1.17亿参数] --> G2[GPT-2 2019<br/>15亿参数]
        G2 --> G3[GPT-3 2020<br/>1750亿参数]
        G3 --> G4[GPT-4 2023<br/>多模态]
        G4 --> G5[GPT-4 Turbo 2024<br/>128K上下文]
    end
```

| 模型 | 发布时间 | 参数量 | 重要突破 |
|------|----------|--------|----------|
| **GPT-1** | 2018.6 | 117M | 证明预训练+微调范式有效 |
| **GPT-2** | 2019.2 | 1.5B | 展示零样本学习能力 |
| **GPT-3** | 2020.6 | 175B | 涌现出上下文学习能力 |
| **GPT-4** | 2023.3 | 未公开 | 多模态，推理能力大幅提升 |
| **GPT-4 Turbo** | 2024 | 未公开 | 128K 上下文，更便宜更快 |

### 8.2 BERT (Google, 2018)

双向编码器，擅长**理解任务**：

```mermaid
graph LR
    subgraph GPT 单向
        G1[我] --> G2[爱] --> G3[你]
    end
    
    subgraph BERT 双向
        B1[我] <--> B2[爱] <--> B3[你]
    end
```

- **掩码语言模型 (MLM)**：随机遮住词，让模型预测
- **下一句预测 (NSP)**：判断两个句子是否连续

### 8.3 技术路线分化

```mermaid
graph TB
    T[Transformer] --> E[Encoder-only<br/>BERT系列]
    T --> D[Decoder-only<br/>GPT系列]
    T --> ED[Encoder-Decoder<br/>T5, BART]
    
    E --> E1[理解任务<br/>分类、NER、问答]
    D --> D1[生成任务<br/>对话、写作、代码]
    ED --> ED1[序列转换<br/>翻译、摘要]
```

| 路线 | 代表模型 | 特点 | 适用场景 |
|------|----------|------|----------|
| **Decoder-only** | GPT, LLaMA, Claude | 自回归生成 | 文本生成、对话、代码 |
| **Encoder-only** | BERT, RoBERTa | 双向理解 | 分类、实体识别、阅读理解 |
| **Encoder-Decoder** | T5, BART, Flan | 编码+生成 | 翻译、摘要、改写 |

### 8.4 2023-2024 主流大模型

| 模型 | 公司 | 特点 |
|------|------|------|
| **GPT-4** | OpenAI | 多模态，强推理 |
| **Claude 3** | Anthropic | 安全对齐，200K 上下文 |
| **Gemini** | Google | 原生多模态 |
| **LLaMA 2/3** | Meta | 开源开放 |
| **Mistral/Mixtral** | Mistral AI | 高效 MoE 架构 |
| **Qwen** | 阿里 | 中文优化 |
| **DeepSeek** | DeepSeek | 高性价比 |

## 9. 语言模型的涌现能力

当模型规模足够大时，会"涌现"出意想不到的能力：

```mermaid
graph LR
    S[规模增大] --> E[涌现能力]
    
    E --> E1[上下文学习<br/>In-Context Learning]
    E --> E2[思维链推理<br/>Chain-of-Thought]
    E --> E3[指令遵循<br/>Instruction Following]
    E --> E4[代码生成<br/>Code Generation]
```

**涌现能力示例**：
- **上下文学习**：无需微调，给几个示例就能学会新任务
- **思维链 (CoT)**：逐步推理复杂问题
- **指令遵循**：理解并执行复杂指令

## 10. 本章小结

```mermaid
mindmap
  root((语言模型发展))
    统计时代
      N-gram
      马尔可夫假设
    神经网络
      词向量
      Word2Vec
    RNN时代
      LSTM
      GRU
      Seq2Seq
    Transformer
      自注意力
      并行计算
    大模型时代
      GPT系列
      BERT
      开源模型
```

**关键里程碑总结**：

| 阶段 | 时间 | 代表技术 | 核心突破 |
|------|------|----------|----------|
| 统计方法 | 1980s-2000s | N-gram | 概率语言建模 |
| 神经网络 | 2003-2013 | Word2Vec | 词向量表示 |
| 循环网络 | 2014-2016 | LSTM, GRU | 序列建模 |
| 注意力 | 2015-2017 | Attention | 动态聚焦 |
| Transformer | 2017-2018 | Transformer | 并行计算 |
| 预训练 | 2018-2020 | GPT, BERT | 预训练范式 |
| 大模型 | 2020-至今 | GPT-3/4, Claude | 涌现能力 |

## 延伸阅读

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165)
- [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805)
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

*下一篇：[Transformer 架构演进](./05-transformer-architecture.md) - 深入理解 Transformer 的内部结构*
