---
sidebar_position: 3
---

# 分词器：文本到数字的桥梁

神经网络无法直接处理文本，需要先将文本转换为数字。分词器 (Tokenizer) 正是完成这一关键转换的组件。本文将深入探讨现代 LLM 中使用的分词技术。

## 1. 为什么需要分词？

```mermaid
graph LR
    A[文本<br/>'Hello, world!'] --> T[分词器<br/>Tokenizer]
    T --> B[Token IDs<br/>'15496, 11, 995, 0']
    B --> E[词嵌入<br/>Embedding]
    E --> M[神经网络<br/>Model]
```

**分词器的核心任务**：
1. **切分文本**：将连续文本切分为离散单元 (tokens)
2. **建立映射**：每个 token 对应一个唯一 ID
3. **构建词表**：所有可能 token 的集合 (vocabulary)

以 "Hello, world!" 为例，分词器会将其切分为 "Hello"、","、" world"、"!" 四个 token，然后分别映射为 15496、11、995、0 这样的数字 ID。

## 2. 分词粒度的选择

```mermaid
graph TB
    subgraph 字符级
        C1["H"] --> C2["e"] --> C3["l"] --> C4["l"] --> C5["o"]
    end
    
    subgraph 词级
        W1["Hello"] --> W2["world"]
    end
    
    subgraph 子词级
        S1["Hel"] --> S2["lo"] --> S3["wor"] --> S4["ld"]
    end
```

### 2.1 字符级 (Character-level)

将文本按单个字符切分，如 "Hello" 变成 H、e、l、l、o 五个 token。

| 优点 | 缺点 |
|------|------|
| 词表小（几百个字符） | 序列过长 |
| 无 OOV（未登录词）问题 | 难以捕捉语义 |
| 对错别字鲁棒 | 计算成本高 |

### 2.2 词级 (Word-level)

按完整单词切分，如 "Hello world" 变成 Hello、world 两个 token。

| 优点 | 缺点 |
|------|------|
| 语义清晰 | 词表巨大（几十万词） |
| 序列短 | 无法处理新词 (OOV) |
| | 无法学习词根词缀 |

### 2.3 子词级 (Subword-level) - 现代主流

将词切分为有意义的子单元，如 "unhappiness" 变成 un、happiness 两个 token，"playing" 变成 play、ing。

| 优点 | 缺点 |
|------|------|
| 平衡词表大小和序列长度 | 切分方式需要学习 |
| 能处理未见过的词 | |
| 捕捉词根、词缀等规律 | |

## 3. 主流分词算法

```mermaid
graph TB
    T[子词分词算法] --> BPE[BPE<br/>字节对编码]
    T --> WP[WordPiece<br/>词片]
    T --> UG[Unigram<br/>一元模型]
    T --> BBPE[Byte-Level BPE<br/>字节级BPE]
    
    BPE --> GPT[GPT-2/3<br/>LLaMA]
    BBPE --> GPT4[GPT-4<br/>Qwen]
    WP --> BERT[BERT<br/>DistilBERT]
    UG --> T5[T5<br/>ALBERT]
```

### 3.1 BPE (Byte Pair Encoding)

**核心思想**：从字符开始，不断合并最频繁出现的相邻 pair。

```mermaid
graph TB
    subgraph BPE训练过程
        I[初始词表<br/>所有字符] --> S1[统计相邻pair频率]
        S1 --> S2[合并最频繁pair]
        S2 --> S3[更新词表]
        S3 --> |重复| S1
        S3 --> |达到目标大小| F[最终词表]
    end
```

**训练示例**：

假设语料是 "low lower lowest"：

1. **初始化**：词表只包含所有单字符 l、o、w、e、r、s、t 和空格
2. **统计频率**：找出最常见的相邻字符对，发现 l-o 出现 3 次最多
3. **第一次合并**：将 l 和 o 合并为新 token "lo"
4. **继续统计**：发现 lo-w 最频繁，合并为 "low"
5. **重复**：直到词表达到目标大小

**分词时**：按照学习到的合并规则进行切分。比如 "lowest" 会被切分为 low 和 est，"slower" 会被切分为 s、low 和 er。

**使用 BPE 的模型**：GPT-2、GPT-3、LLaMA、Mistral

### 3.2 Byte-Level BPE (BBPE)

**核心思想**：直接在 UTF-8 字节序列上应用 BPE，而不是字符。

**优势**：
- 完全消除未登录词问题（任何文本都能编码）
- 更好的多语言支持
- 词表更紧凑

**使用 BBPE 的模型**：GPT-4、GPT-4o、Qwen 2、Claude

### 3.3 WordPiece

**核心思想**：选择使**语言模型似然最大化**的合并，而不是简单选频率最高的。

**特点**：使用 ## 标记非词首 token。比如 "unhappiness" 会被切分为 un、##hap、##pi、##ness，其中 ## 前缀表示这个 token 是某个词的一部分，不是词的开头。

**使用 WordPiece 的模型**：BERT、DistilBERT、ELECTRA

### 3.4 Unigram (SentencePiece)

**核心思想**：从大词表开始，逐步**删除**对似然影响最小的 token。

```mermaid
graph TB
    I[初始化大词表<br/>所有可能子串] --> C[计算每个token的损失]
    C --> D[删除损失最小的token]
    D --> |重复| C
    D --> |达到目标大小| F[最终词表]
```

**过程**：
1. 初始化一个很大的候选词表（包含所有可能的子串）
2. 用期望最大化 (EM) 算法计算每个 token 的概率
3. 删除移除后对总损失影响最小的 token
4. 重复直到达到目标大小

**使用 Unigram 的模型**：T5、ALBERT、XLNet、mBART

## 4. 特殊 Token

现代分词器都包含一些特殊 token：

```mermaid
graph LR
    subgraph 常见特殊Token
        PAD["[PAD]<br/>填充"]
        UNK["[UNK]<br/>未知词"]
        CLS["[CLS]<br/>句子表示"]
        SEP["[SEP]<br/>分隔符"]
        MASK["[MASK]<br/>掩码"]
        BOS["<s><br/>句首"]
        EOS["</s><br/>句尾"]
    end
```

| Token | 作用 | 使用场景 |
|-------|------|----------|
| [PAD] | 填充 | 对齐批次中不同长度的序列 |
| [UNK] | 未知词 | 词表外的 token（现代分词器很少用到） |
| [CLS] | 分类 | BERT 用于获取句子表示 |
| [SEP] | 分隔 | 分隔多个句子 |
| [MASK] | 掩码 | BERT MLM 预训练 |
| \<s\> / \</s\> | 句子边界 | 标记句子开始/结束 |
| \<\|endoftext\|\> | 文档结束 | GPT 系列 |
| \<\|im_start\|\> / \<\|im_end\|\> | 消息边界 | ChatML 格式 |

## 5. 分词器的使用流程

### 5.1 编码过程

分词器的编码过程分为以下步骤：

1. **预处理**：对输入文本进行规范化处理，如大小写转换、Unicode 标准化等
2. **切分**：根据算法（BPE/WordPiece/Unigram）将文本切分为 token 序列
3. **映射**：将每个 token 转换为对应的数字 ID
4. **添加特殊 token**：根据需要添加句首、句尾等特殊标记

例如，对于文本 "Hello, how are you?"，GPT-2 分词器会：
- 切分为 Hello、逗号、空格how、空格are、空格you、问号 六个 token
- 其中空格会被编码到后面的 token 中，用特殊符号 G（代表空格）表示
- 最终得到类似 15496、11、703、389、345、30 这样的 ID 序列

### 5.2 解码过程

解码是编码的逆过程，将数字 ID 序列还原为原始文本。好的分词器应该保证编码后再解码能完全还原原文。

### 5.3 批量处理

处理多个文本时，由于长度不同，需要进行填充（padding）使所有序列长度一致。同时会生成注意力掩码，告诉模型哪些位置是真实内容、哪些是填充。

## 6. 分词器的影响

### 6.1 对模型性能的影响

```mermaid
graph TB
    subgraph 词表大小的权衡
        S[词表太小] --> S1[序列变长<br/>计算成本高]
        L[词表太大] --> L1[嵌入参数多<br/>稀有词学不好]
        M[适中词表] --> M1[平衡效率和效果]
    end
```

**典型词表大小**：
| 模型 | 词表大小 |
|------|----------|
| GPT-2 | 50,257 |
| GPT-4 | 100,256 |
| LLaMA | 32,000 |
| LLaMA-3 | 128,000 |
| Qwen 2 | 151,851 |

### 6.2 压缩率

**压缩率** = 原始字符数 / Token 数

好的分词器能用更少的 token 表示相同文本，这意味着：
- 更低的 API 成本
- 更长的"有效上下文"

### 6.3 多语言挑战

英文分词器在中文上表现差。比如 GPT-2 分词器处理中文"你好世界"时，每个汉字会被拆成多个字节级别的 token，导致效率极低——4 个汉字可能变成 12 个 token。

```mermaid
graph LR
    subgraph 英文优化的分词器
        E["Hello world"] --> |2 tokens| E1[效率高]
        C1["你好世界"] --> |12 tokens| C2[效率低]
    end
    
    subgraph 多语言分词器
        E2["Hello world"] --> |3 tokens| E3[略有损失]
        C3["你好世界"] --> |4 tokens| C4[效率正常]
    end
```

**解决方案**：
- 训练多语言分词器（如 Qwen、Gemma）
- 扩展词表添加中文 token（如 Chinese-LLaMA）
- 使用 Byte-level BPE（如 GPT-4）

## 7. 最新技术进展

### 7.1 领域自适应分词

2024 年的研究（如 ChipNeMo）表明，可以在预训练分词器基础上扩展领域专用词汇。方法是先在领域数据上训练新分词器，识别出基础词表中缺失的领域术语，然后将这些 token 添加到词表中。这种方法可以带来 1.6% 到 3.3% 的任务性能提升。

### 7.2 分词器优化

PickyBPE 等技术通过在训练时剔除低效的合并操作，消除那些很少被使用的"垃圾" token，在保持压缩效率的同时提升下游任务性能。

### 7.3 字节级处理

Meta 的 BLT 架构完全抛弃传统分词，直接处理原始字节序列。通过动态分组字节为"补丁"（patches），实现了约 50% 的推理加速，同时对噪声更鲁棒，对低资源语言支持更好。

## 8. Token 与成本计算

```mermaid
graph LR
    T[文本] --> TK[分词]
    TK --> N[Token数量]
    N --> C[API成本<br/>Token数 × 单价]
```

**主流模型 Token 价格 (2024)**：

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| GPT-4 Turbo | $10/1M tokens | $30/1M tokens |
| GPT-4o | $2.5/1M tokens | $10/1M tokens |
| Claude 3.5 Sonnet | $3/1M tokens | $15/1M tokens |
| Claude 3 Haiku | $0.25/1M tokens | $1.25/1M tokens |

**成本估算**：要估算 API 调用成本，首先需要用对应模型的分词器计算文本的 token 数量，然后乘以单价即可。OpenAI 提供的 tiktoken 库可以方便地进行这一计算。

## 9. 本章小结

```mermaid
mindmap
  root((分词器))
    为什么需要
      文本转数字
      建立词表
    粒度选择
      字符级
      词级
      子词级-主流
    主流算法
      BPE
      WordPiece
      Unigram
      Byte-Level BPE
    实践要点
      特殊Token
      多语言支持
      成本计算
```

**核心要点**：
- 分词器是 LLM 的"翻译官"，将文本转为数字
- 子词分词（BPE、WordPiece、Unigram）是现代主流
- 分词质量直接影响模型性能和推理成本
- 多语言场景需要特别关注分词器的选择
- Byte-Level BPE 正在成为新趋势，提供更好的跨语言支持

## 延伸阅读

- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers)
- Neural Machine Translation of Rare Words with Subword Units (BPE 论文)
- SentencePiece: A simple and language independent subword tokenizer
- [Tiktoken GitHub](https://github.com/openai/tiktoken)
- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

---

*下一篇：[位置编码：让模型理解顺序](./04-positional-encoding.md)*
