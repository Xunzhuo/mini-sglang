---
sidebar_position: 3
---

# 分词器：文本到数字的桥梁

神经网络无法直接处理文本，需要先将文本转换为数字。分词器 (Tokenizer) 正是完成这一关键转换的组件。本文将深入探讨现代 LLM 中使用的分词技术。

## 为什么需要分词？

```
输入: "Hello, world!"
          ↓ 分词器
输出: [15496, 11, 995, 0]  (token IDs)
```

分词器的作用：
1. **切分文本**：将连续文本切分为离散单元 (tokens)
2. **建立映射**：每个 token 对应一个唯一 ID
3. **构建词表**：所有可能 token 的集合

## 分词粒度的选择

### 字符级 (Character-level)

```
"Hello" → ['H', 'e', 'l', 'l', 'o']
```

- ✅ 词表小，无 OOV (Out-of-Vocabulary) 问题
- ❌ 序列过长，难以捕捉语义

### 词级 (Word-level)

```
"Hello world" → ['Hello', 'world']
```

- ✅ 语义清晰
- ❌ 词表巨大，无法处理新词

### 子词级 (Subword-level) ⭐️ 现代主流

```
"unhappiness" → ['un', 'happiness'] 或 ['un', 'happ', 'iness']
```

- ✅ 平衡词表大小和序列长度
- ✅ 能处理未见过的词
- ✅ 捕捉词根、词缀等语言规律

## 主流分词算法

### BPE (Byte Pair Encoding)

**核心思想**：从字符开始，不断合并最频繁出现的相邻 pair。

**训练过程**：

```
初始词表: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd', ...]

Step 1: 最频繁 pair 是 ('e', 's') → 合并为 'es'
Step 2: 最频繁 pair 是 ('es', 't') → 合并为 'est'
Step 3: 最频繁 pair 是 ('l', 'o') → 合并为 'lo'
...
重复直到达到目标词表大小
```

**分词过程**：

```
"lowest" → 按学习到的合并规则切分 → ['lo', 'w', 'est']
```

**使用 BPE 的模型**：GPT 系列、LLaMA、Mistral

### WordPiece

**核心思想**：选择使语言模型似然最大化的合并。

与 BPE 的区别：
- BPE：选择出现频率最高的 pair
- WordPiece：选择合并后能最大化训练数据概率的 pair

**特点**：使用 `##` 标记非词首 token

```
"unhappiness" → ['un', '##hap', '##pi', '##ness']
```

**使用 WordPiece 的模型**：BERT、DistilBERT

### Unigram (SentencePiece)

**核心思想**：从大词表开始，逐步删除对似然影响最小的 token。

**过程**：
1. 初始化一个很大的候选词表
2. 计算每个 token 的损失
3. 删除损失最小的 token
4. 重复直到达到目标大小

**使用 Unigram 的模型**：T5、ALBERT、XLNet

## 特殊 Token

现代分词器都包含一些特殊 token：

| Token | 作用 | 示例 |
|-------|------|------|
| `[PAD]` | 填充 | 对齐不同长度序列 |
| `[UNK]` | 未知词 | 词表外的 token |
| `[CLS]` | 分类 | BERT 句子表示 |
| `[SEP]` | 分隔 | 分隔多个句子 |
| `[MASK]` | 掩码 | BERT 预训练 |
| `<s>`, `</s>` | 句子边界 | 句首/句尾标记 |
| `<\|endoftext\|>` | 文档结束 | GPT 系列 |

## 实战：使用 HuggingFace Tokenizer

```python
from transformers import AutoTokenizer

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 编码
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']

ids = tokenizer.encode(text)
print(f"IDs: {ids}")
# [15496, 11, 703, 389, 345, 30]

# 解码
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
# "Hello, how are you?"
```

**注意**：`Ġ` 表示该 token 前有空格（GPT-2 的表示方式）

## 分词器的影响

### 对模型性能的影响

1. **词表大小**：
   - 太小 → 序列变长，计算成本高
   - 太大 → 嵌入层参数多，稀有词学不好

2. **压缩率**：
   - 好的分词器能用更少的 token 表示相同文本
   - 影响模型的"有效上下文长度"

### 多语言挑战

英文分词器在中文上表现差：

```python
# GPT-2 分词器处理中文
tokenizer.tokenize("你好世界")
# ['ä', '½', 'ł', 'å', '¥', '½', 'ä', '¸', 'ĸ', 'ç', 'ķ', 'Į']
# 每个汉字被拆成多个 byte！
```

**解决方案**：
- 训练多语言分词器
- 扩展词表（如 Chinese-LLaMA）
- 使用 Byte-level BPE（如 GPT-4）

## 训练自己的分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化 BPE 分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 定义训练器
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# 训练
files = ["data/corpus.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my_tokenizer.json")
```

## 本章小结

- 分词器是 LLM 的"翻译官"，将文本转为数字
- 子词分词（BPE、WordPiece、Unigram）是现代主流
- 分词质量直接影响模型性能
- 多语言场景需要特别关注分词器的选择

## 延伸阅读

- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers)
- Neural Machine Translation of Rare Words with Subword Units (BPE 论文)
- SentencePiece: A simple and language independent subword tokenizer

---

*下一篇：[位置编码：让模型理解顺序](./04-positional-encoding.md)*
