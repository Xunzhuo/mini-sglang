---
sidebar_position: 10
---

# 预训练：从海量数据到语言理解

预训练 (Pre-training) 是大语言模型获得强大能力的基础。本文将深入探讨预训练的原理、数据、方法和关键技术。

## 什么是预训练？

预训练是在**大规模无标注文本**上，通过**自监督学习**让模型学习语言的通用表示。

```
互联网文本 (TB级) → 预训练 → 基座模型 (Base Model)
```

预训练后的模型具备：
- 语法和语义理解
- 世界知识
- 推理能力的雏形

## 预训练目标

### Causal Language Modeling (CLM)

**GPT 系列采用的方法**：预测下一个 token。

```
输入: "今天天气"
目标: "真" "好" "！"

损失函数: L = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**特点**：
- 单向注意力（只能看到前文）
- 天然适合文本生成
- 训练效率高

### Masked Language Modeling (MLM)

**BERT 采用的方法**：完形填空。

```
输入: "今天 [MASK] 真好"
目标: 预测 [MASK] = "天气"
```

**特点**：
- 双向注意力
- 擅长理解任务
- 不适合生成

### Span Corruption

**T5 采用的方法**：预测被遮盖的连续片段。

```
输入: "今天<X>真好"
目标: "<X>天气<Y>"
```

## 预训练数据

### 数据来源

| 数据源 | 特点 | 示例 |
|--------|------|------|
| 网页 | 规模大，噪声多 | Common Crawl |
| 书籍 | 质量高，多样性 | Books3, Gutenberg |
| 代码 | 逻辑性强 | GitHub, StackOverflow |
| 论文 | 专业知识 | arXiv, S2ORC |
| 百科 | 结构化知识 | Wikipedia |
| 对话 | 交互模式 | Reddit |

### 数据处理流程

```
原始数据
    ↓ 去重 (Deduplication)
    ↓ 质量过滤 (Quality Filtering)
    ↓ 敏感内容过滤
    ↓ 语言识别
    ↓ 分词
预训练语料
```

### 数据质量的影响

**Chinchilla 论文的发现**：数据量和模型大小同样重要！

```
最优配比: 参数量 (N) 与 Token 数 (D) 应成正比
例如: 70B 参数模型需要约 1.4T tokens
```

### 数据配比

不同来源数据的混合比例影响模型能力：

```python
# LLaMA 数据配比示例
data_mix = {
    "CommonCrawl": 67.0,  # 网页
    "C4": 15.0,           # 清洗后网页
    "GitHub": 4.5,        # 代码
    "Wikipedia": 4.5,     # 百科
    "Books": 4.5,         # 书籍
    "ArXiv": 2.5,         # 论文
    "StackExchange": 2.0  # 问答
}
```

## Scaling Law

### OpenAI Scaling Law

模型性能（损失）与三个因素的幂律关系：

```
L(N, D, C) ≈ (N_c/N)^α + (D_c/D)^β + L_∞

其中:
- N: 模型参数量
- D: 数据量 (tokens)
- C: 计算量 (FLOPs)
```

### 关键发现

1. **模型越大，损失越低**（但边际效益递减）
2. **数据越多，损失越低**
3. **固定计算预算下，存在最优的模型大小和数据量配比**

### Chinchilla Scaling

Google DeepMind 的更新研究：

```
之前认为: 模型大小 > 数据量
Chinchilla: 模型大小 ≈ 数据量

结论: 很多大模型训练不充分 (undertrained)
```

## 预训练技术细节

### 优化器

**AdamW** 是标准选择：

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

### 学习率调度

典型的 **Cosine Annealing with Warmup**：

```
学习率
  ↑
  |   /‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |  /               \
  | /                 \
  |/                   \______
  +------------------------→ steps
   warmup   cosine decay
```

```python
def cosine_schedule(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

### 混合精度训练

使用 BF16/FP16 加速训练：

```python
# PyTorch AMP
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 梯度裁剪

防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 训练稳定性

### 常见问题

1. **Loss Spike**：损失突然飙升
   - 原因：学习率过大、数据异常、数值不稳定
   - 解决：回退 checkpoint，降低学习率

2. **Loss 不下降**
   - 原因：学习率过小、梯度消失
   - 解决：调整超参数，检查模型初始化

### 监控指标

```python
# 关键监控项
metrics = {
    "loss": training_loss,
    "grad_norm": gradient_norm,
    "lr": current_learning_rate,
    "throughput": tokens_per_second,
    "gpu_memory": memory_allocated,
}
```

## 预训练的计算成本

### 估算公式

```
训练 FLOPs ≈ 6 × N × D

其中:
- N: 模型参数量
- D: 训练 token 数
- 6: 前向 (2) + 反向 (4) 的乘数
```

### 成本示例

| 模型 | 参数量 | Token 数 | 估算成本 |
|------|--------|----------|----------|
| GPT-3 | 175B | 300B | ~$5M |
| LLaMA-2 70B | 70B | 2T | ~$3M |
| GPT-4 | ~1.8T (推测) | ? | ~$100M |

## 本章小结

- 预训练通过自监督学习让模型学习语言
- CLM（预测下一个词）是主流预训练目标
- 数据质量和规模同样重要
- Scaling Law 指导模型和数据的配比
- 训练稳定性需要精心设计

## 延伸阅读

- Language Models are Few-Shot Learners (GPT-3)
- Training Compute-Optimal Large Language Models (Chinchilla)
- LLaMA: Open and Efficient Foundation Language Models

---

*下一篇：[监督微调：让模型学会对话](./11-sft.md)*
