---
sidebar_position: 11
---

# 监督微调：让模型学会对话

经过预训练的模型虽然具备强大的语言能力，但它只会"续写"文本，无法理解指令或进行对话。监督微调 (Supervised Fine-Tuning, SFT) 是让模型学会遵循指令的关键步骤。

## 从基座模型到对话模型

```
预训练模型 (Base Model)
    ↓ 监督微调 (SFT)
指令微调模型 (Instruct Model)
    ↓ RLHF/DPO
对齐模型 (Chat Model)
```

### 基座模型的行为

```
用户: 请用Python写一个快速排序
基座模型: 函数。快速排序是一种高效的排序算法，由Tony Hoare于1959年提出...
（只是在续写，不是在回答问题）
```

### SFT 后的行为

```
用户: 请用Python写一个快速排序
SFT模型: 
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    ...
```

## 指令数据格式

### 单轮对话格式

```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is really nice today."
}
```

### 多轮对话格式 (ShareGPT)

```json
{
  "conversations": [
    {"from": "human", "value": "你好，请介绍一下自己"},
    {"from": "gpt", "value": "你好！我是一个AI助手..."},
    {"from": "human", "value": "你能帮我写代码吗？"},
    {"from": "gpt", "value": "当然可以！请告诉我你需要什么..."}
  ]
}
```

### Chat Template

不同模型使用不同的对话模板：

**LLaMA-2 Chat**:
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

{user_message_1} [/INST] {assistant_response_1} </s><s>[INST] {user_message_2} [/INST]
```

**ChatML (Qwen, etc.)**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

## 高质量数据集

### 公开数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| **Alpaca** | 52K | Stanford 用 GPT-3.5 生成 |
| **ShareGPT** | ~90K | 真实用户与 ChatGPT 对话 |
| **OpenAssistant** | 160K | 众包标注 |
| **FLAN** | 1.8M | Google 多任务指令 |
| **WizardLM** | 250K | 复杂指令进化 |
| **UltraChat** | 1.5M | 大规模多轮对话 |

### 数据质量 > 数据数量

**LIMA 论文的发现**：仅用 1000 条高质量数据就能训练出不错的对话模型。

关键是数据的**多样性**和**质量**：
- 覆盖不同任务类型
- 回答准确、详细、有帮助
- 格式规范、无错误

## 训练方法

### 全参数微调 (Full Fine-tuning)

更新模型所有参数：

```python
for batch in dataloader:
    outputs = model(batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

**优点**：效果最好
**缺点**：需要大量显存，容易过拟合

### LoRA (Low-Rank Adaptation)

**核心思想**：冻结原始参数，只训练低秩分解矩阵。

```
原始: W (d × d)
LoRA: W + ΔW = W + BA, 其中 B (d × r), A (r × d), r << d
```

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
    lora_dropout=0.05,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 (0.1% of 7B)
```

**优点**：
- 显存占用小（可微调 7B+ 模型）
- 训练速度快
- 可合并回原模型

### QLoRA

LoRA + 4-bit 量化，进一步降低显存：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

**惊人效果**：单张 24GB GPU 可微调 65B 模型！

## 训练技巧

### 只计算 Response 部分的 Loss

```python
# 不计算 prompt 部分的 loss
labels = input_ids.clone()
labels[:, :prompt_length] = -100  # -100 表示忽略
```

### 学习率设置

SFT 通常使用较小的学习率：

```python
# 预训练: lr = 1e-4 ~ 3e-4
# SFT:    lr = 1e-5 ~ 2e-5
```

### Packing

将多个短样本打包成一个长序列，提高训练效率：

```
样本1: [A1, A2, A3, PAD, PAD, PAD]
样本2: [B1, B2, PAD, PAD, PAD, PAD]
样本3: [C1, C2, C3, C4, PAD, PAD]

打包后: [A1, A2, A3, B1, B2, C1, C2, C3, C4, PAD]
```

### NEFTune

在 embedding 层添加噪声，提升泛化能力：

```python
def noisy_embedding(embed, noise_alpha=5):
    noise = torch.randn_like(embed) * noise_alpha / sqrt(embed.shape[-1])
    return embed + noise
```

## 实战：使用 transformers 训练

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载数据集
dataset = load_dataset("tatsu-lab/alpaca")

# 数据处理
def preprocess(examples):
    prompts = [f"### Instruction:\n{inst}\n### Response:\n{out}" 
               for inst, out in zip(examples["instruction"], examples["output"])]
    return tokenizer(prompts, truncation=True, max_length=512)

dataset = dataset.map(preprocess, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=500,
    bf16=True,
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()
```

## 评估指标

### 自动评估

- **Perplexity**：困惑度，越低越好
- **BLEU/ROUGE**：与参考答案的重合度
- **Pass@k**：代码任务的通过率

### 人工评估

- **Helpfulness**：回答是否有帮助
- **Harmlessness**：回答是否安全
- **Honesty**：回答是否诚实

### Benchmark

| 评测集 | 评估能力 |
|--------|----------|
| MMLU | 多领域知识 |
| GSM8K | 数学推理 |
| HumanEval | 代码生成 |
| TruthfulQA | 真实性 |
| MT-Bench | 多轮对话 |

## 本章小结

- SFT 让预训练模型学会理解和遵循指令
- 数据质量比数量更重要
- LoRA/QLoRA 让个人也能微调大模型
- 合适的训练技巧能显著提升效果

## 延伸阅读

- LIMA: Less Is More for Alignment
- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Finetuning of Quantized LLMs

---

*下一篇：[人类反馈强化学习：对齐人类偏好](./12-rlhf.md)*
