---
sidebar_position: 20
---

# 模型量化：用更少资源运行大模型

模型量化 (Quantization) 是将模型参数从高精度（如 FP16）转换为低精度（如 INT8/INT4）的技术，可以显著降低显存占用和加速推理。

## 为什么需要量化？

### 显存占用对比

```
7B 模型:
- FP32: 7B × 4 bytes = 28 GB
- FP16: 7B × 2 bytes = 14 GB
- INT8: 7B × 1 byte  = 7 GB
- INT4: 7B × 0.5 byte = 3.5 GB

量化让消费级 GPU (24GB) 也能运行 70B 模型！
```

### 量化的好处

- 📉 **显存占用降低**：INT4 只需 FP16 的 1/4
- ⚡ **推理速度提升**：内存带宽是推理瓶颈
- 💰 **部署成本降低**：可用更便宜的硬件

## 量化基础

### 数据类型回顾

| 类型 | 位宽 | 范围 | 精度 |
|------|------|------|------|
| FP32 | 32 | ±3.4×10³⁸ | 高 |
| FP16 | 16 | ±65504 | 中 |
| BF16 | 16 | ±3.4×10³⁸ | 低 |
| INT8 | 8 | -128~127 | 整数 |
| INT4 | 4 | -8~7 | 整数 |

### 量化公式

将浮点数映射到整数：

```
量化: q = round(x / scale) + zero_point
反量化: x' = (q - zero_point) × scale
```

### 量化粒度

| 粒度 | 描述 | 精度 | 开销 |
|------|------|------|------|
| Per-tensor | 整个张量共享 scale | 低 | 低 |
| Per-channel | 每个通道一个 scale | 中 | 中 |
| Per-group | 每 N 个元素一个 scale | 高 | 高 |

## 训练后量化 (PTQ)

### 基本方法

直接在训练好的模型上进行量化，无需重新训练：

```python
import torch

def naive_quantize(tensor, n_bits=8):
    # 计算 scale 和 zero_point
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (2**n_bits - 1)
    zero_point = round(-min_val / scale)
    
    # 量化
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, 0, 2**n_bits - 1)
    
    return q_tensor.to(torch.int8), scale, zero_point
```

### 校准 (Calibration)

使用少量数据确定最佳量化参数：

```python
def calibrate(model, calibration_data):
    # 收集每层激活值的统计信息
    for batch in calibration_data:
        model(batch)
        # 记录 min/max 或直方图
    
    # 确定最佳 scale 和 zero_point
    return quantization_params
```

## LLM.int8()

### 异常值问题

LLM 中存在少量**异常值 (Outliers)**，直接量化会导致精度损失：

```
大部分权重: [-0.5, 0.5]
异常值:     [-10, 10] 或更大

如果用统一的 scale，正常值精度损失严重
```

### 混合精度方案

```
1. 检测异常值（绝对值 > 阈值的维度）
2. 异常维度保持 FP16
3. 其余维度使用 INT8
4. 分别计算后合并
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # LLM.int8()
    device_map="auto",
)
```

## GPTQ

### 核心思想

逐层量化，同时最小化量化误差：

```
目标: min ||WX - Q(W)X||²

每次量化一个权重，调整剩余权重来补偿误差
```

### Optimal Brain Quantization (OBQ)

基于 Hessian 矩阵的最优量化顺序：

```python
# 伪代码
for i in range(n_weights):
    # 选择量化误差最小的权重
    idx = argmin(quant_error)
    
    # 量化该权重
    W[idx] = quantize(W[idx])
    
    # 调整剩余权重补偿误差
    W[remaining] -= H_inv[remaining, idx] * error[idx]
```

### 使用 GPTQ

```python
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",  # 校准数据集
    group_size=128,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## AWQ (Activation-aware Weight Quantization)

### 核心观察

不同权重的重要性不同。**激活值大**的对应权重更重要。

### 方法

```
1. 分析激活值分布
2. 识别重要权重（对应大激活值的列）
3. 对重要权重缩放后再量化
4. 推理时反向缩放
```

```python
# 权重重要性 ∝ 对应激活值的均值
importance = activation.abs().mean(dim=0)

# 缩放因子
scale = (importance / importance.max()) ** alpha

# 缩放后量化
W_scaled = W * scale
W_quant = quantize(W_scaled)

# 推理时: output = (W_quant / scale) @ activation
```

### 使用 AWQ

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    safetensors=True,
)

# 量化
model.quantize(
    tokenizer,
    quant_config={"w_bit": 4, "q_group_size": 128}
)
```

## GGUF/GGML

### 特点

- 专为 CPU 推理优化
- 支持多种量化格式
- 被 llama.cpp 广泛使用

### 量化类型

| 类型 | 描述 | 大小 (7B) |
|------|------|-----------|
| Q2_K | 2-bit | ~2.5 GB |
| Q4_0 | 4-bit | ~4 GB |
| Q4_K_M | 4-bit 混合 | ~4.5 GB |
| Q5_K_M | 5-bit 混合 | ~5 GB |
| Q8_0 | 8-bit | ~7 GB |

### 使用 llama.cpp

```bash
# 转换为 GGUF
python convert.py model_path --outtype f16 --outfile model.gguf

# 量化
./quantize model.gguf model-q4_k_m.gguf q4_k_m

# 推理
./main -m model-q4_k_m.gguf -p "Hello, world"
```

## 量化感知训练 (QAT)

### 与 PTQ 的区别

```
PTQ: 训练完成 → 量化 → 部署
QAT: 训练时模拟量化 → 量化 → 部署
```

### 直通估计器 (STE)

量化操作不可微，使用 STE 近似梯度：

```python
class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point):
        # 前向: 真实量化
        return torch.round(x / scale + zero_point) * scale - zero_point * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向: 直接传递梯度 (STE)
        return grad_output, None, None
```

### QLoRA

结合 LoRA 和量化，在量化模型上高效微调：

```python
from peft import prepare_model_for_kbit_training, LoraConfig

# 4-bit 量化加载
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 准备 QLoRA 训练
model = prepare_model_for_kbit_training(model)

# 添加 LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)
```

## 量化方法对比

| 方法 | 精度损失 | 速度提升 | 显存节省 | 易用性 |
|------|----------|----------|----------|--------|
| LLM.int8() | 小 | 中 | 50% | 高 |
| GPTQ | 小 | 大 | 75% | 中 |
| AWQ | 很小 | 大 | 75% | 中 |
| GGUF Q4 | 中 | 大 | 75% | 高 |

## 实战：量化并部署模型

```python
# 方法1: 使用 bitsandbytes (简单)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 方法2: 使用预量化模型
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-70B-GPTQ",
    device_map="auto",
)
```

## 本章小结

- 量化将模型参数转为低精度，降低显存和加速推理
- PTQ 简单快速，QAT 精度更高
- GPTQ、AWQ 是目前 LLM 量化的主流方法
- 不同场景选择不同方法：
  - 快速部署：bitsandbytes
  - 极致性能：GPTQ/AWQ
  - CPU 推理：GGUF

## 延伸阅读

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- AWQ: Activation-aware Weight Quantization

---

*本章是基础知识系列的最后一篇。接下来，让我们进入推理实战！*
