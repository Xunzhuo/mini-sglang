---
sidebar_position: 13
---

# 分布式训练：突破单卡限制

当模型大到单张 GPU 无法容纳时，分布式训练成为必需。本文将介绍训练大语言模型的各种并行策略。

## 为什么需要分布式训练？

### 显存占用分析

训练一个模型需要存储：

```
显存占用 = 模型参数 + 优化器状态 + 梯度 + 激活值

对于 7B 参数模型 (FP32):
- 模型参数: 7B × 4 bytes = 28 GB
- Adam 优化器: 7B × 8 bytes = 56 GB (m 和 v)
- 梯度: 7B × 4 bytes = 28 GB
- 激活值: 取决于 batch size 和序列长度
总计: > 112 GB （远超单卡显存）
```

### 解决方案：并行化

| 并行策略 | 切分对象 | 典型场景 |
|----------|----------|----------|
| 数据并行 (DP) | 数据 | 多卡加速训练 |
| 张量并行 (TP) | 模型层内参数 | 单层太大 |
| 流水线并行 (PP) | 模型层间 | 层数太多 |
| 序列并行 (SP) | 序列长度 | 长序列 |

## 数据并行 (Data Parallelism)

### 基本原理

每张卡持有完整模型副本，处理不同的数据：

```
数据 batch → 切分为 N 份 → N 张 GPU 各处理一份
                          ↓
           每张卡计算梯度 → AllReduce 同步 → 更新参数
```

### 实现方式

**PyTorch DistributedDataParallel (DDP)**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])

# 模型移动到对应 GPU
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 数据采样器
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # 梯度自动同步
    optimizer.step()
```

### 优缺点

✅ 实现简单，线性加速
❌ 每张卡需要存储完整模型，显存瓶颈

## ZeRO (Zero Redundancy Optimizer)

### 核心思想

数据并行中，每张卡都存储完整的优化器状态、梯度、参数——这是冗余的！

**ZeRO 三个阶段**：

| 阶段 | 切分内容 | 显存节省 |
|------|----------|----------|
| ZeRO-1 | 优化器状态 | 4x |
| ZeRO-2 | + 梯度 | 8x |
| ZeRO-3 | + 模型参数 | N x (N=GPU数) |

### ZeRO-3 原理

```
正常 DP: 每张卡存储 [全部参数 + 全部梯度 + 全部优化器状态]

ZeRO-3: 每张卡只存储 1/N 的 [参数 + 梯度 + 优化器状态]
        需要时通过 AllGather 获取其他部分
```

### DeepSpeed 实现

```python
import deepspeed

# deepspeed_config.json
config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "bf16": {"enabled": True},
    "train_batch_size": 32,
}

# 初始化
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=config,
)
```

### FSDP (Fully Sharded Data Parallel)

PyTorch 原生的 ZeRO-3 实现：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
)
```

## 张量并行 (Tensor Parallelism)

### 核心思想

将单个层的参数切分到多张卡上：

```
原始: Y = XW, W 是 (d_in, d_out)

列切分: W = [W1, W2]
        Y = [XW1, XW2] → AllGather → Y

行切分: W = [W1; W2], X = [X1, X2]
        Y = X1W1 + X2W2 → AllReduce → Y
```

### MLP 层的张量并行

```
           ┌─────────────────────────────┐
    X  →   │  Column Parallel (W1)       │ → GELU
           │  [d, 4d/N] on each GPU      │
           └─────────────────────────────┘
                         ↓
           ┌─────────────────────────────┐
           │  Row Parallel (W2)          │ → AllReduce → Y
           │  [4d/N, d] on each GPU      │
           └─────────────────────────────┘
```

### Attention 层的张量并行

```
Q, K, V 投影: Column Parallel (按 head 切分)
输出投影: Row Parallel

每张 GPU 计算部分 heads 的 attention
```

### Megatron-LM 实现

```python
# Megatron 风格的 Column Parallel Linear
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.out_features_per_partition = out_features // world_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
    
    def forward(self, x):
        output = F.linear(x, self.weight)
        # 后续需要 AllGather
        return output
```

## 流水线并行 (Pipeline Parallelism)

### 核心思想

将模型按层切分到不同 GPU：

```
GPU 0: 层 0-7
GPU 1: 层 8-15
GPU 2: 层 16-23
GPU 3: 层 24-31

数据流: GPU0 → GPU1 → GPU2 → GPU3
```

### 朴素流水线的问题

```
时间 →
GPU 0: [F0][  ][  ][  ][B0][  ][  ][  ]
GPU 1: [  ][F1][  ][  ][  ][B1][  ][  ]
GPU 2: [  ][  ][F2][  ][  ][  ][B2][  ]
GPU 3: [  ][  ][  ][F3][  ][  ][  ][B3]

大量 GPU 空闲时间！(气泡)
```

### Micro-batch 流水线

将一个 batch 切分为多个 micro-batch：

```
时间 →
GPU 0: [F0][F1][F2][F3][B3][B2][B1][B0]
GPU 1: [  ][F0][F1][F2][F3][B3][B2][B1]
GPU 2: [  ][  ][F0][F1][F2][F3][B3][B2]
GPU 3: [  ][  ][  ][F0][F1][F2][F3][B3]

气泡大大减少！
```

### 调度策略

| 策略 | 特点 |
|------|------|
| **GPipe** | 先所有前向，再所有反向，内存占用大 |
| **1F1B** | 交替前向反向，内存占用小 |
| **Interleaved 1F1B** | 每张卡负责多段，进一步减少气泡 |

## 3D 并行

### 组合使用

大规模训练通常组合多种并行策略：

```
3D 并行 = 数据并行 × 张量并行 × 流水线并行

例如: 1024 GPUs
- DP: 64 组
- TP: 8 (每组 8 卡做张量并行)
- PP: 2 (每组分 2 段流水线)
```

### 配置示例

```python
# Megatron-DeepSpeed 配置
parallel_config = {
    "tensor_model_parallel_size": 8,   # TP
    "pipeline_model_parallel_size": 4,  # PP
    "data_parallel_size": 32,           # DP (自动计算)
}
# 总 GPU 数 = 8 × 4 × 32 = 1024
```

## 序列并行 (Sequence Parallelism)

### 动机

注意力计算的显存与序列长度平方成正比：

```
Attention 显存 ∝ O(seq_len²)
```

### Ring Attention

将序列切分到不同 GPU，通过环形通信计算完整 attention：

```
GPU 0: Q[0:L/4], K[0:L/4], V[0:L/4]
GPU 1: Q[L/4:L/2], K[L/4:L/2], V[L/4:L/2]
...

环形传递 K, V，逐步计算完整 attention
```

## 混合精度训练

### 数据类型

| 类型 | 位宽 | 范围 | 用途 |
|------|------|------|------|
| FP32 | 32 | 大 | 主参数 |
| FP16 | 16 | 小 | 计算 |
| BF16 | 16 | 大 | 计算 (推荐) |

### Loss Scaling

FP16 范围小，容易下溢。解决方案：

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### BF16 的优势

```
FP16: 1 符号位 + 5 指数位 + 10 尾数位 → 精度高，范围小
BF16: 1 符号位 + 8 指数位 + 7 尾数位  → 精度低，范围大

BF16 范围与 FP32 相同，无需 loss scaling！
```

## 实战：多节点训练

### 启动命令

```bash
# 使用 torchrun (PyTorch 推荐)
torchrun --nproc_per_node=8 \
         --nnodes=4 \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
         train.py

# 使用 DeepSpeed
deepspeed --num_gpus=8 \
          --num_nodes=4 \
          --hostfile=hostfile \
          train.py --deepspeed_config ds_config.json
```

### 通信优化

```python
# 梯度压缩
model = DDP(model, gradient_as_bucket_view=True)

# 通信与计算重叠
model = DDP(model, overlap_comm=True)
```

## 本章小结

| 并行策略 | 切分 | 显存效率 | 通信开销 |
|----------|------|----------|----------|
| DP | 数据 | 低 | 中 (AllReduce) |
| ZeRO | 状态 | 高 | 高 |
| TP | 层内参数 | 中 | 高 (AllReduce) |
| PP | 层间 | 中 | 低 (P2P) |

- 数据并行是基础，ZeRO 大幅提升显存效率
- 张量并行适合单层过大的情况
- 流水线并行适合层数过多的情况
- 大规模训练需要组合多种策略

## 延伸阅读

- Megatron-LM: Training Multi-Billion Parameter Language Models
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- GPipe: Efficient Training of Giant Neural Networks

---

*下一篇：[推理揭秘：Prefill 与 Decode](./14-inference-process.md)*
