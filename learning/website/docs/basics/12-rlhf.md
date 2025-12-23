---
sidebar_position: 12
---

# 人类反馈强化学习：对齐人类偏好

经过 SFT 的模型已经能够遵循指令，但它的回答可能仍然不够"好"——可能有害、不诚实或无帮助。RLHF (Reinforcement Learning from Human Feedback) 是让模型更好地对齐人类偏好的关键技术。

## 为什么需要 RLHF？

### SFT 的局限性

```
问题: "如何制作炸弹？"

SFT 模型可能会回答:
"首先，你需要准备以下材料：..."
（因为训练数据中可能有类似内容）

RLHF 后的模型:
"我无法提供制作危险物品的信息。如果你对化学感兴趣，
我可以推荐一些安全的科学实验..."
```

### 对齐的三个目标 (HHH)

- **Helpful (有帮助)**：回答有用且相关
- **Harmless (无害)**：不输出有害内容
- **Honest (诚实)**：不编造事实，承认不确定性

## RLHF 三阶段流程

```
阶段1: 监督微调 (SFT)
    基座模型 → SFT 模型

阶段2: 奖励模型训练 (RM)
    人类偏好数据 → 奖励模型

阶段3: 强化学习优化 (RL)
    SFT 模型 + 奖励模型 → 对齐模型
```

## 阶段一：监督微调 (SFT)

（详见上一章）

## 阶段二：奖励模型 (Reward Model)

### 收集偏好数据

让人类标注员对模型回答进行**排序**：

```
问题: "解释什么是量子纠缠"

回答A: "量子纠缠是一种量子力学现象，当两个粒子..." (详细准确)
回答B: "就是两个粒子纠缠在一起" (过于简单)
回答C: "量子纠缠可以用来进行超光速通信" (错误)

人类排序: A > B > C
```

### 训练奖励模型

奖励模型学习预测人类偏好：

```
输入: (问题, 回答)
输出: 分数 (标量)
```

**Bradley-Terry 模型**：

```
P(A > B) = σ(r(A) - r(B)) = 1 / (1 + exp(r(B) - r(A)))
```

**训练目标**：

```python
def reward_loss(r_chosen, r_rejected):
    # r_chosen: 被选中回答的奖励
    # r_rejected: 被拒绝回答的奖励
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
```

### 奖励模型架构

通常基于 LLM，去掉语言模型头，加上一个标量输出头：

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # 取最后一个 token
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
```

## 阶段三：强化学习 (PPO)

### PPO 算法

**Proximal Policy Optimization (PPO)** 是 RLHF 中最常用的 RL 算法。

**核心思想**：
1. 用当前策略生成回答
2. 用奖励模型评分
3. 更新策略以获得更高奖励
4. 限制更新幅度，防止策略崩溃

### PPO 损失函数

```python
def ppo_loss(old_logprobs, new_logprobs, advantages, clip_ratio=0.2):
    # 重要性采样比率
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # Clipped surrogate objective
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    loss = -torch.min(ratio * advantages, clip_adv).mean()
    
    return loss
```

### KL 散度惩罚

防止模型偏离 SFT 版本太远：

```
总奖励 = 奖励模型分数 - β × KL(π_RL || π_SFT)
```

```python
def compute_reward_with_kl(reward, new_logprobs, ref_logprobs, kl_coef=0.1):
    kl_div = new_logprobs - ref_logprobs
    return reward - kl_coef * kl_div
```

### 完整 PPO 训练流程

```python
for batch in dataloader:
    # 1. 生成回答
    with torch.no_grad():
        responses = policy_model.generate(batch["prompts"])
        old_logprobs = policy_model.logprobs(responses)
        ref_logprobs = ref_model.logprobs(responses)
    
    # 2. 计算奖励
    rewards = reward_model(batch["prompts"], responses)
    rewards = compute_reward_with_kl(rewards, old_logprobs, ref_logprobs)
    
    # 3. 计算优势
    advantages = compute_advantages(rewards, values)
    
    # 4. PPO 更新
    for _ in range(ppo_epochs):
        new_logprobs = policy_model.logprobs(responses)
        loss = ppo_loss(old_logprobs, new_logprobs, advantages)
        loss.backward()
        optimizer.step()
```

## DPO：更简单的替代方案

### PPO 的问题

- 训练不稳定
- 需要同时维护多个模型（策略、参考、奖励、价值）
- 超参数敏感

### Direct Preference Optimization (DPO)

**核心洞察**：可以直接从偏好数据优化策略，无需显式奖励模型！

**数学推导**：

```
最优策略: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)

反解奖励: r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + const
```

**DPO 损失函数**：

```python
def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    
    # 计算 log ratio
    policy_ratio = policy_logps_chosen - policy_logps_rejected
    ref_ratio = ref_logps_chosen - ref_logps_rejected
    
    # DPO loss
    logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(logits).mean()
    
    return loss
```

### DPO vs PPO

| 对比 | PPO | DPO |
|------|-----|-----|
| 稳定性 | 较差 | 好 |
| 实现复杂度 | 高 | 低 |
| 模型数量 | 4个 | 2个 |
| 显存占用 | 高 | 低 |
| 效果 | 好 | 接近 |

## 其他对齐方法

### RLAIF (AI Feedback)

用强大的 AI（如 GPT-4）替代人类标注：

```
1. 生成多个回答
2. 让 GPT-4 评判哪个更好
3. 用 AI 偏好数据训练
```

**优点**：成本低、规模大
**缺点**：可能放大 AI 偏见

### Constitutional AI (CAI)

让 AI 自我批评和修正：

```
1. 生成初始回答
2. AI 根据"宪法"原则批评自己
3. 生成改进后的回答
4. 用 (初始, 改进) 对训练
```

### GRPO (Group Relative Policy Optimization)

DeepSeek 提出的方法，无需奖励模型：

```
1. 对每个问题采样多个回答
2. 用组内相对排名作为奖励信号
3. 直接优化策略
```

### KTO (Kahneman-Tversky Optimization)

只需要二元反馈（好/坏），不需要成对比较：

```
数据格式: (问题, 回答, 标签)
标签: 1 (好) 或 0 (坏)
```

## 实战：使用 TRL 训练

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("your_sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("your_sft_model")
tokenizer = AutoTokenizer.from_pretrained("your_sft_model")

# 准备数据集
# 格式: {"prompt": ..., "chosen": ..., "rejected": ...}
dataset = load_dataset("your_preference_data")

# 配置
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)

# 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 对齐税 (Alignment Tax)

RLHF 可能导致模型在某些能力上退化：

- 创造力下降（回答变得保守）
- 拒绝过多（过度安全）
- 性能下降（某些基准测试分数降低）

**平衡之道**：
- 调整 KL 惩罚系数
- 混合 SFT 和 RLHF 数据
- 多目标优化

## 本章小结

- RLHF 让模型学习人类偏好，实现 HHH 对齐
- 传统方法：SFT → RM → PPO 三阶段
- DPO 提供了更简单高效的替代方案
- 对齐是一个持续演进的领域

## 延伸阅读

- Training language models to follow instructions with human feedback (InstructGPT)
- Direct Preference Optimization (DPO)
- Constitutional AI: Harmlessness from AI Feedback

---

*下一篇：[分布式训练：突破单卡限制](./13-distributed-training.md)*
