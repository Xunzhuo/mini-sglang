---
sidebar_position: 2
title: 视频教程
description: LLM 学习路线图与推荐视频课程
---

# 视频教程

## Roadmap 思路

主要拆成 9 个 Phase：

1. 先补齐"学习/优化/泛化"的基本直觉
2. 再搞清 Transformer 这台发动机怎么运转
3. 接着亲手走一遍端到端训练，理解模型能力从哪来
4. 然后用 fine-tuning 把"会续写"变成"能按指令做事"
5. 用 RLHF 让模型真正对齐人类偏好和价值观
6. 再用 RAG 和 context engineering 把可信度做成可控流程
7. 往上扩展到 agent 的工具调用与状态管理
8. 最后进入推理与部署的真实约束（吞吐/延迟/成本）
9. 精读论文把整个体系变成长期可持续迭代的能力

---

## 推荐学习顺序

### Phase 1 — 机器学习 → 深度学习基础

**目标：** 建立"学习/优化/泛化"的基本心智模型

- [李沐 - 机器学习](https://www.bilibili.com/video/BV13U4y1N7Uo)
- [李沐 - 深度学习](https://www.bilibili.com/video/BV1daQAYuEYm)
- [吴恩达 - 深度学习](https://www.bilibili.com/video/BV11H4y1F7uH)

---

### Phase 2 — Transformer（核心架构）

**目标：** Attention 不是玄学，是一种可计算、可优化、有代价的模式

- [吴恩达 - Transformer](https://www.bilibili.com/video/BV13QnYzJE6q)
- [Karpathy - Transformer](https://www.bilibili.com/video/BV1GzwderEYf)

---

### Phase 3 — 从 0 构建 LLM（端到端训练）

**目标：** 理解 LLM 工厂：数据 → 目标 → 训练 loop → 评估 → 生成

- [吴恩达 - LLM](https://www.bilibili.com/video/BV1sMEyzhEM3)
- [从 0 构建 LLM](https://www.bilibili.com/video/BV16AKAzzECq)

---

### Phase 4 — Fine-tuning（从"续写"到"听话"）

**目标：** 搞清楚指令对齐的基本路径与工程套路

- [吴恩达 - Finetune](https://www.bilibili.com/video/BV1DRqbBZEBY)
- [吴恩达 - PyTorch](https://www.bilibili.com/video/BV1ir1WBrEuz)

---

### Phase 5 — RLHF（从"听话"到"对齐"）

**目标：** 理解强化学习如何让模型符合人类偏好，掌握 RLHF 的完整流程

- [李宏毅 - 强化学习](https://www.bilibili.com/video/BV1UE411G78S)
- [OpenAI - Spinning Up in Deep RL](https://spinningup.openai.com/)
- [HuggingFace - RLHF 教程](https://huggingface.co/blog/rlhf)
- [DeepMind - RLHF 讲解](https://www.deepmind.com/blog/learning-through-human-feedback)

---

### Phase 5 — RAG & Context Engineering

**目标：** 把知识注入变成可控流程：检索质量 + 生成质量可评估

- [吴恩达 - RAG](https://www.bilibili.com/video/BV1hD28BXEyR)
- [李宏毅 - Context Engineering](https://www.bilibili.com/video/BV1Wtncz1Erk)

---

### Phase 6 — Agent（从聊天到执行）

**目标：** 理解工具调用、规划、状态管理、以及"可控性"的边界

- [李宏毅 - AI Agent](https://www.bilibili.com/video/BV11nSjB2ErQ)
- [Shunyu Yao - LLM Agents: History & Overview](https://www.youtube.com/watch?v=RM6ZArd2nVc)
- [Shunyu Yao - Formulating and Evaluating Language Agents](https://www.youtube.com/watch?v=qmGu9okiICU)
- [Shunyu Yao - 从语言模型到语言智能体](https://www.bilibili.com/video/BV1ju4y1e7Em)

**相关论文：**

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)  
  让模型一边"想"，一边"做动作"，用外部世界纠偏内部幻觉
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)  
  把模型线性的推理过程，变成树形可探索的决策空间
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)  
  通过语言反思让 Agent 从失败中学习，用自然语言作为强化学习信号

---

### Phase 7 — 大模型推理与部署（决定项目能不能上生产）

**目标：** 掌握吞吐/延迟/成本的真实约束：KV cache、batch、并发、显存、量化等

- [大模型推理](https://space.bilibili.com/1540261574)

---

### Phase 8 — 精读论文（把学习变成持续的竞争力）

**目标：** 把论文阅读变成可复用流程，而不是"看完就忘"

- [李沐 - 精读论文](https://www.bilibili.com/video/BV1H44y1t75x)

