import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: '📚 学习总览',
    },
    {
      type: 'category',
      label: '🎓 学习资源',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'resources/video-tutorials',
          label: '🎬 视频教程',
        },
        {
          type: 'doc',
          id: 'resources/papers',
          label: '📄 经典论文',
        },
        {
          type: 'doc',
          id: 'resources/projects',
          label: '🚀 开源项目',
        },
      ],
    },
    {
      type: 'category',
      label: '🧠 基础知识',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: '第一部分：深度学习基础',
          collapsed: false,
          items: [
            {
              type: 'doc',
              id: 'basics/neural-network-basics',
              label: '神经网络入门',
            },
            {
              type: 'doc',
              id: 'basics/language-model-history',
              label: '语言模型简史',
            },
            {
              type: 'doc',
              id: 'basics/tokenization',
              label: '分词器：文本到数字的桥梁',
            },
            {
              type: 'doc',
              id: 'basics/positional-encoding',
              label: '位置编码：让模型理解顺序',
            },
            {
              type: 'doc',
              id: 'basics/transformer-architecture',
              label: 'Transformer 架构演进',
            },
          ],
        },
        {
          type: 'category',
          label: '第二部分：训练篇',
          collapsed: false,
          items: [
            {
              type: 'doc',
              id: 'basics/pretraining',
              label: '预训练：从海量数据到语言理解',
            },
            {
              type: 'doc',
              id: 'basics/sft',
              label: '监督微调：让模型学会对话',
            },
            {
              type: 'doc',
              id: 'basics/rlhf',
              label: 'RLHF：对齐人类偏好',
            },
            {
              type: 'doc',
              id: 'basics/distributed-training',
              label: '分布式训练：突破单卡限制',
            },
          ],
        },
        {
          type: 'category',
          label: '第三部分：推理篇',
          collapsed: false,
          items: [
            {
              type: 'doc',
              id: 'basics/inference-process',
              label: '推理揭秘：Prefill 与 Decode',
            },
            {
              type: 'doc',
              id: 'basics/kv-cache',
              label: '推理加速基石：KV Cache',
            },
            {
              type: 'doc',
              id: 'basics/memory-management',
              label: '显存管理：PagedAttention',
            },
            {
              type: 'doc',
              id: 'basics/scheduling',
              label: '吞吐量飞跃：Continuous Batching',
            },
            {
              type: 'doc',
              id: 'basics/radix-attention',
              label: '前缀复用：Radix Attention',
            },
            {
              type: 'doc',
              id: 'basics/distributed-inference',
              label: '分布式推理：Tensor Parallelism',
            },
            {
              type: 'doc',
              id: 'basics/quantization',
              label: '模型量化：用更少资源运行大模型',
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '📖 推理实战',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'actions/chapter-01',
          label: '第一章：项目概述与架构设计',
        },
        {
          type: 'doc',
          id: 'actions/chapter-02',
          label: '第二章：核心数据结构与上下文管理',
        },
        {
          type: 'doc',
          id: 'actions/chapter-03',
          label: '第三章：分布式系统与通信机制',
        },
        {
          type: 'doc',
          id: 'actions/chapter-04',
          label: '第四章：推理引擎与调度器系统',
        },
        {
          type: 'doc',
          id: 'actions/chapter-05',
          label: '第五章：KV缓存管理与Radix树优化',
        },
        {
          type: 'doc',
          id: 'actions/chapter-06',
          label: '第六章：高性能内核与CUDA优化',
        },
        {
          type: 'doc',
          id: 'actions/chapter-07',
          label: '第七章：模型层实现与注意力机制',
        },
        {
          type: 'doc',
          id: 'actions/chapter-08',
          label: '第八章：API服务器与系统集成',
        },
      ],
    },
  ],
};

export default sidebars;

