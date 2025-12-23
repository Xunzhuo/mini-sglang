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
          id: 'video-tutorials',
          label: '🎬 视频教程',
        },
        {
          type: 'doc',
          id: 'papers',
          label: '📄 经典论文',
        },
        {
          type: 'doc',
          id: 'projects',
          label: '🚀 开源项目',
        },
      ],
    },
    {
      type: 'category',
      label: '📖 系列教程',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'chapter-01',
          label: '第一章：项目概述与架构设计',
        },
        {
          type: 'doc',
          id: 'chapter-02',
          label: '第二章：核心数据结构与上下文管理',
        },
        {
          type: 'doc',
          id: 'chapter-03',
          label: '第三章：分布式系统与通信机制',
        },
        {
          type: 'doc',
          id: 'chapter-04',
          label: '第四章：推理引擎与调度器系统',
        },
        {
          type: 'doc',
          id: 'chapter-05',
          label: '第五章：KV缓存管理与Radix树优化',
        },
        {
          type: 'doc',
          id: 'chapter-06',
          label: '第六章：高性能内核与CUDA优化',
        },
        {
          type: 'doc',
          id: 'chapter-07',
          label: '第七章：模型层实现与注意力机制',
        },
        {
          type: 'doc',
          id: 'chapter-08',
          label: '第八章：API服务器与系统集成',
        },
      ],
    },
  ],
};

export default sidebars;

