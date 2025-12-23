import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: '从零学习 LLM',
  tagline: '面向 LLM 初学者，从基础知识出发到 SGLang，理解大语言模型',
  favicon: 'img/favicon.ico',

  url: 'https://your-site.netlify.app',
  baseUrl: '/',

  organizationName: 'mini-sglang',
  projectName: 'llm-learning-handbook',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'zh-CN',
    locales: ['zh-CN'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/sgl-project/mini-sglang/tree/main/learning/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    image: 'img/social-card.jpg',
    navbar: {
      title: '从零学习 LLM',
      logo: {
        alt: 'Mini-SGLang Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          href: 'https://github.com/sgl-project/mini-sglang',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '文档',
          items: [
            {
              label: '开始学习',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: '社区',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/sgl-project/mini-sglang',
            },
          ],
        },
        {
          title: '更多',
          items: [
            {
              label: 'SGLang 官网',
              href: 'https://sgl-project.github.io/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} From Scratch to Master LLM.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

