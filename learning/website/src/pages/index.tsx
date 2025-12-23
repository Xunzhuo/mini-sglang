import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            开始学习
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="https://github.com/sgl-project/mini-sglang">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`欢迎来到 ${siteConfig.title}`}
      description="从零学习大语言模型推理系统 - Mini-SGLang 实战手册">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">系统化学习</Heading>
                  <p>
                    8个章节全面覆盖LLM推理系统的核心技术栈，
                    从项目架构到具体实现。
                  </p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">深入浅出</Heading>
                  <p>
                    结合代码实例和架构图，理解分布式系统、
                    GPU优化和注意力机制。
                  </p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">实战导向</Heading>
                  <p>
                    完整的部署指南和性能优化建议，
                    快速上手生产级LLM推理服务。
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

