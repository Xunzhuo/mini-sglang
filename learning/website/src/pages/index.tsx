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
      <div className={styles.bannerImageContainer}>
        <img
          src="/img/banner.png"
          alt="Banner"
          className={styles.bannerImage}
        />
      </div>
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
      title={`${siteConfig.title}`}
      description="从零学习LLM">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">系统化学习</Heading>
                  <p>
                    从神经网络到大模型推理，完整的 LLM 学习路线
                  </p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">深入浅出</Heading>
                  <p>
                    从基础知识出发，最终从 LLM 推理实加深理解
                  </p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md">
                  <Heading as="h3">实战导向</Heading>
                  <p>
                    基于 SGLang 生产级推理框架，进行源码级学习
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

