import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'CAGE',
  description: 'Cybersecurity Agent Gym & Evaluation',
  base: process.env.DOCS_BASE || (process.env.GITHUB_ACTIONS ? '/CAGE/' : '/'),
  cleanUrls: true,
  srcExclude: ['plans/**'],
  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: 'Quick Start', link: '/getting-started/' },
      { text: 'How It Works', link: '/how-a-run-works' },
      { text: 'CLI', link: '/cli-design' },
      { text: 'Reference', link: '/reference/cli' },
      { text: 'GitHub', link: 'https://github.com/AgentCyberRange/CAGE' }
    ],
    sidebar: [
      {
        text: 'Start',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Quick Start', link: '/getting-started/' }
        ]
      },
      {
        text: 'Run Experiments',
        items: [
          { text: 'Running Experiments', link: '/running-experiments/' },
          { text: 'Models', link: '/models' },
          { text: 'Datasets', link: '/datasets' }
        ]
      },
      {
        text: 'Understand CAGE',
        items: [
          { text: 'How a Run Works', link: '/how-a-run-works' },
          { text: 'The CLI', link: '/cli-design' },
          { text: 'Architecture & Principles', link: '/repo-architecture' }
        ]
      },
      {
        text: 'Extend CAGE',
        items: [
          { text: 'Contributing', link: '/developing-cage/' },
          { text: 'Writing Benchmarks', link: '/writing-benchmarks/' },
          { text: 'Adding an Agent', link: '/agent-cage-managed' },
          { text: 'Benchmark-Only (Serve) Mode', link: '/agent-serve-mode' }
        ]
      },
      {
        text: 'Operate',
        items: [
          { text: 'Operations', link: '/operations/' },
          { text: 'Resource GC (cage gc)', link: '/cage-gc' },
          { text: 'External Targets', link: '/serve-external-audience' },
          { text: 'Target Build Debugging', link: '/targets-check' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'CLI', link: '/reference/cli' },
          { text: 'Experiment YAML', link: '/reference/project-yml' },
          { text: 'Core Classes', link: '/reference/classes' },
          { text: 'Docs Deployment', link: '/deployment' }
        ]
      },
      {
        text: 'Benchmarks',
        items: [
          { text: 'AgentPentestBench', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/agent_pentest_bench' },
          { text: 'CVEBench', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/cvebench' },
          { text: 'NYU CTF', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/nyuctfbench' },
          { text: 'AutoPenBench', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/autopenbench' },
          { text: 'HackWorld', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/hackworld' },
          { text: 'StrongREJECT', link: 'https://github.com/AgentCyberRange/CAGE/tree/main/examples/strongreject' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AgentCyberRange/CAGE' }
    ],
    search: {
      provider: 'local'
    }
  }
})
