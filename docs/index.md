---
layout: home

hero:
  name: CAGE
  text: Cybersecurity Agent Gym & Evaluation
  tagline: Drop an installed AI coding agent into an isolated container, run it against security benchmarks, intercept every model call, and score the result — reproducibly and replayably.
  actions:
    - theme: brand
      text: Quick Start
      link: /getting-started/
    - theme: alt
      text: How a Run Works
      link: /how-a-run-works
    - theme: alt
      text: View on GitHub
      link: https://github.com/AgentCyberRange/CAGE

features:
  - icon: 🧠
    title: Installed agents, unchanged
    details: Evaluate Claude Code, Codex, Qwen, Kimi, Hermes, or your own CLI. CAGE drives the real agent binary inside a container — it does not reimplement the agent.
  - icon: 🎯
    title: Security benchmarks with live targets
    details: Web-exploitation apps, multi-host post-exploitation ranges, CTF, and CVE reproduction. Each trial gets its own Dockerized target stack.
  - icon: 📊
    title: Every run is an audit trail
    details: Watch active trials in the inspector, then replay prompts, model traffic, scores, and pre/post state snapshots straight from disk.
---

## One command to learn

In CAGE you mostly type one command: **`cage run`**. Everything else is a *slice*
of that command, or a *post-processing* of what it produced. A single run is:

```text
cage run  =  Framework ( Benchmark , Config )
             └ Layer 1 ┘ └Layer 2 ┘ └Layer 3┘
              the engine   what to     this run's
              (fixed)      evaluate    knobs
```

- **Layer 1 — Framework (`cage/`)** owns the *mechanism* of a run: container,
  proxy, target, scoring, snapshots, resume. It never knows a benchmark name.
- **Layer 2 — Benchmark (`examples/<name>/`)** supplies *what* is evaluated —
  samples, prompts, targets, scorer.
- **Layer 3 — You** supply *how this run goes* — an experiment YAML plus CLI flags.

Hold onto that equation: every command and every YAML field in these docs hangs
off it. The two ideas it encodes — the run equation, and "every command is a
slice of `cage run`" — are unpacked in [How a Run Works](/how-a-run-works) and
[The CLI](/cli-design). If you just want to run something, skip ahead to
[Quick Start](/getting-started/); the design pages keep until you need them.

## Pick your path

| I want to… | Start here |
|---|---|
| Run my first inspected trial | [Quick Start](/getting-started/) |
| Scale up, resume, and score runs | [Running Experiments](/running-experiments/) |
| Understand what a run actually does | [How a Run Works](/how-a-run-works) |
| See how every command maps to `cage run` | [The CLI](/cli-design) |
| Get benchmark datasets | [Datasets](/datasets) |
| Add or maintain a benchmark | [Writing Benchmarks](/writing-benchmarks/) |
| Wrap a new agent CLI | [Adding an Agent](/agent-cage-managed) |
| Operate long runs on a shared host | [Operations](/operations/) |

## Benchmarks

CAGE ships no benchmark data — it is infrastructure. Each benchmark is a
self-contained package with its own runbook:

- [AgentPentestBench](https://github.com/AgentCyberRange/CAGE/tree/main/examples/agent_pentest_bench)
  — WebExploitBench (web targets) and PostExploitBench (multi-host ranges)
- [CVEBench](https://github.com/AgentCyberRange/CAGE/tree/main/examples/cvebench)
- [NYU CTF](https://github.com/AgentCyberRange/CAGE/tree/main/examples/nyuctfbench)
- [AutoPenBench](https://github.com/AgentCyberRange/CAGE/tree/main/examples/autopenbench)
- [HackWorld](https://github.com/AgentCyberRange/CAGE/tree/main/examples/hackworld)
- [StrongREJECT](https://github.com/AgentCyberRange/CAGE/tree/main/examples/strongreject)

## Run this site locally

```bash
cd docs
npm install
npm run dev
```

Config lives in `.vitepress/config.mts`; deployment is covered in
[Docs Deployment](/deployment).
