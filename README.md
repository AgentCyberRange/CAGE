# CAGE

**CAGE (Cybersecurity Agent Gym & Evaluation)** is an evaluation framework for
**already-installed** AI coding agents — Claude Code, Codex, Qwen, Kimi, or your
own. It runs each agent inside its own Docker container against a pluggable
benchmark, **intercepts every LLM call** through an in-container proxy, snapshots
state before and after, and scores the trial. You supply *what to evaluate*;
CAGE owns *how it runs*.

CAGE is **infrastructure** — not a benchmark, not an agent.
Everything domain-specific (samples, prompts, live targets, scoring) lives in a
benchmark package outside the framework.

Two ways to use it: let CAGE **run and record** your agent (the default above),
or run **benchmark-only** — CAGE serves the isolated targets and your own
external agent drives them over an API (`list → launch → attack → submit →
close`). See [Evaluating Your Own Agent](#evaluating-your-own-agent).

## Setup

### 0. Requirements

CAGE runs trials in Docker and Docker Compose, and uses Git LFS for benchmark assets. On
Ubuntu/Debian, install the basic dependencies first:

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Install CAGE

```bash
git clone https://github.com/AgentCyberRange/CAGE.git
cd CAGE
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure a Model
Copy the example model registry:
```
cp config/models.example.yml config/models.yml
cage model list
```

Register a GPT model:
```bash
export OPENAI_API_KEY=...
cage model set gpt-5.5 \
  --provider openai \
  --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 \
  --api-key '${OPENAI_API_KEY}'
```

or a Claude model:
```bash
export ANTHROPIC_API_KEY=...
cage model set claude-opus \
  --provider anthropic \
  --model claude-opus-4-7 \
  --endpoint https://api.anthropic.com \
  --api-key '${ANTHROPIC_API_KEY}'
```

Full model-registry details are in [models.md](docs/models.md).

#### 2.1 Set the reasoning effort

Reasoning effort (and any other inference knob) belongs to the **installed agent
CLI**, not to CAGE — the CLI decides what it accepts and what the levels mean.
CAGE's job is only to hand your choice to that CLI, and it gives you exactly two
channels for doing so:

- **`session_args`** (in `project.yml`) — appended verbatim to the agent's
  launch command. Use it for controls the CLI reads as command-line flags. For
  example, Codex takes reasoning effort as a `-c` config override:

  ```yaml
  agents:
    - id: codex
      kind: codex
      models: [gpt-5.5]        # or codex-chatgpt for ChatGPT/OAuth
      session_args:
        - -c
        - model_reasoning_effort=high
  ```

- **`agent_model_names`** (in `config/models.yml`) — the model string a specific
  agent receives. Use it for CLIs that select effort through the model name. For
  example, Claude Code takes it as a suffix (and only Claude Code sees it):

  ```yaml
  # config/models.yml
  claude-opus-sub:
    provider: anthropic
    model: claude-opus-4-7
    agent_model_names:
      claude_code: claude-opus-4-7[xhigh]
    auth_source: ${CAGE_CLAUDE_AUTH_SOURCE}
  ```

Which flags or model-name forms are valid — and what effort levels exist — is
defined by each agent, so consult that agent's own documentation (Codex, Claude
Code, …); CAGE only forwards what you put in these two fields. The fields
themselves are documented in the
[project.yml reference](docs/reference/project-yml.md) and
[models.md](docs/models.md).

### 3. Prepare Targets

CAGE ships two AgentPentestBench datasets as git submodules. The submodules
include a small bundled subset for smoke tests. The full datasets can be fetched
separately from Hugging Face.

| Benchmark | GitHub (subset) | Hugging Face (full set) |
|---|---|---|
| WebExploitBench | [WebExploitBench (comfyui, dataease, prestashop)](https://github.com/AgentCyberRange/WebExploitBench) | [datasets/WebExploitBench](https://huggingface.co/datasets/AgentCyberRange/WebExploitBench) |
| PostExploitBench | [PostExploitBench (range-4, range-6)](https://github.com/AgentCyberRange/PostExploitBench) | [datasets/PostExploitBench](https://huggingface.co/datasets/AgentCyberRange/PostExploitBench) |

#### 3.1 Initialize Bundled Targets

Make sure Git LFS is enabled before pulling the submodules. Otherwise, large
target assets such as jars and archives may be checked out as LFS pointer files.

```bash
git lfs install
git submodule update --init --recursive \
  examples/agent_pentest_bench/datasets/web_exploit_bench \
  examples/agent_pentest_bench/datasets/post_exploit_bench
# fetch the actual LFS binaries (jars/zips) — submodule update skips these:
git -C examples/agent_pentest_bench/datasets/web_exploit_bench lfs pull
git -C examples/agent_pentest_bench/datasets/post_exploit_bench lfs pull
```

#### 3.2 Optional: Fetch the full datasets

Install the Hugging Face CLI first if you have not already:
```bash
uv pip install huggingface_hub
```

Then use `scripts/fetch` to fetch the full WebExploitBench and PostExploitBench datasets:
```bash
hf auth login
examples/agent_pentest_bench/datasets/web_exploit_bench/scripts/fetch
examples/agent_pentest_bench/datasets/post_exploit_bench/scripts/fetch
```

The fetch scripts only add data on top of the existing checkout and are safe to
re-run.

### 4. Build Agents and Targets

Build the agent images:
```
cage agent build --agent codex --variant pentestenv
cage agent build --agent claude_code --variant pentestenv
```

Prebuild all benchmark targets:
```
cage benchmark build web_exploit_bench --max-concurrent 4
cage benchmark build post_exploit_bench --max-concurrent 4
```

You can also build a single benchmark sample, which is useful for smoke tests,
or for retrying a specific target if it fails during a batch build:
```
cage benchmark build web_exploit_bench --sample pb-comfyui
cage benchmark build post_exploit_bench --sample pb-postexp-range-4
```
> Note: Building agent images and benchmark targets can take a while, especially on the first run, since Docker images and target assets may need to be downloaded and built.
> When many targets are built concurrently, **a small number of samples may fail** due to transient Docker, network, or resource issues. In
> that case, rerun cage benchmark build with **--sample <sample-id>** to **rebuild the failed sample only**.

### 5. Run Evaluations

`web_exploit_bench` scores some vulnerabilities with an LLM judge. Register its
default judge `deepseek-v4-pro` the same way as the models above:

```bash
export DEEPSEEK_API_KEY=...
cage model set deepseek-v4-pro \
  --provider anthropic \
  --model deepseek-v4-pro \
  --endpoint https://api.deepseek.com/anthropic \
  --api-key '${DEEPSEEK_API_KEY}'
```

Default full runs use the benchmark config as-is:
```bash
cage run web_exploit_bench --agent codex --model gpt-5.5
cage run post_exploit_bench --agent codex --model gpt-5.5

# Evaluate a different agent/model pair:
cage run web_exploit_bench --agent claude_code --model claude-opus
```

Single-sample smoke runs. Sample IDs are `pb-<web-target>` and
`pb-postexp-<range>`, so these two use bundled targets and work without the
Hugging Face fetch:
```bash
# Default configs already set benchmark-level concurrency; pass `--max-concurrent N` only to lower the selected agent/model cap.
cage run web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id web-smoke-001

cage run post_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-postexp-range-4 \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id post-smoke-001
```

Prompt levels control how much task information the agent receives. For web tasks, hints may reveal vulnerability location or type. For
post-exploitation tasks, hints may reveal topology or services:
* `l0`: no hints
* `l1`: partial hints
* `l2`: stronger hints

### 6. Inspect, resume, and re-score runs
By default, cage run starts the browser inspector automatically. After the run completes, inspect the results in the browser.

<img src="./docs/assets/inspector.png" width="600">

To continue a named run, pass the same --run-id with --resume:
```
cage run web_exploit_bench --run-id web-smoke-001 --resume
```

To **re-score** a finished run without re-running the agent, use `cage score`.
It reuses the evidence already saved from the run, and scores trials in parallel
when you pass `--max-concurrent`:

```bash
cage score web_exploit_bench --run-id web-smoke-001 --max-concurrent 8
```

By default it re-judges with the benchmark's judge model (`deepseek-v4-pro`). To
judge with a different model, set `eval.benchmark.judge` to any model registered
in `config/models.yml`.

## How a run works

You mostly type one command, `cage run`. A run is the framework executing a
benchmark under your config:

```text
cage run  =  Framework ( Benchmark , Config )
             └ Layer 1 ┘ └Layer 2 ┘ └Layer 3┘
              the engine   what to     this run's
              (fixed)      evaluate    knobs
```

- **Layer 1 — Framework (`cage/`)** owns the run mechanism: container, proxy,
  target, scoring, resume. It never knows a benchmark name.
- **Layer 2 — Benchmark (`examples/<name>/`)** supplies what is evaluated:
  samples, prompts, targets, scorer.
- **Layer 3 — You** supply how this run goes: an experiment YAML plus CLI flags.

Every other command is a slice of `cage run`, and every YAML field parameterizes
one of its steps. The full lifecycle is in
[How a Run Works](docs/how-a-run-works.md).


## Evaluating Your Own Agent

Two ways to put **your own agent** in front of a CAGE benchmark, differing in one
thing: **does CAGE run and record the agent, or just serve the targets?** **Most
users should start with serve mode**; reach for CAGE-managed only when your agent
has no observability of its own and you need CAGE to record the trajectory for you.

- **Benchmark-only / serve** — `cage benchmark serve <benchmark>`
  exposes an isolated, launchable target range, and **your agent drives the loop
  itself** over an HTTP API / zero-dep SDK (`list → launch → attack → submit →
  close`). **Zero integration** — any language, any framework, agent stays a black
  box. CAGE never sees the LLM calls, so there is **no trajectory**, only the final
  score — which is exactly right when your agent already keeps its own logs and UI.
  Ideal for a mature agent or framework, external teams, self-serve leaderboards,
  or agents you can't containerize into CAGE. → [Benchmark-Only (Serve) Mode](docs/agent-serve-mode.md)
  ([中文](docs/agent-serve-mode-CN.md)).

- **CAGE-managed** — for agents with **no frontend or observability of their own**
  — think terminal tools like **Claude Code** or **Codex**. You plug your agent
  into CAGE (a Dockerfile + an `agent.yml` manifest, no framework code) and
  `cage run` owns the whole trial: it builds the container, **intercepts every LLM
  call** through the in-container proxy, snapshots state, and scores — giving you
  the **full step-by-step trajectory** in the inspector and apples-to-apples
  comparability you'd otherwise lack. → [Adding an Agent to CAGE](docs/agent-cage-managed.md)
  ([中文](docs/agent-cage-managed-CN.md)).

| | Benchmark-only / serve *(recommended)* | CAGE-managed |
|---|---|---|
| Who runs the agent | You (external process) | CAGE (`cage run`) |
| Integration cost | None | Dockerfile + `agent.yml` + proxy convention |
| Trajectory (every LLM / tool call) | Not captured — score/verdict only | **Captured** — full inspector view |
| Reproducible / resumable / comparable | Weaker — you own the runtime | Yes |
| Best for | A mature agent / framework with its own logging + UI; external / black-box agents; leaderboards | A terminal tool (Claude Code, Codex, …) with no UI of its own that you want CAGE to record end-to-end |

**Quick check:** if your agent already has its own mature logging and a UI for
inspecting runs, use **serve mode** — CAGE-managed's main value is recording the
trajectory in CAGE's inspector, so if you already have that, its container/proxy
integration is just overhead. Reach for **CAGE-managed** when your agent is a
terminal tool with no observability of its own and you want CAGE to capture and
standardize the trajectory for you.


## Included Benchmarks

CAGE is benchmark-agnostic; these security benchmarks ship as example packages:

| Benchmark | What it evaluates |
|---|---|
| [AgentPentestBench](examples/agent_pentest_bench) | Web exploitation (WebExploitBench) and multi-host post-exploitation ranges (PostExploitBench). The release-facing example, with full datasets on Hugging Face. |
| [CVEBench](examples/cvebench) | Whether an agent can exploit a known CVE in a live target. |
| [NYU CTF](examples/nyuctfbench) | CTF-style capture-the-flag tasks. |
| [AutoPenBench](examples/autopenbench) | Automated penetration-testing tasks. |
| [HackWorld](examples/hackworld) | Web CTF tasks. |
| [StrongREJECT](examples/strongreject) | Safety / refusal behavior (no live target). |

Adding your own is a new `examples/<name>/` package — see
[Writing Benchmarks](docs/writing-benchmarks/README.md). The framework (`cage/`)
never changes.

## Documentation

- [Quick Start](docs/getting-started/README.md) — fresh checkout to one inspected trial.
- [How a Run Works](docs/how-a-run-works.md) — the run lifecycle and runtime internals.
- [The CLI](docs/cli-design.md) — every command as a slice of `cage run`.
- [Running Experiments](docs/running-experiments/README.md) and [Operations](docs/operations/README.md) — scaling, resume, scoring, cleanup.
- [Adding an Agent](docs/agent-cage-managed.md) — evaluate **your own agent**: a LangGraph/LangChain graph (or any program) via an `agent.yml` manifest with **no framework code**, or a built-in Python `AgentType`.
  - [LangGraph / LangChain Agent](docs/langgraph-langchain-agent.md) — focused, hands-on walkthrough for the manifest path, built on the real `references/agentic-poc` example (manifest, proxy wiring, free node-aware tracing).
- [Benchmark-Only (Serve) Mode](docs/agent-serve-mode.md) ([中文](docs/agent-serve-mode-CN.md)) — the other integration path: `cage benchmark serve` exposes the target range and **your external agent drives it** over the PULL API / SDK. Zero integration, but no trajectory. Full HTTP contract in [Serve External Audience](docs/serve-external-audience.md).
- [Writing Benchmarks](docs/writing-benchmarks/README.md) and [Contributing](docs/developing-cage/README.md) — extend CAGE.
