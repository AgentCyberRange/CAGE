# CAGE

**CAGE (Cybersecurity Agent Gym & Evaluation)** is an evaluation framework for
**already-installed** AI coding agents — Claude Code, Codex, Qwen, Kimi, or your
own. It runs each agent inside its own Docker container against a pluggable
benchmark, **intercepts every LLM call** through an in-container proxy, snapshots
state before and after, and scores the trial. You supply *what to evaluate*;
CAGE owns *how it runs*.

CAGE is **infrastructure** — not a benchmark, not an agent, not a model.
Everything domain-specific (samples, prompts, live targets, scoring) lives in a
benchmark package outside the framework.

> 📖 **Documentation:** [`docs/`](docs/) —
> [Quick Start](docs/getting-started/README.md) ·
> [How a Run Works](docs/how-a-run-works.md) ·
> [The CLI](docs/cli-design.md) ·
> [Writing Benchmarks](docs/writing-benchmarks/README.md)

## What you get

- **Installed agents, unchanged** — CAGE drives the real agent CLI; it does not
  reimplement the agent.
- **Every model call observed** — an in-container proxy records and budgets each
  request (rounds / tokens / cost) into `proxy.jsonl`, the source of truth for
  what the model was asked and answered.
- **Isolated live targets** — each trial gets its own Dockerized target stack
  (web apps, multi-host ranges, CTF, CVE).
- **Reproducible, auditable runs** — pre/post state snapshots and one scorer that
  runs identically inline, live, and offline, all under a `.cage_runs` audit
  trail.
- **Live inspection** — watch active trials and replay prompts, model traffic,
  scores, and snapshots in the browser inspector.

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

## Benchmarks

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

---

## Requirements

Requires Docker (Engine + Compose v2), `uv`, `git`, and **`git-lfs`**. git-lfs
is mandatory: the benchmark submodules ship LFS-tracked binaries (`.jar`/`.zip`/
`.tar.gz` …), and without it they check out as text *pointer* files instead of
the real archives. Pulling the full datasets also needs the Hugging Face CLI
(`pip install -U huggingface_hub`).

## Install

```bash
git clone https://github.com/AgentCyberRange/CAGE.git
cd CAGE
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configure A Model

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
cage model set gpt-5.5 \
  --provider openai \
  --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 \
  --api-key '${OPENAI_API_KEY}'
```

…or a Claude model for the Claude Code agent:

```bash
export ANTHROPIC_API_KEY=...
cage model set claude-opus \
  --provider anthropic \
  --model claude-opus-4-7 \
  --endpoint https://api.anthropic.com \
  --api-key '${ANTHROPIC_API_KEY}'
```

Full model-registry details are in [Models](docs/models.md).

## Prepare Targets

CAGE ships two security benchmarks as git submodules. Each is a public
**teaser**: the submodule carries the tooling plus a few ready-to-run targets,
and the full dataset lives on Hugging Face.

| Benchmark | GitHub | Hugging Face (full set) |
|---|---|---|
| WebExploitBench | [AgentCyberRange/WebExploitBench](https://github.com/AgentCyberRange/WebExploitBench) | [datasets/AgentCyberRange/WebExploitBench](https://huggingface.co/datasets/AgentCyberRange/WebExploitBench) |
| PostExploitBench | [AgentCyberRange/PostExploitBench](https://github.com/AgentCyberRange/PostExploitBench) | [datasets/AgentCyberRange/PostExploitBench](https://huggingface.co/datasets/AgentCyberRange/PostExploitBench) |

Initialize the submodules — this gives the bundled subset (web
`comfyui`/`dataease`/`prestashop`, post `range-4`/`range-6`):

> ⚠️ **Install `git-lfs` first**, otherwise the LFS binaries inside the teaser
> come down as pointer files and targets won't build. Install it
> (`sudo apt install git-lfs` / `brew install git-lfs`) and run `git lfs install`
> once. `git submodule update` does **not** fetch LFS content on its own, so
> `git lfs pull` inside each submodule afterwards.

```bash
git lfs install                 # one-time, enables the LFS smudge filter
git submodule update --init --recursive \
  examples/agent_pentest_bench/datasets/web_exploit_bench \
  examples/agent_pentest_bench/datasets/post_exploit_bench
# fetch the actual LFS binaries (jars/zips) — submodule update skips these:
git -C examples/agent_pentest_bench/datasets/web_exploit_bench  lfs pull
git -C examples/agent_pentest_bench/datasets/post_exploit_bench lfs pull
```

If you want to pull the **complete** datasets from Hugging Face (15 web targets, 8 ranges), use `scripts/fetch`.
`scripts/fetch` only adds data on top of the checkout and is safe to re-run; run
`hf auth login` first if a dataset is gated (`pip install -U huggingface_hub`
provides the `hf` CLI):

```bash
examples/agent_pentest_bench/datasets/web_exploit_bench/scripts/fetch
examples/agent_pentest_bench/datasets/post_exploit_bench/scripts/fetch
```

To pull just one target/range instead of the full set, call the Hugging Face
CLI directly with an `--include` filter:

```bash
hf download AgentCyberRange/WebExploitBench --repo-type dataset \
  --local-dir examples/agent_pentest_bench/datasets/web_exploit_bench \
  --include 'siyucms/*'
hf download AgentCyberRange/PostExploitBench --repo-type dataset \
  --local-dir examples/agent_pentest_bench/datasets/post_exploit_bench \
  --include 'range-1/*'
```

Build the agent image, then prebuild the targets:

```bash
cage agent build --agent codex --variant pentestenv
cage agent build --agent claude_code --variant pentestenv
cage benchmark build web_exploit_bench --max-concurrent 4
cage benchmark build post_exploit_bench --max-concurrent 4
```

## Run

Default full runs use the benchmark config as-is:

```bash
cage run web_exploit_bench --agent codex --model gpt-5.5
cage run post_exploit_bench --agent codex --model gpt-5.5
# swap --agent/--model to evaluate another agent, e.g. Claude Code:
cage run web_exploit_bench --agent claude_code --model claude-opus
```

Single-sample smoke runs. Sample IDs are `pb-<web-target>` and
`pb-postexp-<range>`, so these two use bundled targets and work without the
Hugging Face fetch. Both benchmarks share one `--prompt-level` knob: `l0` gives
no hints, `l1`/`l2` progressively reveal more (vulnerability location/type for
web, network topology/services for post):

```bash
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

Default configs already set benchmark-level concurrency; pass
`--max-concurrent N` only to lower the selected agent/model cap.

Resume a named run:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --resume
```

Inspect runs in the browser, or get the exact current CLI surface:

```bash
cage inspect .cage_runs
cage run --help
cage run web_exploit_bench --help
cage run post_exploit_bench --help
```

## Learn more

- [Quick Start](docs/getting-started/README.md) — fresh checkout to one inspected trial.
- [How a Run Works](docs/how-a-run-works.md) — the run lifecycle and runtime internals.
- [The CLI](docs/cli-design.md) — every command as a slice of `cage run`.
- [Running Experiments](docs/running-experiments/README.md) and [Operations](docs/operations/README.md) — scaling, resume, scoring, cleanup.
- [Writing Benchmarks](docs/writing-benchmarks/README.md) and [Contributing](docs/developing-cage/README.md) — extend CAGE.
