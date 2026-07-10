# CLAUDE.md

Principles and decision rules for working in this repo. Operational details
(commands, lifecycle, config schemas, tests) live in `README.md` and
`docs/repo-architecture.md`.

## What Cage Is

Cage is an evaluation framework for **already-installed AI coding agents**
(Claude Code, Codex, Hermes, …). It runs each agent inside its own Docker
container against pluggable benchmarks, intercepts every LLM call through an
in-container proxy, snapshots state, and scores the trial. The user supplies
*what to evaluate*, not *how to run it*.

Cage is **infrastructure**. It is not a benchmark, not an agent, not a model.
Everything domain-specific lives outside the framework.

---

## Three Layers (the design gem)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3 — User                                                 │
│  project.yml + config/models.yml.  Pure config.  No code.       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Benchmark (examples/<name>/)                         │
│  Concrete Benchmark + Scorer subclasses.  Prompts.  Datasets.   │
│  Adapts framework abstractions to a specific evaluation domain. │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — Framework (cage/)                                    │
│  ABCs, container/proxy lifecycle, scoring runtime, hooks,       │
│  exception types, debug interface.  Knows zero benchmark names. │
└─────────────────────────────────────────────────────────────────┘
```

**Layer 1 — Framework (`cage/`)**. Holds ABCs and runtime plumbing only:
`AgentType`, `Benchmark`, `Scorer`, container lifecycle, in-container proxy,
state snapshots, target_server, scoring runtime, hooks, exceptions. **Zero
benchmark names** anywhere in `cage/`. A grep of `cage/` for `"nyu"` /
`"cvebench"` / `"agent_pentest_bench"` should return nothing.

**Layer 2 — Benchmark (`examples/<name>/`)**. Self-contained package with
`benchmark.py` (Benchmark + Scorer subclass), `prompts/`, `datasets/` (git
submodule), and a small example YAML. Adapts framework abstractions to a
domain — chooses what samples look like, what's in the workspace, what
counts as a win. Must not reach into framework internals beyond the ABCs.

**Layer 3 — User (experiment YAML + repo config)**. The whole user surface.
**No Python.** Picks models, agents, prompts, sample limits, hooks, target
runtime. Model endpoints live in the repo-level local `config/models.yml`, which
is created from the committed `config/models.example.yml`; they do not live in
benchmark-local registries. If a user can't get their work done by editing YAML
and writing a `Benchmark` plus a `Scorer`, the framework has a missing
abstraction.

---

## The Layer Test (apply on every change)

Before writing or moving code, ask three questions in order:

1. **Is this true for every benchmark we have or might add?** → Layer 1.
2. **Is this specific to one benchmark domain (CTF / CVE / red-team / coding)?** → Layer 2.
3. **Is this a per-run knob a user might want to flip without code changes?** → Layer 3 (yaml).

Smells:
- A constant named after a benchmark in `cage/` → Layer 2 leaked into Layer 1.
- A new field on a `cage/config/sections.py` dataclass for a single benchmark → Layer 2 in Layer 3.
- A `Benchmark` subclass importing from `cage/experiment/` or `cage/sandbox/` directly → Layer 1 abstraction missing.
- A `project.yml` listing 30 hardcoded sample IDs → Layer 3 doing Layer 2's job.

When in doubt, push **outwards** (towards higher layer numbers). Pulling
inwards is what's expensive to undo.

---

## Core Invariants

These are non-negotiable; break one and downstream things stop working:

- **The framework never knows a benchmark name.** New benchmark → new
  `examples/<name>/` directory. No edits to `cage/`.
- **Stateful vs. stateless is a property of the agent, not the benchmark.**
  An agent is stateful iff `shared_paths` is non-empty.
- **`proxy/sidecar.py` (the in-container proxy) has one runtime dependency: `httpx`.** Don't add more
  — every dep ships in every agent container.
- **One scorer runs at three call sites with the same `ScoringContext`**
  (inline post-trial, mid-trial live monitor, offline `cage score`). If a
  scorer works in one and not another, the abstraction is the bug.
- **Don't mutate samples in `iter_samples()`.** The orchestrator injects
  target info before `prepare_trial`/`build_prompt`. Sample mutation is
  orchestrator-owned.
- **`shared_paths` must live outside the workspace dir.** `reset_directory`
  wipes the workspace; shared paths inside it disappear silently.
- **Yaml first, code second.** Adding a user-facing knob? Prefer extending
  `project.yml` over a Python flag. Env vars and hardcoded paths are smells.

---

## When Working in This Repo

- **Layer test first.** Before adding anything to `cage/`, prove it's universal.
- **Touching the trial lifecycle** (`experiment/engine/conductor.py` +
  `trial_runner.py`) usually means touching `proxy/host.py`, `sandbox/state.py`,
  and at least one example benchmark — keep them in sync.
- **Changing `Benchmark` / `Scorer` / `AgentType` ABCs** ripples to every
  concrete subclass. Update them together.
- **Proxy edits**: prefer `proxy/sidecar.py` (in-container, httpx-only) over
  `proxy/host.py` (host-side). They are deployed separately — host writes
  config JSON, container reads it.
- **Network changes** (Docker networks, host gateway, proxy port) ripple
  through `sandbox/containers.py` (Container), `proxy/host.py`,
  `target/server/network_admin.py`, and the prompts that tell the agent how to
  reach the upstream. Check all.
- **Adding a benchmark**: new `examples/<name>/`, new `benchmark.py`, new
  `prompts/` if needed, optionally a `cage-org/<dataset-repo>` submodule.
  If you find yourself adding a benchmark name to `cage/`, stop and find
  the missing abstraction.
- **Adding an agent**: new `cage/agents/<name>/` package + a
  `docker/<name>/` dir holding its `Dockerfile` (+ any `<variant>.Dockerfile`)
  + an import line in `cage/agents/__init__.py::register_builtin_agents()`. No
  benchmark changes should be needed.

---

## Reference

- [`README.md`](README.md) — quick start, project.yml / config/models.yml schemas,
  CLI command table, "Writing a Benchmark" / "Writing a New Agent Type"
  tutorials, run-output layout.
- [`docs/repo-architecture.md`](docs/repo-architecture.md) — durable
  architecture map: trial lifecycle, configuration model, container/proxy
  runtime, state semantics, target runtime, live-check & verdicts, artifact
  layout, synchronization checklist.
- [`docs/targets-check.md`](docs/targets-check.md) — `cage targets-check`
  guide (parallel target smoke test).
- [`docs/agent-serve-mode.md`](docs/agent-serve-mode.md) —
  benchmark-only / serve (PULL) mode: an external agent drives `cage benchmark
  serve` targets itself (list → launch → prompt → submit → close). The other
  agent-integration path vs [`agent-cage-managed`](docs/agent-cage-managed.md).
- [`docs/serve-external-audience.md`](docs/serve-external-audience.md) —
  the serve HTTP contract: endpoints, two-audience port binding, external token.
