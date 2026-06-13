# Contributing to CAGE

This page is for framework developers changing code under `cage/`, `docker/`,
`scripts/`, shared tests, or the web inspector. If you are only running
experiments, start with [Running Experiments](../running-experiments/).

For how the system fits together — the three layers, trial lifecycle, module
map, proxy, target runtime, resume, and the inspector — read
[Architecture](../repo-architecture.md) first. This page is the contributor
workflow layered on top of it; it does not repeat the architecture.

## The One Rule

CAGE is infrastructure. The framework (`cage/`) must never know a benchmark
name. Anything benchmark-specific lives under `examples/<benchmark>/`; anything
an operator flips per run lives in `project.yml` / `config/models.yml`. Before
adding anything to `cage/`, apply the layer test in
[`CLAUDE.md`](https://github.com/AgentCyberRange/CAGE/blob/main/CLAUDE.md) and
see [Architecture › Configuration Model](../repo-architecture.md#configuration-model)
for the full three-layer boundary.

## Where To Make A Change

| I want to... | Start here |
|---|---|
| Add an agent CLI | [Adding an Agent](../adding-a-new-agent.md) |
| Add or maintain a benchmark | [Writing Benchmarks](../writing-benchmarks/) |
| Change the trial lifecycle | [Architecture › Runtime Flow](../repo-architecture.md#runtime-flow) — keep `conductor.py` and `trial_runner.py` in sync |
| Change model egress / budgets | [Architecture › Proxy Runtime](../repo-architecture.md#proxy-runtime) — prefer `proxy/sidecar.py` (in-container, httpx-only) over `proxy/host.py` |
| Change target launch or networking | [Architecture › Target Runtime](../repo-architecture.md#target-runtime) |
| Change resume / termination behavior | [Architecture › Termination Classification](../repo-architecture.md#termination-classification) |
| Change the inspector | Put meaning in `Benchmark.build_dashboard()`; render generically in `cage/web/` |

Changing a `Benchmark` / `Scorer` / `AgentType` ABC ripples to every concrete
subclass — update them together.

## Testing Strategy

Use focused tests first:

```bash
uv run pytest tests/test_core.py -q
uv run pytest tests/test_orchestrator.py -q
uv run pytest tests/test_proxy.py -q
uv run pytest tests/test_web_live.py -q
```

Run lint on touched files:

```bash
uv run ruff check --select F,E9 <files>
git diff --check -- <files>
```

`lint-imports` enforces the layer boundaries via the import-linter contracts.

Docker-dependent behavior should have either a unit test around command/config
construction, or a small smoke command documented in the relevant benchmark
README.

## Rules Of Thumb

- Keep benchmark semantics out of framework code.
- Keep project-run choices out of benchmark code unless they are benchmark
  constructor parameters.
- Preserve `.cage_runs` artifacts as audit evidence.
- Make CLI output actionable for long-running operations.
- Use labels and run ids for Docker cleanup.
- Prefer benchmark-authored dashboards over generic summaries when meaning
  matters.

## Related Docs

- [Architecture](../repo-architecture.md) — the full system map
- [Writing Benchmarks](../writing-benchmarks/) and [Adding an Agent](../adding-a-new-agent.md)
- [Core Classes](../reference/classes.md) — the ABCs you will subclass
- [Operations](../operations/) — running CAGE at scale
