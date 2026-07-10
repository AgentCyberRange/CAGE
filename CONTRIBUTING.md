# Contributing to Cage

Thanks for your interest in improving Cage. This guide covers the essentials;
the deeper architecture docs live under [`docs/`](docs/).

## What Cage is

Cage is **infrastructure** for evaluating already-installed AI coding agents: it
runs each agent in its own Docker container against a pluggable benchmark,
intercepts every LLM call through an in-container proxy, snapshots state, and
scores the trial. Everything domain-specific (benchmarks, prompts, datasets)
lives outside the framework, under `examples/`.

Before changing anything, read [`CLAUDE.md`](CLAUDE.md) — it explains the
three-layer model (framework / benchmark / user-config) and the invariants that
keep the layers separated. The single most important rule: **the framework
(`cage/`) never knows a benchmark name.**

## Development setup

```bash
pip install -e .
cp config/models.example.yml config/models.yml   # then fill in your endpoints
```

`config/models.yml` is git-ignored — it holds your private model endpoints and
keys. Never commit it; only `config/models.example.yml` (placeholders) is tracked.

## Making a change

- **Adding a benchmark** → new `examples/<name>/` with a `benchmark.py`
  (`Benchmark` + `Scorer` subclasses) and a `default_<name>.yml`. No edits to
  `cage/`. See [`docs/writing-benchmarks/`](docs/writing-benchmarks/).
- **Adding an agent** → new `cage/agents/<name>/` package + a `docker/`
  image + one import line in `cage/agents/__init__.py`. See
  [`docs/agent-cage-managed.md`](docs/agent-cage-managed.md).
- **Touching the framework** → apply the layer test in `CLAUDE.md` first.

Keep agent-facing text (prompts, READMEs, scoring scripts) as plain files under
`examples/<name>/prompts/`, not inline Python strings, so they can be audited and
diffed against upstream sources.

## Before opening a PR

- Run the test suite: `pytest tests/ -q`
- Lint: `ruff check cage tests` and `lint-imports` (the import-linter contracts
  enforce the layer boundaries).
- Don't commit run artifacts (`.cage_runs/`), caches, local model configs, or
  absolute/host-specific paths.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
