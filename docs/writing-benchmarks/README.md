# Writing Benchmarks

This guide is for adding or maintaining a benchmark under `examples/<name>/`.
The benchmark layer should make task semantics explicit while leaving runtime
mechanics to CAGE.

## Benchmark Directory Shape

Recommended layout:

```text
examples/<benchmark>/
  README.md                  benchmark runbook
  benchmark.py               Benchmark + Scorer implementation
  default_<name>.yml         default runnable experiment config
  prompts/                   prompt templates
  datasets/                  submodule, generated data, or local assets
  target_server_adapter.py   optional target adapter
```

The README is part of the benchmark API. A new user should be able to read it
and answer:

- what the benchmark measures;
- how to get the data;
- what Docker/images are required;
- which example YAML to run;
- which CLI parameters are safe for smoke/full runs;
- where artifacts and scores appear.

## Benchmark Responsibilities

A benchmark owns:

- sample ids and metadata;
- prompt construction;
- target files and setup assumptions;
- workspace preparation;
- success/failure semantics;
- scorer output;
- dashboard meaning.

It should not own:

- Docker container lifecycle for the agent;
- model endpoint wiring;
- proxy implementation;
- resume policy;
- global concurrency;
- web UI templates.

## Implement `Benchmark`

Create a subclass in `benchmark.py`:

```python
from cage.benchmarks import Benchmark


class MyBenchmark(Benchmark):
    name = "my-benchmark"

    def iter_samples(self):
        yield {"id": "sample-001", "content": "Task text"}

    def prepare_trial(self, container, sample, workspace_dir):
        pass

    def build_prompt(self, sample):
        return self.render_strict(TEMPLATE, sample)

    def scorer(self):
        return MyScorer()
```

Required methods:

| Method | Purpose |
|---|---|
| `iter_samples()` | Yield stable sample dictionaries |
| `prepare_trial()` | Copy files or initialize workspace before agent starts |
| `build_prompt()` | Return the final agent prompt |
| `scorer()` | Return the default scorer |

Optional hooks:

| Hook | Use |
|---|---|
| `setup()` | Load data, start benchmark-level services |
| `teardown()` | Clean benchmark-level services |
| `on_agent_finish()` | Runs the instant the agent stops, **before** the scoring gather — materialize agent output to the host (e.g. `container.copy_from` the workspace) so gather and serve-only scoring can read it |
| `on_trial_complete()` | Runs **after** the gather, target still alive — target post-mortem only (docker logs/inspect); too late to feed gather |
| `build_dashboard()` | Emit benchmark-owned dashboard sections |
| live-check hooks | Customize mid-trial success confirmation |

Live scoring evidence is no longer a benchmark hook — it moved to the scorer's
`gather()` (see [Scoring](#scoring)). Order per trial:
`on_agent_finish()` → `Scorer.gather()` → `on_trial_complete()`.

## Samples

Every sample should have a stable `id`:

```python
{
    "id": "pb-comfyui",
    "content": "Find vulnerabilities in the target application.",
    "metadata": {...},
}
```

Good sample ids are:

- stable across dataset revisions;
- useful in CLI commands;
- safe as path segments;
- meaningful in dashboards.

Use `cage run <benchmark_id> --sample <id>` and
`cage benchmark check <benchmark_id> --sample <id>` as
part of your authoring loop after the benchmark is registered.

## Prompts

Use templates under `prompts/` and render them through strict helpers:

```python
prompt = self.render_strict(
    template_src,
    sample,
    expected_substrings=(sample["target_url"],),
)
```

Avoid leaking scorer-only fields into prompts. A common pattern is:

```jsonc
{
  "agent_input": {
    "target_url": "http://web:80"
  },
  "internal": {
    "flag": "not rendered",
    "verify_script": "not rendered"
  }
}
```

Validate prompt rendering:

```bash
cage benchmark check <benchmark_id> --sample <sample_id>
```

The command writes the full rendered prompt under `.cage_checks/` and prints a
short preview for one-sample checks. Add `--show-prompt` only when you want the
full prompt in the terminal. For an unregistered local benchmark, use
`cage run examples/<benchmark>/default_<benchmark>.yml --dry-run` to validate project-file
loading before registration.

## Targets

If the benchmark launches targets, describe them as data whenever possible:

- sample manifest points to a challenge directory;
- challenge directory contains `challenge.json`;
- challenge directory contains Docker Compose files and setup assets;
- target server adapter converts that metadata into launch specs.

Build target images through the benchmark-owned hook:

```bash
cage benchmark build <benchmark_id> --sample <sample_id>
```

## Scoring

Scoring lives on the **`Scorer`**, not the benchmark. A benchmark exposes
`scorer()`; the scorer has two methods: `score()` (the offline verdict) and
`gather()` (the live evidence-gathering half). Create a `Scorer`:

```python
from cage.scoring import Scorer, ScoringContext, Score


class MyScorer(Scorer):
    name = "my_benchmark"

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        ok = "FLAG{" in ctx.output
        return {
            self.name: Score(
                value=1.0 if ok else 0.0,
                metadata={"matched": ok},
            )
        }
```

CAGE calls the **same** `score()` in three situations: once right after the
trial, again if you re-run `cage score` later, and — if your benchmark checks
for success while the trial is still running — during the trial too. Because one
method has to serve all three, keep it **pure**: it only reads evidence already
saved on `ctx` and returns a verdict; it must not reach out to a target (by
`cage score` time the target is long gone). `ScoringContext` (defined in
`cage/scoring/context.py`) gives you:

- `ctx.output` — the agent's final answer text;
- `ctx.sample` — the sample dict you yielded from `iter_samples()`;
- `ctx.prompt` — the prompt the agent received;
- `ctx.proxy_log` — every LLM request/response the agent made during the trial;
- `ctx.check_done_output` — evidence your `gather()` collected from the live
  target (see below); `""` if you don't use `gather()`;
- `ctx.live_payload` / `ctx.live_success` — a success verdict recorded while the
  trial was still running, if your benchmark records one;
- `ctx.metadata`;
- files under `ctx.trial_dir`.

### Live evidence: `Scorer.gather`

Some benchmarks can only judge success by looking at the **running target** —
e.g. confirming a file appeared inside a target container, or that a service now
answers. `score()` can't do that, because it may run long after the target is
torn down. Put that check in `gather()` instead: CAGE calls it while the target
is still up, you return any string you like as evidence, and CAGE saves it so a
later `score()` can read it back from `ctx.check_done_output`. If you don't need
to inspect a live target, don't override `gather()` — the default returns `""`.

```python
from cage.scoring import GatherRuntime

    def gather(self, runtime: GatherRuntime) -> str:
        # e.g. docker-exec the target to check a success marker,
        # or query a scoring endpoint the target itself exposes
        return check_target_marker(runtime.sample)
```

`gather()` is handed a `GatherRuntime` (not the agent container), so it can only
reach the *target*:

- `runtime.sample` — how to reach the target (its docker-compose project and any
  endpoints it publishes);
- `runtime.agent_output_dir` — a host directory holding the agent's produced
  files, when CAGE has them on disk; `None` otherwise;
- `runtime.container` — the live agent container; set only when CAGE runs the
  agent itself (`cage run`), and `None` when it doesn't.

That last point matters in one case: when CAGE only *serves* the target and an
outside agent drives it (the [serve mode](/agent-serve-mode) path), there is no
agent container, so `runtime.container` is `None` and scoring works entirely off
`runtime.agent_output_dir`.

### Scoring signals

A vuln/task may require multiple named signals — a scorer emits a `Score` per
signal. Two are conventional: **`verifier`** (an evaluator `verify.py` verdict)
and **`LLM_judge`** (a model call). Compose them by returning several named
`Score`s (or via `CompositeScorer`).

For offline scoring:

```bash
cage score <benchmark_id> --run-id <run_id>
```

For an unregistered local benchmark, pass the project YAML path instead.

## Dashboards

Benchmarks should expose domain meaning through `build_dashboard()`:

```python
from cage.artifacts import Column, Dashboard, Section, Stat


def build_dashboard(self, run_dir):
    return Dashboard(
        title="My Benchmark",
        sections=(
            Section(
                kind="summary",
                title="Overview",
                stats=(Stat("Mean score", "0.42"),),
            ),
            Section(
                kind="table",
                title="Trials",
                columns=(
                    Column("trial_id", "Trial"),
                    Column("score", "Score", align="right"),
                ),
                rows=({...},),
            ),
        ),
    )
```

Open a registered benchmark-id run after changing dashboard logic:

```bash
cage inspect .cage_runs/<agent_label>/<run_id>
```

For a path-based local project run, inspect the project-local artifact tree:

```bash
cage inspect examples/<benchmark>/.cage_runs/<agent_label>/<run_id>
```

## Default YAML

Every benchmark should ship a small default YAML that is safe for a smoke run.
Use names that tell the user what they are running, such as
`default_web.yml` or `default_post_exploit.yml`. Prefer:

```yaml
runtime:
  max_trials_global: 1
  passk: 1
  timeout: 1800
  max_input_tokens: null
  max_output_tokens: null
  max_cost: null
```

Document larger campaign settings in the benchmark README instead of making
the default expensive.

## Benchmark README Template

Use this structure:

```md
# <Benchmark Name>

## What It Measures

## Dataset

## Layout

## Models And Agents

## Example Configs

## Smoke Run

## Full Run

## Important Parameters

## Artifacts And Scoring

## Dashboard

## Resume And Cleanup

## Known Caveats
```

## Smoke Test Checklist

Before considering a benchmark ready:

```bash
cage benchmark check <benchmark_id> --sample <sample_id>
cage benchmark build <benchmark_id> --sample <sample_id> --dry-run
cage benchmark build <benchmark_id> --sample <sample_id>
cage benchmark build <benchmark_id> --max-concurrent 4
cage run <benchmark_id> --sample <sample_id> --run-id smoke-<benchmark>-001
cage inspect .cage_runs --port 8090
```

If the benchmark is not registered yet, use the lower-level project-file forms:
`cage run examples/<benchmark>/default_<benchmark>.yml --dry-run` and
`cage run examples/<benchmark>/default_<benchmark>.yml`. Those path-based runs write under
`examples/<benchmark>/.cage_runs`, so inspect that tree instead.

Registered benchmarks that need prebuilt images should implement
`Benchmark.build_targets(samples, max_workers=N, dry_run=False)`. CAGE's
`benchmark build` command only selects samples and calls that hook; the
benchmark decides whether to run compose builds, dataset scripts, or skip
targets with no prebuild step. In `dry_run=True`, return planned/skipped
results and report image tags without executing build subprocesses. If a
dataset script prebuilds a runtime image that is not a compose `build:` service,
mark the corresponding compose service with `x-cage-prebuild` in
`docker-compose.cage.yml`. The dry-run path should read the image from that
service's `image:` field instead of scraping shell scripts or inventing image
names. Use `--max-concurrent N` for a full prebuild sweep when the benchmark hook
can safely build multiple targets at once.

Add tests for:

- sample loading;
- prompt rendering;
- scorer fixtures;
- dashboard generation;
- target metadata conversion when applicable.

## Related Docs

- Core classes: [../reference/classes.md](../reference/classes.md)
- Experiment YAML: [../reference/project-yml.md](../reference/project-yml.md)
- CLI reference: [../reference/cli.md](../reference/cli.md)
- Framework internals: [../developing-cage/](../developing-cage/)
