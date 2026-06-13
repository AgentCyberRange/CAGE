# The CLI

CAGE has one real command — `cage run` — and a handful of others that each expose
**one slice** of it. Once you see the slices, you never have to memorize the CLI:
you ask "which part of a run do I need right now?" and the command follows.

## Every command is a slice of `cage run`

[How a Run Works](/how-a-run-works) walked the run lifecycle: resolve → prepare →
prompt → execute → score → clean up, all writing artifacts to disk. Each
sub-command is that lifecycle, cut at a different point:

| Command | Which part of a run | Stops before |
|---|---|---|
| `cage run` | the whole pipeline | — |
| `cage benchmark check` | the front half: resolve → `prepare_trial` → `build_prompt` | launching any target or agent |
| `cage benchmark build` | only the target-build hook | the agent, the proxy, scoring |
| `cage benchmark show` / `list` | resolve only, to print the run surface | doing anything |
| `cage agent build` / `debug` | only the image / container half | the benchmark and the model |
| `cage model` | edits the registry a run *reads* | the run itself |
| `cage score` | only the scoring tail, re-run offline | re-executing the agent |
| `cage inspect` | *reads* the artifacts a run produced | writing anything |
| `cage gc` | cleans the Docker side-effects a run left | touching `.cage_runs` |

And `--resume` / `--force` / `--dry-run` are not separate commands — they are
**modes of the same run**: replay a finished run's outcomes, archive-and-restart,
or plan-without-doing.

::: info Why it's built this way
There is no separate "render prompt" engine and no separate "score" engine.
`cage benchmark check` is `cage run` with the launch and execute steps removed;
`cage score` is the same scorer object the live run used, fed a `ScoringContext`
rebuilt from disk. A bug fixed in the run path is fixed in every slice, because
they are the same code reached from a different entry point.
:::

## `cage run` — the whole pipeline

```bash
cage run PROJECT_OR_BENCHMARK [options] [benchmark-owned options]
```

`PROJECT_OR_BENCHMARK` is either a registered id (`web_exploit_bench`) or a
project YAML path (`examples/agent_pentest_bench/default_web_exploit.yml`). The
dynamic help is the authoritative surface for any benchmark:

```bash
cage run web_exploit_bench --help     # samples, agent/model matrix, defaults, benchmark flags
```

A constrained smoke trial:

```bash
cage run web_exploit_bench \
  --agent codex --model gpt-5.5 \
  --sample pb-comfyui --prompt-level l0 \
  --passk 1 --max-concurrent 1 \
  --run-id web-smoke-001
```

Benchmark-owned options (here `--prompt-level`) are parsed after the common ones.
For the scale-up ladder, run ids, and budgets, see
[Running Experiments](/running-experiments/); for every flag, the
[CLI Reference](/reference/cli).

::: info Registered id vs project path
A **registered id** (`web_exploit_bench`) writes runs under the *current working
directory*: `.cage_runs/<agent_label>/<run_id>/`. A **project path**
(`examples/.../default_web_exploit.yml`) writes next to that project file:
`examples/agent_pentest_bench/.cage_runs/...`. Point `cage inspect` at whichever
tree the run actually wrote to.
:::

## `cage benchmark` — the front half of a run

`check` runs everything up to the rendered prompt, then stops — no target, no
agent, no model call. It is the fastest way to validate config and prompts:

```bash
cage benchmark check web_exploit_bench \
  --agent codex --model gpt-5.5 \
  --sample pb-comfyui --prompt-level l0
cage benchmark check web_exploit_bench --sample pb-comfyui --show-prompt
```

`build` runs only the benchmark-owned target-build hook (image builds, dataset
prep), with `--dry-run` to preview:

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui --dry-run
cage benchmark build web_exploit_bench --sample pb-comfyui
```

`list` and `show` resolve the benchmark and print its run surface:

```bash
cage benchmark list
cage benchmark show web_exploit_bench
```

## `cage agent` — the container half of a run

Build the image a run will launch, or drop into a container without any
orchestration to debug the agent itself:

```bash
cage agent list
cage agent build --agent codex --variant pentestenv
cage agent debug --agent codex --model gpt-5.5     # interactive shell, no benchmark
```

`agent debug` is the container/proxy half of a run with the benchmark removed —
ideal for confirming a new agent's launch command and env before a real run. See
[Adding an Agent](/adding-a-new-agent).

## `cage model` — the registry a run reads

Model endpoints live in the repo-level `config/models.yml` (created from
`config/models.example.yml`). `cage model` edits it:

```bash
cage model list
cage model show gpt-5.5
cage model set gpt-5.5 --provider openai --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 --api-key '${OPENAI_API_KEY}'
```

Full field semantics are in the
[Experiment YAML Reference](/reference/project-yml).

## `cage score` — the scoring tail, offline

Re-run scoring without re-executing the agent. Project mode reconstructs the
benchmark scorer and scans that project's runs; run-dir mode applies only an
explicitly supplied scorer:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml
cage score examples/agent_pentest_bench/default_web_exploit.yml --run-id web-smoke-001
cage score .cage_runs/<agent_label>/<run_id> --scorer path/to/scorer.py
```

Because it uses the same scorer the live run used, a re-score and the inline
score agree by construction.

## `cage inspect` — read the artifacts

A read-only browser over `.cage_runs`. It writes nothing:

```bash
cage inspect .cage_runs --host 127.0.0.1 --port 8090
cage inspect .cage_runs --host 0.0.0.0 --port 8090 --no-open    # shared host
```

It shows running trials, model-call steps and token totals, prompts, outputs,
scores, proxy logs, state snapshots, and dashboards. A managed board
(`cage inspect start/status/stop PATH`) keeps one server per tree.

## `cage gc` — clean the Docker side-effects

A run leaves two things: immortal `.cage_runs` artifacts, and disposable Docker
resources labeled with `cage.run_id`. `gc` reclaims the second without ever
touching the first:

```bash
cage gc                       # dry-run (default, safe)
cage gc --apply               # IRREVERSIBLE — reads .cage_runs to spare live runs
cage gc --run-id <id> --apply # one run
```

`--apply` permanently deletes containers, networks, and volumes — read the
[`cage gc` guide](/cage-gc) before using it on a shared host.

## Lifecycle modes, not commands

```bash
cage run web_exploit_bench --run-id r1 --resume --dry-run   # preview what resume would do
cage run web_exploit_bench --run-id r1 --resume             # replay finished trials, re-run retryables
cage run web_exploit_bench --run-id r1 --force              # archive r1, start fresh under the same id
```

Reusing a run id without `--resume` or `--force` is rejected, so an id is a
stable handle you can inspect, resume, compare, and share. Resume semantics —
which trials re-run and why — follow the termination classifier in
[How a Run Works › Termination and resume](/how-a-run-works#termination-and-resume).

## Source-checkout note

In a source checkout, `uv run python -m cage.cli ...` always uses the current
tree and bypasses a stale console script. The
[CLI Reference](/reference/cli) lists every command and flag in full.
