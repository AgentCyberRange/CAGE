# Running Experiments

This guide is for operators running existing CAGE benchmarks. It assumes the
checkout is installed, `config/models.yml` is configured, and the relevant
dataset submodules are initialized.

Use either command form:

```bash
cage run web_exploit_bench ...
cage run examples/agent_pentest_bench/default_web_exploit.yml ...
```

In a source checkout, `uv run python -m cage.cli ...` is the most reliable way
to bypass a stale console script.

## Mental Model

Every run combines:

| Input | Owner | Examples |
|---|---|---|
| Benchmark | Benchmark author | Samples, prompts, target launch, scoring |
| Project YAML | Experiment runner | Agents, models, runtime limits, resume policy |
| CLI flags | Operator | `--agent`, `--model`, `--sample`, `--run-id`, `--resume` |

Registered benchmark-id runs write artifacts under the current working
directory:

```text
.cage_runs/<agent_label>/<run_id>/
```

Path-based project runs write next to the benchmark module or project file. For
example:

```text
examples/agent_pentest_bench/.cage_runs/<agent_label>/<run_id>/
```

The run directory is the source of truth for metadata, proxy logs,
trajectories, state snapshots, scores, dashboards, and copied artifacts.

## Safe Run Ladder

[Getting Started](../getting-started/) takes a fresh checkout to one inspected
smoke trial — install, model config, dataset, agent image, preflight, and a
single run. This ladder continues from there: scale up only once that smoke path
is green.

### 1. Re-check the benchmark surface

```bash
cage benchmark list
cage run web_exploit_bench --help
```

The dynamic help shows sample shape, the agent/model matrix, runtime defaults,
benchmark-owned flags such as `--prompt-level`, and recommended commands. Use
`cage benchmark check ... --show-prompt` to render a prompt without launching
targets or agents.

### 2. One verified smoke trial

```bash
cage run web_exploit_bench \
  --agent codex --model gpt-5.5 \
  --sample pb-comfyui --prompt-level l0 \
  --passk 1 --max-concurrent 1 \
  --run-id smoke-pb-comfyui-001
```

### 3. Small batch

```bash
cage run web_exploit_bench \
  --agent codex --model gpt-5.5 \
  --prompt-level l0 --passk 1 \
  --max-concurrent 2 --max-sample-num 3 \
  --run-id web-l0-small-001
```

### 4. Full campaign

Use the project defaults once the smoke and batch paths are verified:

```bash
cage run web_exploit_bench \
  --agent codex --model gpt-5.5 \
  --run-id web-codex-gpt55-full-001
```

## Run IDs

Use explicit run ids for anything you might inspect, resume, compare, or share.

```bash
cage run web_exploit_bench --run-id web-codex-gpt55-l0-p1-20260603 ...
```

Good ids are short and include benchmark, agent/model, prompt level, pass count,
or date. Reusing an id without `--resume` or `--force` is rejected.

Preview resume:

```bash
cage run web_exploit_bench --run-id web-codex-gpt55-l0-p1-20260603 --resume --dry-run
```

Resume:

```bash
cage run web_exploit_bench --run-id web-codex-gpt55-l0-p1-20260603 --resume
```

Archive and start fresh with the same id:

```bash
cage run web_exploit_bench --run-id web-codex-gpt55-l0-p1-20260603 --force
```

## Selecting Work

Limit sample count:

```bash
cage run web_exploit_bench --max-sample-num 5
```

Run only selected samples:

```bash
cage run web_exploit_bench --sample pb-comfyui
```

Run multiple samples — `--sample` is repeatable, comma-separated, or an `@FILE`
of ids (one per line, `#` comments ok):

```bash
cage run web_exploit_bench --sample pb-comfyui --sample pb-prestashop
cage run web_exploit_bench --sample pb-comfyui,pb-prestashop
cage run web_exploit_bench --sample @subset.txt
```

Slice the ordered sample list Python-style with `--sample-slice` (applied after
`--sample`, before `--max-sample-num`):

```bash
cage run web_exploit_bench --sample-slice :100     # first 100
cage run web_exploit_bench --sample-slice -100:    # last 100
cage run web_exploit_bench --sample-slice 100:200  # a window
cage run web_exploit_bench --sample-slice ::2      # every other sample
```

Select one configured agent:

```bash
cage run web_exploit_bench --agent codex
```

Select one model for the selected agent:

```bash
cage run web_exploit_bench --agent codex --model gpt-5.5
```

Rotate one run across several registered model endpoints (e.g. to load-balance
across replicas of the same model): pass `--model` as a logical key and the
`--model-source` ids CAGE round-robins behind it, per trial. The sources are
ordinary entries in `config/models.yml`:

```bash
cage run web_exploit_bench --agent codex \
  --model my-rotation \
  --model-source source1 --model-source source2
```

Select benchmark-owned levels:

```bash
cage run web_exploit_bench --prompt-level l0,l1
cage run post_exploit_bench --prompt-level l0
```

Use `cage run <benchmark_id> --help` to see which benchmark-owned flags exist.

## Runtime Knobs

Common CLI overrides:

```bash
cage run web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --max-concurrent 2 \
  --max-sample-num 5 \
  --max-trial-num 10 \
  --timeout 7200 \
  --passk 3 \
  --max-rounds 150 \
  --max-input-tokens 10000000 \
  --max-output-tokens 200000 \
  --max-cost 100 \
  --upstream-proxy http://<host-ip>:7890
```

Key YAML fields:

| Field | Meaning |
|---|---|
| `runtime.max_trials_global` | Global in-flight trial cap |
| `agents[].max_concurrent` | Per-agent/model concurrency cap |
| `runtime.max_target_setups` | Target setup concurrency cap |
| `runtime.passk` | Number of independent attempts per sample |
| `runtime.timeout` | Per-trial wall-clock timeout |
| `runtime.max_rounds` | Agent model-call round budget; `-1` defers to benchmark/sample default, `0` skips agent execution |
| `runtime.max_input_tokens` | Per-trial input-token budget |
| `runtime.max_output_tokens` | Per-trial output-token budget |
| `runtime.max_cost` | Per-trial USD cost budget |
| `proxy.request_timeout` | Upstream model request timeout |
| `proxy.upstream_http_proxy` | Host proxy for model egress |
| `target.startup_timeout` | Service readiness timeout |
| `target.compose_up_timeout` | Docker compose up timeout |

When a max-rounds budget is reached, the proxy rejects later model calls and
the orchestrator terminates an agent that keeps retrying. The trial is reported
as `max_rounds_reached`.

## Inspecting Runs

Start or reuse the inspector:

```bash
cage inspect .cage_runs --host 127.0.0.1 --port 8090
```

Remote/shared machine:

```bash
cage inspect .cage_runs \
  --host 0.0.0.0 \
  --port 8090 \
  --no-open
```

The inspector shows:

- running trials and last proxy status;
- model-call request steps and token totals;
- prompt, output, scoring, and dashboard artifacts;
- `proxy/progress.json`, `proxy/proxy.jsonl`, and proxy stderr/stdout;
- pre/post state snapshots and trajectory views.

## Scoring

Re-score all matching run dirs for a project:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml
```

Re-score one run id:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml --run-id web-smoke-001
```

Apply an extra scorer to a run directory:

```bash
cage score .cage_runs/<agent_label>/<run_id> \
  --scorer path/to/scorer.py
```

Score in parallel. `--max-concurrent N` mirrors `cage run` — it scores up to
`N` trials at once, which matters when a scorer re-runs an `LLM_judge` signal
(one model call per trial). Only the scorer call is parallelized; artifacts are
still written serially, so results are identical to serial scoring:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml \
  --run-id web-smoke-001 \
  --max-concurrent 8
```

## Cleanup

Dry-run Docker resource GC:

```bash
cage gc
```

Apply GC:

```bash
cage gc --apply
```

GC reads `.cage_runs` to avoid reclaiming live runs. It does not delete
artifacts.

## Troubleshooting

| Symptom | Command |
|---|---|
| Prompt render failure | `cage benchmark check <benchmark_id> --sample <sample_id> --show-prompt` |
| Target launch failure | `cage benchmark build <benchmark_id> --sample <sample_id>` before rerunning one smoke trial |
| Missing agent image | `cage agent build --agent <id> --variant pentestenv` |
| Need exact CLI surface | `cage run <benchmark_id> --help` |
| Need artifact detail | `cage inspect .cage_runs` for registered-id runs, or inspect the project-local `.cage_runs` for path runs |
| Old console-script behavior | `uv run python -m cage.cli ...` |
