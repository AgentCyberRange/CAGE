# CLI Reference

The `cage` CLI is the operator surface for benchmarks, runs, model registry
edits, artifact inspection, scoring, and Docker cleanup.

From an installed environment:

```bash
cage --help
cage <command> --help
```

From a source checkout, this form always uses the current tree:

```bash
uv run python -m cage.cli --help
uv run python -m cage.cli <command> --help
```

## Top-Level Commands

| Command | Purpose |
|---|---|
| `cage run` | Run or resume a registered benchmark id or project YAML |
| `cage benchmark` | List registered benchmarks, render prompt checks, run benchmark build hooks, serve targets as a range |
| `cage model` | List, show, and edit `config/models.yml` |
| `cage agent` | List, build, and debug agent runtimes |
| `cage inspect` | Browse run artifacts in the web inspector |
| `cage score` | Score or re-score completed runs |
| `cage gc` | Reclaim Docker resources for dead/orphaned runs |

Global options (before the subcommand):

| Option | Meaning |
|---|---|
| `-v` / `--verbose` | Enable debug logging (equivalent to `--log-level DEBUG`) |
| `--log-level [DEBUG\|INFO\|WARNING\|ERROR]` | Console log level (default `INFO`) |

Low-level legacy aliases are not part of the command tree. Use the public
groups below; internal subprocess entrypoints live as Python modules rather
than hidden Click commands.

## `cage run`

```bash
cage run PROJECT_OR_BENCHMARK [options] [benchmark-owned options]
```

`PROJECT_OR_BENCHMARK` can be a registered id such as `web_exploit_bench` or a
project YAML path such as `examples/agent_pentest_bench/default_web_exploit.yml`.

Run landing help:

```bash
cage run --help
```

Benchmark-specific help:

```bash
cage run web_exploit_bench --help
```

Common options:

| Option | Meaning |
|---|---|
| `--agent ID` | Keep only matching `agents[].id`; repeatable |
| `--model ID` | Override the selected agent to one model id; also the run key that `--model-source` rotates behind |
| `--model-source ID` | Registered `config/models.yml` ids this run round-robins across, per trial (repeatable, e.g. `--model-source source1 --model-source source2`); requires `--model <key>` as the logical name the sources rotate behind |
| `--sample ID` | Select sample ids; repeatable **and** comma-separated (`--sample a,b`); `--sample @FILE` reads ids from a file, one per line (`#` comments ok) |
| `--sample-slice SPEC` | Python-style slice of the ordered sample list, e.g. `:100` (first 100), `-100:` (last 100), `100:200`, `::2`; applied after `--sample` and before `--max-sample-num` |
| `--max-sample-num N` | Keep the first `N` selected samples before pass@k expansion |
| `--max-trial-num N` | Run only the first `N` expanded trials this invocation; later resume can finish the rest |
| `--max-concurrent N` | Override selected agent concurrency; with `--resume` and no `--agent`, cap all agents |
| `--passk N` | Override `runtime.passk` |
| `--timeout S` | Override `runtime.timeout` |
| `--max-rounds VALUE` | Override `runtime.max_rounds`. Takes a positive `N`, `unlimited` (no round cap — needs another stop condition), or `-1` (use the benchmark's built-in default) |
| `--max-input-tokens N` | Override `runtime.max_input_tokens` |
| `--max-output-tokens N` | Override `runtime.max_output_tokens` |
| `--max-cost USD` | Override `runtime.max_cost` |
| `--wait-for-model` | Before starting, poll every model endpoint (and each `--model-source`) until it answers — for remotely-launched vLLM servers with unknown boot time |
| `--wait-timeout S` | Max seconds to wait with `--wait-for-model` (default `0.0` = wait indefinitely) |
| `--wait-interval S` | Seconds between `--wait-for-model` polls (default `30.0`) |
| `--upstream-proxy URL` | Override `proxy.upstream_http_proxy` |
| `--set PATH=VALUE` | Override an arbitrary project YAML path |
| `--param KEY=VALUE` | Set a custom-agent param (fills a manifest `{placeholder}`); repeatable; requires one `--agent` if ambiguous |
| `--run-id ID` | Set the run id |
| `--resume` | Resume an existing run id |
| `--force` | Archive an existing run id and start fresh |
| `--dry-run` | Print run/resume plan without launching containers or touching the run directory |
| `--allow-launch-build` | Run the benchmark-owned build hook before target launch |

Benchmark-owned options are parsed after common options. Examples:

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
  --sample pb-postexp-range-1 \
  --prompt-level l0 \
  --passk 1 \
  --run-id post-smoke-001
```

Resume:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --resume --dry-run
cage run web_exploit_bench --run-id web-smoke-001 --resume
```

Force a fresh run with the same id:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --force
```

## `cage benchmark`

List registered benchmarks:

```bash
cage benchmark list
```

Show one benchmark's current run surface:

```bash
cage run web_exploit_bench --help
cage benchmark show web_exploit_bench
```

Render prompts and validate config without launching targets:

```bash
cage benchmark check web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0
```

Useful `benchmark check` options:

| Option | Meaning |
|---|---|
| `--agent ID` | Restrict to configured agent ids |
| `--model ID` | Override the selected agent to one model |
| `--sample ID` | Restrict samples; repeatable and comma-compatible |
| `--limit N` | Check only first `N` selected samples |
| `--passk N` | Override pass@k in rendered config |
| `--max-trials-global N` | Override the global trial cap (`runtime.max_trials_global`) |
| `--max-concurrent N` | Override selected agent concurrency |
| `--timeout S` | Override trial timeout |
| `--max-rounds N` | Override model-call round budget |
| `--max-input-tokens N` | Override input-token budget |
| `--max-output-tokens N` | Override output-token budget |
| `--max-cost USD` | Override cost budget |
| `--upstream-proxy URL` | Override proxy egress |
| `--set PATH=VALUE` | Override a project YAML path |
| `--out DIR` | Write check artifacts to a custom directory |
| `--preview-lines N` | Prompt preview lines to print for one-sample checks (default `12`) |
| `--show-prompt` | Print full rendered prompts |
| `--strict-exit` / `--no-strict-exit` | Exit non-zero if any prompt check fails (default `--strict-exit`) |

Run a registered benchmark build hook without launching targets:

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui --dry-run
cage benchmark build web_exploit_bench --sample pb-comfyui
cage benchmark build web_exploit_bench --max-concurrent 4
```

Build options:

| Option | Meaning |
|---|---|
| `--sample ID` / `--only ID` | Restrict sample ids |
| `--limit N` | Build only first `N` samples |
| `--max-concurrent N` | Build up to `N` benchmark targets concurrently (default `1`) |
| `--dry-run` | Print build hooks and image tags without building |
| `--rebuild` | Rebuild target images even if already built (default: skip images already built) |
| `--set PATH=VALUE` | Override a project YAML path, e.g. `--set eval.benchmark.hint_levels=[0]` |

## `cage benchmark serve`

Stand a benchmark's targets up as a browsable range (serve / PULL mode) without
running a full trial. An external agent then drives the targets itself over the
serve HTTP contract (list → launch → prompt → submit → close).

```bash
cage benchmark serve web_exploit_bench
cage benchmark serve web_exploit_bench --host 0.0.0.0 --port 8000 --open
```

| Option | Meaning |
|---|---|
| `--benchmark-root DIR` | Explicit directory of `challenge.json` targets (overrides the `BENCHMARK` argument); use for a dataset dir not laid out as `examples/<benchmark>/` |
| `--host HOST` | Interface to bind (default `127.0.0.1`; use `0.0.0.0` to expose externally) |
| `--port N` | Port to listen on (default `8000`) |
| `--namespace NAME` | Docker resource namespace isolating this server's targets (default `default`) |
| `--external-token TOKEN` | Enable external-audience mode (see below); empty means every caller is internal (legacy single-audience) |
| `--judge-model MODEL_ID` | Override the judge model for the `LLM_judge` signal; defaults to the benchmark's own declared judge |
| `--prompt-level [l0\|l1\|l2]` | Default hint tier for `GET /prompt` (default `l0`); the effective tier is bound per instance at launch (`GET /launch?prompt_level=`) |
| `--adapter MODULE:CLASS` | Load an extra benchmark adapter; repeatable |
| `--open` | Open the console in a browser after start |

With `--external-token` set, the server runs in two-audience mode: loopback
callers are treated as internal, while non-loopback callers must present
`Authorization: Bearer <token>`. Without the flag every caller is internal
(legacy single-audience behaviour).

## `cage model`

The model registry defaults to `config/models.yml`, as configured by
`config/cage.yml::models_file`.

```bash
cage model list
cage model show gpt-5.5
cage model set gpt-5.5 \
  --provider openai \
  --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 \
  --api-key '${OPENAI_API_KEY}'
```

Useful `model set` options:

| Option | Meaning |
|---|---|
| `--provider openai|anthropic|vllm` | Provider/protocol family |
| `--model NAME` | Model name sent upstream |
| `--endpoint URL` / `--base-url URL` | API base URL |
| `--api-key VALUE` | API key or env placeholder |
| `--auth-source PATH` | Host credential directory for subscription auth |
| `--agent-model-name AGENT=MODEL` | Agent-specific CLI model name |
| `--input-cost-per-1m N` | Input price per 1M tokens |
| `--output-cost-per-1m N` | Output price per 1M tokens |
| `--timeout S` | Model request timeout |
| `--max-retries N` | Upstream retry count |
| `--rl-reward-sink URL` | Enable RL mode: URL the trainer's reward sink listens on. LLM calls then carry an `X-Trial-Id` header and each trial's reward is POSTed here. Pass an empty string to disable |

## `cage agent`

```bash
cage agent list
cage agent list --all
```

Build images:

```bash
cage agent build
cage agent build --all
cage agent build --agent codex
cage agent build --agent codex --variant pentestenv
cage agent build --agent claude_code --variant pentestenv --no-cache
```

Debug commands are primarily for framework/agent adapter development:

```bash
cage agent debug RUN_DIR --trial TRIAL_ID --state pre
cage agent debug --agent codex --model gpt-5.5
```

## `cage inspect`

```bash
cage inspect [PATH] [--host HOST] [--port PORT] [--no-open]
```

`PATH` defaults to `examples/` (scan every project). `--host`, `--port`, and
`--no-open` all default to `None`; unset, they fall back to the values in
`config/cage.yml` (every inspector shares that single port).

Examples:

```bash
cage inspect
cage inspect examples/agent_pentest_bench --host 0.0.0.0 --port 8090 --no-open
```

Managed board commands:

```bash
cage inspect start PATH
cage inspect status PATH
cage inspect stop PATH
```

## `cage score`

Project mode reconstructs the benchmark scorer and scans that project's
`.cage_runs` tree:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml
cage score examples/agent_pentest_bench/default_web_exploit.yml --run-id web-smoke-001
```

Run-directory mode applies only explicitly supplied scorer files:

```bash
cage score .cage_runs/<agent_label>/<run_id> \
  --scorer path/to/scorer.py
```

Scoring is serial by default. `--max-concurrent N` (the same flag as `cage
run`) scores up to `N` trials at once — useful when a scorer re-runs an
`LLM_judge` signal, i.e. one model call per trial. Only the scorer call is
parallelized; every score artifact is still written serially, so the output is
identical to serial scoring:

```bash
cage score examples/agent_pentest_bench/default_web_exploit.yml \
  --run-id web-smoke-001 \
  --max-concurrent 8
```

## `cage gc`

Dry-run:

```bash
cage gc
```

Apply:

```bash
cage gc --apply
```

Common filters:

```bash
cage gc --namespace cage
cage gc --run-id web-smoke-001 --apply
cage gc --root .cage_runs
```

GC reclaims Docker containers, networks, and volumes labelled with CAGE run
metadata. It does not delete `.cage_runs` artifacts.
