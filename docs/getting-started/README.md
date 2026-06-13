# Quick Start

This guide takes a fresh checkout to one inspected WebExploitBench trial, without
full-campaign tuning. For what that single `cage run` actually does under the
hood, read [How a Run Works](../how-a-run-works) afterwards.

## 1. Install

Requirements:

- Linux with Docker and Docker Compose.
- Python 3.11+.
- `uv` or `pip`.
- A reachable LLM endpoint.

```bash
git clone https://github.com/AgentCyberRange/CAGE.git
cd CAGE
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Check the installed command:

```bash
cage --help
```

## 2. Configure A Model

CAGE reads model endpoints from the repo-level file named by
`config/cage.yml::models_file`, which defaults to `config/models.yml`.
That file is intentionally local.

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
```

Show or edit a model entry:

```bash
cage model list
cage model show gpt-5.5
cage model set gpt-5.5 \
  --provider openai \
  --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 \
  --api-key '${OPENAI_API_KEY}'
```

The full entry shape, protocols, multi-key round-robin, decorated agent model
names, and Docker-host endpoint binding are covered in [Models](../models).

## 3. Pull One Dataset

For the first smoke run, initialize WebExploitBench only:

```bash
git submodule update --init --recursive \
  examples/agent_pentest_bench/datasets/web_exploit_bench
```

Other datasets can be initialized later:

```bash
git submodule update --init --recursive examples/agent_pentest_bench/datasets/post_exploit_bench
git submodule update --init --recursive examples/cvebench/datasets
git submodule update --init --recursive examples/nyuctfbench/datasets
git submodule update --init --recursive examples/autopenbench/datasets
```

## 4. Build An Agent Image

List release-facing agents:

```bash
cage agent list
```

Build the image for the agent you plan to run:

```bash
cage agent build --agent codex --variant pentestenv
```

Useful variants:

```bash
cage agent build --agent claude_code --variant pentestenv
cage agent build --agent qwen_code --variant pentestenv
cage agent build --agent kimi_code --variant pentestenv
```

Build all base images:

```bash
cage agent build
```

Build base images plus every Dockerfile variant:

```bash
cage agent build --all
```

## 5. Inspect Benchmark Defaults

Registered benchmarks:

```bash
cage benchmark list
```

WebExploitBench's dynamic run help is the best source for current samples,
agents, models, defaults, benchmark-specific options, and recommended commands:

```bash
cage run web_exploit_bench --help
```

At the time of writing, the default project is:

```text
examples/agent_pentest_bench/default_web_exploit.yml
```

Registered benchmark-id runs write under the current working directory:

```text
.cage_runs/<agent_label>/<run_id>/
```

Path-based runs, such as `cage run examples/agent_pentest_bench/default_web_exploit.yml`,
write next to that project/benchmark module instead.

## 6. Preflight

Render prompts and validate config without launching targets or agents:

```bash
cage benchmark check web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1
```

Print the full prompt for the selected sample:

```bash
cage benchmark check web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --show-prompt
```

Optional target prebuild check, without model calls:

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui --dry-run
```

Remove `--dry-run` when you specifically want to run the benchmark-owned build
hook.

## 7. Run One Trial

```bash
cage run web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id web-smoke-001
```

During the run, CAGE prints a compact terminal progress line. The browser
inspector has the detailed view: active trials, request steps, token totals,
artifacts, proxy logs, trajectories, state snapshots, and dashboards.

Open the inspector:

```bash
cage inspect .cage_runs --host 127.0.0.1 --port 8090
```

For a remote shared machine:

```bash
cage inspect .cage_runs \
  --host 0.0.0.0 \
  --port 8090 \
  --no-open
```

## 8. Resume Or Start Fresh

Preview resume behavior:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --resume --dry-run
```

Resume:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --resume
```

Archive an existing run id and start fresh:

```bash
cage run web_exploit_bench --run-id web-smoke-001 --force
```

## Common First-Run Failures

| Symptom | Check |
|---|---|
| `Model '<id>' not found` | `config/models.yml` has the id used by `--model` or `agents[].models` |
| API key missing | The env var referenced by `api_key` is exported in the shell running CAGE |
| Agent image missing | Run `cage agent build --agent <id> --variant pentestenv` |
| Target images missing | Run `targets-check ... --build` or benchmark-specific build hooks |
| Inspector opens wrong tree | For `cage run <benchmark_id>`, point it at the `.cage_runs` directory under the cwd that launched the run |
| Old `cage` command behavior | Reinstall from this checkout with `uv pip install -e .` and make sure the environment is active |

## Next

- [How a Run Works](../how-a-run-works) — what that single `cage run` actually did.
- [Running Experiments](../running-experiments/) for campaign operations.
- [CLI Reference](../reference/cli.md) for command details.
- [Experiment YAML Reference](../reference/project-yml.md) for config fields.
