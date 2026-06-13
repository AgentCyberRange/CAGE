# Operations

This guide is for long runs, shared machines, expensive model endpoints, and
cleanup after interrupted experiments.

## Operating Model

CAGE creates two durable things:

- `.cage_runs` artifacts on disk;
- Docker resources labeled with `cage.run_id`.

The artifact directory is the source of truth. Docker resources are disposable
and should be recreated or swept by label.

## Before A Large Run

Run this checklist before launching a campaign.

### 1. Confirm dataset state

```bash
git submodule status --recursive examples/agent_pentest_bench
```

Initialize missing submodules:

```bash
git submodule update --init --recursive examples/agent_pentest_bench/datasets/web_exploit_bench
```

### 2. Confirm images

```bash
cage agent build --agent codex --variant pentestenv
cage agent build --agent claude_code --variant pentestenv
docker images | grep -E 'cage/|ctfenv'
```

### 3. Confirm endpoint config

```bash
cage benchmark check web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1
```

### 4. Confirm targets

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui --dry-run
```

### 5. Run one real smoke

```bash
cage run web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id smoke-before-full-001
```

## Capacity Planning

The key controls are:

| Control | What it limits |
|---|---|
| `runtime.max_trials_global` | Total simultaneous trials |
| `runtime.max_target_setups` | Concurrent target launches/readiness waits |
| `runtime.max_input_tokens` | Per-trial cumulative input token budget |
| `runtime.max_output_tokens` | Per-trial cumulative output token budget |
| `runtime.max_cost` | Per-trial cumulative USD budget |
| `agents[].max_concurrent` | Concurrent trials for one agent config |
| model endpoint rate limits | Upstream request throughput |
| host CPU/RAM/disk | Docker target and agent capacity |

For heavy Docker targets, prefer:

```yaml
runtime:
  max_trials_global: 4
  max_target_setups: 1
```

Increase only after target setup and memory pressure are stable.

## Monitoring

Start the inspector:

```bash
cage inspect .cage_runs --host 0.0.0.0 --port 8090 --no-open
```

Watch Docker:

```bash
docker ps --filter label=cage.run_id=<run_id>
docker network ls --filter label=cage.run_id=<run_id>
docker volume ls --filter label=cage.run_id=<run_id>
```

Watch artifacts:

```bash
find .cage_runs -name progress.json | head
```

For one trial, inspect:

```text
trials/<trial_id>/meta.json
trials/<trial_id>/proxy/progress.json
trials/<trial_id>/proxy/stderr.log
trials/<trial_id>/scores/
```

## Resume Policy

Always preview:

```bash
cage run web_exploit_bench \
  --run-id <run_id> \
  --resume \
  --dry-run
```

Then resume:

```bash
cage run web_exploit_bench \
  --run-id <run_id> \
  --resume
```

Use `resume.max_attempts` to avoid infinite retry loops:

```yaml
resume:
  max_attempts: 3
```

Use `resume.retry_reasons` only when you understand the failure class:

```yaml
resume:
  retry_reasons:
    - target_unavailable
    - model_bad_gateway
```

## Cleanup

Dry-run global GC:

```bash
cage gc
```

Apply global GC:

```bash
cage gc --apply
```

Restrict GC:

```bash
cage gc --run-id <run_id>
cage gc --namespace pentestbench
cage gc --root .cage_runs
```

Detailed GC guide: [../cage-gc.md](../cage-gc.md).

## Orphan Resource Triage

Find CAGE-labeled resources:

```bash
docker ps -a --filter label=cage.run_id
docker network ls --filter label=cage.run_id
docker volume ls --filter label=cage.run_id
```

Find resources for one run:

```bash
docker ps -a --filter label=cage.run_id=<run_id>
docker network ls --filter label=cage.run_id=<run_id>
docker volume ls --filter label=cage.run_id=<run_id>
```

If the run is complete and artifacts are preserved, prefer:

```bash
cage gc --run-id <run_id>
cage gc --run-id <run_id> --apply
```

If many old runs are involved, prefer:

```bash
cage gc
cage gc --apply
```

## Sharing The Inspector

Local default:

```bash
cage inspect .cage_runs
```

Remote/shared URL:

```bash
cage inspect .cage_runs --host 0.0.0.0 --port 8090 --no-open
```

If binding to a non-loopback host, configure inspector auth in `config/cage.yml`
when required by the repo config. Do not expose run artifacts containing API
keys, prompts, target secrets, or private datasets to an untrusted network.

## Incident Playbook

| Incident | Action |
|---|---|
| Model endpoint outage | Stop or let current calls timeout, then resume with same run id |
| Docker daemon overloaded | Lower `runtime.max_trials_global`, cleanup orphan resources, resume |
| Target setup storm | Set `runtime.max_target_setups: 1` |
| Disk filling | Archive old `.cage_runs`, clean Docker resources, prune unused images deliberately |
| Many target launch failures | Run `cage benchmark build <benchmark_id> --sample <sample_id>` before a one-sample smoke rerun |
| Trial stuck without progress | Inspect `proxy/progress.json`, `proxy/stderr.log`, Docker logs |

## Campaign Handoff

Use the same handoff shape for every large run:

```md
## Background

Benchmark:
Agent/model:
Prompt levels:
Pass count:
Scale:

## Config changes

- runtime.passk:
- runtime.max_trials_global:
- runtime.max_target_setups:
- runtime.max_input_tokens:
- runtime.max_output_tokens:
- runtime.max_cost:
- agents[].max_concurrent:
- config/models.yml entry or required model env vars:

## Smoke command

```bash
cage run examples/<benchmark>/default_<benchmark>.yml \
  --agent <agent> \
  --sample <sample_id> \
  --max-sample-num 1 \
  --run-id <smoke-run-id>
```

## Full command

```bash
cage run examples/<benchmark>/default_<benchmark>.yml \
  --agent <agent> \
  --prompt-level <levels> \
  --run-id <campaign-run-id>
```

## Monitor

```bash
cage inspect examples/<benchmark>/.cage_runs --host 0.0.0.0 --port 8090 --no-open
```
```

## Related Docs

- Running experiments: [../running-experiments/](../running-experiments/)
- CLI reference: [../reference/cli.md](../reference/cli.md)
- Experiment YAML: [../reference/project-yml.md](../reference/project-yml.md)
- Target checks: [../targets-check.md](../targets-check.md)
