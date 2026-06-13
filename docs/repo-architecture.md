# CAGE Repository Architecture

This note is the durable architecture map for CAGE — the design principles and
the deep mechanics. For the narrative walk-through of a single run, read
[How a Run Works](/how-a-run-works) first; this page is the reference it links
into when syncing runtime, benchmark, proxy, and scoring changes across the repo.

## Purpose

CAGE evaluates installed coding agents inside Docker containers. A run combines:

- an experiment config from `project.yml`
- one or more agent definitions and model subjects
- a benchmark implementation
- an optional benchmark target runtime (CTF / CVE / web pentest, …)
- an in-container LLM API proxy
- trial artifacts and benchmark scores

The orchestrator owns lifecycle and artifact flow. Benchmarks own samples,
workspace preparation, prompts, and scoring semantics.

## Top-Level Modules

| Area | Path | Responsibility |
| --- | --- | --- |
| CLI | `cage/cli/` | User-facing `cage run`, `cage benchmark`, `cage agent`, `cage inspect`, `cage score`, and `cage gc` entry points. |
| Orchestration | `cage/experiment/engine/conductor.py` | Container lifecycle, trial execution, proxy lifecycle, target lifecycle, submit/live-check lifecycle, artifact collection. Also spawns the embedded `cage.target.serve` subprocess used by `cage run`. |
| Terminal run output | `cage/cli/ui/run.py` | Plain progress reporter used by `cage run`; prints inspector entry URLs and a compact run progress line while normal logs continue. |
| Preflight | `cage/experiment/engine/preflight.py` | Runtime preflight checks for configured projects, including image/container checks, optional target reachability, proxy/internet checks, and custom shell commands. |
| Target build | `cage/target/build.py`, `Benchmark.build_targets()` | Build-only path used by `cage benchmark build`; layer 1 loads the project, selects samples, passes `max_workers`/`dry_run`, and dispatches to the benchmark-owned build hook without launching targets. |
| Target diagnostics | `cage/target/check.py` | Lower-level target readiness mechanics used by internal tests and future public target workflows. User-facing target preparation currently goes through `cage benchmark build`. |
| Config | `cage/config/experiment.py` | `project.yml` parsing (`resolve()`) into `ExperimentRun`, including runtime, proxy, target, live-check, agents, and models |
| Agents | `cage/agents/` | Agent adapters such as Claude Code and Codex; install, launch, env vars, parsing, state paths |
| Benchmarks | `cage/benchmarks/base.py`, `examples/*/benchmark.py` | Benchmark ABC plus concrete benchmark integrations |
| Runtime substrate | `cage/sandbox/`, `cage/experiment/engine/` | Docker container wrapper and substrate (`sandbox/`); trial conductor, scheduler, submit/live-check monitors, resource recording (`experiment/engine/`) |
| Target server | `cage/target/server/` | FastAPI service that owns docker-compose target lifecycle. Drives `docker compose up/down`, network admin, readiness probes, subnet allocation, and supports both internal (loopback) and external (token-authed) audiences. See `docs/serve-external-audience.md`. |
| Web inspector | `cage/web/` | Flask app behind `cage inspect`. Browses `.cage_runs/` runs, renders dashboards and per-trial detail, downloads workspace artifacts. |
| Prompt renderer | `cage/benchmarks/prompt_contract.py` | Generic Jinja2 renderer; benchmarks own their own `prompts/` directories under `examples/<name>/prompts/` |
| Storage | `cage/artifacts/run_storage.py` | `.cage_runs` directory layout and artifact persistence |
| Trajectories | `cage/proxy/trajectory.py` | `.traj` reconstruction from proxy logs |
| Tests | `tests/` | Unit and integration coverage for config, orchestrator, proxy, runtime, examples |

## Runtime Flow

The standard serial trial path is:

```text
cage run project.yml
  -> resolve()
  -> build agent/model matrix
  -> _run_single_agent()
  -> _setup_container()
  -> for each Trial:
       _execute_trial()
         1. launch target stack when enabled (via ChallengeClient)
         2. connect agent container to target runtime network
         3. snapshot pre-state
         4. reset workspace
         5. benchmark.prepare_trial()
         6. inject target info into sample
         7. start submit service when benchmark supports flag checks
         8. start in-container LLM proxy
         9. build prompt
        10. start live-success monitors
        11. execute agent CLI
        12. stop proxy and collect logs
        13. snapshot post-state
        14. collect runtime artifacts
        15. score benchmark output, preferring live-success verdicts
        16. cleanup submit service, target network, and target stack
```

When `agents[].max_concurrent > 1`, CAGE uses `_run_trial_isolated()` so
each trial gets its own agent container and target stack lifecycle.

### Target Build And Launch Flow

`cage benchmark build` runs benchmark-owned prebuild hooks without agents or
model calls:

```text
cage benchmark build <benchmark_id>
  -> resolve()
  -> select benchmark samples
  -> call Benchmark.build_targets(samples, max_workers, dry_run)
```

Real target launch happens during `cage run`, where the orchestrator starts an
embedded `cage.target.serve` subprocess and records its log under
`.cage_runs/`.

## Configuration Model

`resolve()` (in `cage/config/experiment.py`) reads `project.yml` and returns `ExperimentRun`.

Important config blocks:

| Block | Dataclass | Notes |
| --- | --- | --- |
| `config/models.yml` / `subjects` | `ModelConfig` | Model subjects are loaded from the repo registry and paired with agents |
| `agents` | `AgentInstance` | Agent kind, image, state mode, home, session args, plugin list |
| `runtime` | `ExecutionConfig` | Timeout, concurrency, `agent_network_mode`, live check settings |
| `proxy` | `ProxyConfig` | Enables in-container proxy and optional system rewriting |
| `target` | `TargetConfig` | Target backend defaults, remote server URL, SSH tunnel, external access, network name |
| `eval.benchmark` | Benchmark loader | Module/class and benchmark-specific constructor options |

Target defaults are enabled unless explicitly changed by config. The
orchestrator builds `TargetConfig` and passes benchmark challenge data
into `ChallengeClient` so remote launch calls can resolve challenge
metadata.

## Container Runtime

`cage/sandbox/containers.py` wraps Docker commands. It owns:

- `docker run` construction
- env var and volume wiring
- `host.docker.internal:host-gateway` mapping
- workspace setup/reset
- command execution and background execution
- network connect/disconnect/sync
- state snapshot file transfer helpers

The agent container may join multiple Docker networks during one trial:

- the base network from `runtime.agent_network_mode`
- a per-trial target stack network

Cleanup must stop trial-local submit daemons and detach target runtime networks
without reverting agent workspace artifacts.

## Proxy Runtime

CAGE starts a proxy inside the agent container.

```text
cage/proxy/host.py
  -> writes config JSON into container
  -> copies cage/proxy/sidecar.py into /opt/cage-proxy/
  -> starts container_proxy.py as agent
  -> health checks /healthz
  -> collects /tmp/cage_proxy_logs/proxy.jsonl
```

`container_proxy.py` translates Anthropic-style calls to the upstream protocol,
logs requests/responses, and reconstructs tool calls. Its upstream HTTP client
can use a host-side proxy such as:

```text
http://<host-ip>:7890
```

Prompt templates now tell agents to use:

```bash
export HTTPS_PROXY=http://<host-ip>:7890 && {install_cmd}
```

when installing external dependencies inside the evaluation container. On
Linux, `host.docker.internal` needs a `host-gateway` mapping; direct host IPs
are usually easier to reason about when the proxy is bound to `0.0.0.0` or a
Docker-reachable interface.

## Target Runtime

Target lifecycle is managed through `cage/target/client.py`.
`ChallengeClient` is the entry point; concrete backends are `LocalBackend`
(runs `docker compose` here) and `RemoteBackend` (calls the embedded
`cage.target.serve` over HTTP). Either way the orchestrator only sees the `ChallengeClient`
interface.

```text
ChallengeClient.get_challenge_data(chal_id)
  -> RemoteBackend.initialize()  (or LocalBackend)
  -> GET /launch/{chal_id}
  -> returns services, runtime network, scoring metadata
  -> _inject_target_info() writes target fields into sample
```

Two target exposure styles exist:

| Mode | Sample fields | Agent guidance |
| --- | --- | --- |
| Alias mode | `target_info`, `network_name` | Use service hostnames such as `target:9090` |
| Network mode | `network_subnet`, `network_name` | Discover target hosts inside the subnet |

CVEBench tasks use built-in target scoring at `target:9091`. NYU and
AutoPenBench use CAGE's in-container `submit` service when enabled.

### Network mode policy (`network_mode` in challenge.json)

Independent of the agent-facing exposure styles above, every launched
target instance lives on one of two Docker network topologies. This is
controlled by `network_mode` in challenge.json (or the adapter's
`runtime_patches`):

| Mode | Default? | Behavior |
| --- | --- | --- |
| `compose_project_local` | **yes** (empty / unset / `default` all resolve here) | Each instance owns its compose-created networks. Services keep their compose-native DNS aliases (`mysql`, `db`, `nacos`, …) and resolve only within the project. The agent joins one of these networks — auto-picked from the declared compose networks (0 declared → `default`, 1 declared → that one, >1 declared → `agent_network` must be set explicitly). |
| `shared_external` (legacy `alias`) | opt-in | Every service is attached to the namespace-shared `cage_bench_<ns>` bridge with project-scoped aliases. Bare service names leak across project boundaries (compose auto-registers the short name on every network), so two passk trials of the same target — or two different targets that both name a service `db` — see each other's containers. Useful for cvebench / autopenbench style runs where the agent expects every target on one stable bridge; not safe for the general pass@k agent_pentest_bench workload. |

Project-local networks get IPAM from `TARGET_SERVER_PROJECT_LOCAL_SUBNET_POOL`
(default `172.31.0.0/16`) with `TARGET_SERVER_PROJECT_LOCAL_SUBNET_PREFIX`
(default `/24`, ≈254 host IPs — sized for Dify-class targets with ~14
services). Each project-local network is labelled with
`cage.target_server.network=<DOCKER_NETWORK>` and
`cage.target_server.role=runtime` so `cleanup_orphan_networks` can
discard them safely without touching another namespace's networks.

For challenges with multiple compose networks that the agent should
choose between (post-exploit ranges with `range1_public_network` +
sub-networks), set `agent_network` in challenge.json to the name of
the network the agent should join. If a single network is declared,
auto-pick covers it.

## Termination Classification

Every trial ends with a structured `termination_reason` in `meta.json`,
written by `cage/experiment/engine/termination.py::classify_trial_termination`. The
classifier is the **single source of truth** — orchestrator / web / resume
all read this field, never re-derive it. Inputs are deliberately
structural to keep the verdict deterministic and re-computable from
on-disk artifacts:

| Input | Source | Used for |
|-------|--------|----------|
| `exit_code` | agent process | timeout / OOM / nonzero-exit gating |
| `timed_out` | orchestrator | `execution_timeout` |
| `terminated_by_limit` | orchestrator | `tool_limit` (legacy) |
| `error` | orchestrator exception text | `trial_error` |
| `proxy.jsonl` records | in-container proxy | `model_*` subtypes via HTTP status + `error.type`; `max_rounds_reached` via successful non-compact round count; token budget fallback totals |
| `proxy/progress.json` | in-container proxy | `max_input_tokens_reached`, `max_output_tokens_reached`, `max_cost_reached` budget hits |
| `max_rounds` | `effective_max_rounds` (orchestrator); falls back to `sample.max_rounds` in reclassify | `max_rounds_reached` budget hit |
| token/cost budgets | `runtime.max_input_tokens`, `runtime.max_output_tokens`, `runtime.max_cost` | token/cost budget hits |

**Detection order** (intentional — earlier branches win):

1. Explicit exception (`termination_info_from_exception`)
2. `timed_out` ⇒ `execution_timeout`
3. `terminated_by_limit` ⇒ `tool_limit`
4. orchestrator `error` text ⇒ `trial_error`
5. exit 137 ⇒ `oom_killed`
6. successful non-compact proxy rounds ≥ `max_rounds > 0` and exit ≠ 0 ⇒ `max_rounds_reached`
7. proxy cumulative usage ≥ token/cost budget and exit ≠ 0 ⇒ `max_input_tokens_reached`, `max_output_tokens_reached`, or `max_cost_reached`
8. last proxy entry has non-200 ⇒ `model_*` subtype (`classify_upstream_error`: error.type substring, then HTTP status with `usage_limit`/`quota`/`billing` disambiguation for 429)
9. exit 0 ⇒ `completed`
10. exit ≠ 0 and nothing above fits ⇒ `agent_exit_nonzero` (truly unknown — defer to trial detail page; resume replays it rather than re-running)

### Why max_rounds_reached is structural

The in-container proxy enforces `max_rounds` by 429-rejecting the
next request after successful non-compact agent rounds reach the budget,
but does **not** call `ProxyRecorder.record()` for the rejection — see
the `_send_json(HTTPStatus.TOO_MANY_REQUESTS)` branch in
`cage/proxy/sidecar.py` which `return`s before recording.
Failed upstream calls and context-compaction bookkeeping calls remain in
`proxy.jsonl` for audit, but they do not spend the round budget.

This branch sits **before** the upstream-error scan so a stray transient
5xx earlier in the run (which the agent recovered from) can't outvote
a deterministic budget hit. A non-zero exit gates the override —
clean exit at exactly `max_rounds` stays `completed`.

### What is NOT used

Agent stdout text is deliberately **not** scanned for keywords like
"502", "rate limit", "usage limit", "context window", or "unauthorized".
Those phrases appear in legitimate agent reasoning (an agent
debugging a target's web service, an agent reading API docs, an agent
narrating its plan) and produced false positives in earlier versions.
The classifier only trusts structural signals — `error.type` strings
emitted by the upstream server itself, HTTP status codes from the
upstream connection, process exit codes from the kernel.

### Resume implication

`_DEFAULT_RESUME_RETRY_REASONS` (in `cage/experiment/engine/conductor.py`) re-runs the
infra subset; everything else replays the prior outcome from disk.
This means `max_rounds_reached` is NOT re-run by default — retrying
a budget-exhausted trial would just exhaust the budget again. Users
can extend or shrink the retry set via project.yml `resume.retry_reasons`
and cap total tries with `resume.max_attempts`.

The token and cost limits follow the same policy. The proxy returns a 429
with `type: "budget_limit_error"` on the next model call after cumulative
usage reaches `runtime.max_input_tokens`, `runtime.max_output_tokens`, or
`runtime.max_cost`. These reasons are completed budget stops, not model
failures, so resume keeps them by default.

#### Filtering which trials resume re-runs

The retry decision lives in one place — `_decide_resume_trial` in
`cage/experiment/engine/conductor.py`, shared by the real run, the `cage score` path, and
the `--dry-run` preview, so all three always agree. Beyond `retry_reasons` /
`max_attempts`, two project.yml knobs narrow *which* failed trials actually
re-run:

```yaml
resume:
  retry_reasons: [model_error]   # opt these reasons into the retry set
  max_attempts: 3                # cap total tries per trial (0 = unlimited)

  select:                        # positive gate (by trial-id regex)
    id_matches: "range-8"        #   ONLY ids matching this may re-run;
                                 #   every other trial replays as-is

  keep_if:                       # veto — KEEP (replay) an otherwise-retryable
    min_rounds: 100              #   trial when ANY threshold matches:
    min_duration_s: 1800         #   ran >= 100 agent rounds / >= 30 min,
    id_matches: "range-3"        #   or its id matches this regex
```

A trial is re-run iff it is retry-eligible (missing/blank meta, or a
non-completed `termination_reason` in the retry set) **AND** passes `select`
**AND** is not vetoed by any `keep_if` threshold **AND** is under
`max_attempts`. `keep_if` is the "this attempt already did enough work — don't
throw it away on a late retryable error" rule: a `model_error` that ran 157
rounds before the upstream rejected one request is kept, while one rejected on
its first request (1 round) still re-runs. Rounds come from
`proxy/progress.json` `total_requests`; duration from `meta.json`
`timing.duration_ms`. `--dry-run` prints the per-trial decision label
(e.g. `keep_if min_rounds (ran 157 ≥ 100)`) so the classification is
auditable before any container starts. The `execution.max_trial` cap (by
ascending trial index) composes with all of the above — it defers trials at
or beyond the cap to a later capless invocation.

### Migration for old runs

Pre-classifier runs carry the legacy `agent_exit_nonzero` reason for every
non-zero exit. A later revision added the `max_rounds_reached` structural
detection. Both upgrades are batched in
`scripts/reclassify_meta.py`:

```bash
python scripts/reclassify_meta.py <run_dir> --dry-run
python scripts/reclassify_meta.py <run_dir>
```

The script preserves prior values under `legacy_termination_reason` /
`legacy_termination_detail` / `legacy_termination_source` /
`legacy_status` so the migration is auditable. Trials whose reason was
already set by a non-classifier path (`completed`, `target_unavailable`,
`execution_timeout`, `user_interrupted`, `tool_limit`, `trial_error`,
`cancelled_before_start`, `live_success`) are skipped — they came from
the orchestrator directly and are already right. `max_rounds_reached`
is rechecked so old runs that counted failed upstream calls against the
round budget can be corrected.

## Live Check And Success Verdicts

CAGE has a unified live-success verdict path for checks that happen while the
agent is still running:

```text
trials/{trial_id}/runtime/live_success.json
```

Benchmark scoring should prefer this verdict over text extraction or post-trial
fallbacks. A successful verdict has:

```json
{
  "success": true,
  "mode": "reactive",
  "source": "submit",
  "benchmark": "nyu_ctf",
  "trial_id": "trial-id",
  "evidence": {}
}
```

The shared helpers live in `cage/artifacts/live_success.py`. Runtime monitors live
in `cage/experiment/engine/live/monitor.py`.

### Submit Service

NYU CTF and AutoPenBench use an in-container Unix-socket submit service instead
of one check container per trial.

```text
orchestrator
  -> copies submit_server.py to /opt/cage-submit/submit_server.py
  -> copies submit_client.py to /usr/local/bin/submit
  -> runs submit_server.py as root via docker exec -i
  -> sends only answer hash + metadata over stdin
  -> leaves stdin open until trial cleanup
```

The server owns the expected-answer hash in memory, listens on:

```text
/run/cage-submit/submit.sock
```

and records every submit attempt to:

```text
trials/{trial_id}/live_checks.jsonl
```

The client is agent-facing:

```bash
submit "<candidate flag>"
```

`submit_server.py` is root-owned and installed under a `0700` directory; the
agent can execute `/usr/local/bin/submit` but cannot read or edit the server
script after install.

### Reactive Mode

Reactive live check watches actions the agent already triggered. It does not
actively poll unless the agent first asks for a check.

Two reactive signals exist:

| Signal | Source artifact | Success action |
| --- | --- | --- |
| `submit` result | `live_checks.jsonl` with `correct: true` | Write `runtime/live_success.json` with `source: submit` |
| Agent touches `:9091` | new tool call in `proxy/proxy.jsonl` | Immediately call `benchmark._tool_check_done(container, sample)` once, then write success if that response is accepted |

The `:9091` detection scans only newly emitted tool-call command fields from the
assistant response. It does not scan the full prompt/history, because CVEBench
prompts mention `target:9091` on every request. If a `:9091` command is detected,
the monitor logs the orchestrator-side check to:

```text
trials/{trial_id}/runtime/check_done_reactive.jsonl
```

The agent's original command is not stored there by default; the artifact stores
the trigger class and `_tool_check_done` output to avoid leaking proof payloads.

### Polling Mode

Polling is separate from reactive mode. When enabled, `CheckDonePoller` calls:

```python
benchmark._tool_check_done(container, sample)
```

at `runtime.live_check.polling.interval_seconds`, regardless of whether the
agent touched `:9091`. Each poll is written to:

```text
trials/{trial_id}/runtime/check_done_polls.jsonl
```

If a poll response parses as success, CAGE writes `runtime/live_success.json`
with `mode: polling` and `source: check_done`.

### Config

`runtime.live_check` controls both modes:

```yaml
runtime:
  live_check:
    enabled: true
    max_calls: 3
    stop_on_success: true
    reactive:
      enabled: true
      check_on_submit: true
      check_on_9091_call: true
    polling:
      enabled: false
      interval_seconds: 5
      stop_on_success: true
```

`stop_on_success` defaults to true for reactive mode. Polling has its own
`polling.stop_on_success`. If a monitor stops the agent after success,
`meta.json` is marked as completed with `termination_reason: live_success`.

`examples/nyuctfbench/default_nyu_ctf.yml` currently enables both reactive and polling, but NYU
only has submit-based live success unless the benchmark implements
`_tool_check_done()`. CVEBench implements `_tool_check_done()` and
`_parse_live_success_output()`, so both `:9091` reactive confirmation and
polling can work there.

## Benchmark Boundaries

Benchmarks implement:

| Method | Owner responsibility |
| --- | --- |
| `iter_samples()` | Discover and normalize sample metadata |
| `prepare_trial(container, sample, workspace)` | Copy files/env into the agent workspace |
| `build_prompt(sample)` | Render benchmark-specific prompt text |
| `score(output, sample, context)` | Convert agent output and artifacts into `Score` objects |

Current examples:

| Benchmark | Path | Scoring style |
| --- | --- | --- |
| StrongReject | `examples/strongreject/benchmark.py` | LLM/judge and safety scoring |
| NYU CTF | `examples/nyuctfbench/benchmark.py` | `runtime/live_success.json` first, then flag extraction |
| AutoPenBench | `examples/autopenbench/benchmark.py` | `runtime/live_success.json` first, then token extraction |
| CVEBench | `examples/cvebench/benchmark.py` | `runtime/live_success.json`, then saved `check_done_output.txt`, then legacy `score_snapshot.json` |

## Artifact Layout

Run artifacts live under `.cage_runs`.

```text
.cage_runs/
  {agent_id}:{model_id}/
    run-{timestamp}/
      dashboard.json
      results.csv
      {mode}/
        trials/
          {trial_id}/
            meta.json
            prompt.txt
            task_output.json
            proxy/proxy.jsonl
            scores/{benchmark}.json
            state_pre/
            state_post/
            runtime/
              live_success.json
              check_done_reactive.jsonl
              check_done_polls.jsonl
```

`proxy.jsonl` is the source of truth for LLM request/response traces. `.traj`
files are derived artifacts for easier human inspection.

Live-check artifacts are intentionally split:

| Artifact | Writer | Meaning |
| --- | --- | --- |
| `live_checks.jsonl` | submit server | Every agent `submit` attempt, hashed answer only |
| `runtime/check_done_reactive.jsonl` | reactive monitor | Each orchestrator-side `_tool_check_done` triggered by an agent `:9091` command |
| `runtime/check_done_polls.jsonl` | polling monitor | Each periodic orchestrator-side `_tool_check_done` call |
| `runtime/live_success.json` | live-success recorder | The first accepted success verdict used by scoring |

## Synchronization Checklist

When changing runtime behavior, keep these areas in sync:

- `cage/experiment/engine/conductor.py` lifecycle changes
- `cage/sandbox/containers.py` Docker networking and env behavior
- `cage/proxy/host.py` and `cage/proxy/sidecar.py` proxy config and logs
- `cage/artifacts/live_success.py` verdict schema and score helper behavior
- `cage/experiment/engine/live/monitor.py` reactive/polling monitor behavior
- `cage/target/services/submit/service.py`, `server.py`, and `client.py`
  for in-container submit lifecycle and permissions
- `cage/config/experiment.py` config defaults and project.yml parsing
- `examples/*/benchmark.py` sample fields consumed by prompt templates
- `examples/*/benchmark.py` score priority when `runtime/live_success.json`
  is present
- `examples/*/prompts/*.j2` and legacy `*_template.txt` prompt templates for
  agent-facing runtime guidance
- `tests/test_orchestrator_live_check.py` for submit and monitor lifecycle behavior
- `tests/test_live_monitor.py` and `tests/test_live_success.py` for live verdicts
- `tests/test_submit_service.py`, `tests/test_submit_server.py`, and
  `tests/test_submit_client.py` for submit lifecycle
- this file for architecture notes

For CVEBench specifically, verify:

- challenge metadata is available in `TargetConfig.challenges`
- `challenge_id` and trial `id` semantics remain distinct when variants exist
- target runtime network is connected before the agent starts
- `target:9091` or `check:9091` tool-call detection does not scan prompt
  history and only reacts to newly emitted tool-call commands
- `_tool_check_done()` can be called from inside the agent container
- `runtime/live_success.json` takes precedence over post-trial
  `runtime/check_done_output.txt`
