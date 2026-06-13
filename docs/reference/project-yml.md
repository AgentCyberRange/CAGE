# Experiment YAML Reference

CAGE commands take an experiment YAML file. Older docs call this file
`project.yml`; release-facing examples can use clearer names such as
`default_web_exploit.yml` and `default_post_exploit.yml`. The file chooses the
benchmark, agents, target runtime, limits, resume policy, and output settings.
The model registry is repo-level configuration in `config/models.yml`.

Benchmark code owns samples, prompts, target setup, scoring, and dashboards.
The experiment YAML owns how you run that benchmark today.

## Minimal Shape

```yaml
project:
  name: agent-pentest-bench
  run_id: smoke-pentest-001

eval:
  benchmark:
    module: ./benchmark.py
    class: AgentPentestBench
    benchmark_root: ./datasets

runtime:
  max_trials_global: 1
  passk: 1
  timeout: 3600
  max_rounds: 100
  max_input_tokens: null
  max_output_tokens: null
  max_cost: null

target:
  enabled: true
  run_mode: remote

proxy:
  enabled: true
  request_timeout: 3600

agents:
  - id: codex
    kind: codex
    models:
      - gpt-5.5
    image: cage/codex:pentestenv
    home: /home/agent/workspace
    max_concurrent: 1
```

## `project`

```yaml
project:
  name: agent-pentest-bench
  run_id: gpt55-l1l2-pass2
```

| Field | Meaning |
|---|---|
| `name` | Human-readable experiment name shown in summaries |
| `run_id` | Optional stable run id; CLI `--run-id` overrides it |

Set a `run_id` for anything you might resume or share. If omitted, CAGE
generates one. Reusing a `run_id` without `--resume` is rejected by default.
Use `cage run ... --force` when you want to archive the previous run and start
fresh with the same id.

## `logging`

```yaml
logging:
  inspect_mode: auto
  terminal_ui: true
  level: INFO
  file_level: DEBUG
  debug_file: false
```

| Field | Meaning |
|---|---|
| `inspect_mode` | `auto`, `on`, or `off`; controls the managed browser inspector |
| `terminal_ui` | Whether to print the small plain progress line |
| `level` | Console log level |
| `file_level` | Run JSONL log level |
| `debug_file` | Whether to write `.cage.debuglog` by default |

Terminal output is deliberately small: normal logs, a startup banner with the
three web inspector base URLs, and one plain progress line. Trial details,
artifacts, model requests, token usage, and dashboards live in the browser
inspector.

## Model Registry

`config/cage.yml` points CAGE at the single repo-level registry:

```yaml
models_file: config/models.yml
```

All release-facing examples use that default. The file is local and ignored by
git; create it from the committed example first:

```bash
cp config/models.example.yml config/models.yml
```

Do not add benchmark-local `models.yml` files; edit `config/models.yml` with
`cage model set` instead.

## `config/models.yml` And `config/models.example.yml`

The committed `config/models.example.yml` documents the complete shape. The
runtime `config/models.yml` uses the same schema:

```yaml
models:
  gpt-5.5:
    provider: openai
    model: gpt-5.5
    agent_model_names: {}
    base_url: https://api.example.com/v1
    api_key: ${OPENAI_API_KEY}
    auth_source: ""
    api_keys: []
    input_cost_per_1m: 0.0
    output_cost_per_1m: 0.0
    timeout: 3600
    max_retries: 2
    extra_headers: {}
```

| Field | Meaning |
|---|---|
| `provider` | `openai`, `vllm`, or `anthropic` |
| `model` | Model name sent upstream |
| `agent_model_names` | Optional per-agent CLI model names, keyed by agent kind |
| `base_url` | API base URL |
| `api_key` | Single API key |
| `api_keys` | List of keys; CAGE pins each trial to one key round-robin |
| `auth_source` | Host credential directory for supported subscription auth |
| `timeout` | Model request timeout |
| `max_retries` | Upstream retry count |
| `input_cost_per_1m` | Input token price per 1M tokens |
| `output_cost_per_1m` | Output token price per 1M tokens |
| `extra_headers` | Headers attached to upstream model requests |

`provider` also defines protocol. `openai` and `vllm` use OpenAI protocol;
`anthropic` uses Anthropic protocol. The proxy can translate when an agent and
model use different protocols.

Keep endpoint identity separate from agent-specific launch strings. For
example, Claude Code can use decorated model names such as
`deepseek-v4-pro[1m]`, but Codex, Qwen Code, Kimi Code, and other agents should
not inherit that suffix. Express that as:

```yaml
models:
  deepseek-v4-pro:
    provider: anthropic
    model: deepseek-v4-pro
    agent_model_names:
      claude_code: deepseek-v4-pro[1m]
    base_url: https://api.deepseek.com/anthropic
    api_key: ${DEEPSEEK_API_KEY}
```

Do not put `max_concurrent` in `config/models.yml`; concurrency belongs to
`agents[].max_concurrent`.

## `eval`

```yaml
eval:
  benchmark:
    module: ./benchmark.py
    class: AgentPentestBench
    benchmark_root: ./datasets
  limit: 3
```

| Field | Meaning |
|---|---|
| `benchmark.module` | Python module containing the benchmark class |
| `benchmark.class` | Optional class name; required if multiple benchmarks exist |
| extra keys | Passed to the benchmark constructor |
| `limit` | Optional sample limit loaded into the benchmark |

CAGE resolves `.cage_runs` under the benchmark module directory, not under an
arbitrary config subdirectory. For `examples/agent_pentest_bench/default_web_exploit.yml`, runs
land in:

```text
examples/agent_pentest_bench/.cage_runs/
```

## `agents`

```yaml
agents:
  - id: codex
    kind: codex
    models:
      - gpt-5.5
    image: cage/codex:pentestenv
    home: /home/agent/workspace
    max_concurrent: 1
    max_rounds: 500
    session_args:
      - -c
      - model_reasoning_effort=xhigh
```

| Field | Meaning |
|---|---|
| `id` | Stable id used by `cage run --agent` |
| `kind` | Registered agent adapter, e.g. `codex`, `claude_code` |
| `model` | Model id from `config/models.yml` |
| `models` | List of model ids; CAGE expands one logical agent across these models |
| `image` | Docker image to run |
| `home` | Agent working directory inside the container |
| `session_args` | Extra CLI args passed to the agent command |
| `shared_paths` | Container paths preserved across stateful trials |
| `skill` | Optional agent skill |
| `plugins` | Optional plugin list |
| `extra_env` | Environment variables injected into the agent |
| `version` | Agent CLI version passed into build/debug flows |
| `max_rounds` | Agent-specific round budget override (same values as `runtime.max_rounds`; absent ⇒ inherit the runtime budget) |
| `context_compaction_threshold` | Proxy/agent compaction threshold |
| `max_concurrent` | Per-agent concurrency cap |

Use `cage agent list` to list registered agent kinds and their default images.

For security benchmarks that run Claude Code inside the isolated target
environment, keep explicit permission bypass args on the agent:

```yaml
agents:
  - id: claude_code
    kind: claude_code
    models:
      - claude-opus-sub
      - claude-opus
      - id: deepseek-v4-pro
        extra_env:
          CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: "1"
          DISABLE_INTERLEAVED_THINKING: "1"
    image: cage/claude-code:pentestenv
    home: /home/agent/workspace
    session_args:
      - --permission-mode
      - bypassPermissions
      - --verbose
```

WebExploitBench and PostExploitBench both need this shape. The agent is inside
the CAGE sandbox, so repeated interactive permission prompts only make the run
less reproducible. Other built-in coding agents wire their own headless
approval mode in the adapter.

## `subjects`

Legacy expansion form:

```yaml
subjects:
  - gpt-5.5

agents:
  - id: codex
    kind: codex
```

If `subjects` is set and an agent does not specify `model`, CAGE expands
agents across subjects. New project files usually put `models` on each agent
class because it is easier to read and maps cleanly to `cage run --agent`.

## `proxy`

```yaml
proxy:
  enabled: true
  request_timeout: 3600
  upstream_http_proxy: http://<host-ip>:7890
  rewrite:
    system: |
      {{ system_raw }}

      Extra system instruction.
```

| Field | Meaning |
|---|---|
| `enabled` | Route agent model calls through the CAGE proxy |
| `request_timeout` | Upstream model request timeout |
| `upstream_http_proxy` | HTTP proxy used by in-container proxy egress |
| `rewrite.system` | Optional Jinja template for system prompt rewrite |

The proxy writes model-call artifacts under each trial's `proxy/` directory.
If the upstream proxy runs on the Docker host, bind it to `0.0.0.0` or a
Docker-reachable host address and point containers at that address. On Linux,
`host.docker.internal` only works in containers that receive the
`host-gateway` mapping.

## `target`

```yaml
target:
  enabled: true
  run_mode: remote
  startup_timeout: 1800
  compose_up_timeout: 3600
  target_scope: per_agent
  parallel_mode: ""
  agent_network_isolation: per_trial_bridge
```

| Field | Meaning |
|---|---|
| `enabled` | Whether target lifecycle is active |
| `run_mode` | `local` or `remote`; most Dockerized benchmarks use `remote` |
| `startup_timeout` | Service readiness timeout |
| `compose_up_timeout` | `docker compose up` timeout |
| `target_scope` | `per_agent`, `per_challenge`, or benchmark default |
| `parallel_mode` | `network`, `alias`, or benchmark default |
| `agent_network_isolation` | `per_trial_bridge` or `trust_server` |

`target.server_url` and `target.embedded` are not user-facing fields. CAGE
spawns a per-run embedded target server and sets the URL internally.

## `runtime`

```yaml
runtime:
  max_trials_global: 4
  max_target_setups: 1
  timeout: 7200
  passk: 3
  max_rounds: 500
  max_input_tokens: null
  max_output_tokens: null
  max_cost: null
  agent_network_mode: bridge
  live_check:
    enabled: false
```

| Field | Meaning |
|---|---|
| `max_trials_global` | Global simultaneous trial cap |
| `max_target_setups` | Concurrent target setup/readiness cap |
| `timeout` | Per-trial wall-clock timeout in seconds; `0` means unlimited |
| `on_failure` | Failure policy; default is `continue` |
| `agent_network_mode` | Agent container network mode |
| `max_rounds` | Model-call round budget. See the value table below. |
| `max_input_tokens` | Cumulative input token budget per trial; `null`/unset means unlimited |
| `max_output_tokens` | Cumulative output token budget per trial; `null`/unset means unlimited |
| `max_cost` | Cumulative per-trial USD budget; `null`/unset means unlimited |
| `passk` | Independent attempts per sample |
| `store_proxy` | Whether to store proxy artifacts |
| `live_check` | Online success checking behavior |

`max_rounds` counts successful non-compact agent decision rounds. Failed
upstream calls and compaction rewrites do not spend the budget.
`max_input_tokens`, `max_output_tokens`, and `max_cost` are enforced by the
per-trial proxy from observed model usage. If the provider does not report
cost, CAGE estimates `max_cost` from `config/models.yml`:
`input_cost_per_1m` and `output_cost_per_1m`.

### `max_rounds` values

`max_rounds` accepts more than a plain integer. The value at each level
(agent → runtime) is resolved as:

| Value | Meaning |
|---|---|
| absent / `null` / `unlimited` | **Unlimited** — no per-trial round cap. This is the default when `max_rounds` is not set. |
| `-1` (or `benchmark`) | Use the **benchmark's built-in default** (e.g. CyberGym's 100); unlimited if the benchmark declares none. |
| `0` | No rounds (the agent gets zero decision rounds). |
| `N` (positive) | Exactly `N` rounds. |

Precedence: an explicit per-agent `max_rounds` wins, then `runtime.max_rounds`,
then (for `-1`) the benchmark sample default. The CLI `--max-rounds` accepts the
same forms (`--max-rounds unlimited`, `--max-rounds 100`, `--max-rounds -1`).

> **Every run must be able to stop.** If the resolved round budget is
> **unlimited** and none of `timeout` (> 0), `max_cost`, `max_input_tokens`, or
> `max_output_tokens` is set, CAGE **refuses to start the run** (and
> `--dry-run` reports the same error), because a trial could run forever. Set at
> least one finite termination condition. For example CyberGym defaults to
> `max_rounds: unlimited` (the agent self-terminates via `submit.sh`) but keeps
> `timeout: 3600` as the hard stop.

## `runtime.live_check`

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
      confirm_polls: 2
```

Use live checks when a benchmark can determine success during the trial.
Disable them when scoring should happen only after the agent stops.

## `resume`

```yaml
resume:
  retry_reasons:
    - target_unavailable
    - model_bad_gateway
  max_attempts: 3
```

| Field | Meaning |
|---|---|
| `retry_reasons` | Extra termination reasons to retry under `--resume` |
| `max_attempts` | Total attempts per trial; `0` means unlimited |

Preview resume:

```bash
cage run examples/agent_pentest_bench/default_web_exploit.yml --run-id <run_id> --resume --dry-run
```

## `output`

```yaml
output:
  dashboard_prompt: false
  dashboard_output: false
  dashboard_reasoning: false
  csv_prompt: false
  csv_output: false
  csv_reasoning: false
```

Use these fields to keep dashboards and CSVs small for large runs. Raw trial
artifacts still remain on disk.

## `admission`

```yaml
admission:
  enabled: true
  memory_pause_at: 0.80
  memory_resume_at: 0.70
  poll_seconds: 3
  log_every_seconds: 30
```

Admission control pauses new trial launches when host memory pressure is high.

## `judge`

```yaml
judge:
  id: judge-model
  temperature: 0.0
  max_tokens: 4096
```

Use when a benchmark scorer needs a judge model.

## `hooks`

Hooks are benchmark/project-specific extension points. They are loaded by
`cage.experiment.engine.hooks.load_hooks()` and should be documented by the benchmark that
uses them.

## CLI Overrides

The most common overrides:

```bash
cage run examples/agent_pentest_bench/default_web_exploit.yml --run-id smoke-001
cage run examples/agent_pentest_bench/default_web_exploit.yml --agent codex
cage run examples/agent_pentest_bench/default_web_exploit.yml --sample pb-comfyui
cage run examples/agent_pentest_bench/default_web_exploit.yml --prompt-level l0,l1
cage run examples/agent_pentest_bench/default_web_exploit.yml --max-sample-num 5
cage run examples/agent_pentest_bench/default_web_exploit.yml --run-id smoke-001 --resume --dry-run
```

## Related Docs

- First run: [../getting-started/](../getting-started/)
- Running experiments: [../running-experiments/](../running-experiments/)
- CLI reference: [cli.md](cli.md)
