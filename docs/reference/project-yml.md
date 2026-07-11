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
| `max_context_size` | Endpoint's real context window in tokens (alias `context_window_size`); `null`/unset ⇒ each agent keeps its CLI's own default. The Claude Code CLI has no window knob, so it cannot honour this |
| `reserved_context_size` | Headroom (tokens) an agent reserves for its next response; `null`/unset ⇒ CLI default |
| `rl_reward_sink` | RL-training URL. When set, this model runs in RL mode: every LLM call carries an `X-Trial-Id` header and each finished trial's reward is POSTed here. Empty ⇒ ordinary model. Also settable via `cage model set --rl-reward-sink` |
| `extra` | Inference knobs routed into the upstream OpenAI/vLLM request body — see below |

`provider` also defines protocol. `openai`, `vllm`, and `sglang` use OpenAI
protocol; `gemini`/`google` use the Google protocol; everything else
(`anthropic`, …) uses Anthropic protocol. The proxy can translate when an agent
and model use different protocols. Note: `cage model set --provider` only
accepts `openai|anthropic|vllm`; `gemini`/`google`/`sglang` must be hand-edited
into `config/models.yml`.

### Model `extra` (upstream inference knobs)

`extra` carries per-request body fields the proxy merges into an OpenAI/vLLM
upstream call — letting the registry pin inference knobs the agent CLI cannot
express itself (e.g. Qwen's `enable_thinking`, or recommended sampling). These
are applied even on the anthropic→openai translation path, so a Claude-Code-driven
run can talk to a vLLM model with a fixed inference config.

```yaml
models:
  qwen3-coder:
    provider: vllm
    model: qwen3-coder
    base_url: http://<host>:8000/v1
    extra:
      enable_thinking: false          # shorthand → chat_template_kwargs
      temperature: 0.7                 # sampling: top_p/top_k/*_penalty too
      chat_template_kwargs: {}         # merged with enable_thinking
      extra_body: {}                   # raw passthrough (lowest precedence)
```

Recognized `extra` keys: `extra_body` (raw dict), the sampling keys
`temperature`/`top_p`/`top_k`/`presence_penalty`/`frequency_penalty`,
`chat_template_kwargs`, and the `enable_thinking` shorthand. `{}` ⇒ nothing
injected.

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
| `sample_slice` | Optional Python-style slice over the ordered (id-filtered) sample list — e.g. `":100"` first 100, `"-100:"` last 100, `"100:200"` a window, `"::2"` every other, `"5"` the 6th, `"-1"` the last |

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
| `kind` | Registered agent adapter, e.g. `codex`, `claude_code`. Omit when using `source` |
| `source` | Path to a **custom-agent** directory (holding `agent.yml` + code) loaded instead of a built-in `kind`. Resolved relative to the experiment file's directory |
| `params` | Optional dict filling the custom agent's manifest params (also via CLI `--param KEY=VALUE`) |
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
      - deepseek-v4-pro
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

`models` is a plain list of model ids; CAGE runs one independent
`agent × model` trial matrix over them. `extra_env` (and other launch knobs)
belong at the **agent** level, not on a per-model entry. A `models` entry may
be a dict only to carry `id`/`model` or a `sources` pool (see below) — no other
per-model launch keys are read there.

WebExploitBench and PostExploitBench both need this shape. The agent is inside
the CAGE sandbox, so repeated interactive permission prompts only make the run
less reproducible. Other built-in coding agents wire their own headless
approval mode in the adapter.

### Multi-endpoint models (`sources`)

A `models` entry may declare `sources` — several registered endpoints the run
round-robins across per trial (load balancing over identical model deployments).
When `sources` is set, the entry **must** set an explicit `id` (the logical key
that groups the run dir / labels / scores; it need not itself be registered),
and all sources must share one protocol.

```yaml
agents:
  - id: qwen_code
    kind: qwen_code
    models:
      - id: qwen3-coder            # logical key
        sources:
          - qwen3-coder-node-a:6   # `id:N` pins per-source concurrency
          - qwen3-coder-node-b:6
          - id: qwen3-coder-node-c # dict form, default even share
            concurrency: 4
```

Each source is `<model_id>`, the `<model_id>:N` short form, or
`{id: <model_id>, concurrency: N}`. Absent concurrency ⇒ the source takes the
default even share of the run's total.

### Custom agents (`source`)

Instead of a built-in `kind`, an agent entry may point `source` at a
self-contained custom-agent directory (its own `agent.yml` manifest + code):

```yaml
agents:
  - id: cairn
    source: ../../cage/agents/custom/cairn   # dir with agent.yml
    params:
      workers: 4                              # fills a manifest param
    models:
      - glm-5.1
```

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
| `network_mode` | Compose launch mode: `""` \| `compose_project_local` \| `shared_external`. Challenge-declared value wins; empty ⇒ server default (`compose_project_local`) |
| `exposure_mode` | Service exposure: `""` \| `host_ports` \| `internal`. Challenge-declared value wins; empty ⇒ server default (`host_ports`) |
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
  agent_network_mode: host
  store_proxy: true
  chunk_size: null
  max_trial: null
  wait_for_model: false
  wait_timeout: 0.0
  wait_interval: 10.0
  live_check:
    enabled: false
```

| Field | Meaning |
|---|---|
| `max_trials_global` | Global simultaneous trial cap |
| `max_target_setups` | Concurrent target setup/readiness cap |
| `timeout` | Per-trial wall-clock timeout in seconds; `0` means unlimited |
| `on_failure` | Failure policy; default is `continue` |
| `agent_network_mode` | Agent container network mode; resolved default is `host` |
| `max_rounds` | Model-call round budget. See the value table below. |
| `max_input_tokens` | Cumulative input token budget per trial; `null`/unset means unlimited |
| `max_output_tokens` | Cumulative output token budget per trial; `null`/unset means unlimited |
| `max_cost` | Cumulative per-trial USD budget; `null`/unset means unlimited |
| `passk` | Independent attempts per sample |
| `store_proxy` | Whether to store proxy artifacts. Effective default when unset is **`true`** (the loader defaults the yaml key on) |
| `chunk_size` | Optional trial-batching chunk size; `null`/unset processes all trials in one batch |
| `max_trial` | Per-invocation execution cap: run only trials whose global index is `< max_trial`, leaving the rest pending (batch with `--resume`); `null`/unset runs all |
| `wait_for_model` | Poll self-hosted (`vllm`/`sglang`) endpoints until they answer before starting; CLI `--wait-for-model` overrides. Default `false` |
| `wait_timeout` | Readiness-wait timeout in seconds; `0` waits indefinitely (default `0`) |
| `wait_interval` | Seconds between readiness polls (default `10`) |
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
  select:
    id_matches: "arvo_1[0-9]{3}"     # only re-run trials whose id matches
  keep_if:
    min_rounds: 100                  # don't re-run trials that got this far
    min_duration_s: 600
    id_matches: "arvo_1507"
```

| Field | Meaning |
|---|---|
| `retry_reasons` | Extra termination reasons to retry under `--resume` |
| `max_attempts` | Total attempts per trial; `0` means unlimited |
| `select.id_matches` | Positive id-regex gate: only trials whose id matches are eligible for re-run |
| `keep_if.min_rounds` | Veto: keep (don't re-run) an otherwise-retryable trial that reached at least this many rounds |
| `keep_if.min_duration_s` | Veto: keep a trial that ran at least this long (seconds) |
| `keep_if.id_matches` | Veto: keep trials whose id matches this regex |

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

`--prompt-level` is **not** a core `cage run` flag — it is a *benchmark-declared*
CLI option (a `BenchmarkOption` the benchmark registers via `cli_options()`,
mapped to a project.yml field). It only exists when the benchmark implements it
(agent_pentest_bench does); against a benchmark that doesn't, the flag is
unknown. Custom-agent params use the core `--param KEY=VALUE` flag instead.

## Related Docs

- First run: [../getting-started/](../getting-started/)
- Running experiments: [../running-experiments/](../running-experiments/)
- CLI reference: [cli.md](cli.md)
