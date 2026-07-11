# Core Class Reference

This page is a human-readable map of the core classes a framework or benchmark
developer will touch. It is not a generated API reference; it explains what
each class owns and where it sits in the three-layer architecture.

## Layer 3: Experiment Configuration

### `ExperimentRun`

Defined in `cage/experiment/engine/run_context.py`.

The single live run object returned by `resolve()` (in `cage/config/experiment.py`).
It is both the parsed project and the run context the conductor threads through
trial execution. It carries:

- project name and project file path;
- benchmark instance and benchmark root;
- loaded model registry;
- agent instances;
- proxy, runtime, target, output, logging, admission, and resume config;
- optional judge and hooks;
- run-context fields wired at run start (run id, `.cage_runs` root, scheduler,
  cleanup registry, reporter).

The conductor consumes `ExperimentRun`; benchmark code should usually not
mutate it directly.

### `ModelConfig`

Defined in `cage/models/endpoint.py`.

Represents one model endpoint from `config/models.yml`.

Important fields:

- `id`: local model id used by project files;
- `provider`: `openai`, `vllm`, `sglang`, `anthropic`, `gemini`, or `google`;
- `model`: endpoint/upstream model name;
- `agent_model_names`: optional per-agent CLI aliases;
- `base_url`;
- `api_key` or `api_keys`;
- `auth_source`;
- `extra_headers`;
- `timeout` (default `360`) and `max_retries`;
- `input_cost_per_1m` / `output_cost_per_1m`: optional cost accounting;
- `max_context_size` / `reserved_context_size`: the endpoint's real context
  window and the response headroom to reserve. `None` ⇒ undeclared (agent falls
  back to its CLI default). The Claude Code CLI has no window knob, so it cannot
  honour these;
- `rl_reward_sink`: URL that puts this model in RL mode (per-call `X-Trial-Id`
  header + per-trial reward POST). Empty ⇒ ordinary model;
- `extra`: free-form inference knobs routed off the registry (sampling,
  `enable_thinking`, `chat_template_kwargs`, `extra_body`).

Important behavior:

- `protocol` maps provider to API protocol (`vllm`/`sglang` → `openai`;
  `gemini`/`google` → `google`; otherwise `anthropic`);
- `model_name_for_agent(agent_name)` returns an agent-specific CLI model name
  when configured, otherwise the endpoint model name;
- `needs_translation(agent_protocol)` tells the proxy whether translation is
  required;
- `upstream_extra_body` builds the per-request body fields merged into an
  OpenAI/vLLM upstream call from `extra`;
- `is_local_endpoint` is true for `vllm`/`sglang` (eligible for
  `--wait-for-model` readiness polling);
- `rl_enabled` is true iff `rl_reward_sink` is set;
- `api_key_pool` returns all keys available for the endpoint.

### `ProxyConfig`

Defined in `cage/config/experiment.py`.

Controls model proxy behavior:

- `enabled`;
- `rewrite_system`;
- `request_timeout`;
- `upstream_http_proxy`.

### `TargetConfig`

Defined in `cage/config/experiment.py`.

Controls target lifecycle behavior:

- whether target runtime is enabled;
- local/remote mode;
- startup and compose timeouts;
- target scoping and parallel mode;
- agent network isolation.

The target server URL is runtime-owned and should not be set in `project.yml`.

### `ExecutionConfig`

Defined in `cage/config/experiment.py`.

Controls scheduling:

- `max_trials_global` (0 = unlimited global cap);
- `max_target_setups`;
- `timeout`;
- `on_failure` (default `continue`);
- `chunk_size`;
- `agent_network_mode` (default `host`);
- `max_rounds` (`"unlimited"` default | `-1` benchmark default | `0` | `N`);
- `max_input_tokens`;
- `max_output_tokens`;
- `max_cost`;
- `store_proxy`;
- `wait_for_model` / `wait_timeout` / `wait_interval`: readiness polling for
  self-hosted endpoints;
- `passk`: independent attempts per sample (pass@k);
- `max_trial`: per-invocation execution cap that leaves the trial plan intact
  (resume-compatible);
- `live_check`.

## Layer 2: Benchmark API

### `Benchmark`

Defined in `cage/benchmarks/base.py`.

Base class for all benchmark adapters. Required methods:

- `iter_samples()`: yield sample dicts;
- `prepare_trial(container, sample, workspace_dir)`: write files or prepare the
  agent workspace before launch;
- `build_prompt(sample)`: return the prompt string;
- `scorer()`: return the default `Scorer`.

Live, target-side evidence gathering lives on the `Scorer` (`gather()`, below),
not on the `Benchmark`.

Lifecycle hooks (run in this order around scoring):

- `setup()` and `teardown()`;
- `on_agent_finish(container, sample, trial_dir)`: runs the instant the agent
  stops, BEFORE the scoring gather — materialize agent output to the host here
  (e.g. copy the workspace) so live/serve-only gather can read it;
- `on_trial_complete(container, sample, trial_dir)`: runs AFTER the gather,
  container still alive — target post-mortem diagnostics only (too late to feed
  gather).

Other optional hooks:

- `container_image_override()`: Docker image this benchmark requires for the
  agent container (`None` ⇒ use the agent's configured image);
- `reward(result)`: scalar RL reward in `[0, 1]` (default: primary numeric
  score, clamped); only consulted when the model declares `rl_reward_sink`;
- `build_targets(samples, ...)`: build benchmark target images for samples;
- `build_dashboard(run_dir)`;
- `cli_options()`: benchmark-owned `cage run` options (`BenchmarkOption`);
- `variant_display_axes()`: active per-run variant axes for the run summary;
- live-check hooks: `live_check_triggers()`, `live_check_confirm_polls()`,
  `live_check_polling_interval()`, `validate_live_verdict()`.

Live-check capability flags (the framework branches on these, never on a
benchmark name; all default `False`):

- `needs_check_service`: live verification needs an in-container check daemon;
- `uses_builtin_check`: the check is built into the agent image (no daemon);
- `needs_submit_service`: live verification needs an in-container flag-submit
  daemon.

### `Scorer`

Defined in `cage/scoring/scorer.py`.

Base class for scoring one trial. Implement:

```python
def score(self, ctx: ScoringContext) -> dict[str, Score]:
    ...
```

Attributes:

- `name`: scorer name (default `""`);
- `strategy` controls when it runs:
  - `per_trial`: score immediately after each trial (the default);
  - `post_run`: score after all trials for an agent finish.

Optional live half:

```python
def gather(self, runtime: GatherRuntime) -> str:
    ...
```

`gather()` is the LIVE evidence-gathering phase — the only scoring step that
requires the target to be up. It observes the running target and returns a
serializable evidence string that `score()` later consumes (via
`ctx.check_done_output` / `ctx.live_payload`). It must run before target
teardown. The default returns `""` (no live evidence).

### `GatherRuntime`

Defined in `cage/scoring/scorer.py`.

Dataclass carrying the inputs to `Scorer.gather()`, decoupled from "the agent
container":

- `sample`: how to reach the *target* (project name to docker-exec into the
  target's containers, and host-published scoring endpoints);
- `agent_output_dir` (`Path | None`, default `None`): host directory holding
  the agent's produced output — set in serve-only mode;
- `container` (`Any`, default `None`): the live agent container, present in
  `cage run` (used only as the agent-output source when `agent_output_dir` is
  absent — never to reach the target).

### `ScoringContext`

Defined in `cage/scoring/context.py`.

The data available to a scorer:

- `trial_id`;
- `trial_index`;
- `sample`;
- agent `output`;
- `exit_code`;
- optional `trial_dir`;
- `run_dir`;
- `canonical_trial_id`;
- `artifact_paths`: canonical artifact-kind → path map;
- `record_metadata`;
- `live_payload`: the live evidence your scorer's `gather()` returned.

When `trial_dir` (or the canonical artifact refs) is set, it lazily loads:

- `prompt`;
- `proxy_log`;
- `live_success`;
- `check_done_output` (returns `live_payload` when present);
- `metadata`.

Use `ScoringContext.from_trial_dir()` for offline scoring.

### `CompositeScorer`

Defined in `cage/scoring/scorer.py`.

Runs multiple scorers and merges their score dictionaries. Use it when a
benchmark ships several default metrics.

### `Dashboard`, `Section`, `Column`, `Stat`

Defined in `cage/artifacts/dashboard.py`.

These classes let benchmarks emit dashboard data without writing HTML.

Section kinds:

- `summary`;
- `table`;
- `note`.

The orchestrator writes the result to:

```text
<run_dir>/dashboard_view.json
```

The web inspector renders it generically.

## Layer 1: Agent Runtime

### `AgentType`

Defined in `cage/agents/base/definition.py`.

Static adapter for an installed agent CLI. Class attributes:

- `name`;
- `state_paths`: shared-state paths (non-empty ⇒ stateful agent);
- `default_image`;
- `dockerfile`;
- `plugin_images`.

Required methods:

- `install_command(version="latest")`;
- `build_launch_command(prompt, *, model, max_rounds=-1, proxy_url="")` —
  everything after `prompt` is keyword-only;
- `parse_output(result: ExecResult) -> str`.

Optional hooks:

- `image_for_variant(variant)`: local image ref for a build/runtime variant;
- `env_vars(*, proxy_url, model, container=None, home_dir="/home/agent",
  workspace_dir="", max_rounds=-1, context_compaction_threshold=None)` —
  `context_compaction_threshold` is `None` unless the user set it (meaning "do
  not impose a threshold");
- `version_command()`;
- `artifact_files()`: `(container_path, artifact_filename)` pairs to pull after
  the agent finishes;
- `setup_container(container, *, home_dir, model=None)`;
- `container_resources(*, home_dir, model)`;
- `validate_auth(model)`;
- `host_run_services(model, *, http_proxy="")`: host-side processes required
  while a run is live;
- `install_plugin(container, *, name, home_dir, agent_id="")`.

Agent adapters live under `cage/agents/`.

### `AgentInstance`

Defined in `cage/agents/base/definition.py`.

Runtime binding of an `AgentType` to one project config entry:

- agent id;
- model;
- image;
- home directory;
- session args;
- shared state paths;
- plugins and extra env;
- max rounds and max concurrency.

`label()` builds the run directory segment:

```text
<agent_id>:<model_id>:<stateful|stateless>
```

### `AgentContainerResources`

Defined in `cage/agents/base/resources.py`.

Extra Docker resources an agent needs before container start:

- `volumes`: host → container bind mounts;
- `group_add`: extra supplementary groups;
- `privileged`: launch the trial container with `--privileged` (off by default;
  needed by Docker-in-Docker agents).

### `HostRunService`

Defined in `cage/agents/base/resources.py`.

A host-side background process an agent needs while a run is live (started once
at run start, terminated at run end, including interrupted runs):

- `name`;
- `argv`;
- `env`;
- `dedup_key`: lets multiple agents share one equivalent service declaration.

## Runtime And Storage

### `RunStorage`

Defined in `cage/artifacts/run_storage.py`.

Owns the filesystem layout for one run:

- config snapshots;
- agent version;
- initial state;
- trial directories;
- prompts;
- task outputs;
- scores;
- metrics and summaries.

### `Container`

Defined in `cage/sandbox/containers.py`.

Thin wrapper around Docker operations used by the orchestrator and debug flows:

- start/stop container;
- exec commands;
- copy files;
- manage mounts, env vars, and workdir.

### Proxy Recorder Classes

Defined in `cage/proxy/sidecar.py` and `cage/proxy/host.py`.

These classes record model requests, responses, errors, usage, and progress.
The in-container proxy is the experiment path; host proxy code supports debug
and compatibility flows.

## Target Server

The target server package under `cage/target/` owns challenge discovery
and launch:

- adapters normalize benchmark challenge definitions;
- launch workflow materializes Docker Compose runtime files;
- network helpers isolate agent-facing and internal services;
- cleanup code sweeps by labels and namespace.

Important concepts:

- `cage.run_id` labels tie resources to experiment artifacts;
- benchmark sources point at roots containing challenge indices;
- target adapters convert benchmark-specific metadata into normalized launch
  specs.

## Web Inspector

The web package under `cage/web/` owns:

- run discovery;
- artifact loading;
- live progress fallback;
- dashboard rendering;
- trial detail rendering;
- trajectory parsing from proxy logs.

Framework web code should stay generic. Benchmark-specific meaning belongs in
`Benchmark.build_dashboard()`.

## Related Docs

- Framework guide: [../developing-cage/](../developing-cage/)
- Benchmark authoring: [../writing-benchmarks/](../writing-benchmarks/)
- Project config: [project-yml.md](project-yml.md)
- CLI: [cli.md](cli.md)
