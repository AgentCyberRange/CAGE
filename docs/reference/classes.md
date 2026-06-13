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
- `provider`: `openai`, `vllm`, or `anthropic`;
- `model`: endpoint/upstream model name;
- `agent_model_names`: optional per-agent CLI aliases;
- `base_url`;
- `api_key` or `api_keys`;
- `auth_source`;
- `extra_headers`;
- `timeout` and `max_retries`.

Important behavior:

- `protocol` maps provider to API protocol;
- `model_name_for_agent(agent_name)` returns an agent-specific CLI model name
  when configured, otherwise the endpoint model name;
- `needs_translation(agent_protocol)` tells the proxy whether translation is
  required.

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

- `max_trials_global`;
- `max_target_setups`;
- `timeout`;
- `agent_network_mode`;
- `max_rounds`;
- `max_input_tokens`;
- `max_output_tokens`;
- `max_cost`;
- `passk`;
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

Optional hooks:

- `setup()` and `teardown()`;
- `on_trial_complete(container, sample, trial_dir)`;
- `check_done(container, sample)`;
- `build_dashboard(run_dir)`;
- live-check hooks such as `live_check_triggers()` and
  `validate_live_verdict()`.

### `Scorer`

Defined in `cage/scoring/scorer.py`.

Base class for scoring one trial. Implement:

```python
def score(self, ctx: ScoringContext) -> dict[str, Score]:
    ...
```

`strategy` controls when it runs:

- `per_trial`: score immediately after each trial;
- `post_run`: score after all trials for an agent finish.

### `ScoringContext`

Defined in `cage/scoring/scorer.py`.

The data available to a scorer:

- `trial_id`;
- `trial_index`;
- `sample`;
- agent `output`;
- `exit_code`;
- optional `trial_dir`;
- optional live payload.

When `trial_dir` is set, it lazily loads:

- `prompt`;
- `proxy_log`;
- `live_success`;
- `check_done_output`;
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

Static adapter for an installed agent CLI. Required methods:

- `install_command(version)`;
- `build_launch_command(prompt, model, max_rounds, proxy_url)`;
- `parse_output(result)`.

Optional hooks:

- `env_vars()`;
- `setup_container()`;
- `container_resources()`;
- `validate_auth()`;
- plugin install hooks.

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

Defined in `cage/agents/base/definition.py`.

Allows an agent adapter to request pre-start Docker resources such as volumes
or extra groups.

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
