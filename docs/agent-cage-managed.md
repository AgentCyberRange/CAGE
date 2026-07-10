# Adding a New Agent to CAGE

**English** · [中文](agent-cage-managed-CN.md)

Cage evaluates an *already-built* agent. "Adding an agent" means telling Cage how
to **launch** yours and **read back its answer** — then
`cage run <benchmark> --agent <your-agent>` scores it like any other. You are not
writing a benchmark, and (almost always) not editing the framework.

> Cage owns the runtime: it launches your agent inside a sandboxed container and
> intercepts every model call through an in-container proxy. Your agent does not
> reach out to Cage; Cage runs your agent. (This is the opposite of "bring your
> own agent that calls our API" platforms — worth keeping in mind if that's your
> mental model.)
>
> **Want the other model** — keep your own harness and have your agent pull
> targets over an API (`list → launch → attack → submit → close`), with no
> integration into Cage? That's [Benchmark-Only (Serve) Mode](agent-serve-mode.md).
> You trade the recorded trajectory for zero integration — see that doc's
> comparison table.

## Pick a path (30 seconds)

| Path | You provide | Framework code? | Pick when |
|---|---|---|---|
| **A · Custom agent** (manifest) | a directory with your code + an `agent.yml` | **none** | your agent is any program/CLI you can start with one command |
| **B · Native adapter** (Python) | a `cage/agents/<name>/` `AgentType` subclass + one register line | yes, inside `cage/` | you're wrapping a third-party coding-agent CLI (Claude Code, Qwen, …) as a reusable, first-class type |

Most people want **Path A** — zero framework code, one generic interpreter
(`cage/agents/custom/`) runs any manifest. Reach for **Path B** only when you
need per-trial control the manifest can't express (rewriting a CLI's config file
each trial, plugin install, subscription auth, protocol selection).

If making your agent work pushes you to edit `cage/sandbox/`, `cage/experiment/`,
or `examples/*/benchmark.py`, stop — the framework is missing an abstraction, not
your agent needing a special case.

## Contents

- [Path A — Custom agent (a manifest, no framework code)](#path-a--custom-agent-a-manifest-no-framework-code)
- [Path B — Native adapter (Python `AgentType`)](#path-b--native-adapter-python-agenttype)
- [Reference — proxy gotchas & troubleshooting](#reference--proxy-gotchas--troubleshooting) *(skip on first read)*
- [Never touch](#never-touch)

---

# Path A — Custom agent (a manifest, no framework code)

Use this for **your own** agent — anything you start with a shell command that
reads a prompt, talks to a model, and prints an answer. You write **zero**
framework code: one generic interpreter drives any agent from its `agent.yml`.

**Real examples in this repo:** [`cage/agents/custom/cairn/`](../cage/agents/custom/cairn/README.md)
(a manifest agent with its own image) and the LangGraph walkthrough in
[`langgraph-langchain-agent.md`](langgraph-langchain-agent.md).

## The shape

A custom agent is a **self-contained directory**: your code plus an `agent.yml`
that declares how Cage runs it.

```
my-agent/                # the agent's own directory
├── agent.yml            # the manifest (below)
├── my_agent/            # your code
└── requirements.txt
```

## The manifest — `agent.yml`

```yaml
name: my_agent                    # label / run-dir name (defaults to the dir name)
image: cage/custom-langgraph:base # runtime base (deps only); your code is copied in
command: >-                       # the ONE launch command; {tokens} are filled by Cage
  python3 -m my_agent {workspace_dir}
  --model {model_name} --max-iterations {max_rounds} --instruction {task_instruction}
env:                              # extra env Cage injects (also templated)
  OPENAI_BASE_URL: "{base_url}"   # point your model client at the sidecar proxy
  OPENAI_API_KEY: "{api_key}"
  OPENAI_MODEL: "{model_name}"
workdir: .                        # optional: cwd for the command, relative to the copied source
output: stdout                    # optional: stdout (default) | {json_field: <key>}
state_paths: []                   # optional: dirs to snapshot for stateful runs
params:                           # optional: your agent's OWN knobs (see "Your own params")
  planning_depth: 3               #   e.g. your planner's depth — override per agent or per run
```

Only `image` and `command` are required. Everything else has a default.

## Tokens Cage fills

These `{token}` names are **reserved** — Cage substitutes them into `command`
and `env`. Any other `{placeholder}` is your own param (next section).

| Token | Value |
|---|---|
| `{task_instruction}` | the benchmark's task for the agent (shell-quoted automatically) |
| `{model_name}` | the selected model name (honours per-agent overrides) |
| `{base_url}` | the endpoint your client should call = the in-container sidecar proxy (so calls are intercepted). OpenAI-protocol models get a `/v1` suffix; Anthropic models don't. |
| `{api_key}` | the model's API key |
| `{max_rounds}` | the per-trial round budget — **the** rounds knob (set via `--max-rounds` / project.yml `max_rounds`). Map it straight to your agent's own iteration/turn flag, e.g. `--max-iterations {max_rounds}`; don't invent a second param for rounds. |
| `{workspace_dir}` | the task workspace path (`/home/agent/workspace`) |
| `{model.<field>}` | any field of the model's `config/models.yml` entry, including `{model.extra.<key>}` — how per-model knobs flow without Cage hardcoding them |

An unknown `{token}` fails at config-load (fast), never mid-trial.

## Your own params

Any `{placeholder}` that isn't a reserved token is a **user param** — a knob Cage
has *no* concept of (planning depth, a reflection toggle, a feature flag), with no
Cage code to add. Its value comes from the first source that sets it, later winning:

1. `params:` in `agent.yml` — the author's default.
2. `agents[].params` in the experiment yaml — per-benchmark override.
3. `cage run … --param KEY=VALUE` — per-run override (repeatable).

```yaml
# experiment yaml — override the manifest default for this benchmark
agents:
  - id: my_agent
    source: ../../path/to/my-agent
    models: [{ id: nex-n2 }]
    params:
      planning_depth: 5
```

```bash
cage run examples/cybergym/default_cybergym.yml --agent my_agent --param planning_depth=8
```

**One knob per concept — don't re-express a Cage-owned concept as a param.**
Rounds, model, base_url, api_key, workspace are Cage's; each has exactly one
canonical knob (rounds via `max_rounds` / `--max-rounds`, model via the model
config). A `param` that reuses a reserved token name is **rejected at load** — map
the reserved `{token}` in your `command` instead. Wall-clock is the same: it's
Cage's single trial timeout (`runtime.timeouts.trial_timeout_s`); never self-time
inside your agent. (`--param` needs an unambiguous agent — pass `--agent <id>`
when the run has more than one.)

## What Cage does per trial

1. Loads `<source>/agent.yml`.
2. Copies the whole source directory into the container at `/opt/cage-agent/src`
   (a `docker cp` — outside the workspace, so the per-trial reset never wipes it).
   Edit your code and re-run with no image rebuild; nothing is bind-mounted.
3. Fills the `{tokens}` in `command` and `env` from the model + proxy + prompt.
4. Runs the command in `workdir`, as the `agent` user.
5. Reads the answer per `output` (default: stdout).

## Wire it into a benchmark

In any benchmark's `agents:` list, an entry with a `source:` **is** a custom
agent (no `kind:` needed). `source` resolves relative to the experiment file.

```yaml
agents:
  - id: my_agent                 # run-dir / label (optional; defaults to manifest name)
    source: ../../path/to/my-agent
    models:
      - id: nex-n2               # an id from config/models.yml
    max_concurrent: 3
```

## Build the image

The manifest's `image` carries only framework deps (e.g. langgraph), never your
code. Most agents reuse a shared base image, built once:

```bash
docker build -f docker/custom_langgraph/Dockerfile -t cage/custom-langgraph:base .
```

If your agent needs its own image (a custom build, or a multi-image bake a single
`docker build` can't express), declare it in the manifest and let
`cage agent build` run it — one command, same as every agent:

```yaml
# agent.yml
build:
  script: docker/my_agent/build.sh   # relative to repo root
```

```bash
cage agent build --agent my_agent
```

(See `cage/agents/custom/cairn/` for a real multi-image example.)

## Constraints

- **Your model client must honour `{base_url}`** (or the `OPENAI_BASE_URL` /
  `ANTHROPIC_BASE_URL` env Cage exports). Hardcode an endpoint and you bypass
  interception. LangChain's `ChatOpenAI()` reads `OPENAI_BASE_URL` by default, so
  usually setting the env just works.
- **No docker-in-docker** (unless you set `privileged: true` and mean it). Your
  agent already runs inside Cage's container; spawning its own Docker adds
  nested-container overhead.
- **Copying solves *code* iteration, not *dependency* iteration** — a new pip dep
  means rebuilding the base image.
- **`output`** supports `stdout` (default) and `{json_field: <key>}` (parse stdout
  as JSON, take that key). Reading a result *file* is not supported.

## Free observability (LangChain / LangGraph)

If your agent is built on LangChain / LangGraph, the runtime base auto-attaches a
global callback (via `CAGE_TRACE`, which Cage sets) that stamps the current
**LangGraph node** on every model request. The proxy records it, so the
inspector's trajectory becomes node-aware for free — no tracing code in your
agent:

- a **node route** strip (e.g. `prepare → global_map → candidate_dev → …`) and a
  per-step **node badge** — the real graph, not a guess;
- each node's **system + user prompt**, surfaced at its first appearance;
- for tools that run in plain Python, the tool **result** paired under each action.

Two caveats: a node that issues no model request (deterministic Python) won't
appear — the trajectory is built from the requests the proxy sees; and a new pip
dependency means rebuilding the base image.

## Files you touch (Path A)

```
path/to/my-agent/agent.yml           # the manifest, beside your code
examples/<bench>/default_<bench>.yml  # +1 agents: entry with `source:`
docker/custom_langgraph/Dockerfile    # a runtime base (reuse across custom agents)
```

Nothing in `cage/` changes — the generic interpreter already handles any manifest.

---

# Path B — Native adapter (Python `AgentType`)

The heavier path: wrapping a third-party coding-agent CLI (Claude Code, Codex,
Hermes, Qwen Code, Kimi CLI, …) as a registered `AgentType` so `cage run` can
drive it. The whole integration is **Layer 1 framework + Layer 2 image** — no
benchmark code ever changes.

**Adapters to copy from:** `cage/agents/qwen_code/agent.py` (env-var config) and
`cage/agents/kimi_code/agent.py` (config-file rewrite).

## Before you code — decide upfront

Five things to nail down first. Getting them wrong wastes a build cycle (each
image is ~22 GB, ~3 min to rebuild).

| Question | Where to look | qwen-code | kimi-cli |
|---|---|---|---|
| Binary name | upstream README | `qwen` | `kimi` |
| Install path | upstream install script (read it, don't trust docs) | `npm i -g @qwen-code/qwen-code` | `uv tool install --python 3.13 kimi-cli` |
| Wire protocol | grep upstream for `chat/completions`, `messages`, `responses` | OpenAI Chat Completions | OpenAI Chat Completions |
| Config surface | env vars vs config file | env (`OPENAI_*`) | TOML (`~/.kimi/config.toml`) |
| Non-interactive flag | the binary's `--help` | `-p 'prompt' --yolo --output-format json` | `--print -p 'prompt' --output-format stream-json` |

Two more knobs to confirm from source:

- **Max-rounds flag** — what counts as a "round", and the flag name? (qwen:
  `--max-session-turns N`, kimi: `--max-steps-per-turn N`, codex: implicit,
  hermes: `agent.max_turns` in YAML.) "Turn"/"step" rarely map 1:1 to API calls.
- **Compaction** — does the CLI auto-compact, at what threshold, configurable?
  (qwen: `model.chatCompression.contextPercentageThreshold`; kimi:
  `[loop_control].compaction_trigger_ratio`.) Most CLIs run the compaction call
  over the same `/v1/chat/completions` as normal steps, so the proxy can't tell
  them apart on the wire.

## 1. The adapter class — `cage/agents/<name>/agent.py`

Subclass `AgentType` (ABC in `cage/agents/base/definition.py`). Required members:

| Member | What it does | Common shape |
|---|---|---|
| `name: str` | Registry key — what users put in `project.yml`'s `kind:` | `"qwen_code"` |
| `state_paths: list[str]` | Dirs to snapshot pre/post trial (relative paths join `/home/agent`) | `[".qwen"]` |
| `default_image: str` | Tag of the prebuilt image | `"cage/qwen-code:pentestenv"` |
| `dockerfile: str` | Path under `docker/` used by `cage agent build` | `"docker/qwen_code/pentestenv.Dockerfile"` |
| `install_command(version)` | Fallback install line if the image lookup misses | `"npm i -g …@<v>"` |
| `build_launch_command(prompt, *, model, max_rounds, proxy_url)` | The per-trial shell command. Single-quote-escape the prompt; map positive `max_rounds` to the CLI's flag (negative = unset, `0` reserved). | `qwen --yolo --output-format json --max-session-turns N -p '<prompt>'` |
| `parse_output(result)` | Pull the final assistant string from `result.stdout`. Try the structured shape first, fall back to raw stdout. | see `qwen_code/agent.py` |
| `@property protocol` | `"openai"` or `"anthropic"` — drives the proxy's translation gate | `"openai"` |

Optional hooks (defaults exist):

| Hook | When to use |
|---|---|
| `env_vars(*, proxy_url, model, container, **kwargs)` | The only hook that runs **per-trial** with the live `proxy_url` **and** the `container` handle — use it to rewrite config files whose port changes each trial. Return only the env you need. |
| `setup_container(container, *, home_dir, model, **kwargs)` | Once-per-container: seed default config, write skill manifests, install plugins, skip first-run UX. Runs after CLI install, before the first trial. |
| `version_command()` | Probe to skip `install_command` when the binary is already baked. Avoid commands that initialise sandboxes. |
| `artifact_files()` | `(container_path, artifact_name)` pairs to pull post-trial. |

Register the class with `@register_agent_type` (keys it by `name`). Forget the
decorator and `cage agent list` won't show your agent.

### Where does the config live? (two patterns)

- **Env-var path** (qwen-code) — all knobs go through `env_vars()`, no config
  file. Simplest, but only viable if auth + provider selection can be fully
  expressed by env. Confirm against upstream source, not marketing.
- **Config-file path with per-trial rewrite** (kimi-cli, hermes) — the proxy port
  changes every trial, so any config file baking in the URL must be re-rendered
  each round:
  - `setup_container()` writes a placeholder so `cage agent debug` and
    `version_command` work before any trial;
  - `env_vars()` rewrites the same file with the live `proxy_url` (`_patch_config`).

Pick whichever the upstream CLI officially recommends for headless mode.

### Wire the compaction knob

`AgentInstance.context_compaction_threshold` (a project.yml field) is plumbed
into both `env_vars()` and `setup_container()`. **Read it.** If the CLI exposes an
auto-compaction trigger, map this 0..1 float to the CLI's native key, clamped to
its range:

```python
# qwen_code → ~/.qwen/settings.json
"model": {"chatCompression": {"contextPercentageThreshold": max(0.0, min(1.0, threshold))}}

# kimi_code → ~/.kimi/config.toml, [loop_control]
compaction_trigger_ratio = max(0.5, min(0.99, threshold))
```

Document the clamp range and any "can't fully disable" caveat in the adapter's
docstring (kimi, for example, also compacts unconditionally once
`context + reserved_context_size >= max_context_size`).

## 2. Register it — `cage/agents/__init__.py`

One import line in `register_builtin_agents()`, alphabetised:

```python
import cage.agents.qwen_code  # noqa: F401
```

The decorator's import side-effect populates the registry. Without this line
`cage run` raises `ValueError: Unknown agent type` even though your package is on
disk. Smoke-test it before building any image:

```bash
PYTHONPATH=. python -c "
from cage.agents import register_builtin_agents; register_builtin_agents()
from cage.agents.base import _AGENT_TYPE_REGISTRY; print(sorted(_AGENT_TYPE_REGISTRY))
"
```

Then validate `build_launch_command` / `env_vars` with no Docker:

```python
from cage.models import ModelConfig
from cage.agents import register_builtin_agents; register_builtin_agents()
from cage.agents.base import _AGENT_TYPE_REGISTRY

mc = ModelConfig(id="x", provider="vllm", model="X", base_url="...", api_key="K")
a = _AGENT_TYPE_REGISTRY["qwen_code"]()
print(a.build_launch_command("hello", model=mc, max_rounds=20, proxy_url="http://localhost:8877"))
print(a.env_vars(proxy_url="http://localhost:8877", model=mc))
```

## 3. Dockerfile — `docker/<name>/pentestenv.Dockerfile`

Base is always `pentestenv:latest` (Ubuntu 22.04 + pentest tools). Layer the CLI
on top. Two templates to copy:

- **Node-based** (claude_code, codex, qwen_code) — install Node 20+,
  `pip install httpx h2` to the system python, `npm install -g <package>`. See
  `docker/qwen_code/pentestenv.Dockerfile`.
- **Python-based** (hermes, kimi_code) — install `uv`, `pip install httpx h2`,
  `uv tool install <pkg>`. See `docker/kimi_code/pentestenv.Dockerfile`. **Set
  `UV_INDEX_URL` to the Tsinghua mirror** or the build hangs for tens of minutes
  from the team network.

Every Dockerfile **must**:

1. `pip install --no-cache-dir httpx h2` to `/usr/bin/python3` — the in-container
   proxy needs them.
2. `COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py` — baked for
   image-cache reuse, but re-copied before every trial, so edits take effect
   without a rebuild.
3. Create the `agent` user + `/home/agent/workspace`, set `HOME=/home/agent`
   (hardcoded in the orchestrator — don't change the convention).
4. Pre-install the CLI so the per-trial `install_command` short-circuits via
   `version_command()`. Pin the version with a `--build-arg`.

```bash
cage agent build --agent <name>             # default pentestenv variant
cage agent build --agent <name> --no-cache  # when debugging the install step
```

A good build leaves `cage/<name>:pentestenv`. Verify the binary and the flags you
bet on:

```bash
docker run --rm cage/<name>:pentestenv bash -c "<binary> --version && <binary> --help | grep -E '<flags>'"
```

## 4. Sanity-check before a real run

Skip these and you'll burn 15 minutes on a real `cage run` only to find the CLI
exits on a UX prompt or your env is wrong.

```bash
# A shell in a fresh container, no orchestration — confirms the image boots, the
# agent user works, and setup_container runs cleanly:
cage agent debug --agent <name> --model <id-from-config-models>
# inside, run your launch command against a tiny prompt to confirm it hits the proxy:
qwen --yolo --output-format json -p 'echo hello'

# Preflight benchmark config + prompt rendering, no targets/agents/model calls:
cage benchmark check <benchmark_id> --sample <sample_id>
```

Only after both pass, run a short smoke trial.

## 5. End-to-end smoke

Pick the smallest benchmark that exercises the proxy and a tool call:

- `examples/strongreject/default_strongreject.yml` — pure LLM-judge, no target,
  ~2 min.
- `examples/cvebench/default_cvebench.yml` — has a target spec but no live target
  unless `ctf.enabled: true`; checks tool-call wiring without docker-compose.

```bash
cage run examples/<bench>/default_<bench>.yml --max-sample-num 1
cage inspect examples/<bench>/.cage_runs/<agent_label>/run-<id>
```

Then inspect `trials/<sample>/proxy/proxy.jsonl`. For an OpenAI-protocol agent
each line should show `status: "success"`, a `system` first message with your
CLI's expected prefix, and `tool_calls` populated when the agent uses tools. If
`tool_calls` is `[]` while `content` looks like `<tool_call><function=…>`, your
model endpoint isn't running with `--enable-auto-tool-choice --tool-call-parser
hermes` — the proxy handles this, don't paper over it in the adapter.

## The `project.yml` a user writes

This is the whole user surface; they never touch your adapter.

```yaml
agents:
  - id: my_agent_baseline
    kind: <name>                 # registry key from your @register_agent_type
    home: /home/agent/workspace
    session_args: [--verbose]    # extra CLI flags, appended verbatim
    max_rounds: 30               # per-agent override; falls back to runtime.max_rounds
    context_compaction_threshold: 0.7   # per-agent (0..1); your adapter maps it to the CLI's knob
    shared_paths:                # make the agent stateful — these dirs persist across trials
      - /home/agent/.<config-dir>
```

## Files you touch (Path B)

```
cage/agents/myagent/                     # new — the AgentType package
cage/agents/__init__.py                  # +1 line in register_builtin_agents()
docker/myagent/pentestenv.Dockerfile     # new — image recipe
examples/<bench>/default_<bench>.yml     # optional — add your agent under agents:
```

---

# Reference — proxy gotchas & troubleshooting

*Skip on first read.* These are real gotchas the proxy already handles; they're
here so the next adapter doesn't relive them, and so you can recognise the symptom
if one regresses.

- **Streaming shape.** Most OpenAI SDKs default to `stream: true`. The proxy
  strips it (for clean 1:1 records) but re-wraps the non-streaming JSON back into
  a one-shot SSE stream (`_chat_completion_to_sse`) when the request had
  `stream: true`. Symptom if it regresses: *"Model stream ended without a finish
  reason"*.
- **`stream_options` 400.** With `stream: true` + `stream_options` stripped, vLLM
  400s on the leftover. The proxy drops `stream_options` too (`_strip_stream_flag`).
  Symptom: an early-fail trial with `status: "error"` + a 400 from upstream.
- **`proxy.rewrite.system` not applied.** The system prepend was historically only
  applied on the Anthropic→OpenAI path; direct OpenAI agents bypassed it. The proxy
  now applies it to the first `system` message of the OpenAI body. Check
  `original_system` vs `modified_system` per proxy.jsonl entry.
- **`upstream_http_proxy` not applied.** The transparent-forward path used to bypass
  the configured `http_proxy`. Fixed via CONNECT tunneling (`_forward_transparent`).
  Symptom: `[Errno -2] Name or service not known` on every entry.
- **Agent ignores proxy 429s.** At `max_requests` the proxy returns 429
  (`rate_limit_error`). Qwen-code retries forever; kimi-cli stops. **Always set the
  CLI's own max-rounds flag** to the same `max_rounds` so the agent self-terminates.
  The same path enforces `runtime.max_input_tokens` / `max_output_tokens` /
  `max_cost` (429 `budget_limit_error`) — a configured stop, not an outage.
- **Compact counter reads 0.** `compact_requests` only increments when the body has
  `_proxy_compact_rewritten`, which today only Claude Code's compact route sets.
  Qwen/Kimi compact over `/v1/chat/completions`, so the counter stays 0 even when
  compaction fires — the CLI's own log is the source of truth.
- **Score JSON shape.** Scorers write `{"<scorer>": {"value", "answer",
  "explanation"}}`; the inspector expects a flat `{"<scorer>": float}` and
  flattens on read (`cage/web/data/__init__.py`). Symptom if it regresses: a run
  page 500 with `TypeError: '>=' not supported between 'dict' and 'float'`.

---

# Never touch

Adding an agent should never make you edit these. If you want to, re-read
CLAUDE.md's layer test — the framework is missing an abstraction.

- `cage/experiment/engine/` — the trial lifecycle is agent-agnostic.
- `cage/agents/base/`, `cage/benchmarks/`, `cage/scoring/` — ABCs are stable.
- `cage/proxy/host.py`, `cage/proxy/sidecar.py` — protocol-level; a change here
  affects every agent and needs cross-validation.
- `cage/web/` — the inspector parses artifacts generically, no per-agent branches.
- any `examples/<bench>/benchmark.py` — benchmarks know nothing about agents.
