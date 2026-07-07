# Adding a New Agent to CAGE

> Reference implementations: `cage/agents/qwen_code.py`, `cage/agents/kimi_code.py`

There are **two ways** to add an agent. Pick by what you're integrating:

| | **Custom agent** (manifest) | **Built-in adapter** (Python) |
|---|---|---|
| For | *your own* agent — a LangGraph/LangChain graph, a script, any framework you launch with a command | wrapping a third-party coding-agent CLI (Claude Code, Codex, Qwen, …) as a first-class, reusable agent type |
| You write | a directory with your code + an `agent.yml` (no Python against the framework) | a `cage/agents/<name>/` `AgentType` subclass |
| Touches `cage/` | no | yes (the adapter + one register line) |
| Read | **the next section** | **sections 0–8 below** |

If at any step you find yourself touching `cage/sandbox/`, `cage/experiment/`,
or `examples/*/benchmark.py` to make the new agent work, stop — that means the
framework is missing an abstraction, not that the agent needs special-casing.

---

## Custom agent — a manifest, no framework code

Use this for **your own** agent: anything you can start with a shell command
and that reads a prompt + talks to a model + prints an answer. You write zero
framework code — one generic interpreter (`cage/agents/custom/`) drives any
such agent from a manifest.

### Shape

A custom agent is a **self-contained directory**: your code plus an `agent.yml`
that declares how Cage runs it.

```
references/agentic-poc/        # the agent's own directory
├── agent.yml                  # the manifest (below)
├── agentic_poc/               # your code
└── requirements.txt
```

### `agent.yml`

```yaml
name: agentic_poc                 # label / run-dir name (defaults to the dir name)
image: cage/custom-langgraph:base # runtime base (deps only); your code is copied in
command: >-                       # the ONE launch command; {tokens} filled by Cage
  python3 -m agentic_poc {workspace_dir}
  --model {model_name} --max-iterations {max_iterations} --instruction {task_instruction}
env:                              # extra env Cage injects (also templated)
  OPENAI_BASE_URL: "{base_url}"   # point your model client at the sidecar proxy
  OPENAI_API_KEY: "{api_key}"
  OPENAI_MODEL: "{model_name}"
workdir: .                        # optional: cwd for the command, relative to the copied source
output: stdout                    # optional: stdout (default) | {json_field: <key>}
state_paths: []                   # optional: dirs to snapshot for stateful runs
params:                           # optional: your own {placeholder} defaults
  max_iterations: 10              #   overridable per agent (yaml) or per run (--param)
```

Only `image` and `command` are required.

### The `{tokens}` Cage fills (the run-parameter surface)

| Token | Value |
|---|---|
| `{task_instruction}` | the benchmark's task instruction for the whole agent (shell-quoted automatically) |
| `{model_name}` | the selected model name (honours per-agent overrides) |
| `{base_url}` | the endpoint your client should call = the in-container sidecar proxy (so calls are intercepted). OpenAI-protocol models get the `/v1` suffix; Anthropic models don't. |
| `{api_key}` | the model's API key |
| `{max_rounds}` | the per-trial round budget (Cage's, not your agent's internal loop count) |
| `{workspace_dir}` | the task workspace path (`/home/agent/workspace`) |
| `{model.<field>}` | **any field of the model's `config/models.yml` entry**, incl. `{model.extra.<key>}` — this is how per-model knobs flow without Cage hardcoding them |

These names are **reserved** (Cage fills them); your own `params` (below) use any
other name. An unknown `{token}` fails at config-load (fast), not mid-trial.

### Your own `params` (everything else is yours)

Any `{placeholder}` that isn't a reserved token is a **user param** — your agent's
own knobs (loop counts, temperatures, feature flags), with no Cage code to add.
A param's value comes from the first of these that sets it, later winning:

1. `params:` in `agent.yml` — the author's default.
2. `agents[].params` in the experiment yaml — per-benchmark override.
3. `cage run … --param KEY=VALUE` — per-run override (repeatable).

```yaml
# experiment yaml — override the manifest default for this benchmark
agents:
  - id: agentic_poc
    source: ../../references/agentic-poc
    models: [{ id: nex-n2 }]
    params:
      max_iterations: 20
```

```bash
cage run examples/cybergym/default_cybergym.yml --agent agentic_poc --param max_iterations=30
```

A reserved Cage token always wins over a same-named param (you can't shadow
`{model_name}` etc.). A `{placeholder}` left with no value anywhere fails at
config-load. `--param` needs an unambiguous agent — pass `--agent <id>` when the
run has more than one.

### What Cage does per trial

1. Loads `<source>/agent.yml`.
2. Copies the whole source directory into the container at `/opt/cage-agent/src`
   (a `docker cp`, the same pattern used for the proxy sidecar — outside the
   workspace so the per-trial reset never wipes it; edit your code and re-run
   with no image rebuild; the process stays sealed in Docker, nothing is
   bind-mounted).
3. Fills the `{tokens}` in `command` and `env` from the resolved model + proxy + prompt.
4. Runs the command in `workdir` (as the `agent` user).
5. Reads the answer per `output` (default: stdout).

### Wire it into a benchmark

In any benchmark's `agents:` list, an entry with a `source:` **is** a custom
agent (no `kind:` needed). `source` resolves relative to the experiment file.

```yaml
agents:
  - id: agentic_poc                      # run-dir / label (optional; defaults to manifest name)
    source: ../../references/agentic-poc # dir with agent.yml + code
    models:
      - id: nex-n2                       # an id from config/models.yml
    max_concurrent: 3
```

### Build the runtime base (once)

The manifest's `image` carries only the framework deps (e.g. langgraph), never
your code. Build it once; rebuild only when *dependencies* change:

```bash
docker build -f docker/custom_langgraph.Dockerfile -t cage/custom-langgraph:base .
```

### Constraints (read these)

- **Your model client must honour `{base_url}`** (or the `OPENAI_BASE_URL` /
  `ANTHROPIC_BASE_URL` env Cage exports). Hardcode an endpoint and you bypass
  interception. LangChain's `ChatOpenAI()` reads `OPENAI_BASE_URL` by default,
  so usually you set the env and it just works.
- **No docker-in-docker.** Your agent already runs inside Cage's container; if
  it spawns its own Docker, you pay nested-container overhead. Do your work with
  the container's shell/files.
- **Copying solves *code* iteration, not *dependency* iteration** — a new pip
  dep means rebuilding the base image.
- **`output`** supports `stdout` (default) and `{json_field: <key>}` (parse
  stdout as JSON, take that key). Reading a result *file* is not supported
  (`parse_output` sees only stdout/stderr).

### Observability — your graph shows up in the trajectory (no setup)

If your agent is built on **LangChain / LangGraph**, the runtime base auto-attaches
a global callback (via `CAGE_TRACE`, which Cage sets for you) that stamps the
current **LangGraph node** on every model request. The in-container proxy records
it in `proxy.jsonl`, so the inspector's trajectory becomes node-aware for free —
no tracing code in your agent:

- a **node route** strip (e.g. `prepare → global_map → candidate_dev → …`) and a
  per-step **node badge** — the real graph, not a guessed structure;
- each node/agent's **system + user prompt**, surfaced at its first appearance;
- for agents whose tools run in plain Python (not LangChain tools), the tool
  **result** paired back under each action.

You write plain LangGraph. Two notes: a node that issues no model request
(deterministic Python, e.g. a `prepare`/`finalize` step) won't appear — the
trajectory is built from the requests the proxy sees; and a new pip dependency
means rebuilding the runtime base image.

### Worked example: `references/agentic-poc` on CyberGym

The manifest above is real. Wired into `examples/cybergym/default_cybergym.yml`,
`cage run` launches, per trial:

```
cd /opt/cage-agent/src && OPENAI_BASE_URL='http://127.0.0.1:<port>/v1' OPENAI_API_KEY='…' OPENAI_MODEL='nex-n2' \
  python3 -m agentic_poc /home/agent/workspace --model nex-n2 --max-iterations 10 --instruction '<cybergym prompt>'
# (--max-iterations comes from params.max_iterations=10; override with --param max_iterations=N)
```

The workspace already holds CyberGym's `README.md` / `submit.sh` / `repo-vul.tar.gz`;
the agent reads them, `submit.sh` posts its PoC, and CyberGym scores via the
grading service. The LLM calls go through the sidecar proxy.

### Files to touch

```
references/<agent>/agent.yml             # the manifest, beside your code
examples/<bench>/default_<bench>.yml     # +1 agents: entry with `source:`
docker/custom_langgraph.Dockerfile       # a runtime base (reuse across custom agents)
```

Nothing in `cage/` changes — the generic `cage/agents/custom/` interpreter
already handles any manifest.

---

## Built-in adapter (Python `AgentType`)

The rest of this guide is the heavier path: wrapping a third-party coding-agent
CLI (Claude Code, Codex, Hermes, Qwen Code, Kimi CLI, …) as a registered
`AgentType` so `cage run` can drive it. The whole integration is **Layer 1
framework + Layer 2 image** — no benchmark code should ever change.

---

## 0. Decide upfront

Five things to nail down before touching code. Getting these wrong wastes a
build cycle (each agent image is ~22 GB and ~3 min to rebuild).

| Question | Where to look | Example: qwen-code | Example: kimi-cli |
|---|---|---|---|
| Binary name | upstream README | `qwen` | `kimi` |
| Install path | upstream install script (fetch and read it, don't trust docs) | `npm i -g @qwen-code/qwen-code` | `uv tool install --python 3.13 kimi-cli` |
| Wire protocol | grep upstream for `chat/completions`, `messages`, `responses` | OpenAI Chat Completions | OpenAI Chat Completions (`openai_legacy` provider) |
| Config surface | env vars vs config file | env (`OPENAI_API_KEY/BASE_URL/MODEL`) | TOML (`~/.kimi/config.toml`) |
| Non-interactive flag | `--help` for the binary | `-p 'prompt' --yolo --output-format json` | `--print -p 'prompt' --output-format stream-json` |

Two more knobs worth confirming from source:

- **Max-rounds flag** — what counts as a "round"? CLI flag name? (qwen:
  `--max-session-turns N`, kimi: `--max-steps-per-turn N`, codex: implicit,
  hermes: `agent.max_turns` in YAML.) Note that "session turn" and "step"
  rarely map 1:1 to LLM API calls.
- **Compaction** — does the CLI auto-compact? At what threshold? Configurable?
  (qwen: `model.chatCompression.contextPercentageThreshold`; kimi:
  `[loop_control].compaction_trigger_ratio` + unconditional `reserved_context_size`
  trigger.) Most CLIs reuse the same `/v1/chat/completions` for the compaction
  call, so the proxy can't distinguish it from a normal agent step on the wire
  — keep that in mind when reading `successful_requests` later.

---

## 1. Add the AgentType adapter — `cage/agents/<name>/`

ABC contract in `cage/agents/base/definition.py`. Required:

| Method/field | What it does | Common shape |
|---|---|---|
| `name: str` | Registry key — what users type in `project.yml`'s `kind:` field | `"qwen_code"` |
| `state_paths: list[str]` | Dirs to snapshot pre/post trial. Relative paths join with `/home/agent`. | `[".qwen"]` |
| `default_image: str` | Tag of the prebuilt image | `"cage/qwen-code:pentestenv"` |
| `dockerfile: str` | Path under `docker/` used by `cage agent build` | `"docker/qwen_code_pentestenv.Dockerfile"` |
| `install_command(version)` | Fallback install line if image lookup misses or `version_command()` reports "unknown" | `"npm i -g @qwen-code/qwen-code@<v>"` |
| `build_launch_command(prompt, *, model, max_rounds, proxy_url)` | The shell command launched per trial. Single-quote-escape the prompt; map positive `max_rounds` to the CLI's own flag; negative values mean unset, and `0` is reserved for no model-call rounds. | `qwen --yolo --output-format json --max-session-turns N -p '<prompt>'` |
| `parse_output(result)` | Pull a final assistant string out of `ExecResult.stdout`. Try the structured shape (`--output-format json` / `stream-json`) first, then fall back to raw stdout. | see qwen_code.py:65–124 |
| `env_vars(*, proxy_url, model, **kwargs)` | Inject env into the agent process. **This is also the only AgentType hook that runs per-trial AND receives the live `proxy_url` AND the `container` handle** — use it to rewrite config files whose port references change per trial. Return only the env you need; the orchestrator merges with `HOME=/home/agent` and `agent.extra_env`. | `{"OPENAI_BASE_URL": proxy_url + "/v1", "OPENAI_API_KEY": model.api_key, "OPENAI_MODEL": model.model}` |
| `@property protocol` | `"openai"` or `"anthropic"`. Drives the proxy's translation gate. | `"openai"` |
| `setup_container(container, *, home_dir, model, context_compaction_threshold, **kwargs)` *(optional)* | Once-per-container hook: seed default config files, write skill manifests, install plugins. Runs after CLI install, before the first trial. | Write `~/.qwen/settings.json` to skip first-run UX; pin `selectedType: openai`; suppress telemetry. |
| `version_command()` *(optional)* | Probe used to skip `install_command` when the binary is already in the image. Avoid commands that initialise sandboxes (codex had to work around this). | `"qwen --version 2>/dev/null \|\| echo unknown"` |

The decorator `@register_agent_type` keys the class by `name` into the registry
the orchestrator pulls from. Forget the decorator and `cage agent list` won't list
your agent.

### Two patterns for "where does the config live"

- **Env-var path** (qwen-code). All knobs go through `env_vars()`, no config
  file needed. Simplest, but only viable if the CLI's authentication +
  provider selection can be fully described by environment variables. Confirm
  this against the upstream source — don't trust marketing pages.
- **Config-file path with per-trial rewrite** (kimi-cli, hermes). The proxy
  port changes every trial, so any config file that bakes in the URL has to
  be re-rendered each round. The hermes pattern:
  - `setup_container()` writes a placeholder config so `cage agent debug` and
    `version_command` work before any trial starts.
  - `env_vars()` rewrites the same file with the live `proxy_url`
    (`_patch_config` in both kimi_code.py and hermes.py).

Choose whichever the upstream CLI officially recommends for headless mode.
For Qwen Code that's env vars (see `packages/cli/src/config/config.ts`);
for Kimi CLI that's TOML (their env-var path only configures one provider
and conflates provider with model selection).

### Wire the compaction knob

`AgentInstance.context_compaction_threshold` (project.yml field of the same
name) is already plumbed by the orchestrator into both `env_vars()` and
`setup_container()`. **Read it.** If the upstream CLI exposes an
auto-compaction trigger, map this 0..1 float to whatever native key it uses,
and clamp to the CLI's supported range:

```python
# qwen_code: writes ~/.qwen/settings.json
"model": {"chatCompression": {"contextPercentageThreshold":
          max(0.0, min(1.0, threshold))}}

# kimi_code: writes ~/.kimi/config.toml under [loop_control]
compaction_trigger_ratio = max(0.5, min(0.99, threshold))
```

Document the clamp range and any "cannot fully disable" caveat in the
adapter's docstring. Kimi for example also fires compaction unconditionally
when `context + reserved_context_size >= max_context_size`, so the trigger
ratio alone never disables it — that's why kimi_code.py exposes
`max_context_size` / `reserved_context_size` via `model.extra` for test rigs.

---

## 2. Register it in `cage/agents/__init__.py`

One import line in `register_builtin_agents()`, alphabetised:

```python
import cage.agents.kimi_code  # noqa: F401
import cage.agents.qwen_code  # noqa: F401
```

The decorator side-effect populates the registry on import. Without this
line `cage run` raises `ValueError: Unknown agent type` even though your
package is on disk.

Smoke-test the registration before building any image:

```bash
PYTHONPATH=. python -c "
from cage.agents import register_builtin_agents
register_builtin_agents()
from cage.agents.base import _AGENT_TYPE_REGISTRY
print(sorted(_AGENT_TYPE_REGISTRY.keys()))
"
```

Then validate the build_launch_command and env_vars shape without ever
starting Docker:

```python
from cage.models import ModelConfig
from cage.agents import register_builtin_agents
register_builtin_agents()
from cage.agents.base import _AGENT_TYPE_REGISTRY

mc = ModelConfig(id="x", provider="vllm", model="X", base_url="...", api_key="K")
a = _AGENT_TYPE_REGISTRY["qwen_code"]()
print(a.build_launch_command("hello", model=mc, max_rounds=20, proxy_url="http://localhost:8877"))
print(a.env_vars(proxy_url="http://localhost:8877", model=mc))
```

---

## 3. Dockerfile — `docker/<name>_pentestenv.Dockerfile`

The base is always `pentestenv:latest` (Ubuntu 22.04 + the pentest tool layer).
Layer the agent CLI on top. Two template Dockerfiles to copy from:

- **Node-based agents** (claude_code, codex, qwen_code) — install Node 20+,
  `pip install httpx h2` to the system python (the in-container proxy needs
  it), `npm install -g <package>`. See `docker/qwen_code_pentestenv.Dockerfile`.
- **Python-based agents** (hermes, kimi_code) — install/upgrade `uv`,
  `pip install httpx h2`, then `uv tool install <pkg>`. See
  `docker/kimi_code_pentestenv.Dockerfile`. **Set `UV_INDEX_URL` to the Tsinghua
  mirror** or the build will hang for tens of minutes against pypi.org from
  the team network.

Every Dockerfile **must**:

1. `RUN /usr/bin/python3 -m pip install --no-cache-dir httpx h2` — the proxy
   that `cage/proxy/host.py::start_container_proxy` copies in at runtime
   uses these.
2. `COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py`
   — present at image build for image-cache reuse, but the orchestrator
   `rm -f`s and re-copies it before every trial, so edits to
   `container_proxy.py` take effect without rebuilding the image.
3. Create the `agent` user, `/home/agent/workspace`, set `HOME=/home/agent`.
   Don't change this convention — it's hardcoded in `_setup_container` and
   `AgentInstance.home` defaults.
4. Pre-install the agent CLI so the per-trial `install_command` short-circuits
   on `version_command()`. Pin the version via a `--build-arg` so reruns are
   deterministic.

Build with:

```bash
cage agent build --agent <name>          # default pentestenv variant
cage agent build --agent <name> --no-cache  # if you're debugging the install step
```

A successful build leaves an image at `cage/<name>:pentestenv`. Verify the binary
runs and `--help` reports the flags you bet on in `build_launch_command`:

```bash
docker run --rm cage/<name>:pentestenv bash -c "<binary> --version && <binary> --help | grep -E '<flags-you-need>'"
```

---

## 4. Sanity-check via `cage agent debug` and `cage benchmark check`

Skip these and you'll burn 15 minutes on a real `cage run` only to discover
the CLI exits with a UX prompt or your env vars are wrong.

```bash
# Drop into a shell in a fresh container, no orchestration. Confirms image
# boots, agent user works, and your setup_container hook runs cleanly.
cage agent debug --agent <name> --model <id-from-config-models>

# Inside, manually run your build_launch_command output against a tiny
# prompt to confirm the agent talks to the proxy:
qwen --yolo --output-format json -p 'echo hello'
```

```bash
# Preflight benchmark config and prompt rendering without targets, agents, or
# model calls. The full rendered prompt is written under .cage_checks/.
cage benchmark check <benchmark_id> --sample <sample_id>
```

Only after both pass should you `cage run` a short smoke trial to exercise the
model endpoint and the agent launch command together.

---

## 5. End-to-end smoke

Pick the smallest benchmark that exercises the proxy and a tool call. Two good
choices:

- `examples/strongreject/default_strongreject.yml` — pure LLM-judge, no container target,
  10 prompts max. Fastest smoke (~2 min).
- `examples/cvebench/default_cvebench.yml` — sample with a target spec but
  no live target unless `ctf.enabled: true`. Lets you check tool-call wiring
  without provisioning docker-compose stacks.

```bash
# Add your agent to one of these default_*.yml files under `agents:`,
# then:
cage run examples/<bench>/default_<bench>.yml --max-sample-num 1
cage inspect examples/<bench>/.cage_runs/<agent_label>/run-<id>
```

Inspect `trials/<sample>/proxy/proxy.jsonl` after the run. For an OpenAI-protocol
agent each line should have:

- `status: "success"`
- `openai_request.messages[0].role == "system"` with your CLI's expected system
  prompt prefix
- `upstream_response.choices[0].message.tool_calls` populated when the agent
  uses tools (if empty, the model emitted XML in `content` or
  `reasoning_content` and the proxy's `_hoist_xml_tool_calls_inplace` should
  have lifted it — verify by checking `finish_reason == "tool_calls"`).

If `tool_calls` is `[]` and `content`/`reasoning` looks like
`<tool_call><function=...>`, your model endpoint isn't running with
`--enable-auto-tool-choice --tool-call-parser hermes`. The proxy already
handles this case; don't paper over it in the adapter.

---

## 6. Pitfalls that have already bitten us

Documented here so the next adapter doesn't relive them.

### Streaming response shape

Most OpenAI SDKs send `stream: true` by default. The proxy strips that flag
(`force_non_streaming = True` for clean 1:1 proxy.jsonl records), but the
client's SDK still parses the response as SSE. The proxy now re-wraps the
non-streaming JSON back into a one-shot SSE stream
(`_chat_completion_to_sse`) when the original request had `stream: true`.
You don't need to do anything in the adapter — but if your CLI errors with
"Model stream ended without a finish reason", that wrap regressed.

### `stream_options` rejection from vLLM

When a client sets both `stream: true` and `stream_options: {...}` and the
proxy strips the stream flag, vLLM 400s on the leftover `stream_options`.
The proxy now drops `stream_options` alongside the flag
(`_strip_stream_flag`). Verify: an early-failure trial with proxy.jsonl
entries `status: "error"` and a 400 from upstream is the signature.

### `proxy.rewrite.system` not reaching the agent

The Jinja-style system prepend (`proxy.rewrite.system` in project.yml) was
historically only applied in the Anthropic→OpenAI translation path. OpenAI
agents hitting `/v1/chat/completions` directly used to bypass it silently
(qwen3 then refused pentest tasks). The proxy now applies the template to
the first `role: system` message of the OpenAI request body
(`_apply_system_template_to_openai_body`). If your agent's behaviour
suggests the system prepend isn't taking effect, check
`proxy.jsonl[].original_system` vs `modified_system` per entry.

### `upstream_http_proxy` not reaching the agent

Same shape as above: the transparent forward path used to bypass the
configured `http_proxy`, so OpenAI agents couldn't reach model endpoints
behind GFW egress. Fixed via CONNECT tunneling in `_forward_transparent`.
Symptom was `[Errno -2] Name or service not known` on every proxy.jsonl
entry; if you see that again the patch regressed.

### Agent doesn't honour proxy 429s

When the proxy hits `max_requests` (= the first non-negative value from
`agent.max_rounds`, `runtime.max_rounds`, then the benchmark sample default),
it returns a 429 with `type: "rate_limit_error"`.
Qwen-code retries indefinitely on this, blowing past the configured round
budget. Kimi-cli stops cleanly. **Always set the CLI's own max-rounds flag**
(`--max-session-turns N` / `--max-steps-per-turn N`) to the same `max_rounds`
value so the agent self-terminates and doesn't depend on proxy 429 graceful
handling.

The same proxy path enforces `runtime.max_input_tokens`,
`runtime.max_output_tokens`, and `runtime.max_cost`. When one of those is
reached, the next model call returns 429 with `type: "budget_limit_error"`.
Treat it as a configured stop condition, not an endpoint outage.

### Score JSON shape

Trial scorers write `{"<scorer>": {"value": float, "answer": str,
"explanation": str}}` to `scores/<scorer>.json`, but the web inspector
template expects a flat `{"<scorer>": float}` view. `cage/web/data/__init__.py` now
flattens on read and stashes the metadata on a parallel `score_details`
key. You shouldn't have to think about this — but if a new run page 500s
with `TypeError: '>=' not supported between instances of 'dict' and 'float'`,
that flatten regressed.

### CAGE's compact counter doesn't see your agent

`proxy.jsonl` and `progress.json` carry a `compact_requests` counter, but it
only increments when the request body has `_proxy_compact_rewritten` —
which currently only Claude Code's `/v1/messages/.../compact` route triggers.
Qwen Code and Kimi CLI run their compaction calls over the same
`/v1/chat/completions` endpoint as agent steps, so for these agents
`compact_requests == 0` even when compaction fires. The agent's internal
log (`~/.qwen/logs/...`, `~/.kimi/logs/kimi.log`) is the only source of truth
for compaction events on the host side. If you need the counter to work,
add a body-sniffing check for the CLI's canonical compaction system prompt
in the proxy (out of scope for the agent adapter itself).

---

## 7. project.yml example for end users

This is what a user writes; they should never need to touch your adapter.

```yaml
agents:
  - id: my_agent_baseline
    kind: <name>                    # registry key from your @register_agent_type
    home: /home/agent/workspace
    # session_args: extra CLI flags appended verbatim to build_launch_command
    session_args:
      - --verbose
    # max_rounds: per-agent override; falls back to runtime.max_rounds
    max_rounds: 30
    # context_compaction_threshold: per-agent override (0..1); your adapter
    # is responsible for translating to whatever the CLI's native knob is.
    context_compaction_threshold: 0.7
    # shared_paths: turn the agent stateful — listed dirs persist across
    # trials in the same container. Defaults to agent_type.state_paths.
    shared_paths:
      - /home/agent/.<config-dir>
```

---

## 8. Files to touch — final checklist

For an agent called `myagent`:

```
cage/agents/myagent/                         # new — the AgentType subclass package
cage/agents/__init__.py                      # +1 line in register_builtin_agents(): import cage.agents.myagent
docker/myagent_pentestenv.Dockerfile             # new — image recipe
examples/<bench>/default_<bench>.yml         # optional — add your agent under agents:
```

What you should **never** touch when adding an agent:

- `cage/experiment/engine/` — lifecycle is agent-agnostic.
- `cage/agents/base/`, `cage/benchmarks/`, `cage/scoring/` — ABCs are stable.
- `cage/proxy/host.py`, `cage/proxy/sidecar.py` — protocol-level;
  changes here affect every agent and need cross-validation.
- `cage/web/` — the inspector parses run artifacts (`proxy.jsonl`, `meta.json`,
  scores) generically, with no per-agent branches, so a new agent needs no
  inspector changes.
- Any `examples/<bench>/benchmark.py` — benchmarks know nothing about agents
  and shouldn't.

If you find yourself wanting to, push back and re-read CLAUDE.md's layer test.
