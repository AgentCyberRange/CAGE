# Putting a LangGraph / LangChain Agent in CAGE

A hands-on guide for dropping **your own** agent — a LangGraph graph, a LangChain
`AgentExecutor`, or any Python program that talks to a model — into Cage so it
can be evaluated on Cage's benchmarks.

> **TL;DR** — You write **zero framework code**. You give Cage (1) your agent's
> code, (2) one launch command in an `agent.yml` manifest, and (3) the env that
> points your LLM client at Cage's proxy. Cage copies your code into a sealed
> container, runs the command per trial, intercepts every model call, snapshots
> state, and scores the result. LangGraph node structure shows up in the
> trajectory viewer automatically.
>
> The fastest way to start is to copy [`references/agentic-poc/`](https://github.com/AgentCyberRange/CAGE/tree/main/references/agentic-poc)
> — a real LangGraph agent already wired this way — and edit it.
>
> `references/agentic-poc/` is a standalone reference; the **in-repo canonical
> home** for a custom agent is `cage/agents/custom/<name>/` (an `agent.yml` with a
> `source:` key), as used by the shipped `qitos` / `cairn` agents. Either layout
> works — the manifest is the same.

There are two ways to add an agent to Cage. This guide is the **custom-agent
(manifest)** path, which is the right one for a LangGraph/LangChain agent you
own. The other path — wrapping a third-party coding-agent *CLI* (Claude Code,
Codex, …) as a registered `AgentType` — is heavier and lives in
[`docs/agent-cage-managed.md`](agent-cage-managed.md). If you find yourself
editing anything under `cage/` to make your LangGraph agent run, stop: the
manifest path needs none of it.

---

## The mental model

```
        your repo                         Cage container (one per trial)
  ┌────────────────────┐   docker cp   ┌──────────────────────────────────┐
  │ my-agent/          │ ────────────▶ │ /opt/cage-agent/src/  (your code)│
  │   agent.yml        │               │ /home/agent/workspace (the task) │
  │   my_agent/...     │               │                                  │
  └────────────────────┘               │   python3 -m my_agent ...         │
                                        │        │  OpenAI/Anthropic call   │
                                        │        ▼                          │
                                        │   sidecar proxy ── records ──▶ proxy.jsonl
                                        │        │                          │
                                        └────────┼──────────────────────────┘
                                                 ▼ upstream model endpoint
```

Three facts follow from this picture, and they are the whole contract:

1. **Your code runs inside the container, not on the host.** It is `docker cp`'d
   to `/opt/cage-agent/src` at trial start (outside the workspace, so the
   per-trial reset never wipes it). Edit your code and re-run with **no image
   rebuild**.
2. **Every model call must go through the in-container proxy** so Cage can record
   and snapshot it. You make that happen by pointing your LLM client at the URL
   Cage hands you (`{base_url}`), nothing more.
3. **The task lives in `/home/agent/workspace`.** The benchmark seeds it (e.g.
   CyberGym drops `README.md` / `submit.sh` / the vulnerable repo there). Your
   agent reads/writes files there like a normal working directory.

---

## What you provide (exactly four things)

| # | Thing | Where |
|---|---|---|
| 1 | Your agent code | any directory in your repo (or a git submodule under `references/`) |
| 2 | `agent.yml` — the launch manifest | beside your code |
| 3 | Your LLM client reads `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | already true for LangChain's `ChatOpenAI()` |
| 4 | A runtime **base image** (deps only) | reuse the shipped `docker/custom_langgraph/Dockerfile` |

You do **not** write a Python class, touch `cage/`, or edit any benchmark.

---

## Quickstart — copy the reference agent

`references/agentic-poc/` is a working LangGraph agent. Use it as the template.

```
references/agentic-poc/
├── agent.yml              # the manifest (see below)
├── requirements.txt
└── agentic_poc/
    ├── __main__.py        # `python3 -m agentic_poc <workspace> --model ... --instruction ...`
    ├── graph.py           # the LangGraph StateGraph
    ├── nodes.py           # nodes (each makes model calls)
    ├── llm.py             # ChatOpenAI built from OPENAI_* env  ← the proxy hook
    └── ...
```

```bash
# 1. Build the runtime base once (deps only; your code is NOT baked in).
docker build -f docker/custom_langgraph/Dockerfile -t cage/custom-langgraph:base .

# 2. Run it on a benchmark (agentic_poc is already wired into cybergym's example).
cage run examples/cybergym/default_cybergym.yml --agent agentic_poc --max-sample-num 1

# 3. Look at the trajectory — node-aware, no tracing code in the agent.
cage inspect examples/cybergym/.cage_runs/<agent_label>/run-<id>
```

To make your **own** agent, copy that directory, drop in your graph, and edit
`agent.yml`'s `command` to launch it.

---

## The manifest — `agent.yml`

The manifest is the entire contract between your agent and Cage. Only `image`
and `command` are required.

```yaml
name: my_agent                      # label / run-dir name (default: the dir name)

image: cage/custom-langgraph:base   # runtime base (deps only). Your code is copied in.

command: >-                         # the ONE launch command; {tokens} are filled by Cage
  python3 -m my_agent {workspace_dir}
  --model {model_name}
  --max-iterations {max_iterations}
  --instruction {task_instruction}

env:                                # extra env Cage injects (also templated)
  OPENAI_BASE_URL: "{base_url}"     # ← point your model client at the sidecar proxy
  OPENAI_API_KEY: "{api_key}"
  OPENAI_MODEL: "{model_name}"

workdir: .                          # optional: cwd for the command, relative to the copied src
output: stdout                      # optional: stdout (default) | {json_field: <key>}
state_paths: []                     # optional: dirs to snapshot for stateful runs
params:                             # optional: YOUR own {placeholder} defaults
  max_iterations: 10
```

### Reserved tokens — what Cage fills

Use these `{tokens}` anywhere in `command` or `env`. Cage substitutes them per
trial/model. They are **reserved** — you can't shadow them with a `param`.

| Token | Value |
|---|---|
| `{task_instruction}` | the benchmark's task prompt (already **shell-quoted** — safe to drop straight into argv) |
| `{model_name}` | the selected model's name (honours per-agent model overrides) |
| `{base_url}` | the URL your client must call = the in-container sidecar proxy. OpenAI-protocol models get a `/v1` suffix; Anthropic models don't. **This is how calls get intercepted.** |
| `{api_key}` | the model's API key |
| `{max_rounds}` | Cage's per-trial round budget (its request cap — not your graph's internal loop count) |
| `{workspace_dir}` | the task workspace path, `/home/agent/workspace` |
| `{model.<field>}` | any field of the model's `config/models.yml` entry — e.g. `{model.base_url}`, `{model.provider}`, and `{model.extra.<key>}` for arbitrary per-model knobs |

An unknown `{token}` fails at **config-load** (fast), never mid-trial.

### Your own params — everything else is yours

Any `{placeholder}` that is **not** a reserved token is your agent's own knob
(loop counts, temperatures, feature flags) — no Cage code to add. A param's
value comes from the first that sets it, later winning:

1. `params:` in `agent.yml` — the author's default.
2. `agents[].params` in the experiment YAML — per-benchmark override.
3. `cage run … --param KEY=VALUE` — per-run override (repeatable; needs
   `--agent <id>` when the run has more than one agent).

```bash
cage run examples/cybergym/default_cybergym.yml --agent my_agent --param max_iterations=30
```

> **`{max_rounds}` vs your own `{max_iterations}`.** They are different
> granularities. `{max_rounds}` is the proxy's per-trial *request* cap; your
> graph may make several model calls per "iteration". Set your own loop bound
> with your own param and decide independently whether to also pass
> `{max_rounds}` to your client. (agentic_poc keeps `max_rounds` unlimited so
> the proxy never cuts it off mid-round, and self-terminates via `submit.sh`.)

---

## Pointing your LLM client at the proxy (the one hard rule)

Cage can only record/snapshot/score what flows through the in-container proxy.
So **your model client must call `{base_url}`** instead of a hardcoded endpoint.

For LangChain this is essentially free, because `ChatOpenAI()` reads
`OPENAI_BASE_URL` / `OPENAI_API_KEY` from the environment by default. The
reference agent's `llm.py`:

```python
from langchain_openai import ChatOpenAI

def build_llm(model=None, temperature=0.0):
    return ChatOpenAI(
        model=model or os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],   # ← the proxy, injected by the manifest's env
        temperature=temperature,
    )
```

…paired with the manifest's `env:` block above. That's the entire integration.

- **OpenAI-protocol model** → set `OPENAI_BASE_URL: "{base_url}"` (it already
  includes `/v1`). `langchain-openai` talks to it directly.
- **Anthropic-protocol model** → set `ANTHROPIC_BASE_URL: "{base_url}"` and use
  `langchain-anthropic` / the Anthropic SDK. Cage's `{base_url}` omits the `/v1`
  suffix for Anthropic automatically.

**Hardcode an endpoint and you bypass interception** — the run produces an empty
`proxy.jsonl` and nothing is scored. If a run shows zero recorded requests, this
is the first thing to check.

---

## The runtime base image

The manifest's `image` is a **deps-only** base. Your agent code is **not** baked
into it — Cage copies it in at runtime — so you rebuild the image only when a
*dependency* changes, not when you edit your agent.

Reuse the shipped recipe, `docker/custom_langgraph/Dockerfile` →
`cage/custom-langgraph:base`. It already does the four things any custom-agent
base must:

```dockerfile
# 1. The agent deps, installed into the *agent user's* python (/usr/bin/python3,
#    NOT root's), or `python3 -m my_agent` can't import langgraph at run time.
RUN /usr/bin/python3 -m pip install --no-cache-dir \
      langgraph 'langchain-core<2' langchain-openai

# 2. The sidecar proxy's lean deps (httpx + h2) on every interpreter.
RUN for py in python3 /usr/bin/python3; do "$py" -m pip install --no-cache-dir httpx h2; done

# 3. The Cage trace runtime on PYTHONPATH → free LangGraph node tracing (next section).
COPY cage/agents/custom/trace_runtime/ /opt/cage-trace/
ENV PYTHONPATH=/opt/cage-trace

# 4. The sidecar proxy + an unprivileged `agent` user with /home/agent/workspace.
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py
```

To add your own dependency, append a `pip install` for **`/usr/bin/python3`** and
rebuild:

```bash
docker build -f docker/custom_langgraph/Dockerfile -t cage/custom-langgraph:base .
```

> **Why `/usr/bin/python3`?** The agent runs as the unprivileged `agent` user,
> whose `python3` is the system interpreter — not root's. Installing your stack
> anywhere else means `python3 -m my_agent` fails with `ModuleNotFoundError` at
> trial time.

---

## Free observability — your graph shows up in the trajectory

If your agent is built on LangChain/LangGraph, you get a **node-aware
trajectory for free** — no tracing code in your agent. Here's the whole
mechanism, so you can trust it (and debug it):

1. Cage sets `CAGE_TRACE=1` in the agent's environment.
2. The base image puts `trace_runtime/` on `PYTHONPATH`; its `sitecustomize.py`
   auto-imports `cage_trace` at interpreter startup (Python loads any
   `sitecustomize` on the path).
3. When `CAGE_TRACE` is set, `cage_trace` registers a **global LangChain
   callback** via the same `register_configure_hook` API LangSmith uses. On
   every model call it reads the current LangGraph node
   (`metadata['langgraph_node']`) and stashes it.
4. It wraps `httpx.send` so each outgoing model request carries `X-Cage-Node` /
   `X-Cage-Run-Id` / `X-Cage-Parent-Id` headers.
5. The sidecar proxy records those headers per request in `proxy.jsonl` (and
   strips them before forwarding upstream).

So the inspector's trajectory becomes structure-aware with no separate trace
file and no agent changes:

- a **node-route strip** (e.g. `prepare → global_map → candidate_dev → …`) and a
  per-step **node badge** — the real graph, not a guess;
- each node's **system + user prompt**, surfaced at its first appearance.

**The one caveat:** the trajectory is built from the requests the proxy sees, so
a node that issues **no model call** (a deterministic Python `prepare`/`finalize`
step) won't appear. That's expected, not a bug.

Everything in the trace runtime is best-effort: no LangChain, no `httpx`, or an
unset `CAGE_TRACE` is a silent no-op (the lean sidecar process is unaffected).

---

## Wire it into a benchmark and run

In any benchmark's experiment YAML, an `agents:` entry with a **`source:`** *is*
a custom agent — no `kind:` needed. `source` resolves relative to the YAML file.

```yaml
# examples/cybergym/default_cybergym.yml (excerpt)
agents:
  - id: my_agent                         # run-dir / label (optional; defaults to manifest name)
    source: ../../references/my-agent    # dir holding agent.yml + your code
    models:
      - id: nex-n2                        # an id from config/models.yml
    max_concurrent: 3
    params:                              # optional per-benchmark param overrides
      max_iterations: 20
```

Then:

```bash
cage run examples/cybergym/default_cybergym.yml --agent my_agent --max-sample-num 1
```

Per trial, that expands to roughly:

```bash
cd /opt/cage-agent/src \
 && OPENAI_BASE_URL='http://127.0.0.1:<port>/v1' OPENAI_API_KEY='…' OPENAI_MODEL='nex-n2' CAGE_TRACE=1 \
    python3 -m my_agent /home/agent/workspace --model nex-n2 --max-iterations 20 --instruction '<task prompt>'
```

---

## What Cage does each trial

1. Loads `<source>/agent.yml`.
2. `docker cp`s the whole source dir to `/opt/cage-agent/src` (outside the
   workspace, sealed in Docker — nothing bind-mounted).
3. Fills the `{tokens}` in `command` and `env` from the resolved model + proxy +
   prompt, and sets `CAGE_TRACE=1`.
4. Runs the command in `workdir` as the unprivileged `agent` user.
5. Reads the agent's answer per `output` (default: stdout), records `proxy.jsonl`,
   snapshots any `state_paths`, and scores the trial.

---

## Constraints & gotchas

- **Honour `{base_url}`.** Hardcoding an endpoint bypasses interception →
  empty `proxy.jsonl`, nothing scored. LangChain `ChatOpenAI()` reads
  `OPENAI_BASE_URL` by default, so usually just setting the env is enough.
- **Install deps into `/usr/bin/python3`**, not root's python — see above.
- **Copying solves *code* iteration, not *dependency* iteration.** Edit your
  agent → just re-run. Add a pip dep → rebuild the base image.
- **No docker-in-docker.** Your agent already runs inside Cage's container; do
  your work with the container's own shell/files, don't spawn nested Docker.
- **`output` supports `stdout` (default) and `{json_field: <key>}`** (parse
  stdout as JSON, take that key). Reading a result *file* is not supported —
  `parse_output` sees only stdout/stderr. If the benchmark scores via a side
  channel (e.g. CyberGym's `submit.sh`), `output: stdout` is fine and stdout is
  just recorded.
- **Stateful runs:** list dirs in `state_paths` (or set `shared_paths` on the
  agent in the experiment YAML) to persist them across trials. They must live
  **outside** `/home/agent/workspace`, which is reset every trial.

---

## Checklist

```
references/my-agent/agent.yml          # the manifest, beside your code
references/my-agent/my_agent/...        # your LangGraph code; llm client reads OPENAI_BASE_URL
docker/custom_langgraph/Dockerfile      # reuse as-is, or add your pip deps to /usr/bin/python3
examples/<bench>/default_<bench>.yml    # +1 agents: entry with `source:`
```

Nothing in `cage/` changes — the generic `cage/agents/custom/` interpreter
already drives any manifest.

---

## Where to look

- [`references/agentic-poc/`](https://github.com/AgentCyberRange/CAGE/tree/main/references/agentic-poc) — the real LangGraph
  example this guide is built on.
- [`cage/agents/custom/manifest.py`](https://github.com/AgentCyberRange/CAGE/blob/main/cage/agents/custom/manifest.py) — the
  authoritative `agent.yml` schema.
- [`cage/agents/custom/agent.py`](https://github.com/AgentCyberRange/CAGE/blob/main/cage/agents/custom/agent.py) — the generic
  interpreter (token filling, `docker cp`, output parsing).
- [`cage/agents/custom/trace_runtime/`](https://github.com/AgentCyberRange/CAGE/tree/main/cage/agents/custom/trace_runtime) —
  the zero-code LangGraph trace hook.
- [`docker/custom_langgraph/Dockerfile`](https://github.com/AgentCyberRange/CAGE/blob/main/docker/custom_langgraph/Dockerfile)
  — the runtime base recipe.
- [`docs/agent-cage-managed.md`](agent-cage-managed.md) — the heavier path for
  wrapping a third-party agent *CLI* as a registered `AgentType`.
```
