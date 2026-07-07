# docker/ — agent container images

One image per **(agent, variant)**. This directory holds the Dockerfiles;
`cage agent build` builds and tags them. Benchmarks never name a Dockerfile —
they pick an agent (and optionally a variant); the framework resolves the image.

## Naming convention

```
<agent>.Dockerfile              # the agent's base image
<agent>_<variant>.Dockerfile    # a build variant of that agent
```

- `<agent>` is exactly the registered agent name (`claude_code`, `codex`,
  `hermes`, `qwen_code`, …). It may contain `_`.
- `<variant>` is the flavor — the base environment it layers on, or a plugin:
  - `pentestenv` — layered on the shared **pentest-env** base (the
    AgentCyberRange web/post-exploit target toolchain).
  - `cyberdebug` — layered on the **oss-fuzz runner** base (CyberGym white-box
    dynamic analysis: gdb/strace + the prebuilt target ABI).
  - `openviking` — adds the **OpenViking** memory plugin.
  - `worker` — Cairn's non-root worker sub-image.

Do **not** put a benchmark name in a Dockerfile name — images are keyed by
agent × variant, never by benchmark (a benchmark only selects them).

## Build

**Native agents** (a Python `AgentType` in `cage/agents/<name>/`):

```
cage agent build --agent <name>                 # base    -> agent.default_image
cage agent build --agent <name> --variant <v>   # docker/<name>_<v>.Dockerfile -> cage/<agent>:<v>
cage agent build --all                          # base + every docker/<name>_*.Dockerfile
```

The variant path is resolved by convention (`docker/<name>_<variant>.Dockerfile`,
see `cage/cli/commands/agent.py`), so a variant only needs a correctly-named file.

**Custom agents** (a manifest `cage/agents/custom/<name>/agent.yml`) run their own
`build.script` and reference their Dockerfiles directly — still named to the
convention:

```
cage agent build --agent cairn   # docker/build_cairn.sh  -> cairn.Dockerfile (+ cairn_worker)
cage agent build --agent qitos   # docker/build_qitos.sh  -> qitos.Dockerfile
```

## Images

★ = the agent's declared default (`agent.default_image` / manifest `image:`).

| Dockerfile | agent | kind | FROM (base) |
|---|---|---|---|
| `claude_code_pentestenv.Dockerfile` | claude_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `claude_code_cyberdebug.Dockerfile` | claude_code | variant | `cybergym/oss-fuzz-base-runner` |
| `claude_code_openviking.Dockerfile` | claude_code | variant | `pursu1ng/cage-images:claude-code-latest` |
| `claude_code.Dockerfile` | claude_code | base layer | `ubuntu:22.04` |
| `codex.Dockerfile` | codex | ★ default | `ubuntu:22.04` |
| `codex_pentestenv.Dockerfile` | codex | variant | `pursu1ng/cage-images:pentest-env` |
| `gemini_cli_pentestenv.Dockerfile` | gemini_cli | ★ default | `pursu1ng/cage-images:pentest-env` |
| `hermes_pentestenv.Dockerfile` | hermes | ★ default | `pursu1ng/cage-images:pentest-env` |
| `hermes_openviking.Dockerfile` | hermes | variant | `cage/hermes:pentestenv` |
| `kimi_code_pentestenv.Dockerfile` | kimi_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `qwen_code_pentestenv.Dockerfile` | qwen_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `custom_langgraph.Dockerfile` | custom_langgraph | trace runtime | `pursu1ng/cage-images:pentest-env` |
| `cairn.Dockerfile` | cairn | custom (engine) | `ubuntu:22.04` |
| `cairn_worker.Dockerfile` | cairn | custom (worker) | `${BASE}` |
| `qitos.Dockerfile` | qitos | custom | `pursu1ng/ctfenv:latest` |

Base images a Dockerfile `FROM`-s (`pursu1ng/cage-images:pentest-env`,
`cybergym/oss-fuzz-base-runner`, …) are pulled from a registry; the images built
here (`cage/<agent>:<variant>`) are local. `build_*.sh` scripts live here too.
