# docker/ — agent container images

One image per **(agent, variant)**. This directory holds the Dockerfiles, one
sub-directory per agent; `cage agent build` builds and tags them. Benchmarks
never name a Dockerfile — they pick an agent (and optionally a variant); the
framework resolves the image.

## Layout

```
docker/<agent>/Dockerfile              # the agent's base image
docker/<agent>/<variant>.Dockerfile    # a build variant of that agent
docker/<agent>/build.sh                # optional: a multi-image build recipe
```

- `<agent>` is exactly the registered agent name (`claude_code`, `codex`,
  `hermes`, `qwen_code`, …). The directory carries the name, so the filenames
  don't repeat it.
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
cage agent build --agent <name> --variant <v>   # docker/<name>/<v>.Dockerfile -> cage/<agent>:<v>
cage agent build --agent <name> --version 0.140.0   # bake a specific CLI version
cage agent build --all                          # base + every docker/<name>/<variant>.Dockerfile
```

The variant path is resolved by convention (`docker/<name>/<variant>.Dockerfile`,
see `cage/cli/commands/agent.py`), so a variant only needs a correctly-placed
file. The bare `docker/<name>/Dockerfile` is the base and is never built as a
variant.

**Version selection.** `--version` (default `latest`) picks which agent-CLI
version gets baked in; it maps to the Dockerfile's `ARG <AGENT>_VERSION`. Every
agent Dockerfile installs its CLI as the **last** build layer, on purpose:
changing `--version` re-runs only that one step — apt, Node, the proxy, and the
agent user all stay cached, so a version rebuild is seconds, not a full rebuild.
Leave it at `latest` for the newest CLI, or pin (`--version 0.140.0`) for a
reproducible image. To keep several versions side by side, give each its own tag
(`docker build -f … --build-arg CODEX_VERSION=0.140.0 -t cage/codex:0.140.0 .`)
and point the agent's image at the tag you want.

**Custom agents** (a manifest `cage/agents/custom/<name>/agent.yml`) run their own
`build.script`, which references their Dockerfiles directly:

```
cage agent build --agent cairn   # docker/cairn/build.sh  -> cairn/Dockerfile (+ cairn/worker.Dockerfile)
cage agent build --agent qitos   # docker/qitos/build.sh  -> qitos/Dockerfile
```

## Images

★ = the agent's declared default (`agent.default_image` / manifest `image:`).

| Dockerfile | agent | kind | FROM (base) |
|---|---|---|---|
| `claude_code/pentestenv.Dockerfile` | claude_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `claude_code/cyberdebug.Dockerfile` | claude_code | variant | `cybergym/oss-fuzz-base-runner` |
| `claude_code/openviking.Dockerfile` | claude_code | variant | `cage/claude-code:pentestenv` |
| `claude_code/Dockerfile` | claude_code | base layer | `ubuntu:22.04` |
| `codex/pentestenv.Dockerfile` | codex | ★ default | `pursu1ng/cage-images:pentest-env` |
| `codex/Dockerfile` | codex | base layer | `ubuntu:22.04` |
| `gemini_cli/pentestenv.Dockerfile` | gemini_cli | ★ default | `pursu1ng/cage-images:pentest-env` |
| `hermes/pentestenv.Dockerfile` | hermes | ★ default | `pursu1ng/cage-images:pentest-env` |
| `hermes/openviking.Dockerfile` | hermes | variant | `cage/hermes:pentestenv` |
| `kimi_code/pentestenv.Dockerfile` | kimi_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `qwen_code/pentestenv.Dockerfile` | qwen_code | ★ default | `pursu1ng/cage-images:pentest-env` |
| `custom_langgraph/Dockerfile` | custom_langgraph | trace runtime | `pursu1ng/cage-images:pentest-env` |
| `cairn/Dockerfile` | cairn | custom (engine) | `ubuntu:22.04` |
| `cairn/worker.Dockerfile` | cairn | custom (worker) | `${BASE}` |
| `qitos/Dockerfile` | qitos | custom | `ubuntu:22.04` |

Base images a Dockerfile `FROM`-s (`pursu1ng/cage-images:pentest-env`,
`cybergym/oss-fuzz-base-runner`, …) are pulled from a registry; the images built
here (`cage/<agent>:<variant>`) are local. Each agent's `build.sh` (when it needs
one) lives in its own sub-directory.
