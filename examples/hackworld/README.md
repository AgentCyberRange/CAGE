# HackWorld

HackWorld is CAGE's **web CTF** benchmark. Each task is a *capture-the-flag*
challenge: CAGE boots a deliberately vulnerable web app in Docker, and the
coding agent must find and exploit a bug to recover a hidden **flag** — a secret
string in a fixed format such as `HTB{...}`. A trial **passes** when CAGE confirms
the flag, either from a live verdict on the running challenge or by finding the
expected flag string in the agent's output. CAGE records every LLM and tool call
through its in-container proxy along the way.

## 1. Get the data

The task catalog (`datasets/hackworld.json`, 36 web challenges) and the per-task
challenge dirs (`datasets/web/<challenge>/`, each with a `challenge.json`,
`Dockerfile`, and `docker-compose.yml`) live **under this example**, but are
gitignored — a fresh clone won't have them. Populate
`examples/hackworld/datasets/` (copy in your own, or set `HACKWORLD_BENCHMARK_ROOT`
to a root containing `hackworld.json` + `web/`), then confirm with
`cage run hackworld --help` (it prints the sample count and catalog path).

## 2. Configure a model

Model endpoints live in the git-ignored `config/models.yml` (it holds your keys):

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
cage model set openai-example --provider openai --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 --api-key '${OPENAI_API_KEY}'
cage model list
```

The default config runs the `claude_code` agent (id `claude_code_baseline`) on
the `openai-example` model. If the container reaches your model only through a
host-side proxy, uncomment `proxy.upstream_http_proxy: http://<host-ip>:7890` in
`default_hackworld.yml` (a Docker-reachable LAN IP, not `localhost`).

## 3. Smoke-test one challenge

Each challenge ships its own Docker target, so the flow is **check → build →
run**. `cage run` never builds targets — an unbuilt target fails pre-flight, so
build the one you're about to run first. Sample ids are the catalog keys from
`hackworld.json` (e.g. `cb-htb-web-very_easy_flag_command`).

```bash
# render this task's prompt + config only (no target, no model call, no cost)
cage benchmark check hackworld --sample cb-htb-web-very_easy_flag_command --show-prompt

# build this challenge's Docker target — REQUIRED before any run
cage benchmark build hackworld --sample cb-htb-web-very_easy_flag_command

# one trial
cage run hackworld --agent claude_code_baseline --model openai-example \
  --sample cb-htb-web-very_easy_flag_command \
  --passk 1 --max-concurrent 1 --run-id hackworld-smoke-001
```

`--passk` is how many independent attempts the challenge gets; `--max-concurrent`
caps trials in flight; `--run-id` names the run (artifacts, inspector, and resume
all key off it).

## 4. Run the full set

Build every target up front (still required), then drop `--sample` to run all 36:

```bash
cage benchmark build hackworld --max-concurrent 4
cage run hackworld --agent claude_code_baseline --model openai-example --run-id hackworld-full-001
```

Narrow with the same flags (`--sample`, `--passk`, `--max-sample-num N`).
`--dry-run` prints the plan without running; `--run-id … --resume` continues a
run, skipping finished trials.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to
browse trials, scores, and full agent trajectories live as the run proceeds.

## Explore the CLI

```bash
cage --help               # top-level: run, benchmark, model, agent, inspect, score, gc
cage benchmark list       # registered benchmarks (hackworld, …)
cage run hackworld --help # this benchmark's samples, agent/model matrix, defaults, flags
cage model list           # model endpoints you've registered
```
