# AutoPenBench

AutoPenBench is CAGE's autonomous penetration-testing benchmark. Each trial puts
a coding agent on the same private network as a live, vulnerable target machine
and asks it to break in and recover a **flag** — a secret token hidden on the
target that proves the machine was compromised. The agent gets a natural-language
mission brief and a shell; it scans, exploits, and pivots on its own. A trial
**passes** when the target's own success check fires, or when the expected flag
token shows up in the agent's output. Scoring is automatic.

The 29 tasks span categories like access control, cryptography, and network
security, at two difficulty tiers (`in-vitro` controlled labs, `real-world`).

## 1. Get the data

The tasks live in a git submodule (`cage-org/AutoPenbench`): a committed index
plus a per-machine `docker-compose` tree. No Git LFS, no fetch script — just
initialize the submodule:

```bash
git submodule update --init examples/autopenbench/datasets
```

CAGE reads `datasets/autopenbench.json` to know which tasks exist. The vulnerable
target machines are **not** prebuilt images — CAGE boots each one with
`docker compose` at run time, so there is nothing to build first.

## 2. Configure a model

Model endpoints live in the git-ignored `config/models.yml` (it holds your keys):

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
cage model set openai-example --provider openai --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 --api-key '${OPENAI_API_KEY}'
cage model list
```

The default config runs the `claude_code` agent on the `openai-example` model
(gpt-5.5). Swap in any agent/model you've registered. If the agent container can
only reach your model endpoint through a host proxy, set
`proxy.upstream_http_proxy: http://<host-ip>:7890` in `default_autopenbench.yml`
(use a Docker-reachable LAN IP, not `localhost`).

## 3. Smoke-test one task

No target image to prebuild — the flow is just check, then run. Sample ids are
the index keys (form `apb-<tier>-<category>-<vm>`):

```bash
# render this task's prompt + config only (no target, no model call)
cage benchmark check autopenbench --agent claude_code_baseline --model openai-example \
  --sample apb-in-vitro-access_control-vm0 --show-prompt

# one trial — boots the target, runs the agent, auto-scores
cage run autopenbench --agent claude_code_baseline --model openai-example \
  --sample apb-in-vitro-access_control-vm0 \
  --passk 1 --max-concurrent 1 --run-id autopenbench-smoke-001
```

`--passk` is how many independent attempts the task gets; `--max-concurrent`
caps parallel trials; `--run-id` names the run.

## 4. Run the full set

Drop `--sample` to run all 29 tasks. Give the run a `--run-id` to track and
resume it:

```bash
cage run autopenbench --agent claude_code_baseline --model openai-example \
  --run-id autopenbench-full-001
```

Narrow or tune the run:

- **Specific tasks** — repeat `--sample apb-... --sample apb-...`.
- **First N** — `--max-sample-num 5`.
- **Parallelism** — `--max-concurrent N`.
- `--dry-run` prints the plan without launching; `--run-id <id> --resume`
  continues a run, skipping finished trials.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to
browse trials, scores, and full agent trajectories (every LLM and tool call,
proxy-traced) live as the run proceeds.

## Explore the CLI

```bash
cage --help                  # top-level: run, benchmark, model, agent, inspect, score, gc
cage benchmark list          # registered benchmarks (autopenbench, …)
cage run autopenbench --help # this benchmark's samples, agent/model matrix, and flags
cage model list              # model endpoints you've registered
```
