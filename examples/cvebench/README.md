# CVEBench

CVEBench is CAGE's web-exploitation benchmark. Each task boots a real, known-vulnerable
web application in Docker and asks a coding agent to **break in** — trigger one of a
fixed menu of effects (denial of service, file read, remote code execution, database
read/write, admin login, privilege escalation, or a forced outbound request). The
target ships a **live-check service** the agent (and CAGE) poll to confirm the
objective; a trial passes when that service reports success. CAGE's in-container proxy
records every model call along the way.

Each task has two difficulty **variants** — `zero_day` (no CVE details) and `one_day`
(the agent is told which CVE) — so a sample id is the challenge id plus the variant,
e.g. `cvb-CVE-2023-37999-zero_day`.

## 1. Get the data

The dataset is a git submodule (`cage-org/CVE-Bench`) holding the task index
(`datasets/cvebench.json`), per-challenge metadata, and the Docker Compose stack that
boots each vulnerable app plus its grading service. Initialize it once:

```bash
git submodule update --init examples/cvebench/datasets
```

There is **no `cage benchmark build` step** — targets are benchmark-owned. CAGE brings
up each task's Compose stack on demand when a trial runs (the first run pulls/builds the
app's images, which can take a while).

## 2. Configure a model

Model endpoints live in the git-ignored `config/models.yml` (it holds your keys):

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
cage model set openai-example --provider openai --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 --api-key '${OPENAI_API_KEY}'
cage model list
```

The default config runs the `claude_code` agent (agent id `claude_code_baseline`) on
the `openai-example` model — point `openai-example` at whatever endpoint you actually
have, or swap in another agent/model. If the agent container can only reach your model
through a host proxy, uncomment `proxy.upstream_http_proxy: http://<host-ip>:7890` in
`default_cvebench.yml` (use a LAN IP the container can reach, not `localhost`).

## 3. Smoke-test one task

No target prebuild needed — just check the prompt, then run one trial:

```bash
# render this task's prompt + config only (no target, no model call, no cost)
cage benchmark check cvebench --sample cvb-CVE-2023-37999-zero_day --show-prompt

# one trial
cage run cvebench --agent claude_code_baseline --model openai-example \
  --sample cvb-CVE-2023-37999-zero_day --passk 1 --max-concurrent 1 \
  --run-id cvebench-smoke-001
```

`--passk` is how many independent attempts the task gets; `--max-concurrent` caps
trials running in parallel; `--run-id` names the run (artifacts, inspector grouping,
and `--resume` all key off it).

## 4. Run the full set

Drop `--sample` to run all 40 samples (20 challenges × 2 variants). Give the run a
`--run-id` so you can track and resume it:

```bash
cage run cvebench --agent claude_code_baseline --model openai-example \
  --run-id cvebench-full-001
```

Narrow it with `--sample` (repeatable, or `@ids.txt`), `--max-sample-num N` (first N),
`--max-concurrent N` (parallelism), or `variants: [one_day]` under `eval.benchmark` in
the config (one difficulty). `--dry-run` prints the plan without running; `--resume`
with a `--run-id` continues a run, skipping finished trials.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to browse
trials, scores, and full agent trajectories live as the run proceeds.

## Explore the CLI

```bash
cage --help                # top-level commands: run, benchmark, model, agent, inspect, score, gc
cage benchmark list        # registered benchmarks (cvebench, …)
cage run cvebench --help   # this benchmark's samples, agent/model matrix, defaults, and flags
cage model list            # model endpoints you've registered
```
