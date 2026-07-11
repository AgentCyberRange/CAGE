# CyberGym

CyberGym is CAGE's vulnerability-reproduction benchmark. It hands a coding agent
a real, known-vulnerable program and asks it to **prove the bug**: write a single
input file — a PoC — that makes the program crash. The bugs are real OSS-Fuzz /
ARVO findings, and the PoC is graded by *actually running it* — a trial passes
only if the PoC crashes the **vulnerable** build and does **not** crash the
**fixed** build. Two catalogs ship:

| Benchmark | Run it with | Catalog |
|---|---|---|
| **CyberGym** (official) | `cybergym` | 1507 tasks (1368 ARVO + 139 OSS-Fuzz) |
| **ARVO external** | `arvo` | 3480 same-kind ARVO tasks not in CyberGym |

## What a run does

Each trial stages one task's vulnerable source tree into the agent's container
(`/home/agent/workspace`), and the agent analyses it and writes a PoC. It submits
candidates with a `submit.sh` script to a grading service that runs the PoC
against the real vulnerable/fixed builds and reports back; the agent iterates
until it lands a crash, then stops. CAGE's in-container proxy records every model
call along the way.

How much the agent is told is the **difficulty level**:

- `level0` — the vulnerable repo only.
- `level1` — repo + a text description of the bug. *(`cybergym` default)*
- `level2` — repo + description + the crash stack trace. *(`arvo` default)*
- `level3` — repo + description + trace + the fixing patch.

## 1. Set up the data (once per machine)

The catalogs are committed (`datasets/cybergym.json`, `datasets/arvo.json`), but
the large, machine-specific data is **not** in git — it resolves through three
gitignored symlinks under `datasets/`. Point each at your own copy:

```bash
cd examples/cybergym/datasets
# Per-task vulnerable repo tarballs staged into the agent container. REQUIRED.
ln -sfn /your/cybergym/data          payloads
# Prebuilt vul/fix binaries — the 'binary' grading backend (~130 GB).
ln -sfn /your/cybergym-server-data   server-binary
# Per-task ARVO image tars — the 'image' grading backend (image mode only).
ln -sfn /your/cybergym/image-cache   image-cache
```

You always need `payloads`, plus **one grading backend** — these are the two main
ways to run:

- **Binary mode** *(the `cybergym` default)* — grade against prebuilt vul/fix
  binaries (`server-binary`, ~130 GB). Covers the full 1507-task catalog and
  needs no image cache.
- **Image mode** *(the `arvo` default)* — grade against per-task `n132/arvo`
  Docker images (`image-cache`, multi-TB, loaded on demand and LRU-evicted).
  Required for the 3480 ARVO-external tasks, because the prebuilt binaries only
  cover the 1507 CyberGym tasks.

`default_cybergym.yml` sets `binary_dir` (binary mode); `default_arvo.yml` leaves
it unset (image mode). Each file's comments show how to switch the other way.

## 2. Configure a model

Model endpoints live in the git-ignored `config/models.yml` (it holds your keys):

```bash
cp config/models.example.yml config/models.yml
export GLM_API_KEY=...
cage model set glm-5.1 --provider openai --model GLM-5.1 \
  --endpoint https://open.bigmodel.cn/api/paas/v4 --api-key '${GLM_API_KEY}'
cage model list
```

The default configs run the `claude_code` agent on the `cage/claude-code:pentestenv`
image — build it with `cage agent build --agent claude_code --variant pentestenv`,
or point the config at your own. If the agent container can only reach your model through a host proxy,
uncomment `proxy.upstream_http_proxy: http://172.17.0.1:7890` in the config.

## 3. Smoke-test one task

There is no target image to prebuild — grading is benchmark-owned — so the flow
is just check, then run. Sample ids are the catalog keys (`arvo:1065`; the
underscore form `arvo_1065` works too):

```bash
# render this task's prompt + config only (no model call, no grading)
cage benchmark check cybergym --agent claude_code --model glm-5.1 \
  --sample arvo:1065 --show-prompt

# one trial
cage run cybergym --agent claude_code --model glm-5.1 \
  --sample arvo:1065 --passk 1 --max-concurrent 1 --run-id cybergym-smoke-001
```

ARVO external is identical, with the `arvo` id:

```bash
cage run arvo --agent claude_code --model glm-5.1 \
  --sample arvo:289 --passk 1 --max-concurrent 1 --run-id arvo-smoke-001
```

## 4. Run the full catalog

Drop `--sample` to run every task (1507 for `cybergym`, 3480 for `arvo`). Give the
run a `--run-id` so you can track and resume it:

```bash
cage run cybergym --agent claude_code --model glm-5.1 --run-id cybergym-full-001
cage run arvo     --agent claude_code --model glm-5.1 --run-id arvo-full-001
```

Narrow a big run several ways:

- **Specific tasks** — `--sample arvo:1065 --sample arvo:1461` (also
  comma-separated, or `--sample @ids.txt` for a file of ids).
- **A pinned subset** — set `eval.benchmark.task_ids: ./datasets/cybergym_images.txt`
  in the config (the shipped file is the official trace-100), or any file of ids.
- **First N / parallelism** — `--max-sample-num 100`, `--max-concurrent N`.
- **Difficulty** — `eval.benchmark.difficulty: level0..level3` (or
  `--set eval.benchmark.difficulty=level0`).

`--dry-run` prints the plan (tasks, budgets) without running; `--run-id … --resume`
continues a run, skipping the trials that already finished.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to
browse trials, scores, and full agent trajectories live as the run proceeds.

## Explore the CLI

```bash
cage --help               # top-level commands: run, benchmark, model, agent, inspect, score, gc
cage benchmark list       # registered benchmarks (cybergym, arvo, …)
cage run cybergym --help  # this benchmark's samples, agent/model matrix, defaults, and flags
cage model list           # model endpoints you've registered
```
