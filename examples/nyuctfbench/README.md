# NYU CTF

NYU CTF is CAGE's capture-the-flag benchmark. It drops a coding agent into a
container with one CTF challenge — a puzzle from a real competition across
categories like **crypto**, **rev** (reverse engineering), **pwn** (binary
exploitation), **web**, **forensics**, and **misc** — and asks it to recover the
hidden **flag** (a secret string of the form `flag{...}`). A trial passes when
the agent surfaces the correct flag: CAGE matches it against the known answer, so
scoring is exact, not judged. The in-container proxy records every model call.

The 192 shipped challenges are all **static-file** tasks: the agent works from a
few handout files (a binary, a script, a zip) staged into its workspace. There is
no live service to attack and **nothing to build**.

## 1. Set up the data (once per machine)

The challenges live in a git submodule (`cage-org/NYUCTF-Bench`) under
`datasets/`. Initialize it:

```bash
git submodule update --init examples/nyuctfbench/datasets
```

That gives you the index (`datasets/nyu_ctf.json`, 192 challenges) and every
challenge's handout files. The repo is large; see its README for shallow/sparse
checkout options.

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
host proxy, set
`proxy.upstream_http_proxy: http://<host-ip>:7890` in the config (a LAN IP the
container can reach, not `localhost`).

## 3. Smoke-test one challenge

Sample ids are the index keys (`<event>-<category>-<name>`). There is **no target
to build** — static-file challenges — so the flow is just check, then run. The
dataset root (`./datasets`) resolves against your current directory, so run from
the example dir:

```bash
cd examples/nyuctfbench

# render this challenge's prompt + config only (no model call, no scoring)
cage benchmark check nyu_ctf --sample 2021f-cry-collision_course --show-prompt

# one trial
cage run nyu_ctf --agent claude_code_baseline --model openai-example \
  --sample 2021f-cry-collision_course --passk 1 --max-concurrent 1 \
  --run-id nyu-ctf-smoke-001
```

(`check` prints a "to build this target" hint — ignore it here; `cage run` stages
the handout files directly and never builds.)

## 4. Run the full set

Drop `--sample` to run all 192 challenges; give the run a `--run-id` to track and
resume it:

```bash
cage run nyu_ctf --agent claude_code_baseline --model openai-example --run-id nyu-ctf-full-001
```

Narrow or steer a run: `--sample <id>` (repeatable) for specific challenges,
`--max-sample-num 20` / `--max-concurrent N` for first-N and parallelism,
`--dry-run` to preview the plan, `--run-id … --resume` to continue one.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to
browse trials, scores, and full agent trajectories live as the run proceeds.

## Explore the CLI

```bash
cage --help               # top-level: run, benchmark, model, agent, inspect, score, gc
cage benchmark list       # registered benchmarks (nyu_ctf, …)
cage run nyu_ctf --help   # this benchmark's samples, agent/model matrix, and flags
cage model list           # model endpoints you've registered
```
