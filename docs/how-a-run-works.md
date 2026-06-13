# How a Run Works

You have run one trial in [Quick Start](/getting-started/). This page explains
what that single `cage run` actually did — the lifecycle it walked, the runtime
machinery behind it, and where every artifact landed. Read it once and the rest
of CAGE stops being a pile of commands and becomes one pipeline with knobs.

## The run equation

A run is the framework executing a benchmark under your config:

```text
cage run  =  Framework ( Benchmark , Config )
             └ Layer 1 ┘ └Layer 2 ┘ └Layer 3┘
```

| Layer | Supplies | Lives in | Example |
|---|---|---|---|
| **1 — Framework** | the *mechanism*: container, proxy, target, scoring, snapshots, resume | `cage/` | "launch the agent container, intercept its model calls, snapshot state" |
| **2 — Benchmark** | *what* is evaluated: samples, prompts, targets, scorer | `examples/<name>/` | "this sample is a ComfyUI app; success = the agent reaches the flag" |
| **3 — You** | *how this run goes*: limits, agent/model, sample selection | experiment YAML + CLI | "Codex on gpt-5.5, 3 samples, pass@1, 2-hour timeout" |

::: info Why it's built this way
The framework holds **zero** benchmark names. A new benchmark is a new
`examples/<name>/` directory, never an edit to `cage/`. This is the single
invariant the whole design protects — it is what lets one runtime serve web
exploitation, CTF, CVE reproduction, and red-teaming without growing a tangle of
`if benchmark == "...":` branches. See [Architecture](/repo-architecture) for the
layer test that keeps it honest.
:::

## The trial lifecycle

`cage run` resolves your config, builds the agent × model × sample × pass@k
matrix, and then drives each **trial** through the same ordered lifecycle. The
benchmark only owns a handful of the steps (marked **L2**); the framework owns
the rest.

```text
cage run project.yml
  └─ resolve()                          parse YAML → one ExperimentRun
  └─ build agent/model/sample matrix
  └─ for each trial:
       SETUP
         1. launch the target stack            (Docker compose, per trial)
         2. connect agent container to target network
         3. snapshot pre-state
         4. reset the agent workspace
         5. prepare_trial()              L2     copy files into the workspace
         6. inject target info into the sample
         7. start the submit service            (when the benchmark uses flags)
       EXECUTE
         8. start the in-container LLM proxy
         9. build_prompt()              L2     render the agent-facing prompt
        10. start live-success monitors
        11. run the agent CLI                   the agent works the task
        12. stop the proxy, collect its logs
       SCORE
        13. snapshot post-state
        14. collect runtime artifacts
        15. score()                    L2     prefer the live verdict, else parse output
       CLEANUP
        16. tear down submit service, target network, target stack
```

The four phases map onto the four things a fair evaluation needs: a **clean,
reproducible environment** (setup), an **observed run** (execute), a **verdict
derived from evidence** (score), and **no leaked resources** (cleanup).

::: info Why it's built this way
Sample mutation (step 6) is **orchestrator-owned**, not benchmark-owned. The
benchmark's `iter_samples()` yields immutable sample dicts; the framework injects
the live target's address/network into the sample just before `prepare_trial`
and `build_prompt`. That keeps a benchmark replayable: the same sample produces
the same trial regardless of which host or port the target happened to get.
:::

When `agents[].max_concurrent > 1`, the framework runs each trial in its own
isolated agent container and target lifecycle, so parallel trials never share a
workspace or a target.

## The runtime substrate

Each box below is one piece of Layer 1 — what it is, why it exists, and the
artifact it leaves behind for you to inspect later.

### Container

`cage/sandbox/containers.py` wraps Docker. One agent container per trial (or per
run, for stateful agents) owns:

- `docker run` construction, env wiring, volumes;
- the `host.docker.internal:host-gateway` mapping;
- workspace setup/reset;
- joining the per-trial target network on top of its base network;
- file transfer for state snapshots.

The agent runs as the unprivileged `agent` user with `HOME=/home/agent`. That
convention is hardcoded across the framework — agents and images must honor it.

### In-container proxy

CAGE starts a proxy **inside** the agent container and points the agent's model
endpoint at it. Every model call the agent makes is therefore observed:

```text
cage/proxy/host.py
  → writes proxy config JSON into the container
  → copies cage/proxy/sidecar.py to /opt/cage-proxy/
  → starts it, health-checks /healthz
  → collects /tmp/cage_proxy_logs/proxy.jsonl
```

The sidecar translates Anthropic-style calls to the upstream protocol when the
agent and model disagree, records every request/response, reconstructs tool
calls, and enforces the run's budgets (rounds / tokens / cost). Its upstream
client can egress through a host proxy (`proxy.upstream_http_proxy`).

::: info Why it's built this way
The in-container sidecar has exactly **one** runtime dependency: `httpx`. Every
agent image ships it, so the proxy works in any container without a heavy
install. `proxy.jsonl` — not the agent's stdout — is the source of truth for what
the model was asked and what it answered. The CLI's text output is convenience;
the proxy log is evidence.
:::

**Artifact:** `trials/<trial>/proxy/proxy.jsonl` (raw traffic) and
`proxy/progress.json` (running token/round/cost totals).

### Target runtime

Targets are managed through `cage/target/client.py`. `ChallengeClient` is the
single interface the orchestrator sees; behind it sit `LocalBackend` (runs
`docker compose` here) and `RemoteBackend` (calls an embedded target server over
HTTP). For `cage run`, the framework spawns that target server as a subprocess
and records its log under `.cage_runs/`.

A target exposes itself to the agent in one of two styles, chosen by the
benchmark:

| Style | Sample fields | Agent guidance |
|---|---|---|
| Alias | `target_info`, `network_name` | reach a service by hostname, e.g. `target:9090` |
| Network | `network_subnet`, `network_name` | discover hosts inside the subnet |

For the deeper network-topology policy (`network_mode`, project-local vs shared
bridges, multi-network ranges) see
[Architecture › Target Runtime](/repo-architecture#target-runtime).

### State snapshots

Steps 3 and 13 snapshot the agent's state directories before and after the
trial. For a **stateful** agent (`shared_paths` non-empty) this captures
persistence across trials; for a **stateless** agent it captures the per-trial
delta. Stateful-vs-stateless is a property of the *agent config*, not the
benchmark.

**Artifact:** `trials/<trial>/state_pre/` and `state_post/`.

::: warning shared_paths must live outside the workspace
`reset_directory` wipes the workspace between trials. Any `shared_paths` entry
inside the workspace disappears silently. Keep shared state outside the workspace
dir.
:::

### Scoring runtime

The benchmark's `score()` runs at step 15, but the **same** scorer with the
**same** `ScoringContext` also runs in two other places: as a mid-trial live
monitor, and offline via `cage score`. One scorer, three call sites. If a scorer
works inline but not offline, the scorer is the bug, not the call site.

Scoring prefers a **live-success verdict** (written while the agent was still
running) over post-hoc text extraction:

```text
trials/<trial>/runtime/live_success.json
```

A benchmark that can tell success *during* the trial (a CTF flag submitted, a
CVE check endpoint touched) writes this verdict; scoring trusts it first. See
[Architecture › Live Check](/repo-architecture#live-check-and-success-verdicts).

**Artifact:** `trials/<trial>/scores/<benchmark>.json`.

### Termination and resume

Every trial ends with a structured `termination_reason` in `meta.json`, derived
by a single classifier from **structural** signals only — process exit code,
proxy HTTP statuses, budget counters — never by scanning the agent's prose for
phrases like "rate limit". That determinism is what makes `--resume` safe: resume
replays a finished outcome from disk and only re-runs the infrastructure failures
you opt into (`resume.retry_reasons`). A budget stop like `max_rounds_reached` is
*not* re-run by default — retrying it would just exhaust the budget again.

The full detection order, the resume filters (`select` / `keep_if`), and the
migration path for old runs are in
[Architecture › Termination Classification](/repo-architecture#termination-classification).

**Artifact:** `trials/<trial>/meta.json` (`termination_reason`, timing, exit code).

### Storage and the inspector

Everything above writes through `cage/artifacts/run_storage.py`, which owns the
`.cage_runs` layout. The run directory is the **source of truth** — Docker
resources are disposable, but artifacts persist as audit evidence.

The web inspector (`cage inspect`) is a read-only renderer over that tree: it
discovers runs, renders dashboards and per-trial detail, and parses trajectories
from `proxy.jsonl`. It has no per-agent or per-benchmark branches — benchmark
meaning reaches the inspector only through `Benchmark.build_dashboard()`.

## Where everything lands

```text
.cage_runs/
  <agent_id>:<model_id>:<lifecycle>/          lifecycle = stateful | stateless
    <run_id>/
      dashboard.json                           per-run aggregate projection
      results.csv
      trials/
        <sample>/                              one dir per trial (pass_N nests here for pass@k)
          meta.json                            termination_reason, timing, exit code
          prompt.txt
          task_output.json
          proxy/proxy.jsonl                    raw model traffic (source of truth)
          proxy/progress.json                  running token/round/cost totals
          scores/<benchmark>.json
          state_pre/  state_post/
          runtime/
            live_success.json                  first accepted success verdict
            check_done_*.jsonl
```

`dashboard.json` is the one-file projection that powers the inspector overview
and `cage gc`'s liveness check. The raw per-trial files are the evidence behind
it.

## Where to go next

- [The CLI](/cli-design) — how `benchmark check`, `score`, `inspect`, and `gc`
  are each just a slice of this lifecycle.
- [Architecture](/repo-architecture) — the design principles, the layer test,
  and the deep mechanics (networking, termination, live check) this page links
  into.
- [Configuring Runs](/reference/project-yml) — the YAML and flags that
  parameterize each step above.
