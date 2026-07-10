# Benchmark-Only (Serve) Mode: External-Agent-Driven Target Ranges

**English** · [中文](benchmark-serve-mode-CN.md)

CAGE's benchmark-only (serve) mode exposes a benchmark as a set of on-demand,
isolated target ranges that an **external agent drives itself** over an HTTP API —
the loop `list → launch → attack → submit → close`. CAGE does not host the
agent's runtime and does not intercept its model calls. Compared with the
integrated path (CAGE-managed, `cage run`), serve mode trades away trajectory
capture for near-zero integration, which fits external, black-box, or non-Python
agents and self-serve evaluation / leaderboards.

## Which mode fits you? (quick check)

**If your agent already has its own mature logging and a frontend for inspecting
runs, benchmark-only (serve) mode is the better fit.** CAGE-managed's main value
is capturing the trajectory and replaying it in CAGE's own inspector; if you
already have that, wrapping your agent in CAGE's container + proxy convention is
pure overhead — point your existing agent at the served range and keep recording
your own way.

Conversely, take on the more involved CAGE-managed integration only when you want
**CAGE to capture and standardize the trajectory for you** — typically a new
agent with no observability of its own.

In short: a mature agent or framework (LangGraph, an existing harness, your team's
in-house agent) → **serve mode**; a new agent you want CAGE to record end-to-end
→ CAGE-managed.

## Two integration paths

CAGE evaluates your own agent in one of two ways; they differ in a single thing:
**does CAGE run and record the agent, or only serve the targets?**

- **CAGE-managed** — see [Adding an Agent](adding-a-new-agent.md). You plug your
  agent into CAGE and `cage run` owns the whole trial: it builds the container,
  **intercepts every model call** through an in-container proxy, snapshots state,
  and scores.
- **Benchmark-only / serve (this document)** — CAGE only serves launchable,
  isolated target ranges; the loop is **driven by the external agent**. CAGE
  never runs the agent and cannot observe its model calls.

**Table 1. The two paths**

| Dimension | Benchmark-only / serve (this doc) | CAGE-managed ([doc](adding-a-new-agent.md)) |
|---|---|---|
| Who runs the agent | You (external process) | CAGE (`cage run`) |
| Integration cost | Near zero — no Dockerfile / `agent.yml` / proxy convention | Dockerfile + `agent.yml`; your model client must call CAGE's `{base_url}` |
| Language / framework | Anything; CAGE never touches your code | Anything, but it runs inside CAGE's container model |
| Trajectory (every LLM / tool call) | Not captured; only the final score + verdict | Fully captured; the proxy records every call, replayable in the inspector |
| Mid-trial snapshots / resume | No | Yes |
| Cross-agent comparability | Weaker; the runtime is yours | Strong; the runtime is standardized by CAGE |
| Loop / retries / concurrency | You | CAGE |
| Results in `.cage_runs` + inspector | Yes (verdict only) | Yes (verdict + full trajectory) |

**Choosing:** if you need CAGE to observe, record, and compare rigorously, and
your agent can be containerized, use **CAGE-managed**; if you only need to point
an external or black-box agent at the range and get a score with zero
integration, use **serve mode**.

---

## 1. Prepare targets

As with any CAGE run, targets are built ahead of time; neither `cage run` nor
serve builds targets during evaluation. The example below uses the bundled,
offline-runnable `web_exploit_bench` target `pb-prestashop`:

```bash
git lfs install
git submodule update --init --recursive examples/agent_pentest_bench/datasets/web_exploit_bench
git -C examples/agent_pentest_bench/datasets/web_exploit_bench lfs pull
cage benchmark build web_exploit_bench --sample pb-prestashop
```

Scoring `web_exploit_bench` targets additionally requires a judge model
configured on the server (see §2); `post_exploit_bench` targets do not.

## 2. Configure the judge model (required for web_exploit_bench)

CAGE's scoring signals fall into two kinds; whether a model is needed depends on
the target type:

**Table 2. Scoring signals and model dependency**

| Target type | Scoring signal | Model needed |
|---|---|---|
| `web_exploit_bench` | `verifier` (the evaluator runs `verify.py`) + **`LLM_judge`** | Yes, for vulns that declare `LLM_judge` |
| `post_exploit_bench` | live target state (markers) | No |

So whenever an evaluation involves `web_exploit_bench`, configure a usable judge
model **on the server**. The model is invoked only at scoring time; its
credentials stay server-side and are never carried on a submission.

**(1) Create the local model registry.** `config/models.yml` is a local,
gitignored file (it holds keys), created from the example:

```bash
cp config/models.example.yml config/models.yml
```

**(2) Register the judge model** (OpenAI-compatible endpoint shown; see
[the model registry](models.md) for all fields):

```bash
export DEEPSEEK_API_KEY=...
cage model set deepseek-v4-pro \
  --provider openai \
  --model deepseek-v4-pro \
  --endpoint https://<your-endpoint>/v1 \
  --api-key '${DEEPSEEK_API_KEY}'
cage model list      # confirm it is registered
```

**(3) Default judge and override.** `agent_pentest_bench`'s `web_exploit_bench`
challenges declare `deepseek-v4-pro` as their default judge, so:

- if `config/models.yml` contains an entry with id `deepseek-v4-pro`,
  `cage benchmark serve agent_pentest_bench` works out of the box, with no extra
  flag;
- to use a different model, override with `--judge-model <id>`, where `<id>` must
  be an entry registered in `config/models.yml`:

```bash
cage benchmark serve agent_pentest_bench --judge-model <your-model-id>
```

The judge model id is resolved against `config/models.yml` at scoring time; if it
is absent, the `LLM_judge` signal of `web_exploit_bench` challenges reports "no
judge model configured" and those vulns cannot pass. `post_exploit_bench` targets
involve no model and can skip this section.

## 3. Start the server

```bash
cage benchmark serve agent_pentest_bench
#   console : http://localhost:8000/     (read-only, for operators)
#   api     : http://localhost:8000/challenges
#   judge   : deepseek-v4-pro (benchmark default)
```

This discovers and serves all of the benchmark's challenge indices
(`web_exploit_bench` and `post_exploit_bench`) together. The server is
**long-lived and lazy**: no target container starts until an agent launches one.
Override the judge with `--judge-model <id>` (§2); expose it to other machines
with `--host 0.0.0.0 --external-token "$(openssl rand -hex 16)"`.

## 4. The evaluation loop (Python SDK)

Rather than hand-rolling HTTP and multipart requests, use the bundled
**standard-library-only** client (`from cage.target.serve_client import
ServeClient`, or copy the single file `cage/target/serve_client.py` into your
agent project):

```python
from cage.target.serve_client import ServeClient

client = ServeClient("http://localhost:8000", client_id="team-red")

# (1) list challenges — no container starts
for ch in client.list_challenges():
    print(ch["id"], ch["category"])

# (2) launch — your own isolated instance: unique run_id + docker network +
#     fresh containers. Many agents may launch the same challenge concurrently
#     without interfering (target_scope=per_agent, network_only=True are defaults).
inst = client.launch("pb-prestashop")
print(inst.run_id, inst.container_addr)   # e.g. ['172.31.x.y:80'] on inst.network_name

# (3) task briefing — the same description a CAGE-managed agent receives (task,
#     target address, final_answer output contract), with THIS instance's live
#     target already filled in. Hand it straight to your agent.
task = inst.task_prompt()

# (4) attack — run your agent as a container on the serve host, joined to the
#     instance network, hitting inst.container_addr (anti-cheat, see §5). Attach a
#     container you already started:
client.attach(inst, "my-agent-container")
#     ... your agent works `task` against the target, writing final_answer/<vuln_id>.json ...

# (5) submit — score against the STILL-RUNNING target. One submission per instance
#     (terminal; a repeat returns the same verdict, already_submitted=True).
verdict = inst.submit(final_answer_dir="./final_answer")
print(verdict["scores"])   # {scorer: {value, answer, explanation, metadata}}

# (6) close — tear the instance down, releasing its containers and network
#     (or pass close=True to submit to do both at once).
client.close(inst)
```

### 4.1 The task briefing and its fields

`inst.task_prompt()` returns the ready-to-use briefing, with **this instance's
live target address(es) already filled in**. If you also want the un-filled
template, `inst.prompt()` returns the full response
`{task_prompt, task_prompt_template, prompt_level, …}`, where
`task_prompt_template` is the same briefing with target addresses shown as
placeholders (e.g. `{{APPLICATION_TARGETS}}`). The briefing is rendered by the
benchmark's own `build_prompt`, so it is identical to what `cage run` hands a
CAGE-managed agent.

### 4.2 Per-instance hint tier

The **hint tier** is bound to the instance at launch, so different instances on
one server can run at different tiers without a restart:

```python
inst = client.launch("pb-postexp-range-4", prompt_level="l1")   # this instance → l1
```

`l0` = no hints; `l1` / `l2` progressively reveal vuln location and topology. If
omitted, the launch uses the server's `--prompt-level` default (`l0`). The tier
is a launch parameter recorded on the instance, chosen by whoever drives the
launch; for a fair evaluation where agents self-launch, drive the launches
operator-side (or set the policy through the default).

### 4.3 Instance teardown

`close=True` destroys the instance right after scoring; otherwise call
`client.close(inst)` explicitly, or use the context manager
`with client.session("pb-prestashop") as inst:` to guarantee teardown on exit.
**Instances are not auto-reaped**: an instance that is never `DELETE`d keeps
holding its containers and network.

### 4.4 Submitting a post_exploit_bench target (markers, no payload)

`post_exploit_bench` targets are scored from live target state (markers), so
`submit` carries no payload — it simply means "I'm done, score my markers":

```python
with client.session("pb-postexp-range-4") as inst:
    client.attach(inst, "my-agent-container")
    # ... your agent lands user/root markers on the range ...
    print(inst.submit()["scores"])
```

### 4.5 Evaluation flow (pseudocode)

The evaluation is implemented by the caller — language, concurrency, and how the
agent is launched are all up to you; CAGE only provides the §4 SDK. The overall
flow (pseudocode):

```
Config:
  LEVELS        # subset of hint tiers to evaluate, e.g. {l0, l1, l2}
  CONCURRENCY   # degree of parallelism
  MAX_ROUNDS    # per-challenge round cap (soft termination)
  TIME_BUDGET   # per-challenge wall-clock budget (hard termination)

Concurrently (up to CONCURRENCY), for each (level, chal) in LEVELS x list_challenges():

    inst  <- launch(chal, prompt_level = level)   # isolated instance: unique run_id + network
    agent <- start an agent container DEDICATED to this task   # key to safe concurrency: one per task
    attach(inst, agent)                           # join the agent to this instance's network (anti-cheat, §5)

    Within MAX_ROUNDS / TIME_BUDGET, have the agent attack inst.container_addr:
        web_exploit_bench  -> produce vuln reports per the Reporting section of inst.task_prompt()
        post_exploit_bench -> land user / root markers on the target

    verdict <- submit(inst, report_dir)   # web: with the report dir; post: no payload; one-shot, terminal

    RECORD the whole verdict (see "Audit"), not just the score
    close(inst); reclaim the agent container
```

**Audit** — `verdict["scores"]` is far more than a single number; it is a
**per-vuln breakdown**: for each vuln, `passed` / `verifier_status` (the
evaluator's `verify.py` verdict) / `judge_status`, plus the raw `verifier_results`
and `judge_findings`. Keep the whole verdict for each challenge — together with
the `task` you handed the agent and the agent's output — as your own audit
record. You need not rely on that record alone: CAGE independently persists the
same verdict server-side under `.cage_runs` (§6), inspectable via `cage inspect`,
so the two sides cross-check each other.

**Concurrency and termination** — serve uses `per_agent`, so every launch is an
isolated instance and the workload is naturally parallel; the only requirement is
that **each concurrent task use its own agent container** (sharing one would join
a single container to multiple instance networks and cross-contaminate).
Termination is two-tier: `MAX_ROUNDS` (soft, enforced by the agent) and
`TIME_BUDGET` (hard, enforced by your scheduler); either one firing ends the
attack and triggers submission.

**Hint tiers** — the task set is `LEVELS x all challenges`, so a challenge can be
evaluated once at each of `l0` / `l1` / `l2`; each is bound to its instance via
`prompt_level = level` at launch (§4.2). To evaluate a single tier, make `LEVELS`
a single element.

The SDK calls (`list_challenges` / `launch` / `session` / `task_prompt` /
`attach` / `submit` / `close`) are covered above; the implementation, concurrency
library, and agent launch mechanism are yours to choose.

## 5. Isolation and anti-cheat

`container_addr` is an address **on the instance's isolated docker network**, not
a host port. The supported, cheat-resistant deployment is to **run your agent as
a container on the serve host and join it to the instance network** — at start
(`docker run --network <inst.network_name> your-agent`) or after the fact
(`client.attach(inst, container)`) — and then reach the target as a network peer.

Note carefully that you **must not reach the target from the host, and must not
grant the agent the docker socket**. `network_only` strips host ports, but it
cannot constrain a process running on the host: the host has a route to *every*
instance's bridge subnet (a host process could hit its own target and others'),
and a mounted `/var/run/docker.sock` lets a process `docker exec` straight into
the target to read flags/markers. Because docker networks are host-local, this
strong isolation holds only for a co-located container attached to just its own
instance network; a genuinely remote agent cannot join that network and must fall
back to the weaker host-published `entry_urls` (`network_only=false`), which
exposes ports on the host.

## 6. Results persistence and audit

Each submission is persisted as one trial under
`.cage_runs/serve__<client_id>/serve/` — the **same `.cage_runs` tree the
inspector reads** — so `cage inspect` lists it alongside `cage run` results.
Semantically, one served benchmark corresponds to "one experiment per external
agent" (each distinct `client_id` has its own scored run), and each submission
appends a trial.

What the caller receives is the **verdict, not a trajectory**:
`trials/<id>/scores/<scorer>.json` holds the full scoring detail (per-vuln
`passed` / `verifier_status` / `judge_status`, plus the raw `verifier_results`
and `judge_findings`), which is exactly what `submit` returns. Because CAGE never
ran your agent, there is **no step-by-step LLM/tool trajectory** — the inherent
limitation of serve mode. If you need the trajectory, use the
[CAGE-managed path](adding-a-new-agent.md).

## Reference

- [Serve External Audience](serve-external-audience.md) — the full HTTP contract:
  every endpoint, the two-audience port-binding model, `--external-token` auth,
  concurrency/isolation internals, and the raw `curl` equivalents of the SDK calls.
- [Adding an Agent](adding-a-new-agent.md) — the other path: let CAGE run and
  record your agent.
