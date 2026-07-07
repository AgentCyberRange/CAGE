# Running the Cairn agent on Cage

[Cairn](https://github.com/oritera/Cairn) is a multi-worker *blackboard* pentest
engine (a Server holding a Fact/Intent/Hint graph, a Dispatcher, and N
Claude-Code workers doing an OODA search). This directory makes Cairn a Cage
**custom agent**: the *whole* Cairn engine runs inside **one Cage trial
container** via Docker-in-Docker, so Cage owns per-trial hard isolation while
Cairn owns the intra-trial multi-worker search. Every worker's LLM call still
goes through the Cage proxy, so a Cairn run is measured exactly like any other
agent.

**What lives where**

| Piece | Path | Travels with Cage? |
|---|---|---|
| Manifest + entrypoint (our glue) | `cage/agents/custom/cairn/` (this dir) | yes (tracked) |
| Cairn engine source (upstream) | `third_party/Cairn/` | yes — a **pinned git submodule** |
| Trial image + worker image | `docker/cairn.Dockerfile`, `docker/cairn_worker.Dockerfile` | yes (tracked) |

The manifest ([`agent.yml`](agent.yml)) is manifest-driven (zero framework
code); the orchestration is [`cairn_cage_entry.py`](cairn_cage_entry.py), copied
into the trial container fresh each run.

---

## Prerequisites

- **Cage installed and working** — you can already run another agent (see the
  repo [`README.md`](../../../../README.md) quick start). `cage --help` works.
- **Docker on the host** that can run **`--privileged`** containers. Cairn runs
  its *own* inner Docker daemon (DinD) inside the trial container, so the trial
  container must be privileged (the manifest sets `privileged: true`).
- **A model** configured in the repo-level `config/models.yml` (created from
  `config/models.example.yml`). **Any** model works — the model is a free
  per-run choice (`--model <id>`), not baked into this agent. Cairn's workers are
  Claude-Code and the Cage proxy translates anthropic↔openai, so **both**
  Anthropic and OpenAI-compatible models run unchanged. The examples below use
  `glm-5.2-sii` purely as one concrete id — substitute your own.

---

## Step 1 — get the Cairn engine source (submodule)

The engine source is a pinned submodule, not vendored into the tree. A fresh
clone must fetch it:

```bash
git submodule update --init third_party/Cairn
```

Verify it landed (you should see the engine package):

```bash
ls third_party/Cairn/cairn/src/cairn        # -> server/ dispatcher/ ... 
git submodule status third_party/Cairn      # -> a commit hash + third_party/Cairn
```

> If you cloned with `git clone --recursive`, this is already done.

## Step 2 — build the images

Build with the standard command — the same `cage agent build` every agent uses:

```bash
cage agent build --agent cairn
```

Cairn's image is a 3-image bake (engine → worker → bake the worker tar into the
final image, so the empty inner Docker daemon can `docker load` it) that a single
`docker build` can't express. The manifest's `build:` points `cage agent build`
at `docker/build_cairn.sh`, which runs all three steps and produces
`cage/cairn:engine`, `cage/cairn-worker:latest`, and `cage/cairn:latest` (the
manifest uses the last). The build **needs the submodule checked out** (Step 1) —
it copies the engine from `third_party/Cairn/`.

Check the images exist (skip the build if they already do):

```bash
docker images | grep -E 'cage/cairn'
```

## Step 3 — pick a model

Any id defined under `models:` in `config/models.yml` works — you pass it with
`--model <id>` at run time; nothing in this agent hardcodes a model. List the
available ids:

```bash
grep -nE '^  [A-Za-z0-9_.-]+:' config/models.yml   # candidate model ids
```

Anthropic and OpenAI-compatible models both work (see Prerequisites). The Step 4
example uses `glm-5.2-sii`, but swap in any id from that list.

## Step 4 — run a trial

Post-exploitation sample, no hints, one trial:

```bash
cage run post_exploit_bench \
  --agent cairn \
  --model glm-5.2-sii \
  --sample pb-postexp-range-1 \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id cairn-quickstart \
  --upstream-proxy http://<HOST_LAN_IP>:7890
```

Notes:

- **`--model`** takes any id from `config/models.yml` — the example's
  `glm-5.2-sii` is not special. Anthropic **and** OpenAI-compatible models both
  work (the proxy translates anthropic↔openai); swap freely.
- **`--upstream-proxy`** is required when the model endpoint isn't reachable
  from *inside* the container (self-hosted vLLM/sglang like the `sii` endpoints,
  or OAuth Anthropic): the container's DNS differs from the host's. Use the
  host's **LAN IP** + your proxy port (e.g. `http://<host-ip>:7890`), **not**
  `172.17.0.1`. If your model endpoint is publicly resolvable, omit this flag.
- **`--sample`** ids are `pb-postexp-range-<N>` (post-exploit) or `pb-<target>`
  (web). `--prompt-level` is `l0` (no hints) / `l1` / `l2`.
- **`--max-concurrent 1`**: one Cairn engine already runs N internal workers
  (the `workers` param, default 2), so keep the outer per-trial concurrency low.

The run prints progress and, by default, opens the browser inspector.

## Step 5 — view the attack graph

Each trial produces Cairn's Fact/Intent attack graph, rendered in **Cairn's own
frontend** inside the Cage inspector.

1. Open the inspector. It auto-starts with `cage run`; otherwise start it with
   `cage inspect` (serves `examples/` on the port from `config/cage.yml`), then
   browse the URL it prints.
2. Open your run → your trial (e.g. `pb-postexp-range-1-l0/pass_1`).
3. Click the **Cairn graph** button in the trial detail. It renders the native
   Cairn graph (Origin→Goal, fact nodes, open-intent diamonds, labeled edges)
   for the trial — live while it runs, and the final snapshot after it ends.

The graph is also written as a plain artifact `workspace/cairn_graph.yaml` in the
trial dir.

---

## Tuning (manifest params)

Override per run with `--param k=v` (or in the experiment yaml under
`agents[].params`):

| Param | Default | Meaning |
|---|---|---|
| `workers` | `2` | N parallel Claude-Code workers inside the trial |
| `worker_image` | `cage/cairn-worker:latest` | inner worker image; for full Kali parity point at `ghcr.io/oritera/cairn-worker-container` |
| `timeout` | `3600` | whole-project wall-clock budget (seconds) |

## Troubleshooting

- **Preflight fails with `curl exited 137`** — the model is a slow *reasoning*
  model (e.g. GLM-5.2 emits `reasoning_content`, ~40s per call). The framework
  preflight already allows for this; if you hit it on an even slower model, the
  fix is in `cage/experiment/engine/preflight.py` (per-ping `--max-time` and the
  temp-container TTL). It is **not** a broken model/endpoint.
- **`403 model_auth_error` / upstream not resolvable** — you forgot
  `--upstream-proxy http://<HOST_LAN_IP>:7890` (see Step 4).
- **Build error: `COPY third_party/Cairn/...` not found** — the submodule isn't
  checked out; run Step 1.
- **Container fails to start its inner daemon / `driver not supported`** — the
  trial container must be privileged (it is, via the manifest) and the inner
  daemon uses the `vfs` storage driver (overlay-on-overlay is unsupported). This
  is handled by the image; if you rebuilt it, don't change the storage driver.
