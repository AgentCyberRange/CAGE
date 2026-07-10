# Running the QitOS agent on Cage

[QitOS](https://github.com/WhitzardAgent/qitos) is an agent framework; the
`cybergym_agent` on top of it is a CyberGym PoC-generation harness (source-audit
→ craft a raw input → submit → iterate on the vul-side crash oracle). This
directory makes QitOS a Cage **custom agent**: the harness runs inside **one Cage
trial container** and its every LLM call goes through the Cage proxy, so a QitOS
run is measured exactly like any other agent (`claude_code`, `cairn`, …).

It needs **no changes to the upstream harness** — Cage's cybergym benchmark
already speaks the upstream CyberGym contract (multipart `submit.sh`, task_id
masking), which is exactly what `cybergym_agent` expects.

**What lives where**

| Piece | Path | Travels with Cage? |
|---|---|---|
| Manifest + entrypoint (our glue) | `cage/agents/custom/qitos/` (this dir) | yes (tracked) |
| QitOS framework (upstream) | `third_party/qitos/` | yes — a **pinned git submodule** |
| CyberGym agent (upstream) | `third_party/cybergym_agent/` | yes — a **pinned git submodule** |
| Trial image | `docker/qitos/Dockerfile` | yes (tracked) |

The manifest ([`agent.yml`](agent.yml)) is manifest-driven (zero framework code);
the orchestration is the thin [`qitos_cage_entry.py`](qitos_cage_entry.py), copied
into the trial container fresh each run (edit it with no image rebuild). It
implements zero QitOS logic — it only reads the grading-service URL out of the
staged `submit.sh` and hands off to `python -m cybergym_agent.run_local`.

---

## Prerequisites

- **Cage installed and working** — `cage --help` runs; you can already run another
  agent (see the repo [`README.md`](../../../../README.md) quick start).
- **The cybergym benchmark data** set up under `examples/cybergym/datasets/`
  (payloads + binary grading data — see that dir's notes).
- **A model endpoint** in `config/models.yml`. QitOS uses an OpenAI-compatible
  client, so pick an `openai`/`vllm` protocol id. This guide uses `glm-5.1-sii`.

## Step 1 — get the QitOS source (submodules)

The harness source is two pinned submodules, not vendored into the tree. A fresh
clone must fetch them:

```bash
git submodule update --init third_party/qitos third_party/cybergym_agent
```

Verify they landed:

```bash
ls third_party/qitos/qitos              # the qitos package
ls third_party/cybergym_agent/run_local.py
git submodule status third_party/qitos third_party/cybergym_agent   # -> a commit hash each
```

> The commit hash printed by `git submodule status` is the exact upstream version
> a build runs — that is the run's traceability record.

## Step 2 — build the image

```bash
docker build -f docker/qitos/Dockerfile -t cage/qitos:latest .
```

The build **needs the submodules checked out** (Step 1) — the Dockerfile copies
the harness from `third_party/`. Only the heavy, stable deps (qitos +
cybergym_agent + the proxy) live in the image; the thin `qitos_cage_entry.py` is
copied in at trial start, so you can edit it without rebuilding.

## Step 3 — run a trial

```bash
cage run cybergym \
  --agent qitos \
  --model glm-5.1-sii \
  --sample arvo_10013 \
  --max-concurrent 1 \
  --run-id qitos-quickstart \
  --upstream-proxy http://<HOST_LAN_IP>:7890
```

Notes:

- **`--upstream-proxy`** is required when the model endpoint isn't reachable from
  *inside* the container (self-hosted vLLM/sglang like the `sii` endpoints):
  the container's DNS differs from the host's. Use the host's **LAN IP** + your
  proxy port (e.g. `http://<host-ip>:7890`), **not** `172.17.0.1`. Omit it if
  your endpoint is publicly resolvable.
- **`--max-rounds N`** caps the harness' agent steps (maps to `--max-steps`).
- Grading is Cage's: the harness submits PoCs to the per-trial grading service
  via the staged `submit.sh`; the authoritative vul+fix verdict is Cage's scorer.

## Choosing which version of the harness to run (the submodules)

The harness source is **not copied into Cage** — it lives in the two submodules
`third_party/qitos` and `third_party/cybergym_agent`, and Cage only remembers
**which commit** of each to use (this remembered commit is called the *pin*). The
image is always built from whatever commit the submodule currently points at.

Two facts worth knowing before the commands:

- **See the current pin at any time:**
  ```bash
  cd /path/to/cage
  git submodule status third_party/cybergym_agent
  # ->  7e3f686c... third_party/cybergym_agent (para_action)
  #     the leading hash is the pinned commit. A leading '+' means the checkout
  #     has moved off the recorded pin and you haven't saved it yet (see below).
  ```
- **`git add third_party/<name>` is the step that actually records your choice.**
  Moving the submodule alone changes nothing for anyone else — the new pin is
  only saved into Cage once you `git add` it (and commit). Skip it and a fresh
  clone still gets the old version.

Each scenario below ends with `git add`, then a rebuild so the image matches the
new pin. All commands run from the Cage repo root
(`/path/to/cage`).

### A — move to the LATEST commit of the branch it already follows (most common)

Each submodule is set to *follow* a branch (recorded in `.gitmodules`:
`cybergym_agent` follows `para_action`, `qitos` follows `qitos_cybergym`).
`--remote` means "fetch that branch and jump to its newest commit":

```bash
cd /path/to/cage
git submodule update --remote third_party/cybergym_agent   # jump to newest commit on para_action
git add third_party/cybergym_agent                         # record the new pin into Cage
cage agent build --agent qitos                             # rebuild the image from the new pin
```

### B — pin to a SPECIFIC commit or branch (most precise; recommended)

Use this when you want an *exact* version — a particular commit hash, or the tip
of some other branch — instead of "whatever is newest on the followed branch":

```bash
cd /path/to/cage/third_party/cybergym_agent
git fetch origin                     # download the latest commits/branches from GitHub
git checkout <commit-hash>           # e.g. a full sha, or a branch tip like: origin/main
cd /path/to/cage
git add third_party/cybergym_agent   # record the new pin into Cage
cage agent build --agent qitos       # rebuild the image from the new pin
```

### C — change WHICH branch the submodule follows (e.g. para_action -> main)

Scenario A follows the branch written in `.gitmodules`. To change that branch
permanently (so future `--remote` updates track the new one):

```bash
cd /path/to/cage
git config -f .gitmodules submodule.third_party/cybergym_agent.branch main
git submodule update --remote third_party/cybergym_agent   # jump to newest commit on main
git add .gitmodules third_party/cybergym_agent             # record both the branch change and the new pin
cage agent build --agent qitos
```

> The same commands work for `third_party/qitos` — just swap the path.
> Nothing in `cage/` changes when you re-pin; the glue only knows how to *launch*
> the harness, not its internals.
