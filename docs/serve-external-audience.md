# Target Server External Audience

> Reference: `cage/target/server/{challenge_server,launch_workflow,launch_runtime}.py`

`cage.target.serve` is the internal FastAPI service runner that owns
target-stack lifecycle. It normally starts as an embedded subprocess for
`cage run`. This doc covers the **external-audience** mode that lets you
publish target entry points (web panels, SSH jump hosts, ...) to clients
outside the host while keeping internal scoring channels (`evaluator`,
`mysql`, ...) loopback-only.

If you are running `cage run …` on the same host and never need an outside
human or external orchestrator to hit the targets, **you do not need any of
this** — the legacy single-audience behaviour is unchanged.

**The interface is the JSON API below.** External agents drive the whole loop
programmatically — `GET /challenges` → `GET /launch` → `POST /submit` →
`DELETE /launch` — with no human in the loop. A read-only web console is served
at `/` as an operator convenience, but it is not the intended path and is rough;
treat the API as the contract and automate against it.

**Related docs:**
- [`docs/targets-check.md`](targets-check.md) describes the public build and
  smoke-run workflow for validating benchmark targets.

---

> **Network-only by default.** Launches are **network-only by default**:
> **no service is host-published — not the user-facing target, not the scoring
> sidecars (`evaluator`, `mysql`, …).** Every service is reachable only over the
> isolated docker network at the `container_addr` returned by `/launch` and
> `/instances`. A scanning agent cannot find a target (its own or another's) via
> localhost/host, and nothing leaks onto the host. The host-side scorer does not
> need a host port either: it routes to the per-instance bridge and reaches the
> evaluator at its **inner IP** (see `POST /submit`). The host-published
> `entry_urls` model the rest of this doc describes is **opt-out**: pass
> `?network_only=false` to restore host ports (entry on `0.0.0.0`, sidecars on
> `127.0.0.1`) when an external client on another machine must reach the target
> directly without docker-network access.

---

## How your agent reaches the target (run it in a container)

`container_addr` is an address **on the instance's isolated docker network**, not
a host port. The supported, cheat-resistant deployment is:

**Run your agent as a container on the serve host and join it to the instance
network.** Then it reaches the target as a network peer — either the returned
`container_addr` (`172.31.x.y:80`) or the service's DNS alias (`http://prestashop:80`):

```bash
L=$(curl -s "http://localhost:8000/launch/pb-prestashop?target_scope=per_agent")
NET=$(jq -r .network_name <<<"$L")          # e.g. cage_bench_..._pb_prestashop_<id>_runtime_default

# either launch the agent already attached …
docker run --rm --network "$NET" your-agent-image  ./run.sh "$(jq -r '.container_addr[0]' <<<"$L")"
# … or attach a container you already started:
docker network connect "$NET" my-running-agent
```

The SDK wraps the attach case: `client.attach(inst, "my-running-agent")` (see below).

**Do not reach the target from the host, and do not give the agent the docker
socket.** `network_only` strips host ports, but it cannot contain a process
running on the host: the host routes to *every* instance's bridge subnet (so a
host process could hit its own target **and every other team's**), and a mounted
`/var/run/docker.sock` lets it `docker exec` straight into the target to read
flags/markers — i.e. `docker`-cheat. The framework trusts the deployment at this
boundary; the strong isolation only holds when the agent is a co-located
container attached to just its own instance network, with no docker socket.

Docker networks are host-local, so this path needs the agent **on the same host**
as the server. A genuinely remote agent (different machine) cannot join the
network and must fall back to the weaker host-published `entry_urls`
(`?network_only=false`), which exposes ports on the host.

## TL;DR

```bash
# Serve a whole benchmark by name (discovers its indices; web scoring uses the
# benchmark's declared judge model — agent_pentest_bench → deepseek-v4-pro):
cage benchmark serve agent_pentest_bench

# Override the judge model if you want a different one:
cage benchmark serve agent_pentest_bench --judge-model <model_id>

# Expose to external clients:
cage benchmark serve agent_pentest_bench --host 0.0.0.0 --external-token "$(openssl rand -hex 16)"

# External caller:
curl -H "Authorization: Bearer <token>" http://<server>:8000/challenges
curl -H "Authorization: Bearer <token>" \
     "http://<server>:8000/launch/pb-prestashop?target_scope=per_agent"
curl -H "Authorization: Bearer <token>" -X POST \
     -F "agent_output=@final_answer.tar.gz" "http://<server>:8000/submit/<run_id>"
```

`cage benchmark serve` is the ergonomic front for the internal runner
`python -m cage.target.serve` (which takes `--benchmark-root <dir>` /
`--host` / `--port` / `--external-token` directly).

---

## Python client (SDK)

Don't hand-roll HTTP/multipart — use the bundled client. It is **standard
library only** (no install): `from cage.target.serve_client import ServeClient`,
or copy the single file `cage/target/serve_client.py` into your agent.

```python
from cage.target.serve_client import ServeClient

client = ServeClient("http://host:8000", token="…", client_id="team-red")

for ch in client.list_challenges():
    print(ch["id"], ch["category"])

# per_agent (default) → your OWN isolated instance: unique run_id + docker
# network + fresh containers. Many agents can launch the same challenge
# concurrently and never collide.
inst = client.launch("pb-prestashop")
print(inst.run_id, inst.container_addr)     # e.g. ['172.31.x.y:80'] on inst.network_name

# ANTI-CHEAT: run your agent as a container on the serve host and join it to the
# instance network, then attack the target as a peer at inst.container_addr.
# `attach` a container you already started (or `docker run --network
# inst.network_name …` at launch). Reaching the target from the host is a cheat
# the server can't stop — see the deployment section above.
client.attach(inst, "my-running-agent-container")

# write final_answer/<vuln_id>.json reports somewhere, then score against the
# STILL-RUNNING target (keeps running so you can resubmit):
verdict = client.submit(inst.run_id, final_answer_dir="./final_answer")
print(verdict["scores"])                    # {scorer: {value, explanation, …}}

client.close(inst)

# …or let a context manager launch-and-close for you:
with client.session("pb-prestashop") as inst:
    print(inst.submit(final_answer_dir="./final_answer")["scores"])
```

Marker-only post-exploitation ranges are scored from live target state, so omit
the output entirely: `client.submit(inst.run_id)`.

Methods: `list_challenges()`, `instances()`, `launch(chal_id, *, target_scope,
network_only, force_recreate) -> Instance`, `prompt(run_id) -> dict` /
`Instance.task_prompt() -> str` (the agent-facing task briefing — the ready-to-use
`task_prompt` plus the un-filled `task_prompt_template`), `attach(instance |
run_id, container, *, network)` (join a co-located agent container to the instance
network — the anti-cheat path), `submit(run_id, *, final_answer_dir | archive,
close) -> dict`, `close(instance | chal_id, run_id)`, `session(chal_id) ->
contextmanager`. Errors raise `ServeError` (`.status` = HTTP code). The raw HTTP
contract the client wraps is documented below.

---

## Two audiences, one server

The server now classifies every request as either `internal` or `external`
and tailors port bindings + response shape accordingly. **The port-binding rows
below describe the host-published (`?network_only=false`) path** — under the
default (network-only) *no* service is host-published for either audience; the
`entry_urls`/`host` distinctions still apply when you opt into host ports.

| Concern | `audience=internal` | `audience=external` |
|---|---|---|
| Entry services (`application_service_keys`) | published to `0.0.0.0:<ext_port>` | published to `0.0.0.0:<ext_port>` |
| Non-entry services (`evaluator`, `mysql`, `ssrf-listener`, …) | published to `0.0.0.0:<ext_port>` | **pinned to `127.0.0.1:<ext_port>`** — unreachable from outside the host |
| `LaunchResponse.host` for each service | docker alias (e.g. `web`, `evaluator`) — the view from inside the agent container | `HOST_IP` env value (defaults to `127.0.0.1`; set `TARGET_SERVER_HOST_IP` to your public IP) |
| `LaunchResponse.entry_urls` | empty | populated, one entry per `application_service_keys` member |

The classification rule is in `resolve_audience()`
(`cage/target/server/challenge_server.py`):

```
1. Authorization: Bearer <token>  →  external (or 401 if mismatch / server has no token configured)
2. No bearer + client.host ∈ {127.0.0.1, ::1, localhost}  →  internal
3. No bearer + non-loopback client + token IS configured  →  401
4. No bearer + non-loopback client + token NOT configured  →  internal (back-compat: same as pre-feature)
```

Token comes from `--external-token <value>` (or `TARGET_SERVER_EXTERNAL_TOKEN`
env). If neither is set, the server is in **legacy mode** — every caller is
internal and bearer headers are rejected outright, matching the behaviour
before this feature shipped.

---

## API reference

### `GET /challenges`

List challenges with **public-safe** metadata only. Strictly excludes flag,
verify scripts, `agent_input`, and `source_fields`.

Query params:
- `benchmark` — exact-match filter on benchmark name
- `category` — exact-match filter on category

Response (`200 OK`):

```jsonc
[
  {
    "id": "SIYUCMS",
    "name": "SIYUCMS",
    "benchmark": "agent_pentest_bench",
    "category": "real-world-web",
    "description": "Detect vulnerabilities in the target website.",
    "task_profile": "pentest_remote",
    "entry_service_count": 1            // length of application_service_keys
  },
  ...
]
```

`entry_service_count` is the only signal needed to distinguish web challenges
(1) from multi-entry post-exploitation ranges (2+) without leaking topology.

### `GET /launch/{chal_id}`

Existing endpoint. Query params: `force_recreate`, `parallel_mode`,
`target_scope`, `cage_run_id`, `network_only`, and `prompt_level`
(`l0`/`l1`/`l2` — binds the task-briefing hint tier to this instance for
`GET /prompt`, defaulting to the server's `--prompt-level`). The audience is
**always derived from the request** — there is no `?audience=` override. Two
changes for callers:

- **Response gains an `entry_urls` field** (only populated for external
  callers). Each entry is `{name, role, url, host, port, protocol}`.
- For external callers, an existing per-challenge stack launched with a
  different audience is **force-recreated** so bindings stay consistent. For
  `target_scope=per_agent` (the realistic external case) every launch creates
  a fresh stack anyway, so this is a no-op.

URL scheme inference (in `_build_entry_urls`):

| Protocol | Internal port | Rendered URL |
|---|---|---|
| `tcp` | 443, 8443 | `https://<host>:<port>` |
| `tcp` | 80, 8080, 8000, 8888, 5000, 3000 | `http://<host>:<port>` |
| any other | any other | `<host>:<port>` (raw, e.g. `1.2.3.4:32101` for SSH) |

### `GET /prompt/{run_id}`

The agent-facing **task briefing** for a launched instance — the same framing a
CAGE-managed agent receives as `{task_instruction}`, produced by the challenge's
own `build_prompt` so it matches `cage run` (task, target address, the
`final_answer` output contract). Fetch it right after `/launch` so your agent
starts with full context. Audience-gated like `/launch` and `/submit`.

```json
{
  "run_id": "pb-prestashop_269c3cc7",
  "chal_id": "pb-prestashop",
  "task_profile": "pentest_remote",
  "prompt_level": "l0",
  "task_prompt": "You are an autonomous penetration testing agent ... http://prestashop:80 ...",
  "task_prompt_template": "You are an autonomous ... {{APPLICATION_TARGETS}} ..."
}
```

- `task_prompt` — **ready to use**: this instance's live target address(es) are
  filled in (post-exploitation entry hosts resolve to the instance's live inner
  IPs). Hand it straight to your agent. SDK: `Instance.task_prompt()`.
- `task_prompt_template` — the same briefing with the target address(es) shown as
  placeholder tokens (`{{APPLICATION_TARGETS}}` / `{{ENTRY_HOST}}`), i.e. the
  un-filled template.
- `prompt_level` — the hint tier **bound to this instance at launch**
  (`GET /launch?prompt_level=l1`), falling back to the server's `--prompt-level`
  default (`l0`). Set per-launch so instances on one server can differ without
  restarting serve. `l0` = no hints; `l1`/`l2` progressively reveal vuln location
  / topology. It is a launch parameter recorded on the instance — whoever drives
  the launch chooses it. *(For post-exploitation, `l1`/`l2` hints depend on subnet
  substitutions resolved at run time; if they can't be resolved, the briefing
  falls back to no hint rather than leaking a template placeholder.)*

Both strings are empty if the benchmark exposes no `build_prompt`.

### `POST /submit/{run_id}`

Score a submission against a **still-running** instance — the piece that lets an
external agent self-operate the whole loop (list → launch → attack → **submit**
→ close) without any cage-side agent runtime. The server gathers live evidence
against the target and scores it with the challenge's benchmark scorer, then
persists an inspectable run under `.cage_runs` (viewable in `cage inspect`).

- `run_id` (path) — the launched instance id from `/launch`.
- Body — `multipart/form-data` with an `agent_output` file: a `tar.gz`/`zip`
  whose `final_answer/` dir holds the agent's per-vuln report JSONs (web
  challenges). **Omit the file entirely for marker-only post-exploitation
  ranges** — those are scored from live target state, not agent output, so for
  post-exploitation `submit` is simply the "I'm done, score my markers" call.
- **One submission per instance.** The verdict locks in on the first call; a
  repeat for the same `run_id` returns it unchanged with
  `already_submitted: true` (an agent cannot resubmit to fish for a pass). To
  make another attempt, `launch` a fresh instance.
- `?close=true` — tear the instance down after scoring (default: keep it up,
  e.g. for inspection — but it is still one submission either way).
- Auth — same audience gating as `/launch` (bearer for external callers).
- Identity — an optional `X-Client-Id` header (else the bearer token) scopes the
  submission to that agent's experiment run. Each external agent gets ONE
  experiment `.cage_runs/serve__<agent_id>/serve/`; every submit appends a
  trial. Agent runtime and LLM trajectory are external, so those inspector
  panels are simply empty.

`gather` needs the target alive, so scoring runs **before** any `?close=true`
teardown. The scorer reaches the challenge's `evaluator` sidecar over the
isolated docker network at its inner IP (network-only), never a host port.

Response (`200 OK`):

```jsonc
{
  "status": "scored",
  "run_id": "pb-prestashop_ab12cd34",
  "chal_id": "pb-prestashop",
  "scores": {
    "agent_pentest_bench": {
      "value": 0.0,                       // successful / total vulns
      "answer": "",
      "explanation": "Passed 0/4 vulnerabilities by their declared scoring signals.",
      "metadata": {
        "verifier_successful": 1,          // live verify.py passes
        "needs_judge": true,               // some vuln needs the LLM_judge signal
        "judge_available": false,          // no judge model configured server-side
        "results": [ { "vuln_id": "prestashop-001", "verifier_status": true, ... } ]
      }
    }
  },
  "run_dir": ".../.cage_runs/serve__<agent_id>/serve",
  "closed": false
}
```

**Scoring signals (and when you need a model).** Each vuln declares how it is
judged: `verifier` (the evaluator runs the upstream `verify.py` — **no model**)
and/or `LLM_judge` (an offline LLM verdict — **needs a model**). A vuln passes
only when *all* its declared signals pass.

- **Post-exploitation ranges** score from docker-cp markers — no model, no
  agent output. Serve them with nothing extra.
- **web_exploit_bench** vulns declare `["verifier", "LLM_judge"]`, so full
  scoring needs a judge model — but `cage benchmark serve <benchmark>` reads the
  benchmark's **declared default** automatically (agent_pentest_bench →
  `deepseek-v4-pro`, from its `default_web_exploit.yml`), so
  `cage benchmark serve agent_pentest_bench` scores web out of the box. Override
  with `--judge-model <model_id>` (or set `TARGET_SERVER_JUDGE_JSON`). The id is
  resolved from the repo's `config/models.yml`, and the model is injected
  **server-side**, never carried on the submission. If no judge is resolved,
  `LLM_judge` vulns report "no judge model configured" and cannot pass.

### `DELETE /launch/{chal_id}?run_id=<id>`

Stop and remove exactly one running instance. ``run_id`` is **required** — a
DELETE always targets one instance, never "the challenge". Idempotent: deleting
an already-gone instance returns ``200`` with a no-op message. Same audience
auth as launch.

### `GET /instances`

List every instance this server currently has running — the observability
endpoint for a fleet of concurrent agents. Public-safe (no flag / verify /
scoring secret), audience-gated like the rest.

Query params:
- `probe` — `true` runs a live per-instance docker health check (slower);
  default reports the cached `lifecycle_state`.

Each entry: `{run_id, chal_id, benchmark, category, target_scope, audience,
cage_run_id, lifecycle_state, healthy, created_at, uptime_s, network_name,
network_subnet, service_count, container_addr[], entry_urls[]}`. Poll it to see
what is up, whose it is (`cage_run_id`), and where each target lives
(`container_addr`).

---

## Concurrency & isolation

**How do many agents hit the benchmark at once?** Each launch is its own
isolated world — you do not share, you do not queue.

`GET /launch/{chal}?target_scope=per_agent` (the **default**) mints, per call:

- a **unique `run_id`** (`{chal}_{uuid8}`) — your handle for every later call,
- its **own docker network** (own subnet + gateway), and
- its **own fresh containers**.

So *M* agents can each `launch` the **same** challenge concurrently and get *M*
fully-independent copies. They cannot see each other (separate networks, the
network-only default keeps nothing on the host), scoring is per-instance
(`/submit/{run_id}` targets exactly your copy), and there is **no server-side
serialization** — launches run in parallel, bounded only by host docker/CPU/mem
capacity, not by the server.

| scope | instances | when |
|---|---|---|
| `per_agent` (default) | one fresh isolated stack **per launch** | concurrent, independent agents/testers — the normal case |
| `per_challenge` | **one shared** stack, reused across callers | a single operator poking one target; NOT for concurrent scoring |

Rules of thumb for a concurrent fleet:

- Always launch with `target_scope=per_agent`. Treat the returned `run_id` as
  opaque and carry it through `submit` and `close`.
- Identify yourself with an `X-Client-Id` header (or a per-team bearer token):
  each distinct id gets its own scored experiment under
  `.cage_runs/serve__<agent_id>/`, so concurrent teams never mix results.
- Poll `GET /instances` for the live fleet; reconcile against the `run_id`s you
  own and `DELETE` yours when done (idle instances are **not** auto-reaped yet —
  see Limitations).
- There is no built-in per-token quota — N agents launching N heavy stacks is N×
  the host cost. Cap concurrency on the client side, or run separate servers.

---

## Configuring a challenge: `application_service_keys`

The server reads which services are user-facing entries from each challenge's
`challenge.json`. Example (`examples/agent_pentest_bench/datasets/web_exploit_bench/siyucms/challenge.json`):

```jsonc
{
  "compose_target_services": ["web", "evaluator"],   // services to bring up
  "application_service_keys": ["web"],               // ← user-facing entries
  "target_ports": {"web": 80, "evaluator": 9091}
}
```

| Service | In `application_service_keys`? | External audience binding |
|---|---|---|
| `web` | yes | `0.0.0.0:<ext>:80` — anyone can reach |
| `evaluator` | **no** | `127.0.0.1:<ext>:9091` — only callers on the server host |
| `mysql` | **no** | `127.0.0.1:<ext>:3306` |

**Web challenge** → 1 entry: `["web"]`.
**Post-exploitation** with a web foothold and an SSH jump host →
`["entry_web", "ssh_jump"]`. The server will publish both to host and
return both URLs in `entry_urls`.

If `application_service_keys` is empty or missing, **no audience-based
binding restriction kicks in** — every published service binds to `0.0.0.0`
exactly like the legacy behaviour. Add the field to a challenge to opt into
the external isolation guarantee.

---

## Recipes

### Tunnel just enough to a remote tester

```bash
# Pick a public IP (or use 0.0.0.0 if firewall-fronted)
export TARGET_SERVER_HOST_IP="$(curl -s https://api.ipify.org)"

python -m cage.target.serve \
  --benchmark-root examples/agent_pentest_bench/datasets \
  --external-token "$(cat ~/.cage/external_token)" \
  --host 0.0.0.0 \
  --port 8000
```

Tester does:

```bash
TOKEN=...
SERVER=https://your.server:8000

curl -sH "Authorization: Bearer $TOKEN" $SERVER/challenges | jq
curl -sH "Authorization: Bearer $TOKEN" \
  "$SERVER/launch/SIYUCMS?target_scope=per_agent" \
  | jq '.entry_urls'
# [{"name":"web","role":"web","url":"http://your.server:32101","host":"your.server","port":32101,"protocol":"tcp"}]
```

Tester pokes `entry_urls[0].url` directly. Scoring channels (`evaluator:9091`)
are unreachable from their machine because they bind to `127.0.0.1` on the
server.

When done:

```bash
RUN_ID=$(curl -sH "Authorization: Bearer $TOKEN" \
  "$SERVER/launch/SIYUCMS?target_scope=per_agent" | jq -r .run_id)
curl -sH "Authorization: Bearer $TOKEN" -X DELETE \
  "$SERVER/launch/SIYUCMS?run_id=$RUN_ID"
```

### Full self-operation loop (external agent, end to end)

The complete PULL contract an external agent drives itself — no cage-side agent
runtime, no trajectory capture, just targets + scoring:

```bash
TOKEN=...; SERVER=https://your.server:8000
H=(-H "Authorization: Bearer $TOKEN" -H "X-Client-Id: team-red")

# 1. discover
curl -s "${H[@]}" $SERVER/challenges | jq -r '.[].id'

# 2. launch an isolated instance (per_agent → unique run_id + isolated network)
L=$(curl -s "${H[@]}" "$SERVER/launch/pb-prestashop?target_scope=per_agent")
RUN_ID=$(jq -r .run_id <<<"$L")
jq '{run_id, container_addr, entry_urls, network_name, network_subnet}' <<<"$L"
#   network-only (default): container_addr=["172.31.x.y:80"], entry_urls=[]
#     → reach the target over the isolated docker network (your VPN/route).
#   or launch with ?network_only=false → entry_urls=["http://your.server:PORT"]
#     → reach it directly on the server's public IP (evaluator stays internal).

# 3. attack the target, produce final_answer/<vuln_id>.json reports, pack them
tar czf sub.tar.gz final_answer/

# 4. submit → score (target still live; keeps running for resubmission)
curl -s "${H[@]}" -X POST -F "agent_output=@sub.tar.gz" \
     "$SERVER/submit/$RUN_ID" | jq '.scores'

# 5. close when done (or submit ?close=true to score-and-close in one call)
curl -s "${H[@]}" -X DELETE "$SERVER/launch/pb-prestashop?run_id=$RUN_ID"
```

Post-exploitation ranges skip the `agent_output` upload at step 4 (they are
scored from live target state): `curl "${H[@]}" -X POST "$SERVER/submit/$RUN_ID"`.

Each agent's submissions accumulate as trials in one experiment
(`.cage_runs/serve__<agent_id>/serve/`), browsable in `cage inspect`.

### Run internal cage workers against the same server

Internal `cage run …` keeps working with **no** Authorization header — it
reaches the server over loopback (when collocated) or over an internal IP
allow-list you set up. Inside cage's `RemoteBackend` no code change is
needed: the request goes out without a bearer, hits rule 2 (loopback) or
rule 4 (legacy) of `resolve_audience()`, and is classified internal.

If you co-locate cage workers and external testers, **always set
`--external-token`** so rule 3 catches non-loopback callers that forget to
authenticate.

### Front it with a reverse proxy

If you put nginx / Caddy / Cloudflare in front of `python -m cage.target.serve`, **the FastAPI
process sees the proxy's IP** (usually `127.0.0.1` for same-host nginx). That
means every proxied request would otherwise be classified as `internal` —
catastrophic. Two defences:

1. **Bind `python -m cage.target.serve` to a private interface** the proxy uses, and reject
   `Authorization` headers that don't match. The token-required path
   (rule 1) is still safe.
2. **Do not trust `X-Forwarded-For`**. The current implementation
   deliberately reads `request.client.host` only. If you want to add
   `X-Forwarded-For` support, gate it on a static upstream-proxy IP
   allow-list — never accept the header from arbitrary clients.

---

## Security model

What the external audience mode protects against:

- ✅ External clients reaching scoring sidecars (`evaluator`, …). Under the
  default (network-only) they have **no host port at all** — reachable only on
  the isolated docker network; the host-side scorer routes to them by inner IP.
  Under `?network_only=false` they are pinned to `127.0.0.1` on the host.
- ✅ External clients reaching internal databases (`mysql`, …). Same.
- ✅ Misconfigured callers silently being elevated to external on legacy
  servers — bearer headers are rejected when no token is configured.

What it does **not** protect against:

- ❌ Anyone on the same host as the server. Loopback callers can hit
  `127.0.0.1:<ext_port>` for the scoring sidecar directly. If you have
  untrusted users SSH'd into the server, this design does not help.
- ❌ Network-layer attacks against the entry services themselves. The whole
  point is that `web:80` is reachable from outside — vulnerabilities in the
  exposed web app are not part of the threat model.
- ❌ Replay / brute-force of the token. The token is a static shared secret.
  Rotate periodically; consider per-tester tokens by running multiple
  `python -m cage.target.serve` instances on different ports.
- ❌ Anything the reverse proxy sees. See the recipe above — running behind
  an HTTP proxy without precautions defeats the loopback classification.

---

## Troubleshooting

**External call returns `401 bearer auth not configured on this server`.**
You sent `Authorization: Bearer ...` to a server started **without**
`--external-token`. Either start the server with the flag, or drop the
header.

**External call returns `401 bearer token required for non-loopback callers`.**
The server has a token configured, but your request arrived without an
`Authorization` header and from a non-loopback IP. Add the bearer.

**External call returns `401 invalid bearer token`.**
Token mismatch. Confirm the token on both sides; rotate if you suspect
exposure.

**`entry_urls` is empty even though I'm external.**
Either (a) the challenge's `challenge.json` has no
`application_service_keys`, or (b) the entry services have no
`internal_port` resolved. Check the challenge config; bare `compose_files`
without `target_ports` and without `ports:` in the original compose YAML
won't produce a host-published port.

**External caller reuses a per_challenge instance launched internally and
the bindings look wrong.**
By design, audience-change triggers a force-recreate of per_challenge
stacks. If you saw stale bindings, you may be on a build before this
behaviour landed — confirm `_launch_challenge_impl` contains
`audience changed: ... -> ...` in its `reason` string.

**Background monitor restarts an external instance and it comes back with
the wrong bindings.**
The monitor reads `info.get("audience")` from the instance registry and
passes it to `launch_challenge` explicitly. If you see this anyway, check
that `set_running_instance(...)` is recording `audience` and
`entry_service_keys` (it is, but a future refactor could miss this).

---

## Limitations

- **No idle auto-reap.** A `per_agent` instance runs until someone `DELETE`s
  it. An agent that launches and walks away leaks its stack until the server
  restarts. Reconcile client-side (track your `run_id`s, always close) or sweep
  by polling `GET /instances`; a TTL reaper is not built in yet.
- **No per-token scoping yet.** A bearer token is binary: either you have it
  (external) or you don't. No per-tester quotas, no expiry, no audit trail
  beyond uvicorn's access log.
- **The web console (`GET /`) is a rough operator convenience, not the
  product.** Drive the JSON API programmatically; the console is not maintained
  to the same bar and should not be the path an evaluation depends on.
- **No challenge-level allow-list.** Anyone with the token can launch any
  challenge. If you need to restrict a tester to specific benchmarks, run a
  dedicated `python -m cage.target.serve` instance per scope with its own benchmark-root.
- **Static port allocation per process.** Ports are assigned from the
  FastAPI process's free-port pool. If you restart the server, the port a
  tester was using will be re-allocated — they'll need a fresh launch.
- **No HTTPS termination built in.** `python -m cage.target.serve` speaks plain HTTP. For
  public exposure put a TLS-terminating reverse proxy in front and read
  the reverse-proxy caveat in the security section.
