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

**Related docs:**
- [`docs/targets-check.md`](targets-check.md) describes the public build and
  smoke-run workflow for validating benchmark targets.

---

## TL;DR

```bash
# Internal-only runner:
python -m cage.target.serve --benchmark-root examples/agent_pentest_bench/datasets

# Expose to external clients (new):
python -m cage.target.serve \
  --benchmark-root examples/agent_pentest_bench/datasets \
  --external-token "$(openssl rand -hex 16)" \
  --host 0.0.0.0 \
  --port 8000

# External caller:
curl -H "Authorization: Bearer <token>" http://<server>:8000/challenges
curl -H "Authorization: Bearer <token>" \
     "http://<server>:8000/launch/SIYUCMS?target_scope=per_agent"
# → response.entry_urls = [{name:"web", url:"http://<server>:<port>", ...}]
```

---

## Two audiences, one server

The server now classifies every request as either `internal` or `external`
and tailors port bindings + response shape accordingly.

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

Existing endpoint. Same query params as before
(`force_recreate`, `parallel_mode`, `target_scope`, `cage_run_id`). The
audience is **always derived from the request** — there is no `?audience=`
override. Two changes for callers:

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

### `DELETE /launch/{chal_id}?run_id=<id>`

Unchanged. Requires the same audience auth as launch.

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

- ✅ External clients reaching scoring sidecars (`evaluator`, …). Pinned to
  `127.0.0.1` on the host.
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

- **No per-token scoping yet.** A bearer token is binary: either you have it
  (external) or you don't. No per-tester quotas, no expiry, no audit trail
  beyond uvicorn's access log.
- **No challenge-level allow-list.** Anyone with the token can launch any
  challenge. If you need to restrict a tester to specific benchmarks, run a
  dedicated `python -m cage.target.serve` instance per scope with its own benchmark-root.
- **Static port allocation per process.** Ports are assigned from the
  FastAPI process's free-port pool. If you restart the server, the port a
  tester was using will be re-allocated — they'll need a fresh launch.
- **No HTTPS termination built in.** `python -m cage.target.serve` speaks plain HTTP. For
  public exposure put a TLS-terminating reverse proxy in front and read
  the reverse-proxy caveat in the security section.
