"""Pre-flight checks — verify everything works before starting trials.

Three layers:

1. **Built-in (always run)** — image, container boot, model API, proxy→model chain.
   These are framework-level and never need user configuration.

2. **Framework-provided, user-enabled** — configured via ``preflight:`` in project.yml.
   The framework knows HOW to check these; the user just says WHAT to check.

   Example::

       preflight:
         # Check URLs are reachable (framework extracts from sample["target_url"])
         check_targets: true
         # Check internet is reachable via a proxy
         check_internet:
           proxy: http://127.0.0.1:7890
         # Arbitrary shell commands
         custom:
           - name: Redis is up
             command: redis-cli ping
             level: warn

3. **User custom commands** — under ``preflight.custom:``, arbitrary shell with
   ``{{image}}``, ``{{model_base_url}}`` etc. template variables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cage.contracts.sample_keys import SAMPLE_MAX_ROUNDS_KEY
from cage.contracts.execution import resolve_max_rounds
from cage.contracts.style import should_color, style

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when a pre-flight check fails."""


@dataclass
class PreflightResult:
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    details: list[str] = field(default_factory=list)
    plain_output: bool = False
    color_enabled: bool = False
    stream: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "details": list(self.details),
        }

    def plain(self, message: str) -> None:
        print(message, file=self.stream or sys.stderr)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run_preflight(
    config: Any,
    samples: list[dict[str, Any]] | None = None,
) -> PreflightResult:
    """Run pre-flight checks. Raises PreflightError on critical failure."""
    terminal_ui = bool(getattr(getattr(config, "logging", None), "terminal_ui", False))
    result = PreflightResult(
        plain_output=terminal_ui,
        color_enabled=should_color(sys.stderr),
        stream=sys.stderr,
    )
    pf_cfg = config.metadata.get("preflight", {}) or {}
    # Backward compat: if preflight is a list, treat it as custom commands
    if isinstance(pf_cfg, list):
        pf_cfg = {"custom": pf_cfg}

    if result.plain_output:
        result.plain(
            style(
                "Pre-flight checks",
                "cyan",
                "bold",
                enabled=result.color_enabled,
            )
        )
    else:
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CHECKS")
        logger.info("=" * 60)

    # ---- Layer 1: Built-in (always) ----
    for agent in config.agents:
        _check_image(agent.effective_image, result)
        _check_container_boot(
            agent.effective_image,
            config.execution.agent_network_mode,
            result,
        )
        if not _agent_may_call_model(agent, samples, config):
            _warn(
                "Proxy → Model chain" if config.proxy.enabled else "Model API",
                result,
                "skipped (max_rounds=0): this run will not issue model-call rounds.",
            )
            continue
        # Multi-source (case 2): every source may serve a trial, so validate
        # each one — not just the representative. Single-source agents iterate a
        # one-element list, i.e. the unchanged behavior.
        for pf_model in (getattr(agent, "model_sources", None) or [agent.model]):
            if config.proxy.enabled:
                # The agent will reach the model via the in-container proxy, so
                # the host-level direct check is irrelevant (and may fail if the
                # host has no direct egress, which is the normal case for our
                # corporate-proxy setup). The proxy chain check exercises the
                # real call path end-to-end.
                #
                # Subscription/OAuth mode (``auth_source`` set, no explicit
                # ``base_url``): the proxy chain CANNOT be validated by a raw
                # curl. Anthropic only accepts OAuth-subscription tokens from
                # the real Claude Code CLI (specific user-agent / device
                # fingerprint); a bare curl with ``Authorization: Bearer ...``
                # gets a fake ``rate_limit_error: Error`` envelope as an
                # anti-abuse decoy. The real call path runs through the CLI
                # inside the trial container, so leave validation to the live
                # trial and emit a warning here instead of a misleading FAIL.
                # Use ``cage agent debug --agent claude_code --model <id> --upstream-proxy <p>``
                # to validate end-to-end before launching a run.
                _sub_mode = (
                    bool(getattr(pf_model, "auth_source", ""))
                    and not pf_model.base_url
                )
                if _sub_mode:
                    _warn(
                        "Proxy → Model chain",
                        result,
                        "skipped (subscription/OAuth mode): Anthropic rejects raw "
                        "API calls with subscription tokens; use `cage agent debug "
                        f"--agent {agent.agent_type.name} --model {pf_model.id} "
                        f"--upstream-proxy {config.proxy.upstream_http_proxy or '<host:port>'}` "
                        "to validate end-to-end.",
                    )
                else:
                    _check_proxy_chain(
                        image=agent.effective_image,
                        model=pf_model,
                        network_mode=config.execution.agent_network_mode,
                        result=result,
                        upstream_http_proxy=config.proxy.upstream_http_proxy,
                    )
            else:
                _check_model_api(pf_model, result)

    # ---- Layer 1 (cont.): target images for benchmarks that bring up a stack ----
    # Build-based targets must be built BEFORE the run (``cage benchmark build``
    # or ``cage run --build``); launch never builds. Without this check the run
    # boots, then every trial fails one-by-one with ``target_unavailable`` once
    # its stack tries to come up — looks like a "completed" run with 0 LLM calls.
    if getattr(getattr(config, "target", None), "enabled", False):
        _check_target_images(config, samples, result)

    # ---- Layer 2: Framework-provided, user-enabled ----
    internet_cfg = pf_cfg.get("check_internet")
    if internet_cfg:
        proxy_url = internet_cfg if isinstance(internet_cfg, str) else internet_cfg.get("proxy", "")
        _check_internet(proxy_url, result)

    # ---- Layer 3: User custom commands ----
    custom_checks = pf_cfg.get("custom", [])
    if custom_checks:
        template_vars = _build_template_vars(config)
        for check in custom_checks:
            if isinstance(check, str):
                check = {"name": check, "command": check}
            _run_user_check(
                name=check.get("name", "Custom"),
                command=_render_template(check.get("command", ""), template_vars),
                level=check.get("level", "error"),
                result=result,
            )

    # ---- Summary ----
    if not result.plain_output:
        logger.info("-" * 60)
    if result.failed > 0:
        if result.plain_output:
            result.plain(
                style(
                    f"Failed: {result.passed} passed, "
                    f"{result.failed} failed, {result.warnings} warnings",
                    "red",
                    "bold",
                    enabled=result.color_enabled,
                )
            )
        else:
            logger.error(
                "PRE-FLIGHT: %d passed, %d FAILED, %d warnings",
                result.passed, result.failed, result.warnings,
            )
        fail_details = [d for d in result.details if d.startswith("FAIL")]
        raise PreflightError(
            f"Pre-flight failed ({result.failed} errors):\n"
            + "\n".join(f"  - {d}" for d in fail_details)
        )

    if result.plain_output:
        summary_color = "yellow" if result.warnings else "green"
        result.plain(
            style(
                f"Ready: {result.passed} passed, {result.warnings} warnings",
                summary_color,
                "bold",
                enabled=result.color_enabled,
            )
        )
        result.plain("")
    else:
        logger.info(
            "PRE-FLIGHT: %d passed, %d warnings — all clear",
            result.passed, result.warnings,
        )
        logger.info("=" * 60)
    return result


# ------------------------------------------------------------------
# Round-budget helpers
# ------------------------------------------------------------------

def _agent_may_call_model(
    agent: Any,
    samples: list[dict[str, Any]] | None,
    config: Any,
) -> bool:
    planned_samples = samples or [{}]
    return any(
        _effective_preflight_max_rounds(agent, sample, config) != 0
        for sample in planned_samples
    )


def _effective_preflight_max_rounds(
    agent: Any,
    sample: dict[str, Any],
    config: Any,
) -> int:
    return resolve_max_rounds(
        getattr(agent, "max_rounds", -1),
        getattr(getattr(config, "execution", None), "max_rounds", -1),
        sample.get(SAMPLE_MAX_ROUNDS_KEY) if isinstance(sample, dict) else None,
    )


# ------------------------------------------------------------------
# Result helpers
# ------------------------------------------------------------------

def _pass(name: str, result: PreflightResult, msg: str = ""):
    result.passed += 1
    result.details.append(f"PASS: {name}" + (f" ({msg})" if msg else ""))
    if result.plain_output:
        status = style("PASS", "green", "bold", enabled=result.color_enabled)
        detail = f"{name}{f' - {msg}' if msg else ''}"
        result.plain(f"  {status} {detail}")
    else:
        logger.info("  [PASS] %s%s", name, f" — {msg}" if msg else "")


def _fail(name: str, result: PreflightResult, msg: str):
    result.failed += 1
    result.details.append(f"FAIL: {name} — {msg}")
    if result.plain_output:
        status = style("FAIL", "red", "bold", enabled=result.color_enabled)
        result.plain(f"  {status} {name} - {msg}")
    else:
        logger.error("  [FAIL] %s — %s", name, msg)


def _warn(name: str, result: PreflightResult, msg: str):
    result.warnings += 1
    result.details.append(f"WARN: {name} — {msg}")
    if result.plain_output:
        status = style("WARN", "yellow", "bold", enabled=result.color_enabled)
        result.plain(f"  {status} {name} - {msg}")
    else:
        logger.warning("  [WARN] %s — %s", name, msg)


def _run_cmd(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, text=True, capture_output=True,
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


# ------------------------------------------------------------------
# Layer 1: Built-in checks
# ------------------------------------------------------------------

def _check_image(image: str, result: PreflightResult):
    name = f"Docker image [{image}]"
    rc, _, _ = _run_cmd(["docker", "image", "inspect", image])
    if rc == 0:
        _pass(name, result)
    else:
        _fail(name, result, "Not found. Run: cage agent build")


def _check_container_boot(image: str, network_mode: str | None, result: PreflightResult):
    name = "Container boot"
    import uuid
    cname = f"cage-preflight-{uuid.uuid4().hex[:8]}"
    # Defensive: remove any leftover container with this name
    _run_cmd(["docker", "rm", "-f", cname], timeout=5.0)
    cmd = ["docker", "run", "-d", "--name", cname]
    if network_mode:
        cmd.extend(["--network", network_mode])
    cmd.extend([image, "sleep", "10"])

    rc, _, err = _run_cmd(cmd, timeout=120.0)
    if rc != 0:
        _fail(name, result, f"Failed to start: {err[:200]}")
        return

    rc2, _, _ = _run_cmd(["docker", "exec", cname, "id", "agent"], timeout=5.0)
    if rc2 != 0:
        _fail(name, result, "User 'agent' not found in image")
    else:
        _pass(name, result, f"network={network_mode}")

    _run_cmd(["docker", "rm", "-f", cname], timeout=10.0)


def _check_model_api(model: Any, result: PreflightResult):
    name = f"Model API [{model.id}]"
    url = f"{str(model.base_url).rstrip('/')}/models"
    rc, out, err = _run_cmd([
        "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
        "--connect-timeout", "5", "--max-time", "10",
        url, "-H", f"Authorization: Bearer {model.api_key}",
    ])
    code = out.strip()
    if rc == 0 and code.startswith("2"):
        _pass(name, result, f"HTTP {code}")
    elif rc == 0:
        _warn(name, result, f"HTTP {code}")
    else:
        _fail(name, result, f"Unreachable: {err[:100]}")


def _validate_proxy_response(body: str, *, curl_exit: int) -> tuple[bool, str]:
    """Validate the proxy → model chain response body.

    Handles two response shapes:

    * **Plain JSON** — typical for ``/v1/chat/completions`` with
      ``stream: false``. FAIL on any top-level ``error`` key (covers both
      ``container_proxy``'s ``{"error": {"type": "proxy_error", ...}}``
      emitted on DNS/upstream failure, and upstream-returned envelopes
      like ``{"error": {"code": 400, ...}}``).

    * **SSE stream** — typical for ``/v1/responses`` against codex-relay
      endpoints that mandate streaming. Parse ``data:`` events; FAIL on
      any event that is itself an error envelope; PASS if at least one
      non-error event arrived.
    """
    if curl_exit != 0:
        return False, f"curl exited {curl_exit} (no response from proxy)"
    if not body or len(body) < 10:
        return False, f"Empty or truncated response ({len(body)} bytes)"

    if body.lstrip().startswith("data:") or "\ndata:" in body:
        events: list[Any] = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if isinstance(event, dict) and (
                "error" in event or event.get("type") == "error"
            ):
                err = event.get("error", event)
                msg = (
                    (err.get("message") or err.get("type") or str(err))
                    if isinstance(err, dict) else str(err)
                )
                return False, f"Upstream SSE error: {str(msg)[:200]}"
        if not events:
            return False, f"SSE stream had no parseable events: {body[:200]}"
        return True, ""

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return False, f"Non-JSON response: {body[:200]}"
    if isinstance(parsed, dict) and "error" in parsed:
        err = parsed["error"]
        if isinstance(err, dict):
            err_type = err.get("type") or ""
            err_msg = err.get("message") or ""
            msg = f"{err_type}: {err_msg}".strip(": ") or str(err)
        else:
            msg = str(err)
        return False, f"Proxy/upstream error: {str(msg)[:200]}"
    return True, ""


def _check_proxy_chain(
    image: str,
    model: Any,
    network_mode: str | None,
    result: PreflightResult,
    *,
    upstream_http_proxy: str = "",
):
    """Temp container → start proxy → send request → verify response."""
    name = "Proxy → Model chain"
    import uuid
    cname = f"cage-preflight-proxy-{uuid.uuid4().hex[:8]}"
    # Defensive: remove any leftover container with this name
    _run_cmd(["docker", "rm", "-f", cname], timeout=5.0)
    cmd = ["docker", "run", "-d", "--name", cname]
    if network_mode:
        cmd.extend(["--network", network_mode])
    cmd.extend(["--add-host", "host.docker.internal:host-gateway"])
    # TTL must outlive this check's own request budget: setup + up to two model
    # pings (``--max-time`` each, see below). A slow reasoning model (e.g. GLM
    # spends ~35s emitting reasoning_content even for a 5-token "hi") otherwise
    # exhausts a short TTL, the container exits, and the in-flight ``docker exec
    # curl`` is SIGKILLed -> a confusing "curl exited 137" instead of a clean
    # timeout. Removed in ``finally`` regardless.
    cmd.extend([image, "sleep", "240"])

    rc, _, err = _run_cmd(cmd, timeout=120.0)
    if rc != 0:
        _fail(name, result, f"Container failed: {err[:200]}")
        return

    try:
        # Copy proxy + write config
        proxy_src = str(Path(__file__).parent / "proxy" / "sidecar.py")
        _run_cmd(["docker", "cp", proxy_src, f"{cname}:/opt/cage-proxy/container_proxy.py"],
                 timeout=30.0)

        # Subscription/OAuth fallback: the agent launch path uses the same rule —
        # when ``auth_source`` is set and no explicit ``base_url``, the
        # proxy forwards to ``api.anthropic.com``. Mirror that here or the
        # preflight curl ends up with an empty upstream URL.
        _sub_mode = bool(getattr(model, "auth_source", "")) and not model.base_url
        upstream_base_url = (
            "https://api.anthropic.com" if _sub_mode else str(model.base_url)
        )
        cfg = json.dumps({
            "upstream_base_url": upstream_base_url,
            "upstream_api_key": str(model.api_key),
            "upstream_protocol": getattr(model, "protocol", "openai"),
            "http_proxy": upstream_http_proxy,
            "extra_headers": {
                str(k): str(v)
                for k, v in (getattr(model, "extra_headers", {}) or {}).items()
            },
            "trial_id": "preflight",
        })

        # Write config via a temp file on host then docker cp (avoids shell escaping issues)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(cfg)
        tmp.close()
        _run_cmd(["docker", "exec", cname, "mkdir", "-p", "/tmp/pl"])
        _run_cmd(["docker", "exec", cname, "chmod", "777", "/tmp/pl"])
        _run_cmd(["docker", "cp", tmp.name, f"{cname}:/tmp/pc.json"])
        _run_cmd(["docker", "exec", cname, "chmod", "644", "/tmp/pc.json"])
        os.unlink(tmp.name)

        # Pick a free port (same mechanism as the real proxy)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _sock:
            _sock.bind(("127.0.0.1", 0))
            proxy_port = _sock.getsockname()[1]

        # Start proxy as agent
        _run_cmd(["docker", "exec", "-d", "-u", "agent", cname, "bash", "-c",
                   f"python3 /opt/cage-proxy/container_proxy.py --port {proxy_port} "
                   f"--config /tmp/pc.json --log-dir /tmp/pl "
                   f">/tmp/pl/stdout.log 2>/tmp/pl/stderr.log"])
        time.sleep(3)

        # Health check
        rc_h, out_h, _ = _run_cmd(
            ["docker", "exec", cname, "curl", "-s", f"http://localhost:{proxy_port}/healthz"],
            timeout=5.0,
        )
        if "ok" not in out_h:
            _, stderr_log, _ = _run_cmd(
                ["docker", "exec", cname, "cat", "/tmp/pl/stderr.log"], timeout=5.0,
            )
            _fail(name, result, f"Proxy failed to start on port {proxy_port}: {stderr_log[:300]}")
            return

        # Send test request — use the right endpoint for the protocol.
        # Pass the real upstream api_key just like a real agent CLI would; the
        # proxy is intentionally dumb about credentials, the harness is what
        # authenticates to the upstream.
        protocol = getattr(model, "protocol", "openai")
        upstream_key = str(model.api_key)
        # Strip Claude-Code's ``[effort]`` decorator (e.g. ``claude-opus-4-7
        # [xhigh]``). The CLI consumes it internally — translates to a
        # ``thinking`` config in the request body — and forwards the bare
        # model id to the API. Raw preflight curl has no such translation,
        # so we send the bare id ourselves.
        _model_bare = re.sub(r"\[[^\]]+\]$", "", str(model.model))
        if protocol == "anthropic":
            # Anthropic: /v1/messages
            payload = json.dumps({
                "model": _model_bare, "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}],
            })
            rc_t, out_t, _ = _run_cmd(
                ["docker", "exec", cname, "curl", "-s", "--max-time", "90",
                 "-X", "POST", f"http://localhost:{proxy_port}/v1/messages",
                 "-H", "Content-Type: application/json",
                 "-H", f"x-api-key: {upstream_key}",
                 "-H", "anthropic-version: 2023-06-01",
                 "-d", payload],
                timeout=100.0,
            )
        elif protocol == "google":
            # Google Gemini: /v1beta/models/{model}:generateContent. Auth is
            # the ``x-goog-api-key`` header — Google 401s if a Bearer token is
            # attached alongside it, so the proxy must not inject one (it skips
            # injection when x-goog-api-key is present).
            payload = json.dumps({
                "contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            })
            rc_t, out_t, _ = _run_cmd(
                ["docker", "exec", cname, "curl", "-s", "--max-time", "90",
                 "-X", "POST",
                 f"http://localhost:{proxy_port}/v1beta/models/{_model_bare}:generateContent",
                 "-H", "Content-Type: application/json",
                 "-H", f"x-goog-api-key: {upstream_key}",
                 "-d", payload],
                timeout=100.0,
            )
        else:
            # OpenAI: /v1/chat/completions (standard) or /v1/responses
            payload = json.dumps({
                "model": _model_bare, "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}],
            })
            rc_t, out_t, _ = _run_cmd(
                ["docker", "exec", cname, "curl", "-s", "--max-time", "90",
                 "-X", "POST", f"http://localhost:{proxy_port}/v1/chat/completions",
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: Bearer {upstream_key}",
                 "-d", payload],
                timeout=100.0,
            )
            # If chat/completions fails (e.g. responses-only API), try
            # /v1/responses. Codex-relay endpoints mandate ``stream: true``
            # and reject ``stream: false`` outright, so send a streaming
            # request and let _validate_proxy_response parse the SSE.
            if rc_t != 0 or not out_t or '"error"' in out_t:
                payload_r = json.dumps({
                    "model": str(model.model),
                    "instructions": "hi",
                    "input": [{"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": "hi"}]}],
                    "store": False, "stream": True,
                })
                rc_t, out_t, _ = _run_cmd(
                    ["docker", "exec", cname, "curl", "-s", "--max-time", "90",
                     "-X", "POST", f"http://localhost:{proxy_port}/v1/responses",
                     "-H", "Content-Type: application/json",
                     "-H", f"Authorization: Bearer {upstream_key}",
                     "-d", payload_r],
                    timeout=100.0,
                )

        ok, fail_reason = _validate_proxy_response(out_t, curl_exit=rc_t)
        if ok:
            _pass(name, result, f"Response received ({len(out_t)} bytes)")
        else:
            _fail(name, result, fail_reason)

    finally:
        _run_cmd(["docker", "rm", "-f", cname], timeout=10.0)


def _image_exists_local(image: str) -> bool:
    rc, _, _ = _run_cmd(["docker", "image", "inspect", image], timeout=10.0)
    return rc == 0


def _check_target_images(config: Any, samples: list[dict[str, Any]] | None, result: PreflightResult):
    """Fail fast when a benchmark target needs build-images that aren't built.

    Mirrors the target server's launch-time ``_missing_build_images`` guard, but
    runs host-side BEFORE any trial starts: resolve each planned sample's target
    challenge, read its compose stack (no containers started), and flag every
    service that declares ``build:`` whose ``image:`` tag is absent locally.
    Pull-only services (``image:`` without ``build:``) are left to launch.
    """
    name = "Target images"
    try:
        from cage.target.adapters.roots import normalize_benchmark_sources
        from cage.target.adapters.source_config import build_default_registry
        from cage.target.compose_files import load_compose_stack
        from cage.target.provisioning import (
            discover_benchmark_root,
            target_challenge_id,
        )
    except Exception as exc:  # noqa: BLE001
        _warn(name, result, f"skipped (target modules unavailable: {exc})")
        return

    # Unique target challenges across the planned samples (same resolver the
    # trial runner uses: ``challenge_id`` else sample ``id``).
    chal_ids: list[str] = []
    seen: set[str] = set()
    for s in samples or []:
        cid = target_challenge_id(s) if isinstance(s, dict) else ""
        if cid and cid not in seen:
            seen.add(cid)
            chal_ids.append(cid)
    if not chal_ids:
        return

    bench_root = discover_benchmark_root(getattr(config, "benchmark", None))
    if bench_root is None:
        _warn(name, result, "skipped (no benchmark_root to resolve target challenges)")
        return

    try:
        registry = build_default_registry()
        sources = normalize_benchmark_sources(
            [{"adapter_kind": "challenge_json", "root": str(bench_root)}]
        )
        challenges = registry.discover_all(sources)
    except Exception as exc:  # noqa: BLE001
        _warn(name, result, f"skipped (could not load challenges: {exc})")
        return

    missing_by_chal: dict[str, list[str]] = {}
    verified_chals: set[str] = set()
    for cid in chal_ids:
        meta = challenges.get(cid)
        if not meta:
            continue  # not a server-launched target (e.g. static / external)
        try:
            adapter = registry.get(meta["adapter_kind"])
            spec = adapter.build_launch_spec(meta)
        except Exception:  # noqa: BLE001
            continue
        if getattr(spec, "mode", "") == "static" or not spec.compose_files:
            continue
        try:
            stack = load_compose_stack(spec.compose_files)
        except Exception:  # noqa: BLE001
            continue
        for _svc, cfg in (stack.get("services") or {}).items():
            if not isinstance(cfg, dict) or "build" not in cfg:
                continue
            image = str(cfg.get("image") or "").strip()
            # Unresolved ``${VAR}`` tags can't be checked statically — skip
            # rather than risk a false failure.
            if not image or "${" in image:
                continue
            # A challenge counts as "verified" only once it declares a real
            # build-image we could inspect — pull-only stacks (nothing to build)
            # contribute no PASS row and no failure.
            verified_chals.add(cid)
            if not _image_exists_local(image):
                missing_by_chal.setdefault(cid, [])
                if image not in missing_by_chal[cid]:
                    missing_by_chal[cid].append(image)

    if not verified_chals:
        return  # nothing build-based to verify in this run
    if not missing_by_chal:
        _pass(name, result, f"{len(verified_chals)} target(s) built")
        return

    all_imgs = sorted({img for imgs in missing_by_chal.values() for img in imgs})
    shown = ", ".join(all_imgs[:6]) + (f" (+{len(all_imgs) - 6} more)" if len(all_imgs) > 6 else "")
    bench_id = str((getattr(config, "metadata", {}) or {}).get("benchmark_id") or "").strip()
    sample_chal = next(iter(missing_by_chal))
    build_target = f"cage benchmark build {bench_id}".rstrip()
    _fail(
        name,
        result,
        f"{len(all_imgs)} image(s) missing for {len(missing_by_chal)} target(s): {shown}. "
        f"Build them first (e.g. `{build_target} --sample {sample_chal} ...`), "
        f"or pass `--build` to build at launch.",
    )


# ------------------------------------------------------------------
# Layer 2: Framework-provided, user-enabled
# ------------------------------------------------------------------


def _check_internet(proxy_url: str, result: PreflightResult):
    """Check internet reachability, optionally via HTTP proxy."""
    name = "Internet access" + (f" (via {proxy_url})" if proxy_url else "")
    cmd = [
        "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
        "--connect-timeout", "5", "--max-time", "10",
    ]
    if proxy_url:
        cmd.extend(["-x", proxy_url])
    cmd.append("https://httpbin.org/status/200")

    rc, out, _ = _run_cmd(cmd)
    code = out.strip()
    if rc == 0 and code.startswith("2"):
        _pass(name, result)
    else:
        _warn(name, result, f"Not reachable (HTTP {code})")


# ------------------------------------------------------------------
# Layer 3: User custom commands
# ------------------------------------------------------------------

def _build_template_vars(config: Any) -> dict[str, str]:
    vs: dict[str, str] = {}
    if config.agents:
        a = config.agents[0]
        vs["image"] = a.effective_image
        vs["model_id"] = a.model.id
        vs["model_name"] = a.model.model
        vs["model_base_url"] = str(a.model.base_url)
        vs["model_api_key"] = str(a.model.api_key)
        vs["network_mode"] = config.execution.agent_network_mode or "bridge"
    return vs


def _render_template(command: str, vars: dict[str, str]) -> str:
    for k, v in vars.items():
        command = command.replace("{{" + k + "}}", v)
    return command


def _run_user_check(name: str, command: str, level: str, result: PreflightResult):
    try:
        r = subprocess.run(
            command, shell=True, text=True,
            capture_output=True, stdin=subprocess.DEVNULL, timeout=30.0,
        )
        if r.returncode == 0:
            _pass(name, result)
        else:
            msg = (r.stderr or r.stdout or "").strip()[:200] or f"exit {r.returncode}"
            (_warn if level == "warn" else _fail)(name, result, msg)
    except subprocess.TimeoutExpired:
        (_warn if level == "warn" else _fail)(name, result, "timeout (30s)")
    except Exception as e:
        (_warn if level == "warn" else _fail)(name, result, str(e)[:200])
