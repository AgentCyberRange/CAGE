"""Trace proxy — observation and intervention on LLM traffic.

Each trial gets its own proxy instance on a dedicated port.
The proxy:
  1. Receives Anthropic-format requests from installed agents
  2. Optionally modifies the system prompt (intervention)
  3. Translates to OpenAI format if the upstream model requires it
  4. Forwards to the upstream model
  5. Translates response back to Anthropic format
  6. Records original and modified requests for audit

Adapted from Snowl's trace_proxy.py with per-trial isolation.
"""

from __future__ import annotations

import httpx
import json
import logging
import shlex
import socket
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from cage.proxy.anthropic_openai_translate import (  # noqa: F401
    ProxyModifyRule,
    _TEXT_TOOL_CALL_RE,
    _XML_FUNCTION_RE,
    _XML_PARAMETER_RE,
    _apply_modify_rules,
    _apply_system_template,
    _build_openai_request,
    _coerce_xml_param_value,
    _extract_text_tool_calls,
    _loads_text_tool_call,
    _normalize_content,
    _normalize_text_tool_input,
    _parse_xml_tool_call,
    _translate_messages_anthropic_to_openai,
    _translate_response_openai_to_anthropic,
    _translate_tools_anthropic_to_openai,
)

logger = logging.getLogger(__name__)


__all__ = [
    "CONTAINER_PROXY_CONFIG_PATH",
    "CONTAINER_PROXY_LOG_DIR",
    "CONTAINER_PROXY_RUNTIME_DIR",
    "ContainerProxyInstance",
    "ProxyInstance",
    "ProxyInstanceConfig",
    "ProxyModifyRule",
    "ProxyRecorder",
    "_TEXT_TOOL_CALL_RE",
    "_XML_FUNCTION_RE",
    "_XML_PARAMETER_RE",
    "_apply_modify_rules",
    "_apply_system_template",
    "_build_openai_request",
    "_coerce_xml_param_value",
    "_estimate_input_tokens",
    "_extract_text_tool_calls",
    "_loads_text_tool_call",
    "_normalize_content",
    "_normalize_text_tool_input",
    "_parse_xml_tool_call",
    "_pick_free_local_port",
    "_translate_messages_anthropic_to_openai",
    "_translate_response_openai_to_anthropic",
    "_translate_tools_anthropic_to_openai",
    "logger",
    "start_container_proxy",
    "start_proxy_instance",
]


logger = logging.getLogger(__name__)


CONTAINER_PROXY_RUNTIME_DIR = "/run/cage-proxy"


CONTAINER_PROXY_CONFIG_PATH = f"{CONTAINER_PROXY_RUNTIME_DIR}/config.json"


CONTAINER_PROXY_LOG_DIR = "/var/lib/cage/proxy"


def _pick_free_local_port() -> int:
    """Reserve an ephemeral localhost port number for a soon-to-start local server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class ProxyInstanceConfig:
    """Configuration for a single proxy instance."""

    upstream_base_url: str
    upstream_api_key: str
    upstream_protocol: str  # "anthropic" | "openai"
    artifact_dir: Path
    trial_id: str
    modify_rules: list[ProxyModifyRule] = field(default_factory=list)
    system_template: str = ""  # Jinja2-style template: {{ system_raw }} → original system
    bind_host: str = "0.0.0.0"
    port: int = 0  # 0 = auto-assign
    request_timeout: float = 3600.0
    max_output_tokens_cap: int | None = None
    http_proxy: str = ""
    # Extra headers injected onto every upstream request, overriding any
    # same-named header the agent CLI sent (see ``ModelConfig.extra_headers``).
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Extra body fields merged into the translated OpenAI/vLLM request,
    # overriding same-named params the agent CLI sent (see
    # ``ModelConfig.upstream_extra_body`` — e.g. Qwen ``chat_template_kwargs``
    # / sampling). ``{}`` ⇒ nothing injected.
    upstream_extra_body: dict[str, Any] = field(default_factory=dict)
    container_log_dir: str = ""
    logs_mounted: bool = False
    max_requests: int = -1  # -1 = unlimited; 0 rejects before the first request
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    # Per-call upstream retry budget (transient HTTP 408/425/429/5xx and
    # provider-specific transient codes wrapped in 4xx — e.g. bigmodel
    # ``code: "1234"``). 0 disables retries.
    upstream_max_retries: int = 4


class ProxyRecorder:
    """Records proxy traffic to disk for a single trial."""

    def __init__(self, artifact_dir: Path, trial_id: str) -> None:
        self.artifact_dir = artifact_dir
        self.trial_id = trial_id
        self._lock = threading.Lock()
        self._counter = 0
        self._tool_call_counter = 0
        artifact_dir.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"req-{self._counter:04d}"

    def record(
        self,
        *,
        request_id: str,
        anthropic_request: dict[str, Any],
        openai_request: dict[str, Any] | None = None,
        upstream_response: dict[str, Any] | None = None,
        anthropic_response: dict[str, Any] | None = None,
        original_system: str = "",
        modified_system: str = "",
        status: str = "success",
        error: str = "",
    ) -> None:
        entry = {
            "request_id": request_id,
            "trial_id": self.trial_id,
            "ts_ms": int(time.time() * 1000),
            "status": status,
            "original_system": original_system,
            "modified_system": modified_system,
            "anthropic_request": anthropic_request,
            "openai_request": openai_request,
            "upstream_response": upstream_response,
            "anthropic_response": anthropic_response,
            "error": error,
        }
        self._append_jsonl(entry)

        # Detect tool_use blocks in the response and append to tool_calls.jsonl
        response = upstream_response or anthropic_response or {}
        self._count_tool_uses(response)

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with (self.artifact_dir / "proxy.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _count_tool_uses(self, response: dict[str, Any]) -> None:
        """Extract tool_use blocks from a response and append to tool_calls.jsonl."""
        content_blocks: list[dict[str, Any]] = []

        if isinstance(response.get("content"), list):
            content_blocks = response["content"]
        else:
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                for tc in (msg.get("tool_calls") or []):
                    func = tc.get("function", {})
                    content_blocks.append({
                        "type": "tool_use",
                        "name": func.get("name", ""),
                    })

        tool_uses = [
            b for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not tool_uses:
            return

        with self._lock:
            for block in tool_uses:
                self._tool_call_counter += 1
                entry = {
                    "trial_id": self.trial_id,
                    "tool_name": block.get("name", ""),
                    "call_index": self._tool_call_counter,
                    "ts_ms": int(time.time() * 1000),
                }
                line = json.dumps(entry, ensure_ascii=False)
                with (self.artifact_dir / "tool_calls.jsonl").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


def _estimate_input_tokens(body: dict[str, Any]) -> int:
    parts: list[str] = []
    system = _normalize_content(body.get("system", ""))
    if system:
        parts.append(system)
    for msg in body.get("messages", []) or []:
        parts.append(_normalize_content(msg.get("content", "")))
    text = "\n".join(p for p in parts if p)
    return max(1, (len(text) + 3) // 4) if text else 1


@dataclass
class ProxyInstance:
    """A running proxy instance bound to a specific trial (host-side)."""

    host: str
    port: int
    trial_id: str
    httpd: ThreadingHTTPServer
    recorder: ProxyRecorder
    _thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self.host == "0.0.0.0":
            return f"http://172.17.0.1:{self.port}"
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self._thread.start()
        logger.info("Proxy started: %s (trial=%s)", self.base_url, self.trial_id)

    def stop(self, *, artifact_dir: Path | None = None) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Proxy stopped (trial=%s)", self.trial_id)


@dataclass
class ContainerProxyInstance:
    """A proxy running inside a container, managed from the host."""

    container: Any  # Container instance
    port: int
    pid: str
    trial_id: str
    config_path: str  # path to config JSON inside container
    log_dir: str  # path to log dir inside container
    logs_mounted: bool = False

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def resource_metadata(self) -> dict[str, object]:
        """Return ResourceLedger-safe metadata for this live proxy process.

        The container proxy config file contains upstream endpoint details,
        API keys, and extra headers. This method intentionally records only
        operational identifiers that help inspect and cleanup understand the
        container-internal process: container name, local port, pid, runtime
        paths, and whether logs were host-mounted. It never reads or serializes
        the proxy config contents, upstream routing fields, traffic artifacts,
        or host artifact paths.
        """

        metadata: dict[str, object] = {
            "base_url": self.base_url,
            "config_path": self.config_path,
            "container_name": str(getattr(self.container, "name", "") or ""),
            "log_dir": self.log_dir,
            "logs_mounted": bool(self.logs_mounted),
            "pid": str(self.pid),
            "port": int(self.port),
            "trial_id": self.trial_id,
        }
        return dict(sorted(metadata.items()))

    def stop(self, *, artifact_dir: Path | None = None) -> None:
        """Stop the proxy process and collect logs."""
        if self.pid:
            self.container.kill_process(self.pid, signal="TERM")
            time.sleep(0.5)
            # Force kill if still running
            if self.container.is_process_running(self.pid):
                self.container.kill_process(self.pid, signal="KILL")

        # Collect logs only when they are not already backed by a host bind mount.
        if artifact_dir is not None and not self.logs_mounted:
            for filename in ("proxy.jsonl", "tool_calls.jsonl", "progress.json"):
                container_path = f"{self.log_dir}/{filename}"
                check = self.container.exec(
                    f"test -f {shlex.quote(container_path)}",
                    timeout=5.0,
                )
                if check.exit_code == 0:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    self.container.copy_from(container_path, str(artifact_dir / filename))

        logger.info("Container proxy stopped (trial=%s)", self.trial_id)


def start_container_proxy(
    container: Any,
    config: ProxyInstanceConfig,
    *,
    port: int = 0,
) -> ContainerProxyInstance:
    """Start a proxy inside the container for a single trial.

    Writes config JSON into the container, starts container_proxy.py
    as a background process, and waits for it to become healthy.
    """
    from cage.sandbox.containers import Container

    assert isinstance(container, Container)

    requested_port = port or config.port
    if requested_port == 0:
        requested_port = _pick_free_local_port()

    # Write config JSON into container
    proxy_config = {
        "upstream_base_url": str(config.upstream_base_url),
        "upstream_api_key": str(config.upstream_api_key),
        "upstream_protocol": str(config.upstream_protocol),
        "system_template": str(config.system_template),
        "max_output_tokens_cap": config.max_output_tokens_cap,
        "request_timeout": config.request_timeout,
        "http_proxy": config.http_proxy,
        "extra_headers": {str(k): str(v) for k, v in (config.extra_headers or {}).items()},
        "upstream_extra_body": dict(config.upstream_extra_body or {}),
        "trial_id": str(config.trial_id),
        "max_requests": config.max_requests,
        "max_input_tokens": config.max_input_tokens,
        "max_output_tokens": config.max_output_tokens,
        "max_cost": config.max_cost,
        "input_cost_per_1m": config.input_cost_per_1m,
        "output_cost_per_1m": config.output_cost_per_1m,
        "upstream_max_retries": int(config.upstream_max_retries),
    }
    config_json = json.dumps(proxy_config, ensure_ascii=False, indent=2)
    config_path = CONTAINER_PROXY_CONFIG_PATH
    log_dir = config.container_log_dir or CONTAINER_PROXY_LOG_DIR
    quoted_runtime_dir = shlex.quote(CONTAINER_PROXY_RUNTIME_DIR)
    quoted_config_path = shlex.quote(config_path)
    quoted_log_dir = shlex.quote(log_dir)

    container.exec("rm -f /opt/cage-proxy/container_proxy.py")
    container.copy_to(
        str(Path(__file__).with_name("sidecar.py")),
        "/opt/cage-proxy/container_proxy.py",
    )
    log_files = (
        "proxy.jsonl",
        "tool_calls.jsonl",
        "progress.json",
        "stdout.log",
        "stderr.log",
    )
    rm_targets = " ".join(shlex.quote(f"{log_dir}/{filename}") for filename in log_files)
    setup_cmd = (
        f"(mkdir -p {quoted_runtime_dir} {quoted_log_dir} && "
        f"chmod 0700 {quoted_runtime_dir}) && rm -f {rm_targets}"
    )
    if not config.logs_mounted:
        setup_cmd = f"{setup_cmd} && chown -R agent:agent {quoted_log_dir}"
    setup_result = container.exec(setup_cmd, timeout=5.0)
    if setup_result.exit_code != 0:
        raise RuntimeError(
            "Failed to prepare container proxy log dir "
            f"{log_dir}: {setup_result.stderr[:500] or setup_result.stdout[:500]}"
        )
    container.write_file(config_path, config_json)
    config_permissions_cmd = f"chmod 0600 {quoted_config_path}"
    if not config.logs_mounted:
        config_permissions_cmd = (
            f"chown -R agent:agent {quoted_runtime_dir} && {config_permissions_cmd}"
        )
    config_permissions = container.exec(config_permissions_cmd, timeout=5.0)
    if config_permissions.exit_code != 0:
        raise RuntimeError(
            "Failed to secure container proxy config "
            f"{config_path}: "
            f"{config_permissions.stderr[:500] or config_permissions.stdout[:500]}"
        )

    # Start proxy as background process
    stdout_log = f"{log_dir}/stdout.log"
    stderr_log = f"{log_dir}/stderr.log"
    cmd = (
        f"python3 /opt/cage-proxy/container_proxy.py --port {requested_port} "
        f"--config {quoted_config_path} --log-dir {quoted_log_dir} "
        f">{shlex.quote(stdout_log)} 2>{shlex.quote(stderr_log)}"
    )
    if config.logs_mounted:
        pid = container.exec_background(cmd)
    else:
        pid = container.exec_background(cmd, user="agent")

    # Health check: wait for proxy to become ready
    max_retries = 30
    for i in range(max_retries):
        time.sleep(0.2)
        check = container.exec(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{requested_port}/healthz",
            timeout=3.0,
        )
        if "200" in check.stdout:
            logger.info(
                "Container proxy ready on port %d (trial=%s)",
                requested_port,
                config.trial_id,
            )
            break
    else:
        raise RuntimeError(
            f"Container proxy failed to start on port {requested_port} after {max_retries} retries"
        )

    return ContainerProxyInstance(
        container=container,
        port=requested_port,
        pid=pid,
        trial_id=config.trial_id,
        config_path=config_path,
        log_dir=log_dir,
        logs_mounted=config.logs_mounted,
    )


def start_proxy_instance(config: ProxyInstanceConfig) -> ProxyInstance:
    """Start a proxy instance for a single trial."""
    recorder = ProxyRecorder(config.artifact_dir, config.trial_id)
    needs_translation = config.upstream_protocol == "openai"

    class Handler(BaseHTTPRequestHandler):
        server_version = "CageProxy/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path == "/healthz":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_json()
                route = urlsplit(self.path).path

                if route == "/v1/messages/count_tokens":
                    self._send_json(
                        HTTPStatus.OK,
                        {"input_tokens": _estimate_input_tokens(body)},
                    )
                    return

                if route != "/v1/messages":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                request_id = recorder.next_id()

                if needs_translation:
                    openai_req, orig_sys, mod_sys = _build_openai_request(
                        body,
                        modify_rules=list(config.modify_rules),
                        system_template=config.system_template,
                        max_output_tokens_cap=config.max_output_tokens_cap,
                        upstream_extra_body=config.upstream_extra_body,
                    )
                    upstream_resp = self._forward_openai(openai_req)
                    anthropic_resp = _translate_response_openai_to_anthropic(
                        request_id=request_id,
                        model=str(body.get("model") or ""),
                        response=upstream_resp,
                    )
                    recorder.record(
                        request_id=request_id,
                        anthropic_request=body,
                        openai_request=openai_req,
                        upstream_response=upstream_resp,
                        anthropic_response=anthropic_resp,
                        original_system=orig_sys,
                        modified_system=mod_sys,
                    )
                    self._send_json(HTTPStatus.OK, anthropic_resp)
                else:
                    # Upstream is Anthropic — forward as-is, just apply modify rules
                    system_content = _normalize_content(body.get("system", ""))
                    orig_sys, mod_sys = _apply_modify_rules(
                        system_content, list(config.modify_rules)
                    )
                    modified_body = dict(body)
                    if mod_sys != system_content:
                        modified_body["system"] = mod_sys
                    # Disable streaming — we need complete responses
                    modified_body.pop("stream", None)

                    upstream_resp = self._forward_anthropic(modified_body)
                    recorder.record(
                        request_id=request_id,
                        anthropic_request=body,
                        anthropic_response=upstream_resp,
                        original_system=orig_sys,
                        modified_system=mod_sys,
                    )
                    self._send_json(HTTPStatus.OK, upstream_resp)

            except Exception as exc:  # noqa: BLE001
                request_id = locals().get("request_id", recorder.next_id())
                body_safe = locals().get("body", {})
                if not isinstance(body_safe, dict):
                    body_safe = {}
                recorder.record(
                    request_id=request_id,
                    anthropic_request=body_safe,
                    status="error",
                    error=str(exc),
                )
                logger.error(
                    "proxy_request_error",
                    extra={
                        "request_id": request_id,
                        "error": str(exc),
                        "route": urlsplit(self.path).path,
                        "trial_id": config.trial_id,
                    },
                )
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {
                        "type": "error",
                        "error": {"type": "proxy_error", "message": str(exc)},
                    },
                )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def _forward_openai(self, payload: dict[str, Any]) -> dict[str, Any]:
            url = f"{config.upstream_base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if config.upstream_api_key:
                headers["Authorization"] = f"Bearer {config.upstream_api_key}"
            started = time.time()
            try:
                with httpx.Client(
                    timeout=config.request_timeout, trust_env=False
                ) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                elapsed_ms = int((time.time() - started) * 1000)
                logger.debug(
                    "proxy_forward_success",
                    extra={"upstream": "openai", "model": payload.get("model", ""), "duration_ms": elapsed_ms},
                )
                return data
            except httpx.TimeoutException:
                elapsed_ms = int((time.time() - started) * 1000)
                logger.error(
                    "proxy_forward_timeout",
                    extra={"upstream": "openai", "model": payload.get("model", ""), "duration_ms": elapsed_ms, "timeout": config.request_timeout},
                )
                raise
            except httpx.HTTPStatusError as exc:
                elapsed_ms = int((time.time() - started) * 1000)
                logger.error(
                    "proxy_forward_http_error",
                    extra={
                        "upstream": "openai",
                        "model": payload.get("model", ""),
                        "status_code": exc.response.status_code,
                        "duration_ms": elapsed_ms,
                    },
                )
                raise

        def _forward_anthropic(self, payload: dict[str, Any]) -> dict[str, Any]:
            url = f"{config.upstream_base_url.rstrip('/')}/v1/messages"
            headers = {"Content-Type": "application/json"}
            if config.upstream_api_key:
                headers["x-api-key"] = config.upstream_api_key
                headers["anthropic-version"] = "2023-06-01"
            started = time.time()
            try:
                with httpx.Client(
                    timeout=config.request_timeout, trust_env=False
                ) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                elapsed_ms = int((time.time() - started) * 1000)
                logger.debug(
                    "proxy_forward_success",
                    extra={"upstream": "anthropic", "model": payload.get("model", ""), "duration_ms": elapsed_ms},
                )
                return data
            except httpx.TimeoutException:
                elapsed_ms = int((time.time() - started) * 1000)
                logger.error(
                    "proxy_forward_timeout",
                    extra={"upstream": "anthropic", "model": payload.get("model", ""), "duration_ms": elapsed_ms, "timeout": config.request_timeout},
                )
                raise
            except httpx.HTTPStatusError as exc:
                elapsed_ms = int((time.time() - started) * 1000)
                logger.error(
                    "proxy_forward_http_error",
                    extra={
                        "upstream": "anthropic",
                        "model": payload.get("model", ""),
                        "status_code": exc.response.status_code,
                        "duration_ms": elapsed_ms,
                    },
                )
                raise

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    httpd = ThreadingHTTPServer((config.bind_host, config.port), Handler)
    actual_port = httpd.server_address[1]

    instance = ProxyInstance(
        host=str(httpd.server_address[0]),
        port=actual_port,
        trial_id=config.trial_id,
        httpd=httpd,
        recorder=recorder,
    )
    instance.start()
    return instance

