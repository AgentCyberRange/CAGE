"""Manual target launcher for debugging benchmark services.

KNOWN LAYER-1 LEAK (tracked): the ``_is_wordpress_target`` / ``_dify_*`` /
``_is_prestashop_target`` / ``_fix_*_public_url`` / ``_ensure_dify_setup`` helpers
hardcode TARGET-APPLICATION knowledge (wordpress/dify/prestashop) into this Layer-1
debug tool. They are woven into ``_wait_for_application_ready``'s readiness state
machine, so the correct fix is a generic ``Benchmark.debug_target_fixups`` hook with
the app-specific logic relocated to a shared Layer-2 module. Deferred as its own
focused task — it is unverifiable here (no live app containers, the relevant
example dataset absent). See docs/plans/2026-06-09-god-file-split-design.md. This is a manual
debug tool off the trial hot path, and these are app names (softer than the
benchmark-name invariant, which already holds).
"""
from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from cage.benchmarks import sample_id_matches
from cage.config.experiment import resolve

if TYPE_CHECKING:
    # Type-only: ExperimentRun appears only in signatures here. Keeping it out
    # of the runtime imports breaks the target↔experiment.engine cycle.
    from cage.experiment.engine.run_context import ExperimentRun

logger = logging.getLogger(__name__)

Echo = Callable[[str], None]

DIFY_ADMIN_EMAIL = "admin@example.com"
DIFY_ADMIN_PASSWORD = "Admin123!"


def _print_flush(message: str) -> None:
    print(message, flush=True)


class TargetDebugError(RuntimeError):
    """User-facing target-debug failure."""


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    return slug or "target"


def _default_run_id(sample_id: str) -> str:
    # Keep this reasonably short because it also becomes a docker label value.
    return f"debug-{_safe_slug(sample_id)[:32]}-{secrets.token_hex(4)}"


def _detect_public_host() -> str:
    configured = (
        os.getenv("TARGET_SERVER_HOST_IP")
        or os.getenv("CAGE_PUBLIC_HOST")
        or os.getenv("HOST_IP")
        or ""
    ).strip()
    if configured:
        return configured

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No packet is sent for UDP connect; this just asks the kernel
            # which source address would be used for non-loopback traffic.
            sock.connect(("8.8.8.8", 80))
            detected = str(sock.getsockname()[0])
            if detected:
                return detected
    except OSError:
        pass
    return "127.0.0.1"


def _load_config_and_sample(
    project_file: Path,
    sample_id: str,
) -> tuple[ExperimentRun, dict[str, Any], list[str]]:
    challenge_client_logger = logging.getLogger("ChallengeClient")
    prev_level = challenge_client_logger.level
    challenge_client_logger.setLevel(logging.WARNING)
    try:
        run = resolve(project_file)
        benchmark = run.benchmark
        if hasattr(benchmark, "setup"):
            benchmark.setup()
        samples = list(benchmark.iter_samples_limited(None))
    finally:
        challenge_client_logger.setLevel(prev_level)

    sample_ids = [str(sample.get("id") or sample.get("challenge_id") or "") for sample in samples]
    for sample in samples:
        if sample_id_matches(sample, [sample_id]):
            return run, sample, sample_ids
    raise TargetDebugError(
        f"sample not found: {sample_id}. Available samples: {', '.join(sorted(sample_ids))}"
    )


def _spawn_server(
    *,
    run_id: str,
    benchmark_root: Path | None,
    log_path: Path,
    extra_env: dict[str, str],
):
    from cage.target.provisioning import spawn_embedded_target_server

    return spawn_embedded_target_server(
        run_id=run_id,
        benchmark_root=benchmark_root,
        log_path=log_path,
        extra_env=extra_env,
    )


def _benchmark_root(run: ExperimentRun) -> Path | None:
    from cage.target.provisioning import discover_benchmark_root

    return discover_benchmark_root(run.benchmark)


def _runtime_args(run: ExperimentRun, sample: dict[str, Any]) -> dict[str, str]:
    from cage.target.provisioning import target_runtime_args

    args = target_runtime_args(run, sample)
    args["target_scope"] = "per_agent"
    return args


def _parse_http_error(exc: urllib.error.HTTPError) -> str:
    raw = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        raw = ""
    if raw:
        try:
            parsed = json.loads(raw)
            detail = parsed.get("detail", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(detail, str):
                return detail
            return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))[:1200]
        except json.JSONDecodeError:
            return raw[:1200]
    return str(exc)


def _request_json(
    method: str,
    url: str,
    *,
    token: str = "",
    timeout: float | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if method.upper() == "DELETE" and exc.code == 404:
            return {}
        raise TargetDebugError(f"HTTP {exc.code}: {_parse_http_error(exc)}") from exc
    except urllib.error.URLError as exc:
        raise TargetDebugError(str(exc)) from exc
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TargetDebugError(f"non-JSON response from target_server: {body[:400]}") from exc
    if not isinstance(parsed, dict):
        raise TargetDebugError(f"unexpected target_server response: {body[:400]}")
    return parsed


def _infer_url(host: str, external_port: int, internal_port: int | None, protocol: str) -> str:
    proto = (protocol or "tcp").lower()
    if proto in {"http", "https"}:
        return f"{proto}://{host}:{external_port}"
    if proto == "tcp" and internal_port in {443, 8443}:
        return f"https://{host}:{external_port}"
    if proto == "tcp" and internal_port in {80, 8080, 8000, 8888, 5000, 3000}:
        return f"http://{host}:{external_port}"
    return f"{host}:{external_port}"


def _entry_urls(target_data: dict[str, Any], public_host: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for item in target_data.get("entry_urls") or []:
        if isinstance(item, dict) and item.get("url"):
            role = str(item.get("role") or item.get("name") or "entry")
            entries.append((role, str(item["url"])))
    if entries:
        return entries

    for svc in target_data.get("services") or []:
        if not isinstance(svc, dict) or svc.get("external_port") is None:
            continue
        try:
            external_port = int(svc["external_port"])
            internal_port = int(svc["internal_port"]) if svc.get("internal_port") else None
        except (TypeError, ValueError):
            continue
        role = str(svc.get("service_name") or "entry")
        protocol = str(svc.get("protocol") or "tcp")
        entries.append((role, _infer_url(public_host, external_port, internal_port, protocol)))
    return entries


def _first_http_entry_url(
    target_data: dict[str, Any],
    public_host: str,
    service_name: str | None = None,
) -> str:
    for role, url in _entry_urls(target_data, public_host):
        if service_name is not None and role != service_name:
            continue
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def _join_origin_path(origin: str, path: str) -> str:
    return urllib.parse.urljoin(f"{origin.rstrip('/')}/", path.lstrip("/"))


def _wordpress_debug_urls(
    sample_id: str,
    target_data: dict[str, Any],
    public_host: str,
) -> list[tuple[str, str]]:
    if not _is_wordpress_target(sample_id, target_data):
        return []
    origin = _first_http_entry_url(target_data, public_host, "wordpress")
    if not origin:
        origin = _first_http_entry_url(target_data, public_host)
    if not origin:
        return []
    return [
        ("homepage", origin),
        ("login", _join_origin_path(origin, "wp-login.php")),
        ("admin dashboard", _join_origin_path(origin, "wp-admin/")),
        ("ARMember directory", _join_origin_path(origin, "?pagename=armember-directory")),
        (
            "Everest contact form",
            _join_origin_path(origin, "?pagename=webexploitbench-everest-contact"),
        ),
        ("REST API index", _join_origin_path(origin, "index.php?rest_route=/")),
    ]


def _is_dify_target(sample_id: str, target_data: dict[str, Any]) -> bool:
    labels = [
        sample_id,
        str(target_data.get("chal_id") or ""),
        str(target_data.get("challenge_name") or ""),
        str(target_data.get("name") or ""),
    ]
    return any("dify" in label.lower() for label in labels)


def _dify_debug_urls(
    sample_id: str,
    target_data: dict[str, Any],
    public_host: str,
) -> list[tuple[str, str]]:
    if not _is_dify_target(sample_id, target_data):
        return []
    origin = _first_http_entry_url(target_data, public_host, "nginx")
    if not origin:
        origin = _first_http_entry_url(target_data, public_host)
    if not origin:
        return []
    return [
        ("homepage", origin),
        ("apps", _join_origin_path(origin, "apps")),
        ("sign in", _join_origin_path(origin, "signin")),
        ("tools", _join_origin_path(origin, "tools")),
        ("setup status", _join_origin_path(origin, "console/api/setup")),
        ("system features", _join_origin_path(origin, "console/api/system-features")),
    ]


def _agent_input_urls(sample: dict[str, Any]) -> dict[str, urllib.parse.SplitResult]:
    agent_input = sample.get("agent_input") if isinstance(sample, dict) else None
    if not isinstance(agent_input, dict):
        return {}
    text = str(agent_input.get("application_targets") or "")
    urls: dict[str, urllib.parse.SplitResult] = {}
    for match in re.finditer(r"https?://[^\s,]+", text):
        parsed = urllib.parse.urlsplit(match.group(0).strip())
        if parsed.hostname:
            urls[parsed.hostname] = parsed
    return urls


def _apply_agent_input_entry_urls(
    target_data: dict[str, Any],
    sample: dict[str, Any],
    public_host: str,
) -> None:
    """Prefer the benchmark's agent-facing scheme over generic port guesses."""
    by_host = _agent_input_urls(sample)
    if not by_host:
        return

    rewritten: list[dict[str, Any]] = []
    for svc in target_data.get("services") or []:
        if not isinstance(svc, dict):
            continue
        service_name = str(svc.get("service_name") or "")
        parsed = by_host.get(service_name)
        if parsed is None or svc.get("external_port") is None:
            continue
        port = int(svc["external_port"])
        path = parsed.path or ""
        if parsed.query:
            path = f"{path}?{parsed.query}"
        if parsed.fragment:
            path = f"{path}#{parsed.fragment}"
        rewritten.append({
            "name": service_name,
            "role": service_name,
            "url": f"{parsed.scheme}://{public_host}:{port}{path}",
            "host": public_host,
            "port": port,
            "protocol": parsed.scheme,
        })

    if rewritten:
        target_data["entry_urls"] = rewritten


def _format_service_lines(target_data: dict[str, Any], public_host: str) -> list[str]:
    services = target_data.get("services") or []
    entry_url_by_role = {
        str(item.get("role") or item.get("name") or ""): str(item.get("url") or "")
        for item in (target_data.get("entry_urls") or [])
        if isinstance(item, dict)
    }
    entry_roles = {
        role for role, url in entry_url_by_role.items() if url
    }
    has_entry_urls = bool(entry_roles)

    lines: list[str] = []
    for svc in services:
        if not isinstance(svc, dict):
            continue
        external_port = svc.get("external_port")
        if external_port is None:
            continue
        service_name = str(svc.get("service_name") or "?")
        protocol = str(svc.get("protocol") or "tcp")
        internal_port = svc.get("internal_port")
        try:
            external_port_int = int(external_port)
        except (TypeError, ValueError):
            continue
        try:
            internal_port_int = int(internal_port) if internal_port is not None else None
        except (TypeError, ValueError):
            internal_port_int = None

        if has_entry_urls and service_name not in entry_roles:
            bind = "127.0.0.1"
        else:
            bind = "0.0.0.0"
        url_host = "127.0.0.1" if bind == "127.0.0.1" else public_host
        url = entry_url_by_role.get(service_name) or _infer_url(
            url_host,
            external_port_int,
            internal_port_int,
            protocol,
        )
        target = f"{internal_port_int}/{protocol}" if internal_port_int else protocol
        lines.append(
            f"  {service_name}: {bind}:{external_port_int} -> {target}  url={url}"
        )
    return lines


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def _default_port_for_scheme(scheme: str) -> int | None:
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def _is_bad_entry_redirect(base_url: str, location: str) -> bool:
    if not location:
        return False
    base = urllib.parse.urlsplit(base_url)
    dest = urllib.parse.urlsplit(urllib.parse.urljoin(base_url, location))
    dest_host = (dest.hostname or "").lower()
    base_port = base.port or _default_port_for_scheme(base.scheme)
    dest_port = dest.port or _default_port_for_scheme(dest.scheme)
    if dest_host in {"localhost", "127.0.0.1", "::1"}:
        return (
            (dest.hostname or "").lower() != (base.hostname or "").lower()
            or dest_port != base_port
        )
    if dest_host == (base.hostname or "").lower() and dest_port != base_port:
        return True
    return dest_host != (base.hostname or "").lower()


def _probe_entry_url(url: str, *, timeout: float = 5.0) -> tuple[bool, str, bool]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "cage-target-debug/0.1", "Accept": "*/*"},
        method="GET",
    )
    try:
        with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
            code = int(resp.getcode() or 0)
            if 200 <= code < 400 or code in {401, 403, 405}:
                return True, f"HTTP {code}", False
            return False, f"HTTP {code}", False
    except urllib.error.HTTPError as exc:
        code = int(exc.code or 0)
        location = exc.headers.get("Location", "")
        if 300 <= code < 400:
            if _is_bad_entry_redirect(url, location):
                return False, f"HTTP {code} redirects to {location}", True
            suffix = f" redirects to {location}" if location else ""
            return True, f"HTTP {code}{suffix}", False
        if code in {401, 403, 405}:
            return True, f"HTTP {code}", False
        return False, f"HTTP {code}", False
    except TimeoutError:
        return False, "timeout", False
    except OSError as exc:
        return False, str(exc), False


def _is_prestashop_target(sample_id: str, target_data: dict[str, Any]) -> bool:
    if "prestashop" in sample_id.lower():
        return True
    return any(
        isinstance(svc, dict) and str(svc.get("service_name") or "").lower() == "prestashop"
        for svc in target_data.get("services") or []
    )


def _fix_prestashop_public_url(
    *,
    sample_id: str,
    target_data: dict[str, Any],
    entry_url: str,
    echo: Echo,
) -> bool:
    if not _is_prestashop_target(sample_id, target_data):
        return False
    project_name = str(target_data.get("project_name") or "")
    if not project_name:
        return False
    parsed = urllib.parse.urlsplit(entry_url)
    public_domain = parsed.netloc
    if not public_domain:
        return False

    escaped_domain = public_domain.replace("'", "''")
    sql = (
        "UPDATE ps_shop_url "
        f"SET domain='{escaped_domain}', domain_ssl='{escaped_domain}' "
        "WHERE id_shop_url=1; "
        "UPDATE ps_configuration "
        f"SET value='{escaped_domain}' "
        "WHERE name IN ('PS_SHOP_DOMAIN','PS_SHOP_DOMAIN_SSL');"
    )
    proc = subprocess.run(
        [
            "docker", "compose", "-p", project_name, "exec", "-T",
            "db", "mysql", "-uroot", "-padmin", "prestashop", "-e", sql,
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if proc.returncode != 0:
        logger.debug("PrestaShop URL fix not ready: %s", (proc.stderr or "").strip())
        return False

    subprocess.run(
        [
            "docker", "compose", "-p", project_name, "exec", "-T",
            "prestashop", "sh", "-lc",
            "rm -rf /var/www/html/var/cache/prod/* /var/www/html/var/cache/dev/* || true",
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    echo(f"PrestaShop public URL configured: {public_domain}")
    return True


def _is_wordpress_target(sample_id: str, target_data: dict[str, Any]) -> bool:
    if "wordpress" in sample_id.lower():
        return True
    return any(
        isinstance(svc, dict) and str(svc.get("service_name") or "").lower() == "wordpress"
        for svc in target_data.get("services") or []
    )


def _fix_wordpress_public_url(
    *,
    sample_id: str,
    target_data: dict[str, Any],
    entry_url: str,
    echo: Echo,
) -> bool:
    if not _is_wordpress_target(sample_id, target_data):
        return False
    project_name = str(target_data.get("project_name") or "")
    if not project_name:
        return False
    parsed = urllib.parse.urlsplit(entry_url)
    public_url = f"{parsed.scheme}://{parsed.netloc}"
    if not parsed.scheme or not parsed.netloc:
        return False

    escaped_url = public_url.replace("'", "''")
    sql = (
        "UPDATE wp_options "
        f"SET option_value='{escaped_url}' "
        "WHERE option_name IN ('siteurl','home');"
    )
    proc = subprocess.run(
        [
            "docker", "compose", "-p", project_name, "exec", "-T",
            "db", "mysql", "-uwordpress", "-pwordpress", "wordpress", "-e", sql,
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if proc.returncode != 0:
        logger.debug("WordPress URL fix not ready: %s", (proc.stderr or "").strip())
        return False

    echo(f"WordPress public URL configured: {public_url}")
    return True


def _fix_public_url(
    *,
    sample_id: str,
    target_data: dict[str, Any],
    entry_url: str,
    echo: Echo,
) -> bool:
    return _fix_prestashop_public_url(
        sample_id=sample_id,
        target_data=target_data,
        entry_url=entry_url,
        echo=echo,
    ) or _fix_wordpress_public_url(
        sample_id=sample_id,
        target_data=target_data,
        entry_url=entry_url,
        echo=echo,
    )


def _compose_service_state(project_name: str, service_name: str) -> tuple[str, int | None] | None:
    if not project_name:
        return None
    ps = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--filter",
            f"label=com.docker.compose.service={service_name}",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if ps.returncode != 0:
        logger.debug("docker ps failed while checking %s: %s", service_name, ps.stderr.strip())
        return None
    container_ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not container_ids:
        return None

    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}} {{.State.ExitCode}}",
            container_ids[0],
        ],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if inspect.returncode != 0:
        logger.debug(
            "docker inspect failed while checking %s: %s",
            service_name,
            inspect.stderr.strip(),
        )
        return None
    parts = inspect.stdout.strip().split()
    if not parts:
        return None
    exit_code: int | None = None
    if len(parts) >= 2:
        try:
            exit_code = int(parts[1])
        except ValueError:
            exit_code = None
    return parts[0], exit_code


def _dify_request_json(
    opener: urllib.request.OpenerDirector,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any, str]:
    data = None
    headers = {"Accept": "application/json", "User-Agent": "cage-target-debug/0.1"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=10.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = int(resp.getcode() or 0)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        code = int(exc.code or 0)
    if not raw.strip():
        return code, {}, raw
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return code, parsed, raw


def _ensure_dify_setup(origin: str) -> tuple[bool, str]:
    if not origin:
        return False, "no Dify entry URL"
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    setup_url = _join_origin_path(origin, "console/api/setup")
    try:
        code, parsed, raw = _dify_request_json(opener, "GET", setup_url)
        if code != 200:
            return False, f"setup status HTTP {code}: {raw[:160]}"

        step = parsed.get("step") if isinstance(parsed, dict) else None
        if step == "not_started":
            init_url = _join_origin_path(origin, "console/api/init")
            _dify_request_json(
                opener,
                "POST",
                init_url,
                {"password": DIFY_ADMIN_PASSWORD},
            )
            code, parsed, raw = _dify_request_json(
                opener,
                "POST",
                setup_url,
                {
                    "email": DIFY_ADMIN_EMAIL,
                    "name": "Admin",
                    "password": DIFY_ADMIN_PASSWORD,
                },
            )
            if code not in {200, 201}:
                return False, f"setup HTTP {code}: {raw[:160]}"
            step = "finished"

        if step != "finished":
            return False, f"setup step {step!r}"

        login_url = _join_origin_path(origin, "console/api/login")
        code, _parsed, raw = _dify_request_json(
            opener,
            "POST",
            login_url,
            {
                "email": DIFY_ADMIN_EMAIL,
                "password": DIFY_ADMIN_PASSWORD,
                "remember_me": True,
            },
        )
        if code != 200:
            return False, f"login HTTP {code}: {raw[:160]}"
        return True, f"setup finished, login ok ({DIFY_ADMIN_EMAIL})"
    except (OSError, TimeoutError) as exc:
        return False, str(exc)


def _readiness_probe_urls(
    sample_id: str,
    target_data: dict[str, Any],
    public_host: str,
) -> list[tuple[str, str]]:
    probes: list[tuple[str, str]] = []
    if _is_wordpress_target(sample_id, target_data):
        urls = dict(_wordpress_debug_urls(sample_id, target_data, public_host))
        for label in ("login", "ARMember directory", "Everest contact form", "REST API index"):
            if urls.get(label):
                probes.append((f"wordpress {label}", urls[label]))
    if _is_dify_target(sample_id, target_data):
        urls = dict(_dify_debug_urls(sample_id, target_data, public_host))
        for label in ("apps", "sign in", "setup status", "system features"):
            if urls.get(label):
                probes.append((f"dify {label}", urls[label]))
    return probes


def _wait_for_application_ready(
    *,
    sample_id: str,
    target_data: dict[str, Any],
    public_host: str,
    readiness_timeout: float,
    echo: Echo,
) -> bool:
    entries = _entry_urls(target_data, public_host)
    if not entries:
        echo("")
        echo("Application status: no exposed entry URL to probe.")
        return False

    echo("")
    echo("Application status: setting up (not ready to use)")
    echo("Waiting for entry URL to answer from the host ...")

    deadline = time.monotonic() + max(0.0, readiness_timeout)
    last_note = ""
    last_print = 0.0
    public_url_fixed = False
    wordpress_init_done = not _is_wordpress_target(sample_id, target_data)
    dify_setup_done = not _is_dify_target(sample_id, target_data)
    project_name = str(target_data.get("project_name") or "")

    while True:
        if not wordpress_init_done:
            state = _compose_service_state(project_name, "wordpress-init")
            if state is not None:
                status, exit_code = state
                current = f"wordpress-init {status}"
                if status == "exited" and exit_code is not None:
                    current = f"{current} ({exit_code})"
                if status == "exited" and exit_code == 0:
                    echo("WordPress setup complete: wordpress-init exited 0")
                    wordpress_init_done = True
                    entry_url = _first_http_entry_url(target_data, public_host, "wordpress")
                    if entry_url and not public_url_fixed:
                        public_url_fixed = _fix_wordpress_public_url(
                            sample_id=sample_id,
                            target_data=target_data,
                            entry_url=entry_url,
                            echo=echo,
                        )
                elif status == "exited":
                    echo(f"Application status: setup failed ({current})")
                    return False
                else:
                    now = time.monotonic()
                    if current != last_note or now - last_print >= 15.0:
                        echo(f"  still setting up: {current}")
                        last_note = current
                        last_print = now
                    if time.monotonic() >= deadline:
                        echo("Application status: still setting up (not ready to use)")
                        echo(f"  last probe: {current}")
                        return False
                    time.sleep(3.0)
                    continue
            else:
                wordpress_init_done = True

        if not dify_setup_done:
            origin = _first_http_entry_url(target_data, public_host, "nginx")
            if not origin:
                origin = _first_http_entry_url(target_data, public_host)
            ready, note = _ensure_dify_setup(origin)
            current = f"dify setup {origin} -> {note}"
            if ready:
                echo(f"Dify setup complete: {note}")
                dify_setup_done = True
            else:
                now = time.monotonic()
                if current != last_note or now - last_print >= 15.0:
                    echo(f"  still setting up: {current}")
                    last_note = current
                    last_print = now
                if time.monotonic() >= deadline:
                    echo("Application status: still setting up (not ready to use)")
                    echo(f"  last probe: {current}")
                    return False
                time.sleep(3.0)
                continue

        all_ready = True
        ready_notes: list[str] = []
        probe_targets = entries + _readiness_probe_urls(sample_id, target_data, public_host)
        for role, url in probe_targets:
            ready, note, bad_redirect = _probe_entry_url(url)
            current = f"{role} {url} -> {note}"
            if ready:
                ready_notes.append(current)
                continue
            all_ready = False
            if bad_redirect and not public_url_fixed:
                public_url_fixed = _fix_public_url(
                    sample_id=sample_id,
                    target_data=target_data,
                    entry_url=url,
                    echo=echo,
                )
                if public_url_fixed:
                    last_note = ""
                    time.sleep(2.0)
                    break
            now = time.monotonic()
            if current != last_note or now - last_print >= 15.0:
                echo(f"  still setting up: {current}")
                last_note = current
                last_print = now
        if all_ready:
            joined = "; ".join(ready_notes)
            echo(f"Application status: ready to use ({joined})")
            return True
        else:
            if time.monotonic() >= deadline:
                echo("Application status: still setting up (not ready to use)")
                if last_note:
                    echo(f"  last probe: {last_note}")
                return False
            time.sleep(3.0)
            continue
        if time.monotonic() >= deadline:
            echo("Application status: still setting up (not ready to use)")
            if last_note:
                echo(f"  last probe: {last_note}")
            return False


def _launch_url(
    server_url: str,
    sample_id: str,
    *,
    cage_run_id: str,
    runtime_args: dict[str, str],
    force_recreate: bool,
) -> str:
    params: dict[str, str] = dict(runtime_args)
    params["cage_run_id"] = cage_run_id
    if force_recreate:
        params["force_recreate"] = "true"
    qs = urllib.parse.urlencode(params)
    return f"{server_url}/launch/{urllib.parse.quote(sample_id, safe='')}?{qs}"


def _delete_url(server_url: str, sample_id: str, target_run_id: str) -> str:
    qs = urllib.parse.urlencode({"run_id": target_run_id})
    return f"{server_url}/launch/{urllib.parse.quote(sample_id, safe='')}?{qs}"


def _print_launch_summary(
    *,
    echo: Echo,
    target_data: dict[str, Any],
    sample_id: str,
    cage_run_id: str,
    public_host: str,
    log_path: Path,
    server_pid: int | None,
) -> None:
    echo("")
    echo(f"Target launched: {sample_id}")
    echo(f"  status: {target_data.get('status') or ''}")
    echo(f"  cage_run_id: {cage_run_id}")
    echo(f"  target_run_id: {target_data.get('run_id') or ''}")
    echo(f"  project: {target_data.get('project_name') or ''}")
    if target_data.get("network_name"):
        echo(f"  network: {target_data.get('network_name')}")
    if server_pid:
        echo(f"  target_server_pid: {server_pid}")
    echo(f"  target_server_log: {log_path}")

    entry_urls = [
        item for item in (target_data.get("entry_urls") or [])
        if isinstance(item, dict) and item.get("url")
    ]
    if entry_urls:
        echo("")
        echo("Entry URLs:")
        for item in entry_urls:
            role = item.get("role") or item.get("name") or "entry"
            echo(f"  {role}: {item['url']}")
    else:
        fallback = [
            _infer_url(
                public_host,
                int(svc["external_port"]),
                int(svc["internal_port"]) if svc.get("internal_port") is not None else None,
                str(svc.get("protocol") or "tcp"),
            )
            for svc in (target_data.get("services") or [])
            if isinstance(svc, dict) and svc.get("external_port") is not None
        ]
        if fallback:
            echo("")
            echo("Candidate URLs:")
            for url in fallback:
                echo(f"  {url}")

    service_lines = _format_service_lines(target_data, public_host)
    if service_lines:
        echo("")
        echo("Exposed ports:")
        for line in service_lines:
            echo(line)
    else:
        echo("")
        echo("Exposed ports: none")

    useful_urls = _wordpress_debug_urls(sample_id, target_data, public_host)
    if useful_urls:
        echo("")
        echo("Useful URLs:")
        for label, url in useful_urls:
            echo(f"  {label}: {url}")
        echo("  credentials: admin/admin123, author/author123, victim/victim123")

    useful_urls = _dify_debug_urls(sample_id, target_data, public_host)
    if useful_urls:
        echo("")
        echo("Useful URLs:")
        for label, url in useful_urls:
            echo(f"  {label}: {url}")
        echo(f"  credentials: {DIFY_ADMIN_EMAIL}/{DIFY_ADMIN_PASSWORD}")

    echo("")
    echo("Debug:")
    project_name = target_data.get("project_name") or ""
    echo(f"  docker ps --filter label=com.docker.compose.project={project_name}")
    echo("  Press Ctrl-C to stop and clean up.")


def debug_target(
    *,
    project_file: str | Path,
    sample_id: str,
    public_host: str = "",
    run_id: str = "",
    force_recreate: bool = False,
    compose_up_timeout: float | None = None,
    startup_timeout: float | None = None,
    readiness_timeout: float | None = None,
    keep: bool = False,
    wait: bool = True,
    echo: Echo = _print_flush,
) -> dict[str, Any]:
    """Launch one target stack and keep it available for manual debugging."""
    project_path = Path(project_file).expanduser().resolve()
    run, sample, _sample_ids = _load_config_and_sample(project_path, sample_id)
    sample_id = str(sample.get("id") or sample_id)
    bench_root = _benchmark_root(run)
    public_host = (public_host or "").strip() or _detect_public_host()
    cage_run_id = (run_id or "").strip() or _default_run_id(sample_id)
    token = secrets.token_urlsafe(18)

    log_dir = Path.cwd() / ".cage_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"target_server-{cage_run_id}.log"

    extra_env: dict[str, str] = {
        "TARGET_SERVER_EXTERNAL_TOKEN": token,
        "TARGET_SERVER_HOST_IP": public_host,
    }
    if compose_up_timeout is not None:
        extra_env["TARGET_SERVER_COMPOSE_UP_TIMEOUT_S"] = str(compose_up_timeout)
    if startup_timeout is not None:
        extra_env["TARGET_SERVER_STARTUP_TIMEOUT_S"] = str(startup_timeout)

    embedded = None
    target_data: dict[str, Any] = {}
    target_run_id = ""
    try:
        embedded = _spawn_server(
            run_id=cage_run_id,
            benchmark_root=bench_root,
            log_path=log_path,
            extra_env=extra_env,
        )
        runtime_args = _runtime_args(run, sample)
        launch_url = _launch_url(
            embedded.server_url,
            sample_id,
            cage_run_id=cage_run_id,
            runtime_args=runtime_args,
            force_recreate=force_recreate,
        )
        echo(f"Launching {sample_id} via {embedded.server_url} ...")
        target_data = _request_json("GET", launch_url, token=token, timeout=None)
        _apply_agent_input_entry_urls(target_data, sample, public_host)
        target_run_id = str(target_data.get("run_id") or "")
        _print_launch_summary(
            echo=echo,
            target_data=target_data,
            sample_id=sample_id,
            cage_run_id=cage_run_id,
            public_host=public_host,
            log_path=log_path,
            server_pid=getattr(getattr(embedded, "process", None), "pid", None),
        )

        if not wait or str(target_data.get("status") or "") == "static":
            return target_data

        _wait_for_application_ready(
            sample_id=sample_id,
            target_data=target_data,
            public_host=public_host,
            readiness_timeout=(
                readiness_timeout
                if readiness_timeout is not None
                else (startup_timeout if startup_timeout is not None else 300.0)
            ),
            echo=echo,
        )

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        echo("")
        echo("Stopping target ...")
        return target_data
    finally:
        if embedded is not None and not keep:
            if target_run_id:
                try:
                    _request_json(
                        "DELETE",
                        _delete_url(embedded.server_url, sample_id, target_run_id),
                        timeout=120.0,
                    )
                    echo(f"Stopped target: {target_run_id}")
                except TargetDebugError as exc:
                    echo(f"Warning: target cleanup failed: {exc}")
            try:
                embedded.stop()
                echo("Stopped target_server.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("target_server stop failed: %s", exc)
                echo(f"Warning: target_server stop failed: {exc}")
        elif embedded is not None and keep:
            echo("")
            echo("Keeping target and target_server running.")
            echo(f"  target_server: {embedded.server_url}")
            echo(f"  target_server_pid: {getattr(getattr(embedded, 'process', None), 'pid', '')}")
            if target_run_id:
                echo(
                    "  stop target: "
                    f"curl -X DELETE '{_delete_url(embedded.server_url, sample_id, target_run_id)}'"
                )
            echo(f"  cleanup fallback: cage gc --run-id {cage_run_id} --apply")
