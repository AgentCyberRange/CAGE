"""OpenViking memory-plugin lifecycle shared by the agent implementations.

Seeding ``~/.openviking/ov.conf`` and starting ``openviking-server`` is the
same dance for every agent CLI — only the memory-namespace key written into
the conf and the suggested ``:openviking`` image tag differ. The per-agent
classes used to carry near-identical private copies of both (hermes' even
said "same shape as claude_code"); this module is the one implementation.

Agents that support the plugin keep a thin public
``start_openviking_server`` method delegating here — the engine and the CLI
discover support via ``hasattr``, so the method must exist only on agents
that actually run the server.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Fixed in-container port the OpenViking server listens on.
OPENVIKING_PORT = 1933


def seed_conf(
    container: Any,
    *,
    home_dir: str,
    agent_id: str = "",
    namespace_key: str = "",
) -> None:
    """Create a minimal ``~/.openviking/ov.conf`` if absent.

    The config only needs ``server.host`` and ``server.port`` — the plugin
    hooks use these to reach the OpenViking server. Embedding and VLM
    settings live server-side.

    When ``namespace_key`` is given (e.g. ``"claude_code"``), ``agent_id`` is
    written into ``<namespace_key>.agentId`` so each cage agent instance gets
    its own isolated memory namespace within a shared OV server.
    """

    conf_dir = f"{home_dir}/.openviking"
    conf_path = f"{conf_dir}/ov.conf"
    check = container.exec(f"test -f {conf_path}", timeout=5.0)
    if check.exit_code == 0:
        return  # user or stateful restore already provided one

    conf: dict = {"server": {"host": "127.0.0.1", "port": OPENVIKING_PORT}}
    if namespace_key and agent_id:
        conf[namespace_key] = {"agentId": agent_id}
    container.exec(f"mkdir -p {conf_dir}", timeout=5.0)
    container.write_file(conf_path, json.dumps(conf, indent=2))
    container.exec(f"chown -R agent:agent {conf_dir}", timeout=5.0)


def start_server(container: Any, *, home_dir: str, image_hint: str = "") -> str:
    """Start ``openviking-server`` as a background process.

    Returns the PID (or empty string). Only works on images that have the
    ``openviking`` Python package pre-installed; ``image_hint`` names the
    agent's ``:openviking`` image variant in the skip warning.

    Skips if the server is already running (idempotent for stateful
    containers that persist across trials).
    """

    # Already running? Use pgrep -x to match the exact process name,
    # avoiding false positives from the grep/pgrep command itself.
    check = container.exec(
        "pgrep -x openviking-serv >/dev/null 2>&1", timeout=5.0,
    )
    if check.exit_code == 0:
        logger.info("OpenViking server already running, skipping start")
        return ""

    have = container.exec(
        "python3 -c 'import openviking' 2>/dev/null", timeout=5.0,
    )
    if have.exit_code != 0:
        logger.warning(
            "openviking package not installed in image — skipping server "
            "start. Use %s image.", image_hint or "an :openviking",
        )
        return ""

    log_dir = f"{home_dir}/.openviking/logs"
    container.exec(f"mkdir -p {log_dir}", timeout=5.0)
    container.exec(
        f"chown -R agent:agent {home_dir}/.openviking 2>/dev/null || true",
        timeout=5.0,
    )

    conf_path = f"{home_dir}/.openviking/ov.conf"
    cmd = (
        f"OPENVIKING_CONFIG_FILE={conf_path} HOME={home_dir} "
        f"openviking-server "
        f">{log_dir}/server.log 2>&1"
    )
    pid = container.exec_background(cmd, user="agent")
    logger.info("OpenViking server starting (pid=%s)", pid)

    # Health-check: OV server can take 5-10s to initialize.
    # Poll /health for up to 30s.
    for _ in range(60):
        time.sleep(0.5)
        hc = container.exec(
            f"curl -s http://127.0.0.1:{OPENVIKING_PORT}/health 2>/dev/null",
            timeout=3.0,
        )
        if '"healthy":true' in hc.stdout or '"status":"ok"' in hc.stdout:
            logger.info("OpenViking server ready on :%s", OPENVIKING_PORT)
            return pid

    tail = container.exec(
        f"tail -20 {log_dir}/server.log 2>/dev/null", timeout=5.0,
    )
    logger.warning(
        "OpenViking server failed health-check after 30s. "
        "Log tail:\n%s", tail.stdout[:1000],
    )
    return pid
