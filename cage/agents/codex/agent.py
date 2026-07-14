"""Codex CLI agent type.

Codex speaks OpenAI protocol natively.
"""

from __future__ import annotations

import getpass
import json
import logging
import socket
import stat
import sys
from pathlib import Path
from typing import Any

from cage.agents.base import openviking
from cage.agents.base import (
    AgentContainerResources,
    AgentType,
    HostRunService,
    register_agent_type,
)
from cage.agents.codex.output import parse_codex_event_stream
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult

logger = logging.getLogger(__name__)

# Host-side OAuth refresher launched for the lifetime of a ChatGPT-subscription
# run (see ``host_run_services``). The Codex twin of Claude Code's refresher;
# lives at the repo root beside the ``cage`` package.
_OAUTH_REFRESH_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "refresh_codex_oauth.py"
)


def _container_mounts_target(container: Any, target_path: str) -> bool:
    for container_path in getattr(container, "volumes", {}).values():
        mounted_path = str(container_path).split(":", 1)[0]
        if mounted_path == target_path:
            return True
    return False


@register_agent_type
class CodexAgent(AgentType):
    name = "codex"
    state_paths = [".codex"]
    default_image = "cage/codex:pentestenv"
    dockerfile = "docker/codex/pentestenv.Dockerfile"

    def install_command(self, version: str = "latest") -> str:
        return f"npm install -g @openai/codex@{version}"

    @property
    def subscription_upstream_base_url(self) -> str:
        """ChatGPT Codex backend — reached with the OAuth bearer in sub mode.

        Codex appends ``/responses`` to its provider base_url; in OAuth mode
        that provider base is the proxy root, so Codex POSTs ``<proxy>/responses``
        and the proxy rewrites it onto ``<this>/responses`` =
        ``https://chatgpt.com/backend-api/codex/responses`` (the ChatGPT backend
        inference endpoint).
        """
        return "https://chatgpt.com/backend-api/codex"

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        escaped = prompt.replace("'", "'\\''")
        # Static codex config (model_provider, sandbox, auto-compact disable)
        # lives in ~/.codex/config.toml, seeded by setup_container. The only
        # per-trial dynamic value is the proxy base URL, since the in-container
        # proxy binds a fresh port each trial.
        #
        # OAuth/ChatGPT-subscription mode: the proxy forwards to the ChatGPT
        # Codex backend, whose inference path is ``<backend>/responses``. So
        # Codex's provider base must be the proxy ROOT (no ``/v1``) → Codex
        # POSTs ``<proxy>/responses``, which the proxy rewrites onto the
        # backend. API-key mode keeps the ``/v1`` OpenAI-style base.
        oauth = bool(model.auth_source) and not model.base_url
        if proxy_url:
            root = proxy_url.rstrip("/")
            base_url = root if oauth else f"{root}/v1"
        else:
            base_url = model.base_url or ""
        base_flag = (
            f' -c model_providers.cage.base_url="{base_url}"' if base_url else ""
        )
        return (
            f"codex exec '{escaped}' --model {model.model}"
            f"{base_flag}"
            f" --dangerously-bypass-approvals-and-sandbox"
            f" --skip-git-repo-check"
            f" --cd /home/agent/workspace"
            f" --json"
        )

    def parse_output(self, result: ExecResult) -> str:
        if result.exit_code != 0 and not result.stdout.strip():
            return f"[Agent exited with code {result.exit_code}]\n{result.stderr[:1000]}"
        summary = parse_codex_event_stream(result.stdout)
        if summary.is_event_stream:
            parsed = summary.final_output()
            if parsed:
                return parsed
        return result.stdout.strip()

    def env_vars(
        self,
        *,
        proxy_url: str,
        model: ModelConfig,
        container: Any = None,
        home_dir: str = "/home/agent",
        workspace_dir: str = "",
        max_rounds: int = -1,
        context_compaction_threshold: float | None = None,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        # OAuth/ChatGPT-subscription mode: auth comes from the bind-mounted
        # ~/.codex/auth.json OAuth tokens and routing from config.toml's
        # ``requires_openai_auth`` provider (base_url set per-trial in
        # build_launch_command). Inject NO OPENAI_API_KEY — a key present would
        # flip Codex back to API-key mode and bypass the ChatGPT tokens.
        if bool(model.auth_source) and not model.base_url:
            return env
        if proxy_url:
            env["OPENAI_BASE_URL"] = proxy_url
        elif model.base_url:
            env["OPENAI_BASE_URL"] = model.base_url
        if model.api_key:
            env["OPENAI_API_KEY"] = model.api_key
        return env

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
    ) -> None:
        """Seed ``~/.codex/auth.json`` and ``~/.codex/config.toml``.

        Static codex configuration lives in ``config.toml`` (the supported
        upstream path) so ``codex exec`` invocations stay short. Two auth modes
        share this method:

        * **API-key** (default): ``auth.json`` carries ``OPENAI_API_KEY`` and
          the ``cage`` provider reads it via ``env_key``.
        * **ChatGPT-subscription / OAuth** (``model.auth_source`` set, no
          ``base_url``): ``auth.json`` carries the host's ChatGPT OAuth tokens
          (normally bind-mounted by :meth:`container_resources`), and the
          ``cage`` provider declares ``requires_openai_auth = true`` with NO
          ``env_key`` — so Codex authenticates with those tokens and emits the
          OAuth ``Bearer`` + ``ChatGPT-Account-ID`` headers, which the proxy
          forwards to the ChatGPT Codex backend verbatim.

        The proxy ``base_url`` is set per-trial via ``-c`` in
        :meth:`build_launch_command` because the in-container proxy chooses a
        free port at each trial. Per-experiment knobs such as
        ``model_reasoning_effort`` belong in ``project.yml`` under
        ``agents[].session_args`` rather than here.
        """
        codex_dir = f"{home_dir.rstrip('/')}/.codex"
        container.exec(f"mkdir -p {codex_dir}", timeout=5.0)

        oauth = (
            model is not None and bool(model.auth_source) and not model.base_url
        )

        if oauth:
            # OAuth tokens come from the host auth.json (bind-mounted, or copied
            # in for legacy callers). ``requires_openai_auth = true`` + NO
            # ``env_key`` keeps Codex in ChatGPT-subscription mode.
            self._seed_oauth_auth(
                container, codex_dir=codex_dir, source=model.auth_source,
            )
            provider_block = (
                '[model_providers.cage]\n'
                'name = "Cage Proxy"\n'
                'requires_openai_auth = true\n'
                'wire_api = "responses"\n'
            )
        else:
            # Auth — real API key when known, placeholder otherwise. codex exec
            # tolerates a placeholder when OPENAI_API_KEY is set in env, but the
            # interactive TUI insists on a non-empty auth.json.
            api_key = (model.api_key if model is not None else "") or "placeholder"
            container.write_file(
                f"{codex_dir}/auth.json", json.dumps({"OPENAI_API_KEY": api_key}),
            )
            provider_block = (
                '[model_providers.cage]\n'
                'name = "Cage Proxy"\n'
                'env_key = "OPENAI_API_KEY"\n'
                'wire_api = "responses"\n'
            )

        model_name = model.model if model is not None else ""
        model_line = f'model = "{model_name}"\n' if model_name else ""
        config_toml = (
            f'{model_line}'
            f'model_provider = "cage"\n'
            f'approval_policy = "never"\n'
            f'sandbox_mode = "danger-full-access"\n'
            f'\n'
            f'{provider_block}'
        )
        container.write_file(f"{codex_dir}/config.toml", config_toml)
        # Own the dir + config for the agent user. In OAuth mode the
        # bind-mounted auth.json shares the host file's inode, so we must NOT
        # ``chown`` it (that would rewrite host ownership) — the agent reaches
        # it through Docker ``--group-add`` instead (see container_resources).
        if oauth:
            container.exec(
                f"chown agent:agent {codex_dir} {codex_dir}/config.toml",
                timeout=5.0,
            )
        else:
            container.exec(f"chown -R agent:agent {codex_dir}", timeout=5.0)

    # ------------------------------------------------------------------ #
    # ChatGPT-subscription / OAuth support (mirrors claude_code)
    # ------------------------------------------------------------------ #

    def validate_auth(self, model: ModelConfig) -> None:
        """Verify a usable ChatGPT ``auth.json`` exists before any trial runs.

        Only checked in subscription mode (``auth_source`` set). Requires an
        ``auth.json`` carrying ``tokens.access_token``; otherwise raises
        ``ValueError`` naming host / user / how to fix.
        """
        if not model.auth_source:
            return
        try:
            auth_path = self._resolve_oauth_auth_path(model.auth_source)
            data = json.loads(auth_path.read_text())
            if (data.get("tokens") or {}).get("access_token"):
                return
        except (RuntimeError, OSError, json.JSONDecodeError):
            pass
        host = socket.gethostname()
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001
            user = "?"
        raise ValueError(
            f"config/models.yml::{model.id}: auth_source={model.auth_source!r} "
            f"has no usable ChatGPT auth.json (tokens.access_token) on this host.\n"
            f"  Host: {host}\n"
            f"  User: {user}\n"
            f"Fix one of:\n"
            f"  1. Sign in with ChatGPT on this host (`codex login`) so "
            f"~/.codex/auth.json gets created with OAuth tokens.\n"
            f"  2. Point auth_source at the correct host-local path.\n"
            f"  3. Remove auth_source from this model entry to fall back to "
            f"api_key auth."
        )

    def container_resources(
        self, *, home_dir: str, model: ModelConfig,
    ) -> AgentContainerResources:
        """Bind-mount ONLY the ChatGPT ``auth.json`` into the container.

        We mount the single OAuth file (not the whole ``~/.codex`` dir, which
        holds session/history state that would leak across stateless trials).
        A single-file bind mount shares the host inode, so the host-side
        refresher's in-place writes keep the container's token fresh. The agent
        user reaches the host-owned file through Docker ``--group-add``.
        """
        if not model.auth_source:
            return AgentContainerResources()
        auth_path = self._resolve_oauth_auth_path(model.auth_source)
        group_id = self._prepare_credential_file_mount(auth_path)
        home = home_dir.rstrip("/")
        return AgentContainerResources(
            volumes={str(auth_path): f"{home}/.codex/auth.json"},
            group_add=[str(group_id)],
        )

    def host_run_services(
        self, model: ModelConfig, *, http_proxy: str = "",
    ) -> list[HostRunService]:
        """Keep the OAuth access token fresh on disk while the run is live.

        Only subscription mode needs it (``auth_source`` set, no ``base_url``):
        trial containers bind-mount ``auth.json`` and cannot refresh it
        themselves, so a single host-side daemon keeps its access token's
        remaining lifetime above the trial duration. Codex ROTATES the refresh
        token on every refresh, so exactly one daemon may run — its ``flock``
        makes it a singleton and the orchestrator dedups by the auth path.
        """
        if not (model.auth_source and not model.base_url):
            return []
        if not _OAUTH_REFRESH_SCRIPT.is_file():
            logger.warning(
                "subscription mode but Codex OAuth refresher missing at %s; "
                "trials may hit HTTP 401 once the on-disk access token expires",
                _OAUTH_REFRESH_SCRIPT,
            )
            return []
        auth_path = self._resolve_oauth_auth_path(model.auth_source)
        env: dict[str, str] = {}
        if http_proxy:
            env["HTTPS_PROXY"] = http_proxy
            env["https_proxy"] = http_proxy
        argv = [
            sys.executable,
            str(_OAUTH_REFRESH_SCRIPT),
            "--daemon",
            "--creds",
            str(auth_path),
        ]
        return [
            HostRunService(
                name="codex-oauth-refresh",
                argv=argv,
                env=env,
                dedup_key=str(auth_path),
            )
        ]

    def _seed_oauth_auth(
        self, container: Any, *, codex_dir: str, source: str,
    ) -> None:
        """Make the host ChatGPT ``auth.json`` visible at ``{codex_dir}/auth.json``.

        Normally a bind mount created before container start (see
        :meth:`container_resources`); legacy callers without that mount get a
        copy of the host file. Missing tokens are fatal — a subscription trial
        is pointless without them.
        """
        auth_path = self._resolve_oauth_auth_path(source)
        target = f"{codex_dir}/auth.json"
        if _container_mounts_target(container, target):
            logger.info("Codex OAuth auth.json bind-mounted from %s", auth_path)
            return
        container.write_file(target, auth_path.read_text())
        logger.info("Codex OAuth auth.json copied from %s", auth_path)

    def _resolve_oauth_auth_path(self, source: str) -> Path:
        """Resolve ``auth_source`` to the ``auth.json`` file (accepts dir or file)."""
        src = Path(source).expanduser()
        if src.is_file():
            return src
        candidate = src / "auth.json"
        if candidate.is_file():
            return candidate
        raise RuntimeError(
            f"auth_source={source!r}: no auth.json at {candidate} or at {src}"
        )

    def _prepare_credential_file_mount(self, cred_path: Path) -> int:
        """Make a file bind mount readable/writable by the container agent.

        Docker bind mounts preserve the host file's uid/gid/mode. Codex writes
        ``auth.json`` ``0600`` owned by the operator, while the container's
        ``agent`` user is uid/gid 1000. We keep the mount scoped to the single
        file and grant access through its group plus Docker ``--group-add`` —
        group-writable so the CLI can still write if it ever refreshes.
        """
        st = cred_path.stat()
        mode = stat.S_IMODE(st.st_mode)
        needed = stat.S_IRGRP | stat.S_IWGRP
        if (mode & needed) != needed:
            try:
                cred_path.chmod(mode | needed)
            except OSError as exc:
                raise RuntimeError(
                    f"auth_source credential file {cred_path} must be group "
                    "readable/writable for bind mounting into the agent "
                    f"container; chmod failed: {exc}"
                ) from exc
        return st.st_gid

    def version_command(self) -> str:
        # NOTE: do NOT run 'codex --version' — it initialises a Landlock
        # sandbox as root, which permanently blocks later user-level execs.
        return "which codex >/dev/null 2>&1 && dpkg-query -W codex 2>/dev/null || (ls /usr/bin/codex >/dev/null 2>&1 && echo 'codex installed') || echo unknown"

    # ------------------------------------------------------------------ #
    # Plugin installation
    # ------------------------------------------------------------------ #

    def _plugin_installed(
        self, container: Any, *, name: str, home_dir: str,
    ) -> bool:
        """Check if MCP server is already registered in config.toml."""
        config = f"{home_dir}/.codex/config.toml"
        result = container.exec(
            f"grep -q '{name}' {config} 2>/dev/null", timeout=5.0,
        )
        return result.exit_code == 0

    def _do_install_plugin(
        self, container: Any, *, name: str, home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Register the plugin's MCP server via ``codex mcp add``.

        The server binary is read from the mounted marketplace directory
        at ``/opt/cage-plugins/{name}-marketplace/plugins/{name}/``.
        """
        server = (
            f"/opt/cage-plugins/{name}-marketplace"
            f"/plugins/{name}/servers/memory-server.js"
        )
        container.exec(
            f"codex mcp add {name} -- node {server}",
            user="agent", timeout=10.0,
        )

        if name == "openviking-memory":
            openviking.seed_conf(container, home_dir=home_dir)

    @property
    def protocol(self) -> str:
        return "openai"
