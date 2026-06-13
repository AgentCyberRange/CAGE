"""Claude Code agent type.

Claude Code speaks Anthropic protocol. When paired with a non-Anthropic
model (vLLM, etc.), the proxy auto-translates.

State paths: .claude/ contains MEMORY.md, settings.json, session data.
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

from cage.agents.base.output import extract_stream_json_text, failure_banner
from cage.agents.base import openviking
from cage.agents.base import (
    AgentContainerResources,
    AgentType,
    HostRunService,
    register_agent_type,
)
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult

logger = logging.getLogger(__name__)

# The single host-side OAuth refresher, launched by the orchestrator for the
# lifetime of any subscription-mode run (see ``host_run_services``). It lives
# at the repo root beside the ``cage`` package.
_OAUTH_REFRESH_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "refresh_claude_oauth.py"
)

# Fields kept from a host ``~/.claude.json`` when seeding subscription
# auth into a container. The full host file is ~58 KB of cached state,
# personal browsing history (``projects``), MCP server configs, growth-
# book flags, etc. — none of which the OAuth flow needs. Keeping only
# this slim subset is enough for the CLI to identify the account and
# skip onboarding, without leaking unrelated user state into containers.
_SUBSCRIPTION_CLAUDE_JSON_KEEP = (
    "oauthAccount",
    "hasCompletedOnboarding",
    "hasAvailableSubscription",
    "numStartups",
    "firstStartTime",
    "installMethod",
)


def _container_mounts_target(container: Any, target_path: str) -> bool:
    for container_path in getattr(container, "volumes", {}).values():
        mounted_path = str(container_path).split(":", 1)[0]
        if mounted_path == target_path:
            return True
    return False


@register_agent_type
class ClaudeCodeAgent(AgentType):
    name = "claude_code"
    state_paths = [".claude"]
    default_image = "cage/claude-code:pentestenv"
    dockerfile = "docker/claude_code_pentestenv.Dockerfile"
    plugin_images = {"openviking-memory": "openviking"}

    def install_command(self, version: str = "latest") -> str:
        return f"npm install -g @anthropic-ai/claude-code@{version}"

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        escaped = prompt.replace("'", "'\\''")
        cli_model = model.model_name_for_agent(self.name)
        max_turns_flag = (
            f" --max-turns {int(max_rounds)}" if max_rounds >= 0 else ""
        )
        return (
            f"claude -p '{escaped}' "
            f"--output-format stream-json "
            f"--model {cli_model}"
            f"{max_turns_flag}"
        )

    def parse_output(self, result: ExecResult) -> str:
        """Parse Claude Code's stream-json output.

        Claude Code outputs NDJSON lines. We look for the final
        assistant message with type="result".
        """
        banner = failure_banner(result)
        if banner is not None:
            return banner
        last_text, _ = extract_stream_json_text(result.stdout)
        return last_text or result.stdout[:2000]

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
        """Build Claude Code env vars.

        Splits into three layers:
          - **Always**: proxy endpoint + telemetry kill (sandbox policy).
          - **Anthropic upstream**: real ``sk-ant-*`` key, stock model names,
            all features on. Behave like a vanilla Claude Code install.
          - **Non-Anthropic upstream** (vLLM / sglang / openai-compat): bearer
            auth, rewrite model ids to the real upstream name, and disable
            Anthropic-specific feature probes unsupported by the backend.

        ``context_compaction_threshold`` is the fraction of the context
        window (0.0–1.0) at which Claude Code triggers auto-compaction;
        it maps to ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`` (an integer percent).
        When ``None`` (the default — user did not opt in) we inject nothing
        and Claude Code keeps its own built-in threshold (~83.5%); we do NOT
        silently halve it. Note Claude Code caps the override at its default,
        so values above ~0.835 are clamped down by the CLI.
        """
        env: dict[str, str] = {}
        cli_model = model.model_name_for_agent(self.name)

        # Always: point Claude Code at our in-container proxy.
        if proxy_url:
            env["ANTHROPIC_BASE_URL"] = proxy_url

        # Always: we're running benchmark containers, no telemetry allowed.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["DISABLE_TELEMETRY"] = "1"

        # Subscription auth: credentials seeded into ``~/.claude/`` by
        # ``setup_container`` carry the OAuth tokens. Skip ANTHROPIC_API_KEY
        # so the CLI's ``apiKeySource`` resolves to ``"none"`` and it
        # enters subscription mode; the proxy passes the OAuth ``Bearer``
        # header through verbatim.
        if model.auth_source:
            return env

        if model.protocol == "anthropic":
            # Real Anthropic backend — nothing provider-specific to paper over.
            if model.api_key:
                env["ANTHROPIC_API_KEY"] = model.api_key
            # When the upstream is an Anthropic-compat endpoint that hosts a
            # non-Claude model (e.g. DeepSeek's https://api.deepseek.com/anthropic
            # serving ``deepseek-v4-pro``), still rewrite Claude Code's default
            # model ids so the outgoing request carries the real model name.
            # Otherwise Claude Code sends ``claude-opus-4-7`` and DeepSeek
            # silently maps unknown names to ``deepseek-v4-flash``.
            if cli_model and not cli_model.startswith("claude-"):
                env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = cli_model
                env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = cli_model
                env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = cli_model
            return env

        # ---- Non-Anthropic backends (vLLM / sglang / openai-compat) ---- #

        # Claude Code 2.1.110 recognizes ANTHROPIC_API_KEY as the active
        # credential source for custom endpoints; ANTHROPIC_AUTH_TOKEN does
        # not show up in apiKeySource and requests never leave the client.
        if model.api_key:
            env["ANTHROPIC_API_KEY"] = model.api_key

        # Model routing. By default Claude asks for ``claude-sonnet-4-6`` etc.
        # Our proxy forwards ``body.model`` as-is to chat/completions, and a
        # vLLM/sglang endpoint rejects the unknown name — observed as
        # UND_ERR_SOCKET on the client. Pin all three tiers to the real id.
        if cli_model:
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = cli_model
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = cli_model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = cli_model

        # Anthropic-only features that probe capabilities; disable so vLLM
        # doesn't 400 on them. Autocompact threshold is configurable via the
        # ``context_compaction_threshold`` agent field (fraction in 0..1);
        # CLAUDE_AUTOCOMPACT_PCT_OVERRIDE expects an integer percent.
        env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
        if context_compaction_threshold is not None:
            pct = max(1, min(100, int(round(context_compaction_threshold * 100))))
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(pct)

        return env

    def version_command(self) -> str:
        return "claude --version 2>/dev/null || echo unknown"

    def validate_auth(self, model: ModelConfig) -> None:
        """Verify subscription credentials exist before any trial runs.

        Resolves ``model.auth_source`` (with ``~`` expansion) and checks
        that one of the two accepted layouts contains a
        ``.credentials.json``. If neither does, raises ``ValueError``
        with hostname / user / tried paths so the user knows exactly
        what to fix.
        """
        if not model.auth_source:
            return
        src = Path(model.auth_source).expanduser()
        candidates = (
            src / ".credentials.json",
            src / ".claude" / ".credentials.json",
        )
        if any(c.is_file() for c in candidates):
            return
        host = socket.gethostname()
        try:
            user = getpass.getuser()
        except Exception:
            user = "?"
        tried = "\n  ".join(str(c) for c in candidates)
        raise ValueError(
            f"config/models.yml::{model.id}: auth_source={model.auth_source!r} "
            f"has no .credentials.json on this host.\n"
            f"  Host:  {host}\n"
            f"  User:  {user}\n"
            f"  Tried:\n  {tried}\n"
            f"Fix one of:\n"
            f"  1. Authenticate Claude Code on this host (run "
            f"`claude /login`) so {src}/.credentials.json gets created.\n"
            f"  2. Edit config/models.yml::{model.id} and point auth_source at "
            f"the correct host-local path (api_key entries in this same "
            f"file are already host-specific).\n"
            f"  3. Remove auth_source from this model entry to fall back "
            f"to api_key auth."
        )

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
    ) -> None:
        """Seed ~/.claude.json (and, in subscription mode, OAuth credentials).

        Without this, `claude` (interactive) opens a welcome screen that does a
        hardcoded connectivity check against api.anthropic.com — which fails
        inside a sandboxed container even when ANTHROPIC_BASE_URL points at
        our local proxy. The onboarding marker short-circuits that screen.

        ``home_dir`` is the OS user home (e.g. ``/home/agent``) — not the agent
        workspace. The marker sits beside (not inside) the ``.claude/`` state
        dir so snapshot/restore of ``state_paths=[".claude"]`` won't touch it.

        When ``model.auth_source`` is non-empty it must point at a host
        directory containing ``.credentials.json`` (OAuth access/refresh
        tokens), and is typically accompanied by a ``.claude.json`` account
        metadata file. The credential file is normally bind-mounted into the
        container by :meth:`container_resources`; a slim subset of
        ``.claude.json`` is written per container to avoid leaking unrelated
        personal state into the trial. Path validity is asserted by the
        experiment loader, so a missing ``.credentials.json`` here means the
        path disappeared between config-load and trial start.
        """
        home = home_dir.rstrip("/")
        marker_path = f"{home}/.claude.json"
        auth_source = model.auth_source if model is not None else ""

        if auth_source:
            self._seed_subscription_credentials(
                container, home_dir=home, source=auth_source,
                marker_path=marker_path,
            )
            return

        # Classic api-key mode: just the onboarding marker.
        container.write_file(marker_path, json.dumps({"hasCompletedOnboarding": True}))
        container.exec(f"chown agent:agent {marker_path}", timeout=5.0)

    def container_resources(
        self, *, home_dir: str, model: ModelConfig,
    ) -> AgentContainerResources:
        """Bind-mount only the OAuth credential file needed by Claude Code.

        We intentionally do not mount the whole host ``.claude`` directory:
        project/session/cache files there would leak state across stateless
        trials. The marker file ``~/.claude.json`` is still written as a slim
        post-start stub by :meth:`setup_container`.

        Freshness invariant (load-bearing, easy to break):
        The bind-mounted access token must outlive each trial. That is kept
        true by a single host-side refresher (``scripts/refresh_claude_oauth.py
        --daemon``), which the orchestrator now launches and tears down
        automatically for the lifetime of every subscription run — see
        :meth:`host_run_services`. It is the ONLY process that calls the OAuth
        refresh endpoint — one refresher means no refresh-token rotation race.
        That daemon must write the file IN PLACE (same inode), never tmp+rename:
        a single-file bind mount binds the inode at container start, so a
        rename (new inode) would leave already-running containers reading the
        orphaned old inode forever. Do not switch the daemon back to an
        atomic tmp+rename write, or running trials will silently stop seeing
        refreshed tokens.
        """
        if not model.auth_source:
            return AgentContainerResources()

        cred_path, _json_path = self._resolve_subscription_auth_paths(
            model.auth_source,
        )
        group_id = self._prepare_credential_file_mount(cred_path)
        home = home_dir.rstrip("/")
        return AgentContainerResources(
            volumes={
                str(cred_path): f"{home}/.claude/.credentials.json",
            },
            group_add=[str(group_id)],
        )

    def host_run_services(
        self, model: ModelConfig, *, http_proxy: str = "",
    ) -> list[HostRunService]:
        """Keep the OAuth access token fresh on disk while the run is live.

        Only subscription mode needs this: ``auth_source`` is set (OAuth
        credentials on the host) and ``base_url`` is empty (upstream is the
        real Anthropic API, reached with the OAuth bearer). In that mode trial
        containers copy / bind-mount ``.credentials.json`` and cannot refresh
        it themselves, so a single host-side daemon must keep its access
        token's remaining lifetime above the trial duration — this is the
        freshness invariant documented on :meth:`container_resources`. Without
        it, every trial that starts after the on-disk token expires gets
        ``HTTP 401`` from Anthropic.

        Returns no service for API-key mode or a custom ``base_url`` (which is
        not the OAuth endpoint). The refresher's own ``flock`` makes it a
        singleton across concurrent runs, so it is safe for every subscription
        run to declare it; the orchestrator additionally deduplicates by the
        credentials path.
        """
        if not (model.auth_source and not model.base_url):
            return []
        if not _OAUTH_REFRESH_SCRIPT.is_file():
            logger.warning(
                "subscription mode but OAuth refresher missing at %s; trials "
                "may hit HTTP 401 once the on-disk access token expires",
                _OAUTH_REFRESH_SCRIPT,
            )
            return []
        cred_path, _ = self._resolve_subscription_auth_paths(model.auth_source)
        env: dict[str, str] = {}
        if http_proxy:
            env["HTTPS_PROXY"] = http_proxy
            env["https_proxy"] = http_proxy
        argv = [
            sys.executable,
            str(_OAUTH_REFRESH_SCRIPT),
            "--daemon",
            "--creds",
            str(cred_path),
        ]
        return [
            HostRunService(
                name="claude-oauth-refresh",
                argv=argv,
                env=env,
                dedup_key=str(cred_path),
            )
        ]

    def _seed_subscription_credentials(
        self, container: Any, *, home_dir: str, source: str,
        marker_path: str,
    ) -> None:
        """Prepare Claude Code subscription files in the container home.

        Two files are made visible inside the container:

        * ``{home_dir}/.claude/.credentials.json`` — OAuth access + refresh
          tokens. In normal runs this is a bind mount created before
          container start; legacy callers without that mount still get a copy.
        * ``{home_dir}/.claude.json`` — slim subset of the host account
          marker. ``CLAUDE_CONFIG_DIR``-style sources keep it at
          ``{source}/.claude.json``; stock ``~/.claude`` sources keep it at
          the sibling ``~/.claude.json``. Carries ``oauthAccount`` (account
          UUID / org / subscription type) plus onboarding markers so the CLI
          recognises the account these tokens belong to.

        Missing ``.credentials.json`` is fatal — there is no point starting
        a subscription-mode trial without tokens. Missing ``.claude.json``
        falls back to the minimal onboarding stub.
        """
        cred_path, json_path = self._resolve_subscription_auth_paths(source)
        marker_text = self._build_subscription_marker_text(
            source=source,
            json_path=json_path,
        )

        cred_target = f"{home_dir}/.claude/.credentials.json"
        mounted_credentials = _container_mounts_target(container, cred_target)
        container.exec(f"mkdir -p {home_dir}/.claude", timeout=5.0)
        if not mounted_credentials:
            cred_text = cred_path.read_text()
            container.write_file(cred_target, cred_text)
        container.write_file(marker_path, marker_text)
        if mounted_credentials:
            container.exec(
                f"chown agent:agent {home_dir}/.claude {marker_path} && "
                f"chmod 600 {marker_path}",
                timeout=5.0,
            )
        else:
            container.exec(
                f"chown -R agent:agent {home_dir}/.claude {marker_path} && "
                f"chmod 600 {cred_target} {marker_path}",
                timeout=5.0,
            )
        logger.info(
            "Subscription credentials seeded from %s into %s",
            cred_path,
            "bind mount" if mounted_credentials else cred_target,
        )

    def _resolve_subscription_auth_paths(self, source: str) -> tuple[Path, Path]:
        src_dir = Path(source).expanduser()
        # Accept both forms: ``~/.claude`` (the credentials dir itself) and
        # ``~`` (the parent of ``.claude``). Normalize to the credentials dir.
        if (src_dir / ".credentials.json").is_file():
            json_path = src_dir / ".claude.json"
            if not json_path.is_file():
                json_path = src_dir.parent / ".claude.json"
            return src_dir / ".credentials.json", json_path
        if (src_dir / ".claude" / ".credentials.json").is_file():
            return src_dir / ".claude" / ".credentials.json", src_dir / ".claude.json"
        raise RuntimeError(
            f"auth_source={source!r}: no .credentials.json found at "
            f"{src_dir}/.credentials.json or {src_dir}/.claude/.credentials.json"
        )

    def _build_subscription_marker_text(self, *, source: str, json_path: Path) -> str:
        if json_path.is_file():
            try:
                full = json.loads(json_path.read_text())
                slim = {
                    k: full[k] for k in _SUBSCRIPTION_CLAUDE_JSON_KEEP
                    if k in full
                }
                if "hasCompletedOnboarding" not in slim:
                    slim["hasCompletedOnboarding"] = True
                return json.dumps(slim)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "auth_source=%s: failed to parse %s (%s); using "
                    "minimal onboarding stub. CLI may re-prompt for "
                    "subscription details on first launch.",
                    source, json_path, exc,
                )
                return json.dumps({"hasCompletedOnboarding": True})

        logger.info(
            "auth_source=%s: no .claude.json at %s; using minimal "
            "onboarding stub.", source, json_path,
        )
        return json.dumps({"hasCompletedOnboarding": True})

    def _prepare_credential_file_mount(self, cred_path: Path) -> int:
        """Make a file bind mount readable/writable by the container agent.

        Docker bind mounts preserve the host file's uid/gid/mode. Host Claude
        credentials are usually ``0600`` and owned by the operator, while the
        container's ``agent`` user is uid/gid 1000. We keep the mount scoped to
        the single credential file and grant access through the file's existing
        group plus Docker ``--group-add``.
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

    # ------------------------------------------------------------------ #
    # Plugin installation
    # ------------------------------------------------------------------ #

    def _plugin_installed(
        self, container: Any, *, name: str, home_dir: str,
    ) -> bool:
        """Check Claude Code's installed_plugins.json for *name*."""
        path = f"{home_dir}/.claude/plugins/installed_plugins.json"
        result = container.exec(
            f"grep -q '{name}' {path} 2>/dev/null", timeout=5.0,
        )
        return result.exit_code == 0

    def _do_install_plugin(
        self, container: Any, *, name: str, home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Install via local marketplace add + plugin install.

        The marketplace directory is mounted read-only at
        ``/opt/cage-plugins/{name}-marketplace``.  ``claude plugin
        marketplace add`` accepts a local path, and ``claude plugin
        install`` copies files + registers hooks/MCP — all offline.

        For plugins that need a config file (e.g. OpenViking's
        ``~/.openviking/ov.conf``), a minimal default is seeded so hooks
        can start immediately without manual setup.
        """
        src = f"/opt/cage-plugins/{name}-marketplace"
        container.exec(
            f"claude plugin marketplace add {src}",
            user="agent", timeout=300.0,
        )
        container.exec(
            f"claude plugin install {name}",
            user="agent", timeout=300.0,
        )

        # Seed plugin-specific config files if they don't exist yet.
        if name == "openviking-memory":
            openviking.seed_conf(
                container, home_dir=home_dir, agent_id=agent_id,
                namespace_key="claude_code",
            )

    # ------------------------------------------------------------------ #
    # OpenViking server lifecycle
    # ------------------------------------------------------------------ #

    def start_openviking_server(
        self, container: Any, *, home_dir: str,
    ) -> str:
        """Start ``openviking-server`` (shared lifecycle in base.openviking)."""
        return openviking.start_server(
            container, home_dir=home_dir,
            image_hint="cage/claude-code:openviking",
        )

    @property
    def protocol(self) -> str:
        """Claude Code speaks Anthropic protocol."""
        return "anthropic"
