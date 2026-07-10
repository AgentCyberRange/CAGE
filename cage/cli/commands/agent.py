"""Agent runtime Cage CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from cage.artifacts.run_storage import (
    trial_path,
    trial_prompt_path,
    trial_state_dir_path,
)
from cage.cli.commands import model as model_commands
from cage.cli.paths import display_path


@click.command("debug")
@click.argument("run_dir", required=False, type=click.Path(exists=True))
@click.option(
    "--agent", "agent_name",
    default="",
    help="Agent type to launch (e.g. claude_code, codex). Required if RUN_DIR not given.",
)
@click.option(
    "--model", "model_id",
    default="",
    help="Model ID from config/models.yml. Required if RUN_DIR not given.",
)
@click.option(
    "--models", "models_file",
    type=click.Path(exists=True),
    default=None,
    hidden=True,
    help="Override the repo default model registry.",
)
@click.option("--skill", default="", help="Agent skill to install")
@click.option(
    "--plugin",
    "plugins",
    multiple=True,
    help="Plugin to install (repeatable, e.g. --plugin openviking-memory)",
)
@click.option("--version", default="", help="Agent CLI version to install")
@click.option("--no-proxy", is_flag=True, help="Skip proxy (agent talks directly to upstream)")
@click.option(
    "--upstream-proxy", "upstream_http_proxy", default="",
    help="HTTP proxy URL the in-container cage-proxy should use to reach the model "
         "upstream (e.g. http://<host-ip>:7890). Without RUN_DIR, this is the only "
         "way to set it; with RUN_DIR it overrides the value read from config.yaml.",
)
@click.option("--image", default="", help="Override Docker image")
@click.option(
    "--trial", "trial_id", default="",
    help="Restore state from a specific trial (requires RUN_DIR)",
)
@click.option(
    "--state", "state_type",
    type=click.Choice(["pre", "post"]),
    default="pre",
    help="Which trial state to restore: pre-execution or post-execution (default: pre)",
)
def debug(
    run_dir: str | None,
    agent_name: str,
    model_id: str,
    models_file: str | None,
    skill: str,
    plugins: tuple[str, ...],
    version: str,
    no_proxy: bool,
    upstream_http_proxy: str,
    image: str,
    trial_id: str,
    state_type: str,
) -> None:
    """Launch an interactive agent container for debugging.

    Two modes:

    \b
    1. From a run directory (reads config.yaml for agent/model info):
         cage agent debug .cage_runs/run-.../agent_label
         cage agent debug .cage_runs/run-.../agent_label --trial trial_0003 --state post

    \b
    2. From scratch (requires --agent and --model):
         cage agent debug --agent claude_code --model glm-5.1-sii
         cage agent debug --agent claude_code --model glm-5.1-sii --skill self-improving-agent
         cage agent debug --agent claude_code --model glm-5.1-sii --plugin openviking-memory
    """
    import json
    import subprocess
    import time

    import yaml

    from cage.agents.base import AgentInstance, get_agent_type
    from cage.models import load_models
    from cage.proxy.host import ProxyInstanceConfig
    from cage.sandbox.containers import Container
    from cage.sandbox.state import restore_state, snapshot_state

    # ------------------------------------------------------------------ #
    # Resolve agent + model config from either run_dir or explicit flags
    # ------------------------------------------------------------------ #
    run_path: Path | None = None
    restore_trial_id = trial_id
    restore_state_type = state_type
    initial_state_dir: Path | None = None

    if run_dir:
        # Mode 1: from run directory
        run_path = Path(run_dir)
        config_path = run_path / "config.yaml"
        version_path = run_path / "agent_version.json"

        if not config_path.exists():
            click.echo(f"Error: {config_path} not found. Not a valid run directory.")
            raise SystemExit(1)

        run_config = yaml.safe_load(config_path.read_text())
        agent_label = run_config.get("agent", "")
        model_id_from_run = run_config.get("model", "")

        # Parse agent_label to extract agent type
        # Label format: "agent_type:model_id:stateful_or_stateless"
        # or just the agent type if it was a simple run
        agent_name = agent_name or agent_label.split(":")[0] if agent_label else agent_name
        model_id = model_id or model_id_from_run

        # Load agent version from run
        if version_path.exists() and not version:
            version_info = json.loads(version_path.read_text())
            version = version_info.get("version", "latest")

        # Check for initial state
        init_dir = run_path / "initial_state"
        if init_dir.exists() and any(init_dir.iterdir()):
            initial_state_dir = init_dir

        # Check for trial state
        if restore_trial_id:
            trial_dir = trial_path(run_path, restore_trial_id)
            if not trial_dir.exists():
                click.echo(f"Error: trial '{restore_trial_id}' not found in {run_dir}")
                raise SystemExit(1)
            state_dir = trial_dir / f"state_{restore_state_type}"
            if not state_dir.exists():
                click.echo(
                    f"Error: state_{restore_state_type} not found for "
                    f"trial {restore_trial_id}"
                )
                raise SystemExit(1)

        click.echo(f"Debug from run: {run_dir}")
        if restore_trial_id:
            click.echo(f"  Restoring trial {restore_trial_id} ({restore_state_type}-state)")

    # Validate required params
    if not agent_name:
        click.echo("Error: --agent is required when RUN_DIR is not specified.")
        raise SystemExit(1)
    if not model_id:
        click.echo("Error: --model is required when RUN_DIR is not specified.")
        raise SystemExit(1)

    # Load model config
    models_path = model_commands._model_registry_path(models_file)
    models = load_models(models_path)
    if model_id not in models:
        click.echo(
            "Error: model "
            f"'{model_id}' not found in {display_path(models_path)}"
        )
        click.echo(f"Available models: {', '.join(models.keys())}")
        raise SystemExit(1)

    model = models[model_id]
    agent_type = get_agent_type(agent_name)
    if not version:
        version = "latest"

    # Build agent instance
    default_home = (
        run_config.get("home", "/home/agent/workspace")
        if run_path
        else "/home/agent/workspace"
    )
    agent = AgentInstance(
        agent_type=agent_type,
        model=model,
        id="debug",
        # session_timeout removed — debug mode has no timeout
        version=version,
        image=image,
        skill=skill,
        plugins=list(plugins),
        shared_paths=run_config.get("shared_paths", []) if run_path else [],
        home=default_home,
    )

    effective_image = agent.effective_image
    container_name = f"cage-debug-{int(time.time()) % 100000}"

    # Resolve plugin volume mounts
    plugin_volumes: dict[str, str] = {}
    if agent.plugins:
        from cage.experiment.engine.trial_runner import _resolve_plugin_volumes
        plugin_volumes = _resolve_plugin_volumes(
            agent.plugins, Path.cwd(),
        )

    click.echo("Starting debug container...")
    click.echo(f"  Agent:   {agent_name} (v{version})")
    click.echo(f"  Model:   {model_id} ({model.model})")
    click.echo(f"  Image:   {effective_image}")
    click.echo(f"  Stateful: {agent.stateful}")
    if skill:
        click.echo(f"  Skill:   {skill}")
    for p in agent.plugins:
        click.echo(f"  Plugin:  {p}")

    # Auto-mount host ov.conf for OpenViking plugin
    if "openviking-memory" in agent.plugins:
        host_ov_conf = Path.home() / ".openviking" / "ov.conf"
        if host_ov_conf.exists():
            plugin_volumes[str(host_ov_conf)] = "/home/agent/.openviking/ov.conf:ro"
            click.echo(f"  OV conf: {host_ov_conf} (mounted)")
        else:
            click.echo(f"  Warning: {host_ov_conf} not found — OV server will fail to start")
            click.echo("           Create it with embedding + VLM config for OpenViking")

    agent_resources = agent_type.container_resources(
        home_dir="/home/agent",
        model=model,
    )
    plugin_volumes.update(agent_resources.volumes)

    # Start container
    container = Container(
        name=container_name,
        image=effective_image,
        env_vars={"HOME": "/home/agent"},
        volumes=plugin_volumes,
        group_add=agent_resources.group_add,
        network_mode="host",
    )
    container.start()
    container.setup_workspace(agent.home)

    # Install agent CLI (skip if already pre-installed in image)
    version_check = container.exec(
        agent_type.version_command(), timeout=10.0
    )
    if version_check.exit_code == 0 and version_check.stdout.strip() not in ("", "unknown"):
        click.echo(f"  Agent CLI already installed: {version_check.stdout.strip()}")
    else:
        click.echo("  Installing agent CLI...")
        install_result = container.exec(
            agent_type.install_command(version), timeout=120.0
        )
        if install_result.exit_code != 0:
            click.echo(f"Error: CLI install failed: {install_result.stderr[:500]}")
            container.stop()
            raise SystemExit(1)

    # Agent-specific container setup (e.g. skip Claude Code onboarding)
    agent_type.setup_container(
        container, home_dir="/home/agent", model=model,
    )

    # Install plugins (skip if already present from stateful restore)
    for plugin_name in agent.plugins:
        click.echo(f"  Installing plugin: {plugin_name}...")
        agent_type.install_plugin(
            container, name=plugin_name, home_dir="/home/agent",
        )

    # Start plugin backend servers (e.g. OpenViking)
    if "openviking-memory" in agent.plugins and hasattr(agent_type, "start_openviking_server"):
        click.echo("  Starting OpenViking server...")
        agent_type.start_openviking_server(container, home_dir="/home/agent")

    # Restore initial state if available from run_dir
    if initial_state_dir and any(initial_state_dir.iterdir()):
        click.echo("  Restoring initial state from run...")
        snapshot_state(
            container,
            state_paths=agent.effective_state_paths,
            home_dir="/home/agent",
            output_dir=Path.cwd() / ".cage-debug-state",
        )
        # Restore from the run's initial state
        from cage.sandbox.state import StateSnapshot
        initial_snapshot = StateSnapshot(
            snapshot_dir=initial_state_dir,
            state_paths=tuple(agent.effective_state_paths),
            timestamp_ms=0,
        )
        restore_state(
            container,
            snapshot=initial_snapshot,
            home_dir="/home/agent",
        )

    # Restore trial state if requested
    if restore_trial_id and run_path:
        trial_state_dir = trial_state_dir_path(run_path, restore_trial_id, restore_state_type)
        if trial_state_dir.exists() and any(trial_state_dir.iterdir()):
            click.echo(f"  Restoring trial {restore_trial_id} ({restore_state_type}-state)...")
            snapshot_state(
                container,
                state_paths=agent.effective_state_paths,
                home_dir="/home/agent",
                output_dir=Path.cwd() / ".cage-debug-state-current",
            )
            from cage.sandbox.state import StateSnapshot
            trial_restore_snapshot = StateSnapshot(
                snapshot_dir=trial_state_dir,
                state_paths=tuple(agent.effective_state_paths),
                timestamp_ms=0,
            )
            restore_state(
                container,
                snapshot=trial_restore_snapshot,
                home_dir="/home/agent",
            )
            # Also restore workspace files if task_output has sample
            prompt_file = trial_prompt_path(run_path, restore_trial_id)
            if prompt_file.exists():
                container.write_file(
                    f"{agent.home}/note.md",
                    prompt_file.read_text(),
                )

    # Start proxy if enabled — match trial config when run_dir provided
    proxy = None
    proxy_url = ""
    if not no_proxy:
        from cage.proxy.host import start_container_proxy
        proxy_section = (run_config.get("proxy") or {}) if run_path else {}
        runtime_section = (run_config.get("runtime") or {}) if run_path else {}
        # Subscription/OAuth fallback — mirror the preflight rule: when
        # ``auth_source`` is set and no explicit ``base_url``, point the
        # proxy at ``api.anthropic.com`` so the OAuth Bearer reaches the
        # real subscription backend. Without this, an empty base_url makes
        # the proxy try to connect to nowhere.
        _sub_mode = bool(model.auth_source) and not model.base_url
        upstream_base = "https://api.anthropic.com" if _sub_mode else model.base_url
        # --upstream-proxy overrides anything from config.yaml when given.
        effective_http_proxy = (
            upstream_http_proxy
            or proxy_section.get("upstream_http_proxy", "")
        )
        proxy_config = ProxyInstanceConfig(
            upstream_base_url=upstream_base,
            upstream_api_key=model.api_key,
            upstream_protocol=model.protocol,
            artifact_dir=Path.cwd() / ".cage-debug-proxy",
            trial_id="debug",
            system_template=proxy_section.get("rewrite_system", ""),
            port=0,
            request_timeout=proxy_section.get("request_timeout", 3600.0),
            http_proxy=effective_http_proxy,
            max_requests=runtime_section.get("max_rounds", 0),
            max_input_tokens=runtime_section.get("max_input_tokens"),
            max_output_tokens=runtime_section.get("max_output_tokens"),
            max_cost=runtime_section.get("max_cost"),
            input_cost_per_1m=model.input_cost_per_1m,
            output_cost_per_1m=model.output_cost_per_1m,
            upstream_extra_body=dict(model.upstream_extra_body or {}),
        )
        proxy = start_container_proxy(container, proxy_config)
        proxy_url = proxy.base_url
        click.echo(f"  Proxy:   {proxy_url} (protocol: {model.protocol})")
        max_requests_label = (
            proxy_config.max_requests if proxy_config.max_requests >= 0 else "unlimited"
        )
        click.echo(
            f"  Proxy timeout: {proxy_config.request_timeout}s, "
            f"max_requests: {max_requests_label}"
        )
    else:
        click.echo(f"  Proxy:   disabled (direct to {model.base_url})")

    # Build env vars for the agent
    agent_env = agent_type.env_vars(
        proxy_url=proxy_url,
        model=model,
        context_compaction_threshold=agent.context_compaction_threshold,
    )

    # Write env vars to a file inside the container (for scripts that source it)
    env_lines = [f"export {k}='{v}'" for k, v in agent_env.items()]
    env_script = "\n".join(env_lines) + "\n"
    container.exec("mkdir -p /home/agent", timeout=10.0)
    container.exec(
        f"cat > /home/agent/.cage_env << 'CAGE_EOF'\n{env_script}CAGE_EOF",
        timeout=10.0,
    )

    # Codex-specific: bake proxy URL + real API key into ~/.codex/ so
    # `codex` interactive works without manual flags.
    # Use a CUSTOM model_provider (not the built-in `openai`) so codex skips
    # WebSocket probing and server-side /compact endpoint (which most relays
    # like NewAPI don't implement). Equivalent to user-style local config.
    if agent_name == "codex" and proxy_url:
        codex_dir = "/home/agent/.codex"
        container.exec(f"mkdir -p {codex_dir}", timeout=5.0)
        # Auth (real key, not placeholder)
        auth = json.dumps({"OPENAI_API_KEY": model.api_key})
        container.write_file(f"{codex_dir}/auth.json", auth)
        check_cfg = container.exec(
            f"grep -q '^model_provider = \"cage\"' {codex_dir}/config.toml 2>/dev/null"
        )
        if check_cfg.exit_code != 0:
            cfg_lines = (
                f'model = "{model.model}"\n'
                f'model_provider = "cage"\n'
                f'approval_policy = "never"\n'
                f'sandbox_mode = "danger-full-access"\n'
                f'\n'
                f'[model_providers.cage]\n'
                f'name = "Cage Proxy"\n'
                f'base_url = "{proxy_url}/v1"\n'
                f'env_key = "OPENAI_API_KEY"\n'
                f'wire_api = "responses"\n'
            )
            container.exec(
                f"cat >> {codex_dir}/config.toml << 'CAGE_EOF'\n{cfg_lines}CAGE_EOF",
                timeout=5.0,
            )
        click.echo("  Codex: ~/.codex/config.toml seeded (custom provider 'cage')")

    # Keep debug setup ownership scoped. Subscription credentials may be a
    # bind-mounted host file under /home/agent/.claude, so never chown the
    # whole home directory here.
    container.exec("chown agent:agent /home/agent /home/agent/.cage_env", timeout=10.0)
    if agent_name == "codex":
        container.exec(
            "chown -R agent:agent /home/agent/.codex 2>/dev/null || true",
            timeout=10.0,
        )

    # Install skill if specified
    if skill:
        click.echo(f"  Installing skill: {skill}...")
        container.exec(
            f"su agent -c 'cd {agent.home} && claude skill install {skill}'",
            timeout=60.0,
        )

    click.echo()
    click.echo("=" * 60)
    click.echo("Interactive debug session started.")
    click.echo("  - Agent CLI is installed and configured")
    click.echo("  - Environment variables:")
    for k, v in agent_env.items():
        click.echo(f"      {k}={v}")
    if agent_name == "codex" and proxy_url:
        click.echo("  - Codex pre-configured. Just type: codex")
    elif agent_name == "claude_code":
        click.echo("  - Claude Code pre-configured. Just type: claude")
    if restore_trial_id:
        click.echo(f"  - Trial {restore_trial_id} {restore_state_type}-state restored")
    click.echo("  - Type 'exit' to stop the container")
    click.echo("=" * 60)
    click.echo()

    # Drop into interactive bash (with env vars injected via -e)
    try:
        # Build docker exec command with -e flags for each env var
        exec_cmd = ["docker", "exec", "-it", "-u", "agent"]
        exec_cmd.extend(["-e", "HOME=/home/agent", "-w", agent.home])
        for k, v in agent_env.items():
            exec_cmd.extend(["-e", f"{k}={v}"])
        exec_cmd.extend([container_name, "bash"])
        subprocess.run(exec_cmd)
    except KeyboardInterrupt:
        pass
    finally:
        click.echo("\nStopping debug container...")
        if proxy:
            proxy.stop(artifact_dir=Path.cwd() / ".cage-debug-proxy")
        container.stop()
        click.echo("Debug container removed.")


def _custom_agent_dirs(cage_root: Path) -> list[Path]:
    """In-repo custom-agent manifest dirs (``cage/agents/custom/<name>/agent.yml``)."""
    custom_root = cage_root / "cage" / "agents" / "custom"
    return [p.parent for p in sorted(custom_root.glob("*/agent.yml"))]


def _custom_agent_build_script(name: str, cage_root: Path) -> Path | None:
    """Absolute path of an in-repo custom agent's declared ``build.script``.

    Returns ``None`` when no in-repo custom agent is named ``name`` (or it has no
    build recipe). Lets ``cage agent build --agent <name>`` be the ONE build
    entry point for a custom agent whose image needs more than a single
    ``docker build`` (it declares ``build: {script: ...}`` in its manifest).
    """
    from cage.agents.custom.manifest import load_manifest

    for d in _custom_agent_dirs(cage_root):
        try:
            manifest = load_manifest(str(d), d.parent)
        except Exception:
            continue
        if manifest.name != name:
            continue
        script = (manifest.build or {}).get("script")
        if not script:
            return None
        script_path = (cage_root / script).resolve()
        if not script_path.is_file():
            raise click.ClickException(
                f"build script not found: {script} (declared by custom agent '{name}')"
            )
        return script_path
    return None


def _custom_agent_build_names(cage_root: Path) -> list[str]:
    """Names of in-repo custom agents that declare a ``build.script``."""
    from cage.agents.custom.manifest import load_manifest

    names: list[str] = []
    for d in _custom_agent_dirs(cage_root):
        try:
            manifest = load_manifest(str(d), d.parent)
        except Exception:
            continue
        if (manifest.build or {}).get("script"):
            names.append(manifest.name)
    return names


@click.command("build")
@click.option(
    "--agent", "agent_name",
    default="",
    help="Build image for a specific agent only (default: all agents)",
)
@click.option(
    "--variant",
    default="",
    help="Build a variant image (e.g. 'openviking' → cage/claude-code:openviking)",
)
@click.option(
    "--version", default="",
    help="Override agent CLI version (default: use Dockerfile's pinned version)",
)
@click.option(
    "--no-cache", is_flag=True,
    help="Build without Docker cache",
)
@click.option(
    "--all", "build_all", is_flag=True,
    help="Build ALL images: base + every variant found in docker/",
)
def build(agent_name: str, variant: str, version: str, no_cache: bool, build_all: bool) -> None:
    """Build Docker images for agent types.

    Pre-installs the agent CLI so containers start faster.

    \b
    Examples:
      cage agent build                                        # build all base images
      cage agent build --all                                  # build base + all variants
      cage agent build --agent claude_code                    # build only claude_code
      cage agent build --agent claude_code --variant pentestenv   # build pentestenv variant
      cage agent build --version 1.0.0                        # install specific CLI version
    """
    import subprocess

    # Find cage package root (where docker/ dir lives)
    import cage
    from cage.agents.base import _AGENT_TYPE_REGISTRY
    cage_root = Path(cage.__file__).resolve().parent.parent
    docker_dir = cage_root / "docker"

    agents_to_build = []
    if agent_name:
        if agent_name not in _AGENT_TYPE_REGISTRY:
            # Not a native AgentType — try an in-repo custom agent that ships its
            # own build recipe (e.g. cairn's multi-image Docker-in-Docker bake a
            # single `docker build` can't express). This keeps ONE build command
            # for every agent instead of a separate script the user must know.
            script_path = _custom_agent_build_script(agent_name, cage_root)
            if script_path is not None:
                click.echo(
                    f"Building custom agent '{agent_name}' via "
                    f"{script_path.relative_to(cage_root)}..."
                )
                if subprocess.run(["bash", str(script_path)], text=True).returncode != 0:
                    click.echo(f"Error: build failed for custom agent '{agent_name}'")
                    raise SystemExit(1)
                click.echo(f"  Built: custom agent '{agent_name}'")
                return
            native = ", ".join(sorted(_AGENT_TYPE_REGISTRY.keys()))
            custom = ", ".join(_custom_agent_build_names(cage_root)) or "(none)"
            click.echo(
                f"Error: unknown agent '{agent_name}'. "
                f"Native: {native}. Custom (buildable): {custom}"
            )
            raise SystemExit(1)
        agents_to_build = [(agent_name, _AGENT_TYPE_REGISTRY[agent_name])]
    else:
        agents_to_build = sorted(_AGENT_TYPE_REGISTRY.items())

    # Collect (dockerfile_path, image_tag) pairs
    build_targets: list[tuple[Path, str]] = []

    for name, cls in agents_to_build:
        agent = cls()

        if variant:
            # Single variant → docker/<name>/<variant>.Dockerfile
            vdf = docker_dir / name / f"{variant}.Dockerfile"
            if not vdf.exists():
                click.echo(f"Error: variant Dockerfile not found: {name}/{variant}.Dockerfile")
                raise SystemExit(1)
            build_targets.append((vdf, agent.image_for_variant(variant)))
        elif build_all:
            # Base image
            if agent.dockerfile:
                base_df = cage_root / agent.dockerfile
                if base_df.exists():
                    build_targets.append((base_df, agent.default_image))
            # All variants: docker/<name>/<variant>.Dockerfile
            # (the bare docker/<name>/Dockerfile is the base, handled above)
            agent_dir = docker_dir / name
            if agent_dir.is_dir():
                for vdf in sorted(agent_dir.glob("*.Dockerfile")):
                    if vdf.name == "Dockerfile":
                        continue
                    vname = vdf.stem  # e.g. "pentestenv"
                    build_targets.append((vdf, agent.image_for_variant(vname)))
        else:
            # Base image only
            if not agent.dockerfile:
                click.echo(f"Skipping {name}: no Dockerfile defined")
                continue
            base_df = cage_root / agent.dockerfile
            if not base_df.exists():
                click.echo(f"Error: Dockerfile not found at {base_df}")
                raise SystemExit(1)
            build_targets.append((base_df, agent.default_image))

    if not build_targets:
        click.echo("Nothing to build.")
        return

    build_context = cage_root
    for dockerfile_path, image_tag in build_targets:
        click.echo(f"Building {image_tag} from {dockerfile_path.relative_to(cage_root)}...")
        cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_tag,
        ]
        if version:
            # Override every known agent CLI version ARG; the Dockerfile
            # only consumes the one it declares, the rest are ignored.
            for arg in (
                "CLAUDE_CODE_VERSION",
                "CODEX_VERSION",
                "KIMI_CLI_VERSION",
                "QWEN_CODE_VERSION",
            ):
                cmd += ["--build-arg", f"{arg}={version}"]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(str(build_context))

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            click.echo(f"Error: build failed for {image_tag}")
            raise SystemExit(1)
        click.echo(f"  Built: {image_tag}")
        click.echo()


_RELEASE_AGENT_ORDER = ("codex", "claude_code", "qwen_code", "kimi_code")

_AGENT_LIST_HELP: dict[str, tuple[str, str, str]] = {
    "codex": (
        "stable",
        "Codex CLI runner for OpenAI / Responses-compatible models.",
        "Use model ids from config/models.yml; benchmark presets usually select the image.",
    ),
    "claude_code": (
        "stable",
        "Claude Code runner for Anthropic, subscription, and Anthropic-compatible models.",
        "Use model ids from config/models.yml; benchmark presets may add permission flags.",
    ),
    "qwen_code": (
        "stable",
        "Qwen Code runner for OpenAI-compatible Qwen endpoints.",
        "Use model ids from config/models.yml; Cage prepares the CLI config at runtime.",
    ),
    "kimi_code": (
        "stable",
        "Kimi Code runner for OpenAI-compatible Kimi endpoints.",
        "Use model ids from config/models.yml; Cage prepares the CLI config at runtime.",
    ),
    "hermes": (
        "experimental",
        "Hermes runner for Anthropic-compatible local providers.",
        "Not part of the default release benchmark presets; use only from an explicit project.yml.",
    ),
}


def _agent_image_tags(name: str, agent: object) -> list[str]:
    default_image = str(getattr(agent, "default_image", "") or "")
    tags: list[str] = []
    if default_image:
        tags.append(default_image)

    import cage
    cage_root = Path(cage.__file__).resolve().parent.parent
    docker_dir = cage_root / "docker"
    image_for_variant = getattr(agent, "image_for_variant", None)
    if default_image and callable(image_for_variant):
        for dockerfile in sorted(docker_dir.glob(f"{name}_*.Dockerfile")):
            variant = dockerfile.stem.removeprefix(f"{name}_")
            tag = image_for_variant(variant)
            if tag not in tags:
                tags.append(tag)
    return tags


@click.command("list")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Also show experimental or internal agent runtimes.",
)
def agents(show_all: bool) -> None:
    """List available agent types."""
    from cage.agents.base import _AGENT_TYPE_REGISTRY

    order = list(_RELEASE_AGENT_ORDER)
    if show_all:
        order.extend(name for name in sorted(_AGENT_TYPE_REGISTRY) if name not in order)

    click.echo("Agent runtimes")
    if show_all:
        click.echo("Showing stable and experimental agents because --all was used.")
    else:
        click.echo("Showing stable release-facing agents. Use --all for experimental runtimes.")
    click.echo("Use with: cage run <benchmark> --agent <runner> --model <model-id>")
    click.echo(f"Stable --agent values: {', '.join(_RELEASE_AGENT_ORDER)}")
    if show_all:
        experimental = [
            name for name in order
            if name not in _RELEASE_AGENT_ORDER
        ]
        if experimental:
            click.echo(
                "Experimental runtimes shown below may require explicit "
                f"project.yml configuration: {', '.join(experimental)}"
            )
    click.echo("Models come from config/models.yml unless the project preset supplies one.")
    click.echo(
        "Container images are Cage image tags; benchmark/project configs usually "
        "select them automatically."
    )
    click.echo()
    for name in order:
        cls = _AGENT_TYPE_REGISTRY.get(name)
        if cls is None:
            continue
        agent = cls()
        status, summary, config_hint = _AGENT_LIST_HELP.get(
            name,
            (
                "experimental",
                "Agent runtime registered by Cage.",
                "Use only if your project.yml selects this agent kind explicitly.",
            ),
        )
        click.echo(f"{name}  [{status}]")
        click.echo(
            f"  container images: "
            f"{', '.join(_agent_image_tags(name, agent)) or '(none declared)'}"
        )
        click.echo(f"  runner: {summary}")
        click.echo(f"  config: {config_hint}")
        click.echo()


@click.group(name="agent")
def agent_group() -> None:
    """List, build, and debug agent runtimes."""


agent_group.add_command(agents)
agent_group.add_command(build)
agent_group.add_command(debug)
