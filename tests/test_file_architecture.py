from pathlib import Path


def test_batch_one_core_imports_are_removed() -> None:
    """Batch 1 must stop importing base framework concepts from old core."""

    root = Path(__file__).resolve().parents[1]
    old_prefix = ".".join(("cage", "core"))
    forbidden = (
        f"{old_prefix}.agent_type",
        f"{old_prefix}.model",
        f"{old_prefix}.benchmark",
        f"{old_prefix}.scorer",
        f"{old_prefix}.types",
    )
    scanned_roots = (root / "cage", root / "examples", root / "tests")
    offenders: list[str] = []
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(root)} imports {token}")

    assert offenders == []


def test_batch_two_experiment_core_imports_are_removed() -> None:
    """Batch 2 must move experiment and artifact contracts out of old core."""

    root = Path(__file__).resolve().parents[1]
    old_prefix = ".".join(("cage", "core"))
    forbidden = (
        f"{old_prefix}.experiment",
        f"{old_prefix}.experiment_contracts",
        f"{old_prefix}.experiment_artifacts",
        f"{old_prefix}.experiment_legacy_adapter",
        f"{old_prefix}.hooks",
    )
    scanned_roots = (root / "cage", root / "examples", root / "tests")
    offenders: list[str] = []
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(root)} imports {token}")

    assert offenders == []


def test_core_package_is_deleted() -> None:
    """The framework must not keep the old core package as a boundary."""

    root = Path(__file__).resolve().parents[1]
    old_prefix = ".".join(("cage", "core"))
    assert not (root / "cage" / "core").exists()

    scanned_roots = (root / "cage", root / "examples", root / "tests")
    offenders: list[str] = []
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if old_prefix in text:
                offenders.append(f"{path.relative_to(root)} mentions {old_prefix}")

    assert offenders == []


def test_flat_agent_modules_are_removed() -> None:
    """Concrete agents must live in package directories, not flat modules."""

    root = Path(__file__).resolve().parents[1]
    flat_modules = (
        "claude_code.py",
        "codex.py",
        "codex_output.py",
        "qwen_code.py",
        "kimi_code.py",
        "hermes.py",
    )
    offenders = [
        str((root / "cage" / "agents" / module).relative_to(root))
        for module in flat_modules
        if (root / "cage" / "agents" / module).exists()
    ]
    assert offenders == []


def test_cli_uses_agent_package_registration() -> None:
    """CLI startup should register agents through the agents package."""

    root = Path(__file__).resolve().parents[1]
    cli_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden_imports = (
        "import cage.agents.claude_code",
        "import cage.agents.codex",
        "import cage.agents.hermes",
        "import cage.agents.kimi_code",
        "import cage.agents.qwen_code",
    )
    offenders = [token for token in forbidden_imports if token in cli_text]
    assert offenders == []


def test_cli_modules_live_under_cli_package() -> None:
    """CLI and terminal UI modules must not live at the package root.

    The ANSI style primitives are an exception: they are dependency-free and
    shared by the runtime preflight gate, so they live on the layer-0 floor at
    ``cage.contracts.style`` rather than under the CLI.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden_root_files = (
        root / "cage" / "cli.py",
        root / "cage" / "run_ui.py",
        root / "cage" / "terminal_style.py",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_root_files
        if path.exists()
    ]
    assert offenders == []
    assert (root / "cage" / "cli" / "main.py").is_file()
    assert (root / "cage" / "cli" / "ui" / "run.py").is_file()
    assert not (root / "cage" / "cli" / "ui" / "style.py").exists()
    assert (root / "cage" / "contracts" / "style.py").is_file()


def test_benchmark_cli_commands_live_under_commands_package() -> None:
    """The benchmark command group should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "benchmark.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "@main.group(name=\"benchmark\")",
        "def benchmark_group",
        "def benchmark_list",
        "def benchmark_show",
        "def benchmark_check",
        "def benchmark_build",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_model_cli_commands_live_under_commands_package() -> None:
    """The model command group should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "model.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "@main.group(name=\"model\")",
        "def model_group",
        "def model_list",
        "def model_show",
        "def model_set",
        "def _load_model_registry_yaml",
        "def _write_model_registry_yaml",
        "def _ensure_model_entry_shape",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_target_hidden_cli_wrappers_are_removed() -> None:
    """Target workflows should live in targets, not hidden Click wrappers."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert not (commands_dir / "target.py").exists()
    assert (root / "cage" / "target" / "check.py").is_file()
    assert (root / "cage" / "target" / "debug.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "target_commands",
        "@main.command(\"targets-check\"",
        "@main.command(\"target-debug\"",
        "def targets_check",
        "def target_debug",
        "def _run_targets_check",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_inspect_cli_commands_live_under_commands_package() -> None:
    """The inspect command group should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "inspect.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "class InspectGroup",
        "def inspect(",
        "def _serve_inspector",
        "def inspect_serve",
        "def inspect_start",
        "def inspect_status",
        "def inspect_stop",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []

    inspect_text = (commands_dir / "inspect.py").read_text(encoding="utf-8")
    assert '@inspect.command("serve"' not in inspect_text
    assert "def inspect_serve" not in inspect_text


def test_agent_cli_commands_live_under_commands_package() -> None:
    """Agent runtime commands should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "agent.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "def debug(",
        "def build(",
        "def agents(",
        "def agent_group",
        "def _agent_image_tags",
        "def _visible_command_alias",
        "_RELEASE_AGENT_ORDER",
        "_AGENT_LIST_HELP",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_gc_cli_commands_live_under_commands_package() -> None:
    """GC commands should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "gc.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "def cleanup(",
        "def gc(",
        "from cage.gc.runner import",
        "from cage.target.local_cleanup import sweep_run",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_target_server_runner_is_not_a_click_command() -> None:
    """Internal target-server startup should not be hidden in the Click tree."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert not (commands_dir / "serve.py").exists()
    assert (root / "cage" / "target" / "serve.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "serve_commands",
        "def serve(",
        "TARGET_SERVER_BENCHMARK_SOURCES_JSON",
        "TARGET_SERVER_ADAPTER_MODULES",
        "TARGET_SERVER_EXTERNAL_TOKEN",
        "uvicorn.run(",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_dashboard_hidden_cli_wrappers_are_removed() -> None:
    """Dashboard projections should not be exposed as hidden Click commands."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert not (commands_dir / "dashboard.py").exists()
    # The standalone dashboard/ package is dissolved: the projection view-model
    # and readers are an artifact schema (artifacts/), and the writers are
    # run-finalization (experiment/engine/reporting.py).
    assert not (root / "cage" / "dashboard").exists()
    assert (root / "cage" / "artifacts" / "dashboard.py").is_file()
    assert (root / "cage" / "experiment" / "engine" / "reporting.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "dashboard_commands",
        "def show(",
        "def dashboard(",
        "def _load_show_dashboard",
        "def _show_dashboard_from_canonical_record",
        "def _show_scores_from_trial_record",
        "def _show_flatten_score_payload",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_check_hidden_cli_wrapper_is_removed() -> None:
    """Project checks should be public benchmark checks, not hidden commands."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert not (commands_dir / "check.py").exists()
    assert (commands_dir / "benchmark.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "check_commands",
        "def check(",
        "check_benchmark(",
        "discover_template_source",
        "keep-services",
        "strict-exit",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_score_cli_command_lives_under_commands_package() -> None:
    """The score command should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "score.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "def score(",
        "def _score_output_path_for_context",
        "def _find_project_run_dirs",
        "def _run_dir_project_candidates",
        "run.score_summary",
        "load_scorer_from_module",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_run_cli_command_lives_under_commands_package() -> None:
    """The run command should not be implemented in cli/main.py."""

    root = Path(__file__).resolve().parents[1]
    commands_dir = root / "cage" / "cli" / "commands"
    assert (commands_dir / "__init__.py").is_file()
    assert (commands_dir / "run.py").is_file()

    main_text = (root / "cage" / "cli" / "main.py").read_text(encoding="utf-8")
    forbidden = (
        "def run(",
        "def _print_resume_dry_run",
        "def _print_experiment_plan_dry_run",
        "def _normalize_explicit_sample_ids",
        "prepare_project_for_run(",
        "run_experiment(",
    )
    offenders = [token for token in forbidden if token in main_text]
    assert offenders == []


def test_runtime_live_modules_live_under_live_package() -> None:
    """Live runtime checks should live in ``cage.experiment.engine.live``.

    The live-success *verdict file* helper is persistence, not a live check, so
    it lives in ``cage.artifacts.live_success`` (layer 1) where both the engine
    monitor and the scorer import it downward — the genuine live-check modules
    (monitor / liveness / fs_signals) live under ``experiment.engine.live``.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden_files = (
        root / "cage" / "runtime" / "live_monitor.py",
        root / "cage" / "runtime" / "live_success.py",
        root / "cage" / "runtime" / "liveness.py",
        root / "cage" / "runtime" / "fs_signals.py",
        root / "cage" / "runtime" / "live",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_files
        if path.exists()
    ]
    assert offenders == []
    live = root / "cage" / "experiment" / "engine" / "live"
    assert (live / "monitor.py").is_file()
    assert (live / "liveness.py").is_file()
    assert (live / "fs_signals.py").is_file()
    assert not (live / "success.py").exists()
    assert (root / "cage" / "artifacts" / "live_success.py").is_file()


def test_target_services_live_under_target_services_package() -> None:
    """Check and submit services are target-side and live under the target package.

    They no longer live under ``cage.runtime`` (the runtime junk-drawer is being
    dissolved); they sit next to the rest of the target server.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden_files = (
        root / "cage" / "runtime" / "check_server.py",
        root / "cage" / "runtime" / "check_service.py",
        root / "cage" / "runtime" / "submit_client.py",
        root / "cage" / "runtime" / "submit_server.py",
        root / "cage" / "runtime" / "submit_service.py",
        root / "cage" / "runtime" / "services",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_files
        if path.exists()
    ]
    assert offenders == []
    services = root / "cage" / "target" / "services"
    assert (services / "check" / "server.py").is_file()
    assert (services / "check" / "service.py").is_file()
    assert (services / "submit" / "client.py").is_file()
    assert (services / "submit" / "server.py").is_file()
    assert (services / "submit" / "service.py").is_file()


def test_container_primitives_live_under_sandbox_package() -> None:
    """Container primitives belong to the ``sandbox`` substrate package.

    They use plural domain names (``containers``/``shell``), not the old
    single-file names, and they no longer live under ``cage.runtime`` — the
    runtime junk-drawer is being dissolved by domain.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden_files = (
        root / "cage" / "sandbox" / "container.py",
        root / "cage" / "sandbox" / "persistent_shell.py",
        root / "cage" / "runtime" / "containers.py",
        root / "cage" / "runtime" / "shell.py",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_files
        if path.exists()
    ]
    assert offenders == []
    assert (root / "cage" / "sandbox" / "containers.py").is_file()
    assert (root / "cage" / "sandbox" / "shell.py").is_file()


def test_proxy_modules_live_under_proxy_package() -> None:
    """Proxy host and sidecar code should not live under ``cage.runtime``."""

    root = Path(__file__).resolve().parents[1]
    forbidden_runtime_files = (
        root / "cage" / "runtime" / "proxy.py",
        root / "cage" / "runtime" / "container_proxy.py",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_runtime_files
        if path.exists()
    ]
    assert offenders == []
    assert (root / "cage" / "proxy" / "host.py").is_file()
    assert (root / "cage" / "proxy" / "sidecar.py").is_file()


def test_proxy_usage_and_trajectory_helpers_live_under_proxy_package() -> None:
    """Proxy-derived usage and trajectory helpers should not live at package root."""

    root = Path(__file__).resolve().parents[1]
    forbidden_root_files = (
        root / "cage" / "token_usage.py",
        root / "cage" / "traj.py",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_root_files
        if path.exists()
    ]
    assert offenders == []
    assert (root / "cage" / "proxy" / "usage.py").is_file()
    assert (root / "cage" / "proxy" / "trajectory.py").is_file()


def test_run_storage_lives_under_artifacts_package() -> None:
    """Run-directory storage belongs to durable artifacts, not the package root."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "storage.py").exists()
    assert (root / "cage" / "artifacts" / "run_storage.py").is_file()


def test_project_overlays_live_under_engine_package() -> None:
    """Project override merging is part of the experiment engine."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "project_overlays.py").exists()
    assert not (root / "cage" / "experiments" / "overlays.py").exists()
    assert (root / "cage" / "experiment" / "engine" / "overlays.py").is_file()


def test_benchmark_registry_lives_under_benchmarks_package() -> None:
    """Benchmark discovery and default project lookup belong to benchmarks."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "benchmark_registry.py").exists()
    assert (root / "cage" / "benchmarks" / "registry.py").is_file()


def test_offline_gc_lives_under_gc_package() -> None:
    """Offline resource garbage collection belongs to the cage.gc package.

    The package is named ``gc`` (matching the ``cage gc`` command), distinct
    from in-run teardown (``runtime.run_cleanup``) and the target server's local
    sweep (``target_server.local_cleanup``), so "cleanup" no longer names three
    different concepts.
    """

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "gc.py").exists()
    assert not (root / "cage" / "cleanup").exists()
    assert (root / "cage" / "gc" / "runner.py").is_file()


def test_inspect_board_lives_under_web_package() -> None:
    """Managed inspector board helpers belong beside the web inspector."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "inspect_board.py").exists()
    assert (root / "cage" / "web" / "inspect_board.py").is_file()


def test_target_commands_live_under_target_package() -> None:
    """Target build, check, and debug workflows belong to the target package.

    targets/ and target_server/ are merged into one ``cage.target`` package —
    there is no longer a plural ``targets`` package or a separate target server.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden_files = (
        root / "cage" / "target_build.py",
        root / "cage" / "targets_check.py",
        root / "cage" / "target_debug.py",
        root / "cage" / "targets",
        root / "cage" / "target_server",
    )
    offenders = [
        str(path.relative_to(root))
        for path in forbidden_files
        if path.exists()
    ]
    assert offenders == []
    assert (root / "cage" / "target" / "__init__.py").is_file()
    assert (root / "cage" / "target" / "build.py").is_file()
    assert (root / "cage" / "target" / "check.py").is_file()
    assert (root / "cage" / "target" / "debug.py").is_file()
    assert (root / "cage" / "target" / "provisioning.py").is_file()


def test_preflight_lives_under_engine_package() -> None:
    """Preflight checks start containers and belong to the experiment engine."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "preflight.py").exists()
    assert not (root / "cage" / "runtime" / "preflight.py").exists()
    assert (root / "cage" / "experiment" / "engine" / "preflight.py").is_file()


def test_web_data_lives_in_data_package() -> None:
    """Inspector data loading should be a package that can split by view."""

    root = Path(__file__).resolve().parents[1]
    assert not (root / "cage" / "web" / "data.py").exists()
    assert (root / "cage" / "web" / "data" / "__init__.py").is_file()


def test_framework_source_never_names_a_benchmark() -> None:
    """The CLAUDE.md grep test, machine-enforced — including comments.

    Layer 1 must know zero benchmark names. ``pentest_remote`` in
    ``target/check_probes.py`` is NOT a benchmark name: it is a declarative
    check-profile key that datasets select via ``challenge.json``, so the
    token list below deliberately uses the benchmark directory names only.
    """

    root = Path(__file__).resolve().parents[1]
    tokens = (
        "cvebench",
        "autopenbench",
        "agent_pentest_bench",
        "pentest_bench",
        "strongreject",
        "hackworld",
        "ctfbench",
        "cybergym",
        "nyu",
    )
    offenders: list[str] = []
    for path in sorted((root / "cage").rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for token in tokens:
            if token in text:
                offenders.append(f"{path.relative_to(root)}: {token}")
    assert offenders == []
