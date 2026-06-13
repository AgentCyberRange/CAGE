"""Run-surface display shared by ``cage run`` and the benchmark commands.

Echo helpers for the no-args landing page and the per-benchmark run surface
(agent/model matrix, runtime defaults, sample-id notes). Private to
``cage.cli`` — commands import the public names below.
"""

from __future__ import annotations

import click

from cage.benchmarks import BenchmarkOption
from cage.benchmarks import registry
from cage.benchmarks.registry import UnknownBenchmarkError
from cage.cli.paths import display_path as _display_path


def echo_benchmark_list(*, include_footer: bool = False) -> None:
    click.echo("ID                  Display Name        Default project")
    click.echo("------------------  ------------------  ----------------")
    for spec in registry.list_benchmarks():
        click.echo(
            f"{spec.id:18s}  {spec.display_name:18s}  "
            f"{_display_path(spec.resolved_project_file)}"
        )
    if include_footer:
        click.echo()
        click.echo("Run help: cage run <id> --help")
        click.echo("Run by ID: cage run <id> ...")
        click.echo("Customize by file: cage run <default-project.yml> ...")

def echo_run_landing_help() -> None:
    click.echo("CAGE RUN")
    click.echo()
    click.echo("Usage: cage run PROJECT_OR_BENCHMARK [options]")
    click.echo()
    click.echo(
        "Run is the main Cage workflow. Pick a benchmark, inspect its run "
        "surface, check/build benchmark targets, then launch or resume an "
        "evaluation."
    )
    click.echo()
    click.echo("Step 1. Choose a benchmark")
    click.echo()
    echo_benchmark_list(include_footer=False)
    click.echo()
    click.echo("Step 2. Inspect benchmark-specific run help")
    click.echo()
    click.echo("  cage run <id> --help")
    click.echo()
    click.echo(
        "This shows sample IDs, prompt/hint levels, agents, models, runtime "
        "defaults, target settings, benchmark-owned options, and example run "
        "commands."
    )
    click.echo()
    click.echo("Step 3. Check and build benchmark targets")
    click.echo()
    click.echo("  cage benchmark check <id> --sample <sample_id>")
    click.echo("  cage benchmark check <id> --sample <sample_id> --show-prompt")
    click.echo("  cage benchmark build <id> --sample <sample_id>")
    click.echo("  cage benchmark build <id> --max-concurrent 4")
    click.echo()
    click.echo(
        "Use check to load config and render prompts without launching targets. "
        "Use build to prepare benchmark-owned target images/assets before agents "
        "spend model calls."
    )
    click.echo()
    click.echo("Step 4. Run one smoke trial")
    click.echo()
    click.echo(
        "  cage run <id> --agent <agent> --model <model-id> "
        "--sample <sample_id> --passk 1 --max-concurrent 1 "
        "--run-id smoke-001"
    )
    click.echo()
    click.echo("Step 5. Scale up or resume")
    click.echo()
    click.echo("  cage run <id> --agent <agent> --run-id full-001")
    click.echo("  cage run <id> --run-id full-001 --resume --dry-run")
    click.echo("  cage run <id> --run-id full-001 --resume")
    click.echo()
    click.echo("Common run options:")
    click.echo("  --agent ID              Run only this configured agent (repeatable).")
    click.echo("  --model ID              Override the model for one selected agent.")
    click.echo("  --model-source ID       Rotate one run across these registered models (repeatable; needs --model as the key).")
    click.echo("  --wait-for-model        Poll model endpoints until they boot before starting (--wait-timeout/--wait-interval).")
    click.echo("  --sample ID             Run specific benchmark sample IDs (repeatable).")
    click.echo("  --sample-slice SPEC     Run a Python-style slice of samples, e.g. :100, -100:-1, 100:200, ::2.")
    click.echo("  --max-sample-num N      Use only the first N selected benchmark samples.")
    click.echo("  --max-trial-num N       Run only the first N expanded trials this invocation.")
    click.echo("  --max-concurrent N      Cap one agent; with --resume, cap all if no --agent.")
    click.echo("  --passk N               Run N independent attempts per sample.")
    click.echo("  --run-id ID             Set the run directory id.")
    click.echo("  --resume                Continue an existing run id.")
    click.echo("  --force                 Archive an existing run id and start fresh.")
    click.echo("  --dry-run               Show the plan without launching containers or targets.")
    click.echo("  --allow-launch-build    Run benchmark build hook before target launch.")
    click.echo("  --timeout SECONDS       Override per-trial timeout.")
    click.echo("  --max-rounds N          Override max agent turns/rounds.")
    click.echo("  --max-input-tokens N    Stop after this input-token budget.")
    click.echo("  --max-output-tokens N   Stop after this output-token budget.")
    click.echo("  --max-cost USD          Stop after this model-cost budget.")
    click.echo("  --upstream-proxy URL    Route model traffic through an HTTP proxy.")
    click.echo("  --set PATH=VALUE        Override a project.yml field.")
    click.echo()
    click.echo("Benchmark-owned options are shown by `cage run <id> --help`.")

def _echo_mapping_section(
    title: str,
    raw: object,
    keys: tuple[str, ...],
) -> None:
    if not isinstance(raw, dict):
        return
    present = [(key, raw[key]) for key in keys if key in raw]
    if not present:
        return
    click.echo()
    click.echo(f"{title}:")
    for key, value in present:
        click.echo(f"  {key}: {value}")

def _echo_runtime_defaults(raw: object, *, defer_label: str | None = None) -> None:
    if not isinstance(raw, dict):
        return
    keys = (
        "timeout",
        "max_trials_global",
        "max_target_setups",
        "passk",
        "max_rounds",
        "max_input_tokens",
        "max_output_tokens",
        "max_cost",
        "agent_network_mode",
        "store_proxy",
    )
    present = [(key, raw[key]) for key in keys if key in raw]
    if not present:
        return
    click.echo()
    click.echo("Runtime defaults:")
    for key, value in present:
        if key == "max_rounds" and str(value) == "-1":
            if defer_label:
                value = f"{defer_label} (benchmark default; project.yml sets max_rounds: -1 to defer)"
            else:
                value = "-1 (defers to benchmark/sample default)"
        elif key == "max_rounds" and str(value) == "0":
            value = "0 (no model-call rounds)"
        elif key == "max_trials_global" and str(value) in ("0", "None", ""):
            value = "unlimited"
        if (
            key in {"max_input_tokens", "max_output_tokens", "max_cost"}
            and value in (None, "")
        ):
            value = "unset (unlimited)"
        click.echo(f"  {key}: {value}")
    click.echo(
        "  Concurrency: runtime.max_trials_global caps in-flight trials across "
        "the whole run (default unlimited); use --max-concurrent to cap one "
        "selected agent/model from the CLI."
    )

def _model_id_and_cap(item: object) -> tuple[str, object | None]:
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        model_id = str(item.get("id") or item.get("model") or "").strip()
        return model_id, item.get("max_concurrent")
    return "", None

def _agent_model_rows(raw: dict[str, object]) -> list[tuple[str, str, str]]:
    matrix = registry.project_agent_model_matrix(raw)
    agent_entries = [
        agent for agent in raw.get("agents", []) or []
        if isinstance(agent, dict)
    ]
    rows: list[tuple[str, str, str]] = []
    for agent_id, models in matrix:
        entry = next(
            (
                agent for agent in agent_entries
                if str(
                    agent.get("id")
                    or agent.get("kind")
                    or agent.get("agent_type")
                    or ""
                ) == agent_id
            ),
            {},
        )
        cap = entry.get("max_concurrent") if isinstance(entry, dict) else None
        model_caps: list[str] = []
        if isinstance(entry, dict) and isinstance(entry.get("models"), list):
            for item in entry["models"]:
                model_id, model_cap = _model_id_and_cap(item)
                if model_id and model_cap is not None:
                    model_caps.append(f"{model_id}={model_cap}")
        meta_parts: list[str] = []
        if cap is not None:
            meta_parts.append(f"max_concurrent: {cap}")
        if model_caps:
            meta_parts.append(f"model caps: {', '.join(model_caps)}")
        meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
        model_text = ", ".join(models) if models else "(default model registry order)"
        rows.append((agent_id, meta, model_text))
    return rows

def _benchmark_level_option(options: list[BenchmarkOption]) -> str:
    flags = {option.flag for option in options}
    if "--prompt-level" in flags:
        return "--prompt-level"
    if "--hint-level" in flags:
        return "--hint-level"
    return ""

def _benchmark_level_default(options: list[BenchmarkOption]) -> str:
    flag = _benchmark_level_option(options)
    if flag == "--prompt-level":
        return "l0"
    if flag == "--hint-level":
        return "0"
    return ""

def _echo_sample_id_note(options: list[BenchmarkOption]) -> None:
    flag = _benchmark_level_option(options)
    if flag == "--prompt-level":
        click.echo(
            "Sample IDs: prompt-level sweeps expand base tasks into "
            "<challenge_id>-l0/-l1/-l2; pass --prompt-level to narrow."
        )
        click.echo(
            "You may pass the base challenge id or an expanded level id; "
            "target builds collapse level ids to the underlying target."
        )
    elif flag == "--hint-level":
        click.echo(
            "Sample IDs: hint-level sweeps expand base tasks into "
            "<challenge_id>-l0/-l1/-l2; pass --hint-level to narrow."
        )
        click.echo(
            "You may pass the base challenge id or an expanded level id; "
            "target builds collapse level ids to the underlying target."
        )

def echo_benchmark_run_surface(benchmark_id: str) -> None:
    try:
        spec = registry.resolve_benchmark(benchmark_id)
    except UnknownBenchmarkError as exc:
        raise click.UsageError(str(exc)) from exc
    project_file = spec.resolved_project_file.resolve()
    raw = registry.load_project_yaml(project_file)
    options = registry.load_benchmark_options(project_file)
    project_name = raw.get("project", {}).get("name", "") if isinstance(raw.get("project"), dict) else ""
    benchmark_root = registry.project_benchmark_root(project_file, raw)

    click.echo(f"{spec.display_name} ({spec.id})")
    click.echo(f"Benchmark ID: {spec.id} (use with cage run)")
    click.echo(spec.description)
    click.echo(f"Default project: {_display_path(project_file)}")
    if project_name:
        click.echo(f"Project name: {project_name} (used in run metadata/artifacts)")
    if benchmark_root is not None:
        click.echo(f"Benchmark root: {_display_path(benchmark_root)}")
    click.echo(f"Samples: {registry.project_sample_summary(project_file, raw)}")
    _echo_sample_id_note(options)
    defer_label = registry.project_round_budget_default_label(project_file, raw)
    _echo_runtime_defaults(raw.get("runtime", {}), defer_label=defer_label)
    _echo_mapping_section("Target defaults", raw.get("target", {}), (
        "enabled",
        "run_mode",
        "startup_timeout",
        "compose_up_timeout",
        "target_scope",
        "parallel_mode",
    ))
    agent_rows = _agent_model_rows(raw)
    if agent_rows:
        click.echo()
        click.echo("Agents / models:")
        for agent_id, meta, model_text in agent_rows:
            click.echo(f"  {agent_id}{meta}: {model_text}")
        click.echo("  If --model is omitted, the selected agent runs all listed models.")
        click.echo("  If --agent is omitted, all configured agent/model pairs run.")
    if options:
        click.echo()
        click.echo("Benchmark options:")
        for option in options:
            choices = f" [{'|'.join(option.choices)}]" if option.choices else ""
            repeatable = " (repeatable, comma-allowed)" if option.multiple else ""
            help_text = f" - {option.help}" if option.help else ""
            click.echo(
                f"  {option.flag}{choices}{repeatable} -> {option.config_path}{help_text}"
            )
    click.echo()
    level_flag = _benchmark_level_option(options)
    level_default = _benchmark_level_default(options)
    level_part = f" {level_flag} {level_default}" if level_flag and level_default else ""
    click.echo("Recommended workflow:")
    click.echo()
    click.echo("Step 1. Check config and prompt rendering")
    click.echo(f"  cage benchmark check {spec.id} --sample <sample_id>")
    click.echo(f"  cage benchmark check {spec.id} --sample <sample_id> --show-prompt")
    click.echo()
    click.echo("Step 2. Build or verify benchmark targets")
    click.echo(f"  cage benchmark build {spec.id} --sample <sample_id>")
    click.echo(f"  cage benchmark build {spec.id} --max-concurrent 4")
    click.echo()
    click.echo("Step 3. Run one smoke trial")
    click.echo(
        f"  cage run {spec.id} --agent <agent> --model <model-id> "
        f"--sample <sample_id>{level_part} --passk 1 --max-concurrent 1 "
        f"--run-id smoke-{spec.id}-001"
    )
    click.echo()
    click.echo("Step 4. Scale up or resume")
    click.echo(f"  cage run {spec.id} --agent <agent> --run-id full-{spec.id}-001")
    click.echo(f"  cage run {spec.id} --run-id full-{spec.id}-001 --resume --dry-run")
    click.echo(f"  cage run {spec.id} --run-id full-{spec.id}-001 --resume")
    click.echo()
    click.echo("Customize by file:")
    click.echo(f"  cage run {_display_path(project_file)} ...")
