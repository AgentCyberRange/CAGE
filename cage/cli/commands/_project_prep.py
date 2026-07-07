"""Project preparation shared by ``cage run`` and the benchmark commands.

CLI-side plumbing that turns (project / benchmark id, CLI overrides) into an
effective project file plus the per-run agent and sample restrictions. Private
to ``cage.cli`` — commands import the public names below; nothing outside the
CLI package should.
"""

from __future__ import annotations

from pathlib import Path

import click

from cage.benchmarks import BenchmarkOption, registry, sample_id_matches
from cage.benchmarks.registry import UnknownBenchmarkError
from cage.cli.ids import split_cli_ids as _split_cli_ids
from cage.experiment.engine.overlays import (
    apply_set_expressions,
    clone_project,
    materialize_effective_project,
    merge_selected_agent_params,
    override_selected_agent_field,
    override_selected_agent_model,
    parse_set_expression,
    set_project_path,
)


def _coerce_benchmark_option_value(option: BenchmarkOption, raw: str) -> object:
    if option.choices and raw not in option.choices:
        choices = ", ".join(option.choices)
        raise click.UsageError(
            f"{option.flag}: expected one of [{choices}], got {raw!r}"
        )
    if option.value_type == "int":
        try:
            return int(raw)
        except ValueError as exc:
            raise click.UsageError(f"{option.flag}: expected an integer") from exc
    if option.value_type == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise click.UsageError(f"{option.flag}: expected a number") from exc
    if option.value_type == "bool":
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise click.UsageError(f"{option.flag}: expected true or false")
    return raw

def _parse_benchmark_run_args(
    args: list[str],
    options: list[BenchmarkOption],
) -> list[tuple[str, object]]:
    """Parse benchmark-owned args captured by Click's unknown-option mode."""
    if not args:
        return []
    by_flag = {option.flag: option for option in options}
    multi_values: dict[str, list[object]] = {}
    overlays: list[tuple[str, object]] = []

    i = 0
    while i < len(args):
        token = args[i]
        option = by_flag.get(token)
        if option is None:
            available = ", ".join(sorted(by_flag)) or "(none)"
            raise click.UsageError(
                f"Unknown run option {token!r}. Benchmark options: {available}"
            )
        if i + 1 >= len(args):
            raise click.UsageError(f"{token} requires a value")
        raw_values = [args[i + 1]]
        if option.multiple:
            raw_values = [
                part.strip() for part in args[i + 1].split(",") if part.strip()
            ]
        values = [
            _coerce_benchmark_option_value(option, raw_value)
            for raw_value in raw_values
        ]
        if option.multiple:
            multi_values.setdefault(option.config_path, []).extend(values)
        else:
            overlays.append((option.config_path, values[0]))
        i += 2

    for path, values in multi_values.items():
        overlays.append((path, values))
    return overlays

def _resolve_project_or_benchmark(value: str) -> tuple[Path, str]:
    """Return ``(project_file, benchmark_id)`` for a path or registered id."""
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve(), ""

    looks_like_path = (
        "/" in value
        or "\\" in value
        or candidate.suffix in {".yml", ".yaml"}
    )
    if looks_like_path:
        raise click.UsageError(f"Project file not found: {value}")

    try:
        spec = registry.resolve_benchmark(value)
    except UnknownBenchmarkError as exc:
        raise click.UsageError(str(exc)) from exc
    return spec.resolved_project_file.resolve(), spec.id


def benchmark_source_dir(benchmark_id: str) -> Path:
    """Directory holding a registered benchmark's project.yml (examples/<id>/).

    Run output (``.cage_runs``) belongs next to the benchmark's project.yml —
    same as running that project.yml path directly — not under the cwd the
    operator happened to launch from. Using the registered source dir keeps
    ``cage run <id>``, ``cage score <id>`` and ``cage benchmark check <id>``
    all pointed at the same per-benchmark run tree.
    """
    return registry.resolve_benchmark(benchmark_id).resolved_project_file.resolve().parent


def prepare_project_for_run(
    project_or_benchmark: str,
    *,
    extra_args: list[str],
    agent_ids: tuple[str, ...],
    resume: bool,
    models_file: str | None,
    model_id: str,
    model_sources: tuple[str, ...] = (),
    upstream_http_proxy: str,
    timeout: float | None,
    max_trials_global: int | None,
    max_concurrent: int | None,
    passk: int | None,
    max_rounds: int | str | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
    max_cost: float | None,
    set_values: tuple[str, ...],
    param_values: tuple[str, ...] = (),
) -> tuple[Path, Path | None, str]:
    """Resolve and materialize the effective project for a run.

    Returns ``(project_file, temp_project_file, benchmark_id)``. The caller
    owns deleting ``temp_project_file`` after the run.
    """
    source_project, benchmark_id = _resolve_project_or_benchmark(project_or_benchmark)
    raw = clone_project(registry.load_project_yaml(source_project))
    changed = False

    if models_file:
        set_project_path(raw, "models_file", str(Path(models_file).expanduser().resolve()))
        changed = True
    if upstream_http_proxy:
        set_project_path(raw, "proxy.upstream_http_proxy", upstream_http_proxy)
        changed = True
    if timeout is not None:
        set_project_path(raw, "runtime.timeout", timeout)
        changed = True
    if max_trials_global is not None:
        set_project_path(raw, "runtime.max_trials_global", max_trials_global)
        changed = True
    if max_concurrent is not None:
        if resume and not agent_ids:
            _override_all_agents_field(
                raw,
                field="max_concurrent",
                value=max_concurrent,
                flag="--max-concurrent",
            )
        else:
            try:
                override_selected_agent_field(
                    raw,
                    agent_ids=agent_ids,
                    field="max_concurrent",
                    value=max_concurrent,
                    flag="--max-concurrent",
                )
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc
        changed = True
    if passk is not None:
        set_project_path(raw, "runtime.passk", passk)
        changed = True
    if max_rounds is not None:
        set_project_path(raw, "runtime.max_rounds", max_rounds)
        changed = True
    if max_input_tokens is not None:
        set_project_path(raw, "runtime.max_input_tokens", max_input_tokens)
        changed = True
    if max_output_tokens is not None:
        set_project_path(raw, "runtime.max_output_tokens", max_output_tokens)
        changed = True
    if max_cost is not None:
        set_project_path(raw, "runtime.max_cost", max_cost)
        changed = True

    if extra_args:
        options = registry.load_benchmark_options(source_project)
        for path, value in _parse_benchmark_run_args(extra_args, options):
            set_project_path(raw, path, value)
            changed = True

    if set_values:
        try:
            apply_set_expressions(raw, set_values)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        changed = True

    if param_values:
        params: dict[str, object] = {}
        for item in param_values:
            try:
                key, value = parse_set_expression(item)
            except ValueError as exc:
                raise click.UsageError(f"--param: {exc}") from exc
            params[key] = value
        try:
            merge_selected_agent_params(
                raw, agent_ids=agent_ids, params=params, flag="--param",
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        changed = True

    if model_id or model_sources:
        try:
            override_selected_agent_model(
                raw,
                agent_ids=agent_ids,
                model_id=model_id,
                sources=model_sources,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        changed = True

    if _filter_project_agents(raw, agent_ids, source_label=project_or_benchmark):
        changed = True

    if not changed:
        return source_project, None, benchmark_id
    effective = materialize_effective_project(source_project, raw)
    return effective, effective, benchmark_id

def _override_all_agents_field(
    raw: dict[str, object],
    *,
    field: str,
    value: object,
    flag: str,
) -> None:
    agents = raw.get("agents", []) or []
    if not isinstance(agents, list):
        raise click.UsageError(f"agents must be a list before {flag} can be applied")
    count = 0
    for agent in agents:
        if isinstance(agent, dict):
            agent[field] = value
            count += 1
    if count == 0:
        raise click.UsageError(f"Cannot apply {flag} {value!r}: project has no agents")

def _filter_project_agents(
    raw: dict[str, object],
    agent_ids: tuple[str, ...],
    *,
    source_label: str,
) -> bool:
    requested = {str(agent_id) for agent_id in agent_ids if str(agent_id)}
    if not requested:
        return False

    agents = raw.get("agents", []) or []
    if not isinstance(agents, list):
        raise click.UsageError("agents must be a list before --agent can be applied")

    kept: list[object] = []
    matched: set[str] = set()
    available: set[str] = set()
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = str(agent.get("id") or "")
        if agent_id:
            available.add(agent_id)
        if agent_id in requested:
            kept.append(agent)
            matched.add(agent_id)

    if not kept:
        raise click.UsageError(
            f"--agent {sorted(requested)} matched no agent in {source_label}. "
            f"Available: {sorted(available)}"
        )
    missing = requested - matched
    if missing:
        raise click.UsageError(
            f"--agent: unknown id(s) {sorted(missing)}. Available: "
            f"{sorted(available)}"
        )
    raw["agents"] = kept
    return True

def run_benchmark_image_build(
    project_file: Path,
    *,
    limit: int | None,
    only: tuple[str, ...],
    max_workers: int,
    dry_run: bool,
    rebuild: bool = False,
) -> None:
    from cage.target.build import build_benchmark_targets, print_build_summary

    try:
        summary = build_benchmark_targets(
            Path(project_file),
            limit=limit,
            only=_split_cli_ids(only) or None,
            max_workers=max_workers,
            dry_run=dry_run,
            rebuild=rebuild,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    print_build_summary(summary, compact=dry_run)
    if summary.failed:
        raise SystemExit(1)

def restrict_config_agents(
    config: object,
    agent_ids: tuple[str, ...],
    *,
    source_label: str,
) -> None:
    if not agent_ids:
        return
    requested = set(agent_ids)
    agents = list(getattr(config, "agents", []))
    kept = [agent for agent in agents if getattr(agent, "id", "") in requested]
    available = sorted({str(getattr(agent, "id", "")) for agent in agents})
    if not kept:
        raise click.UsageError(
            f"--agent {sorted(requested)} matched no agent in {source_label}. "
            f"Available: {available}"
        )
    missing = requested - {str(getattr(agent, "id", "")) for agent in kept}
    if missing:
        raise click.UsageError(
            f"--agent: unknown id(s) {sorted(missing)}. Available: {available}"
        )
    config.agents = kept

def resolve_sample_ids_for_benchmark(
    benchmark: object,
    sample_ids: tuple[str, ...],
) -> list[str]:
    requested = [str(item).strip() for item in sample_ids if str(item).strip()]
    if not requested:
        return []

    samples = list(benchmark.iter_samples_limited(None))
    missing = [
        sample_id for sample_id in requested
        if not any(sample_id_matches(sample, [sample_id]) for sample in samples)
    ]
    if missing:
        available = sorted(str(sample.get("id") or sample.get("challenge_id") or "") for sample in samples)
        raise click.UsageError(
            f"--sample id(s) not found: {missing}. Available samples: {available}"
        )

    resolved: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        if not sample_id_matches(sample, requested):
            continue
        sample_id = str(sample.get("id") or sample.get("challenge_id") or "").strip()
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            resolved.append(sample_id)
    return resolved
