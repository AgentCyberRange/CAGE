"""Benchmark-related Cage CLI commands and run-surface helpers."""

from __future__ import annotations

from pathlib import Path

import click

from cage.benchmarks import sample_id_matches
from cage.benchmarks import registry
from cage.benchmarks.registry import UnknownBenchmarkError
from cage.cli.ids import split_cli_ids as _split_cli_ids
from cage.cli.paths import display_path as _display_path
from cage.cli.commands import _project_prep as project_prep
from cage.cli.commands import _run_surface as run_surface


def _split_prompt_levels(values: tuple[str, ...]) -> list[str]:
    levels: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            level = part.strip()
            if level:
                levels.append(level)
    return levels


def _apply_prompt_levels(benchmark: object, levels: tuple[str, ...]) -> None:
    parsed = _split_prompt_levels(levels)
    if not parsed:
        return
    setter = getattr(benchmark, "set_prompt_levels", None)
    if not callable(setter):
        name = getattr(benchmark, "name", type(benchmark).__name__)
        raise click.UsageError(
            f"Benchmark {name!r} does not support --prompt-level"
        )
    try:
        setter(parsed)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@click.group(name="benchmark")
def benchmark_group() -> None:
    """List benchmarks, check their configs, and build their targets."""


@benchmark_group.command("list")
def benchmark_list() -> None:
    """List benchmarks that can be used with ``cage run <benchmark>``."""
    run_surface.echo_benchmark_list(include_footer=True)


@benchmark_group.command("show")
@click.argument("benchmark_id")
def benchmark_show(benchmark_id: str) -> None:
    """Show the default configuration surface for one benchmark."""
    run_surface.echo_benchmark_run_surface(benchmark_id)


@benchmark_group.command(
    "check",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("benchmark_id")
@click.option(
    "--models",
    "models_file",
    type=click.Path(exists=True),
    default=None,
    hidden=True,
    help="Override the repo default model registry for this check.",
)
@click.option(
    "--model",
    "model_id",
    default="",
    help="Override the model for exactly one selected agent.",
)
@click.option(
    "--agent", "agent_ids", multiple=True,
    help="Restrict the check to specific agent IDs/classes.",
)
@click.option(
    "--sample", "sample_ids", multiple=True,
    help="Restrict to sample ids (repeatable; comma-allowed; @FILE reads a list).",
)
@click.option("--limit", type=int, default=None, help="Check only the first N samples.")
@click.option("--passk", type=int, default=None, help="Override runtime.passk.")
@click.option("--max-trials-global", type=int, default=None, help="Override runtime.max_trials_global.")
@click.option(
    "--max-concurrent",
    "--agent-max-concurrent",
    "max_concurrent",
    type=int,
    default=None,
    help=(
        "Override agents[].max_concurrent for exactly one selected agent — the "
        "number of trials run AT ONCE (concurrency cap, not a total count)."
    ),
)
@click.option("--timeout", type=float, default=None, help="Override runtime.timeout.")
@click.option("--max-rounds", type=int, default=None, help="Override runtime.max_rounds.")
@click.option(
    "--max-input-tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Override runtime.max_input_tokens. Unset means unlimited.",
)
@click.option(
    "--max-output-tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Override runtime.max_output_tokens. Unset means unlimited.",
)
@click.option(
    "--max-cost",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Override runtime.max_cost in USD. Unset means unlimited.",
)
@click.option(
    "--upstream-proxy",
    "upstream_http_proxy",
    default="",
    help="Override proxy.upstream_http_proxy for this check.",
)
@click.option(
    "--set",
    "set_values",
    multiple=True,
    help="Override a project.yml path, e.g. --set runtime.timeout=7200.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for check artifacts. Default: .cage_checks/<check-id>.",
)
@click.option(
    "--preview-lines",
    type=int,
    default=12,
    show_default=True,
    help="Prompt preview lines to print for one-sample checks.",
)
@click.option("--show-prompt", is_flag=True, help="Print full rendered prompts.")
@click.option(
    "--strict-exit/--no-strict-exit",
    default=True,
    help="Exit non-zero if any prompt check fails.",
)
@click.pass_context
def benchmark_check(
    ctx: click.Context,
    benchmark_id: str,
    models_file: str | None,
    model_id: str,
    agent_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    limit: int | None,
    passk: int | None,
    max_trials_global: int | None,
    max_concurrent: int | None,
    timeout: float | None,
    max_rounds: int | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
    max_cost: float | None,
    upstream_http_proxy: str,
    set_values: tuple[str, ...],
    out_dir: str | None,
    preview_lines: int,
    show_prompt: bool,
    strict_exit: bool,
) -> None:
    """Check benchmark config and render prompts without launching targets."""
    from cage.config.experiment import resolve

    effective_project, temp_project, resolved_id = project_prep.prepare_project_for_run(
        benchmark_id,
        extra_args=list(ctx.args),
        agent_ids=agent_ids,
        resume=False,
        models_file=models_file,
        model_id=model_id,
        upstream_http_proxy=upstream_http_proxy,
        timeout=timeout,
        max_trials_global=max_trials_global,
        max_concurrent=max_concurrent,
        passk=passk,
        max_rounds=max_rounds,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_cost=max_cost,
        set_values=set_values,
    )
    if temp_project is not None:
        ctx.call_on_close(lambda path=temp_project: path.unlink(missing_ok=True))

    try:
        config = resolve(effective_project)
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, FileNotFoundError):
            detail = str(exc)
        raise click.UsageError(
            f"Unable to load benchmark config for {resolved_id or benchmark_id}: "
            f"{detail}. If this benchmark uses local datasets, make sure the "
            "dataset files are present before checking prompts."
        ) from exc
    if resolved_id:
        config.benchmark_dir = project_prep.benchmark_source_dir(resolved_id)
    project_prep.restrict_config_agents(config, agent_ids, source_label=benchmark_id)

    wanted_samples = _split_cli_ids(sample_ids)
    check_dir = (
        Path(out_dir).expanduser()
        if out_dir
        else _default_check_dir(resolved_id or benchmark_id)
    )
    try:
        result = _run_benchmark_prompt_check(
            config,
            benchmark_id=resolved_id or benchmark_id,
            project_file=effective_project,
            sample_ids=wanted_samples,
            limit=limit,
            out_dir=check_dir,
            preview_lines=preview_lines,
            show_prompt=show_prompt,
        )
    finally:
        try:
            config.benchmark.teardown()
        except Exception as exc:  # noqa: BLE001
            click.echo(click.style(f"teardown warning: {exc}", fg="yellow"))

    if strict_exit and not result["ok"]:
        raise SystemExit(1)


@benchmark_group.command("build")
@click.argument("benchmark_id")
@click.option(
    "--limit", "limit", type=int, default=None,
    help="Only build the first N samples.",
)
@click.option(
    "--sample", "sample_ids", multiple=True, default=(),
    help="Restrict to sample ids (repeatable; comma-allowed).",
)
@click.option(
    "--only", "only_ids", multiple=True, default=(),
    help="Alias for --sample.",
)
@click.option(
    "--max-concurrent",
    "max_concurrent",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Build up to N benchmark targets concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the build targets and image tags that would be built, without building them.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Rebuild target images even if they already exist "
    "(default: skip images that are already built).",
)
@click.option(
    "--set",
    "set_values",
    multiple=True,
    help="Override a project.yml path, e.g. --set eval.benchmark.hint_levels=[0].",
)
@click.pass_context
def benchmark_build(
    ctx: click.Context,
    benchmark_id: str,
    limit: int | None,
    sample_ids: tuple[str, ...],
    only_ids: tuple[str, ...],
    max_concurrent: int,
    dry_run: bool,
    rebuild: bool,
    set_values: tuple[str, ...],
) -> None:
    """Run a registered benchmark's build hook without launching targets."""
    try:
        spec = registry.resolve_benchmark(benchmark_id)
    except UnknownBenchmarkError as exc:
        raise click.UsageError(str(exc)) from exc
    project_file = spec.resolved_project_file
    effective_project = project_file

    if set_values:
        effective_project, temp_project, _ = project_prep.prepare_project_for_run(
            benchmark_id,
            extra_args=[],
            agent_ids=(),
            resume=False,
            models_file=None,
            model_id="",
            upstream_http_proxy="",
            timeout=None,
            max_trials_global=None,
            max_concurrent=None,
            passk=None,
            max_rounds=None,
            max_input_tokens=None,
            max_output_tokens=None,
            max_cost=None,
            set_values=set_values,
        )
        if temp_project is not None:
            ctx.call_on_close(lambda path=temp_project: path.unlink(missing_ok=True))

    click.echo(f"Benchmark: {spec.id}")
    click.echo(f"Default project: {_display_path(project_file)}")
    project_prep.run_benchmark_image_build(
        effective_project,
        limit=limit,
        only=tuple(_split_cli_ids((*sample_ids, *only_ids))),
        max_workers=max_concurrent,
        dry_run=dry_run,
        rebuild=rebuild,
    )


def _default_check_dir(benchmark_id: str) -> Path:
    from datetime import datetime
    from uuid import uuid4

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(".cage_checks") / f"{stamp}-{benchmark_id}-{uuid4().hex[:8]}"


def _safe_artifact_name(value: object) -> str:
    import re

    text = str(value or "sample").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "sample"


def _sample_matches(sample: dict[str, object], wanted: set[str]) -> bool:
    return sample_id_matches(sample, wanted)


def _iter_check_samples(
    benchmark: object,
    *,
    sample_ids: list[str],
    limit: int | None,
) -> list[dict[str, object]]:
    wanted = set(sample_ids)
    samples: list[dict[str, object]] = []
    for sample in benchmark.iter_samples():
        if wanted and not _sample_matches(sample, wanted):
            continue
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _prompt_preview(rendered: str, max_lines: int) -> str:
    lines = rendered.strip().splitlines()
    if max_lines <= 0:
        return ""
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += "\n..."
    return preview


def _public_sample_payload(sample: dict[str, object]) -> dict[str, object]:
    allowed = (
        "id",
        "challenge_id",
        "benchmark",
        "name",
        "challenge_name",
        "category",
        "task_profile",
        "content",
        "task",
        "prompt_level",
        "hint_level",
        "agent_input",
        "vulnerability_hints",
    )
    return {key: sample[key] for key in allowed if key in sample}


def _run_benchmark_prompt_check(
    config: object,
    *,
    benchmark_id: str,
    project_file: Path,
    sample_ids: list[str],
    limit: int | None,
    out_dir: Path,
    preview_lines: int,
    show_prompt: bool,
) -> dict[str, object]:
    import hashlib
    import json

    from cage.benchmarks.prompt_contract import check_sample, discover_template_source

    benchmark = config.benchmark
    samples = _iter_check_samples(benchmark, sample_ids=sample_ids, limit=limit)
    if not samples:
        detail = f" for {sample_ids}" if sample_ids else ""
        raise click.UsageError(f"No benchmark samples matched{detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_root = out_dir / "prompts"
    prompt_root.mkdir(parents=True, exist_ok=True)
    template_src = discover_template_source(benchmark)

    records: list[dict[str, object]] = []
    first_prompt = ""

    for sample in samples:
        sample_id = str(sample.get("id") or "<unknown>")
        rendered = ""
        prompt_path: Path | None = None
        issues: list[str] = []
        missing_vars: list[str] = []
        ok = True
        try:
            rendered = benchmark.build_prompt(sample)
            report = check_sample(template_src, sample, rendered=rendered)
            ok = report.ok
            issues = list(report.issues)
            missing_vars = list(report.missing_vars)
        except Exception as exc:  # noqa: BLE001
            ok = False
            issues = [f"build_prompt raised: {type(exc).__name__}: {exc}"]

        if rendered:
            sample_dir = prompt_root / _safe_artifact_name(sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = sample_dir / "prompt.md"
            prompt_path.write_text(rendered, encoding="utf-8")
            (sample_dir / "sample.json").write_text(
                json.dumps(
                    _public_sample_payload(sample),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            if not first_prompt:
                first_prompt = rendered

        records.append({
            "sample_id": sample_id,
            "challenge_id": sample.get("challenge_id", sample_id),
            "ok": ok,
            "issues": issues,
            "missing_vars": missing_vars,
            "rendered_chars": len(rendered),
            "prompt_path": str(prompt_path) if prompt_path else "",
            "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if rendered else "",
        })

    passk = getattr(getattr(config, "execution", object()), "passk", 1)
    agents = list(getattr(config, "agents", []))
    agent_labels = [getattr(agent, "label", lambda: getattr(agent, "id", ""))() for agent in agents]
    total_trials = len(samples) * len(agents) * int(passk)
    ok = all(bool(record["ok"]) for record in records)
    payload = {
        "benchmark": benchmark_id,
        "project_file": str(project_file),
        "ok": ok,
        "agents": agent_labels,
        "samples": records,
        "passk": passk,
        "total_trials": total_trials,
    }
    (out_dir / "check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    click.echo(f"Benchmark: {benchmark_id}")
    click.echo(f"Config: {_display_path(Path(project_file))}")
    click.echo()
    click.echo("Plan:")
    click.echo(
        f"  agents: {len(agents)}"
        + (f" ({', '.join(agent_labels)})" if agent_labels else "")
    )
    sample_list = ", ".join(str(record["sample_id"]) for record in records[:5])
    if len(records) > 5:
        sample_list += f", ... (+{len(records) - 5} more)"
    click.echo(
        f"  samples: {len(samples)}"
        + (f" selected ({sample_list})" if sample_list else "")
    )
    prompt_levels = sorted(
        {
            str(sample.get("prompt_level"))
            for sample in samples
            if sample.get("prompt_level") not in (None, "")
        }
    )
    hint_levels = sorted(
        {
            str(sample.get("hint_level"))
            for sample in samples
            if sample.get("hint_level") not in (None, "")
        }
    )
    if prompt_levels:
        click.echo(f"  prompt levels: {', '.join(prompt_levels)}")
    elif hint_levels:
        click.echo(f"  hint levels: {', '.join(hint_levels)}")
    click.echo(f"  passk: {passk} attempt(s) per agent/sample")
    click.echo(f"  total trials: {total_trials} agent run(s)")
    click.echo()
    click.echo("Prompt render:")
    for record in records:
        status = "OK" if record["ok"] else "FAIL"
        click.echo(
            f"  {status} {record['sample_id']} "
            f"({record['rendered_chars']} chars)"
        )
        for issue in record["issues"]:
            click.echo(click.style(f"    - {issue}", fg="red"))
    if len(records) == 1 and first_prompt and not show_prompt:
        preview = _prompt_preview(first_prompt, preview_lines)
        if preview:
            click.echo()
            click.echo(f"Prompt preview (first {preview_lines} lines):")
            for line in preview.splitlines():
                click.echo(f"  {line}")
    if show_prompt:
        for record in records:
            path = record.get("prompt_path")
            if not path:
                continue
            text = Path(str(path)).read_text(encoding="utf-8")
            label = str(record["sample_id"])
            click.echo()
            click.echo(f"--- PROMPT {label} BEGIN ---")
            click.echo(text.rstrip())
            click.echo(f"--- PROMPT {label} END ---")

    click.echo()
    click.echo(f"Prompt artifacts directory: {_display_path(out_dir.resolve())}")
    if len(records) == 1 and records[0].get("prompt_path"):
        prompt_path = Path(str(records[0]["prompt_path"])).resolve()
        click.echo(f"Rendered prompt file: {_display_path(prompt_path)}")
    if len(records) == 1:
        sample_id = str(records[0]["sample_id"])
        click.echo("Target not launched. To build this target image, run:")
        click.echo(f"  cage benchmark build {benchmark_id} --sample {sample_id}")
    else:
        click.echo(
            "Target not launched. To build target images, use: "
            f"cage benchmark build {benchmark_id} [--sample <sample_id>] "
            "[--max-concurrent N]"
        )
    return payload
