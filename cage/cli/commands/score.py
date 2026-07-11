"""Score Cage CLI command."""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from typing import Any

import click

from cage.benchmarks.registry import UnknownBenchmarkError, resolve_benchmark
from cage.cli.paths import display_path


def _find_project_run_dirs(
    cage_runs: Path,
    config: object,
    run_id_filter: str,
) -> list[Path]:
    """Return project-owned run dirs under a ``.cage_runs`` root."""

    agents = list(getattr(config, "agents", []) or [])
    matches: list[Path] = []
    for agent in agents:
        label_fn = getattr(agent, "label", None)
        if not callable(label_fn):
            continue
        agent_dir = cage_runs / label_fn()
        if not agent_dir.is_dir():
            continue
        if run_id_filter:
            candidate = agent_dir / run_id_filter
            if candidate.is_dir():
                matches.append(candidate)
            continue
        for child in sorted(agent_dir.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                matches.append(child)
    return matches


def _run_root_for_run_dir(run_dir: Path) -> Path | None:
    """Return the project root that owns a run directory, if it is inferable."""

    for parent in run_dir.resolve().parents:
        if parent.name == ".cage_runs":
            return parent.parent
    return None


def _run_dir_project_candidates(run_dir: Path) -> list[tuple[Path, Path | None]]:
    """Find project files that may have produced a standalone run directory."""

    import yaml

    candidates: list[tuple[Path, Path | None]] = []
    snapshot_base_dirs: list[Path] = []

    def add_snapshot_base(value: str) -> None:
        if not value.strip():
            return
        base = Path(value).expanduser().parent
        if base.is_dir():
            snapshot_base_dirs.append(base)

    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        try:
            run_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            run_config = {}
        if isinstance(run_config, dict):
            for key in ("project_file", "source_project_file"):
                value = str(run_config.get(key) or "").strip()
                if value:
                    project_path = Path(value).expanduser()
                    candidates.append((project_path, None))
                    add_snapshot_base(value)
            benchmark_dir = str(run_config.get("benchmark_dir") or "").strip()
            if benchmark_dir:
                base = Path(benchmark_dir).expanduser()
                if base.is_dir():
                    snapshot_base_dirs.append(base)

    root = _run_root_for_run_dir(run_dir)
    if root is not None:
        candidates.extend((path, None) for path in (root / "project.yml", root / "project.yaml"))
        candidates.extend((path, None) for path in sorted(root.glob("example*.yml")))
        candidates.extend((path, None) for path in sorted(root.glob("example*.yaml")))

    snapshot_paths = [run_dir / "project.yml", run_dir / "project.yaml"]
    for snapshot_path in snapshot_paths:
        for base in snapshot_base_dirs:
            candidates.append((snapshot_path, base))
        candidates.append((snapshot_path, None))

    seen: set[tuple[Path, Path | None]] = set()
    unique: list[tuple[Path, Path | None]] = []
    for candidate, base in candidates:
        resolved = candidate.resolve()
        resolved_base = base.resolve() if base is not None else None
        key = (resolved, resolved_base)
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        unique.append((resolved, resolved_base))
    return unique


def _score_output_path_for_context(ctx: Any, score_name: str) -> Path | None:
    """Return where ``cage score`` should write one score artifact."""

    if ctx.run_dir is not None and ctx.canonical_trial_id:
        trial_parts = [
            part if part and part not in {".", ".."} else "_"
            for part in str(ctx.canonical_trial_id).split("/")
        ]
        filename = str(score_name or "score").replace("/", "_")
        if filename in {"", ".", ".."}:
            filename = "score"
        return ctx.run_dir / "scores" / "trials" / Path(*trial_parts) / f"{filename}.json"
    if ctx.trial_dir is None:
        return None
    return ctx.trial_dir / "scores" / f"{score_name}.json"


@click.command()
@click.argument("target", metavar="BENCHMARK_OR_PROJECT_OR_RUN")
@click.option(
    "--scorer", "scorer_paths", multiple=True,
    help="Path to a Python file defining a Scorer subclass (repeatable)",
)
@click.option(
    "--run-id", "run_id_filter",
    default="",
    help=(
        "When TARGET is a benchmark id or project.yml, restrict re-scoring "
        "to a single run id under .cage_runs/<agent_label>/<run_id>/. "
        "Without this flag, every matching run dir is rescored."
    ),
)
@click.option(
    "--max-concurrent",
    "max_concurrent",
    type=int,
    default=None,
    help=(
        "Score up to N trials AT ONCE (a concurrency cap, same flag as "
        "`cage run`). The slow part of scoring is the scorer itself — an "
        "LLM_judge signal makes one model call per trial — so N-way "
        "concurrency reruns that many judges in parallel. Only the expensive "
        "scorer.score() is parallelized; every artifact/manifest write stays "
        "serialized, so output is identical to serial scoring. Unset means "
        "serial (N=1)."
    ),
)
def score(
    target: str,
    scorer_paths: tuple[str, ...],
    run_id_filter: str,
    max_concurrent: int | None,
) -> None:
    """Score (or re-score) trial results from a completed run."""

    from cage.scoring import (
        Scorer,
        ScoringContext,
        load_scorer_from_module,
    )

    target_path = Path(target).expanduser()
    benchmark_id = ""
    benchmark_label = ""
    project_target: Path | None = None
    cage_runs: Path | None = None
    run_dir_project: Path | None = None
    run_dir_load_error = ""
    is_benchmark_mode = False

    is_project_mode = (
        target_path.exists()
        and target_path.is_file()
        and target_path.suffix.lower() in (".yml", ".yaml")
    )
    if not target_path.exists():
        try:
            spec = resolve_benchmark(target)
        except UnknownBenchmarkError:
            raise click.UsageError(
                "TARGET must be a registered benchmark id, a project.yml file, "
                f"or a run directory: {target}"
            ) from None
        benchmark_id = spec.id
        benchmark_label = spec.display_name
        project_target = spec.resolved_project_file
        # Run output lives next to the benchmark's project.yml (examples/<id>/),
        # the same per-benchmark tree cage run / benchmark check write to.
        cage_runs = project_target.resolve().parent / ".cage_runs"
        is_benchmark_mode = True
    elif is_project_mode:
        project_target = target_path
        cage_runs = target_path.resolve().parent / ".cage_runs"
    elif not target_path.is_dir():
        raise click.UsageError(
            "TARGET must be a registered benchmark id, a project.yml file, "
            f"or a run directory: {target}"
        )

    benchmark_scorer: Scorer | None = None
    bench_label = ""
    run_paths: list[Path]
    if project_target is not None and cage_runs is not None:
        from cage.config.experiment import resolve

        try:
            config = resolve(project_target)
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, FileNotFoundError):
                detail = str(exc)
            label = benchmark_id or str(project_target)
            raise click.UsageError(
                f"Unable to load benchmark config for {label}: {detail}. "
                "If this benchmark uses local datasets, make sure the dataset "
                "files are present before scoring."
            ) from exc
        if is_benchmark_mode:
            config.benchmark_dir = project_target.resolve().parent
        benchmark_scorer = config.benchmark.scorer()
        bench_label = (
            benchmark_label
            or config.benchmark.name
            or benchmark_scorer.name
            or "benchmark"
        )
        if not cage_runs.is_dir():
            click.echo(f"Error: no runs found at {cage_runs}")
            raise SystemExit(1)
        run_paths = _find_project_run_dirs(cage_runs, config, run_id_filter)
        if not run_paths:
            hint = f" with --run-id {run_id_filter!r}" if run_id_filter else ""
            click.echo(f"Error: no run dirs matched{hint} under {cage_runs}")
            raise SystemExit(1)
    else:
        if not scorer_paths:
            from cage.config.experiment import resolve

            for candidate, base_dir in _run_dir_project_candidates(target_path):
                try:
                    config = resolve(candidate, base_dir=base_dir)
                except Exception as exc:  # noqa: BLE001
                    base_note = f" (base_dir={base_dir})" if base_dir else ""
                    run_dir_load_error = (
                        f"{candidate}{base_note}: {type(exc).__name__}: {exc}"
                    )
                    continue
                benchmark_scorer = config.benchmark.scorer()
                bench_label = (
                    config.benchmark.name
                    or benchmark_scorer.name
                    or "benchmark"
                )
                run_dir_project = candidate
                break
        run_paths = [target_path]

    scorers: list[Scorer] = []
    if benchmark_scorer is not None:
        scorers.append(benchmark_scorer)
    for sp in scorer_paths:
        p = Path(sp)
        if not p.exists():
            click.echo(f"Error: scorer not found: {sp}")
            raise SystemExit(1)
        scorers.append(load_scorer_from_module(p))

    if not scorers:
        extra = ""
        if run_dir_load_error:
            extra = f" Last project auto-detect error: {run_dir_load_error}"
        click.echo(
            "Error: no scorers available. Pass a benchmark id or project.yml "
            "so the benchmark's own scorer is loaded, or specify "
            f"--scorer <path/to/scorer.py>.{extra}"
        )
        raise SystemExit(1)

    if project_target is not None:
        if is_benchmark_mode:
            click.echo(f"Benchmark: {benchmark_id} ({bench_label})")
            click.echo(f"Default project: {display_path(project_target)}")
        else:
            click.echo(f"Project: {target} ({bench_label})")
        click.echo(f"Run dirs: {len(run_paths)}")
        for rp in run_paths:
            click.echo(f"  - {rp}")
    else:
        click.echo(f"Scoring run: {target}")
        if run_dir_project is not None:
            click.echo(
                "Project context: "
                f"{display_path(run_dir_project)} ({bench_label})"
            )
    click.echo(f"Scorers: {', '.join(s.name for s in scorers)}")

    from cage.artifacts.reader import ExperimentArtifactReader
    from cage.artifacts.run_storage import (
        EXPERIMENT_RECORD_FILENAME,
        TASK_OUTPUT_FILENAME,
        iter_live_trial_dirs,
    )
    from cage.artifacts.writer import ExperimentArtifactWriter

    scoring_contexts: list[ScoringContext] = []
    seen_trial_ids: set[str] = set()
    for rp in run_paths:
        canonical_snapshot_loaded = False
        if (rp / EXPERIMENT_RECORD_FILENAME).is_file():
            snapshot = ExperimentArtifactReader(rp).try_load_snapshot()
            if snapshot is not None:
                canonical_snapshot_loaded = True
                for trial_record in snapshot.trial_records:
                    ctx = ScoringContext.from_trial_record(rp, trial_record)
                    if ctx is None or ctx.trial_id in seen_trial_ids:
                        continue
                    scoring_contexts.append(ctx)
                    seen_trial_ids.add(ctx.trial_id)
        if canonical_snapshot_loaded:
            continue

        for td in iter_live_trial_dirs(rp):
            if not (td / TASK_OUTPUT_FILENAME).exists():
                continue
            ctx = ScoringContext.from_trial_dir(td)
            if ctx is None or ctx.trial_id in seen_trial_ids:
                continue
            scoring_contexts.append(ctx)
            seen_trial_ids.add(ctx.trial_id)

    click.echo(f"Found {len(scoring_contexts)} trials")

    # Concurrency cap, mirroring `cage run --max-concurrent`: how many trials to
    # score AT ONCE. Only the expensive scorer.score() (an LLM_judge signal is
    # one model call per trial) is fanned out; every disk/manifest mutation is
    # applied serially below, so N=1 and N>1 produce byte-identical artifacts.
    workers = max(1, max_concurrent or 1)
    if scoring_contexts:
        workers = min(workers, len(scoring_contexts))
    else:
        workers = 1

    def _compute_scores(
        ctx: ScoringContext,
    ) -> tuple[ScoringContext, list[tuple[Scorer, Any, Exception | None]]]:
        """Run every scorer for one trial (no disk writes — thread-safe)."""
        computed: list[tuple[Scorer, Any, Exception | None]] = []
        for scorer in scorers:
            try:
                computed.append((scorer, scorer.score(ctx), None))
            except Exception as exc:  # noqa: BLE001
                computed.append((scorer, None, exc))
        return ctx, computed

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        click.echo(f"Scoring with {workers} concurrent workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            computed_trials = list(executor.map(_compute_scores, scoring_contexts))
    else:
        computed_trials = [_compute_scores(ctx) for ctx in scoring_contexts]

    results: dict[str, dict[str, dict[str, object]]] = {}
    canonical_run_score_values: dict[Path, dict[str, list[float]]] = {}
    for ctx, computed in computed_trials:
        trial_scores: dict[str, dict[str, object]] = {}
        for scorer, scores_map, compute_error in computed:
            try:
                if compute_error is not None:
                    raise compute_error
                for name, s in scores_map.items():
                    trial_scores[name] = {
                        "value": s.value,
                        "answer": s.answer,
                        "explanation": s.explanation,
                        "metadata": s.metadata,
                    }
                    score_path = _score_output_path_for_context(ctx, name)
                    if score_path is None:
                        continue
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    score_path.write_text(
                        json_mod.dumps({name: trial_scores[name]}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    if ctx.run_dir is not None and ctx.canonical_trial_id:
                        try:
                            score_ref = score_path.relative_to(ctx.run_dir).as_posix()
                            writer = ExperimentArtifactWriter(ctx.run_dir)
                            writer.mark_trial_artifact(
                                ctx.canonical_trial_id,
                                artifact_id=f"trial.{ctx.canonical_trial_id}.score.{name}",
                                path=score_ref,
                                kind="trial_score",
                                schema_version="trial_score.v1",
                                producer="cage score",
                                replayability="replayable",
                            )
                            writer.mark_trial_scored(
                                ctx.canonical_trial_id,
                                score_ref=score_ref,
                                scoring_id=name,
                            )
                            if isinstance(s.value, (int, float)):
                                by_name = canonical_run_score_values.setdefault(
                                    ctx.run_dir,
                                    {},
                                )
                                by_name.setdefault(name, []).append(float(s.value))
                        except Exception:
                            pass
            except Exception as exc:
                click.echo(f"  Error scoring {ctx.trial_id} with {scorer.name}: {exc}")
                trial_scores[scorer.name or "scorer"] = {"value": 0.0, "error": str(exc)}

        results[ctx.trial_id] = trial_scores

        scores_line = "  ".join(
            f"{name}={d['value']:.2f}" for name, d in trial_scores.items()
            if isinstance(d.get("value"), (int, float))
        )
        click.echo(f"  {ctx.trial_id}: {scores_line}")

    click.echo()
    all_names: set[str] = set()
    for trial_scores in results.values():
        all_names.update(trial_scores.keys())
    for name in sorted(all_names):
        values = [
            r[name]["value"]
            for r in results.values()
            if name in r
            and "error" not in r[name]
            and isinstance(r[name].get("value"), (int, float))
        ]
        if values:
            mean = sum(values) / len(values)
            click.echo(f"  {name}: mean={mean:.4f} ({len(values)} trials)")
    for run_dir, score_values in canonical_run_score_values.items():
        summary_scores = {
            name: {
                "count": len(values),
                "mean": sum(values) / len(values),
            }
            for name, values in sorted(score_values.items())
            if values
        }
        if not summary_scores:
            continue
        summary_path = run_dir / "scores" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json_mod.dumps(
                {
                    "schema_version": "score_summary.v1",
                    "scores": summary_scores,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            summary_ref = summary_path.relative_to(run_dir).as_posix()
            writer = ExperimentArtifactWriter(run_dir)
            writer.mark_run_artifact(
                artifact_id="run.score_summary",
                path=summary_path,
                kind="score_summary",
                schema_version="score_summary.v1",
                producer="cage score",
                replayability="replayable",
            )
            writer.mark_run_scored(summary_ref=summary_ref)
        except Exception:
            pass
    click.echo()
