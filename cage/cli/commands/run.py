"""Run Cage CLI command."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import click

from cage.benchmarks import parse_sample_slice, registry
from cage.benchmarks.registry import UnknownBenchmarkError
from cage.cli.commands import _project_prep as project_prep
from cage.cli.commands import _run_surface as run_surface
from cage.contracts.execution import max_rounds_config_label

if TYPE_CHECKING:
    from cage.experiment.model import ExperimentPlan, ExperimentSpec


class _RunCommand(click.Command):
    """``cage run`` command with a self-explanatory usage line.

    Click's default one-liner (``Usage: cage run [OPTIONS] PROJECT_OR_BENCHMARK``)
    is opaque on errors. We spell out what the positional means and where to look
    next, so a mistyped flag or a bad ``--run-id`` points the user somewhere.
    """

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] <benchmark-id | path/to/project.yml>",
        )
        formatter.write_paragraph()
        with formatter.indentation():
            formatter.write_text(
                "Examples: `cage run cybergym --model nex-n2`  ·  "
                "`cage run ./project.yml --resume --run-id <id>`. "
                "See `cage run --help`, or `cage run <benchmark> --help` for a "
                "benchmark's samples/agents/defaults."
            )


def _assert_run_can_terminate(
    *,
    agent_round_budgets: list,
    execution_max_rounds: object,
    timeout: float | None,
    max_cost: float | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
) -> None:
    """Reject a run that could never stop (fail fast, also blocks --dry-run)."""

    from cage.contracts.execution import run_lacks_termination_condition

    if run_lacks_termination_condition(
        agent_round_budgets=agent_round_budgets,
        execution_max_rounds=execution_max_rounds,
        timeout=timeout,
        max_cost=max_cost,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    ):
        raise click.UsageError(
            "This run has no termination condition: max_rounds is unlimited and "
            "none of runtime.timeout / max_cost / max_input_tokens / "
            "max_output_tokens is set, so a trial could run forever. Set at least "
            "one finite stop condition (e.g. runtime.max_rounds: 100, or "
            "runtime.timeout: 3600, or max_cost)."
        )


def _print_resume_dry_run(config: object, plans: list) -> None:
    """Print a human-readable preview of what ``--resume`` would do.

    Output is grouped per agent: counts at the top, then the trials slated for
    re-run (grouped by category, with each trial's detail label) and the kept
    trials — completed / opted-out replays are summarized as counts, while the
    *interesting* keeps (salvaged by ``keep_if``, excluded by ``select``,
    retry-capped) are listed individually so the operator can sanity-check the
    classification.
    """
    from collections import Counter

    from cage.experiment.engine.resume import _resume_keep_if_summary

    click.echo("=" * 70)
    title = (
        "DRY-RUN: cage run --resume preview (no containers, no disk writes)"
        if config.resume
        else "DRY-RUN: cage run plan (no containers, no disk writes)"
    )
    click.echo(title)
    click.echo("=" * 70)
    click.echo(f"Project file:        {config.project_file}")
    click.echo(f"Run ID:              {config.run_id or '(none)'}")
    if config.resume:
        click.echo("Resume mode:         on")
    else:
        click.echo("Plan mode:           fresh run (every planned trial would run)")
    if config.resume:
        extras = list(config.resume_retry_reasons) or "[]"
        click.echo(f"Retry reasons:       default set + {extras}")
        max_attempts = (
            str(config.resume_max_attempts)
            if config.resume_max_attempts > 0
            else "unlimited"
        )
        click.echo(f"Max attempts:        {max_attempts}")
        click.echo(
            f"Select (id_matches): {config.resume_select_id_pattern or '(all trials)'}"
        )
        click.echo(f"Keep-if (veto):      {_resume_keep_if_summary(config.resume_keep_if)}")
    execution = getattr(config, "execution", object())
    click.echo()
    click.echo("Runtime:")
    click.echo(f"  max_trials_global:  {_fmt_global_cap(getattr(execution, 'max_trials_global', '(unset)'))}")
    click.echo(f"  max_target_setups:  {getattr(execution, 'max_target_setups', '(unset)')}")
    click.echo(
        f"  max_rounds:         "
        f"{max_rounds_config_label(getattr(execution, 'max_rounds', None))}"
    )
    click.echo(f"  max_input_tokens:   {getattr(execution, 'max_input_tokens', '(unset)')}")
    click.echo(f"  max_output_tokens:  {getattr(execution, 'max_output_tokens', '(unset)')}")
    click.echo(f"  max_cost:           {getattr(execution, 'max_cost', '(unset)')}")
    click.echo(f"  passk:              {getattr(execution, 'passk', '(unset)')}")
    max_trial = getattr(execution, "max_trial", None)
    if max_trial is not None:
        click.echo(f"  max_trial:          {max_trial}")
    agents = list(getattr(config, "agents", []) or [])
    if agents:
        caps = []
        for agent in agents:
            label = (
                agent.label()
                if callable(getattr(agent, "label", None))
                else str(getattr(agent, "id", "agent"))
            )
            cap = getattr(agent, "max_concurrent", 0) or "unbounded"
            caps.append(f"{label}={cap}")
        click.echo(f"  max_concurrent:     {', '.join(caps)}")
    click.echo()

    total_rerun = 0
    for plan in plans:
        click.echo(f"Agent: {plan.agent_label}")
        click.echo(f"  Run dir:           {plan.run_dir}")
        click.echo(f"  Planned trials:    {plan.total}")
        if plan.max_trial is not None and plan.considered != plan.total:
            click.echo(
                f"  ↳ max_trial cap:   {plan.max_trial} "
                f"(considering {plan.considered}/{plan.total} this run; "
                f"{plan.total - plan.considered} deferred)"
            )
        click.echo(f"  ↳ Re-run:          {len(plan.rerun)}")
        click.echo(f"  ↳ Replay (kept):   {len(plan.replay)}")
        if plan.capped:
            click.echo(f"  ↳ Retry-capped:    {len(plan.capped)} "
                       f"(would retry but max_attempts hit)")
        if plan.archives_present:
            click.echo(f"  ↳ Archives on disk: {plan.archives_present} "
                       f"(.before_resume_* from earlier resume cycles)")
        total_rerun += len(plan.rerun)

        if plan.rerun:
            # Group rerun trials by category for a quick "what's failing" tally
            by_cat = Counter(category for _, category, _ in plan.rerun)
            click.echo()
            click.echo("  Re-run by category:")
            for category, n in by_cat.most_common():
                click.echo(f"    {n:4d}  {category}")
            click.echo()
            click.echo("  Trials to re-run:")
            for trial_id, _category, label in plan.rerun:
                click.echo(f"    {trial_id:50s} {label}")

        if plan.replay:
            # Tally every kept category, then itemize EVERY kept trial (id + why
            # kept), grouped by category — a dry-run is a detailed preview, so
            # show the complete list rather than just counts.
            by_cat = Counter(category for _, category, _ in plan.replay)
            grouped: dict[str, list[tuple[str, str]]] = {}
            for tid, category, label in plan.replay:
                grouped.setdefault(category, []).append((tid, label))
            click.echo()
            click.echo("  Kept (not re-run) by reason:")
            for category, n in by_cat.most_common():
                click.echo(f"    {n:4d}  {category}")
            click.echo()
            click.echo("  Kept trials (id — why kept):")
            for category, _n in by_cat.most_common():
                for trial_id, label in grouped[category]:
                    click.echo(f"    {trial_id:50s} {label}")

        if plan.capped:
            click.echo()
            click.echo("  Retry-capped (left as final result; raise resume.max_attempts to retry):")
            for trial_id, attempts in plan.capped:
                click.echo(f"    {trial_id:50s} (attempts so far: {attempts})")
        click.echo()

    click.echo("-" * 70)
    if total_rerun == 0:
        if config.resume:
            click.echo("Nothing to re-run. cage run --resume would be a no-op.")
        else:
            click.echo("No trials selected.")
    else:
        label = "re-run" if config.resume else "run"
        click.echo(f"Total trials slated to {label}: {total_rerun}")
        click.echo("Run with: drop --dry-run from this command line.")
    click.echo()


def _print_experiment_plan_dry_run(
    spec: ExperimentSpec,
    plan: ExperimentPlan,
) -> None:
    """Print a fresh-run dry-run summary from the new Experiment contracts.

    This printer is intentionally independent of the resolved ``ExperimentRun``
    and resume analyzer. It is used before benchmark setup, agent construction,
    target launch, or artifact directory mutation, so the output must be derived
    only from serializable ``ExperimentSpec`` and ``ExperimentPlan`` data. The
    wording keeps a few stable labels from the old dry-run output because users
    and tests already key off those terms while the backend migration proceeds.
    """

    click.echo("=" * 70)
    click.echo("DRY-RUN: cage run plan (no containers, no targets, no disk writes)")
    click.echo("=" * 70)
    click.echo(f"Project file:        {spec.project_file}")
    click.echo(f"Run ID:              {spec.identity.run_id or '(none)'}")
    click.echo(f"Experiment:          {spec.identity.experiment_id}")
    click.echo(f"Benchmark:           {spec.benchmark.id}")
    click.echo(f"Plan ID:             {plan.plan_id}")
    click.echo("Plan mode:           fresh run (every planned trial would run)")
    click.echo()

    click.echo("Runtime:")
    click.echo(f"  max_trials_global:  {_fmt_global_cap(spec.runtime.scheduler.max_trials_global)}")
    click.echo(f"  max_target_setups:  {spec.runtime.scheduler.max_target_setups}")
    click.echo(f"  max_rounds:         {max_rounds_config_label(spec.protocol.max_rounds)}")
    click.echo(f"  max_input_tokens:   {_dry_run_limit(spec.protocol.max_input_tokens)}")
    click.echo(f"  max_output_tokens:  {_dry_run_limit(spec.protocol.max_output_tokens)}")
    click.echo(f"  max_cost:           {_dry_run_limit(spec.protocol.max_cost)}")
    click.echo(f"  passk:              {spec.workload.passk}")
    if spec.workload.task_selection.max_trial_num is not None:
        click.echo(f"  max_trial:          {spec.workload.task_selection.max_trial_num}")
    if plan.subjects:
        caps = []
        for subject in plan.subjects:
            cap = subject.max_concurrent if subject.max_concurrent is not None else "unbounded"
            caps.append(f"{subject.subject_id}={cap}")
        click.echo(f"  max_concurrent:     {', '.join(caps)}")
    click.echo()

    click.echo("Selection:")
    selected_samples = ", ".join(spec.workload.task_selection.samples) or "(none)"
    variant_parts = [
        f"{axis}={','.join(values)}"
        for axis, values in sorted(spec.workload.variants.items())
    ]
    click.echo(f"  samples:            {selected_samples}")
    click.echo(f"  variants:           {', '.join(variant_parts) if variant_parts else '(none)'}")
    click.echo(f"  subjects:           {len(plan.subjects)}")
    click.echo(f"  benchmark tasks:    {len(plan.tasks)}")
    click.echo(f"  Planned trials:    {len(plan.trials)}")
    click.echo()

    if plan.trials:
        tasks_by_id = {task.task_id: task for task in plan.tasks}
        click.echo("Trials to run:")
        for trial in plan.trials:
            task = tasks_by_id.get(trial.task_id)
            sample = task.source_sample_id if task is not None else trial.task_id
            click.echo(
                f"  {trial.trial_id} "
                f"(sample={sample}, subject={trial.subject_id}, pass={trial.pass_index})"
            )
        click.echo()

    click.echo("-" * 70)
    if plan.trials:
        click.echo(f"Total trials slated to run: {len(plan.trials)}")
        click.echo("Run with: drop --dry-run from this command line.")
    else:
        click.echo("No trials selected.")
    click.echo()


def _dry_run_limit(value: object) -> object:
    """Format nullable dry-run limits without implying the field is missing."""

    return "unlimited" if value is None else value


def _fmt_global_cap(value: object) -> object:
    """Render ``max_trials_global`` for display; 0 reads as unlimited."""

    if value in (0, None, ""):
        return "unlimited"
    return value


def _model_endpoint_reachable(base_url: str, *, timeout: float = 8.0) -> bool:
    """True if an HTTP server answers at ``base_url`` (any status counts).

    A 401/403/404 still means the server is up — only connection errors /
    timeouts mean "not yet". Used by --wait-for-model to poll remotely-launched
    vLLM endpoints until they come online.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _wait_for_model_endpoints(
    config: object, *, timeout: float, interval: float
) -> None:
    """Block until every model endpoint this run uses is reachable.

    Only self-hosted endpoints (``provider: vllm`` / ``sglang``, i.e.
    ``ModelConfig.is_local_endpoint``) are polled — managed SaaS providers are
    assumed always up, and subscription/OAuth endpoints have no ``base_url`` to
    probe. Polls each distinct ``base_url`` across all agents (including every
    ``model_sources`` entry) every ``interval`` seconds. ``timeout`` of 0 waits
    indefinitely (the operator does not know when the remote server will boot);
    a positive value aborts the run with a clear error once exceeded.
    """
    import time as _time

    seen: set[str] = set()
    pending: list[tuple[str, str]] = []
    for agent in getattr(config, "agents", []) or []:
        models = list(getattr(agent, "model_sources", None) or [agent.model])
        for mdl in models:
            if not getattr(mdl, "is_local_endpoint", False):
                continue
            base_url = str(getattr(mdl, "base_url", "") or "").strip()
            if not base_url or base_url in seen:
                continue
            seen.add(base_url)
            pending.append((str(getattr(mdl, "id", "") or base_url), base_url))
    if not pending:
        click.echo(
            "--wait-for-model: no self-hosted (vllm/sglang) endpoints to poll; "
            "starting run."
        )
        return

    start = _time.monotonic()
    while pending:
        still: list[tuple[str, str]] = []
        for mid, base_url in pending:
            if _model_endpoint_reachable(base_url):
                click.echo(f"  model endpoint up: {mid} ({base_url})")
            else:
                still.append((mid, base_url))
        pending = still
        if not pending:
            break
        elapsed = _time.monotonic() - start
        if timeout and elapsed >= timeout:
            down = ", ".join(mid for mid, _ in pending)
            raise click.UsageError(
                f"--wait-for-model timed out after {int(elapsed)}s; still "
                f"unreachable: {down}"
            )
        names = ", ".join(mid for mid, _ in pending)
        click.echo(
            f"  waiting for {len(pending)} model endpoint(s) to come online: "
            f"{names} — retrying in {int(interval)}s"
        )
        _time.sleep(interval)
    click.echo("All model endpoints reachable; starting run.")


def run_dirs_for_config(config: object) -> list[Path]:
    """Per-agent run directories for *config* (.cage_runs/<agent>/<run_id>).

    Lets the CLI locate trial artifacts (e.g. to print a results table) without
    a summary object — works on the interrupt path too.
    """
    from cage.experiment.engine.resume import _cage_runs_root
    from cage.sandbox.naming import _parse_agent_label

    cage_runs = _cage_runs_root(config)
    run_id = getattr(config, "run_id", "") or ""
    out: list[Path] = []
    for agent in getattr(config, "agents", []) or []:
        agent_dir_name, _mode = _parse_agent_label(agent.label())
        out.append(cage_runs / agent_dir_name / run_id)
    return out


def _normalize_explicit_sample_ids(sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize CLI sample selectors before side-effect-free task planning.

    Legacy dry-run resolved samples by importing the benchmark and matching
    against ``iter_samples()`` case-insensitively. The contract planner cannot
    import the benchmark, so explicit selectors are treated as task ids and
    normalized to the lowercase id convention used by the release benchmarks.
    """

    return tuple(
        str(sample_id).strip().lower()
        for sample_id in sample_ids
        if str(sample_id).strip()
    )
@click.command(
    cls=_RunCommand,
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("project_file", metavar="PROJECT_OR_BENCHMARK", required=False)
@click.option(
    "--help",
    "show_help",
    is_flag=True,
    is_eager=True,
    help=(
        "Show run help. With a benchmark id, show that benchmark's default "
        "configuration and run examples."
    ),
)
@click.option(
    "--models",
    "models_file",
    type=click.Path(exists=True),
    default=None,
    hidden=True,
    help="Override the repo default model registry for this run.",
)
@click.option(
    "--model",
    "model_id",
    default="",
    help="Override the model for exactly one selected agent.",
)
@click.option(
    "--model-source",
    "model_sources",
    multiple=True,
    help=(
        "Registered model id this run round-robins across per trial (repeatable, "
        "e.g. --model-source glm-5.1-sii1 --model-source glm-5.1-sii2). Requires "
        "--model <logical-id> as the run key the sources rotate behind."
    ),
)
@click.option(
    "--upstream-proxy",
    "upstream_http_proxy",
    default="",
    help="Override proxy.upstream_http_proxy for this run.",
)
@click.option(
    "--wait-for-model", "wait_for_model", is_flag=True, default=False,
    help=(
        "Before starting, poll every model endpoint (and each --model-source) "
        "until it answers — for remotely-launched vLLM servers whose boot time "
        "is unknown. Pair with --wait-timeout / --wait-interval."
    ),
)
@click.option(
    "--wait-timeout", "wait_timeout", type=float, default=0.0,
    help="Max seconds to wait with --wait-for-model (0 = wait indefinitely).",
)
@click.option(
    "--wait-interval", "wait_interval", type=float, default=10.0,
    help="Seconds between --wait-for-model polls (default 10).",
)
@click.option("--timeout", type=float, default=None, help="Override runtime.timeout.")
@click.option(
    "--max-concurrent",
    "--agent-max-concurrent",
    "max_concurrent",
    type=int,
    default=None,
    help=(
        "Override agents[].max_concurrent — how many trials one agent runs AT "
        "ONCE (a concurrency cap, NOT a total-trial count). Non-resume runs "
        "require exactly one --agent; --resume without --agent caps all agents."
    ),
)
@click.option("--passk", type=int, default=None, help="Override runtime.passk.")
@click.option(
    "--max-rounds", type=str, default=None,
    help=(
        "Override runtime.max_rounds: a positive N, 'unlimited' (no round cap), "
        "or -1 (use the benchmark's built-in default). An unlimited budget needs "
        "another stop condition (--timeout / --max-cost / token caps)."
    ),
)
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
    "--set",
    "set_values",
    multiple=True,
    help="Override a project.yml path, e.g. --set runtime.timeout=7200.",
)
@click.option(
    "--max-sample-num",
    "limit",
    type=int,
    default=None,
    help=(
        "Use only the first N benchmark samples after --sample filtering. "
        "This limits samples, not expanded pass@k/agent trials."
    ),
)
@click.option(
    "--max-trial-num", "max_trial", type=int, default=None,
    help=(
        "Run only the first N expanded trials this invocation. The full trial "
        "plan is still recorded so --resume can finish the rest."
    ),
)
@click.option(
    "--sample", "sample_ids", multiple=True,
    help=(
        "Restrict the run to specific sample IDs (repeatable / comma-separated, "
        "e.g. ``--sample <id-1> --sample <id-2>``). Use ``@FILE`` to read ids "
        "from a file, one per line (# comments ok): ``--sample @subset.txt``. "
        "If unset, the project's full sample set is used."
    ),
)
@click.option(
    "--sample-slice", "sample_slice", default=None,
    help=(
        "Run a Python-style slice of the ordered sample list, e.g. ':100' "
        "(first 100), '-100:' (last 100), '-100:-1', '100:200', '::2'. "
        "Applied after --sample filtering and before --max-sample-num. "
        "Overrides eval.sample_slice from project.yml."
    ),
)
@click.option(
    "--agent", "agent_ids", multiple=True,
    help=(
        "Restrict the run to specific agent IDs/classes from ``project.yml`` "
        "(repeatable, e.g. ``--agent codex``). If unset, every agent/model "
        "pair in the project is run."
    ),
)
@click.option(
    "--run-id", "run_id",
    default="",
    help=(
        "Run identifier (≤48 chars, [a-z0-9][a-z0-9_-]*). Overrides project.run_id. "
        "If omitted on both sides, cage generates run-<ts>-<uuid8>. Reuse triggers "
        "an error unless --resume or --force is set."
    ),
)
@click.option(
    "--resume", is_flag=True,
    help=(
        "Resume an existing run: reuse the directory pointed at by --run-id (or "
        "project.run_id) and skip trials whose meta.json shows status=completed. "
        "Requires a run id to be set."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "If run_id already exists, archive the previous run directory and start "
        "a fresh run with the same run_id. Mutually exclusive with --resume."
    ),
)
@click.option(
    "--dry-run", "dry_run", is_flag=True,
    help=(
        "Print which trials WOULD be re-run vs replayed under the current "
        "resume policy, then exit without launching containers, targets, or "
        "touching the run directory. Use with --resume to preview a resume; "
        "without --resume it lists every planned trial as 'rerun'."
    ),
)
@click.option(
    "--allow-launch-build",
    is_flag=True,
    help=(
        "Before launching targets, run the benchmark-owned build hook for the "
        "selected samples. This does not enable target-server compose build "
        "fallbacks; target launch still uses existing images only."
    ),
)
@click.pass_context
def run(
    ctx: click.Context,
    project_file: str | None,
    show_help: bool,
    models_file: str | None,
    model_id: str,
    model_sources: tuple[str, ...],
    upstream_http_proxy: str,
    wait_for_model: bool,
    wait_timeout: float,
    wait_interval: float,
    timeout: float | None,
    max_concurrent: int | None,
    passk: int | None,
    max_rounds: str | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
    max_cost: float | None,
    set_values: tuple[str, ...],
    limit: int | None,
    max_trial: int | None,
    sample_ids: tuple[str, ...],
    sample_slice: str | None,
    agent_ids: tuple[str, ...],
    run_id: str,
    resume: bool,
    force: bool,
    dry_run: bool,
    allow_launch_build: bool,
) -> None:
    """Run or resume an evaluation — the main CAGE workflow.

    Use ``cage run <benchmark> --help`` to inspect benchmark-specific samples,
    agents, models, defaults, and benchmark-owned options before launching a
    run.
    """
    from cage.config.experiment import resolve
    from cage.experiment.engine.conductor import (
        ResumeCompatibilityError,
        analyze_resume_plan,
        run_experiment,
    )

    if show_help:
        if project_file:
            try:
                spec = registry.resolve_benchmark(project_file)
            except UnknownBenchmarkError:
                run_surface.echo_run_landing_help()
            else:
                run_surface.echo_benchmark_run_surface(spec.id)
            return
        run_surface.echo_run_landing_help()
        return

    if not project_file:
        raise click.UsageError(
            "Missing argument 'PROJECT_OR_BENCHMARK'. "
            "Use `cage run --help` or `cage run <benchmark> --help`."
        )

    if force and resume:
        raise click.UsageError("--force and --resume are mutually exclusive")

    # Expand --sample values: comma-lists and ``@file`` refs (a file of ids, one
    # per line) become a flat id tuple before any sample filtering downstream.
    from cage.cli.ids import split_cli_ids

    try:
        sample_ids = tuple(split_cli_ids(sample_ids))
    except FileNotFoundError as exc:
        raise click.UsageError(f"--sample {exc}") from exc

    effective_project, temp_project, benchmark_id = project_prep.prepare_project_for_run(
        project_file,
        extra_args=list(ctx.args),
        agent_ids=agent_ids,
        resume=resume,
        models_file=models_file,
        model_id=model_id,
        model_sources=model_sources,
        upstream_http_proxy=upstream_http_proxy,
        timeout=timeout,
        max_trials_global=None,
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

    if dry_run and not resume and sample_ids and not allow_launch_build:
        from cage.experiment.model import (
            build_experiment_plan,
            load_experiment_spec,
        )

        spec = load_experiment_spec(
            effective_project,
            sample_ids=_normalize_explicit_sample_ids(sample_ids),
            max_sample_num=limit,
            max_trial_num=max_trial,
        )
        if run_id:
            spec = replace(
                spec,
                identity=replace(spec.identity, run_id=run_id),
            )
        _assert_run_can_terminate(
            agent_round_budgets=[None],  # spec carries the run-level budget only
            execution_max_rounds=spec.protocol.max_rounds,
            timeout=spec.runtime.timeouts.trial_timeout_s,
            max_cost=spec.protocol.max_cost,
            max_input_tokens=spec.protocol.max_input_tokens,
            max_output_tokens=spec.protocol.max_output_tokens,
        )
        plan = build_experiment_plan(spec)
        _print_experiment_plan_dry_run(spec, plan)
        return

    config = resolve(effective_project)
    if benchmark_id:
        config.benchmark_dir = project_prep.benchmark_source_dir(benchmark_id)
        config.metadata["benchmark_id"] = benchmark_id

    # A run must be able to stop: if the round budget is unlimited, require some
    # other finite termination condition (timeout / cost / token cap).
    _execution = getattr(config, "execution", None)
    if _execution is not None:
        _assert_run_can_terminate(
            agent_round_budgets=[
                getattr(a, "max_rounds", None)
                for a in (getattr(config, "agents", None) or [])
            ],
            execution_max_rounds=getattr(_execution, "max_rounds", "unlimited"),
            timeout=getattr(_execution, "timeout", 0.0),
            max_cost=getattr(_execution, "max_cost", None),
            max_input_tokens=getattr(_execution, "max_input_tokens", None),
            max_output_tokens=getattr(_execution, "max_output_tokens", None),
        )

    if limit is not None:
        config.sample_limit = limit

    # CLI --max-trial-num overrides any runtime.max_trial from project.yml.
    if max_trial is not None:
        config.execution.max_trial = max_trial

    if sample_ids:
        config.sample_ids = tuple(
            project_prep.resolve_sample_ids_for_benchmark(
                config.benchmark,
                sample_ids,
            )
        )

    # CLI --sample-slice overrides eval.sample_slice from project.yml. Applied
    # by iter_samples_limited after --sample filtering and before --max-sample-num.
    if sample_slice is not None and str(sample_slice).strip():
        config.sample_slice = parse_sample_slice(sample_slice)

    project_prep.restrict_config_agents(
        config,
        agent_ids,
        source_label=project_file,
    )

    if allow_launch_build:
        config.metadata["launch_build"] = "benchmark-hook"
        click.echo(
            "Launch build: running benchmark build hook for selected samples."
            if not dry_run
            else "Launch build dry-run: benchmark build hook plan for selected samples.",
        )
        project_prep.run_benchmark_image_build(
            Path(effective_project),
            limit=limit,
            only=sample_ids,
            max_workers=max(
                1,
                int(getattr(getattr(config, "execution", None), "max_target_setups", 1) or 1),
            ),
            dry_run=dry_run,
        )
    else:
        config.metadata["launch_build"] = "disabled"

    # CLI --run-id wins over project.yml::project.run_id.
    if run_id:
        config.run_id = run_id

    # --force vs --resume is validated once up-front (see the check right after
    # the PROJECT argument is resolved); the engine re-checks the resolved
    # ``run.force``/``run.resume`` as a defense for programmatic callers.
    config.force = bool(force)

    if resume:
        if not config.run_id:
            raise click.UsageError("--resume requires --run-id (or project.run_id) to be set")
        config.resume = True

    if config.logging.terminal_ui:
        from cage.cli.ui.run import print_run_banner

        print_run_banner()

    if dry_run:
        try:
            plans = analyze_resume_plan(config)
        except ResumeCompatibilityError as exc:
            raise click.UsageError(str(exc)) from exc
        _print_resume_dry_run(config, plans)
        return

    # CLI --wait-for-model overrides runtime.wait_for_model; the timeout/interval
    # fall back to the yml values when the operator did not pass them.
    _exec = getattr(config, "execution", None)
    _wait_enabled = wait_for_model or bool(getattr(_exec, "wait_for_model", False))
    if _wait_enabled:
        _timeout = wait_timeout or float(getattr(_exec, "wait_timeout", 0.0) or 0.0)
        _interval = wait_interval or float(getattr(_exec, "wait_interval", 10.0) or 10.0)
        _wait_for_model_endpoints(config, timeout=_timeout, interval=_interval)

    from cage.cli.ui.progress_reporter import (
        create_run_reporter_with_contract,
        print_inspector_hint,
        print_run_results,
    )

    def _show_results(*, interrupted: bool = False) -> None:
        # Reuse the inspector's shared data layer to print a per-trial results
        # table. Best-effort: never let result rendering break the run command.
        try:
            from cage.contracts.logging import clear_active_progress_line
            if interrupted:
                clear_active_progress_line()
            print_run_results(run_dirs_for_config(config), interrupted=interrupted)
        except Exception:  # noqa: BLE001
            pass

    def _inspect_command() -> str:
        # Durable re-open command: the run dir survives, so `cage inspect <dir>`
        # works even after the managed board is stopped. Use the common root when
        # several agents each got their own run dir.
        dirs = run_dirs_for_config(config)
        if not dirs:
            return ""
        if len(dirs) == 1:
            return f"cage inspect {dirs[0]}"
        import os.path as _osp
        try:
            root = _osp.commonpath([str(d) for d in dirs])
        except ValueError:
            root = ""
        return f"cage inspect {root or dirs[0]}"

    def _show_inspector_hint(run_summary: dict | None) -> None:
        # End-of-run reminder: the live browser URL (the managed board outlives
        # the run) plus the durable `cage inspect` command. Best-effort.
        try:
            from cage.cli.ui.run import browser_urls_from_summary
            print_inspector_hint(browser_urls_from_summary(run_summary or {}), _inspect_command())
        except Exception:  # noqa: BLE001
            pass

    try:
        summary = run_experiment(config, make_reporter=create_run_reporter_with_contract)
    except ResumeCompatibilityError as exc:
        raise click.UsageError(str(exc)) from exc
    except KeyboardInterrupt:
        # Rare fallback: a KeyboardInterrupt that escaped the run's own SIGINT
        # handler (e.g. raised before handlers were installed). The graceful
        # drain and forced-exit Ctrl+C paths are handled inside run_experiment.
        _show_results(interrupted=True)
        _show_inspector_hint(None)
        raise SystemExit(130)

    # Per-trial results table — shown in every mode, including the live UI path
    # that otherwise returns before any summary. The graceful Ctrl+C drain
    # returns a summary with status "interrupted", so reflect that in the header.
    run_interrupted = str(summary.get("status") or "") == "interrupted"
    _show_results(interrupted=run_interrupted)

    if config.logging.terminal_ui:
        _show_inspector_hint(summary)
        return

    click.echo("\n" + "=" * 60)
    click.echo(f"Experiment: {summary['experiment']}")
    click.echo(f"Run ID: {summary['run_id']}")
    click.echo("=" * 60)

    agents_summary = summary.get("agents", {})
    for agent_label, agent_summary in agents_summary.items():
        click.echo(f"\n  {agent_label}:")
        click.echo(f"    Total: {agent_summary['total']}")
        click.echo(f"    Completed: {agent_summary['completed']}")
        click.echo(f"    Failed: {agent_summary['failed']}")
        if agent_summary.get("mean_scores"):
            for metric, value in agent_summary["mean_scores"].items():
                click.echo(f"    {metric}: {value}")

    agent_run_dirs = [
        a["run_dir"] for a in agents_summary.values() if a.get("run_dir")
    ]
    if len(agent_run_dirs) <= 1:
        run_dir_str = summary.get("run_dir") or (agent_run_dirs[0] if agent_run_dirs else "")
        dashboard_str = summary.get("dashboard_path") or (
            str(Path(run_dir_str) / "dashboard.json") if run_dir_str else ""
        )
        if dashboard_str:
            click.echo(f"\n  Dashboard: {dashboard_str}")
        if run_dir_str:
            click.echo(f"  View with: cage inspect {run_dir_str}")
    else:
        click.echo("\n  Dashboards:")
        for agent_label, agent_summary in agents_summary.items():
            rd = agent_summary.get("run_dir")
            if rd:
                click.echo(f"    {agent_label}: cage inspect {rd}")
        import os.path as _osp
        try:
            inspect_root = _osp.commonpath(agent_run_dirs)
        except ValueError:
            inspect_root = ""
        if inspect_root:
            click.echo(f"\n  Inspect all: cage inspect {inspect_root}")
    click.echo()
