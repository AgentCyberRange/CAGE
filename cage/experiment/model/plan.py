"""Side-effect-free execution plan derived from an ``ExperimentSpec``."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cage.contracts.sample_keys import sample_pass_index
from cage.experiment.model._serde import _json_ready, _stable_id
from cage.experiment.model.spec import (
    AgentSelection,
    ExperimentSpec,
    ProtocolControls,
)
from cage.experiment.model.trial import Trial
from cage.experiment.model.trial_id import (
    parse_trial_id,
    runtime_trial_subpath,
)


@dataclass(frozen=True)
class PlanSource:
    """Source metadata explaining how an ``ExperimentPlan`` was derived."""

    project_file: Path
    benchmark_id: str
    cli_overrides: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SubjectPlan:
    """Resolved subject participating in trial expansion."""

    subject_id: str
    agent: str
    kind: str
    profile: str
    model: str
    max_concurrent: int | None = None


@dataclass(frozen=True)
class BenchmarkTaskPlan:
    """Plan-level benchmark task generated without launching benchmark runtime."""

    task_id: str
    source_sample_id: str
    variant_id: str
    axis_values: Mapping[str, str]


@dataclass(frozen=True)
class TrialPlan:
    """One planned trial produced by subject, task, and pass expansion."""

    trial_id: str
    subject_id: str
    task_id: str
    pass_index: int
    runtime_id: str = ""
    """Runtime trial subpath under ``trials/`` (``<task>`` or ``<task>/pass_<n>``).

    The physical directory the trial runner writes to. The canonical record ref
    derives from this so the durable record co-locates with runtime artifacts in
    one trial tree, while ``trial_id`` keeps the subject-qualified canonical id
    as logical identity. Empty falls back to deriving the path from ``trial_id``.
    """


@dataclass(frozen=True)
class ExperimentPlan:
    """Side-effect-free execution plan consumed by dry-run and future runtime."""

    schema_version: str
    plan_id: str
    source: PlanSource
    subjects: tuple[SubjectPlan, ...]
    tasks: tuple[BenchmarkTaskPlan, ...]
    trials: tuple[TrialPlan, ...]
    controls: ProtocolControls


def build_experiment_plan(spec: ExperimentSpec) -> ExperimentPlan:
    """Build a side-effect-free ``ExperimentPlan`` from an ``ExperimentSpec``.

    This planner intentionally supports only task ids already supplied by user
    selection. Full benchmark task discovery will later call the new Layer 2
    ``Benchmark.tasks()`` API, but this first contract slice must not import or
    instantiate benchmarks. Requiring sample ids makes that limitation explicit
    instead of silently performing runtime work.
    """

    if not spec.workload.task_selection.samples:
        raise ValueError(
            "sample ids are required for side-effect-free task planning until "
            "Benchmark.tasks() is available"
        )
    subjects = _subject_plans(spec.workload.subjects)
    sample_ids = spec.workload.task_selection.samples
    if spec.workload.task_selection.max_sample_num is not None:
        sample_cap = max(0, spec.workload.task_selection.max_sample_num)
        sample_ids = sample_ids[:sample_cap]
    tasks = _task_plans(
        sample_ids,
        spec.workload.variants,
    )
    trials = _trial_plans(subjects, tasks, spec.workload.passk)
    if spec.workload.task_selection.max_trial_num is not None:
        trials = trials[: spec.workload.task_selection.max_trial_num]

    source = PlanSource(
        project_file=spec.project_file,
        benchmark_id=spec.benchmark.id,
        cli_overrides=_cli_overrides(spec),
    )
    return create_experiment_plan(
        source=source,
        subjects=subjects,
        tasks=tasks,
        trials=trials,
        controls=spec.protocol,
    )


def plan_from_trial_sequence(
    spec: ExperimentSpec,
    *,
    subject: SubjectPlan,
    trials: Sequence[Trial],
) -> ExperimentPlan:
    """Project the conductor's resolved trial sequence into the canonical plan.

    The canonical plan is derived from the *actual* trial sequence rather than
    re-expanded from the spec, because a ``pre_run`` hook may legitimately
    rewrite the sequence and the recorded plan must match what ran. Trial
    order mirrors the scheduler's resolved order exactly — resume and
    dashboards depend on the recorded indexes and pass-major pass@k ordering.
    """

    tasks: list[BenchmarkTaskPlan] = []
    seen: set[str] = set()
    for trial in trials:
        task_id = _runtime_task_id(trial)
        if task_id in seen:
            continue
        seen.add(task_id)
        tasks.append(
            BenchmarkTaskPlan(
                task_id=task_id,
                source_sample_id=str(trial.sample_id or task_id),
                variant_id="default",
                axis_values={},
            )
        )
    trial_plans = tuple(
        TrialPlan(
            # trial_id IS the runtime id (single on-disk trial tree); the
            # subject is kept in subject_id, never prefixed into the id.
            trial_id=str(trial.id or _runtime_task_id(trial)),
            subject_id=subject.subject_id,
            task_id=_runtime_task_id(trial),
            pass_index=trial_pass_index(trial),
            runtime_id=str(trial.id or _runtime_task_id(trial)),
        )
        for trial in trials
    )
    return create_experiment_plan(
        source=PlanSource(
            project_file=spec.project_file,
            benchmark_id=spec.benchmark.id,
            cli_overrides=(
                (
                    {
                        "path": "workload.task_selection.samples",
                        "value": list(spec.workload.task_selection.samples),
                    },
                )
                if spec.workload.task_selection.samples
                else ()
            ),
        ),
        subjects=(subject,),
        tasks=tuple(tasks),
        trials=trial_plans,
        controls=spec.protocol,
    )


def trial_pass_index(trial: Trial) -> int:
    """Resolved pass index: a structured ``sample["pass_index"]`` wins.

    pass@k expansion stores the structured value before execution, so it takes
    precedence over the ``…/pass_N`` id suffix.
    """

    structured = sample_pass_index(trial.sample)
    if structured is not None:
        return structured
    return parse_trial_id(str(trial.id or ""))[1]


def _runtime_task_id(trial: Trial) -> str:
    """The runtime task id of a trial — its id without the pass suffix."""

    raw = str(trial.id or trial.sample_id or "trial")
    return parse_trial_id(raw)[0]


def create_experiment_plan(
    *,
    source: PlanSource,
    subjects: tuple[SubjectPlan, ...],
    tasks: tuple[BenchmarkTaskPlan, ...],
    trials: tuple[TrialPlan, ...],
    controls: ProtocolControls,
) -> ExperimentPlan:
    """Create an ``ExperimentPlan`` with Cage's canonical stable ``plan_id``.

    Benchmark adapters, compatibility layers, and dry-run code should use this
    constructor instead of assembling ``ExperimentPlan`` manually. Keeping plan
    id derivation centralized makes saved runs comparable even while different
    planning frontends are migrated onto the same contract.
    """

    plan_payload = _plan_hash_payload(
        source=source,
        subjects=subjects,
        tasks=tasks,
        trials=trials,
        controls=controls,
    )
    plan_id = _stable_id("plan", plan_payload)
    return ExperimentPlan(
        schema_version="experiment_plan.v1",
        plan_id=plan_id,
        source=source,
        subjects=subjects,
        tasks=tasks,
        trials=trials,
        controls=controls,
    )


def experiment_plan_to_mapping(plan: ExperimentPlan) -> dict[str, Any]:
    """Return a JSON-ready mapping for a resolved ``ExperimentPlan``.

    This is the public serialization boundary for dry-run, future plan
    snapshots, inspector views, and tests that need to diff plan contents. The
    returned mapping contains only standard JSON-compatible containers and
    scalar values; dataclasses, tuples, and ``Path`` objects are normalized.
    """

    return _json_ready(plan)


def experiment_plan_to_json(plan: ExperimentPlan, *, indent: int | None = 2) -> str:
    """Serialize an ``ExperimentPlan`` to deterministic JSON text.

    The output sorts mapping keys so saved plan snapshots are easy to diff
    across runs. A trailing newline is included because these snapshots are
    intended to be written as ordinary text artifacts.
    """

    return (
        json.dumps(
            experiment_plan_to_mapping(plan),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def _subject_plans(selections: tuple[AgentSelection, ...]) -> tuple[SubjectPlan, ...]:
    """Expand agent/model selections into resolved subject plan rows."""

    subjects: list[SubjectPlan] = []
    for selection in selections:
        for model in selection.models:
            subject_id = f"{selection.agent}:{model.model}:{selection.profile}"
            subjects.append(
                SubjectPlan(
                    subject_id=subject_id,
                    agent=selection.agent,
                    kind=selection.kind,
                    profile=selection.profile,
                    model=model.model,
                    max_concurrent=(
                        model.max_concurrent
                        if model.max_concurrent is not None
                        else selection.max_concurrent
                    ),
                )
            )
    return tuple(subjects)


def _task_plans(
    sample_ids: tuple[str, ...],
    variants: Mapping[str, tuple[str, ...]],
) -> tuple[BenchmarkTaskPlan, ...]:
    """Expand sample ids and benchmark-owned variant axes into task plans."""

    tasks: list[BenchmarkTaskPlan] = []
    variant_rows = _variant_rows(variants)
    for sample_id in sample_ids:
        for variant_id, axis_values in variant_rows:
            task_id = sample_id if variant_id == "default" else f"{sample_id}:{variant_id}"
            tasks.append(
                BenchmarkTaskPlan(
                    task_id=task_id,
                    source_sample_id=sample_id,
                    variant_id=variant_id,
                    axis_values=axis_values,
                )
            )
    return tuple(tasks)


def _variant_rows(
    variants: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, Mapping[str, str]], ...]:
    """Return stable cartesian rows for benchmark-owned variants."""

    if not variants:
        return (("default", {}),)
    axes = [(axis, tuple(values)) for axis, values in variants.items() if values]
    if not axes:
        return (("default", {}),)
    rows: list[tuple[str, Mapping[str, str]]] = []
    for values in itertools.product(*(axis_values for _, axis_values in axes)):
        axis_values = {axis: str(value) for (axis, _), value in zip(axes, values)}
        variant_id = ",".join(f"{axis}={value}" for axis, value in axis_values.items())
        rows.append((variant_id, axis_values))
    return tuple(rows)


def _trial_plans(
    subjects: tuple[SubjectPlan, ...],
    tasks: tuple[BenchmarkTaskPlan, ...],
    passk: int,
) -> tuple[TrialPlan, ...]:
    """Expand subjects, tasks, and pass indices into stable trial ids."""

    trials: list[TrialPlan] = []
    for subject in subjects:
        for task in tasks:
            for pass_index in range(1, max(1, passk) + 1):
                trials.append(
                    TrialPlan(
                        # trial_id IS the runtime id; subject stays in subject_id.
                        trial_id=runtime_trial_subpath(
                            task.task_id, pass_index, passk
                        ),
                        subject_id=subject.subject_id,
                        task_id=task.task_id,
                        pass_index=pass_index,
                        runtime_id=runtime_trial_subpath(
                            task.task_id, pass_index, passk
                        ),
                    )
                )
    return tuple(trials)


def _cli_overrides(spec: ExperimentSpec) -> tuple[Mapping[str, Any], ...]:
    """Record side-effect-free invocation overlays visible in the plan source."""

    overrides: list[Mapping[str, Any]] = []
    if spec.workload.task_selection.samples:
        overrides.append(
            {
                "path": "workload.task_selection.samples",
                "value": list(spec.workload.task_selection.samples),
            }
        )
    if spec.workload.task_selection.max_sample_num is not None:
        overrides.append(
            {
                "path": "workload.task_selection.max_sample_num",
                "value": spec.workload.task_selection.max_sample_num,
            }
        )
    if spec.workload.task_selection.max_trial_num is not None:
        overrides.append(
            {
                "path": "workload.task_selection.max_trial_num",
                "value": spec.workload.task_selection.max_trial_num,
            }
        )
    return tuple(overrides)


def _plan_hash_payload(
    *,
    source: PlanSource,
    subjects: tuple[SubjectPlan, ...],
    tasks: tuple[BenchmarkTaskPlan, ...],
    trials: tuple[TrialPlan, ...],
    controls: ProtocolControls,
) -> Mapping[str, Any]:
    """Return normalized plan content used to derive ``plan_id``.

    Local absolute paths are deliberately excluded from this payload. Two users
    planning the same benchmark workload from different checkout directories
    should get the same ``plan_id``; pathful provenance still remains available
    in ``ExperimentPlan.source`` and in serialized plan snapshots.
    """

    return {
        "schema_version": "experiment_plan.v1",
        "source": {
            "benchmark_id": source.benchmark_id,
            "cli_overrides": source.cli_overrides,
        },
        "subjects": subjects,
        "tasks": tasks,
        "trials": trials,
        "controls": controls,
    }
