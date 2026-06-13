import json
from pathlib import Path

import pytest

from cage.experiment.model import (
    build_experiment_plan,
    create_experiment_record,
    create_trial_records,
    experiment_plan_to_json,
    experiment_plan_to_mapping,
    experiment_record_to_json,
    experiment_record_to_mapping,
    load_experiment_spec,
    trial_record_to_mapping,
)


def _write_side_effect_benchmark(project_dir: Path) -> None:
    benchmark_py = project_dir / "benchmark.py"
    benchmark_py.write_text(
        """
raise RuntimeError("benchmark module should not be imported by spec planning")
""".lstrip(),
        encoding="utf-8",
    )


def _write_contract_project(project_dir: Path) -> Path:
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: contract-smoke
  run_id: contract-run-001

proxy:
  enabled: true
  request_timeout: 180
  upstream_http_proxy: http://10.1.2.3:7890

eval:
  limit: 2
  benchmark:
    module: ./benchmark.py
    class: DemoBenchmark
    benchmark_root: ./datasets
    prompt_levels: [l0, l2]

target:
  enabled: true
  startup_timeout: 120
  compose_up_timeout: 240

runtime:
  timeout: 600
  max_trials_global: 3
  max_target_setups: 2
  passk: 2
  max_rounds: 7
  max_input_tokens: 1000
  max_output_tokens: 2000
  max_cost: 3.5

agents:
  - id: claude_code
    kind: claude_code
    models:
      - deepseek-v4-pro
      - id: glm-5.1
        max_concurrent: 2
    max_concurrent: 4
  - id: codex
    kind: codex
    model: gpt-5.5
""".lstrip(),
        encoding="utf-8",
    )
    return project_file


def test_load_experiment_spec_does_not_import_or_setup_benchmark(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)

    spec = load_experiment_spec(project_file)

    assert spec.identity.experiment_id == "contract-smoke"
    assert spec.identity.run_id == "contract-run-001"
    assert spec.benchmark.module == "./benchmark.py"
    assert spec.workload.variants == {"prompt_level": ("l0", "l2")}
    assert spec.runtime.proxy.upstream_http_proxy == "http://10.1.2.3:7890"


def test_build_experiment_plan_expands_subjects_tasks_and_trials(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    spec = load_experiment_spec(
        project_file,
        sample_ids=("pb-siyucms",),
    )

    plan = build_experiment_plan(spec)

    assert plan.source.project_file == project_file.resolve()
    assert [subject.subject_id for subject in plan.subjects] == [
        "claude_code:deepseek-v4-pro:stateless",
        "claude_code:glm-5.1:stateless",
        "codex:gpt-5.5:stateless",
    ]
    assert [task.task_id for task in plan.tasks] == [
        "pb-siyucms:prompt_level=l0",
        "pb-siyucms:prompt_level=l2",
    ]
    assert len(plan.trials) == 12
    # trial_id is the runtime id (no subject prefix); the subject is carried in
    # the separate subject_id field, so two subjects running the same task share
    # a trial_id but remain distinct plan rows.
    assert plan.trials[0].trial_id == "pb-siyucms:prompt_level=l0/pass_1"
    assert plan.trials[0].subject_id == "claude_code:deepseek-v4-pro:stateless"
    assert plan.trials[-1].trial_id == "pb-siyucms:prompt_level=l2/pass_2"
    assert plan.trials[-1].subject_id == "codex:gpt-5.5:stateless"
    assert plan.controls.max_rounds == 7
    assert plan.controls.max_input_tokens == 1000
    assert plan.plan_id == build_experiment_plan(spec).plan_id


def test_build_experiment_plan_applies_sample_cap_before_variant_expansion(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    spec = load_experiment_spec(
        project_file,
        sample_ids=("pb-siyucms", "pb-wordpress"),
        max_sample_num=1,
    )

    plan = build_experiment_plan(spec)

    assert [task.source_sample_id for task in plan.tasks] == [
        "pb-siyucms",
        "pb-siyucms",
    ]
    assert len(plan.trials) == 12


def test_experiment_plan_serializes_to_json_ready_mapping(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    spec = load_experiment_spec(project_file, sample_ids=("pb-siyucms",))
    plan = build_experiment_plan(spec)

    mapping = experiment_plan_to_mapping(plan)
    encoded = experiment_plan_to_json(plan)

    assert json.loads(json.dumps(mapping)) == mapping
    assert json.loads(encoded) == mapping
    assert mapping["schema_version"] == "experiment_plan.v1"
    assert mapping["plan_id"] == plan.plan_id
    assert mapping["source"]["project_file"] == str(project_file.resolve())
    assert mapping["subjects"][0]["subject_id"] == "claude_code:deepseek-v4-pro:stateless"


def test_experiment_plan_id_ignores_local_project_file_path(tmp_path: Path) -> None:
    left_dir = tmp_path / "left" / "project"
    right_dir = tmp_path / "right" / "project"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    _write_side_effect_benchmark(left_dir)
    _write_side_effect_benchmark(right_dir)
    left_project = _write_contract_project(left_dir)
    right_project = _write_contract_project(right_dir)

    left = build_experiment_plan(
        load_experiment_spec(left_project, sample_ids=("pb-siyucms",))
    )
    right = build_experiment_plan(
        load_experiment_spec(right_project, sample_ids=("pb-siyucms",))
    )

    assert left.source.project_file != right.source.project_file
    assert left.plan_id == right.plan_id


def test_create_experiment_record_from_plan_is_json_ready(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    plan = build_experiment_plan(
        load_experiment_spec(project_file, sample_ids=("pb-siyucms",))
    )

    record = create_experiment_record(
        plan,
        run_id="record-smoke",
        created_at="2026-06-05T00:00:00Z",
    )
    mapping = experiment_record_to_mapping(record)

    assert json.loads(json.dumps(mapping)) == mapping
    assert json.loads(experiment_record_to_json(record)) == mapping
    assert mapping["schema_version"] == "experiment_record.v1"
    assert mapping["run_id"] == "record-smoke"
    assert mapping["status"] == "planned"
    assert mapping["plan_ref"] == "experiment_plan.json"
    assert mapping["trials"]["total"] == len(plan.trials)
    assert mapping["trials"]["completed"] == 0
    assert mapping["trials"]["records"][0]["trial_id"] == plan.trials[0].trial_id
    assert mapping["subjects"][0]["subject_id"] == plan.subjects[0].subject_id
    assert mapping["record_id"] == create_experiment_record(
        plan,
        run_id="record-smoke",
        created_at="different timestamp does not affect record id",
    ).record_id


def test_create_trial_records_from_plan_are_planned_and_serializable(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    plan = build_experiment_plan(
        load_experiment_spec(project_file, sample_ids=("pb-siyucms",))
    )

    records = create_trial_records(plan, run_id="record-smoke")
    first = trial_record_to_mapping(records[0])

    assert len(records) == len(plan.trials)
    assert first["schema_version"] == "trial_record.v1"
    assert first["run_id"] == "record-smoke"
    assert first["trial_id"] == plan.trials[0].trial_id
    assert first["subject_id"] == plan.trials[0].subject_id
    assert first["task_id"] == plan.trials[0].task_id
    assert first["status"] == "planned"
    assert first["termination"] == {
        "reason": None,
        "signal": None,
        "exit_code": None,
    }
    assert first["scoring"]["status"] == "not_scored"


def test_build_experiment_plan_requires_samples_for_side_effect_free_tasks(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_side_effect_benchmark(project_dir)
    project_file = _write_contract_project(project_dir)
    spec = load_experiment_spec(project_file)

    with pytest.raises(ValueError, match="sample ids"):
        build_experiment_plan(spec)


def test_spec_mapper_defaults_match_resolved_section_defaults() -> None:
    """One project mapping, two readers — their fallback defaults must agree.

    ``resolve()`` (the live run) and ``experiment_spec_from_project_mapping``
    (run.spec, dry-run, canonical snapshots) read the same raw mapping with
    independently written fallbacks. If a default drifts, the recorded spec
    silently disagrees with what actually executed. Pin the shared fields to
    the authoritative ``cage.config.sections`` defaults.
    """

    from cage.config.sections import ExecutionConfig, ProxyConfig
    from cage.experiment.model import experiment_spec_from_project_mapping

    spec = experiment_spec_from_project_mapping(
        {}, project_file=Path("project.yml"), base_dir=Path(".")
    )
    assert spec.runtime.proxy.request_timeout_s == ProxyConfig().request_timeout
    assert spec.runtime.timeouts.request_timeout_s == ProxyConfig().request_timeout
    assert spec.runtime.proxy.enabled == ProxyConfig().enabled
    assert spec.workload.passk == ExecutionConfig().passk
    assert spec.runtime.scheduler.max_trials_global == ExecutionConfig().max_trials_global
    assert spec.runtime.timeouts.trial_timeout_s == ExecutionConfig().timeout
    assert spec.protocol.max_rounds == ExecutionConfig().max_rounds


def test_resolve_carries_the_spec_for_canonical_snapshots(tmp_path: Path) -> None:
    """``run.spec`` is the same parse the canonical snapshot derives from.

    The benchmark variant axes must survive into the carried spec — the old
    post-hoc projection recorded ``variants={}`` for every run, so variant
    provenance was silently lost.
    """

    from cage.experiment.model import experiment_spec_from_project_mapping

    raw = {
        "project": {"name": "contract-demo"},
        "eval": {
            "benchmark": {
                "module": "./benchmark.py",
                "prompt_levels": ["l1", "l2"],
            }
        },
        "runtime": {"passk": 2},
    }
    spec = experiment_spec_from_project_mapping(
        raw, project_file=tmp_path / "project.yml", base_dir=tmp_path
    )
    assert spec.workload.variants == {"prompt_level": ("l1", "l2")}
    assert spec.workload.passk == 2
    assert spec.identity.experiment_id == "contract-demo"


def test_resolved_sections_derive_from_the_spec() -> None:
    """The resolved sections take overlapping fields FROM the spec.

    One parse: legacy aliases and edge values must flow through the spec
    mapper into the resolved sections — these used to be parsed twice with
    diverging semantics (0-token budgets recorded as 0 but executed as
    unlimited; ``max_running_trials`` honoured by the run but recorded as 1).
    """

    from cage.config.experiment import _resolve_execution, _resolve_proxy
    from cage.experiment.model import experiment_spec_from_project_mapping

    raw = {
        "project": {"name": "derive-demo"},
        "runtime": {
            "max_running_trials": 8,        # legacy max_trials_global alias
            "max_sample_target_setups": 3,  # legacy max_target_setups alias
            "timeout_seconds": "300",       # legacy timeout alias, string-typed
            "max_rounds": 7,
            "max_input_tokens": 0,          # 0 = unlimited -> None everywhere
            "max_output_tokens": 50,
            "max_cost": 0,                  # 0 = unlimited -> None everywhere
            "passk": 2,
            "max_trial": 4,
        },
        "proxy": {
            "enabled": False,
            "request_timeout": 120,
            "upstream_http_proxy": "http://10.0.0.1:7890",
        },
    }
    spec = experiment_spec_from_project_mapping(
        raw, project_file=Path("project.yml"), base_dir=Path(".")
    )
    execution = _resolve_execution(raw, spec)
    proxy = _resolve_proxy(raw, spec)

    assert spec.runtime.scheduler.max_trials_global == 8
    assert execution.max_trials_global == 8
    assert spec.runtime.scheduler.max_target_setups == 3
    assert execution.max_target_setups == 3
    assert execution.timeout == 300.0
    assert execution.max_rounds == spec.protocol.max_rounds == 7
    assert spec.protocol.max_input_tokens is None
    assert execution.max_input_tokens is None
    assert execution.max_output_tokens == spec.protocol.max_output_tokens == 50
    assert spec.protocol.max_cost is None and execution.max_cost is None
    assert execution.passk == spec.workload.passk == 2
    assert execution.max_trial == 4
    assert proxy.enabled is False
    assert proxy.request_timeout == spec.runtime.proxy.request_timeout_s == 120.0
    assert proxy.upstream_http_proxy == "http://10.0.0.1:7890"
