"""Unit tests for the serve-only submission scoring closure (docker-free).

Covers the pure helpers of ``cage.target.server.submit`` — benchmark-module
resolution, sample reconstruction, and score serialization — plus an
end-to-end ``score_submission`` against a stub benchmark whose scorer records
what it received. No docker, no network: the stub scorer's ``gather``/``score``
are deterministic, so this exercises the closure's wiring (sample shape,
GatherRuntime, persistence, ScoringContext) without a live target.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cage.contracts import RUNTIME_STATE_KEY
from cage.target.server import submit as submit_mod
from cage.target.server.submit import (
    SubmissionError,
    _prompt_level_to_hint,
    reconstruct_sample,
    render_task_prompt,
    resolve_benchmark_module,
    score_submission,
)


def test_resolve_benchmark_module_walks_up_to_benchmark_py(tmp_path):
    bench_dir = tmp_path / "examples" / "demo_bench"
    challenge_dir = bench_dir / "datasets" / "web" / "chal1"
    challenge_dir.mkdir(parents=True)
    (bench_dir / "benchmark.py").write_text("# stub", encoding="utf-8")

    resolved = resolve_benchmark_module({"full_path": str(challenge_dir)})
    assert resolved == bench_dir / "benchmark.py"


def test_resolve_benchmark_module_missing_raises(tmp_path):
    challenge_dir = tmp_path / "no_benchmark_here"
    challenge_dir.mkdir()
    with pytest.raises(SubmissionError):
        resolve_benchmark_module({"full_path": str(challenge_dir)})


def test_resolve_benchmark_module_no_full_path_raises():
    with pytest.raises(SubmissionError):
        resolve_benchmark_module({})


def test_reconstruct_sample_builds_target_info_and_metadata():
    class _Svc:  # ServiceInfo duck-type
        def __init__(self, **kw):
            self.__dict__.update(kw)

    instance = {
        "chal_id": "pb-demo",
        "project_name": "proj_abc",
        "network_name": "net_abc",
        "scoring": {"k": "v"},
        "services": [
            _Svc(service_name="evaluator", alias="evaluator",
                 inner_ip="172.31.0.5", inner_port=9091,
                 external_host="127.0.0.1", external_port=54215),
            _Svc(service_name="web", alias="web",
                 inner_ip="172.31.0.4", inner_port=80,
                 external_host="127.0.0.1", external_port=None),
        ],
    }
    challenge = {
        "id": "pb-demo",
        "name": "demo",
        "source_fields": {
            "verifier_kind": "vuln_scripts",
            "vulnerabilities": [{"vuln_id": "demo-001", "scoring": ["verifier"]}],
            "marker_stages": [],
            "marker_services": [],
            "marker_user_path": "",
            "marker_root_path": "",
        },
    }

    sample = reconstruct_sample("pb-demo_run1", instance, challenge)

    # target_info keyed by service name; evaluator carries host-published addr
    assert sample["target_info"]["evaluator"]["external_host"] == "127.0.0.1"
    assert sample["target_info"]["evaluator"]["external_port"] == 54215
    assert sample["target_info"]["web"]["inner_ip"] == "172.31.0.4"
    # runtime_state carries the compose project name for docker-cp markers
    assert sample[RUNTIME_STATE_KEY]["project_name"] == "proj_abc"
    assert sample[RUNTIME_STATE_KEY]["run_id"] == "pb-demo_run1"
    # metadata carries the scorer's vuln/marker inputs
    assert sample["metadata"]["verifier_kind"] == "vuln_scripts"
    assert sample["metadata"]["vulnerabilities"][0]["vuln_id"] == "demo-001"


def test_reconstruct_sample_accepts_dict_services():
    instance = {
        "chal_id": "c",
        "project_name": "p",
        "services": [
            {"service_name": "evaluator", "external_host": "127.0.0.1",
             "external_port": 1234, "inner_ip": "10.0.0.2", "inner_port": 9091},
        ],
    }
    sample = reconstruct_sample("c_r", instance, {"id": "c", "source_fields": {}})
    assert sample["target_info"]["evaluator"]["external_port"] == 1234


def test_score_to_dict_handles_dataclass_and_ducktype():
    @dataclasses.dataclass
    class _Score:
        value: float
        answer: str = ""
        explanation: str = ""
        metadata: dict = dataclasses.field(default_factory=dict)

    assert submit_mod._score_to_dict(_Score(1.0, "a", "e"))["value"] == 1.0

    class _Duck:
        value = 0.5
        answer = "x"
        explanation = "y"
        metadata = {"m": 1}

    d = submit_mod._score_to_dict(_Duck())
    assert d["value"] == 0.5 and d["metadata"] == {"m": 1}


# --- task-prompt rendering (serve counterpart of {task_instruction}) --- #


def test_prompt_level_to_hint_clamps():
    got = [_prompt_level_to_hint(x) for x in ["l0", "l1", "l2", "2", "l9", "lX", None, ""]]
    assert got == [0, 1, 2, 2, 2, 0, 0, 0]


def test_render_task_prompt_threads_operator_level_and_agent_input(tmp_path):
    # Uses the stub benchmark's build_prompt, which echoes the sample.
    challenge = _write_stub_benchmark(tmp_path)
    challenge["task_profile"] = "pentest_remote"
    challenge["source_fields"]["agent_input"] = {"application_targets": "http://t:80"}
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}

    task_prompt, template = render_task_prompt("rid", instance, challenge, prompt_level="l2")
    assert "http://t:80" in task_prompt        # agent_input from source_fields flows in
    assert "hint=2" in task_prompt             # OPERATOR level is threaded, not agent-chosen
    assert "profile=pentest_remote" in task_prompt
    # the template is the same briefing with the target masked to a placeholder
    assert "http://t:80" not in template
    assert "{{APPLICATION_TARGETS}}" in template


def test_render_task_prompt_empty_when_no_build_prompt(tmp_path):
    # A benchmark whose module raises on load → best-effort empty, never throws.
    challenge = {
        "id": "x", "name": "x",
        "full_path": str(tmp_path / "nope" / "chal"),
        "source_fields": {},
    }
    assert render_task_prompt("rid", {"chal_id": "x", "services": []}, challenge) == ("", "")


# --- end-to-end closure against a stub benchmark (no docker) --- #

_STUB_BENCHMARK = '''
import json
from cage.benchmarks.base import Benchmark
from cage.scoring.scorer import Scorer
from cage.contracts import RUNTIME_STATE_KEY

class Score:
    def __init__(self, value, answer="", explanation="", metadata=None):
        self.value = value
        self.answer = answer
        self.explanation = explanation
        self.metadata = metadata or {}

class _StubScorer(Scorer):
    def __init__(self, judge=None):
        self.judge = judge
    def gather(self, runtime):
        # record what the closure handed us as the "evidence"
        rs = runtime.sample.get(RUNTIME_STATE_KEY, {})
        return json.dumps({
            "project_name": rs.get("project_name"),
            "agent_output_dir_set": runtime.agent_output_dir is not None,
            "container_is_none": runtime.container is None,
        })
    def score(self, ctx):
        return {"stub": Score(value=1.0, answer="ok", explanation=ctx.check_done_output)}

class StubBenchmark(Benchmark):
    def __init__(self, judge=None):
        self._judge = judge
    def iter_samples(self):
        yield {"id": "s0"}
    def prepare_trial(self, sample, workspace_dir):
        return None
    def build_prompt(self, sample):
        ai = sample.get("agent_input") or {}
        return (
            f"TASK {sample.get('id')} profile={sample.get('task_profile')} "
            f"targets={ai.get('application_targets','')} hint={ai.get('hint_level')}"
        )
    def scorer(self):
        return _StubScorer(judge=self._judge)
'''


def _write_stub_benchmark(tmp_path) -> dict:
    bench_dir = tmp_path / "examples" / "stub_bench"
    chal_dir = bench_dir / "datasets" / "chalX"
    chal_dir.mkdir(parents=True)
    (bench_dir / "benchmark.py").write_text(_STUB_BENCHMARK, encoding="utf-8")
    return {
        "id": "stub-chalX",
        "name": "chalX",
        "full_path": str(chal_dir),
        "source_fields": {"verifier_kind": "vuln_scripts", "vulnerabilities": []},
    }


def test_score_submission_end_to_end_persists_inspectable_run(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "projX", "services": []}
    agent_out = tmp_path / "agent_out"
    (agent_out / "final_answer").mkdir(parents=True)
    runs = tmp_path / "runs"

    result = score_submission(
        run_id="stub-chalX_r1",
        agent_output_dir=agent_out,
        instance=instance,
        challenge=challenge,
        runs_root=runs,
        uuid_hex="abcd1234",
        now=0.0,
    )

    # verdict surfaced
    assert result["scores"]["stub"]["value"] == 1.0
    assert result["chal_id"] == "stub-chalX"

    # gather saw the reconstructed sample + serve-only handles (container=None)
    evidence = json.loads(result["evidence"])
    assert evidence["project_name"] == "projX"
    assert evidence["agent_output_dir_set"] is True
    assert evidence["container_is_none"] is True

    # inspectable .cage_runs layout: one experiment per agent (default "local"),
    # this submission is a trial appended to it.
    run_dir = Path(result["run_dir"])
    assert run_dir.parent.name == "serve__local"
    assert run_dir.name == "serve"
    trial_dir = run_dir / "trials" / "stub-chalX__stub-chalX_r1_abcd1234"
    assert (trial_dir / "task_output.json").is_file()
    assert (trial_dir / "runtime" / "check_done_output.txt").is_file()
    assert (trial_dir / "meta.json").is_file()
    assert (trial_dir / "scores" / "stub.json").is_file()

    # score file nested under scorer name (inspector shape); score() received the
    # gather evidence via ScoringContext.check_done_output
    score_json = json.loads((trial_dir / "scores" / "stub.json").read_text())
    assert set(score_json) == {"stub"}
    assert json.loads(score_json["stub"]["explanation"])["project_name"] == "projX"


def test_persisted_run_is_inspector_readable(tmp_path):
    """The persisted trial renders through the real cage inspector loader."""
    from cage.web.data import load_trial_score_details

    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "projX", "services": []}
    agent_out = tmp_path / "ao"; agent_out.mkdir()

    result = score_submission(
        run_id="stub-chalX_r3",
        agent_output_dir=agent_out,
        instance=instance,
        challenge=challenge,
        runs_root=tmp_path / "runs",
        uuid_hex="cafef00d",
        now=0.0,
    )
    trial_dir = Path(result["run_dir"]) / "trials" / "stub-chalX__stub-chalX_r3_cafef00d"

    # the inspector's score loader reads value/answer/explanation from scores/*.json
    details = load_trial_score_details(trial_dir)
    assert "stub" in details
    assert details["stub"]["value"] == 1.0
    assert details["stub"]["answer"] == "ok"


def test_two_submissions_append_to_one_agent_experiment(tmp_path):
    """Two submissions from one agent land as two trials in ONE experiment run."""
    from cage.artifacts.reader import ExperimentArtifactReader

    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "projX", "services": []}
    agent_out = tmp_path / "ao"; agent_out.mkdir()
    runs = tmp_path / "runs"

    r1 = score_submission(
        run_id="inst1", agent_output_dir=agent_out, instance=instance,
        challenge=challenge, agent_id="agent_abc", runs_root=runs,
        uuid_hex="1111", now=0.0,
    )
    r2 = score_submission(
        run_id="inst2", agent_output_dir=agent_out, instance=instance,
        challenge=challenge, agent_id="agent_abc", runs_root=runs,
        uuid_hex="2222", now=0.0,
    )
    # same experiment run dir for both
    assert r1["run_dir"] == r2["run_dir"]
    run_dir = Path(r1["run_dir"])
    assert run_dir == runs / "serve__agent_abc" / "serve"

    # the experiment record now lists BOTH trials, and both render
    record = ExperimentArtifactReader(run_dir).load_record()
    trial_ids = {ref.trial_id for ref in record.trials.records}
    assert trial_ids == {
        "stub-chalX__inst1_1111",
        "stub-chalX__inst2_2222",
    }
    from cage.web.data import load_trial_score_details
    for tid in trial_ids:
        details = load_trial_score_details(run_dir / "trials" / tid)
        assert details["stub"]["value"] == 1.0


def test_distinct_agents_get_distinct_experiments(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = tmp_path / "ao"; agent_out.mkdir()
    runs = tmp_path / "runs"
    a = score_submission(run_id="i", agent_output_dir=agent_out, instance=instance,
                         challenge=challenge, agent_id="agent_a", runs_root=runs,
                         uuid_hex="aa", now=0.0)
    b = score_submission(run_id="i", agent_output_dir=agent_out, instance=instance,
                         challenge=challenge, agent_id="agent_b", runs_root=runs,
                         uuid_hex="bb", now=0.0)
    assert Path(a["run_dir"]).parent.name == "serve__agent_a"
    assert Path(b["run_dir"]).parent.name == "serve__agent_b"
    assert a["run_dir"] != b["run_dir"]


def test_score_submission_injects_judge_config(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = tmp_path / "ao"; agent_out.mkdir()

    result = score_submission(
        run_id="stub-chalX_r2",
        agent_output_dir=agent_out,
        instance=instance,
        challenge=challenge,
        judge={"model_id": "some-model"},
        runs_root=tmp_path / "runs",
        uuid_hex="deadbeef",
        now=0.0,
    )
    # judge config reached the benchmark → scorer (StubScorer stored it; a full
    # pass just needs the closure not to crash when a judge is provided)
    assert result["scores"]["stub"]["value"] == 1.0
