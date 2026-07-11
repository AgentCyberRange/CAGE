"""Unit tests for the serve-only submission scoring closure (docker-free).

Covers the pure helpers of ``cage.target.server.submit`` — benchmark-module
resolution, sample reconstruction, and score serialization — plus an
end-to-end ``score_submission`` against a stub benchmark whose scorer records
what it received. No docker, no network: the stub scorer's ``gather``/``score``
are deterministic, so this exercises the closure's wiring (sample shape,
GatherRuntime, persistence, ScoringContext, the serve-native record) without a
live target.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cage.contracts import RUNTIME_STATE_KEY
from cage.target.server import serve_log
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
        # the judge input the offline re-judge needs: findings copied to the
        # submission dir. Echo whether they are readable from ctx.trial_dir.
        fa = ctx.trial_dir / "workspace" / "final_answer"
        found = sorted(p.name for p in fa.glob("*.json")) if fa.is_dir() else []
        return {"stub": Score(value=1.0, answer=",".join(found), explanation=ctx.check_done_output)}

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
    # a grader asset so the record's scoring_assets_digest is non-empty
    (chal_dir / "metadata.json").write_text('{"id": "stub-chalX"}', encoding="utf-8")
    return {
        "id": "stub-chalX",
        "name": "chalX",
        "full_path": str(chal_dir),
        "source_fields": {"verifier_kind": "vuln_scripts", "vulnerabilities": []},
    }


def _agent_out_with_findings(tmp_path, name="ao") -> Path:
    agent_out = tmp_path / name
    fa = agent_out / "final_answer"
    fa.mkdir(parents=True)
    (fa / "001.json").write_text('{"vuln_id": "v1"}', encoding="utf-8")
    return agent_out


def test_score_submission_writes_serve_native_record(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "projX", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)
    serve = tmp_path / "serve"

    result = score_submission(
        run_id="inst-42",
        agent_output_dir=agent_out,
        instance=instance,
        challenge=challenge,
        serve_root=serve,
        uuid_hex="abcd1234",
        now=0.0,
    )

    # verdict surfaced
    assert result["scores"]["stub"]["value"] == 1.0
    assert result["chal_id"] == "stub-chalX"
    assert result["submission_id"] == "stub-chalX__inst-42_abcd1234"

    # serve-native layout: .cage_serve/<client>/<submission_id>/
    sub_dir = Path(result["submission_dir"])
    assert sub_dir == serve / "local" / "stub-chalX__inst-42_abcd1234"
    assert (sub_dir / "record.json").is_file()
    assert (sub_dir / "task_output.json").is_file()
    assert (sub_dir / "runtime" / "check_done_output.txt").is_file()
    assert (sub_dir / "scores" / "stub.json").is_file()

    # gather saw the reconstructed sample + serve-only handles (container=None)
    evidence = json.loads(result["evidence"])
    assert evidence["project_name"] == "projX"
    assert evidence["container_is_none"] is True

    # FINDINGS PERSISTED — the offline-rejudge precondition (and the bug fix):
    # the agent's final_answer survives into the submission dir, and score()
    # could read it (the stub echoes the filenames it saw).
    assert (sub_dir / "workspace" / "final_answer" / "001.json").is_file()
    assert result["scores"]["stub"]["answer"] == "001.json"

    # the record round-trips and is provenance-complete
    rec = serve_log.load_record(sub_dir)
    assert rec is not None
    assert rec.submission_id == "stub-chalX__inst-42_abcd1234"
    assert rec.client_id == "local"
    assert rec.challenge_id == "stub-chalX"
    assert rec.target_launch.launch_id == "inst-42"       # bound to the boot
    assert rec.target_launch.project_name == "projX"
    assert rec.submission.has_findings is True
    assert rec.submission.findings_digest.startswith("sha256:")
    assert rec.inputs_digest.startswith("sha256:")
    assert rec.grader.verifier_kind == "vuln_scripts"
    assert rec.grader.scoring_assets_digest.startswith("sha256:")
    # exactly one canonical pass, verifier-only (no judge configured)
    assert len(rec.passes) == 1
    p = rec.passes[0]
    assert p.pass_id == rec.canonical_pass_id
    assert p.pass_id == "verifier-only@1970-01-01T00:00:00Z"
    assert p.judge_models == ()
    assert p.provenance.mode == "serve"
    assert p.provenance.target_launch_id == "inst-42"
    assert p.provenance.inputs_digest == rec.inputs_digest


def test_label_prefixes_dir_and_is_recorded(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)
    serve = tmp_path / "serve"

    result = score_submission(
        run_id="inst1", agent_output_dir=agent_out, instance=instance,
        challenge=challenge, serve_root=serve, label="my exp/A!", uuid_hex="ff00",
        now=0.0,
    )
    sub_dir = Path(result["submission_dir"])
    # label sanitized and used as the leaf-dir prefix, submission_id embedded
    assert sub_dir.name == "my_exp_A___stub-chalX__inst1_ff00"
    assert result["label"] == "my_exp_A_"
    assert serve_log.load_record(sub_dir).label == "my_exp_A_"
    # submission_id stays pure (no label) for programmatic lookup
    assert result["submission_id"] == "stub-chalX__inst1_ff00"


def test_pass_id_keyed_on_judge_model_and_time(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)

    result = score_submission(
        run_id="inst1", agent_output_dir=agent_out, instance=instance,
        challenge=challenge, serve_root=tmp_path / "serve",
        judge={"model_id": "some-model"}, uuid_hex="aa", now=0.0,
    )
    rec = serve_log.load_record(Path(result["submission_dir"]))
    assert result["pass_id"] == "some-model@1970-01-01T00:00:00Z"
    assert rec.passes[0].judge_models == ("some-model",)
    assert rec.passes[0].provenance.judge_models == ("some-model",)


def test_two_submissions_same_client_two_records(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "projX", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)
    serve = tmp_path / "serve"

    r1 = score_submission(run_id="inst1", agent_output_dir=agent_out, instance=instance,
                          challenge=challenge, agent_id="agent_abc", serve_root=serve,
                          uuid_hex="1111", now=0.0)
    r2 = score_submission(run_id="inst2", agent_output_dir=agent_out, instance=instance,
                          challenge=challenge, agent_id="agent_abc", serve_root=serve,
                          uuid_hex="2222", now=0.0)

    # two distinct submission dirs under the ONE client, each self-describing
    assert r1["submission_dir"] != r2["submission_dir"]
    client_dir = serve / "agent_abc"
    leaves = sorted(p.name for p in client_dir.iterdir())
    assert leaves == ["stub-chalX__inst1_1111", "stub-chalX__inst2_2222"]
    for r in (r1, r2):
        rec = serve_log.load_record(Path(r["submission_dir"]))
        assert rec is not None and rec.client_id == "agent_abc"


def test_distinct_clients_get_distinct_dirs(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)
    serve = tmp_path / "serve"
    a = score_submission(run_id="i", agent_output_dir=agent_out, instance=instance,
                         challenge=challenge, agent_id="agent_a", serve_root=serve,
                         uuid_hex="aa", now=0.0)
    b = score_submission(run_id="i", agent_output_dir=agent_out, instance=instance,
                         challenge=challenge, agent_id="agent_b", serve_root=serve,
                         uuid_hex="bb", now=0.0)
    assert Path(a["submission_dir"]).parent == serve / "agent_a"
    assert Path(b["submission_dir"]).parent == serve / "agent_b"


def test_score_submission_injects_judge_config(tmp_path):
    challenge = _write_stub_benchmark(tmp_path)
    instance = {"chal_id": "stub-chalX", "project_name": "p", "services": []}
    agent_out = _agent_out_with_findings(tmp_path)

    result = score_submission(
        run_id="inst-r2",
        agent_output_dir=agent_out,
        instance=instance,
        challenge=challenge,
        judge={"model_id": "some-model"},
        serve_root=tmp_path / "serve",
        uuid_hex="deadbeef",
        now=0.0,
    )
    # judge config reached the benchmark → scorer (StubScorer stored it; a full
    # pass just needs the closure not to crash when a judge is provided)
    assert result["scores"]["stub"]["value"] == 1.0
