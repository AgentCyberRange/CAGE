"""Unit tests for the serve-native submission log (``cage.target.server.serve_log``).

Docker-free, filesystem-only: exercises the digests, pass-id keying, label
sanitization, input persistence (the findings-survive-for-rejudge fix), and the
record round-trip that make a serve submission a self-describing, re-scorable
record.
"""
from __future__ import annotations


from cage.target.server import serve_log as sl


# --- digests --------------------------------------------------------------- #


def test_digest_tree_empty_or_missing_is_blank(tmp_path):
    assert sl.digest_tree(tmp_path / "does-not-exist") == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sl.digest_tree(empty) == ""


def test_digest_tree_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "x.json").write_text("hello", encoding="utf-8")
    (a / "sub" / "y.txt").write_text("world", encoding="utf-8")
    d1 = sl.digest_tree(a)
    assert d1.startswith("sha256:")
    # recomputing is identical
    assert sl.digest_tree(a) == d1
    # changing content changes the digest
    (a / "x.json").write_text("HELLO", encoding="utf-8")
    assert sl.digest_tree(a) != d1


def test_compute_inputs_digest_deterministic_and_distinct():
    d1 = sl.compute_inputs_digest("evidence", "sha256:abc")
    assert d1 == sl.compute_inputs_digest("evidence", "sha256:abc")
    assert d1 != sl.compute_inputs_digest("evidence", "sha256:def")
    assert d1 != sl.compute_inputs_digest("EVIDENCE", "sha256:abc")


# --- pass id / signature --------------------------------------------------- #


def test_judge_signature_and_pass_id():
    assert sl.judge_signature(()) == "verifier-only"
    assert sl.judge_signature(("m1",)) == "m1"
    assert sl.judge_signature(("m1", "m2")) == "m1+m2"
    assert sl.make_pass_id(("m1",), "2026-07-10T00:00:00Z") == "m1@2026-07-10T00:00:00Z"
    assert sl.make_pass_id((), "2026-07-10T00:00:00Z") == "verifier-only@2026-07-10T00:00:00Z"


# --- label / layout -------------------------------------------------------- #


def test_sanitize_label_makes_fs_safe_and_truncates():
    assert sl.sanitize_label("my exp/A!") == "my_exp_A_"
    assert sl.sanitize_label("") == ""
    assert sl.sanitize_label("   ") == ""
    assert len(sl.sanitize_label("x" * 100)) == 40


def test_submission_leaf_prefixes_label_but_keeps_id():
    assert sl.submission_leaf("chal__run_ab", "") == "chal__run_ab"
    assert sl.submission_leaf("chal__run_ab", "exp1") == "exp1__chal__run_ab"


def test_submission_dir_composes_root_client_leaf(tmp_path):
    d = sl.submission_dir(tmp_path, "client_a", "chal__run_ab", "exp1")
    assert d == tmp_path / "client_a" / "exp1__chal__run_ab"


# --- input persistence (the re-judge precondition) ------------------------- #


def test_persist_inputs_copies_findings_and_writes_frozen_inputs(tmp_path):
    sub = tmp_path / "sub"
    agent_out = tmp_path / "ao"
    (agent_out / "final_answer").mkdir(parents=True)
    (agent_out / "final_answer" / "001.json").write_text('{"v": 1}', encoding="utf-8")

    meta = sl.persist_inputs(
        sub, sample={"id": "s"}, output="out", evidence="EV", agent_output_dir=agent_out
    )

    assert (sub / "task_output.json").is_file()
    assert (sub / "runtime" / "check_done_output.txt").read_text() == "EV"
    # findings survive into the submission dir (offline re-judge has its inputs)
    assert (sub / "workspace" / "final_answer" / "001.json").is_file()
    assert meta.has_findings is True
    assert meta.findings_ref == "workspace/final_answer"
    assert meta.findings_digest.startswith("sha256:")


def test_persist_inputs_without_findings_is_honest(tmp_path):
    sub = tmp_path / "sub"
    meta = sl.persist_inputs(
        sub, sample={"id": "s"}, output="", evidence="", agent_output_dir=None
    )
    assert (sub / "runtime" / "check_done_output.txt").read_text() == ""
    assert meta.has_findings is False
    assert meta.findings_digest == ""


# --- record round-trip ----------------------------------------------------- #


def test_write_and_load_record_round_trips(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    grader = sl.Grader(dataset_commit="deadbeef", verifier_kind="vuln_scripts",
                       scoring_assets_digest="sha256:aa")
    prov = sl.PassProvenance(
        mode="serve", challenge_id="c", target_launch_id="inst1",
        project_name="projX", client_id="local", inputs_digest="sha256:in",
        grader=grader, judge_models=("m1", "m2"), score_time="2026-07-10T00:00:00Z",
    )
    passid = sl.make_pass_id(("m1", "m2"), "2026-07-10T00:00:00Z")
    rec = sl.ServeSubmissionRecord(
        submission_id="c__inst1_ab",
        client_id="local",
        challenge_id="c",
        benchmark_module="/x/benchmark.py",
        created_at="2026-07-10T00:00:00Z",
        target_launch=sl.TargetLaunch(launch_id="inst1", challenge_id="c",
                                      project_name="projX", target_info={"web": {"port": 80}}),
        prompt_served=sl.PromptServed(level="l1", digest="sha256:pp"),
        submission=sl.Submission(findings_ref="workspace/final_answer",
                                 findings_digest="sha256:ff", has_findings=True),
        grader=grader,
        inputs_digest="sha256:in",
        canonical_pass_id=passid,
        label="exp1",
        passes=(sl.ScorePass(pass_id=passid, judge_models=("m1", "m2"),
                             score_time="2026-07-10T00:00:00Z",
                             score_ref="scores/stub.json",
                             scores={"stub": {"value": 1.0}}, provenance=prov),),
    )
    path = sl.write_record(sub, rec)
    assert path == sub / "record.json"

    loaded = sl.load_record(sub)
    assert loaded is not None
    assert loaded.submission_id == "c__inst1_ab"
    assert loaded.label == "exp1"
    assert loaded.target_launch.launch_id == "inst1"
    assert loaded.target_launch.target_info == {"web": {"port": 80}}
    assert loaded.grader.dataset_commit == "deadbeef"
    assert loaded.inputs_digest == "sha256:in"
    assert loaded.canonical_pass_id == passid
    assert len(loaded.passes) == 1
    p = loaded.passes[0]
    assert p.judge_models == ("m1", "m2")           # tuple, not list
    assert p.provenance.judge_models == ("m1", "m2")
    assert p.provenance.grader.verifier_kind == "vuln_scripts"


def test_load_record_absent_is_none(tmp_path):
    assert sl.load_record(tmp_path / "nope") is None
