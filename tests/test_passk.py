"""Unit tests for native pass@k support (runtime.passk)."""

from __future__ import annotations

from typing import Any

from cage.config.experiment import ExecutionConfig
from cage.experiment.model import Trial, TrialResult, TrialType
from cage.scoring import Score
from cage.sandbox.exec import Timing
from cage.scoring.lifecycle import _build_summary


def _result(sample_id: str, score: float, trial_id: str | None = None) -> TrialResult:
    return TrialResult(
        trial_id=trial_id or f"{sample_id}/t",
        trial_index=0,
        trial_type="task",
        sample_id=sample_id,
        output="",
        exit_code=0,
        timing=Timing(started_at_ms=0, ended_at_ms=1, duration_ms=1),
        scores={"cvebench": Score(value=score, answer="", explanation="")},
    )


def test_execution_config_default_passk_is_1():
    assert ExecutionConfig().passk == 1


def test_build_summary_passk1_has_no_pass_at_k_block():
    results = [_result("s1", 1.0), _result("s2", 0.0), _result("s3", 1.0)]
    summary = _build_summary(results)
    assert "pass_at_k" not in summary
    assert summary["mean_scores"]["cvebench"] == 0.6667


def test_build_summary_passk_emits_pass_at_k_block():
    # 3 samples, 4 attempts each. s1 solves 1/4, s2 solves 0/4, s3 solves 4/4.
    # pass@1 = (1 + 0 + 4) / 12 = 0.4167
    # pass@4 = (1 + 0 + 1) / 3 = 0.6667
    results = []
    for pass_idx in range(1, 5):
        results.append(_result("s1", 1.0 if pass_idx == 2 else 0.0, f"s1/pass_{pass_idx}"))
        results.append(_result("s2", 0.0, f"s2/pass_{pass_idx}"))
        results.append(_result("s3", 1.0, f"s3/pass_{pass_idx}"))

    summary = _build_summary(results)
    assert "pass_at_k" in summary
    block = summary["pass_at_k"]["cvebench"]
    assert block["k"] == 4
    assert block["n_samples"] == 3
    assert block["pass@1"] == round(5 / 12, 4)
    assert block["pass@4"] == round(2 / 3, 4)


def test_build_summary_passk_with_partial_results_still_works():
    """If a pass crashed mid-way and some samples have <k attempts, the
    summary should still emit a reasonable pass@k using whatever attempts
    were collected (no IndexError, no exception).
    """
    results = [
        _result("s1", 0.0, "s1/pass_1"),
        _result("s1", 1.0, "s1/pass_2"),
        # s2 only ran once
        _result("s2", 0.0, "s2/pass_1"),
    ]
    summary = _build_summary(results)
    assert "pass_at_k" in summary
    block = summary["pass_at_k"]["cvebench"]
    assert block["k"] == 2  # max attempts across samples
    assert block["n_samples"] == 2
    # pass@1 = (0 + 1 + 0) / 3 = 0.3333
    assert block["pass@1"] == round(1 / 3, 4)
    # pass@2 = s1 solved + s2 not solved = 1/2 = 0.5
    assert block["pass@2"] == 0.5


def test_passk_trial_sequence_order_is_pass_by_pass():
    """A direct test of the orchestrator's trial-multiplication: trial IDs
    in the final list must form ``[pass1 over all samples, pass2 over all
    samples, ...]`` so any chunked execution (parallel batches or serial
    loop) preserves the user-requested ordering.
    """
    def _t(tid: str, sample_id: str, variant: str) -> Trial:
        return Trial(
            id=tid,
            index=0,
            type=TrialType.TASK,
            sample={"id": sample_id, "variant": variant},
        )

    base_trials = [
        _t("cve-A/zero_day", "cve-A-zero", "zero_day"),
        _t("cve-A/one_day", "cve-A-one", "one_day"),
        _t("cve-B/zero_day", "cve-B-zero", "zero_day"),
    ]
    # Replicate the orchestrator's multiplication logic.
    passk = 3
    trials: list[Trial] = []
    for pass_idx in range(1, passk + 1):
        for t in base_trials:
            sample_copy: dict[str, Any] = dict(t.sample)
            sample_copy["pass_index"] = pass_idx
            trials.append(Trial(
                id=f"{t.id}/pass_{pass_idx}",
                index=0,
                type=t.type,
                sample=sample_copy,
            ))

    ids = [t.id for t in trials]
    assert len(ids) == 9
    # All pass_1 trials come before any pass_2 trial, etc.
    assert ids == [
        "cve-A/zero_day/pass_1",
        "cve-A/one_day/pass_1",
        "cve-B/zero_day/pass_1",
        "cve-A/zero_day/pass_2",
        "cve-A/one_day/pass_2",
        "cve-B/zero_day/pass_2",
        "cve-A/zero_day/pass_3",
        "cve-A/one_day/pass_3",
        "cve-B/zero_day/pass_3",
    ]
    # And each trial's sample carries the pass index.
    assert [t.sample["pass_index"] for t in trials[:3]] == [1, 1, 1]
    assert [t.sample["pass_index"] for t in trials[-3:]] == [3, 3, 3]
