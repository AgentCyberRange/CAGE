from __future__ import annotations

import json

from cage.artifacts.live_success import (
    load_live_success,
    parse_live_checks_success,
    record_live_success,
)


def test_records_and_loads_live_success_without_plain_answer(tmp_path):
    trial_dir = tmp_path / "trials" / "trial-1"

    verdict = record_live_success(
        trial_dir=trial_dir,
        trial_id="trial-1",
        benchmark="nyu_ctf",
        mode="reactive",
        source="submit",
        evidence={"answer_hash": "sha256:abc"},
    )

    loaded = load_live_success(trial_dir)
    assert loaded == verdict
    assert loaded["success"] is True
    assert loaded["mode"] == "reactive"
    assert loaded["source"] == "submit"
    assert "flag{" not in json.dumps(loaded)


def test_parse_live_checks_success_requires_correct_true():
    entry = parse_live_checks_success(
        '{"correct": true, "source": "container-submit", "answer_hash": "sha256:abc"}'
    )

    assert entry is not None
    assert entry["source"] == "container-submit"
    assert parse_live_checks_success('{"correct": false}') is None
    assert parse_live_checks_success("not json") is None


