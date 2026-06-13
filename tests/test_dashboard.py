"""Unit tests for ``cage.artifacts.dashboard``.

Schema-level only: dataclass serialization, ``load_dashboard_view``
robustness, and the Benchmark ABC default returning None. The
agent_pentest_bench-specific builder is exercised by the e2e CLI test (and the
inspector smoke check in the same suite).
"""

from __future__ import annotations

import json
from pathlib import Path

from cage.benchmarks import Benchmark
from cage.artifacts.dashboard import (
    SCHEMA_VERSION,
    Column,
    Dashboard,
    Section,
    Stat,
    load_dashboard_view,
)
from cage.scoring import Scorer


class _MinimalBench(Benchmark):
    name = "x"

    def iter_samples(self):
        return iter(())

    def prepare_trial(self, container, sample, workspace_dir):  # pragma: no cover
        pass

    def build_prompt(self, sample):
        return ""

    def scorer(self) -> Scorer:  # pragma: no cover
        raise NotImplementedError


def test_default_build_dashboard_returns_none(tmp_path: Path):
    bench = _MinimalBench()
    assert bench.build_dashboard(tmp_path) is None


def test_summary_section_serialization():
    d = Dashboard(
        title="T",
        subtitle="S",
        sections=(
            Section(
                kind="summary",
                title="head",
                stats=(
                    Stat(label="trials", value="3"),
                    Stat(label="score", value="0.5", hint="explain"),
                ),
            ),
        ),
    )
    out = d.to_dict()
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["title"] == "T"
    assert out["sections"][0]["kind"] == "summary"
    stats = out["sections"][0]["stats"]
    assert stats == [
        {"label": "trials", "value": "3"},
        {"label": "score", "value": "0.5", "hint": "explain"},
    ]


def test_table_section_serialization():
    sec = Section(
        kind="table",
        title="trials",
        columns=(
            Column(key="sample", label="Sample"),
            Column(key="score", label="Score", align="right"),
        ),
        rows=(
            {"sample": "a", "score": "0.10"},
            {"sample": "b", "score": "0.20"},
        ),
    )
    out = sec.to_dict()
    assert out["columns"] == [
        {"key": "sample", "label": "Sample", "align": "left"},
        {"key": "score", "label": "Score", "align": "right"},
    ]
    assert out["rows"] == [
        {"sample": "a", "score": "0.10"},
        {"sample": "b", "score": "0.20"},
    ]


def test_table_section_serialization_supports_typed_columns_and_cells():
    sec = Section(
        kind="table",
        title="trials",
        columns=(
            Column(key="score", label="Score", align="right", unit="ratio", hint="Higher is better"),
        ),
        rows=(
            {
                "score": {
                    "display_value": "0.25",
                    "raw_value": 0.25,
                    "sort_key": 0.25,
                    "unit": "ratio",
                    "severity": "warning",
                    "hint": "Partial benchmark result",
                },
            },
        ),
    )

    out = sec.to_dict()

    assert out["columns"] == [
        {
            "key": "score",
            "label": "Score",
            "align": "right",
            "unit": "ratio",
            "hint": "Higher is better",
        },
    ]
    assert out["rows"][0]["score"] == {
        "display_value": "0.25",
        "raw_value": 0.25,
        "sort_key": 0.25,
        "unit": "ratio",
        "severity": "warning",
        "hint": "Partial benchmark result",
    }


def test_note_section_keeps_body():
    out = Section(kind="note", title="n", body="hello\nworld").to_dict()
    assert out["body"] == "hello\nworld"


def test_write_and_load_roundtrip(tmp_path: Path):
    d = Dashboard(title="t", sections=(Section(kind="note", body="x"),))
    p = tmp_path / "run" / "dashboard_view.json"
    d.write(p)
    assert p.exists()
    loaded = load_dashboard_view(tmp_path / "run")
    assert loaded is not None
    assert loaded["title"] == "t"
    assert loaded["sections"][0]["body"] == "x"


def test_load_dashboard_view_missing_returns_none(tmp_path: Path):
    assert load_dashboard_view(tmp_path) is None


def test_load_dashboard_view_malformed_returns_none(tmp_path: Path):
    (tmp_path / "dashboard_view.json").write_text("not json", encoding="utf-8")
    assert load_dashboard_view(tmp_path) is None


def test_load_dashboard_view_wrong_root_type_returns_none(tmp_path: Path):
    (tmp_path / "dashboard_view.json").write_text(
        json.dumps(["not", "a", "dict"]), encoding="utf-8"
    )
    assert load_dashboard_view(tmp_path) is None
