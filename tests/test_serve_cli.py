"""Unit tests for `cage serve <benchmark>` name → source-root resolution."""
from __future__ import annotations

import json

import click
import pytest

from cage.cli.commands import serve as serve_mod


def test_looks_like_challenge_index_distinguishes_index_from_challenge(tmp_path):
    index = tmp_path / "bench.json"
    index.write_text(json.dumps({
        "pb-a": {"benchmark": "b", "challenge": "a", "path": "web/a"},
        "pb-b": {"benchmark": "b", "challenge": "b", "path": "web/b"},
    }))
    challenge = tmp_path / "challenge.json"
    challenge.write_text(json.dumps({"adapter_kind": "challenge_json", "name": "a"}))
    other = tmp_path / "notindex.json"
    other.write_text(json.dumps({"some": "config", "values": [1, 2]}))

    assert serve_mod._looks_like_challenge_index(index) is True
    assert serve_mod._looks_like_challenge_index(challenge) is False  # named challenge.json
    assert serve_mod._looks_like_challenge_index(other) is False      # no path-bearing entries


def test_resolve_benchmark_sources_finds_indices_at_multiple_depths(tmp_path, monkeypatch):
    # fake benchmark: a web index at datasets/ and a post index one level deeper
    examples = tmp_path / "examples"
    bench = examples / "demo_bench" / "datasets"
    (bench / "web").mkdir(parents=True)
    (bench / "post_exploit" / "range-1").mkdir(parents=True)
    (bench / "demo_bench.json").write_text(json.dumps({"d-a": {"path": "web/a"}}))
    (bench / "web" / "a" ).mkdir()
    (bench / "web" / "a" / "challenge.json").write_text(json.dumps({"adapter_kind": "challenge_json"}))
    (bench / "post_exploit" / "post_ranges.json").write_text(json.dumps({"d-r1": {"path": "range-1"}}))
    (bench / "post_exploit" / "range-1" / "challenge.json").write_text(json.dumps({"adapter_kind": "challenge_json"}))

    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: examples)

    sources = serve_mod._resolve_benchmark_sources("demo_bench")
    roots = {s["root"] for s in sources}
    assert all(s["adapter_kind"] == "challenge_json" for s in sources)
    assert roots == {str(bench), str(bench / "post_exploit")}  # both index dirs, challenge.json ignored


def test_resolve_benchmark_sources_skips_duplicate_indices(tmp_path, monkeypatch):
    # A dataset can ship a top-level index AND a per-collection one that re-lists
    # the same challenges (agent_pentest_bench.json vs web_exploit_bench/
    # pentest_bench.json). Only the shallower one should become a root — feeding
    # both to discovery would raise a duplicate-id error.
    examples = tmp_path / "examples"
    ds = examples / "demo_bench" / "datasets"
    (ds / "web_exploit_bench").mkdir(parents=True)
    (ds / "post_exploit_bench").mkdir(parents=True)
    # top-level web index + a duplicate of it one level down (SAME ids)
    web_ids = {"d-a": {"path": "web_exploit_bench/a"}, "d-b": {"path": "web_exploit_bench/b"}}
    (ds / "demo_bench.json").write_text(json.dumps(web_ids))
    (ds / "web_exploit_bench" / "pentest_bench.json").write_text(
        json.dumps({"d-a": {"path": "a"}, "d-b": {"path": "b"}})  # same ids, different paths
    )
    # a genuinely distinct collection (post) — must still be picked up
    (ds / "post_exploit_bench" / "post_ranges.json").write_text(json.dumps({"d-r1": {"path": "range-1"}}))

    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: examples)
    roots = {s["root"] for s in serve_mod._resolve_benchmark_sources("demo_bench")}
    # top-level web index wins; the duplicate under web_exploit_bench is skipped;
    # the distinct post collection is kept.
    assert roots == {str(ds), str(ds / "post_exploit_bench")}
    assert str(ds / "web_exploit_bench") not in roots


def test_resolve_benchmark_judge_reads_declared_default(tmp_path, monkeypatch):
    # The benchmark declares its web judge in a default_*.yml; serving it should
    # pick that up so web scoring works without --judge-model.
    examples = tmp_path / "examples"
    bench = examples / "demo_bench"
    bench.mkdir(parents=True)
    (bench / "default_post.yml").write_text("project:\n  name: demo\n")  # no judge
    (bench / "default_web.yml").write_text(
        "eval:\n  scoring:\n    judge:\n      models:\n        - deepseek-v4-pro\n"
    )
    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: examples)
    assert serve_mod._resolve_benchmark_judge("demo_bench") == {"models": ["deepseek-v4-pro"]}


def test_resolve_benchmark_judge_none_when_absent(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    bench = examples / "demo_bench"
    bench.mkdir(parents=True)
    (bench / "default.yml").write_text("project:\n  name: demo\n")
    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: examples)
    assert serve_mod._resolve_benchmark_judge("demo_bench") is None


def test_resolve_benchmark_sources_missing_benchmark_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: tmp_path / "examples")
    with pytest.raises(click.BadParameter):
        serve_mod._resolve_benchmark_sources("nope")


def test_resolve_benchmark_sources_no_index_raises(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    (examples / "empty_bench" / "datasets").mkdir(parents=True)
    monkeypatch.setattr(serve_mod, "_repo_examples_root", lambda: examples)
    with pytest.raises(click.BadParameter):
        serve_mod._resolve_benchmark_sources("empty_bench")
