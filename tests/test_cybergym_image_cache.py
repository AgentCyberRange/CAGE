"""Tests for the CyberGym grading-image cache: offline resolution + storage floor.

Covers the two core guarantees:
  1. images resolve local-docker -> local-tar-cache -> fail (never a registry);
  2. free space never drops below ``min_free_gb`` — evict to recover, else abort.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "examples" / "cybergym" / "benchmark.py"


def load_bench_module():
    spec = importlib.util.spec_from_file_location("cybergym_benchmark", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_bench_module()


def _run_result(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------------- #
# Offline resolution: local docker -> tar cache -> fail (no registry pull)
# --------------------------------------------------------------------------- #

def test_ensure_image_present_is_noop(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _run_result(0)  # already loaded
        raise AssertionError(f"unexpected docker call: {cmd}")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    bench._ensure_image("n132/arvo:123-vul", None)
    # only the inspect happened — no load, no pull
    assert all(c[:3] == ["docker", "image", "inspect"] for c in calls)


def test_ensure_image_missing_no_cache_raises(monkeypatch):
    monkeypatch.setattr(
        bench.subprocess, "run",
        lambda cmd, **kw: _run_result(1, stderr="No such image"),
    )
    with pytest.raises(bench._ImageUnavailable):
        bench._ensure_image("n132/arvo:404-vul", None)


def test_ensure_image_loads_from_tar(monkeypatch, tmp_path):
    image = "n132/arvo:55-vul"
    tar = bench._cache_tar_path(image, tmp_path)
    tar.parent.mkdir(parents=True, exist_ok=True)
    tar.write_bytes(b"fake-tar")

    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd[:2])
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _run_result(1)  # not loaded yet
        if cmd[:2] == ["docker", "load"]:
            return _run_result(0)
        return _run_result(0)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    # plenty of free space so the floor never trips
    monkeypatch.setattr(bench, "_free_bytes", lambda *a, **k: 10 ** 13)
    bench._ensure_image(image, tmp_path)
    assert ["docker", "load"] in seen


def test_ensure_image_tar_load_failure_raises(monkeypatch, tmp_path):
    image = "n132/arvo:66-vul"
    tar = bench._cache_tar_path(image, tmp_path)
    tar.parent.mkdir(parents=True, exist_ok=True)
    tar.write_bytes(b"corrupt")

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _run_result(1)
        if cmd[:2] == ["docker", "load"]:
            return _run_result(1, stderr="archive/tar: invalid header")
        return _run_result(0)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "_free_bytes", lambda *a, **k: 10 ** 13)
    with pytest.raises(bench._ImageUnavailable):
        bench._ensure_image(image, tmp_path)


def test_grade_uses_pull_never(monkeypatch, tmp_path):
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _run_result(0, stdout="ok")

    monkeypatch.setattr(bench, "_ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    bench._grade("arvo:99", poc, "vul", cache_dir=None)
    assert "--pull=never" in captured["cmd"]


# --------------------------------------------------------------------------- #
# Storage floor: evict to recover, abort if it cannot clear the floor
# --------------------------------------------------------------------------- #

def _fresh_mgr(monkeypatch, *, min_free_gb=50.0, evict=True):
    mgr = bench._ImageCacheManager()
    mgr.configure(
        cache_dir=None, pinned_images=set(),
        max_evictable_gb=500, evict=evict, min_free_gb=min_free_gb,
    )
    return mgr


def test_ensure_room_noop_when_above_floor(monkeypatch):
    mgr = _fresh_mgr(monkeypatch)
    monkeypatch.setattr(bench, "_free_bytes", lambda *a, **k: 200 * 1024 ** 3)
    monkeypatch.setattr(
        bench, "_managed_image_sizes",
        lambda: (_ for _ in ()).throw(AssertionError("should not query sizes")),
    )
    mgr.ensure_room()  # no raise, no eviction


def test_ensure_room_evicts_then_passes(monkeypatch):
    mgr = _fresh_mgr(monkeypatch)
    mgr.touch("n132/arvo:1-vul")
    mgr.touch("n132/arvo:2-vul")
    free = {"v": 10 * 1024 ** 3}  # start below the 50 GB floor

    monkeypatch.setattr(bench, "_free_bytes", lambda *a, **k: free["v"])
    monkeypatch.setattr(
        bench, "_managed_image_sizes",
        lambda: {"n132/arvo:1-vul": 30 * 1024 ** 3, "n132/arvo:2-vul": 30 * 1024 ** 3},
    )

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "rmi"]:
            free["v"] += 30 * 1024 ** 3  # eviction recovers space
            return _run_result(0)
        return _run_result(0)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    mgr.ensure_room()
    assert free["v"] >= 50 * 1024 ** 3


def test_ensure_room_aborts_when_all_pinned(monkeypatch):
    mgr = bench._ImageCacheManager()
    mgr.configure(
        cache_dir=None, pinned_images={"n132/arvo:1-vul"},
        max_evictable_gb=500, evict=True, min_free_gb=50.0,
    )
    monkeypatch.setattr(bench, "_free_bytes", lambda *a, **k: 10 * 1024 ** 3)
    # the only loaded image is pinned -> not evictable -> cannot recover
    monkeypatch.setattr(
        bench, "_managed_image_sizes",
        lambda: {"n132/arvo:1-vul": 30 * 1024 ** 3},
    )
    monkeypatch.setattr(bench.subprocess, "run", lambda cmd, **kw: _run_result(0))
    with pytest.raises(bench._DiskExhausted):
        mgr.ensure_room()


def test_disk_exhausted_not_image_unavailable():
    # the scorer catches _ImageUnavailable (unscorable) but must let
    # _DiskExhausted propagate as fatal — they are unrelated types.
    assert not issubclass(bench._DiskExhausted, bench._ImageUnavailable)
