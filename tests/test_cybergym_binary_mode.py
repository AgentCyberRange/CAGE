"""Tests for CyberGym binary-only grading (``binary_dir`` set).

Binary mode mirrors upstream ``run_container_binary``: a small pre-pulled base
runner image executes prebuilt vul/fix binaries bind-mounted from
``<binary_dir>/<subset>/<subid>/<mode>/`` — no per-task image, no tar cache, no
eviction. These tests assert the docker command is built byte-for-byte like
upstream and that the image-cache machinery is bypassed.
"""

from __future__ import annotations

import importlib.util
import json
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


def _make_arvo_binary(binary_dir: Path, subid: str, mode: str, *, runner: str | None = None):
    d = binary_dir / "arvo" / subid / mode
    (d / "out").mkdir(parents=True, exist_ok=True)
    (d / "libs").mkdir(parents=True, exist_ok=True)
    (d / "arvo").write_text("#!/bin/bash\n/out/xml /tmp/poc\n")
    (d / "out" / "xml").write_bytes(b"\x7fELF")
    if runner is not None:
        (d / "runner").write_text(runner)
    return d


def _make_ossfuzz_binary(binary_dir: Path, subid: str, mode: str, fuzzer: str):
    d = binary_dir / "oss-fuzz" / subid / mode
    (d / "out").mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"fuzz_target": fuzzer}))
    (d / "out" / fuzzer).write_bytes(b"\x7fELF")
    (d / "out" / f"{fuzzer}.options").write_text("[libfuzzer]\n")
    return d


def _capture_docker(monkeypatch, *, returncode=0, stdout="ok"):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _run_result(returncode, stdout=stdout)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    return captured


# --------------------------------------------------------------------------- #
# arvo binary grading
# --------------------------------------------------------------------------- #

def test_grade_binary_arvo_builds_runner_command(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_arvo_binary(bin_dir, "25210", "vul")
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    captured = _capture_docker(monkeypatch)
    code, _out = bench._grade_binary("arvo:25210", poc, "vul", binary_dir=bin_dir)
    cmd = captured["cmd"]
    joined = " ".join(cmd)

    assert code == 0
    # small pre-pulled base runner, NOT the heavy per-task image
    assert bench.DEFAULT_RUNNER_IMAGE in cmd
    assert "n132/arvo" not in joined
    # offline + isolated, exactly like upstream (network_mode="none")
    assert "--pull=never" in cmd
    assert "--network" in cmd and "none" in cmd
    # the prebuilt artifacts are bind-mounted at the upstream targets
    assert "target=/arvo,readonly" in joined
    assert "target=/out-libs,readonly" in joined
    assert "target=/out/xml,readonly" in joined
    assert "target=/tmp/poc,readonly" in joined
    # arvo entrypoint command
    assert "LD_LIBRARY_PATH=/out-libs" in joined
    assert "/bin/bash /arvo" in joined


def test_grade_binary_uses_per_task_runner_file(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_arvo_binary(
        bin_dir, "20050", "vul", runner="cybergym/oss-fuzz-base-runner:20200102"
    )
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    captured = _capture_docker(monkeypatch)
    bench._grade_binary("arvo:20050", poc, "vul", binary_dir=bin_dir)
    # the per-task runner file overrides DEFAULT_RUNNER_IMAGE
    assert "cybergym/oss-fuzz-base-runner:20200102" in captured["cmd"]


def test_grade_binary_ossfuzz_uses_testcase_and_reproduce(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_ossfuzz_binary(bin_dir, "368076871", "vul", "mruby_fuzzer")
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    captured = _capture_docker(monkeypatch)
    bench._grade_binary("oss-fuzz:368076871", poc, "vul", binary_dir=bin_dir)
    joined = " ".join(captured["cmd"])
    # oss-fuzz mounts the PoC at /testcase and runs `reproduce <fuzzer>`
    assert "target=/testcase,readonly" in joined
    assert "target=/out/mruby_fuzzer,readonly" in joined
    assert "reproduce mruby_fuzzer" in joined


# --------------------------------------------------------------------------- #
# exit-code contract (shared with the image path)
# --------------------------------------------------------------------------- #

def test_grade_binary_timeout_normalised_to_zero(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_arvo_binary(bin_dir, "1", "vul")
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    _capture_docker(monkeypatch, returncode=bench.DOCKER_KILL_EXIT)
    code, out = bench._grade_binary("arvo:1", poc, "vul", binary_dir=bin_dir)
    assert code == 0  # SIGKILL-by-timeout is "did not crash"


def test_grade_binary_infra_exit_raises_unavailable(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_arvo_binary(bin_dir, "1", "vul")
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    _capture_docker(monkeypatch, returncode=bench.DOCKER_INFRA_EXIT, stdout="no such image")
    with pytest.raises(bench._ImageUnavailable):
        bench._grade_binary("arvo:1", poc, "vul", binary_dir=bin_dir)


def test_grade_binary_missing_dir_raises_unavailable(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    bin_dir.mkdir()
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")
    # no arvo/404/vul tree -> unscorable, NOT a false crash
    with pytest.raises(bench._ImageUnavailable):
        bench._grade_binary("arvo:404", poc, "vul", binary_dir=bin_dir)


# --------------------------------------------------------------------------- #
# dispatch: _grade(binary_dir=...) routes to binary path, bypasses image cache
# --------------------------------------------------------------------------- #

def test_grade_dispatches_to_binary_and_bypasses_image_cache(monkeypatch, tmp_path):
    bin_dir = tmp_path / "server-data"
    _make_arvo_binary(bin_dir, "7", "vul")
    poc = tmp_path / "poc.bin"
    poc.write_bytes(b"x")

    # if the image path were taken, _ensure_image would run (and fail offline)
    monkeypatch.setattr(
        bench, "_ensure_image",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("image path taken")),
    )
    captured = _capture_docker(monkeypatch)
    code, _ = bench._grade("arvo:7", poc, "vul", cache_dir=None, binary_dir=bin_dir)
    assert code == 0
    assert bench.DEFAULT_RUNNER_IMAGE in captured["cmd"]
