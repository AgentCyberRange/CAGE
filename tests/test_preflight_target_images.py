"""Pre-flight ``Target images`` check — fail fast before launching a run.

Build-based targets must be built before launch (the server never builds at
launch). This guard reads each planned sample's compose stack host-side and
flags ``build:`` services whose ``image:`` tag is absent locally, so a run with
missing target images is rejected up front instead of failing trial-by-trial
with ``target_unavailable``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


from cage.experiment.engine import preflight
from cage.experiment.engine.preflight import PreflightResult, _check_target_images


def _make_benchmark_root(tmp_path, *, services: str) -> str:
    # The challenge_json adapter discovers from an index file at the root that
    # maps challenge_id -> {path: <dir with challenge.json>}.
    (tmp_path / "index.json").write_text(json.dumps({"mychal": {"path": "mychal"}}))
    chal_dir = tmp_path / "mychal"
    chal_dir.mkdir()
    (chal_dir / "challenge.json").write_text(
        json.dumps(
            {
                "adapter_kind": "challenge_json",
                "challenge_name": "mychal",
                "name": "mychal",
                "compose_files": ["docker-compose.yml"],
            }
        )
    )
    (chal_dir / "docker-compose.yml").write_text(services)
    return str(tmp_path)


_COMPOSE = """
services:
  app:
    build: .
    image: example/app:v1
  evaluator:
    build: .
    image: example/evaluator:v1
  cache:
    image: redis:7
"""


def _config(root: str):
    return SimpleNamespace(
        target=SimpleNamespace(enabled=True),
        benchmark=SimpleNamespace(benchmark_root=root, name="b"),
        metadata={"benchmark_id": "mybench"},
    )


def _discovered_challenge_id(root: str) -> str:
    from cage.target.adapters.roots import normalize_benchmark_sources
    from cage.target.adapters.source_config import build_default_registry

    registry = build_default_registry()
    sources = normalize_benchmark_sources(
        [{"adapter_kind": "challenge_json", "root": root}]
    )
    return next(iter(registry.discover_all(sources)))


def test_missing_build_image_fails(tmp_path, monkeypatch):
    root = _make_benchmark_root(tmp_path, services=_COMPOSE)
    cid = _discovered_challenge_id(root)
    # Only the evaluator image exists locally; the app image is missing.
    monkeypatch.setattr(
        preflight, "_image_exists_local", lambda img: img == "example/evaluator:v1"
    )
    result = PreflightResult()
    _check_target_images(_config(root), [{"id": "s0", "challenge_id": cid}], result)
    assert result.failed == 1
    fail = next(d for d in result.details if d.startswith("FAIL"))
    assert "example/app:v1" in fail
    # The pull-only service (no build:) is never flagged.
    assert "redis:7" not in fail
    # The present build-image is not flagged either.
    assert "example/evaluator:v1" not in fail


def test_all_images_present_passes(tmp_path, monkeypatch):
    root = _make_benchmark_root(tmp_path, services=_COMPOSE)
    cid = _discovered_challenge_id(root)
    monkeypatch.setattr(preflight, "_image_exists_local", lambda img: True)
    result = PreflightResult()
    _check_target_images(_config(root), [{"id": "s0", "challenge_id": cid}], result)
    assert result.failed == 0
    assert result.passed == 1


def test_pull_only_missing_image_is_ignored(tmp_path, monkeypatch):
    # A stack with no build: services — a missing pull-only image is the
    # launcher's job, not ours; preflight must not fail on it.
    root = _make_benchmark_root(
        tmp_path,
        services="services:\n  cache:\n    image: redis:7\n",
    )
    cid = _discovered_challenge_id(root)
    monkeypatch.setattr(preflight, "_image_exists_local", lambda img: False)
    result = PreflightResult()
    _check_target_images(_config(root), [{"id": "s0", "challenge_id": cid}], result)
    assert result.failed == 0
    # No build: services means nothing to verify -> no PASS row emitted.
    assert result.passed == 0


def test_unknown_challenge_is_skipped(tmp_path, monkeypatch):
    root = _make_benchmark_root(tmp_path, services=_COMPOSE)
    monkeypatch.setattr(preflight, "_image_exists_local", lambda img: False)
    result = PreflightResult()
    # Sample points at a challenge the benchmark doesn't define.
    _check_target_images(_config(root), [{"id": "s0", "challenge_id": "nope"}], result)
    assert result.failed == 0
    assert result.passed == 0
