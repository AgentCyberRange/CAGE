from __future__ import annotations

import json
from pathlib import Path


def test_build_benchmark_targets_delegates_to_benchmark_hook(tmp_path: Path) -> None:
    import cage.agents.codex  # noqa: F401
    from cage.target.build import build_benchmark_targets

    project_file = _write_build_hook_project(tmp_path)

    summary = build_benchmark_targets(
        project_file,
        limit=1,
        only=["demo-target"],
        max_workers=3,
        dry_run=True,
    )

    seen = json.loads((tmp_path / "build-hook-seen.json").read_text(encoding="utf-8"))
    assert seen == {
        "dry_run": True,
        "max_workers": 3,
        "samples": [{"id": "demo-target-l0", "challenge_id": "demo-target"}],
    }
    assert summary.total == 1
    assert summary.planned == 1
    assert summary.built == 0
    assert summary.failed == 0


def test_build_benchmark_targets_does_not_validate_agent_auth(
    tmp_path: Path,
) -> None:
    import cage.agents.claude_code  # noqa: F401
    import cage.agents.codex  # noqa: F401
    from cage.target.build import build_benchmark_targets

    project_file = _write_build_hook_project(tmp_path)
    (tmp_path / "models.yml").write_text(
        f"""
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
  blocked:
    provider: anthropic
    model: claude-opus
    auth_source: {tmp_path / "missing-claude-auth"}
""",
        encoding="utf-8",
    )
    project_file.write_text(
        project_file.read_text(encoding="utf-8").replace(
            """agents:
  - id: demo_agent
    kind: codex
    model: base
""",
            """agents:
  - id: demo_agent
    kind: codex
    model: base
  - id: blocked_claude
    kind: claude_code
    model: blocked
""",
        ),
        encoding="utf-8",
    )

    summary = build_benchmark_targets(
        project_file,
        only=["demo-target"],
        dry_run=True,
    )

    assert summary.total == 1
    assert summary.planned == 1
    assert summary.failed == 0


def test_build_benchmark_targets_prints_professional_progress(
    tmp_path: Path,
    capsys,
) -> None:
    import cage.agents.codex  # noqa: F401
    from cage.target.build import build_benchmark_targets, print_build_summary

    project_file = _write_build_hook_project(tmp_path)

    summary = build_benchmark_targets(project_file, limit=1, only=["demo-target"])
    print_build_summary(summary)

    output = capsys.readouterr().out
    assert "Benchmark build" in output
    assert "Project:" in output
    assert "Selected samples: 1" in output
    assert "Build workers: 1" in output
    assert "[1/1] demo-target" in output
    assert "demo-build demo-target" in output
    assert "built" in output
    assert "Summary: total=1 built=1 skipped=0 failed=0" in output


def test_build_benchmark_targets_prints_dry_run_mode(
    tmp_path: Path,
    capsys,
) -> None:
    import cage.agents.codex  # noqa: F401
    from cage.target.build import build_benchmark_targets, print_build_summary

    project_file = _write_build_hook_project(tmp_path)

    summary = build_benchmark_targets(project_file, limit=1, only=["demo-target"], dry_run=True)
    print_build_summary(summary)

    output = capsys.readouterr().out
    assert "Benchmark build dry-run" in output
    assert "Mode: dry-run (no build commands executed)" in output
    assert "[1/1] demo-target: would build" in output
    assert "[plan]  demo-target" in output


def test_build_progress_prints_failure_exit_code_and_error(capsys) -> None:
    from cage.target.build import _print_build_event

    _print_build_event(
        "finish",
        {
            "index": 2,
            "total": 15,
            "target_id": "pb-phpbb",
            "status": "failed",
            "duration_s": 0.5,
            "returncode": 17,
            "error": "missing base image",
        },
    )

    output = capsys.readouterr().out
    assert "[2/15] pb-phpbb: failed (exit 17) in 0.5s" in output
    assert "error: missing base image" in output


def test_build_progress_can_label_dry_run_pull_action(capsys) -> None:
    from cage.target.build import _print_build_event

    _print_build_event(
        "dry-run",
        {
            "index": 1,
            "total": 1,
            "target_id": "pb-openmetadata",
            "kind": "compose images",
            "action": "pull",
            "status": "planned",
            "command": ["docker", "compose", "pull", "db"],
            "detail": "pull-images=db->mysql:8.0",
        },
    )

    output = capsys.readouterr().out
    assert "[1/1] pb-openmetadata: would pull compose images" in output
    assert "command: docker compose pull db" in output
    assert "detail: pull-images=db->mysql:8.0" in output


def test_build_summary_keeps_failure_line_compact(capsys) -> None:
    from cage.benchmarks import BenchmarkBuildResult, BenchmarkBuildSummary
    from cage.target.build import print_build_summary

    print_build_summary(
        BenchmarkBuildSummary(
            total=1,
            built=0,
            skipped=0,
            failed=1,
            results=[
                BenchmarkBuildResult(
                    target_id="pb-phpbb",
                    status="failed",
                    error="#21 ERROR: missing source tree\n------\nmore detail",
                    duration_s=0.6,
                )
            ],
        )
    )

    output = capsys.readouterr().out
    assert "[fail]  pb-phpbb (0.6s): #21 ERROR: missing source tree" in output
    assert "more detail" not in output


def _write_build_hook_project(tmp_path: Path) -> Path:
    (tmp_path / "benchmark.py").write_text(
        """
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from cage.benchmarks import Benchmark, BenchmarkBuildResult, BenchmarkBuildSummary
from cage.scoring import Scorer


class DemoBench(Benchmark):
    name = "demo"

    def __init__(self, benchmark_root: str) -> None:
        self.benchmark_root = Path(benchmark_root)
        if not self.benchmark_root.is_absolute():
            self.benchmark_root = Path(__file__).resolve().parent / self.benchmark_root

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        yield {
            "id": "demo-target-l0",
            "challenge_id": "demo-target",
            "content": "demo",
        }

    def prepare_trial(self, container, sample, workspace_dir):
        return None

    def build_prompt(self, sample):
        return sample["content"]

    def scorer(self) -> Scorer:
        raise NotImplementedError

    def build_targets(self, samples, reporter=None, max_workers=1, dry_run=False, rebuild=False):
        Path(__file__).resolve().parent.joinpath("build-hook-seen.json").write_text(
            json.dumps(
                {
                    "dry_run": dry_run,
                    "max_workers": max_workers,
                    "samples": [
                        {"id": sample["id"], "challenge_id": sample["challenge_id"]}
                        for sample in samples
                    ],
                }
            ),
            encoding="utf-8",
        )
        if reporter is not None:
            reporter(
                "dry-run" if dry_run else "start",
                {
                    "index": 1,
                    "total": len(samples),
                    "target_id": "demo-target",
                    "kind": "demo-build",
                    "command": ["demo-build", "demo-target"],
                },
            )
            if not dry_run:
                reporter(
                    "finish",
                    {
                        "index": 1,
                        "total": len(samples),
                        "target_id": "demo-target",
                        "status": "built",
                        "duration_s": 0.01,
                    },
                )
        return BenchmarkBuildSummary(
            total=len(samples),
            built=0 if dry_run else len(samples),
            skipped=0,
            failed=0,
            results=[
                BenchmarkBuildResult(
                    target_id=sample["challenge_id"],
                    status="planned" if dry_run else "built",
                    command=["demo-build", sample["challenge_id"]],
                    detail="images=demo->demo-image:latest" if dry_run else "",
                )
                for sample in samples
            ],
            planned=len(samples) if dry_run else 0,
        )
""",
        encoding="utf-8",
    )
    (tmp_path / "models.yml").write_text(
        """
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
""",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        """
project:
  name: demo-build-project
models_file: models.yml
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBench
    benchmark_root: ./datasets
agents:
  - id: demo_agent
    kind: codex
    model: base
""",
        encoding="utf-8",
    )
    return project_file
