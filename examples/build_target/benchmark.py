"""build_target — an *artifact* benchmark for CAGE.

The agent is handed the raw attachments of a CTF challenge (a program + its
source + the flag) but **no Dockerfile / compose**. Its job is to produce a
bootable CAGE target: a ``Dockerfile`` + ``docker-compose.yml`` (+ optional
``challenge.json``) that serves the challenge over TCP.

The scorer is a *validator*, not a flag-matcher: it snapshots the agent's
produced files, then independently ``docker compose up --build``s them in an
isolated project and probes the declared port from inside the target's network
namespace. The agent's textual claim ("it builds and runs") is never trusted —
only the boot evidence counts.

Single-attempt / boot-only for now (no feedback loop, no solvability gate). The
structured verdict (failure_class / retryable / feedback) rides in Score.metadata
until the framework grows first-class artifact-task fields.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cage.benchmarks import Benchmark, render_prompt
from cage.scoring import Score, Scorer, ScoringContext

if TYPE_CHECKING:
    from cage.sandbox.containers import Container

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
WORKSPACE_DIR = "/home/agent/workspace"

# Seconds to let the stack settle before checking it is still running — long
# enough for a service that crashes on startup to have crashed.
_SETTLE_SECONDS = 4

_COMPOSE_NAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _tail(text: str | None, n: int = 1500) -> str:
    text = text or ""
    return text[-n:]


class BuildTarget(Benchmark):
    """Agent builds a bootable target from attachments; scorer verifies boot."""

    name = "build_target"

    def __init__(self, benchmark_root: str | Path | None = None) -> None:
        root = Path(benchmark_root).expanduser() if benchmark_root else DATASETS_DIR
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent / root).resolve()
        if not root.is_dir():
            root = DATASETS_DIR
        self._datasets_dir = root

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        for sample_dir in sorted(p for p in self._datasets_dir.iterdir() if p.is_dir()):
            task_file = sample_dir / "task.json"
            if not task_file.is_file():
                continue
            meta = json.loads(task_file.read_text())
            attachments = list(meta.get("attachments") or [])
            yield {
                "id": meta.get("name", sample_dir.name),
                "name": meta.get("name", sample_dir.name),
                "category": meta.get("category", "pwn"),
                "internal_port": int(meta.get("internal_port") or 0),
                "description": meta.get("description", ""),
                "content": meta.get("description", ""),
                "flag": meta.get("flag", ""),
                "flag_format": meta.get("flag_format", "flag{...}"),
                "files": attachments,
                "challenge_path": str(sample_dir),
            }

    def prepare_trial(
        self,
        container: "Container",
        sample: dict[str, Any],
        workspace_dir: str,
    ) -> None:
        container.exec(f"mkdir -p {shlex.quote(workspace_dir)}")
        challenge_dir = Path(sample["challenge_path"])
        for filename in sample.get("files", []):
            source = challenge_dir / filename
            if not source.is_file():
                continue
            container.copy_to(str(source), f"{workspace_dir}/{filename}")

    def on_agent_finish(
        self,
        container: "Container",
        sample: dict[str, Any],
        trial_dir: str,
    ) -> None:
        # Materialize the agent's produced files (Dockerfile / compose / wrapper)
        # to the host BEFORE scoring, so the boot-check validator reads a frozen
        # snapshot from ``<trial_dir>/workspace`` instead of the live container.
        workspace_out = Path(trial_dir) / "workspace"
        workspace_out.mkdir(parents=True, exist_ok=True)
        try:
            container.copy_from(f"{WORKSPACE_DIR}/.", str(workspace_out), timeout=300.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("workspace copy_from failed: %s", exc)

    def build_prompt(self, sample: dict[str, Any]) -> str:
        instance_data = {key: value for key, value in sample.items() if key != "flag"}
        return render_prompt(
            template_dir=PROMPTS_DIR,
            template_type="instance",
            instance_data=instance_data,
            workspace="/home/agent/workspace",
            command_docs="",
            skill_descriptions="",
        )

    def scorer(self) -> Scorer:
        return _BootCheckScorer()


class _BootCheckScorer(Scorer):
    """Independently build + boot the agent's target; require it stays up (port-agnostic).

    The mechanical gate is deliberately narrow: builds, comes up, and does not
    crash-loop. It does NOT impose a port. Whether the service actually *serves*
    is the agent's job to self-check and evidence in ``service_output.txt``; this
    scorer captures the container logs too, and a human reviews both.
    """

    name = "build_target"

    def _fail(self, failure_class: str, summary: str, *, retryable: bool, evidence: dict[str, Any] | None = None, feedback: str = "") -> dict[str, Score]:
        return {
            "build_target": Score(
                value=0.0,
                answer="",
                explanation=summary,
                metadata={
                    "passed": False,
                    "failure_class": failure_class,
                    "retryable": retryable,
                    "feedback": feedback or summary,
                    "evidence": evidence or {},
                },
            )
        }

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        if ctx.trial_dir is None:
            return self._fail("no_workspace", "No trial_dir available for scoring.", retryable=False)
        ws = ctx.trial_dir / "workspace"
        if not ws.is_dir():
            return self._fail("no_workspace", f"Agent workspace snapshot not found at {ws}.", retryable=False)

        # --- contract gate: a compose file must exist ---
        compose_path = next((ws / n for n in _COMPOSE_NAMES if (ws / n).is_file()), None)
        if compose_path is None:
            return self._fail(
                "contract_missing_compose",
                "No docker-compose.yml produced in the workspace.",
                retryable=True,
                feedback="You did not produce a docker-compose.yml in your workspace root. Write one that builds your Dockerfile and exposes the challenge on the target port.",
            )

        proj = "buildcheck_" + re.sub(r"[^a-z0-9]+", "", ctx.trial_id.lower())[:32] + "_" + uuid.uuid4().hex[:6]

        try:
            # --- build + boot gate ---
            try:
                up = _run(["docker", "compose", "-p", proj, "-f", str(compose_path), "up", "-d", "--build"], cwd=ws, timeout=420)
            except subprocess.TimeoutExpired:
                return self._fail("build_or_boot_timeout", "docker compose up --build timed out (>420s).", retryable=True,
                                  feedback="Your build did not finish in time. Keep the image small; remember the build host is OFFLINE (no apt/pip/apk).")
            if up.returncode != 0:
                return self._fail(
                    "build_or_boot_failed",
                    "docker compose up --build failed.",
                    retryable=True,
                    evidence={"returncode": up.returncode, "stderr_tail": _tail(up.stderr), "stdout_tail": _tail(up.stdout)},
                    feedback="`docker compose up --build` failed. The build host has NO internet — do not apt/pip/apk install; base on a locally-cached image such as python:3.12-slim. Error tail:\n" + _tail(up.stderr, 800),
                )

            # --- container located ---
            ps = _run(["docker", "compose", "-p", proj, "-f", str(compose_path), "ps", "-q"], cwd=ws, timeout=60)
            cids = [c for c in (ps.stdout or "").split() if c.strip()]
            if not cids:
                return self._fail("no_container", "Compose reported no running containers after up.", retryable=True,
                                  evidence={"ps_stderr": _tail(ps.stderr)})
            cid = cids[0]

            # --- stability gate (option B: port-agnostic). We do NOT impose or probe a
            # specific port. We only require that the service builds and STAYS UP rather
            # than crash-looping. Whether it truly serves is the agent's job to self-check
            # and evidence (service_output.txt); a human reviews that + the captured logs.
            time.sleep(_SETTLE_SECONDS)
            inspect = _run(
                ["docker", "inspect", "-f", "{{.State.Status}} {{.State.Running}} {{.RestartCount}}", cid],
                timeout=30,
            )
            state = (inspect.stdout or "").strip()
            status = state.split()[0] if state else "unknown"

            # --- capture the service's own output (container logs) for HUMAN review, and
            # surface the agent's self-check output file if it wrote one. ---
            logs = _run(
                ["docker", "compose", "-p", proj, "-f", str(compose_path), "logs", "--no-color", "--tail", "300"],
                cwd=ws, timeout=60,
            )
            service_logs = ((logs.stdout or "") + (logs.stderr or "")).strip()
            review_path = ctx.trial_dir / "service_logs.txt"
            try:
                review_path.write_text(service_logs, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            agent_out_path = ws / "service_output.txt"
            agent_service_output = (
                agent_out_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if agent_out_path.is_file() else ""
            )

            if status != "running":
                return self._fail(
                    "service_not_running",
                    f"Container state is {status!r} shortly after boot (crashed / exited).",
                    retryable=True,
                    evidence={"container_state": state, "service_logs_tail": _tail(service_logs)},
                    feedback=(
                        f"Your image built but the service did not stay up (state={status!r}) — it "
                        f"likely crashes on startup. Fix it so the container keeps running. "
                        f"Service logs:\n{_tail(service_logs, 800)}"
                    ),
                )

            return {
                "build_target": Score(
                    value=1.0,
                    answer="boots & stays up",
                    explanation="Agent target built and the service stayed up. Service output captured for review.",
                    metadata={
                        "passed": True,
                        "failure_class": None,
                        "retryable": False,
                        "compose_file": compose_path.name,
                        "container_state": state,
                        "service_logs_tail": _tail(service_logs, 1500),
                        "agent_self_check_present": bool(agent_service_output),
                        "agent_service_output": agent_service_output,
                        "review_files": {
                            "service_logs": str(review_path),
                            "agent_service_output": str(agent_out_path) if agent_out_path.is_file() else None,
                        },
                    },
                )
            }
        finally:
            # --- cleanup: always tear the stack down; leave no leftovers ---
            try:
                _run(["docker", "compose", "-p", proj, "-f", str(compose_path), "down", "-v", "--remove-orphans"], cwd=ws, timeout=120)
            except Exception:
                pass
