"""Serve-native submission log — a target-centric record for PULL-mode scoring.

Serve (benchmark-only / PULL) mode has no Cage-launched agent: an *external*
client launches an isolated target instance (``GET /launch``), attacks it, and
submits its output (``POST /submit``). Cage never sees the client's model,
trajectory, prompt handling, or container — so the agent-centric ``cage run``
experiment record (subjects = agent x model, trials, proxy logs, state
snapshots) does not fit. Forcing serve through it means faking an experiment and
leaving the agent panels empty.

This module is that record done natively. Each submission is one self-describing
directory under ``.cage_serve/`` carrying exactly the trial's information *minus
the agent half*:

  - which challenge and which **target launch** produced the result (the launch
    id / project / target addresses — the thing a verdict is only meaningful
    relative to),
  - the **prompt served** (level + digest of the briefing the server offered),
  - the **submission** itself (the agent's findings, persisted — never deleted —
    so an offline re-judge has its inputs),
  - the frozen **verifier evidence** and an ``inputs_digest`` over it,
  - the **grader** version (dataset commit + scoring-asset digest — which
    verifier / ground truth graded it), and
  - one or more **scoring passes**, each a ``(judge signature, score time)``
    keyed, provenance-complete verdict. The first submit writes one canonical
    pass; the layout admits many (re-judge with a different judge model later
    without re-running the target).

Layer-1 clean: no benchmark name appears here. The record embeds a benchmark's
scorer output but never names one.

On-disk layout (one directory per submission)::

    .cage_serve/<client_id>/<submission_id>/
      record.json                     # ServeSubmissionRecord (this module)
      task_output.json                # {output, sample} — the scorer's sample
      runtime/check_done_output.txt   # frozen verifier evidence (gather output)
      workspace/final_answer/...       # persisted agent findings (judge input)
      scores/<scorer>.json            # the canonical pass's score
      scores/judge_io.jsonl           # judge audit log (written by the scorer)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cage.artifacts.records import _sha256_artifact_path, _sha256_file

SERVE_ROOT_DIRNAME = ".cage_serve"

_RECORD_FILENAME = "record.json"
_TASK_OUTPUT_FILENAME = "task_output.json"
_RUNTIME_DIRNAME = "runtime"
_CHECK_DONE_FILENAME = "check_done_output.txt"
_WORKSPACE_DIRNAME = "workspace"
_FINAL_ANSWER_DIRNAME = "final_answer"
_SCORES_DIRNAME = "scores"

SCHEMA_VERSION = "serve_submission.v1"

# Curated grader/ground-truth filenames whose content pins the scoring version.
# Mirrors what the web scorer's ``_load_official_vulnerabilities`` reads.
_GRADER_ASSET_NAMES = ("verify.py", "verify.sh", "metadata.json", "report.md")
_GRADER_ASSET_DIRS = ("exploits",)


# ---------------------------------------------------------------------------
# digests / provenance helpers
# ---------------------------------------------------------------------------
def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_tree(root: Path) -> str:
    """Stable content digest of a directory tree, blank if missing/empty.

    Delegates the tree walk + hashing to the shared artifact hasher
    (:func:`cage.artifacts.records._sha256_artifact_path`) so serve and the
    managed run compute digests the same way; only the "missing/empty ⇒ blank"
    honesty is layered on top, so an absent or empty findings dir is recorded as
    "no digest" rather than a hash of nothing.
    """
    root = Path(root)
    if not root.is_dir() or not any(p.is_file() for p in root.rglob("*")):
        return ""
    return "sha256:" + _sha256_artifact_path(root)


def _git_commit(path: Path) -> str:
    """Best-effort ``git rev-parse HEAD`` of the submodule holding ``path``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _grader_assets_digest(challenge_dir: Path) -> str:
    """Content digest of the challenge's verifier + ground-truth assets.

    Combines the shared per-file / per-tree hashers over a curated asset set, so
    a change to any verifier script or ground-truth file changes the digest.
    """
    if not challenge_dir.is_dir():
        return ""
    parts: list[str] = []
    for name in _GRADER_ASSET_NAMES:
        f = challenge_dir / name
        if f.is_file():
            parts.append(f"{name}={_sha256_file(f)}")
    for dirname in _GRADER_ASSET_DIRS:
        tree = digest_tree(challenge_dir / dirname)
        if tree:
            parts.append(f"{dirname}={tree}")
    return _sha256_text("\n".join(parts)) if parts else ""


def _challenge_dir(challenge: Mapping[str, Any]) -> Path | None:
    full_path = str(challenge.get("full_path") or "")
    return Path(full_path).resolve() if full_path else None


def _instance_launch_time(instance: Mapping[str, Any]) -> str:
    """Best-effort launch timestamp from the instance registry (ISO8601 or "")."""
    for key in ("launched_at", "created_at", "launch_time", "start_time", "started_at"):
        val = instance.get(key)
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# record dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetLaunch:
    """The target-range boot a verdict was produced against (its identity)."""

    launch_id: str  # = run_id, the launched instance id
    challenge_id: str
    project_name: str = ""
    launch_time: str = ""
    network_name: str = ""
    target_info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptServed:
    """The briefing the server offered the external client (not proof of use)."""

    level: str = ""
    digest: str = ""
    delivery: str = "served"  # serve = agent fetched it via /prompt


@dataclass(frozen=True)
class Submission:
    """The external client's output — persisted so a re-judge has its inputs."""

    received_at: str = ""
    findings_ref: str = ""  # rel path to workspace/final_answer (or "")
    findings_digest: str = ""
    has_findings: bool = False


@dataclass(frozen=True)
class Grader:
    """Which verifier / ground-truth version graded this submission."""

    dataset_commit: str = ""
    verifier_kind: str = ""
    scoring_assets_digest: str = ""


@dataclass(frozen=True)
class PassProvenance:
    """Denormalized provenance stamped on every pass so it is self-describing.

    ``mode`` discriminates serve vs managed; serve honestly carries no
    agent/model fields (Cage cannot see the external client's model), only the
    client's self-declared id.
    """

    mode: str = "serve"
    challenge_id: str = ""
    target_launch_id: str = ""
    project_name: str = ""
    launch_time: str = ""
    client_id: str = ""
    client_source: str = "declared"
    prompt_level: str = ""
    prompt_digest: str = ""
    inputs_digest: str = ""
    grader: Grader = field(default_factory=Grader)
    judge_models: tuple[str, ...] = ()
    score_time: str = ""


@dataclass(frozen=True)
class ScorePass:
    """One scoring pass — a ``(judge signature, score time)``-keyed verdict."""

    pass_id: str
    judge_models: tuple[str, ...]
    score_time: str
    score_ref: str  # rel path to the scores/<scorer>.json this pass wrote
    scores: Mapping[str, Any]
    provenance: PassProvenance


@dataclass(frozen=True)
class ServeSubmissionRecord:
    """The durable, self-describing record of one serve submission."""

    submission_id: str
    client_id: str
    challenge_id: str
    benchmark_module: str
    created_at: str
    target_launch: TargetLaunch
    prompt_served: PromptServed
    submission: Submission
    grader: Grader
    inputs_digest: str
    canonical_pass_id: str
    # A human-friendly, caller-supplied name for easy lookup. NOT an identity —
    # may be empty, may repeat. The machine primary key is ``submission_id``.
    label: str = ""
    passes: tuple[ScorePass, ...] = ()
    verifier_evidence_ref: str = f"{_RUNTIME_DIRNAME}/{_CHECK_DONE_FILENAME}"
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# pass-id / signature
# ---------------------------------------------------------------------------
def judge_signature(judge_models: tuple[str, ...]) -> str:
    """A stable config signature for a pass. Empty judge set → verifier-only."""
    if not judge_models:
        return "verifier-only"
    return "+".join(judge_models)


def make_pass_id(judge_models: tuple[str, ...], score_time: str) -> str:
    """Auto pass id keyed on ``(judge signature, score time)`` — no manual tag."""
    return f"{judge_signature(judge_models)}@{score_time}"


# ---------------------------------------------------------------------------
# layout + persistence
# ---------------------------------------------------------------------------
def sanitize_label(label: str) -> str:
    """Filesystem-safe form of a caller-supplied label (empty stays empty)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(label or "").strip())
    return safe[:40]


def submission_leaf(submission_id: str, label: str = "") -> str:
    """The submission's leaf directory name.

    When a ``label`` is given it prefixes the dir (``<label>__<submission_id>``)
    so ``ls`` shows a human-navigable name, while the stable ``submission_id``
    stays embedded for programmatic lookup. No label → just ``submission_id``.
    """
    safe = sanitize_label(label)
    return f"{safe}__{submission_id}" if safe else submission_id


def submission_dir(
    serve_root: Path, client_id: str, submission_id: str, label: str = ""
) -> Path:
    return Path(serve_root) / client_id / submission_leaf(submission_id, label)


def persist_inputs(
    sub_dir: Path,
    *,
    sample: Mapping[str, Any],
    output: str,
    evidence: str,
    agent_output_dir: Path | None,
) -> Submission:
    """Write the scorer-visible inputs into the submission dir and freeze them.

    Lays out ``task_output.json``, ``runtime/check_done_output.txt``, and copies
    the agent findings to ``workspace/final_answer`` — the exact layout the
    benchmark scorer reads (``ctx.trial_dir/workspace/final_answer`` for the
    LLM judge, ``ctx.check_done_output`` for the verifier signal). Persisting
    the findings is what makes the pass re-judgeable offline and fixes the serve
    bug where the upload was deleted before the judge could read it.

    Returns the :class:`Submission` metadata (findings ref + digest).
    """
    sub_dir = Path(sub_dir)
    (sub_dir / _RUNTIME_DIRNAME).mkdir(parents=True, exist_ok=True)
    (sub_dir / _TASK_OUTPUT_FILENAME).write_text(
        json.dumps({"output": output, "sample": sample}, ensure_ascii=False),
        encoding="utf-8",
    )
    (sub_dir / _RUNTIME_DIRNAME / _CHECK_DONE_FILENAME).write_text(
        evidence or "", encoding="utf-8"
    )

    findings_ref = ""
    findings_digest = ""
    has_findings = False
    workspace = sub_dir / _WORKSPACE_DIRNAME
    if agent_output_dir is not None and Path(agent_output_dir).is_dir():
        # Mirror the managed path's _copy_agent_workspace: the uploaded output
        # dir (its final_answer/ holds the reports) maps to workspace/, giving
        # workspace/final_answer the judge reads.
        shutil.copytree(agent_output_dir, workspace, dirs_exist_ok=True)
        final_answer = workspace / _FINAL_ANSWER_DIRNAME
        if final_answer.is_dir():
            findings_ref = f"{_WORKSPACE_DIRNAME}/{_FINAL_ANSWER_DIRNAME}"
            findings_digest = digest_tree(final_answer)
            has_findings = bool(findings_digest)

    return Submission(
        findings_ref=findings_ref,
        findings_digest=findings_digest,
        has_findings=has_findings,
    )


def build_grader(challenge: Mapping[str, Any], sample: Mapping[str, Any]) -> Grader:
    """Pin the verifier + ground-truth version that graded this submission."""
    chal_dir = _challenge_dir(challenge)
    verifier_kind = str((sample.get("metadata") or {}).get("verifier_kind") or "")
    if chal_dir is None:
        return Grader(verifier_kind=verifier_kind)
    return Grader(
        dataset_commit=_git_commit(chal_dir),
        verifier_kind=verifier_kind,
        scoring_assets_digest=_grader_assets_digest(chal_dir),
    )


def build_target_launch(
    *,
    launch_id: str,
    challenge_id: str,
    instance: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> TargetLaunch:
    return TargetLaunch(
        launch_id=launch_id,
        challenge_id=challenge_id,
        project_name=str(instance.get("project_name") or ""),
        launch_time=_instance_launch_time(instance),
        network_name=str(instance.get("network_name") or ""),
        target_info=dict(sample.get("target_info") or {}),
    )


def compute_inputs_digest(evidence: str, findings_digest: str) -> str:
    """One digest over the frozen scorer inputs (verifier evidence + findings).

    Two passes with the same ``inputs_digest`` scored the *same* launch + the
    *same* findings — a true re-judge, comparable. A different digest means a
    different target boot, not the same result judged differently.
    """
    return _sha256_text((evidence or "") + "\n\x1e\n" + (findings_digest or ""))


def prompt_digest(prompt_text: str) -> str:
    return _sha256_text(prompt_text or "")


def _to_json(record: ServeSubmissionRecord) -> str:
    return json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2)


def write_record(sub_dir: Path, record: ServeSubmissionRecord) -> Path:
    """Serialize the record to ``<submission_dir>/record.json``."""
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / _RECORD_FILENAME
    path.write_text(_to_json(record), encoding="utf-8")
    return path


def load_record(sub_dir: Path) -> ServeSubmissionRecord | None:
    """Read back a submission record (for offline re-scoring / inspection)."""
    path = Path(sub_dir) / _RECORD_FILENAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _record_from_dict(raw)


def _record_from_dict(raw: Mapping[str, Any]) -> ServeSubmissionRecord:
    def _grader(d: Mapping[str, Any]) -> Grader:
        keys = ("dataset_commit", "verifier_kind", "scoring_assets_digest")
        return Grader(**{k: d.get(k, "") for k in keys})

    passes = tuple(
        ScorePass(
            pass_id=p["pass_id"],
            judge_models=tuple(p.get("judge_models") or ()),
            score_time=p.get("score_time", ""),
            score_ref=p.get("score_ref", ""),
            scores=p.get("scores") or {},
            provenance=PassProvenance(
                **{
                    **{k: v for k, v in (p.get("provenance") or {}).items() if k != "grader"},
                    "judge_models": tuple((p.get("provenance") or {}).get("judge_models") or ()),
                    "grader": _grader((p.get("provenance") or {}).get("grader") or {}),
                }
            ),
        )
        for p in (raw.get("passes") or [])
    )
    tl = raw.get("target_launch") or {}
    ps = raw.get("prompt_served") or {}
    sb = raw.get("submission") or {}
    return ServeSubmissionRecord(
        submission_id=raw["submission_id"],
        client_id=raw.get("client_id", ""),
        challenge_id=raw.get("challenge_id", ""),
        benchmark_module=raw.get("benchmark_module", ""),
        created_at=raw.get("created_at", ""),
        target_launch=TargetLaunch(
            launch_id=tl.get("launch_id", ""),
            challenge_id=tl.get("challenge_id", ""),
            project_name=tl.get("project_name", ""),
            launch_time=tl.get("launch_time", ""),
            network_name=tl.get("network_name", ""),
            target_info=tl.get("target_info") or {},
        ),
        prompt_served=PromptServed(
            level=ps.get("level", ""),
            digest=ps.get("digest", ""),
            delivery=ps.get("delivery", "served"),
        ),
        submission=Submission(
            received_at=sb.get("received_at", ""),
            findings_ref=sb.get("findings_ref", ""),
            findings_digest=sb.get("findings_digest", ""),
            has_findings=bool(sb.get("has_findings")),
        ),
        grader=_grader(raw.get("grader") or {}),
        inputs_digest=raw.get("inputs_digest", ""),
        canonical_pass_id=raw.get("canonical_pass_id", ""),
        label=raw.get("label", ""),
        passes=passes,
        verifier_evidence_ref=raw.get(
            "verifier_evidence_ref", f"{_RUNTIME_DIRNAME}/{_CHECK_DONE_FILENAME}"
        ),
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
    )
