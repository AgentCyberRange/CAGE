"""Storage — manages the per-project run directory layout.

Layout:
  .cage_runs/{agent_label}/run-{timestamp}/
    config.yaml
    dashboard.json
    results.csv
    agent_version.json
    initial_state/
    trials/
      {trial_id}/
        meta.json
        prompt.txt
        state_pre/
        state_post/
        proxy/
          proxy.jsonl
        task_output.json
        scores/
      {trial_id}.before_resume_{ts}/   ← preserved by `cage --resume`
                                          (siblings of the live trial dir;
                                           ignored by all trial scanners)
    metrics.json
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Sentinel substring in a directory name marking it as a snapshot of a
# previous trial attempt, kept aside by ``--resume`` so the re-run starts
# from a clean directory without losing failure evidence. Sibling dirs of
# this shape are NOT live trial dirs:
#   * trial scanners (web index, fs_signals, cli score) skip them entirely
#   * the trial-detail page surfaces them as a collapsed "Previous attempts"
#     list so users can drill into past failures on demand.
RESUME_ARCHIVE_MARKER = ".before_resume_"


def is_resume_archive_name(name: str) -> bool:
    """True iff ``name`` is a resume-archive sibling (skip during scans)."""
    return RESUME_ARCHIVE_MARKER in name


# --------------------------------------------------------------------------- #
# On-disk layout vocabulary — the SINGLE source of truth for where artifacts
# live inside a run directory. Any other module that needs these names must
# import them from here rather than hardcoding the literal: this is what makes
# ``RunStorage`` the one object that knows the filesystem layout. A grep of the
# codebase for ``"trials"`` / ``"proxy.jsonl"`` / ``"meta.json"`` outside this
# file is a layer smell (the read side re-deriving the layout).
# --------------------------------------------------------------------------- #
TRIALS_DIRNAME = "trials"
META_FILENAME = "meta.json"
TASK_OUTPUT_FILENAME = "task_output.json"
PROMPT_FILENAME = "prompt.txt"
SCORES_DIRNAME = "scores"
STATE_PRE_DIRNAME = "state_pre"
STATE_POST_DIRNAME = "state_post"
PROXY_DIRNAME = "proxy"
PROXY_LOG_FILENAME = "proxy.jsonl"
PROGRESS_FILENAME = "progress.json"
DASHBOARD_FILENAME = "dashboard.json"
RECORD_FILENAME = "record.json"
EXPERIMENT_RECORD_FILENAME = "experiment_record.json"
PLANNED_TRIALS_FILENAME = "planned_trials.json"
INITIAL_STATE_DIRNAME = "initial_state"
CAGE_RUNS_DIRNAME = ".cage_runs"

# Run-level aggregates + the canonical-record system's files. Declared here so
# ``RunStorage`` stays the one place that knows the layout. Some of these still
# appear as bare literals in ``writer.py`` / ``canonical_marks.py`` / the
# duplicate ``record_snapshots.RUN_HISTORY_FILE``; folding those onto these
# constants is the deferred consumer-migration step (record_snapshots imports
# run_storage, so the dependency only flows one way — into here).
CONFIG_FILENAME = "config.yaml"
METRICS_FILENAME = "metrics.json"
SUMMARY_FILENAME = "summary.json"
AGENT_VERSION_FILENAME = "agent_version.json"
RUN_HISTORY_FILENAME = "run_history.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
EXPERIMENT_SPEC_FILENAME = "experiment_spec.json"
EXPERIMENT_PLAN_FILENAME = "experiment_plan.json"
ARTIFACT_INDEX_FILENAME = "artifact_index.json"
PREFLIGHT_FILENAME = "preflight.json"


@dataclass(frozen=True)
class ArtifactSpec:
    """One artifact file's declared semantics + how to read it.

    The single human- and machine-readable answer to *"what is this file and
    which RunStorage method gives it to me?"*. ``RUN_ARTIFACTS`` /
    ``TRIAL_ARTIFACTS`` below enumerate every artifact a run produces.

    Worked example — *"to see an agent×model's results on a benchmark, which
    file do I read?"*: the run dir is
    ``examples/<benchmark>/.cage_runs/<agent>:<model>:<lifecycle>/<run-id>/`` and
    the aggregate + per-trial scoreboard lives in ``dashboard.json`` →
    ``RunStorage(run_dir).load_dashboard()`` (whole projection) or
    ``.iter_trial_summaries()`` (just the per-trial rows).
    """

    name: str       # filename / dir name on disk
    level: str      # "run" | "trial"
    fmt: str        # json | jsonl | text | csv | yaml | dir
    purpose: str    # one line: what it holds / which question it answers
    accessor: str   # RunStorage method that reads it ("" = path-only)


# The run's scoreboard + provenance. Read these for run-level questions.
RUN_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(DASHBOARD_FILENAME, "run", "json",
        "Aggregate + per-trial results projection — the run's scoreboard. THE "
        "file for 'how did this agent×model do on this benchmark'.",
        "load_dashboard / iter_trial_summaries"),
    ArtifactSpec(EXPERIMENT_RECORD_FILENAME, "run", "json",
        "Canonical run record: status, timestamps, trial record refs, score summary.",
        "load_experiment_record"),
    ArtifactSpec(ARTIFACT_INDEX_FILENAME, "run", "json",
        "Canonical manifest of every durable artifact in the run (one per run).",
        "load_artifact_index / has_artifact_index"),
    ArtifactSpec(EXPERIMENT_SPEC_FILENAME, "run", "json",
        "Frozen request: models, agents, benchmark, limits.", "load_experiment_spec"),
    ArtifactSpec(EXPERIMENT_PLAN_FILENAME, "run", "json",
        "Expanded plan: the concrete trials to run.", "load_experiment_plan"),
    ArtifactSpec(PLANNED_TRIALS_FILENAME, "run", "json",
        "Legacy-compat list of planned trial rows.", "load_planned_trials"),
    ArtifactSpec(RUN_HISTORY_FILENAME, "run", "json",
        "Invocation history (each run/--resume that touched this dir).", "load_run_history"),
    ArtifactSpec(RUN_MANIFEST_FILENAME, "run", "json",
        "Run manifest written on resume.", "load_run_manifest"),
    ArtifactSpec(PREFLIGHT_FILENAME, "run", "json",
        "Preflight target/config check captured before trials ran.", "load_preflight"),
    ArtifactSpec(CONFIG_FILENAME, "run", "yaml",
        "The resolved run config.", "load_config"),
    ArtifactSpec(METRICS_FILENAME, "run", "json",
        "Run-level aggregate metrics.", "load_metrics"),
    ArtifactSpec(SUMMARY_FILENAME, "run", "json",
        "Run-level human summary.", "load_summary"),
    ArtifactSpec(AGENT_VERSION_FILENAME, "run", "json",
        "Agent build/version fingerprint.", "load_agent_version"),
    ArtifactSpec(INITIAL_STATE_DIRNAME, "run", "dir",
        "Shared workspace snapshot before any trial ran.", "initial_state_dir"),
)

# Per-trial artifacts under ``trials/<id>/``. Read these for trial-level
# questions. ``load_trial_*`` accept the trial directory.
TRIAL_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(META_FILENAME, "trial", "json",
        "Trial status/timing/termination — source of truth for out-of-band "
        "reclassification.", "load_trial_meta"),
    ArtifactSpec(TASK_OUTPUT_FILENAME, "trial", "json",
        "Agent's final task output + the sample it ran.", "load_trial_output"),
    ArtifactSpec(SCORES_DIRNAME, "trial", "dir",
        "One JSON per scorer: value/answer/explanation.", "load_trial_scores"),
    ArtifactSpec(PROGRESS_FILENAME, "trial", "json",
        "Live proxy progress: request counts, tokens, cost, last activity.",
        "load_trial_progress"),
    ArtifactSpec(PROMPT_FILENAME, "trial", "text",
        "The rendered task prompt the agent received.", "load_trial_prompt"),
    ArtifactSpec(PROXY_LOG_FILENAME, "trial", "jsonl",
        "Raw intercepted LLM request/response audit log (the trajectory).",
        "load_trial_proxy_log"),
    ArtifactSpec(RECORD_FILENAME, "trial", "json",
        "Canonical per-trial durable record (status, artifacts, scoring refs).",
        "(via ExperimentArtifactReader)"),
)

ALL_ARTIFACTS: tuple[ArtifactSpec, ...] = RUN_ARTIFACTS + TRIAL_ARTIFACTS


def _read_json(path: Path) -> Any:
    """``json.loads`` of ``path``; ``None`` on missing/unreadable/bad JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str:
    """Text of ``path``; ``""`` on missing/unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""

# Directories that never hold runs; pruned during run discovery so the
# inspector never descends into vendored datasets / caches (gigabytes that
# used to make a cold index load take many seconds).
DISCOVERY_PRUNE_DIRS = frozenset({
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".worktrees",
    "__pycache__",
    "node_modules",
    "venv",
})

# Marker files that identify a directory as a live trial dir (any one present).
_TRIAL_DIR_MARKERS = (
    (META_FILENAME,),
    (TASK_OUTPUT_FILENAME,),
    (PROMPT_FILENAME,),
    (PROXY_DIRNAME, PROGRESS_FILENAME),
    (PROXY_DIRNAME, PROXY_LOG_FILENAME),
)

_NATURAL_KEY_SPLIT = re.compile(r"(\d+)")


def _natural_path_key(path: Path) -> list[Any]:
    """Sort key that treats embedded integers as numbers, not text.

    Without this, ``sorted(...)`` orders ``range-10`` between ``range-1`` and
    ``range-2`` (lexicographic). Trial scanners rely on natural
    ``challenge → level → pass`` ordering.
    """
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in _NATURAL_KEY_SPLIT.split(path.name)
    ]


def is_trial_dir(path: Path) -> bool:
    """True iff ``path`` looks like a live trial directory (has a marker file)."""
    return any(
        path.joinpath(*marker).exists() for marker in _TRIAL_DIR_MARKERS
    )


def discover_run_dirs(root: Path) -> list[Path]:
    """Find the ``.cage_runs/`` directories at or one level below ``root``.

    Canonical layout: runs live in the inspect root itself
    (``<root>/.cage_runs/``) or exactly one level below it
    (``<root>/<project>/.cage_runs/`` — e.g. every ``examples/<benchmark>/``).
    Only these two depths are scanned, so discovery never descends into a
    project's ``datasets/``, vendored submodules, or image caches.
    """
    root = root.resolve()
    if root.name == CAGE_RUNS_DIRNAME:
        return [root]
    found: list[Path] = []
    top_level = root / CAGE_RUNS_DIRNAME
    if top_level.is_dir():
        found.append(top_level)
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        children = []
    for child in children:
        if child.name == CAGE_RUNS_DIRNAME or child.name in DISCOVERY_PRUNE_DIRS:
            continue
        candidate = child / CAGE_RUNS_DIRNAME
        if candidate.is_dir():
            found.append(candidate)
    return found


@dataclass(frozen=True)
class RunRef:
    """A discovered run located by its on-disk coordinates.

    Parsed purely from the
    ``.cage_runs/<agent>:<model>:<lifecycle>/<run-id>/`` path convention — the
    directory naming is storage's own domain. Answers "where is agent×model's
    run on benchmark X?" with zero presentation: display grouping/labels stay in
    ``cage/web``. Hand a ``run_dir`` to ``RunStorage(run_dir)`` to read it.
    """

    benchmark: str   # project dir name, e.g. "cybergym" (or "." at the inspect root)
    agent: str       # agent code, e.g. "claude_code"
    model: str       # model label, e.g. "nex-n2"
    lifecycle: str   # "stateless" | "stateful" | ...
    run_id: str      # the run dir name
    run_dir: Path    # absolute path to the run directory


def _split_agent_label(label: str) -> tuple[str, str, str]:
    """Split ``<agent>:<model>:<lifecycle>`` → (agent, model, lifecycle).

    Tolerant of colons inside the model segment and of 1-/2-segment labels:
    the FIRST segment is the agent, the LAST is the lifecycle (when ≥3
    segments), and everything between is the model.
    """
    parts = label.split(":")
    if len(parts) >= 3:
        return parts[0], ":".join(parts[1:-1]), parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return label, "", ""


def _is_run_dir(path: Path) -> bool:
    """True iff ``path`` looks like a run dir (has a dashboard / record / plan)."""
    return (
        (path / DASHBOARD_FILENAME).exists()
        or (path / EXPERIMENT_RECORD_FILENAME).exists()
        or (path / PLANNED_TRIALS_FILENAME).exists()
        or (path / TRIALS_DIRNAME).is_dir()
    )


def discover_runs(root: Path) -> list[RunRef]:
    """Find every run under ``root`` as structured ``RunRef`` coordinates.

    Layout-aware navigation: for each ``.cage_runs/`` dir (see
    :func:`discover_run_dirs`), the benchmark is its parent's name, each child is
    an ``<agent>:<model>:<lifecycle>`` dir, and each grandchild (that looks like
    a run, not a ``.before_resume_*`` archive) is a run. This is the storage-level
    answer to "which runs exist for agent×model on benchmark X" without touching
    any per-run file.
    """
    refs: list[RunRef] = []
    for cage_runs in discover_run_dirs(root):
        benchmark = cage_runs.parent.name or "."
        try:
            agent_dirs = sorted(p for p in cage_runs.iterdir() if p.is_dir())
        except OSError:
            continue
        for agent_dir in agent_dirs:
            agent, model, lifecycle = _split_agent_label(agent_dir.name)
            try:
                run_dirs = sorted(p for p in agent_dir.iterdir() if p.is_dir())
            except OSError:
                continue
            for run_dir in run_dirs:
                if is_resume_archive_name(run_dir.name) or not _is_run_dir(run_dir):
                    continue
                refs.append(RunRef(
                    benchmark=benchmark,
                    agent=agent,
                    model=model,
                    lifecycle=lifecycle,
                    run_id=run_dir.name,
                    run_dir=run_dir,
                ))
    return refs


def trial_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's directory (does NOT create it).

    The read-only counterpart to :meth:`RunStorage.trial_dir`, which mkdirs.
    Use this on read/analysis paths (web, resume dry-run, score) so browsing a
    run never pollutes it with empty directories.
    """
    return run_dir / TRIALS_DIRNAME / trial_id


def trial_meta_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's ``meta.json`` (does NOT create it)."""
    return trial_path(run_dir, trial_id) / META_FILENAME


def trial_output_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's ``task_output.json`` (does NOT create it)."""
    return trial_path(run_dir, trial_id) / TASK_OUTPUT_FILENAME


def trial_prompt_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's ``prompt.txt`` (does NOT create it)."""
    return trial_path(run_dir, trial_id) / PROMPT_FILENAME


def trial_proxy_log_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's raw proxy log (does NOT create it)."""
    return trial_path(run_dir, trial_id) / PROXY_DIRNAME / PROXY_LOG_FILENAME


def trial_progress_path(run_dir: Path, trial_id: str) -> Path:
    """Pure path to one trial's proxy ``progress.json`` (does NOT create it)."""
    return trial_path(run_dir, trial_id) / PROXY_DIRNAME / PROGRESS_FILENAME


def trial_record_ref(runtime_id: str) -> str:
    """Run-relative path to one trial's canonical ``record.json`` (pure string).

    The single authority for where a trial's durable record lives: inside the
    very ``trials/<runtime_id>/`` directory the trial runner writes its runtime
    artifacts to, so one trial is one directory on disk. The canonical artifact
    writer derives every trial record ref from here instead of re-deriving a
    subject-prefixed path, which is what kept a parallel record tree alive.

    ``runtime_id`` is the runtime trial subpath (``<task>`` or
    ``<task>/pass_<n>``) carried on the plan — exactly ``trial.id`` at runtime.
    """
    sub = str(runtime_id).strip("/")
    return f"{TRIALS_DIRNAME}/{sub}/{RECORD_FILENAME}"


def trial_state_dir_path(run_dir: Path, trial_id: str, phase: str) -> Path:
    """Pure path to one trial's ``state_pre``/``state_post`` dir (no create)."""
    return trial_path(run_dir, trial_id) / f"state_{phase}"


def iter_live_trial_dirs(run_dir: Path) -> list[Path]:
    """Find all *live* trial directories under one run dir.

    Supports the flat layout (``run-xxx/trials/<id>/``), the nested layout with
    a per-challenge subdirectory (``run-xxx/trials/<challenge>/<variant>/``) used
    by benchmarks that emit a ``variant`` sample key, and the legacy layout with
    a mode subdirectory (``run-xxx/stateless/trials/``).

    Resume archives (``<id>.before_resume_<ts>/`` siblings) are normally *not*
    returned — they are past attempts of the same logical trial. Exception: a
    parent with only archive children and no live sibling surfaces its most
    recent archive, so every logical trial keeps at least one row.
    """
    trials: list[Path] = []

    def _scan(scan_root: Path) -> None:
        if not scan_root.is_dir():
            return
        live_names: set[str] = set()
        archives_by_live: dict[str, list[Path]] = {}
        recurse_targets: list[Path] = []
        for td in sorted(scan_root.iterdir(), key=_natural_path_key):
            if not td.is_dir():
                continue
            if is_resume_archive_name(td.name):
                live_name = td.name.split(RESUME_ARCHIVE_MARKER, 1)[0]
                archives_by_live.setdefault(live_name, []).append(td)
                continue
            if is_trial_dir(td):
                trials.append(td)
                live_names.add(td.name)
            else:
                recurse_targets.append(td)
        for live_name, archives in archives_by_live.items():
            if live_name in live_names:
                continue
            if (scan_root / live_name).exists():
                continue
            trials.append(max(archives, key=lambda p: p.name))
        for td in recurse_targets:
            _scan(td)

    direct = run_dir / TRIALS_DIRNAME
    if direct.is_dir():
        _scan(direct)
        if trials:
            return trials

    for child in sorted(run_dir.iterdir(), key=_natural_path_key):
        if not child.is_dir():
            continue
        trials_root = child / TRIALS_DIRNAME
        if trials_root.is_dir():
            _scan(trials_root)
    return trials


@dataclass
class RunStorage:
    """Manages artifact storage for a single experiment run."""

    run_dir: Path
    agent_label: str = ""

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Run-level artifacts
    # ------------------------------------------------------------------ #

    def save_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def save_agent_version(self, version_info: dict[str, Any]) -> None:
        (self.run_dir / "agent_version.json").write_text(
            json.dumps(version_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def initial_state_dir(self) -> Path:
        d = self.run_dir / INITIAL_STATE_DIRNAME
        d.mkdir(exist_ok=True)
        return d

    def log_file_path(self) -> Path:
        """Path to the structured JSONL log file for this run."""
        return self.run_dir / ".cage.runlog"

    def debug_log_path(self) -> Path:
        """Path to the debug log file for this run."""
        return self.run_dir / ".cage.debuglog"

    # ------------------------------------------------------------------ #
    # Trial-level artifacts
    # ------------------------------------------------------------------ #

    def trial_dir(self, trial_id: str) -> Path:
        d = self.run_dir / TRIALS_DIRNAME / trial_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def trial_state_pre_dir(self, trial_id: str) -> Path:
        d = self.trial_dir(trial_id) / STATE_PRE_DIRNAME
        d.mkdir(exist_ok=True)
        return d

    def trial_state_post_dir(self, trial_id: str) -> Path:
        d = self.trial_dir(trial_id) / STATE_POST_DIRNAME
        d.mkdir(exist_ok=True)
        return d

    def trial_proxy_dir(self, trial_id: str) -> Path:
        d = self.trial_dir(trial_id) / PROXY_DIRNAME
        d.mkdir(exist_ok=True)
        return d

    def save_trial_prompt(self, trial_id: str, prompt: str) -> None:
        (self.trial_dir(trial_id) / PROMPT_FILENAME).write_text(prompt, encoding="utf-8")

    def save_trial_meta(self, trial_id: str, meta: dict[str, Any]) -> None:
        (self.trial_dir(trial_id) / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_trial_output(self, trial_id: str, output: dict[str, Any]) -> None:
        output = self._make_json_safe(output)
        (self.trial_dir(trial_id) / TASK_OUTPUT_FILENAME).write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_trial_scores(
        self, trial_id: str, scorer_id: str, scores: dict[str, Any]
    ) -> None:
        scores_dir = self.trial_dir(trial_id) / SCORES_DIRNAME
        scores_dir.mkdir(exist_ok=True)
        (scores_dir / f"{scorer_id}.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Read & discovery — the layout authority's read half. Consumers
    # (web, cli, resume) should reach disk THROUGH these instead of
    # re-deriving ``trials/<id>/...`` paths by hand.
    # ------------------------------------------------------------------ #

    @classmethod
    def discover(cls, root: Path) -> list[Path]:
        """Return every ``.cage_runs/`` directory at or one level below ``root``."""
        return discover_run_dirs(Path(root))

    def iter_trial_dirs(self) -> list[Path]:
        """Return all live trial directories under this run (resume-archive aware)."""
        return iter_live_trial_dirs(self.run_dir)

    def trial_path(self, trial_id: str) -> Path:
        """Pure path to one trial's directory (does NOT create it)."""
        return trial_path(self.run_dir, trial_id)

    def proxy_log_path(self, trial_id: str) -> Path:
        """Path to one trial's raw proxy log (may not exist)."""
        return trial_proxy_log_path(self.run_dir, trial_id)

    def dashboard_path(self) -> Path:
        """Path to the run-level dashboard projection (may not exist)."""
        return self.run_dir / DASHBOARD_FILENAME

    def experiment_record_path(self) -> Path:
        """Path to the canonical run-level experiment record (may not exist)."""
        return self.run_dir / EXPERIMENT_RECORD_FILENAME

    def load_dashboard(self) -> dict[str, Any] | None:
        """Load ``dashboard.json`` if present, else ``None``."""
        path = self.dashboard_path()
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- run-level reads ------------------------------------------------- #
    # Every artifact in RUN_ARTIFACTS has a loader here. Missing/unreadable
    # files yield an empty container (not an exception) so browsing a partial
    # run never crashes.

    def load_experiment_record(self) -> dict[str, Any]:
        """Canonical run record (``experiment_record.json``); ``{}`` if absent."""
        return _read_json(self.experiment_record_path()) or {}

    def load_experiment_spec(self) -> dict[str, Any]:
        """Frozen experiment spec (``experiment_spec.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / EXPERIMENT_SPEC_FILENAME) or {}

    def load_experiment_plan(self) -> dict[str, Any]:
        """Expanded plan (``experiment_plan.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / EXPERIMENT_PLAN_FILENAME) or {}

    def load_planned_trials(self) -> list[dict[str, Any]]:
        """Legacy-compat planned trial rows (``planned_trials.json``); ``[]``."""
        data = _read_json(self.run_dir / PLANNED_TRIALS_FILENAME)
        return data if isinstance(data, list) else []

    def load_run_history(self) -> Any:
        """Invocation history (``run_history.json``); ``[]``/``{}`` shape preserved."""
        data = _read_json(self.run_dir / RUN_HISTORY_FILENAME)
        return data if data is not None else []

    def load_run_manifest(self) -> dict[str, Any]:
        """Resume manifest (``run_manifest.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / RUN_MANIFEST_FILENAME) or {}

    def load_preflight(self) -> dict[str, Any]:
        """Preflight check (``preflight.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / PREFLIGHT_FILENAME) or {}

    def load_metrics(self) -> dict[str, Any]:
        """Run aggregate metrics (``metrics.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / METRICS_FILENAME) or {}

    def load_summary(self) -> dict[str, Any]:
        """Run human summary (``summary.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / SUMMARY_FILENAME) or {}

    def load_agent_version(self) -> dict[str, Any]:
        """Agent build fingerprint (``agent_version.json``); ``{}`` if absent."""
        return _read_json(self.run_dir / AGENT_VERSION_FILENAME) or {}

    def load_config(self) -> dict[str, Any]:
        """Resolved run config (``config.yaml``); ``{}`` if absent."""
        try:
            data = yaml.safe_load((self.run_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def has_artifact_index(self) -> bool:
        """Whether this run uses the canonical ``artifact_index.json``."""
        return (self.run_dir / ARTIFACT_INDEX_FILENAME).is_file()

    def load_artifact_index(self) -> dict[str, Any]:
        """Raw canonical artifact index (``artifact_index.json``); ``{}`` if absent.

        Returns the raw mapping. For typed access use
        :meth:`cage.artifacts.reader.ExperimentArtifactReader.load_artifact_index`.
        """
        return _read_json(self.run_dir / ARTIFACT_INDEX_FILENAME) or {}

    def iter_trial_summaries(self) -> list[dict[str, Any]]:
        """Per-trial result rows from the ``dashboard.json`` projection (one read).

        The fast path for the run overview: ``dashboard.json`` already aggregates
        every trial's status/scores/score_details/duration_ms/exit_code/
        termination_*/usage/sample_id, so the overview needs no per-trial walk.
        Returns ``[]`` when the run has no dashboard yet (live or crashed) so
        callers fall back to scanning trial dirs.
        """
        dashboard = self.load_dashboard()
        if not isinstance(dashboard, dict):
            return []
        agents = dashboard.get("agents") or {}
        rows: list[dict[str, Any]] = []
        if isinstance(agents, dict):
            for agent_data in agents.values():
                if not isinstance(agent_data, dict):
                    continue
                for trial in agent_data.get("trials") or []:
                    if isinstance(trial, dict):
                        rows.append(trial)
        return rows

    # -- trial-level reads ----------------------------------------------- #
    # Accept the trial DIRECTORY (what read/analysis callers hold). Where the
    # canonical record system attaches the artifact, prefer it; otherwise fall
    # back to the legacy fixed filename. Best-effort canonical: a legacy run
    # (no record) or an unmatched trial simply uses the legacy file.

    def load_trial_meta(self, trial_dir: Path) -> dict[str, Any]:
        """Trial ``meta.json`` (status/timing/termination); ``{}`` if absent."""
        return _read_json(trial_dir / META_FILENAME) or {}

    def load_trial_progress(self, trial_dir: Path) -> dict[str, Any]:
        """Trial ``proxy/progress.json`` (live counters); ``{}`` if absent."""
        return _read_json(trial_dir / PROXY_DIRNAME / PROGRESS_FILENAME) or {}

    def load_trial_prompt(self, trial_dir: Path) -> str:
        """Trial rendered prompt (canonical ``rendered_prompt`` → ``prompt.txt``)."""
        for art in self._resolved_trial_artifacts(trial_dir):
            if art.kind in ("rendered_prompt", "prompt"):
                text = _read_text(Path(art.path))
                if text:
                    return text
        return _read_text(trial_dir / PROMPT_FILENAME)

    def load_trial_output(self, trial_dir: Path) -> dict[str, Any]:
        """Trial ``task_output.json`` (canonical first, legacy fallback); ``{}``."""
        for art in self._resolved_trial_artifacts(trial_dir):
            if art.kind == "task_output":
                data = _read_json(Path(art.path))
                if isinstance(data, dict):
                    return data
        data = _read_json(trial_dir / TASK_OUTPUT_FILENAME)
        return data if isinstance(data, dict) else {}

    def load_trial_scores(self, trial_dir: Path) -> dict[str, Any]:
        """Trial scores merged across scorers (canonical first, legacy fallback).

        Returns the RAW per-scorer mapping; numeric flattening is a presentation
        concern that stays in the consumer (e.g. the web inspector).
        """
        scores: dict[str, Any] = {}
        for art in self._resolved_trial_artifacts(trial_dir):
            if art.kind == "trial_score":
                data = _read_json(Path(art.path))
                if isinstance(data, dict):
                    scores.update(data)
        if scores:
            return scores
        scores_dir = trial_dir / SCORES_DIRNAME
        if scores_dir.is_dir():
            for score_file in sorted(scores_dir.glob("*.json")):
                data = _read_json(score_file)
                if isinstance(data, dict):
                    scores.update(data)
        return scores

    def load_trial_proxy_log(self, trial_dir: Path) -> Path:
        """Resolve a trial's raw proxy JSONL path (legacy or canonical).

        Returns the path (which the caller streams); the legacy
        ``proxy/proxy.jsonl`` wins when present, else the canonical artifact, else
        the legacy path (which may not exist — caller checks ``.is_file()``).
        """
        legacy = trial_dir / PROXY_DIRNAME / PROXY_LOG_FILENAME
        if legacy.is_file():
            return legacy
        for art in self._resolved_trial_artifacts(trial_dir):
            if art.kind in ("proxy_log", "proxy_jsonl") and Path(art.path).is_file():
                return Path(art.path)
        return legacy

    @staticmethod
    def _split_trial_dir(trial_dir: Path) -> tuple[Path, str]:
        """``.../<run>/trials/<id...>`` → (run_dir, trial runtime id).

        Splits on the last ``trials`` path segment so flat
        (``trials/<id>``) and nested (``trials/<challenge>/<variant>``) layouts
        both resolve. For the legacy mode layout (``<run>/<mode>/trials/<id>``)
        the returned run_dir is the mode dir — harmless: such runs predate the
        canonical record, so the reader finds nothing and callers use the legacy
        file.
        """
        parts = trial_dir.parts
        trials_indices = [i for i, part in enumerate(parts) if part == TRIALS_DIRNAME]
        if not trials_indices:
            return trial_dir.parent, trial_dir.name
        idx = trials_indices[-1]
        run_dir = Path(*parts[:idx]) if idx else Path(parts[0])
        trial_id = "/".join(parts[idx + 1:])
        return run_dir, trial_id

    def _resolved_trial_artifacts(self, trial_dir: Path) -> list[Any]:
        """Canonical artifacts attached to ``trial_dir`` (best-effort, never raises).

        Delegates the "on the TrialRecord AND in the ArtifactIndex, resolved"
        policy to :class:`ExperimentArtifactReader`. Returns ``[]`` for legacy
        runs or unmatched trials so every ``load_trial_*`` falls back to its
        legacy file. Imported lazily to avoid an import cycle.
        """
        try:
            from cage.artifacts.reader import ExperimentArtifactReader

            run_dir, trial_id = self._split_trial_dir(trial_dir)
            reader = ExperimentArtifactReader(run_dir)
            record = reader.trial_record_by_id(trial_id)
            if record is None:
                return []
            return list(reader.resolve_trial_artifacts(record))
        except Exception:
            return []

    def _make_json_safe(self, obj: Any) -> Any:
        """Convert trial output values into JSON-serializable data."""
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        return obj

    # ------------------------------------------------------------------ #
    # Run-level aggregation
    # ------------------------------------------------------------------ #

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RUN_ID_MAX_LEN = 48


def create_run_id() -> str:
    """Auto-generate a run id as ``run-<YYYYMMDDHHMMSS>-<uuid8>``.

    All-digit timestamp (no ``T`` separator) so the value also satisfies
    :func:`validate_run_id`'s user-facing ``[a-z0-9][a-z0-9_-]*`` contract.
    """
    ts = time.strftime("%Y%m%d%H%M%S")
    return f"run-{ts}-{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` after enforcing the user-facing format contract.

    User-supplied run ids must match ``[a-z0-9][a-z0-9_-]*`` and be at most
    48 characters so they survive being embedded into Docker container names,
    network names, and on-disk paths. Auto-generated ids from
    :func:`create_run_id` are also accepted (they fit this format).
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if len(run_id) > _RUN_ID_MAX_LEN:
        raise ValueError(
            f"run_id too long ({len(run_id)}>{_RUN_ID_MAX_LEN}): {run_id!r}"
        )
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"run_id must match [a-z0-9][a-z0-9_-]*: {run_id!r}"
        )
    return run_id
