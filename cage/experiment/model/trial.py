"""Trial — the atomic unit of evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from cage.contracts.execution import Timing
from cage.contracts.scoring import Score


class TrialType(str, Enum):
    TASK = "task"  # normal benchmark task
    REFLECTION = "reflection"  # meta-trial injected by hook
    WARMUP = "warmup"  # benign warm-up trial


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


@dataclass
class Trial:
    """A single trial in an experiment."""

    id: str
    index: int  # position in the trial sequence
    type: TrialType
    sample: dict[str, Any]  # raw sample from Benchmark.iter_samples()

    # Filled after execution
    status: TrialStatus = TrialStatus.PENDING
    output: str = ""
    exit_code: int | None = None
    timing: Timing | None = None
    error: str | None = None

    # Paths to artifacts (filled by storage)
    artifacts_dir: Path | None = None

    @property
    def sample_id(self) -> str:
        return str(self.sample.get("id", self.id))

    @property
    def content(self) -> str:
        return str(self.sample.get("content", ""))


@dataclass(frozen=True)
class TrialResult:
    """Immutable result of a completed trial."""

    trial_id: str
    trial_index: int
    trial_type: str
    sample_id: str
    output: str
    exit_code: int
    timing: Timing
    error: str | None = None

    # Paths to persisted data
    proxy_log: Path | None = None
    state_pre: Path | None = None
    state_post: Path | None = None

    # Scores (filled after scoring)
    scores: dict[str, Score] = field(default_factory=dict)

    # Raw metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tool call monitoring
    tool_call_count: int = 0
    terminated_by_limit: bool = False
