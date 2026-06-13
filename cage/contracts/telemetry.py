"""Run telemetry events.

Small data records describing what happened during a run as it happens, emitted
by lower layers (the proxy monitor) and rendered by higher ones (the CLI progress
reporter, the web inspector). They live in ``contracts`` (layer 0) so the
producer never imports up into the renderer and vice versa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ModelRequestEvent:
    """One model request observed by the in-container proxy for a trial."""

    trial_id: str
    step: int
    status: str
    latency_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    error: str = ""
    timestamp: float = field(default_factory=time.time)


__all__ = ["ModelRequestEvent"]
