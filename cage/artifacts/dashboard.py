"""Dashboard — benchmark-defined visualization spec for a finished run.

A benchmark may return a :class:`Dashboard` from :meth:`Benchmark.build_dashboard`
after a run completes. The orchestrator writes it to
``<run_dir>/dashboard_view.json`` next to the existing manifest ``dashboard.json``.
The inspector renders the spec generically — it knows about
:class:`Section` *kinds*, not benchmark domains.

Schema design goals:
  * Benchmark fills data, framework renders. No HTML in benchmark code.
  * Per-benchmark / per-category variation comes from emitting different
    section sets, not from new section kinds.
  * Forward compat: unknown section kinds are rendered as a fenced JSON
    blob, never crash the inspector.

Section kinds (v1):
  * ``summary``  — list of label/value/hint stat chips.
  * ``table``    — typed columns + opaque-shaped row dicts keyed by ``column.key``.
  * ``note``     — free-form text (rendered as preformatted; no HTML).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Stat:
    """A single labelled value in a ``summary`` section."""

    label: str
    value: str
    hint: Optional[str] = None


@dataclass(frozen=True)
class Column:
    """A column definition for a ``table`` section.

    ``align`` is a hint to the renderer: ``"left" | "right" | "center"``.
    Numeric columns should use ``"right"`` for visual alignment.
    ``unit`` and ``hint`` are optional display metadata for generic renderers.
    """

    key: str
    label: str
    align: str = "left"
    unit: Optional[str] = None
    hint: Optional[str] = None


@dataclass(frozen=True)
class Section:
    """One renderable block in a dashboard.

    Different kinds use different fields; unused fields are simply absent
    in the serialized JSON.
    """

    kind: str  # "summary" | "table" | "note"
    title: str = ""
    subtitle: str = ""
    stats: tuple[Stat, ...] = ()
    columns: tuple[Column, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "title": self.title}
        if self.subtitle:
            out["subtitle"] = self.subtitle
        if self.kind == "summary":
            out["stats"] = [
                {"label": s.label, "value": s.value, **({"hint": s.hint} if s.hint else {})}
                for s in self.stats
            ]
        elif self.kind == "table":
            out["columns"] = [
                {
                    "key": c.key,
                    "label": c.label,
                    "align": c.align,
                    **({"unit": c.unit} if c.unit else {}),
                    **({"hint": c.hint} if c.hint else {}),
                }
                for c in self.columns
            ]
            out["rows"] = list(self.rows)
        elif self.kind == "note":
            out["body"] = self.body
        return out


@dataclass(frozen=True)
class Dashboard:
    """Top-level dashboard spec returned by :meth:`Benchmark.build_dashboard`."""

    title: str = ""
    subtitle: str = ""
    sections: tuple[Section, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [s.to_dict() for s in self.sections],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_dashboard_view(run_dir: Path) -> Optional[dict[str, Any]]:
    """Read ``<run_dir>/dashboard_view.json``, return its parsed dict or ``None``.

    The inspector calls this — it deliberately returns a plain dict (not a
    :class:`Dashboard`) so unknown section kinds from a newer benchmark
    flow through untouched and the template can fall back gracefully.
    """
    path = run_dir / "dashboard_view.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


__all__ = [
    "SCHEMA_VERSION",
    "Stat",
    "Column",
    "Section",
    "Dashboard",
    "load_dashboard_view",
]
