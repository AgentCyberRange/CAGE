#!/usr/bin/env python3
"""Build a CyberGym task catalog (the Layer-2 "which tasks exist" index).

Two task families, two catalogs — both normalised to the same schema (a dict
keyed by ``task_id``), analogous to ``cvebench.json`` / ``nyu_ctf.json``:

  * ``cybergym.json`` — the **official CyberGym benchmark**. Built from the
    canonical manifest ``cybergym_data/tasks.json`` (arvo + oss-fuzz).
        python scripts/build_cybergym_index.py from-manifest \
            --manifest /path/to/cybergym_data/tasks.json --out datasets/cybergym.json

  * ``arvo.json`` — same-kind ARVO tasks that are **not** in the benchmark
    (the "arvo-external" set). Built by globbing a payload tree.
        python scripts/build_cybergym_index.py from-data-dir \
            --data-dir /path/to/arvo-external/data --out datasets/arvo.json

Both catalogs carry a ``path`` relative to their own ``data_dir`` so the
benchmark can resolve payloads regardless of where the (large) data lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _task_source_id(task_id: str) -> tuple[str, str]:
    source, _, ident = task_id.partition(":")
    return source, ident


def entry_from_manifest(task: dict) -> tuple[str, dict] | None:
    task_id = str(task.get("task_id") or "").strip()
    if not task_id or ":" not in task_id:
        return None
    source, ident = _task_source_id(task_id)
    return task_id, {
        "benchmark": "cybergym",
        "benchmark_family": "cybergym",
        "source": source,
        "task_id": task_id,
        "id_num": ident,
        "project": task.get("project_name"),
        "language": task.get("project_language"),
        "description": task.get("vulnerability_description"),
        "crash_type": None,
        "fix_commit": None,
        "path": f"{source}/{ident}",
    }


def build_from_manifest(manifest: Path) -> dict[str, dict]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    tasks = raw if isinstance(raw, list) else list(raw.values())
    out: dict[str, dict] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        built = entry_from_manifest(task)
        if built:
            out[built[0]] = built[1]
    return out


def build_from_data_dir(data_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for source in ("arvo", "oss-fuzz"):
        root = data_dir / source
        if not root.is_dir():
            continue
        dirs = sorted(
            (p for p in root.iterdir()
             if p.is_dir() and (p / "repo-vul.tar.gz").is_file()),
            key=lambda p: (int(p.name) if p.name.isdigit() else 0, p.name),
        )
        for d in dirs:
            ident = d.name
            meta = {}
            mp = d / "meta.json"
            if mp.is_file():
                try:
                    loaded = json.loads(mp.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        meta = loaded
                except (OSError, json.JSONDecodeError):
                    meta = {}
            task_id = f"{source}:{ident}"
            out[task_id] = {
                "benchmark": "cybergym",
                "benchmark_family": "cybergym",
                "source": source,
                "task_id": task_id,
                "id_num": ident,
                "project": meta.get("project"),
                "language": None,
                "description": None,
                "crash_type": meta.get("crash_type"),
                "fix_commit": meta.get("fix_commit"),
                "path": f"{source}/{ident}",
            }
    return out


def _write(entries: dict[str, dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    by_source: dict[str, int] = {}
    for e in entries.values():
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
    print(f"wrote {len(entries)} tasks to {out}")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("from-manifest", help="build from cybergym_data/tasks.json")
    m.add_argument("--manifest", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)

    d = sub.add_parser("from-data-dir", help="glob a payload tree (external set)")
    d.add_argument("--data-dir", required=True, type=Path)
    d.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.mode == "from-manifest":
        _write(build_from_manifest(args.manifest.expanduser().resolve()), args.out)
    else:
        _write(build_from_data_dir(args.data_dir.expanduser().resolve()), args.out)


if __name__ == "__main__":
    main()
