"""Backfill ``marker_stages`` into each post-exploitation range's
``challenge.json`` by parsing the canonical ``scripts/verify.sh``.

Stage line format (uniform across all ranges):

    stage <stage_id> '<description>' <service> <ENV_NAME> "<path_expr>"

where ``ENV_NAME`` is one of ``USER_MARKER_PATH``, ``ROOT_MARKER_PATH``,
``FILE_READ_MARKER_PATH`` (others tolerated as ``kind=other``). ``path_expr``
is typically ``"$ENV_NAME"``; the literal default is resolved from the
script's ``${VERIFY_..._MARKER_PATH:-/path}`` fallback at the top.

Output is intentionally aggregate-only — counts, not identifiers.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

POSTEXP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "agent_pentest_bench"
    / "datasets"
    / "post_exploit_bench"
)

ENV_KIND = {
    "USER_MARKER_PATH": "user",
    "ROOT_MARKER_PATH": "root",
    "FILE_READ_MARKER_PATH": "file_read",
}

# ``USER_MARKER_PATH="${2:-${VERIFY_USER_MARKER_PATH:-/tmp/range1_user_shell_marker}}"``
DEFAULT_RX = re.compile(
    r'^\s*(?P<env>[A-Z_]+)="?\$\{\d+:-\$\{VERIFY_\1:-(?P<default>[^}]+)\}\}"?',
    re.MULTILINE,
)
# Also accept simpler forms like ``FOO="${VERIFY_FOO:-/path}"`` or ``FOO=/path``.
SIMPLE_DEFAULT_RX = re.compile(
    r'^\s*(?P<env>[A-Z_]+)=(?:"?\$\{VERIFY_\1:-(?P<def1>[^}]+)\}"?|"?(?P<def2>/[^"\s]+)"?)\s*$',
    re.MULTILINE,
)


def _join_backslash_lines(text: str) -> str:
    return re.sub(r"\\\n\s*", " ", text)


def _resolve_defaults(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in DEFAULT_RX.finditer(text):
        out[m.group("env")] = m.group("default")
    for m in SIMPLE_DEFAULT_RX.finditer(text):
        env = m.group("env")
        val = m.group("def1") or m.group("def2")
        if env not in out and val:
            out[env] = val
    return out


def _resolve_path(expr: str, defaults: dict[str, str]) -> tuple[str, str | None]:
    """Return (resolved_path, env_name_or_None). ``env_name`` lets us tag
    ``marker_kind``; literal paths get no env tag."""
    e = expr.strip().strip('"').strip("'")
    if e.startswith("$"):
        env = e.lstrip("$").strip("{}")
        return defaults.get(env, ""), env
    return e, None


def _parse_stages(verify_sh: Path) -> list[dict[str, str]]:
    raw = verify_sh.read_text(encoding="utf-8")
    text = _join_backslash_lines(raw)
    defaults = _resolve_defaults(raw)

    stages: list[dict[str, str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("stage "):
            continue
        try:
            tokens = shlex.split(s)
        except ValueError:
            continue
        if len(tokens) < 6 or tokens[0] != "stage":
            continue
        stage_id, description, service, env_name, path_expr = tokens[1:6]
        marker_path, env_used = _resolve_path(path_expr, defaults)
        kind = ENV_KIND.get(env_name) or ENV_KIND.get(env_used or "") or "other"
        stages.append({
            "id": stage_id,
            "description": description,
            "service": service,
            "marker_env": env_name,
            "marker_path": marker_path,
            "marker_kind": kind,
        })
    return stages


def _backfill_one(range_dir: Path) -> tuple[int, int]:
    """Returns (n_stages_written, n_stages_unresolved). Silent on identifiers."""
    verify_sh = range_dir / "scripts" / "verify.sh"
    challenge_json = range_dir / "challenge.json"
    if not verify_sh.is_file() or not challenge_json.is_file():
        return 0, 0
    stages = _parse_stages(verify_sh)
    unresolved = sum(1 for st in stages if not st["marker_path"])

    payload = json.loads(challenge_json.read_text(encoding="utf-8"))
    src = payload.get("source_fields")
    if not isinstance(src, dict):
        src = {}
        payload["source_fields"] = src
    src["marker_stages"] = stages
    challenge_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(stages), unresolved


def main() -> int:
    if not POSTEXP_ROOT.is_dir():
        print(f"postexp root missing: {POSTEXP_ROOT}", file=sys.stderr)
        return 2
    total_stages = 0
    total_unresolved = 0
    range_count = 0
    for d in sorted(POSTEXP_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("range-"):
            continue
        n, u = _backfill_one(d)
        if n:
            range_count += 1
            total_stages += n
            total_unresolved += u
            print(f"{d.name}: wrote {n} stages  (unresolved={u})")
    print(f"---\ntotal: {range_count} ranges, {total_stages} stages, {total_unresolved} unresolved paths")
    return 0 if total_unresolved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
