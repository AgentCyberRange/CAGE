"""Small parsers for repeatable comma-aware CLI id options."""

from __future__ import annotations

from pathlib import Path


def _read_id_file(path_str: str) -> list[str]:
    """Read ids from a file: one per line, ``#`` comments and blanks skipped,
    commas within a line allowed. Path is resolved relative to the CWD."""

    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"id file not found: {path} (from '@{path_str}')")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def split_cli_ids(values: tuple[str, ...]) -> list[str]:
    """Expand repeatable CLI id values that may also contain commas.

    A value of the form ``@PATH`` is read as a file of ids (one per line, ``#``
    comments and blank lines skipped), so a large subset can be passed without
    listing every id on the command line — e.g. ``--sample @subset.txt``. Plain
    values are split on commas as before. The two forms can be mixed and
    repeated: ``--sample arvo_1 --sample @more.txt``.
    """

    ids: list[str] = []
    for tok in values:
        tok = str(tok)
        if tok.startswith("@"):
            ids.extend(_read_id_file(tok[1:]))
            continue
        for part in tok.split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids
