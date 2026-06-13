"""Tests for CLI id parsing, incl. ``@file`` subset expansion."""

from __future__ import annotations

import pytest

from cage.cli.ids import split_cli_ids


def test_plain_and_comma_values():
    assert split_cli_ids(("a", "b,c", " d ")) == ["a", "b", "c", "d"]


def test_at_file_reads_one_per_line(tmp_path):
    f = tmp_path / "subset.txt"
    f.write_text("arvo:1065\n# a comment\narvo:368\n\noss-fuzz:7  # trailing\n")
    assert split_cli_ids((f"@{f}",)) == ["arvo:1065", "arvo:368", "oss-fuzz:7"]


def test_at_file_allows_commas_within_a_line(tmp_path):
    f = tmp_path / "subset.txt"
    f.write_text("arvo:1, arvo:2\narvo:3\n")
    assert split_cli_ids((f"@{f}",)) == ["arvo:1", "arvo:2", "arvo:3"]


def test_mix_inline_ids_and_file(tmp_path):
    f = tmp_path / "subset.txt"
    f.write_text("arvo:2\narvo:3\n")
    assert split_cli_ids(("arvo_1", f"@{f}")) == ["arvo_1", "arvo:2", "arvo:3"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        split_cli_ids(("@/no/such/subset.txt",))
