"""Tests for the stats-table TSV formatter (`dac.gui.stat_to_tsv`)."""

import pytest

pytest.importorskip("PyQt5")

from dac.gui import stat_to_tsv


def test_full_table_with_headers():
    stats = {
        "title": "t",
        "headers": {"row": ["r1", "r2"], "col": ["a", "b"]},
        "data": [[1, 2], [3, 4]],
    }
    assert stat_to_tsv(stats) == "\ta\tb\nr1\t1\t2\nr2\t3\t4"


def test_missing_row_headers():
    stats = {"headers": {"col": ["a", "b"]}, "data": [[1, 2]]}
    assert stat_to_tsv(stats) == "\ta\tb\n\t1\t2"


def test_empty_data():
    stats = {"headers": {"row": [], "col": ["a"]}, "data": []}
    assert stat_to_tsv(stats) == "\ta"


def test_missing_headers():
    assert stat_to_tsv({"data": [[1, 2]]}) == "\t1\t2"
    assert stat_to_tsv({}) == ""


def test_sanitizes_tabs_and_newlines():
    stats = {"headers": {"col": ["a\tb"]}, "data": [["x\ny"]]}
    assert stat_to_tsv(stats) == "\ta b\n\tx y"
