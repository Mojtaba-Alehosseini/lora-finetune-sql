"""
Tests for eval_compare utility functions: extract_sql, exec_sql, rows_match.
These must run without GPU/model downloads.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_compare import extract_sql, exec_sql, rows_match

DB_PATH = Path(__file__).parent.parent / "data" / "sample.sqlite"


@pytest.fixture(scope="module")
def conn():
    db = sqlite3.connect(DB_PATH)
    yield db
    db.close()


# ── extract_sql ────────────────────────────────────────────────────────────── #

def test_extract_sql_plain():
    assert extract_sql("SELECT * FROM products;") == "SELECT * FROM products;"


def test_extract_sql_strips_sql_marker():
    text = "### SQL\nSELECT name FROM products;"
    assert extract_sql(text) == "SELECT name FROM products;"


def test_extract_sql_strips_fence():
    text = "```sql\nSELECT 1;\n```"
    result = extract_sql(text)
    assert "SELECT 1" in result


def test_extract_sql_first_statement_only():
    text = "SELECT 1; SELECT 2;"
    result = extract_sql(text)
    assert "SELECT 2" not in result


def test_extract_sql_adds_semicolon():
    result = extract_sql("SELECT * FROM products")
    assert result.endswith(";")


def test_extract_sql_legacy_marker():
    text = "SQL: SELECT * FROM products; I hope this helps."
    result = extract_sql(text)
    assert result.startswith("SELECT")


# ── exec_sql ──────────────────────────────────────────────────────────────── #

def test_exec_valid_sql(conn):
    rows = exec_sql(conn, "SELECT COUNT(*) FROM products;")
    assert rows is not None
    assert len(rows) == 1


def test_exec_invalid_sql_returns_none(conn):
    result = exec_sql(conn, "SELECT * FROM nonexistent_table;")
    assert result is None


def test_exec_syntax_error_returns_none(conn):
    result = exec_sql(conn, "SELEKT * FORM products;")
    assert result is None


# ── rows_match ─────────────────────────────────────────────────────────────── #

def test_rows_match_identical():
    a = frozenset([frozenset({1, 2}), frozenset({3, 4})])
    assert rows_match(a, a)


def test_rows_match_different():
    a = frozenset([frozenset({1})])
    b = frozenset([frozenset({2})])
    assert not rows_match(a, b)


def test_rows_match_none_pred():
    a = frozenset([frozenset({1})])
    assert not rows_match(a, None)


def test_rows_match_none_gold():
    b = frozenset([frozenset({1})])
    assert not rows_match(None, b)


def test_rows_match_both_none():
    assert not rows_match(None, None)


def test_rows_match_order_insensitive(conn):
    # Both queries return same rows in different order
    r1 = exec_sql(conn, "SELECT product_id FROM products ORDER BY product_id;")
    r2 = exec_sql(conn, "SELECT product_id FROM products ORDER BY product_id DESC;")
    assert rows_match(r1, r2)
