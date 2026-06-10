"""
Dataset validity tests — all SQL in train.jsonl and heldout.jsonl must execute against
data/sample.sqlite without error, and the files must have the expected size and fields.
"""

import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "sample.sqlite"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
HELDOUT_PATH = ROOT / "data" / "heldout.jsonl"


@pytest.fixture(scope="module")
def conn():
    db = sqlite3.connect(DB_PATH)
    yield db
    db.close()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── basic structure ────────────────────────────────────────────────────────── #

def test_train_file_exists():
    assert TRAIN_PATH.exists(), "data/train.jsonl missing — run data/build_dataset.py"


def test_heldout_file_exists():
    assert HELDOUT_PATH.exists(), "data/heldout.jsonl missing — run data/build_dataset.py"


def test_db_file_exists():
    assert DB_PATH.exists(), "data/sample.sqlite missing — run data/build_dataset.py"


def test_train_size():
    records = load_jsonl(TRAIN_PATH)
    assert len(records) >= 60, f"Expected at least 60 train records, got {len(records)}"


def test_heldout_size():
    records = load_jsonl(HELDOUT_PATH)
    assert len(records) >= 30, f"Expected at least 30 heldout records, got {len(records)}"


def test_required_fields():
    for path in [TRAIN_PATH, HELDOUT_PATH]:
        for i, rec in enumerate(load_jsonl(path)):
            for field in ("question", "schema", "sql"):
                assert field in rec, f"{path.name} row {i} missing field '{field}'"


def test_no_empty_sql():
    for path in [TRAIN_PATH, HELDOUT_PATH]:
        for i, rec in enumerate(load_jsonl(path)):
            assert rec["sql"].strip(), f"{path.name} row {i} has empty sql"


def test_no_duplicate_questions():
    train = load_jsonl(TRAIN_PATH)
    heldout = load_jsonl(HELDOUT_PATH)
    all_q = [r["question"] for r in train + heldout]
    assert len(all_q) == len(set(all_q)), "Duplicate questions found across train + heldout"


# ── SQL execution ──────────────────────────────────────────────────────────── #

def test_all_train_sql_executes(conn):
    for i, rec in enumerate(load_jsonl(TRAIN_PATH)):
        try:
            conn.execute(rec["sql"]).fetchall()
        except Exception as exc:
            pytest.fail(f"train row {i} SQL failed: {exc}\n  SQL: {rec['sql']}")


def test_all_heldout_sql_executes(conn):
    for i, rec in enumerate(load_jsonl(HELDOUT_PATH)):
        try:
            conn.execute(rec["sql"]).fetchall()
        except Exception as exc:
            pytest.fail(f"heldout row {i} SQL failed: {exc}\n  SQL: {rec['sql']}")


# ── DB content sanity ──────────────────────────────────────────────────────── #

def test_db_products_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert n == 20, f"Expected 20 products, got {n}"


def test_db_customers_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    assert n == 100, f"Expected 100 customers, got {n}"


def test_db_orders_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert n == 200, f"Expected 200 orders, got {n}"


def test_db_order_items_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]
    assert n > 400, f"Expected >400 order_items, got {n}"


def test_db_categories(conn):
    cats = {r[0] for r in conn.execute("SELECT DISTINCT category FROM products").fetchall()}
    expected = {"Electronics", "Furniture", "Stationery", "Appliances", "Accessories"}
    assert expected == cats, f"Category mismatch: {cats}"


def test_db_order_statuses(conn):
    statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM orders").fetchall()}
    assert statuses.issubset({"pending", "shipped", "delivered", "cancelled"})


def test_db_referential_integrity_orders(conn):
    orphans = conn.execute(
        "SELECT COUNT(*) FROM orders o "
        "WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)"
    ).fetchone()[0]
    assert orphans == 0


def test_db_referential_integrity_items(conn):
    orphans = conn.execute(
        "SELECT COUNT(*) FROM order_items oi "
        "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = oi.order_id)"
    ).fetchone()[0]
    assert orphans == 0
