#!/usr/bin/env python
"""Test numeric_data_store.py: seed mock employees -> verify distribution -> run_sql, no LLM involved."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.orchestrator.numeric_data_store import (
    _DEPARTMENT_TARGETS,
    _TOTAL_EMPLOYEES,
    run_sql,
    seed_mock_employees,
)


def test_numeric_data_store() -> bool:
    print("=" * 70)
    print("NUMERIC DATA STORE TEST")
    print("=" * 70)

    try:
        print("\n[1/3] Seeding mock employees (first run)...")
        seeded = seed_mock_employees()
        assert seeded == _TOTAL_EMPLOYEES, f"expected {_TOTAL_EMPLOYEES} seeded, got {seeded}"
        print(f"   OK: seeded {seeded} employees")

        print("\n[2/3] Seeding again (idempotency check)...")
        reseeded = seed_mock_employees()
        assert reseeded == 0, f"expected 0 on already-seeded table, got {reseeded}"
        total_rows = run_sql("SELECT COUNT(*) AS cnt FROM users")[0]["cnt"]
        assert total_rows == _TOTAL_EMPLOYEES, f"expected {_TOTAL_EMPLOYEES} rows, got {total_rows}"
        print(f"   OK: no duplicate insert, still {total_rows} rows")

        print("\n[3/3] Verifying department distribution via run_sql()...")
        rows = run_sql("SELECT department, COUNT(*) AS cnt FROM users GROUP BY department ORDER BY department")
        actual = {row["department"]: row["cnt"] for row in rows}
        expected = dict(_DEPARTMENT_TARGETS)
        assert actual == expected, f"distribution mismatch:\n  expected {expected}\n  actual   {actual}"
        print(f"   OK: {actual}")

        print("\nPASSED: numeric_data_store.py seeds and serves mock employee data correctly.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_numeric_data_store()
    sys.exit(0 if success else 1)