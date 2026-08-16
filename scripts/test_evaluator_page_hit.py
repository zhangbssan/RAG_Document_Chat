#!/usr/bin/env python
"""Test evaluator.py::_page_hit_score() against a multi-page Source block."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.evaluator import _page_hit_score
from app.schemas import Source


def test_page_hit_score() -> bool:
    print("=" * 70)
    print("EVALUATOR _page_hit_score() TEST")
    print("=" * 70)

    try:
        print("\n[1/3] A multi-page block hits an expected page anywhere in its range...")
        multi_page = Source(text="...", document="doc.pdf", page=2, pages=[2, 3], chunk="4-5")
        assert _page_hit_score([multi_page], "doc.pdf", 3) == 1.0, "expected page 3 to hit inside pages=[2,3]"
        assert _page_hit_score([multi_page], "doc.pdf", 2) == 1.0
        assert _page_hit_score([multi_page], "doc.pdf", 5) == 0.0
        print("   OK")

        print("\n[2/3] A single-page block (pages=None) falls back to .page...")
        single_page = Source(text="...", document="doc.pdf", page=7, chunk=1)
        assert _page_hit_score([single_page], "doc.pdf", 7) == 1.0
        assert _page_hit_score([single_page], "doc.pdf", 8) == 0.0
        print("   OK")

        print("\n[3/3] Wrong document never hits regardless of page...")
        assert _page_hit_score([multi_page], "other.pdf", 2) == 0.0
        print("   OK")

        print("\nPASSED: _page_hit_score() checks the full page range, not just the first page.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_page_hit_score()
    sys.exit(0 if success else 1)
