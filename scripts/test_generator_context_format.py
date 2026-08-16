#!/usr/bin/env python
"""Test generator.py's pure string formatting: format_context() / fallback_answer()
with a multi-page Source and a citable link."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.generator import fallback_answer, format_context
from app.schemas import Source


def test_generator_context_format() -> bool:
    print("=" * 70)
    print("GENERATOR CONTEXT FORMATTING TEST")
    print("=" * 70)

    source = Source(
        text="Termination requires 30 days written notice.",
        document="service_agreement.pdf",
        page=2,
        pages=[2, 3],
        chunk="4-5",
        score=0.031,
        link="doc:hash1#p2-3",
    )

    try:
        print("\n[1/2] format_context() shows the merged page range and the citable link...")
        context = format_context([source])
        assert "[service_agreement.pdf-P2-3-S4-5]" in context, context
        assert "Link: doc:hash1#p2-3" in context, context
        assert "Termination requires 30 days written notice." in context
        print("   OK")

        print("\n[2/2] fallback_answer() includes the link alongside the citation...")
        answer = fallback_answer([source])
        assert "[service_agreement.pdf-P2-3-S4-5]" in answer, answer
        assert "(doc:hash1#p2-3)" in answer, answer
        assert "Rank score: 0.0310" in answer, answer
        print("   OK")

        print("\nPASSED: generator.py formatting threads page ranges and citable links through.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_generator_context_format()
    sys.exit(0 if success else 1)
