#!/usr/bin/env python
"""Test hybrid_schema.py builders in isolation: field/function/index shape only."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from pymilvus import DataType, FunctionType, MilvusClient

from app.rag.hybrid_schema import build_realtime_pdf_index_params, build_realtime_pdf_schema


def test_hybrid_schema() -> bool:
    print("=" * 70)
    print("HYBRID SCHEMA BUILDER TEST")
    print("=" * 70)

    client = MilvusClient.__new__(MilvusClient)  # only need create_schema/prepare_index_params factories

    try:
        print("\n[1/3] Building schema...")
        schema = build_realtime_pdf_schema(client, dim=384)
        fields = {f.name: f for f in schema.fields}
        assert set(fields) == {"id", "text", "embedding", "sparse_vector"}, f"unexpected fields: {set(fields)}"
        assert fields["id"].is_primary is True
        assert fields["text"].dtype == DataType.VARCHAR
        assert fields["text"].params.get("enable_analyzer") is True, f"text field params: {fields['text'].params}"
        assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
        assert fields["embedding"].params.get("dim") == 384
        assert fields["sparse_vector"].dtype == DataType.SPARSE_FLOAT_VECTOR
        print("   OK: id/text/embedding/sparse_vector fields present, text has enable_analyzer=True")

        print("\n[2/3] Checking the BM25 Function...")
        assert len(schema.functions) == 1, f"expected exactly one Function, got {schema.functions}"
        fn = schema.functions[0]
        assert fn.type == FunctionType.BM25, f"expected BM25, got {fn.type}"
        assert fn.input_field_names == ["text"]
        assert fn.output_field_names == ["sparse_vector"]
        print(f"   OK: Function(name={fn.name!r}, type=BM25, input=['text'], output=['sparse_vector'])")

        print("\n[3/3] Checking index params...")
        index_params = build_realtime_pdf_index_params(client)
        by_field = {p.field_name: p.to_dict() for p in index_params}
        assert by_field["embedding"]["metric_type"] == "COSINE"
        assert by_field["sparse_vector"]["metric_type"] == "BM25"
        print(f"   OK: {dict(by_field)}")

        print("\nPASSED: hybrid_schema.py builders produce the expected shape.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_hybrid_schema()
    sys.exit(0 if success else 1)
