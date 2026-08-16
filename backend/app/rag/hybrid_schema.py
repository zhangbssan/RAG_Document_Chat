from __future__ import annotations

from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.orm.schema import CollectionSchema
from pymilvus.milvus_client.index import IndexParams


def build_realtime_pdf_schema(client: MilvusClient, dim: int) -> CollectionSchema:
    """Schema for realtime_pdf_collection: dense embedding + Milvus-native BM25 sparse vector.

    `text` has enable_analyzer=True (required for the BM25 Function to tokenize it) and
    `sparse_vector` is a Function output field — never written to directly by add_chunks().
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"type": "standard"},
    )
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(
        Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    return schema


def build_realtime_pdf_index_params(client: MilvusClient) -> IndexParams:
    """Index params for realtime_pdf_collection: dense AUTOINDEX/COSINE + sparse AUTOINDEX/BM25."""
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse_vector", index_type="AUTOINDEX", metric_type="BM25")
    return index_params