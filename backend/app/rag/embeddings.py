from __future__ import annotations

import os
from functools import lru_cache

from app.config import EMBEDDING_MODEL


class LocalEmbeddingFunction:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = load_embedding_model(model_name)

    def name(self) -> str:
        return "sentence_transformer"

    def __call__(self, input):
        return self.model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        restore_env("HF_HUB_OFFLINE", previous_hf_offline)
        restore_env("TRANSFORMERS_OFFLINE", previous_transformers_offline)
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
