from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import Settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(Settings.embedding_model_name)
