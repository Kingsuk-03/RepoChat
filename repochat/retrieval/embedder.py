"""Sentence-transformers embedder.

Loaded lazily because the model download (~80MB on first run) can take 10s
even on a fast connection, and we don't want to block app startup. After the
first call, the model lives in memory for the process lifetime.

Output vectors are L2-normalized so downstream cosine similarity reduces to
a dot product, which is what Chroma's default distance computes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from repochat.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_MODEL: "SentenceTransformer | None" = None


def _load() -> "SentenceTransformer":
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy import
        log.info("Loading embedding model %s ...", settings.embedding_model)
        _MODEL = SentenceTransformer(settings.embedding_model)
        log.info("Embedding model ready")
    return _MODEL


def embed_texts(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Encode a batch of texts. Returns a list of float lists (Chroma-friendly)."""
    if not texts:
        return []
    model = _load()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return _load().get_sentence_embedding_dimension()
