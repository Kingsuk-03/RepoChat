"""Cross-encoder reranker.

Cross-encoders are slower than bi-encoders (they jointly encode query+doc
rather than scoring against pre-computed embeddings) but much more accurate
for the top of the list. We only ever rerank ~20 candidates, so latency is
fine on CPU.

This module is a no-op when USE_RERANKER=false. The hybrid search results
are returned unchanged.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from repochat.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

_MODEL: "CrossEncoder | None" = None


def _load() -> "CrossEncoder | None":
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder %s ...", settings.reranker_model)
            _MODEL = CrossEncoder(settings.reranker_model)
            log.info("Cross-encoder ready")
        except Exception as e:
            log.error("Failed to load reranker: %s. Continuing without rerank.", e)
            _MODEL = None
    return _MODEL


def rerank(query: str, candidates: list[dict], *, top_k: int) -> list[dict]:
    """Rerank candidates by cross-encoder relevance to the query.

    If reranker is disabled or fails to load, returns candidates[:top_k]
    in their original order — degrades gracefully.
    """
    if not candidates:
        return []
    if not settings.use_reranker:
        return candidates[:top_k]

    model = _load()
    if model is None:
        return candidates[:top_k]

    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
