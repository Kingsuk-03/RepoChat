"""Hybrid retrieval: BM25 + dense vectors fused via Reciprocal Rank Fusion.

Why RRF instead of weighted score sum: the two retrievers produce scores on
totally different scales (BM25 is unbounded, cosine sim is [0,1]). Min-max
normalizing across queries is fragile; RRF only cares about ranks, which
makes it robust without any tuning.

RRF formula: score(d) = sum over retrievers of 1 / (k + rank(d))
We use k=60 (the canonical value from the original RRF paper).
"""
from __future__ import annotations

import logging
from typing import Iterable

from repochat.config import settings
from repochat.retrieval import keyword_store, vector_store

log = logging.getLogger(__name__)

_RRF_K = 60


def _rrf_merge(
    *result_lists: list[dict],
    k: int = _RRF_K,
) -> list[dict]:
    """Reciprocal rank fusion over multiple ranked lists, deduped by chunk_id.

    Returns a list of merged dicts (chunk_id, content, metadata, rrf_score),
    sorted by rrf_score descending.
    """
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            contribution = 1.0 / (k + rank + 1)  # rank is 0-indexed; +1 for paper formula
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": contribution,
                }
            else:
                fused[cid]["rrf_score"] += contribution
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(
    repo_url: str,
    commit_sha: str,
    query: str,
    *,
    top_k_bm25: int | None = None,
    top_k_vector: int | None = None,
) -> list[dict]:
    """Return RRF-fused candidates from BM25 and vector search.

    Caller is expected to apply reranking (or take top_k_final directly) on
    the returned list.
    """
    top_k_bm25 = top_k_bm25 or settings.top_k_bm25
    top_k_vector = top_k_vector or settings.top_k_vector

    bm25_hits = keyword_store.query(repo_url, commit_sha, query, top_k=top_k_bm25)
    vec_hits = vector_store.query(repo_url, commit_sha, query, top_k=top_k_vector)

    log.debug(
        "hybrid_search: bm25=%d vec=%d for query=%r",
        len(bm25_hits), len(vec_hits), query[:60],
    )

    return _rrf_merge(bm25_hits, vec_hits)
