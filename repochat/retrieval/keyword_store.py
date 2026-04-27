"""BM25 keyword index.

Why BM25 alongside vectors: embeddings handle semantic similarity, BM25
handles exact-token matches. Real questions like "where is the parse_url
function?" need exact lexical match — and dense retrievers are surprisingly
bad at exact rare tokens (the `parse_url` token vector is dominated by
context, not the token itself).

Pickled to disk because rank-bm25 has no native persistence. Re-tokenizing
a 500-file corpus takes <1s, so this is cheap.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from repochat.config import settings
from repochat.ingestion.chunker import Chunk
from repochat.retrieval.vector_store import collection_name

log = logging.getLogger(__name__)

# Code-aware tokenizer: split on non-word boundaries, on underscores, and on
# case transitions, so `getUserById` and `get_user_by_id` both produce
# ['get', 'user', 'by', 'id']. Underscores must be split explicitly because
# Python's \w considers them word characters.
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")  # any non-alphanum, including _


def _tokenize(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text)
    tokens: list[str] = []
    for p in parts:
        if not p:
            continue
        for sub in _CAMEL_RE.split(p):
            sub = sub.lower()
            if sub and len(sub) <= 60:  # drop obvious garbage like base64 blobs
                tokens.append(sub)
    return tokens


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]
    contents: list[str]
    metadatas: list[dict]


def _index_path(repo_url: str, commit_sha: str) -> Path:
    settings.ensure_dirs()
    return settings.cache_dir / "bm25" / f"{collection_name(repo_url, commit_sha)}.pkl"


def build(repo_url: str, commit_sha: str, chunks: list[Chunk]) -> None:
    """Build and persist a BM25 index for a repo's chunks."""
    if not chunks:
        return
    tokenized = [_tokenize(c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    idx = BM25Index(
        bm25=bm25,
        chunk_ids=[c.chunk_id for c in chunks],
        contents=[c.content for c in chunks],
        metadatas=[c.as_metadata() for c in chunks],
    )
    path = _index_path(repo_url, commit_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(idx, f)
    log.info("Built BM25 index for %d chunks at %s", len(chunks), path)


def _load(repo_url: str, commit_sha: str) -> BM25Index | None:
    path = _index_path(repo_url, commit_sha)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def query(
    repo_url: str,
    commit_sha: str,
    query_text: str,
    *,
    top_k: int,
) -> list[dict]:
    idx = _load(repo_url, commit_sha)
    if idx is None:
        return []
    tokens = _tokenize(query_text)
    if not tokens:
        return []
    scores = idx.bm25.get_scores(tokens)
    # Take top_k by score, descending.
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[dict] = []
    for i, score in pairs:
        if score <= 0:
            continue
        out.append(
            {
                "chunk_id": idx.chunk_ids[i],
                "content": idx.contents[i],
                "metadata": idx.metadatas[i],
                "score": float(score),
            }
        )
    return out


def index_exists(repo_url: str, commit_sha: str) -> bool:
    return _index_path(repo_url, commit_sha).exists()


def delete_index(repo_url: str, commit_sha: str) -> None:
    path = _index_path(repo_url, commit_sha)
    if path.exists():
        path.unlink()
