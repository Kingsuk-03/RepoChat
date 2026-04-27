"""ChromaDB vector store.

One collection per repo, named by a slugified repo identifier. Each repo has
its own collection so queries don't bleed across repos and we can delete a
repo's index in one call.
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from repochat.config import settings
from repochat.ingestion.chunker import Chunk
from repochat.retrieval.embedder import embed_query, embed_texts

if TYPE_CHECKING:
    import chromadb

log = logging.getLogger(__name__)

_CLIENT: "chromadb.api.ClientAPI | None" = None


def _client():
    """Lazy-init a persistent Chroma client.

    Chroma's PersistentClient writes a SQLite DB plus per-collection HNSW
    indexes under chroma_dir. Telemetry is disabled because (a) we don't
    need it and (b) it occasionally throws warnings on HF Spaces.
    """
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        settings.ensure_dirs()
        _CLIENT = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    return _CLIENT


def collection_name(repo_url: str, commit_sha: str) -> str:
    """Stable collection name. Chroma allows alphanumerics, _, -, with length
    constraints — hashing keeps us safely inside that envelope."""
    h = hashlib.sha1(f"{repo_url}@{commit_sha}".encode()).hexdigest()[:16]
    return f"repo_{h}"


def get_or_create_collection(repo_url: str, commit_sha: str):
    return _client().get_or_create_collection(
        name=collection_name(repo_url, commit_sha),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(repo_url: str, commit_sha: str, chunks: list[Chunk]) -> None:
    """Embed and insert chunks. Idempotent: re-adding the same chunk_id upserts."""
    if not chunks:
        return
    coll = get_or_create_collection(repo_url, commit_sha)
    contents = [c.content for c in chunks]
    embeddings = embed_texts(contents)
    coll.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=contents,
        metadatas=[c.as_metadata() for c in chunks],
    )
    log.info("Indexed %d chunks into %s", len(chunks), coll.name)


def query(
    repo_url: str,
    commit_sha: str,
    query_text: str,
    *,
    top_k: int,
) -> list[dict]:
    """Vector search. Returns a list of dicts with chunk_id, content, metadata, score.

    Score is similarity in [0, 1] (we use cosine + normalized embeddings).
    """
    coll = get_or_create_collection(repo_url, commit_sha)
    if coll.count() == 0:
        return []
    q_emb = embed_query(query_text)
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        # cosine distance is 1 - cosine_similarity; convert for downstream RRF
        # to compare directly with BM25 scores (handled by the fusion ranker).
        out.append(
            {
                "chunk_id": cid,
                "content": doc,
                "metadata": meta,
                "score": 1.0 - float(dist),
            }
        )
    return out


def delete_collection(repo_url: str, commit_sha: str) -> None:
    try:
        _client().delete_collection(collection_name(repo_url, commit_sha))
    except Exception as e:  # collection may not exist
        log.debug("delete_collection: %s", e)


def collection_exists(repo_url: str, commit_sha: str) -> bool:
    try:
        coll = _client().get_collection(collection_name(repo_url, commit_sha))
        return coll.count() > 0
    except Exception:
        return False
