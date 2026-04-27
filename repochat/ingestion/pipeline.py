"""Ingestion pipeline: orchestrates the whole index-a-repo flow.

A typical run goes through these phases (each surfaced via the progress callback):
    1. clone        — git clone --depth 1
    2. filter       — drop junk, enforce MAX_FILES with priority truncation
    3. chunk        — tree-sitter where possible, sliding-window otherwise
    4. embed/index  — Chroma + BM25
    5. register     — record in the LRU cache manifest

The progress callback gets `(phase: str, current: int | None, total: int | None, msg: str)`.
For phases without a known total (cloning, filtering), we pass None.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from repochat.config import settings
from repochat.ingestion import chunker, cloner, filter as flt
from repochat.retrieval import keyword_store, vector_store
from repochat.utils import cache
from repochat.utils.github import RepoRef

log = logging.getLogger(__name__)


ProgressFn = Callable[[str, Optional[int], Optional[int], str], None]


def _noop_progress(phase: str, current: Optional[int], total: Optional[int], msg: str) -> None:
    pass


@dataclass
class IngestionResult:
    ref: RepoRef
    commit_sha: str
    local_path: Path
    files_kept: int
    files_seen: int
    chunk_count: int
    truncated: bool
    seconds: float


def index_repo(
    ref: RepoRef,
    *,
    progress: ProgressFn = _noop_progress,
    embed_batch_size: int = 64,
) -> IngestionResult:
    """End-to-end index. Idempotent: if already indexed at this commit, returns cached state."""
    started = time.time()

    # Phase 1: clone
    progress("clone", None, None, f"Cloning {ref.slug}...")
    cloned = cloner.clone_repo(ref)
    repo_url = cloned.ref.web_url

    # Short-circuit if we've already indexed this exact commit.
    existing = cache.get(repo_url, cloned.commit_sha)
    if existing and vector_store.collection_exists(repo_url, cloned.commit_sha) \
            and keyword_store.index_exists(repo_url, cloned.commit_sha):
        log.info("Repo %s @ %s already indexed; skipping", ref.slug, cloned.commit_sha[:8])
        # The freshly-cloned dir is redundant; clean it up.
        cloner.remove_clone(cloned.local_path)
        cache.touch(repo_url, cloned.commit_sha)
        return IngestionResult(
            ref=cloned.ref,
            commit_sha=cloned.commit_sha,
            local_path=Path(existing.local_path),
            files_kept=0,
            files_seen=0,
            chunk_count=existing.chunk_count,
            truncated=False,
            seconds=time.time() - started,
        )

    # Phase 2: filter
    progress("filter", None, None, "Filtering files...")
    files, total_seen = flt.filter_repo(
        cloned.local_path,
        max_file_size_kb=settings.max_file_size_kb,
        max_files=settings.max_files,
    )
    truncated = total_seen > settings.max_files
    progress(
        "filter", len(files), total_seen,
        f"Kept {len(files)} of {total_seen} files{' (truncated)' if truncated else ''}",
    )

    # Phase 3: chunk
    progress("chunk", None, None, "Chunking code with tree-sitter...")
    chunks: list[chunker.Chunk] = []
    for i, f in enumerate(files, 1):
        chunks.extend(chunker.chunk_file(f, repo_url=repo_url, commit_sha=cloned.commit_sha))
        if i % 25 == 0 or i == len(files):
            progress("chunk", i, len(files), f"Chunked {i}/{len(files)} files ({len(chunks)} chunks)")

    if not chunks:
        # Nothing usable. Clean up and bail.
        cloner.remove_clone(cloned.local_path)
        raise RuntimeError(
            f"No indexable content in {ref.slug}. Repo may be empty or contain only "
            "filtered file types (binaries, images, etc.)."
        )

    # Phase 4: embed + index
    progress("embed", 0, len(chunks), "Loading embedding model...")
    # Embed in batches so we can report progress and avoid OOM on big repos.
    for start in range(0, len(chunks), embed_batch_size):
        batch = chunks[start : start + embed_batch_size]
        vector_store.add_chunks(repo_url, cloned.commit_sha, batch)
        progress(
            "embed", min(start + len(batch), len(chunks)), len(chunks),
            f"Embedded {min(start + len(batch), len(chunks))}/{len(chunks)} chunks",
        )

    progress("bm25", None, None, "Building keyword index...")
    keyword_store.build(repo_url, cloned.commit_sha, chunks)

    # Phase 5: register in LRU cache
    cache.register(
        repo_url=repo_url,
        commit_sha=cloned.commit_sha,
        branch=cloned.branch,
        local_path=cloned.local_path,
        chunk_count=len(chunks),
    )

    elapsed = time.time() - started
    progress("done", None, None, f"Indexed in {elapsed:.1f}s")
    log.info(
        "Indexed %s @ %s: %d chunks from %d/%d files in %.1fs",
        ref.slug, cloned.commit_sha[:8], len(chunks), len(files), total_seen, elapsed,
    )

    return IngestionResult(
        ref=cloned.ref,
        commit_sha=cloned.commit_sha,
        local_path=cloned.local_path,
        files_kept=len(files),
        files_seen=total_seen,
        chunk_count=len(chunks),
        truncated=truncated,
        seconds=elapsed,
    )
