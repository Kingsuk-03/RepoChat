"""LRU cache for cloned repos and their indexes.

We track per-repo cache state in a single JSON manifest under cache_dir.
On every successful index, we write a fresh entry; on each query, we touch
the entry's `last_used` timestamp.

When total cache size exceeds MAX_CACHE_GB, we evict least-recently-used
entries until under the cap. Evicting a repo deletes its cloned source,
its Chroma collection, and its BM25 pickle.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from repochat.config import settings
from repochat.retrieval import keyword_store, vector_store

log = logging.getLogger(__name__)

_LOCK = threading.Lock()


@dataclass
class CacheEntry:
    repo_url: str
    commit_sha: str
    branch: str
    local_path: str  # str so it serializes to JSON cleanly
    indexed_at: float
    last_used: float
    chunk_count: int


def _manifest_path() -> Path:
    settings.ensure_dirs()
    return settings.cache_dir / "manifest.json"


def _load_manifest() -> dict[str, CacheEntry]:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {k: CacheEntry(**v) for k, v in raw.items()}
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("Cache manifest corrupt (%s); resetting", e)
        return {}


def _save_manifest(manifest: dict[str, CacheEntry]) -> None:
    _manifest_path().write_text(
        json.dumps({k: asdict(v) for k, v in manifest.items()}, indent=2)
    )


def _key(repo_url: str, commit_sha: str) -> str:
    return f"{repo_url}@{commit_sha}"


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def total_cache_bytes() -> int:
    return _dir_size(settings.cache_dir)


def register(
    *,
    repo_url: str,
    commit_sha: str,
    branch: str,
    local_path: Path,
    chunk_count: int,
) -> None:
    """Record a freshly indexed repo and trigger eviction if over budget."""
    with _LOCK:
        manifest = _load_manifest()
        now = time.time()
        manifest[_key(repo_url, commit_sha)] = CacheEntry(
            repo_url=repo_url,
            commit_sha=commit_sha,
            branch=branch,
            local_path=str(local_path),
            indexed_at=now,
            last_used=now,
            chunk_count=chunk_count,
        )
        _save_manifest(manifest)
    _evict_if_needed()


def touch(repo_url: str, commit_sha: str) -> None:
    """Update last_used. Called on every query against this repo."""
    with _LOCK:
        manifest = _load_manifest()
        entry = manifest.get(_key(repo_url, commit_sha))
        if entry:
            entry.last_used = time.time()
            _save_manifest(manifest)


def get(repo_url: str, commit_sha: str) -> CacheEntry | None:
    return _load_manifest().get(_key(repo_url, commit_sha))


def list_recent(limit: int = 5) -> list[CacheEntry]:
    """Return the N most-recently-used cache entries, newest first."""
    entries = list(_load_manifest().values())
    entries.sort(key=lambda e: e.last_used, reverse=True)
    return entries[:limit]


def evict(repo_url: str, commit_sha: str) -> None:
    """Hard-delete a repo's clone and indexes. Used by eviction and manual reset."""
    with _LOCK:
        manifest = _load_manifest()
        entry = manifest.pop(_key(repo_url, commit_sha), None)
        if entry:
            local = Path(entry.local_path)
            if local.exists():
                shutil.rmtree(local, ignore_errors=True)
            vector_store.delete_collection(repo_url, commit_sha)
            keyword_store.delete_index(repo_url, commit_sha)
            _save_manifest(manifest)
            log.info("Evicted %s @ %s", repo_url, commit_sha[:8])


def _evict_if_needed() -> None:
    cap_bytes = int(settings.max_cache_gb * 1024 * 1024 * 1024)
    while total_cache_bytes() > cap_bytes:
        manifest = _load_manifest()
        if not manifest:
            return
        oldest = min(manifest.values(), key=lambda e: e.last_used)
        log.info(
            "Cache over budget (%.2fGB > %.2fGB); evicting %s",
            total_cache_bytes() / 1e9, settings.max_cache_gb, oldest.repo_url,
        )
        evict(oldest.repo_url, oldest.commit_sha)
