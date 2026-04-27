"""Central configuration. Single source of truth for env vars, paths, and constants.

Detects the runtime environment (local vs Hugging Face Spaces) and picks
appropriate cache paths. Everything downstream imports `settings` from here
rather than reading os.environ directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env if present. On HF Spaces, secrets are injected into os.environ
# directly, so dotenv finding nothing is fine.
load_dotenv()


def _is_hf_spaces() -> bool:
    """HF Spaces sets SPACE_ID for every running Space."""
    return os.getenv("SPACE_ID") is not None


def _default_cache_dir() -> Path:
    """On HF Spaces, /data is the persistent-storage mount (when enabled).

    If persistent storage isn't enabled on the Space, /data won't be writable
    and we fall back to /tmp, which means the cache evaporates on restart.
    That's acceptable: re-indexing on cold start is cheap relative to user time.
    """
    if _is_hf_spaces():
        persistent = Path("/data")
        if persistent.exists() and os.access(persistent, os.W_OK):
            return persistent
        return Path("/tmp/repochat_cache")
    return Path(os.getenv("CACHE_DIR", "./.repochat_cache")).resolve()


class Settings(BaseModel):
    # --- LLM ---
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    llm_provider: Literal["groq", "gemini"] = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq")  # type: ignore
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )

    # --- Embeddings / reranker ---
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    reranker_model: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    use_reranker: bool = Field(
        default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() == "true"
    )

    # --- Retrieval ---
    top_k_bm25: int = Field(default_factory=lambda: int(os.getenv("TOP_K_BM25", "15")))
    top_k_vector: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K_VECTOR", "15"))
    )
    top_k_final: int = Field(default_factory=lambda: int(os.getenv("TOP_K_FINAL", "5")))

    # --- Ingestion limits ---
    max_files: int = Field(default_factory=lambda: int(os.getenv("MAX_FILES", "500")))
    max_file_size_kb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_KB", "300"))
    )

    # --- Cache ---
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    max_cache_gb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_CACHE_GB", "5"))
    )

    # --- Misc ---
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    github_token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))

    # --- Rate limiting (Groq free tier is 30 RPM; we leave headroom) ---
    rpm_limit: int = 25

    # Derived paths
    @property
    def repos_dir(self) -> Path:
        return self.cache_dir / "repos"

    @property
    def chroma_dir(self) -> Path:
        return self.cache_dir / "chroma"

    @property
    def overviews_dir(self) -> Path:
        return self.cache_dir / "overviews"

    @property
    def is_hf_spaces(self) -> bool:
        return _is_hf_spaces()

    def ensure_dirs(self) -> None:
        for p in (self.cache_dir, self.repos_dir, self.chroma_dir, self.overviews_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()


def configure_logging() -> None:
    """Idempotent. Safe to call from multiple entry points (app.py, eval runner)."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
