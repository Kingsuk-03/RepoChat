"""File filtering for repo ingestion.

The filter is the single most important quality lever in the pipeline: every
junk file kept is one that pollutes retrieval and burns embedding compute.
Rules are intentionally aggressive — when in doubt, drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Directory names that are dropped wherever they appear in the tree.
DROP_DIRS: frozenset[str] = frozenset(
    {
        ".git", "node_modules", "dist", "build", ".next", "__pycache__",
        "venv", ".venv", "env", "target", "vendor", "bin", "obj",
        ".idea", ".vscode", "coverage", ".pytest_cache", ".mypy_cache",
        ".tox", "out", ".cache", ".gradle", ".terraform",
    }
)

# Extensions we never index. Note: the leading dot is included.
DROP_EXTS: frozenset[str] = frozenset(
    {
        ".lock", ".log", ".min.js", ".min.css", ".map",
        ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
        ".mp4", ".mp3", ".wav", ".avi", ".mov",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".pdf", ".exe", ".dll", ".so", ".dylib", ".a",
        ".pyc", ".class", ".o", ".obj",
    }
)

# Specific filenames to drop regardless of extension.
DROP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json", "yarn.lock", "poetry.lock",
        "Cargo.lock", "pnpm-lock.yaml", "Pipfile.lock",
        "composer.lock", "Gemfile.lock",
    }
)

# Source extensions we want chunked with tree-sitter when possible.
SOURCE_EXTS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
        ".java", ".go", ".rs", ".rb", ".cpp", ".cc", ".cxx",
        ".c", ".h", ".hpp", ".swift", ".kt", ".kts", ".cs",
        ".php", ".scala", ".lua", ".r", ".jl", ".dart",
    }
)

DOC_EXTS: frozenset[str] = frozenset({".md", ".rst", ".txt", ".adoc"})
CONFIG_EXTS: frozenset[str] = frozenset({".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"})

# Priority ranking for over-MAX_FILES truncation. Lower = higher priority.
_PRIORITY_DIRS = ("src/", "lib/", "app/", "pkg/", "internal/")


@dataclass
class FilteredFile:
    abs_path: Path
    rel_path: str   # POSIX-style, repo-root-relative
    size_bytes: int


def _has_drop_dir_in_path(rel_path: Path) -> bool:
    return any(part in DROP_DIRS for part in rel_path.parts)


def _is_dropped_filename(name: str) -> bool:
    if name in DROP_FILENAMES:
        return True
    lower = name.lower()
    # Compound extensions like .min.js need substring check, not just suffix.
    for ext in DROP_EXTS:
        if lower.endswith(ext):
            return True
    return False


def _priority_key(rel_path: str) -> tuple[int, int, str]:
    """Sort key for truncation when a repo exceeds MAX_FILES.

    Lower tuple sorts first => kept first. We prefer:
    1. Files inside conventional source directories (src/, lib/, etc.)
    2. Then top-level files (depth 1)
    3. Then everything else, deeper paths last
    """
    depth = rel_path.count("/")
    for i, prefix in enumerate(_PRIORITY_DIRS):
        if rel_path.startswith(prefix):
            return (0, i, rel_path)
    if depth == 0:
        return (1, 0, rel_path)
    return (2, depth, rel_path)


def filter_repo(
    repo_root: Path,
    *,
    max_file_size_kb: int,
    max_files: int,
) -> tuple[list[FilteredFile], int]:
    """Walk repo_root and return (kept_files, total_seen_before_truncation).

    The second return value lets callers tell the user "we indexed 500 of 2,134
    files" without re-walking the tree.
    """
    max_bytes = max_file_size_kb * 1024
    kept: list[FilteredFile] = []
    repo_root = repo_root.resolve()

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue  # symlink escape; skip

        if _has_drop_dir_in_path(rel):
            continue
        if _is_dropped_filename(path.name):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > max_bytes:
            continue

        kept.append(
            FilteredFile(
                abs_path=path,
                rel_path=rel.as_posix(),
                size_bytes=size,
            )
        )

    total_seen = len(kept)
    if total_seen > max_files:
        kept.sort(key=lambda f: _priority_key(f.rel_path))
        kept = kept[:max_files]

    return kept, total_seen


def is_source_file(rel_path: str) -> bool:
    suffix = Path(rel_path).suffix.lower()
    return suffix in SOURCE_EXTS


def is_doc_file(rel_path: str) -> bool:
    return Path(rel_path).suffix.lower() in DOC_EXTS


def is_config_file(rel_path: str) -> bool:
    return Path(rel_path).suffix.lower() in CONFIG_EXTS
