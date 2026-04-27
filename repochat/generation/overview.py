"""Capability 1: project overview generation.

Single LLM call. We prepare four context sections (README, file tree,
tech-stack manifests, sampled source files) and stuff them into the
overview prompt.

The cost of being thorough here is paid once per `(repo_url, commit_sha)` —
results cached on disk so repeat visits are instant.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from repochat.config import settings
from repochat.generation import llm
from repochat.ingestion.filter import FilteredFile
from repochat.prompts.overview_prompt import OVERVIEW_PROMPT
from repochat.utils.github import RepoRef

log = logging.getLogger(__name__)

README_CANDIDATES = ("README.md", "README.rst", "README.txt", "Readme.md", "readme.md")
MANIFEST_FILES = (
    "package.json", "requirements.txt", "pyproject.toml", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "Gemfile", "composer.json",
    "setup.py", "Pipfile",
)


def _read_readme(repo_root: Path) -> str:
    for name in README_CANDIDATES:
        p = repo_root / name
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                # Trim very long READMEs to stay within prompt budget.
                if len(content) > 8000:
                    content = content[:8000] + "\n\n[... README truncated ...]"
                return content
            except OSError:
                continue
    return "(no README found)"


def _build_file_tree(repo_root: Path, max_depth: int = 3, max_entries: int = 200) -> str:
    """Compact text rendering of the top-N levels of the directory tree."""
    repo_root = repo_root.resolve()
    lines: list[str] = []
    count = 0
    for path in sorted(repo_root.rglob("*")):
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        # Skip the standard junk directories.
        if any(p in {".git", "node_modules", "__pycache__", ".venv", "venv",
                     "dist", "build", "target", "vendor"} for p in rel.parts):
            continue
        indent = "  " * (depth - 1)
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{rel.parts[-1]}{suffix}")
        count += 1
        if count >= max_entries:
            lines.append("[... tree truncated ...]")
            break
    return "\n".join(lines)


def _read_manifests(repo_root: Path) -> str:
    """Concatenate any tech-stack manifest files we find at repo root."""
    out_parts: list[str] = []
    for name in MANIFEST_FILES:
        p = repo_root / name
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > 4000:
                    content = content[:4000] + "\n[... truncated ...]"
                out_parts.append(f"### {name}\n{content}\n")
            except OSError:
                continue
    return "\n".join(out_parts) if out_parts else "(no standard manifest files found)"


def _sample_files(files: list[FilteredFile], n: int = 6) -> list[FilteredFile]:
    """Pick representative source files for the LLM to skim.

    Strategy:
    1. Anything that looks like an entry point (main.py, app.py, index.{js,ts}, etc.)
    2. Top-level files in src/ or lib/ (the public surface)
    3. Pad with arbitrary source files
    """
    entry_names = {
        "main.py", "app.py", "__main__.py", "cli.py",
        "index.js", "index.ts", "main.js", "main.ts", "server.js", "server.ts",
        "main.go", "main.rs", "Main.java",
    }
    entry_points = [f for f in files if Path(f.rel_path).name in entry_names]
    src_top = [
        f for f in files
        if (f.rel_path.startswith("src/") or f.rel_path.startswith("lib/"))
        and f.rel_path.count("/") <= 2
        and f not in entry_points
    ]
    others = [f for f in files if f not in entry_points and f not in src_top]

    picked: list[FilteredFile] = []
    for bucket in (entry_points, src_top, others):
        for f in bucket:
            if len(picked) >= n:
                return picked
            if f.size_bytes < 60_000:  # avoid stuffing one giant file
                picked.append(f)
    return picked


def _format_sampled(files: list[FilteredFile]) -> str:
    chunks: list[str] = []
    budget = 0
    for f in files:
        try:
            content = f.abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Per-file cap so a single big file can't dominate.
        if len(content) > 4000:
            content = content[:4000] + "\n[... file truncated ...]"
        budget += len(content)
        chunks.append(f"### {f.rel_path}\n```\n{content}\n```\n")
        if budget > 20_000:
            chunks.append("[... remaining samples truncated ...]")
            break
    return "\n".join(chunks) if chunks else "(no source files sampled)"


def _cache_path(ref: RepoRef, commit_sha: str) -> Path:
    settings.ensure_dirs()
    return settings.overviews_dir / f"{ref.owner}__{ref.repo}__{commit_sha[:12]}.md"


def generate_overview(
    *,
    ref: RepoRef,
    commit_sha: str,
    repo_root: Path,
    files: list[FilteredFile],
    use_cache: bool = True,
) -> str:
    cache_file = _cache_path(ref, commit_sha)
    if use_cache and cache_file.exists():
        log.info("Overview cache hit: %s", cache_file)
        return cache_file.read_text(encoding="utf-8")

    readme = _read_readme(repo_root)
    tree = _build_file_tree(repo_root)
    manifests = _read_manifests(repo_root)
    sampled = _sample_files(files)
    sampled_text = _format_sampled(sampled)

    prompt = OVERVIEW_PROMPT.format(
        readme_content=readme,
        file_tree=tree,
        manifests=manifests,
        sampled_files=sampled_text,
    )

    log.info("Generating overview for %s (prompt length=%d chars)", ref.slug, len(prompt))
    result = llm.complete(prompt, temperature=0.2, max_tokens=1200)
    overview = result.text.strip()

    cache_file.write_text(overview, encoding="utf-8")
    # Sidecar metadata for debugging.
    cache_file.with_suffix(".json").write_text(
        json.dumps(
            {
                "repo": ref.slug,
                "commit": commit_sha,
                "provider": result.provider,
                "model": result.model,
                "files_sampled": [f.rel_path for f in sampled],
            },
            indent=2,
        )
    )
    return overview
