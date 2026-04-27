"""AST-aware code chunker.

Strategy:
- Source files in supported languages: parse with tree-sitter, extract
  top-level functions/methods/classes as one chunk each. Long functions
  get split into overlapping windows.
- Source files in unsupported languages or on parse failure: 50-line
  sliding windows with 10-line overlap.
- Markdown: split on `##` headers.
- Other text (config, docs): treat as single chunk if small, else windowed.

Each chunk gets a stable `chunk_id` (uuid4) plus enough metadata to (a) cite
back to GitHub, and (b) reconstruct context for the LLM prompt.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from repochat.ingestion.filter import FilteredFile, is_doc_file, is_source_file

log = logging.getLogger(__name__)

# tree-sitter grammar names per file extension. Keys must be lowercase.
# Values must match grammars exposed by tree-sitter-language-pack.
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
}

# tree-sitter node types that we treat as "top-level chunks" per language.
# Different grammars name things differently — this mapping is empirical.
CHUNKABLE_NODES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "generator_function_declaration",
        "lexical_declaration",  # const/let assigning a function
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
    },
    "tsx": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c": {"function_definition", "struct_specifier"},
    "ruby": {"method", "class", "module"},
}

# Long-function splitting parameters (line counts).
LONG_FN_THRESHOLD = 80
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
# Fallback sliding window when AST parsing fails or language unsupported.
FALLBACK_WINDOW = 50
FALLBACK_OVERLAP = 10

ChunkType = Literal["function", "class", "block", "markdown"]


@dataclass
class Chunk:
    chunk_id: str
    filepath: str  # repo-root-relative, POSIX-style
    language: str  # tree-sitter grammar name, or "text" / "markdown"
    chunk_type: ChunkType
    name: str | None  # function/class name when extractable
    line_start: int  # 1-indexed, inclusive
    line_end: int    # 1-indexed, inclusive
    content: str
    repo_url: str
    commit_sha: str
    metadata: dict = field(default_factory=dict)

    def as_metadata(self) -> dict:
        """Flatten to a Chroma-compatible metadata dict (no nested objects)."""
        return {
            "chunk_id": self.chunk_id,
            "filepath": self.filepath,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name or "",
            "line_start": self.line_start,
            "line_end": self.line_end,
            "repo_url": self.repo_url,
            "commit_sha": self.commit_sha,
        }


# --- tree-sitter loader (lazy, cached) -------------------------------------

_PARSER_CACHE: dict[str, object] = {}


def _get_parser(language: str):
    """Lazy-load a tree-sitter parser. Cached across calls.

    Importing tree-sitter-language-pack at module load is expensive (~200ms),
    and parsers themselves take ~50ms each. Cache aggressively.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        parser = get_parser(language)
        _PARSER_CACHE[language] = parser
        return parser
    except Exception as e:
        log.warning("tree-sitter parser for %s unavailable: %s", language, e)
        _PARSER_CACHE[language] = None
        return None


# --- chunkers ---------------------------------------------------------------


def _extract_name(node, source_bytes: bytes) -> str | None:
    """Best-effort name extraction from a tree-sitter node.

    Most grammars expose a child named 'name' for functions/classes. When they
    don't, we return None and let the caller fall back to chunk content.
    """
    name_node = node.child_by_field_name("name")
    if name_node:
        try:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _split_long(text: str, line_start: int) -> Iterable[tuple[int, int, str]]:
    """Yield (line_start, line_end, content) windows for a long function body."""
    lines = text.splitlines(keepends=True)
    n = len(lines)
    step = WINDOW_LINES - WINDOW_OVERLAP
    i = 0
    while i < n:
        window = lines[i : i + WINDOW_LINES]
        if not window:
            break
        ls = line_start + i
        le = ls + len(window) - 1
        yield ls, le, "".join(window)
        if i + WINDOW_LINES >= n:
            break
        i += step


def _chunk_with_ast(
    rel_path: str,
    source_text: str,
    language: str,
    repo_url: str,
    commit_sha: str,
) -> list[Chunk]:
    parser = _get_parser(language)
    if parser is None:
        return _chunk_sliding(rel_path, source_text, language, repo_url, commit_sha)

    try:
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception as e:
        log.warning("AST parse failed for %s: %s; falling back to sliding window", rel_path, e)
        return _chunk_sliding(rel_path, source_text, language, repo_url, commit_sha)

    chunkable = CHUNKABLE_NODES.get(language, set())
    chunks: list[Chunk] = []

    def visit(node, depth: int = 0):
        # Only consume top-level definitions (and class members one level deep).
        # Beyond that, deeply nested closures aren't useful as standalone chunks.
        if node.type in chunkable:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            name = _extract_name(node, source_bytes)
            ctype: ChunkType = (
                "class" if "class" in node.type or "interface" in node.type or "struct" in node.type or "trait" in node.type
                else "function"
            )

            line_count = end_line - start_line + 1
            if line_count > LONG_FN_THRESHOLD:
                for ls, le, sub in _split_long(content, start_line):
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            filepath=rel_path,
                            language=language,
                            chunk_type=ctype,
                            name=name,
                            line_start=ls,
                            line_end=le,
                            content=sub,
                            repo_url=repo_url,
                            commit_sha=commit_sha,
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        filepath=rel_path,
                        language=language,
                        chunk_type=ctype,
                        name=name,
                        line_start=start_line,
                        line_end=end_line,
                        content=content,
                        repo_url=repo_url,
                        commit_sha=commit_sha,
                    )
                )
            # For class bodies, also descend one level to capture methods.
            if ctype == "class" and depth == 0:
                for child in node.children:
                    visit(child, depth + 1)
            return

        # Recurse into top-level container nodes (module, program, source_file).
        if depth == 0:
            for child in node.children:
                visit(child, depth)

    visit(tree.root_node, 0)

    # If the AST visit produced nothing (e.g. a module with only top-level
    # statements and no defs), fall back to a sliding window so we don't lose
    # the file entirely.
    if not chunks:
        return _chunk_sliding(rel_path, source_text, language, repo_url, commit_sha)

    return chunks


def _chunk_sliding(
    rel_path: str,
    source_text: str,
    language: str,
    repo_url: str,
    commit_sha: str,
) -> list[Chunk]:
    lines = source_text.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []
    step = FALLBACK_WINDOW - FALLBACK_OVERLAP
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        window = lines[i : i + FALLBACK_WINDOW]
        if not window:
            break
        ls = i + 1
        le = ls + len(window) - 1
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                filepath=rel_path,
                language=language,
                chunk_type="block",
                name=None,
                line_start=ls,
                line_end=le,
                content="".join(window),
                repo_url=repo_url,
                commit_sha=commit_sha,
            )
        )
        if i + FALLBACK_WINDOW >= n:
            break
        i += step
    return chunks


def _chunk_markdown(
    rel_path: str, source_text: str, repo_url: str, commit_sha: str
) -> list[Chunk]:
    """Split markdown on `##` headers. Each chunk is a section."""
    lines = source_text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    section_lines: list[str] = []
    section_start = 1
    section_name: str | None = None

    def flush(end_line: int) -> None:
        nonlocal section_lines, section_start, section_name
        if section_lines:
            content = "".join(section_lines).strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        filepath=rel_path,
                        language="markdown",
                        chunk_type="markdown",
                        name=section_name,
                        line_start=section_start,
                        line_end=end_line,
                        content="".join(section_lines),
                        repo_url=repo_url,
                        commit_sha=commit_sha,
                    )
                )
        section_lines = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # Treat any heading depth as a split point. Top-level docs often only
        # use `#`, while READMEs mix `#` and `##`.
        if stripped.startswith("#") and not stripped.startswith("#!"):
            flush(idx - 1)
            section_start = idx
            section_name = stripped.lstrip("#").strip() or None
        section_lines.append(line)
    flush(len(lines))

    if not chunks and source_text.strip():
        # No headers at all — emit the whole file as one chunk.
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                filepath=rel_path,
                language="markdown",
                chunk_type="markdown",
                name=None,
                line_start=1,
                line_end=len(lines) or 1,
                content=source_text,
                repo_url=repo_url,
                commit_sha=commit_sha,
            )
        )
    return chunks


# --- public entry point -----------------------------------------------------


def chunk_file(
    f: FilteredFile,
    *,
    repo_url: str,
    commit_sha: str,
) -> list[Chunk]:
    """Chunk a single file. Returns [] on read failure (already logged)."""
    try:
        text = f.abs_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        log.warning("Could not read %s: %s", f.rel_path, e)
        return []

    if not text.strip():
        return []

    if is_doc_file(f.rel_path) and Path(f.rel_path).suffix.lower() == ".md":
        return _chunk_markdown(f.rel_path, text, repo_url, commit_sha)

    if is_source_file(f.rel_path):
        ext = Path(f.rel_path).suffix.lower()
        language = EXT_TO_LANG.get(ext)
        if language:
            return _chunk_with_ast(f.rel_path, text, language, repo_url, commit_sha)
        return _chunk_sliding(f.rel_path, text, "text", repo_url, commit_sha)

    # Configs and other docs: one chunk if small, else windowed.
    if len(text.splitlines()) <= FALLBACK_WINDOW:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                filepath=f.rel_path,
                language="text",
                chunk_type="block",
                name=None,
                line_start=1,
                line_end=max(1, len(text.splitlines())),
                content=text,
                repo_url=repo_url,
                commit_sha=commit_sha,
            )
        ]
    return _chunk_sliding(f.rel_path, text, "text", repo_url, commit_sha)


def chunk_files(
    files: Iterable[FilteredFile],
    *,
    repo_url: str,
    commit_sha: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(f, repo_url=repo_url, commit_sha=commit_sha))
    return chunks
