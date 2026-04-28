"""Capability 2: cited Q&A.

Pipeline: question -> hybrid retrieve -> rerank -> format snippets -> stream
LLM -> parse citations into clickable GitHub links.

The citation regex deliberately accepts what real models produce. We've seen
all of these in practice:
  [src/auth/login.py:12-45]
  [src/auth/login.py:12]
  (src/auth/login.py:12-45)

We accept both bracket styles. We also de-duplicate consecutive identical
citations because models love to over-cite.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from repochat.config import settings
from repochat.generation import llm
from repochat.prompts.qa_prompt import QA_PROMPT, REFUSAL_PREFIX
from repochat.retrieval import hybrid, reranker
from repochat.utils.github import RepoRef

log = logging.getLogger(__name__)

# Match [path:start-end] or [path:line] or (path:start-end). Path may include
# slashes, dots, dashes, underscores; line numbers must be digits.
_CITATION_RE = re.compile(
    r"[\[\(]"
    r"(?P<path>[A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)"
    r":"
    r"(?P<start>\d+)"
    r"(?:-(?P<end>\d+))?"
    r"[\]\)]"
)

# Project-level questions don't have specific keywords retrieval can ground on.
# "What does this project do?" wouldn't retrieve relevant code chunks because
# no specific code chunk *is* the answer — the answer lives in the overview
# we already generated. We detect these questions and route them to the
# cached overview directly instead of running retrieval.
_PROJECT_QUESTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat (does|is) (this|the)\s+(project|repo|repository|codebase|library|app|application|tool)",
        r"^\s*what (is|does) (this|it)\s*\??\s*$",
        r"^\s*what (does|do) (this|it) do\s*\??\s*$",
        r"\bexplain (this|the)\s+(project|repo|repository|codebase|library)",
        r"\bdescribe (this|the)\s+(project|repo|repository|codebase|library)",
        r"\bgive (me )?an? (overview|summary|introduction)",
        r"\bsummari[sz]e (this|the)\s+(project|repo|repository|codebase)",
        r"\bpurpose of (this|the)\s+(project|repo|repository|codebase|library)",
        r"\btell me about (this|the)\s+(project|repo|repository|codebase)",
        r"\bwhat'?s (this|the)\s+(project|repo|repository|codebase|library) (about|for|do)",
    )
]


def _is_project_question(q: str) -> bool:
    """True if the question is a project-level meta-question best answered
    from the cached overview rather than via retrieval."""
    return any(p.search(q) for p in _PROJECT_QUESTION_PATTERNS)


def _load_cached_overview(ref: RepoRef, commit_sha: str) -> str | None:
    """Read the overview from disk if it was generated during indexing.

    Returns None if no overview exists for this repo+commit.
    """
    from pathlib import Path
    overview_path = Path(
        settings.overviews_dir / f"{ref.owner}__{ref.repo}__{commit_sha[:12]}.md"
    )
    if overview_path.exists():
        try:
            return overview_path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


@dataclass
class Citation:
    filepath: str
    line_start: int
    line_end: int
    url: str

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "url": self.url,
        }


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # the retrieved chunks shown to LLM
    citations: list[Citation] = field(default_factory=list)  # parsed from text
    refused: bool = False


def _format_chunks(candidates: list[dict]) -> str:
    parts: list[str] = []
    for c in candidates:
        meta = c["metadata"]
        parts.append(
            f"[file: {meta['filepath']}, lines {meta['line_start']}-{meta['line_end']}, "
            f"language: {meta.get('language', 'text')}]\n{c['content']}\n---"
        )
    return "\n".join(parts)


def parse_citations(text: str, ref: RepoRef) -> list[Citation]:
    """Extract citations from answer text. De-dups identical consecutive citations."""
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(text):
        path = m.group("path")
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Citation(
                filepath=path,
                line_start=start,
                line_end=end,
                url=ref.file_url(path, start, end),
            )
        )
    return out


def render_citations_html(text: str, ref: RepoRef) -> str:
    """Replace [path:lines] tokens with HTML pill links for Streamlit rendering."""
    def _replace(m: re.Match) -> str:
        path = m.group("path")
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        url = ref.file_url(path, start, end)
        label = f"{path}:{start}" if start == end else f"{path}:{start}-{end}"
        return (
            f'<a href="{url}" target="_blank" rel="noopener" '
            f'class="citation-pill">{label}</a>'
        )
    return _CITATION_RE.sub(_replace, text)


def answer_question(
    *,
    ref: RepoRef,
    repo_url: str,
    commit_sha: str,
    question: str,
) -> Answer:
    """Synchronous version. For streaming, use stream_answer below."""
    # Short-circuit project-level questions to the cached overview. These
    # questions don't have specific keywords retrieval can ground on, and the
    # overview was generated with full project context (README, manifests,
    # file tree, sampled files) — it's the right answer source.
    if _is_project_question(question):
        overview = _load_cached_overview(ref, commit_sha)
        if overview:
            return Answer(text=overview, sources=[], citations=[], refused=False)

    candidates = hybrid.hybrid_search(repo_url, commit_sha, question)
    if not candidates:
        return Answer(
            text=(
                f"{REFUSAL_PREFIX} The retrieved snippets cover nothing relevant, "
                "but not the specific question asked."
            ),
            refused=True,
        )

    top = reranker.rerank(question, candidates, top_k=settings.top_k_final)
    prompt = QA_PROMPT.format(
        repo_url=repo_url,
        retrieved_chunks=_format_chunks(top),
        question=question,
    )
    result = llm.complete(prompt, temperature=0.1, max_tokens=1200)
    text = result.text.strip()
    return Answer(
        text=text,
        sources=top,
        citations=parse_citations(text, ref),
        refused=text.lstrip().startswith(REFUSAL_PREFIX),
    )


def stream_answer(
    *,
    ref: RepoRef,
    repo_url: str,
    commit_sha: str,
    question: str,
) -> tuple[Iterator[str], list[dict]]:
    """Stream tokens. Returns (token_iterator, sources_used).

    Sources are returned eagerly so the UI can render the "Sources" expander
    before the answer finishes streaming.
    """
    # Short-circuit project-level questions to the cached overview. We yield
    # it in word-sized chunks so the UI's streaming loop still gets the
    # progressive-render feel even though no LLM call is happening.
    if _is_project_question(question):
        overview = _load_cached_overview(ref, commit_sha)
        if overview:
            def _stream_overview() -> Iterator[str]:
                # Split on whitespace but keep separators so reconstruction
                # is exact. Yielding ~20 words at a time gives a nice typing
                # feel without being so slow it feels artificial.
                tokens = re.split(r"(\s+)", overview)
                buf = ""
                for tok in tokens:
                    buf += tok
                    if len(buf) >= 40:  # flush every ~40 chars
                        yield buf
                        buf = ""
                if buf:
                    yield buf
            return _stream_overview(), []

    candidates = hybrid.hybrid_search(repo_url, commit_sha, question)
    if not candidates:
        def _empty() -> Iterator[str]:
            yield (
                f"{REFUSAL_PREFIX} The retrieved snippets cover nothing relevant, "
                "but not the specific question asked."
            )
        return _empty(), []

    top = reranker.rerank(question, candidates, top_k=settings.top_k_final)
    prompt = QA_PROMPT.format(
        repo_url=repo_url,
        retrieved_chunks=_format_chunks(top),
        question=question,
    )
    return llm.stream(prompt, temperature=0.1, max_tokens=1200), top