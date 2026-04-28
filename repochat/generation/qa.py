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


# Conversational pleasantries — "thanks", "great", "ok cool" — aren't questions
# about the code at all. Without this short-circuit, retrieval finds nothing
# meaningful and the model refuses with the unhelpful "couldn't find this in
# the indexed code" message. We respond with a brief friendly reply instead.
_CHITCHAT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        # Thanks variants
        r"^\s*(thanks|thank you|thx|ty|cheers|much appreciated|appreciate(d| it))\s*[!.?]*\s*$",
        r"^\s*(thanks|thank you|thx|ty)[,!]?\s+(a lot|so much|much|tons|loads|mate|buddy|man|friend)\s*[!.?]*\s*$",
        # Acknowledgments / approval
        r"^\s*(great|awesome|nice|cool|perfect|excellent|amazing|wonderful|fantastic|lovely|brilliant|sweet)\s*[!.?]*\s*$",
        r"^\s*(good|ok|okay|alright|got it|gotcha|understood|makes sense|i see|got ya|ic)\s*[!.?]*\s*$",
        r"^\s*(very (good|nice|cool|helpful)|works (great|well|perfectly)|that('s| is) (great|helpful|useful|perfect))\s*[!.?]*\s*$",
        r"^\s*(makes sense|i understand|that helps|that helped|helpful)\s*[!.?]*\s*$",
        # Greetings
        r"^\s*(hi|hello|hey|yo|hiya|sup|howdy|greetings)\s*[!.?]*\s*$",
        r"^\s*(good (morning|afternoon|evening|day))\s*[!.?]*\s*$",
        # Goodbyes
        r"^\s*(bye|goodbye|see (ya|you)( later)?|cya|later|farewell|take care)\s*[!.?]*\s*$",
        # Affirmations
        r"^\s*(yes|yep|yeah|yup|sure|right|correct|exactly|true|of course)\s*[!.?]*\s*$",
        r"^\s*(no|nope|nah|not really|not quite)\s*[!.?]*\s*$",
        # Compliments to the assistant
        r"^\s*(you'?re (great|awesome|helpful|amazing|the best|cool))\s*[!.?]*\s*$",
        r"^\s*(good (job|work|bot|answer))\s*[!.?]*\s*$",
        r"^\s*(well done|nicely done|good one)\s*[!.?]*\s*$",
    )
]


def _is_chitchat(q: str) -> bool:
    """True if the message is a conversational pleasantry, not a code question."""
    # Cap length: real questions get long; chitchat is almost always under 40 chars.
    # This guards against a question that happens to start with a chitchat word.
    if len(q.strip()) > 40:
        return False
    return any(p.match(q.strip()) for p in _CHITCHAT_PATTERNS)


def _chitchat_reply(q: str) -> str:
    """Pick a short, contextually appropriate reply for a chitchat message.

    Uses simple keyword matching rather than the LLM to keep latency near zero
    and avoid burning rate-limit slots on social niceties.
    """
    q_lower = q.strip().lower()

    # Thanks family
    if any(w in q_lower for w in ("thank", "thx", "ty", "cheers", "appreciate")):
        return (
            "You're welcome! Happy to help. Ask me anything else about this codebase — "
            "I can explain how things work, find specific functions, or trace through logic."
        )
    # Goodbye family
    if any(w in q_lower for w in ("bye", "later", "farewell", "see ya", "see you", "cya", "take care")):
        return "See you later! The indexed repo will still be here when you come back."
    # Greetings
    if any(w in q_lower for w in ("hi", "hello", "hey", "yo", "hiya", "sup", "howdy", "greetings", "morning", "afternoon", "evening")):
        return (
            "Hi! I've indexed this repository and I'm ready to answer questions about it. "
            "Try asking how something works, where a specific function lives, or what a particular module does."
        )
    # Compliments
    if any(w in q_lower for w in ("you're", "youre", "good job", "good work", "good bot", "good answer", "well done", "nicely done")):
        return "Thanks! Got another question about the code?"
    # Approval / acknowledgment (great, nice, cool, ok, got it, etc.)
    return (
        "Glad that helped. What else would you like to know about the codebase? "
        "You can ask about specific functions, classes, files, or how features are implemented."
    )


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
    # Short-circuit conversational pleasantries — "thanks", "great", "ok cool".
    # These aren't questions about the code; retrieval would fail and the model
    # would refuse with a confusing "couldn't find this in the indexed code"
    # message. Return a brief friendly reply instead, without burning an LLM call.
    if _is_chitchat(question):
        return Answer(text=_chitchat_reply(question), sources=[], citations=[], refused=False)

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
    # Short-circuit conversational pleasantries. These are short by definition,
    # so we just yield the whole reply as a single chunk — the UI's streaming
    # loop handles single-chunk yields fine, and a 1-token "stream" still
    # animates correctly.
    if _is_chitchat(question):
        reply = _chitchat_reply(question)
        def _stream_chitchat() -> Iterator[str]:
            # Word-by-word for a natural feel, even though the response is short.
            for word in re.split(r"(\s+)", reply):
                if word:
                    yield word
        return _stream_chitchat(), []

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