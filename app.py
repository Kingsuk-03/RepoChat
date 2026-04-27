"""RepoChat — Streamlit entry point.

Single-page app. Sidebar holds repo input, recent repos, and settings.
Main area has two sections: AI overview (collapsible) and chat (streaming).

State management: we store the active repo's RepoRef + commit_sha in
st.session_state so questions know which collection to query. Chat history
is also session-scoped — refreshing the page clears it.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import streamlit as st

from repochat.config import configure_logging, settings
from repochat.generation import overview as overview_gen
from repochat.generation import qa
from repochat.generation.llm import LLMError, RateLimitError, time_until_next_slot
from repochat.ingestion import pipeline
from repochat.ingestion.filter import filter_repo
from repochat.utils import cache
from repochat.utils.github import InvalidRepoURL, RepoRef, parse_github_url

configure_logging()
log = logging.getLogger("repochat.app")

st.set_page_config(
    page_title="RepoChat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
}
.gradient-title {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}
.tagline { color: #4c51bf; font-size: 1rem; margin-bottom: 1.5rem; opacity: 0.85; }
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}
.repo-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.citation-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.13), rgba(118, 75, 162, 0.13));
    border: 1px solid rgba(102, 126, 234, 0.4);
    font-size: 0.85rem;
    margin: 0 4px;
    color: #4c51bf !important;
    text-decoration: none !important;
    transition: all 0.15s ease;
}
.citation-pill:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.25), rgba(118, 75, 162, 0.25));
    transform: translateY(-1px);
}
.user-message {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08), rgba(118, 75, 162, 0.08));
    border-radius: 12px;
    padding: 12px 16px;
    margin-left: 20%;
    margin-bottom: 12px;
}
.assistant-message {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
    margin-right: 20%;
    margin-bottom: 12px;
}
@media (prefers-color-scheme: dark) {
    .stApp { background: linear-gradient(135deg, #1a1d2e 0%, #2d1b4e 100%); }
    .repo-card { background: #252842; color: #e2e8f0; }
    .assistant-message { background: #252842; color: #e2e8f0; border-color: #3d4263; }
    .citation-pill {
        color: #a3bffa !important;
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("active_ref", None)        # RepoRef
    st.session_state.setdefault("active_commit", None)     # str
    st.session_state.setdefault("active_repo_url", None)   # str
    st.session_state.setdefault("overview", None)          # str (markdown)
    st.session_state.setdefault("chat", [])                # list[dict(role,text,citations,sources)]
    st.session_state.setdefault("indexing", False)


_init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="gradient-title">RepoChat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Paste a GitHub URL → instant AI overview → ask anything with cited answers.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: repo input + recent + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Index a repository")
    url_input = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/tiangolo/fastapi",
        key="url_input",
        label_visibility="collapsed",
    )
    index_btn = st.button("Index repo", use_container_width=True, disabled=st.session_state.indexing)

    # Active repo indicator
    if st.session_state.active_ref:
        ref: RepoRef = st.session_state.active_ref
        st.success(f"Active: {ref.slug} @ {st.session_state.active_commit[:8]}")

    # Recent repos
    st.divider()
    st.subheader("Recent")
    recent = cache.list_recent(limit=5)
    if not recent:
        st.caption("No recently indexed repos.")
    else:
        for entry in recent:
            slug = entry.repo_url.replace("https://github.com/", "")
            if st.button(f"📁 {slug}", key=f"recent_{entry.commit_sha}", use_container_width=True):
                ref = parse_github_url(entry.repo_url)
                ref = RepoRef(owner=ref.owner, repo=ref.repo, branch=entry.branch)
                st.session_state.active_ref = ref
                st.session_state.active_commit = entry.commit_sha
                st.session_state.active_repo_url = entry.repo_url
                st.session_state.chat = []
                # Re-load any cached overview
                overview_path = settings.overviews_dir / f"{ref.owner}__{ref.repo}__{entry.commit_sha[:12]}.md"
                st.session_state.overview = overview_path.read_text() if overview_path.exists() else None
                cache.touch(entry.repo_url, entry.commit_sha)
                st.rerun()

    # Settings
    st.divider()
    with st.expander("Settings"):
        st.write(f"**Provider:** {settings.llm_provider}")
        st.write(f"**Model:** {settings.llm_model}")
        st.write(f"**Reranker:** {'on' if settings.use_reranker else 'off'}")
        st.write(f"**Top-k final:** {settings.top_k_final}")
        if st.session_state.active_ref:
            if st.button("Clear active repo", use_container_width=True):
                st.session_state.active_ref = None
                st.session_state.active_commit = None
                st.session_state.active_repo_url = None
                st.session_state.overview = None
                st.session_state.chat = []
                st.rerun()

    # Footer
    st.divider()
    st.caption(f"Cache: {cache.total_cache_bytes() / 1e6:.1f} MB / {settings.max_cache_gb} GB")
    st.caption("**Cost: $0.00** — all free-tier services")


# ---------------------------------------------------------------------------
# Indexing handler
# ---------------------------------------------------------------------------

def _handle_index(url: str) -> None:
    try:
        ref = parse_github_url(url)
    except InvalidRepoURL as e:
        st.error(f"Invalid GitHub URL: {e}")
        return

    st.session_state.indexing = True
    progress_bar = st.progress(0.0)
    status = st.empty()

    # Phase weights for a smoother progress bar.
    PHASE_WEIGHTS = {
        "clone": 0.10, "filter": 0.10, "chunk": 0.20,
        "embed": 0.50, "bm25": 0.05, "done": 0.05,
    }
    phase_starts = {}
    cum = 0.0
    for p, w in PHASE_WEIGHTS.items():
        phase_starts[p] = cum
        cum += w

    def _progress(phase, current, total, msg):
        base = phase_starts.get(phase, 0.0)
        weight = PHASE_WEIGHTS.get(phase, 0.0)
        if current is not None and total:
            frac = base + weight * (current / total)
        else:
            frac = base + weight * 0.5
        progress_bar.progress(min(1.0, frac))
        status.info(msg)

    try:
        result = pipeline.index_repo(ref, progress=_progress)
    except Exception as e:
        st.session_state.indexing = False
        progress_bar.empty()
        status.empty()
        st.error(f"Indexing failed: {e}")
        log.exception("index_repo failed")
        return

    # Generate overview (separate try so an LLM error doesn't kill indexing)
    status.info("Generating project overview...")
    try:
        # Need the file list again for overview sampling. Cheap re-walk.
        files, _ = filter_repo(
            result.local_path,
            max_file_size_kb=settings.max_file_size_kb,
            max_files=settings.max_files,
        )
        overview_md = overview_gen.generate_overview(
            ref=result.ref,
            commit_sha=result.commit_sha,
            repo_root=result.local_path,
            files=files,
        )
    except (LLMError, RateLimitError) as e:
        overview_md = f"_Overview generation failed: {e}_"
    except Exception as e:
        overview_md = f"_Overview generation failed: {e}_"
        log.exception("overview generation failed")

    # Commit to session state
    st.session_state.active_ref = result.ref
    st.session_state.active_commit = result.commit_sha
    st.session_state.active_repo_url = result.ref.web_url
    st.session_state.overview = overview_md
    st.session_state.chat = []
    st.session_state.indexing = False
    progress_bar.progress(1.0)
    status.success(
        f"Indexed {result.ref.slug}: {result.chunk_count} chunks from "
        f"{result.files_kept}/{result.files_seen} files in {result.seconds:.1f}s"
        + (" (truncated)" if result.truncated else "")
    )
    time.sleep(1.0)
    st.rerun()


if index_btn and url_input:
    _handle_index(url_input.strip())


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not st.session_state.active_ref:
    st.markdown(
        '<div class="repo-card">'
        '<h3>👋 Paste a GitHub URL in the sidebar to get started</h3>'
        '<p>RepoChat will:</p>'
        '<ol>'
        '<li>Clone the repo and chunk it with tree-sitter</li>'
        '<li>Generate an AI overview of what it does</li>'
        '<li>Answer questions about the code with cited line ranges</li>'
        '</ol>'
        '<p><b>Try:</b> <code>https://github.com/tiangolo/fastapi</code></p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Overview section
if st.session_state.overview:
    with st.expander("📋 Project overview", expanded=True):
        st.markdown(st.session_state.overview)

# Chat section
st.subheader("💬 Ask anything about this codebase")

# Render chat history
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["text"]}</div>', unsafe_allow_html=True)
    else:
        rendered = qa.render_citations_html(msg["text"], st.session_state.active_ref)
        st.markdown(f'<div class="assistant-message">{rendered}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    meta = s["metadata"]
                    url = st.session_state.active_ref.file_url(
                        meta["filepath"], meta["line_start"], meta["line_end"]
                    )
                    st.markdown(
                        f"**[{meta['filepath']}:{meta['line_start']}-{meta['line_end']}]({url})** "
                        f"_({meta.get('language', 'text')})_"
                    )
                    st.code(s["content"][:1500], language=meta.get("language", "text"))


# Chat input
question = st.chat_input("e.g., How does authentication work?")
if question:
    st.session_state.chat.append({"role": "user", "text": question})
    cache.touch(st.session_state.active_repo_url, st.session_state.active_commit)

    # Render the user message immediately
    st.markdown(f'<div class="user-message">{question}</div>', unsafe_allow_html=True)

    # Rate-limit countdown if needed
    wait = time_until_next_slot()
    if wait > 0.5:
        with st.spinner(f"Waiting {wait:.0f}s for rate limit..."):
            pass

    placeholder = st.empty()
    full_text = ""
    sources: list[dict] = []
    try:
        token_iter, sources = qa.stream_answer(
            ref=st.session_state.active_ref,
            repo_url=st.session_state.active_repo_url,
            commit_sha=st.session_state.active_commit,
            question=question,
        )
        for token in token_iter:
            full_text += token
            # Re-render whole bubble each token. For longer answers Streamlit's
            # diff makes this cheap enough.
            placeholder.markdown(
                f'<div class="assistant-message">{qa.render_citations_html(full_text, st.session_state.active_ref)}▌</div>',
                unsafe_allow_html=True,
            )
        # Final render without cursor
        placeholder.markdown(
            f'<div class="assistant-message">{qa.render_citations_html(full_text, st.session_state.active_ref)}</div>',
            unsafe_allow_html=True,
        )
    except RateLimitError:
        full_text = "_Rate limit hit. Please wait a minute and try again._"
        placeholder.markdown(f'<div class="assistant-message">{full_text}</div>', unsafe_allow_html=True)
    except LLMError as e:
        full_text = f"_LLM error: {e}_"
        placeholder.markdown(f'<div class="assistant-message">{full_text}</div>', unsafe_allow_html=True)
    except Exception as e:
        full_text = f"_Unexpected error: {e}_"
        placeholder.markdown(f'<div class="assistant-message">{full_text}</div>', unsafe_allow_html=True)
        log.exception("stream_answer failed")

    st.session_state.chat.append({
        "role": "assistant",
        "text": full_text,
        "sources": sources,
    })

    if sources:
        with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
            for s in sources:
                meta = s["metadata"]
                url = st.session_state.active_ref.file_url(
                    meta["filepath"], meta["line_start"], meta["line_end"]
                )
                st.markdown(
                    f"**[{meta['filepath']}:{meta['line_start']}-{meta['line_end']}]({url})** "
                    f"_({meta.get('language', 'text')})_"
                )
                st.code(s["content"][:1500], language=meta.get("language", "text"))
