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
    page_title="RepoChat — chat with any GitHub repo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
# Aesthetic: editorial/refined. Serif display + grotesk body + mono for code.
# Warm paper background in light mode; deep ink in dark mode. Ink-blue primary
# with terracotta accent. Deliberately avoids the generic purple-pink SaaS look.
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Design tokens ---------- */
:root {
    --paper:      #f8f5ef;
    --paper-2:    #f1ece1;
    --ink:        #1a1d23;
    --ink-soft:   #4a4f5c;
    --ink-faint:  #8b8f99;
    --rule:       #e3ddcf;
    --primary:    #2a4d8f;
    --primary-2:  #1a3766;
    --accent:     #c46a4a;
    --accent-2:   #a8543a;
    --success:    #5a7a3e;
    --warn:       #b8862b;
    --error:      #a83838;
    --code-bg:    #ede7d8;
    --shadow-1:   0 1px 2px rgba(26,29,35,0.04), 0 2px 8px rgba(26,29,35,0.04);
    --shadow-2:   0 2px 6px rgba(26,29,35,0.06), 0 8px 24px rgba(26,29,35,0.06);
    --radius:     6px;
    --radius-lg:  10px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --paper:      #161821;
        --paper-2:    #1c1f2a;
        --ink:        #e8e4d9;
        --ink-soft:   #b2afa4;
        --ink-faint:  #6e6f78;
        --rule:       #2a2d3a;
        --primary:    #7fa3e0;
        --primary-2:  #9bb8e8;
        --accent:     #e08a6a;
        --accent-2:   #ed9a7a;
        --success:    #88a86b;
        --warn:       #d4a653;
        --error:      #d96a6a;
        --code-bg:    #1f2230;
        --shadow-1:   0 1px 2px rgba(0,0,0,0.2), 0 2px 8px rgba(0,0,0,0.2);
        --shadow-2:   0 2px 6px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.3);
    }
}

/* ---------- App canvas ---------- */
.stApp {
    background: var(--paper);
    background-image:
        radial-gradient(circle at 0% 0%, rgba(42,77,143,0.04) 0%, transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(196,106,74,0.04) 0%, transparent 40%);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main .block-container {
    padding-top: 2.5rem;
    max-width: 920px;
}
section[data-testid="stSidebar"] {
    background: var(--paper-2);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}

/* Hide the default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ---------- Typography ---------- */
.brand-mark {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}
.brand-mark .glyph {
    color: var(--accent);
    font-style: italic;
    font-weight: 400;
}
.brand-mark .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    margin-bottom: 0.35rem;
}
.brand-tagline {
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--ink-soft);
    margin: 0.4rem 0 1.75rem 0;
    font-weight: 400;
    line-height: 1.4;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-faint);
    margin: 0 0 0.6rem 0;
    font-weight: 500;
}
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: var(--ink-faint);
    text-transform: uppercase;
}

/* Streamlit-injected headings, lists, and paragraphs */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.stMarkdown h2 { font-size: 1.4rem; margin-top: 1.5rem; }
.stMarkdown h3 { font-size: 1.15rem; margin-top: 1.2rem; }
.stMarkdown p, .stMarkdown li { color: var(--ink-soft); line-height: 1.65; }
.stMarkdown code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    background: var(--code-bg);
    color: var(--ink);
    padding: 1px 6px;
    border-radius: 4px;
    border: 1px solid var(--rule);
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    background: var(--ink);
    color: var(--paper);
    border: 1px solid var(--ink);
    border-radius: var(--radius);
    padding: 0.55rem 1.1rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.18s ease;
    box-shadow: var(--shadow-1);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--primary);
    border-color: var(--primary);
    transform: translateY(-1px);
    box-shadow: var(--shadow-2);
}
.stButton > button:disabled {
    background: var(--rule);
    color: var(--ink-faint);
    border-color: var(--rule);
    cursor: not-allowed;
    transform: none;
}

/* Ghost-style secondary buttons (recent repos in sidebar) */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: var(--ink-soft);
    border: 1px solid var(--rule);
    text-align: left;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    padding: 0.45rem 0.7rem;
    box-shadow: none;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--paper);
    color: var(--ink);
    border-color: var(--ink-soft);
    transform: none;
    box-shadow: var(--shadow-1);
}
/* The primary "Index repo" button keeps the dark style even in sidebar */
section[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    padding: 0.6rem 1rem;
}
section[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: var(--primary);
    border-color: var(--primary);
}

/* ---------- Inputs ---------- */
.stTextInput > div > div > input, .stTextArea textarea {
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 0.8rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(42,77,143,0.12);
    outline: none;
}
.stChatInput > div {
    background: var(--paper-2);
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-1);
}
.stChatInput textarea {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* ---------- Cards ---------- */
.empty-hero {
    background: var(--paper-2);
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    padding: 2.25rem 2.25rem 1.75rem;
    margin: 0.5rem 0 1.5rem;
    box-shadow: var(--shadow-1);
    position: relative;
    overflow: hidden;
}
.empty-hero::before {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 140px; height: 140px;
    background: radial-gradient(circle, var(--accent) 0%, transparent 65%);
    opacity: 0.08;
    pointer-events: none;
}
.empty-hero h2 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 1.5rem;
    color: var(--ink);
    margin: 0.5rem 0 0.6rem 0;
    letter-spacing: -0.01em;
}
.empty-hero p {
    color: var(--ink-soft);
    line-height: 1.65;
    margin: 0 0 1.25rem 0;
    font-size: 0.95rem;
}
.steps-list {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin: 1.25rem 0 0;
}
.step-tile {
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 0.85rem 0.95rem;
    background: var(--paper);
}
.step-tile .step-num {
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.4rem;
    color: var(--accent);
    line-height: 1;
    display: block;
    margin-bottom: 0.35rem;
}
.step-tile .step-text {
    font-size: 0.82rem;
    color: var(--ink-soft);
    line-height: 1.45;
}
@media (max-width: 700px) {
    .steps-list { grid-template-columns: 1fr; }
}
.try-list {
    margin-top: 1.5rem;
    border-top: 1px solid var(--rule);
    padding-top: 1.25rem;
}
.try-list .eyebrow { margin-bottom: 0.7rem; display: block; }
.try-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px dotted var(--rule);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    gap: 1rem;
}
.try-row:last-child { border-bottom: none; }
.try-row .repo-slug { color: var(--ink); }
.try-row .repo-desc { color: var(--ink-faint); font-size: 0.78rem; }

/* ---------- Active repo strip ---------- */
.active-strip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.75rem;
    background: var(--paper);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--success);
    border-radius: var(--radius);
    margin: 0.75rem 0 0.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.active-strip .pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 0 rgba(90,122,62,0.6);
    animation: pulse 2.2s infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(90,122,62,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(90,122,62,0); }
    100% { box-shadow: 0 0 0 0 rgba(90,122,62,0); }
}
.active-strip .slug { color: var(--ink); font-weight: 500; }
.active-strip .commit { color: var(--ink-faint); }

/* ---------- Chat ---------- */
.message {
    display: flex;
    gap: 0.75rem;
    margin: 1.25rem 0;
    animation: fade-in 0.35s ease-out;
}
.message .avatar {
    width: 28px; height: 28px;
    flex-shrink: 0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.message.user .avatar {
    background: var(--ink);
    color: var(--paper);
}
.message.assistant .avatar {
    background: var(--accent);
    color: var(--paper);
}
.message .body {
    flex: 1;
    min-width: 0;
}
.message .role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    margin-bottom: 0.3rem;
}
.message .text {
    color: var(--ink);
    line-height: 1.7;
    font-size: 0.95rem;
}
.message .text p:first-child { margin-top: 0; }
.message .text p:last-child { margin-bottom: 0; }
.message .text code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    background: var(--code-bg);
    padding: 1px 5px;
    border-radius: 3px;
}
.message .text pre {
    background: var(--code-bg);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 0.75rem;
    overflow-x: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
}
@keyframes fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Streaming cursor */
.cursor {
    display: inline-block;
    width: 0.5em;
    height: 1em;
    background: var(--accent);
    vertical-align: text-bottom;
    margin-left: 1px;
    animation: blink 1s step-end infinite;
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* ---------- Citation pills ---------- */
.citation-pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    background: var(--code-bg);
    border: 1px solid var(--rule);
    border-bottom: 2px solid var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78em;
    margin: 0 2px;
    color: var(--ink) !important;
    text-decoration: none !important;
    transition: all 0.15s ease;
    line-height: 1.4;
}
.citation-pill:hover {
    background: var(--accent);
    color: var(--paper) !important;
    border-color: var(--accent);
    transform: translateY(-1px);
}

/* ---------- Sources expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    background: transparent;
    margin: 0.5rem 0 0.75rem;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-soft);
    padding: 0.6rem 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stExpander"] summary:hover { color: var(--ink); }

/* Code blocks */
.stCode, [data-testid="stCodeBlock"] {
    border-radius: var(--radius) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ---------- Sidebar specifics ---------- */
section[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 1.25rem 0;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    font-size: 0.82rem;
}
.sidebar-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-faint);
    line-height: 1.7;
    padding: 0.5rem 0;
}
.sidebar-meta .key { color: var(--ink-soft); }
.sidebar-meta .val { color: var(--ink); }
.cost-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    background: rgba(90,122,62,0.1);
    color: var(--success);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    border: 1px solid rgba(90,122,62,0.25);
}

/* ---------- Progress bar ---------- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
}
.stProgress > div > div > div {
    background: var(--rule);
}

/* ---------- Alert boxes ---------- */
[data-testid="stAlert"] {
    border-radius: var(--radius);
    border: 1px solid var(--rule);
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
}

/* ---------- Mobile ---------- */
@media (max-width: 700px) {
    .main .block-container { padding-top: 1.5rem; }
    .brand-mark { font-size: 2rem; }
    .empty-hero { padding: 1.5rem; }
    .message { gap: 0.55rem; }
    .message .avatar { width: 24px; height: 24px; font-size: 0.65rem; }
    .try-row { flex-direction: column; align-items: flex-start; gap: 0.2rem; }
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
    st.session_state.setdefault("pending_question", None)  # set by suggested-question chips


_init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 class="brand-mark">'
    '<span>Repo</span>'
    '<span class="glyph">Chat</span>'
    '<span class="dot"></span>'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="brand-tagline">'
    'Read any GitHub repository the way an engineer would — '
    'with grounded answers and exact citations.'
    '</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: repo input + recent + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="section-label">→ Index a repository</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
        key="url_input",
        label_visibility="collapsed",
    )
    index_btn = st.button("Index repo", use_container_width=True, disabled=st.session_state.indexing)

    # Active repo indicator — replaces st.success with a more refined strip
    if st.session_state.active_ref:
        ref: RepoRef = st.session_state.active_ref
        commit_short = st.session_state.active_commit[:8]
        st.markdown(
            f'<div class="active-strip">'
            f'<span class="pulse"></span>'
            f'<span class="slug">{ref.slug}</span>'
            f'<span class="commit">@ {commit_short}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Recent repos
    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">↻ Recently indexed</div>', unsafe_allow_html=True)
    recent = cache.list_recent(limit=5)
    if not recent:
        st.markdown(
            '<div class="sidebar-meta" style="opacity:0.7;">No recent repos yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for entry in recent:
            slug = entry.repo_url.replace("https://github.com/", "")
            if st.button(f"{slug}", key=f"recent_{entry.commit_sha}", use_container_width=True):
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
    st.markdown('<hr/>', unsafe_allow_html=True)
    with st.expander("Settings"):
        st.markdown(
            f'<div class="sidebar-meta">'
            f'<div><span class="key">provider</span> · <span class="val">{settings.llm_provider}</span></div>'
            f'<div><span class="key">model</span> · <span class="val">{settings.llm_model}</span></div>'
            f'<div><span class="key">reranker</span> · <span class="val">{"on" if settings.use_reranker else "off"}</span></div>'
            f'<div><span class="key">top-k</span> · <span class="val">{settings.top_k_final}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.active_ref:
            if st.button("Clear active repo", use_container_width=True):
                st.session_state.active_ref = None
                st.session_state.active_commit = None
                st.session_state.active_repo_url = None
                st.session_state.overview = None
                st.session_state.chat = []
                st.rerun()

    # Footer
    st.markdown('<hr/>', unsafe_allow_html=True)
    cache_mb = cache.total_cache_bytes() / 1e6
    cache_pct = (cache_mb / (settings.max_cache_gb * 1024)) * 100
    st.markdown(
        f'<div class="sidebar-meta">'
        f'<div><span class="key">cache</span> · <span class="val">{cache_mb:.1f} MB</span> '
        f'<span style="color:var(--ink-faint);">({cache_pct:.0f}%)</span></div>'
        f'</div>'
        f'<div style="margin-top:0.6rem;"><span class="cost-badge">● $0.00 / month</span></div>',
        unsafe_allow_html=True,
    )


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

EXAMPLE_REPOS = [
    ("tiangolo/fastapi", "Modern Python web framework"),
    ("psf/requests", "HTTP for humans"),
    ("expressjs/express", "Node.js web framework"),
]

if not st.session_state.active_ref:
    # Empty state — editorial hero with steps and example repos.
    st.markdown(
        '<div class="empty-hero">'
        '<span class="eyebrow">Get started</span>'
        '<h2>Paste a GitHub URL to begin.</h2>'
        '<p>RepoChat clones the repository, parses code with tree-sitter, '
        'builds a hybrid search index, and uses a 70B language model to '
        'answer questions — every claim cited back to the exact lines.</p>'

        '<div class="steps-list">'
        '<div class="step-tile">'
        '<span class="step-num">i.</span>'
        '<span class="step-text">Paste any public GitHub URL into the sidebar.</span>'
        '</div>'
        '<div class="step-tile">'
        '<span class="step-num">ii.</span>'
        '<span class="step-text">Wait ~60s while the codebase is parsed and indexed.</span>'
        '</div>'
        '<div class="step-tile">'
        '<span class="step-num">iii.</span>'
        '<span class="step-text">Ask questions — answers cite the source.</span>'
        '</div>'
        '</div>'

        '<div class="try-list">'
        '<span class="eyebrow">Try one of these</span>'
        + "".join(
            f'<div class="try-row">'
            f'<span class="repo-slug">github.com/{slug}</span>'
            f'<span class="repo-desc">{desc}</span>'
            f'</div>'
            for slug, desc in EXAMPLE_REPOS
        )
        + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Overview section
if st.session_state.overview:
    with st.expander("◐  Project overview", expanded=True):
        st.markdown(st.session_state.overview)

# Chat section
st.markdown(
    '<div class="section-label" style="margin-top:1.5rem;">→ Conversation</div>',
    unsafe_allow_html=True,
)


def _render_user(text: str) -> str:
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="message user">'
        f'<div class="avatar">You</div>'
        f'<div class="body"><div class="role">You</div>'
        f'<div class="text">{safe}</div></div>'
        f'</div>'
    )


def _render_assistant(html_text: str, *, streaming: bool = False) -> str:
    cursor = '<span class="cursor"></span>' if streaming else ''
    return (
        f'<div class="message assistant">'
        f'<div class="avatar">RC</div>'
        f'<div class="body"><div class="role">RepoChat</div>'
        f'<div class="text">{html_text}{cursor}</div></div>'
        f'</div>'
    )


# Render chat history
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(_render_user(msg["text"]), unsafe_allow_html=True)
    else:
        rendered = qa.render_citations_html(msg["text"], st.session_state.active_ref)
        st.markdown(_render_assistant(rendered), unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"Sources · {len(msg['sources'])}", expanded=False):
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


# Suggested questions — only when there's no chat yet, to give users a starting point
SUGGESTED_QUESTIONS = [
    "What does this project do?",
    "Walk me through the architecture.",
    "Where is the main entry point?",
    "What are the core modules?",
]

if not st.session_state.chat:
    st.markdown(
        '<div class="section-label" style="margin-top:0.75rem;">⌁ Try asking</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# Chat input — accepts typed questions OR a pending suggested question
typed_question = st.chat_input("Ask anything about this codebase…")
question = typed_question or st.session_state.pending_question
if st.session_state.pending_question:
    st.session_state.pending_question = None  # consume

if question:
    st.session_state.chat.append({"role": "user", "text": question})
    cache.touch(st.session_state.active_repo_url, st.session_state.active_commit)

    # Render the user message immediately
    st.markdown(_render_user(question), unsafe_allow_html=True)

    # Rate-limit countdown if needed
    wait = time_until_next_slot()
    if wait > 0.5:
        with st.spinner(f"Rate-limit cooldown — {wait:.0f}s"):
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
            placeholder.markdown(
                _render_assistant(
                    qa.render_citations_html(full_text, st.session_state.active_ref),
                    streaming=True,
                ),
                unsafe_allow_html=True,
            )
        # Final render without cursor
        placeholder.markdown(
            _render_assistant(
                qa.render_citations_html(full_text, st.session_state.active_ref),
                streaming=False,
            ),
            unsafe_allow_html=True,
        )
    except RateLimitError:
        full_text = "_Rate limit hit. Please wait a minute and try again._"
        placeholder.markdown(_render_assistant(full_text), unsafe_allow_html=True)
    except LLMError as e:
        full_text = f"_LLM error: {e}_"
        placeholder.markdown(_render_assistant(full_text), unsafe_allow_html=True)
    except Exception as e:
        full_text = f"_Unexpected error: {e}_"
        placeholder.markdown(_render_assistant(full_text), unsafe_allow_html=True)
        log.exception("stream_answer failed")

    st.session_state.chat.append({
        "role": "assistant",
        "text": full_text,
        "sources": sources,
    })

    if sources:
        with st.expander(f"Sources · {len(sources)}", expanded=False):
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