---
title: RepoChat
emoji: 🔍
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Chat with any GitHub repo. Cited AI answer.
---

# RepoChat

Paste a GitHub URL → get an instant AI overview → ask anything about the code with cited line-range answers.

See the full README in this repo for architecture, configuration, and eval details.

## How to use

1. Paste a GitHub URL in the sidebar (e.g. `https://github.com/tiangolo/fastapi`)
2. Wait for indexing (typically 30-90 seconds for a medium repo)
3. Read the auto-generated project overview
4. Ask questions. Every answer cites the exact files and line ranges it draws from — click a citation to jump straight to GitHub.

## Required configuration

Add a repository secret in **Settings → Repository secrets**:

- `GROQ_API_KEY` — get a free one at <https://console.groq.com>

Optional secrets:
- `GEMINI_API_KEY` for fallback when Groq is rate-limited
- `GITHUB_TOKEN` to raise GitHub's clone rate limit from 60→5000/hr

## Hardware

Free CPU basic (2 vCPU, 16GB RAM) is sufficient. The embedding and reranker models run on CPU.
