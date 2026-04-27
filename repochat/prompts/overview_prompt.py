"""Prompt template for Capability 1: project overview."""

OVERVIEW_PROMPT = """You are a senior engineer summarizing a GitHub repository for someone seeing it for the first time.

Below are: README contents, file tree (top 3 levels), tech stack manifests, and 6 sampled source files.

Produce a markdown overview with EXACTLY these 4 sections, in this order:

## What it does
2–3 sentences. Plain English. No marketing language. Focus on the actual function.

## Tech stack
Bulleted list grouped by: Language, Framework, Database, Infrastructure, Key libraries.

## Architecture
2 short paragraphs explaining how the codebase is organized. Reference real directory names. Explain the relationship between major directories.

## Main modules
3–5 key files or modules. Format: `path/to/file` — one-line description of its responsibility.

Rules:
- Be concrete. Name real files, real components.
- Don't invent features that aren't in the provided content.
- If README is missing or sparse, infer from code only and note this.
- Total length under 400 words.

---
README:
{readme_content}

---
File tree (top 3 levels):
{file_tree}

---
Tech stack manifests:
{manifests}

---
Sampled source files:
{sampled_files}
"""
