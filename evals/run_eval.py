"""Eval runner.

Runs the question suite against the live pipeline. For each question we check:

- For `location` questions: did the answer (or its citations) reference a file
  whose path contains any expected substring?
- For `project` and `mechanism` questions: did the answer mention all expected
  topics (case-insensitive substring match)? Topics are lowercased; we don't
  do stemming because the topics are deliberately picked to be unambiguous.
- For `refusal` questions: did the answer start with the refusal prefix?

Results land in evals/results.md as a markdown report with per-question
pass/fail and aggregate pass rate.

Usage:
    python evals/run_eval.py [--limit N] [--repos repo1,repo2]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `repochat` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repochat.config import configure_logging, settings  # noqa: E402
from repochat.generation import qa  # noqa: E402
from repochat.generation.qa import parse_citations  # noqa: E402
from repochat.ingestion import pipeline  # noqa: E402
from repochat.prompts.qa_prompt import REFUSAL_PREFIX  # noqa: E402
from repochat.utils.github import parse_github_url  # noqa: E402

configure_logging()
log = logging.getLogger("eval")


@dataclass
class QuestionResult:
    repo: str
    question: str
    qtype: str
    answer: str
    cited_files: list[str]
    passed: bool
    reason: str
    seconds: float


@dataclass
class RepoResults:
    repo: str
    indexed_seconds: float
    chunks: int
    questions: list[QuestionResult] = field(default_factory=list)


def _check_location(answer: str, citations: list[str], expected_files: list[str]) -> tuple[bool, str]:
    haystack_paths = [c.lower() for c in citations]
    answer_lower = answer.lower()
    for expected in expected_files:
        e = expected.lower()
        if any(e in p for p in haystack_paths):
            return True, f"citation matched {expected}"
        if e in answer_lower:
            return True, f"answer mentioned {expected}"
    return False, f"no citation/mention matched any of {expected_files}"


def _check_topics(answer: str, expected_topics: list[str]) -> tuple[bool, str]:
    answer_lower = answer.lower()
    missing = [t for t in expected_topics if t.lower() not in answer_lower]
    if missing:
        return False, f"missing topics: {missing}"
    return True, "all topics present"


def _check_refusal(answer: str) -> tuple[bool, str]:
    if answer.lstrip().startswith(REFUSAL_PREFIX):
        return True, "refused as expected"
    return False, "did not refuse"


def run_one_question(
    repo_url: str,
    commit_sha: str,
    ref,
    question_obj: dict,
) -> QuestionResult:
    q = question_obj["q"]
    qtype = question_obj["type"]
    started = time.time()

    try:
        answer = qa.answer_question(
            ref=ref,
            repo_url=repo_url,
            commit_sha=commit_sha,
            question=q,
        )
    except Exception as e:
        return QuestionResult(
            repo=repo_url, question=q, qtype=qtype,
            answer=f"<error: {e}>", cited_files=[],
            passed=False, reason=f"exception: {e}",
            seconds=time.time() - started,
        )

    cited_files = sorted({c.filepath for c in answer.citations})

    if qtype == "location":
        passed, reason = _check_location(answer.text, cited_files, question_obj["expected_files"])
    elif qtype in {"project", "mechanism"}:
        passed, reason = _check_topics(answer.text, question_obj["expected_topics"])
    elif qtype == "refusal":
        passed, reason = _check_refusal(answer.text)
    else:
        passed, reason = False, f"unknown qtype: {qtype}"

    return QuestionResult(
        repo=repo_url, question=q, qtype=qtype,
        answer=answer.text, cited_files=cited_files,
        passed=passed, reason=reason,
        seconds=time.time() - started,
    )


def write_report(repo_results: list[RepoResults], out_path: Path) -> None:
    lines: list[str] = ["# RepoChat Eval Results\n"]

    total_q = sum(len(r.questions) for r in repo_results)
    total_pass = sum(1 for r in repo_results for q in r.questions if q.passed)
    pass_rate = (total_pass / total_q * 100) if total_q else 0.0

    lines.append(f"**Overall: {total_pass}/{total_q} passed ({pass_rate:.1f}%)**\n")
    lines.append(f"_Provider: {settings.llm_provider} ({settings.llm_model}), reranker: {settings.use_reranker}_\n")

    # Per-question-type breakdown
    type_stats: dict[str, list[bool]] = {}
    for r in repo_results:
        for q in r.questions:
            type_stats.setdefault(q.qtype, []).append(q.passed)
    lines.append("## By question type\n")
    lines.append("| Type | Passed | Total | Rate |")
    lines.append("|---|---|---|---|")
    for t, results in sorted(type_stats.items()):
        p, n = sum(results), len(results)
        lines.append(f"| {t} | {p} | {n} | {p / n * 100:.0f}% |")
    lines.append("")

    for r in repo_results:
        passed = sum(1 for q in r.questions if q.passed)
        lines.append(f"## {r.repo}\n")
        lines.append(f"- Indexed in {r.indexed_seconds:.1f}s ({r.chunks} chunks)")
        lines.append(f"- Passed {passed}/{len(r.questions)}\n")
        lines.append("| # | Type | Q | Result | Reason |")
        lines.append("|---|---|---|---|---|")
        for i, q in enumerate(r.questions, 1):
            mark = "✅" if q.passed else "❌"
            esc_q = q.question.replace("|", "\\|")
            esc_reason = q.reason.replace("|", "\\|")
            lines.append(f"| {i} | {q.qtype} | {esc_q} | {mark} | {esc_reason} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Limit questions per repo (smoke test)")
    p.add_argument("--repos", type=str, default=None,
                   help="Comma-separated list of repo URLs to filter to")
    p.add_argument("--questions", type=str,
                   default=str(Path(__file__).parent / "test_questions.json"))
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parent / "results.md"))
    args = p.parse_args()

    suite = json.loads(Path(args.questions).read_text())
    if args.repos:
        wanted = {x.strip() for x in args.repos.split(",")}
        suite = [r for r in suite if r["repo"] in wanted]

    all_results: list[RepoResults] = []

    for entry in suite:
        repo_url = entry["repo"]
        log.info("=== %s ===", repo_url)
        ref = parse_github_url(repo_url)
        try:
            ingest = pipeline.index_repo(ref)
        except Exception as e:
            log.error("Failed to index %s: %s", repo_url, e)
            all_results.append(RepoResults(repo=repo_url, indexed_seconds=0, chunks=0))
            continue

        repo_url_canonical = ingest.ref.web_url
        repo_results = RepoResults(
            repo=repo_url_canonical,
            indexed_seconds=ingest.seconds,
            chunks=ingest.chunk_count,
        )

        questions = entry["questions"]
        if args.limit:
            questions = questions[: args.limit]

        for q in questions:
            log.info("Q: %s", q["q"])
            qr = run_one_question(repo_url_canonical, ingest.commit_sha, ingest.ref, q)
            log.info("  -> %s (%s)", "PASS" if qr.passed else "FAIL", qr.reason)
            repo_results.questions.append(qr)

        all_results.append(repo_results)

    out_path = Path(args.out)
    write_report(all_results, out_path)
    log.info("Wrote %s", out_path)

    total_q = sum(len(r.questions) for r in all_results)
    total_pass = sum(1 for r in all_results for q in r.questions if q.passed)
    print(f"\nDone: {total_pass}/{total_q} passed -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
