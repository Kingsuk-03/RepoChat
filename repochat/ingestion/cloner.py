"""Repo cloner.

Clones to a UUID-named directory under settings.repos_dir so the same repo
URL can be re-cloned without name collisions, and we can wire LRU eviction
later by inspecting directory mtimes.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, Repo

from repochat.config import settings
from repochat.utils.github import RepoRef

log = logging.getLogger(__name__)


class CloneError(RuntimeError):
    pass


@dataclass
class ClonedRepo:
    ref: RepoRef
    local_path: Path
    commit_sha: str
    branch: str  # the branch we actually checked out (resolved if input was None)


def _auth_url(ref: RepoRef) -> str:
    """Inject GITHUB_TOKEN into the clone URL if configured.

    GitHub accepts personal access tokens via basic auth in the URL. Anonymous
    clones are limited to 60 requests/hour per IP — fine for demos, painful in
    eval runs.
    """
    if settings.github_token:
        return f"https://{settings.github_token}@github.com/{ref.owner}/{ref.repo}.git"
    return ref.clone_url


def clone_repo(ref: RepoRef, *, depth: int = 1) -> ClonedRepo:
    """Shallow-clone a public GitHub repo. Raises CloneError on failure.

    Shallow clones (depth=1) save ~70% bandwidth and disk on most repos. We
    don't need history — we only ever read working-tree state.
    """
    settings.ensure_dirs()
    target = settings.repos_dir / f"{ref.owner}__{ref.repo}__{uuid.uuid4().hex[:8]}"

    try:
        log.info("Cloning %s into %s (depth=%d)", ref.slug, target, depth)
        kwargs: dict = {"depth": depth}
        if ref.branch:
            kwargs["branch"] = ref.branch
        repo = Repo.clone_from(_auth_url(ref), target, **kwargs)
        commit_sha = repo.head.commit.hexsha
        # Resolve the actual branch name we landed on. For shallow clones with
        # no explicit branch, this is the default branch GitHub redirected us to.
        try:
            branch = repo.active_branch.name
        except TypeError:
            # Detached HEAD on shallow clone; fall back to ref.branch or 'HEAD'.
            branch = ref.branch or "HEAD"
        log.info("Cloned %s @ %s (branch=%s)", ref.slug, commit_sha[:8], branch)
        return ClonedRepo(
            ref=RepoRef(owner=ref.owner, repo=ref.repo, branch=branch),
            local_path=target,
            commit_sha=commit_sha,
            branch=branch,
        )
    except GitCommandError as e:
        # Clean up any partial clone before re-raising.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        msg = str(e).lower()
        if "not found" in msg or "could not read from remote" in msg:
            raise CloneError(
                f"Repository {ref.slug} not found or is private. "
                f"If it's private, set GITHUB_TOKEN in your environment."
            ) from e
        if "rate limit" in msg:
            raise CloneError(
                "GitHub rate limit hit. Set GITHUB_TOKEN to raise the limit "
                "from 60 to 5000 requests per hour."
            ) from e
        raise CloneError(f"Failed to clone {ref.slug}: {e}") from e
    except Exception as e:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise CloneError(f"Unexpected error cloning {ref.slug}: {e}") from e


def remove_clone(local_path: Path) -> None:
    """Best-effort removal. Used by LRU eviction and on errors."""
    if local_path.exists():
        shutil.rmtree(local_path, ignore_errors=True)
