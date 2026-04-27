"""GitHub URL parsing.

Accepts the messy URLs users actually paste:
- https://github.com/owner/repo
- https://github.com/owner/repo.git
- https://github.com/owner/repo/tree/main
- https://github.com/owner/repo/tree/main/some/subpath  (subpath ignored, full repo cloned)
- git@github.com:owner/repo.git
- owner/repo  (shorthand)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_HTTPS_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+).*)?/?$"
)
_SSH_RE = re.compile(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$")
_SHORTHAND_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


class InvalidRepoURL(ValueError):
    pass


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    branch: str | None  # None means "default branch"

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"

    @property
    def web_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    def file_url(self, filepath: str, line_start: int, line_end: int) -> str:
        """Build a GitHub link to a file with a line range highlighted.

        Uses the resolved branch if available, otherwise falls back to HEAD,
        which GitHub redirects to the default branch.
        """
        ref = self.branch or "HEAD"
        anchor = f"L{line_start}-L{line_end}" if line_end > line_start else f"L{line_start}"
        return f"https://github.com/{self.owner}/{self.repo}/blob/{ref}/{filepath}#{anchor}"


def parse_github_url(url: str) -> RepoRef:
    url = url.strip().rstrip("/")
    if not url:
        raise InvalidRepoURL("empty URL")

    if m := _HTTPS_RE.match(url):
        owner, repo, branch = m.group(1), m.group(2), m.group(3)
        return RepoRef(owner=owner, repo=repo, branch=branch)
    if m := _SSH_RE.match(url):
        return RepoRef(owner=m.group(1), repo=m.group(2), branch=None)
    if m := _SHORTHAND_RE.match(url):
        return RepoRef(owner=m.group(1), repo=m.group(2), branch=None)

    raise InvalidRepoURL(f"could not parse as a GitHub repo URL: {url!r}")
