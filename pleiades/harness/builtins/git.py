"""
git.py — Git as a first-class tool, not a shell one-liner.

Every coding task touches git. Having it as structured tools (not ad-hoc
`run_shell("git ...")` calls) means:
  * consistent output format the model can parse reliably
  * read-only ops are safe=True (no approval gate)
  * write ops (commit, checkout) are gated — the model can't silently rewrite history
  * the workspace root is automatically the git root of the cwd

All tools run with the git repo resolved from `cwd` (defaults to the process's
working directory, so `Config.workspace_root` applies when set).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..tools import tool

_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _git(args: list[str], cwd: str | None = None,
         timeout: int = 30) -> tuple[str, int]:
    """Run a git command; return (output, returncode)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd or None, env=_GIT_ENV,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out.strip(), proc.returncode
    except FileNotFoundError:
        return "Error: git not found on PATH.", 1
    except subprocess.TimeoutExpired:
        return f"Error: git timed out after {timeout}s.", 1


def _repo_root(path: str | None = None) -> str | None:
    """Return the git root for the given path, or None if not in a repo."""
    out, rc = _git(["rev-parse", "--show-toplevel"], cwd=path)
    return out if rc == 0 else None


# ---------------------------------------------------------------------------
# READ-ONLY tools (safe=True)
# ---------------------------------------------------------------------------

@tool(safe=True, tags=("git", "read"))
def git_status(cwd: str = ".") -> str:
    """Show the working tree status (modified, staged, untracked files).

    cwd: directory inside the repo (default current directory).
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."
    out, _ = _git(["status", "--short", "--branch"], cwd=root)
    return out or "(clean — nothing to report)"


@tool(safe=True, tags=("git", "read"))
def git_diff(ref: str = "", staged: bool = False, path: str = ".",
             stat_only: bool = False) -> str:
    """Show changes in the working tree or between commits.

    ref: compare against this ref (branch, commit, tag). Empty = working-tree diff.
    staged: if True, show staged (--cached) changes instead of unstaged.
    path: limit diff to this file or directory.
    stat_only: if True, show only the diffstat (file names + line counts), not full patch.
    """
    root = _repo_root(path)
    if root is None:
        return f"Error: {path!r} is not inside a git repository."

    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    if ref:
        args.append(ref)
    args += ["--", path]

    out, rc = _git(args, cwd=root, timeout=60)
    if rc != 0:
        return f"git diff failed:\n{out}"
    return out or "(no diff)"


@tool(safe=True, tags=("git", "read"))
def git_log(n: int = 12, file: str = "", oneline: bool = True,
            cwd: str = ".") -> str:
    """Show recent commit history.

    n: number of commits to show (default 12).
    file: limit log to commits that touched this file (optional).
    oneline: compact one-line format (default True); False for full metadata.
    cwd: directory inside the repo.
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."

    fmt = "--oneline" if oneline else "--format=%H %an %ad %s --date=short"
    args = ["log", f"-{n}", fmt]
    if file:
        args += ["--", file]

    out, rc = _git(args, cwd=root)
    return out or "(no commits)" if rc == 0 else f"git log failed:\n{out}"


@tool(safe=True, tags=("git", "read"))
def git_blame(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Show who last changed each line of a file (git blame).

    path: file to blame.
    start_line: first line to show (1-indexed, default 1).
    end_line: last line to show (0 = to end of file).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: not a file: {path}"
    root = _repo_root(str(p.parent))
    if root is None:
        return f"Error: {path!r} is not inside a git repository."

    args = ["blame", "--porcelain"]
    if end_line > 0:
        args += [f"-L{start_line},{end_line}"]
    elif start_line > 1:
        args += [f"-L{start_line},"]
    args += ["--", str(p)]

    raw, rc = _git(args, cwd=root, timeout=60)
    if rc != 0:
        return f"git blame failed:\n{raw}"

    # Parse porcelain into readable output
    lines_out: list[str] = []
    commit = author = summary = ""
    for line in raw.splitlines():
        if not line:
            continue
        if line.startswith("\t"):
            code = line[1:]
            lines_out.append(f"{commit[:8]} {author:<20} {summary[:40]:<40} │ {code}")
        elif line.startswith("author "):
            author = line[7:]
        elif line.startswith("summary "):
            summary = line[8:]
        elif len(line) == 40 + 1 + 6 + 1 + 5 and line[:40].replace("0","").isalnum():
            parts = line.split()
            if len(parts) >= 1:
                commit = parts[0]

    return "\n".join(lines_out) or raw


@tool(safe=True, tags=("git", "read"))
def git_show(ref: str, path: str = ".") -> str:
    """Show the content or metadata of a specific commit.

    ref: commit hash, tag, or ref (e.g. "HEAD", "main~3", "abc1234").
    path: directory or file inside the repo.
    """
    root = _repo_root(path)
    if root is None:
        return f"Error: {path!r} is not inside a git repository."
    out, rc = _git(["show", "--stat", ref], cwd=root, timeout=30)
    return out if rc == 0 else f"git show failed:\n{out}"


@tool(safe=True, tags=("git", "read"))
def git_branches(cwd: str = ".") -> str:
    """List local and remote branches, with the current branch marked.

    cwd: directory inside the repo.
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."
    out, rc = _git(["branch", "-a", "-v"], cwd=root)
    return out if rc == 0 else f"git branch failed:\n{out}"


# ---------------------------------------------------------------------------
# WRITE tools (safe=False — gated by exec_policy)
# ---------------------------------------------------------------------------

@tool(tags=("git",))
def git_commit(message: str, add_all: bool = False, cwd: str = ".") -> str:
    """Create a git commit. GATED — requires approval.

    message: the commit message.
    add_all: if True, stage all tracked modifications before committing (git commit -a).
    cwd: directory inside the repo.
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."
    args = ["commit", "-m", message]
    if add_all:
        args.insert(1, "-a")
    out, rc = _git(args, cwd=root, timeout=30)
    return out if rc == 0 else f"git commit failed:\n{out}"


@tool(tags=("git",))
def git_add(paths: list, cwd: str = ".") -> str:
    """Stage files for the next commit. GATED — requires approval.

    paths: list of file or directory paths to stage (use ["."] for everything).
    cwd: directory inside the repo.
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."
    out, rc = _git(["add", "--"] + list(paths), cwd=root)
    return out or "Staged." if rc == 0 else f"git add failed:\n{out}"


@tool(tags=("git",))
def git_checkout(ref: str, create_branch: bool = False, cwd: str = ".") -> str:
    """Switch branches or restore files. GATED — requires approval.

    ref: branch name, commit, or file path.
    create_branch: if True, creates the branch with -b.
    cwd: directory inside the repo.
    """
    root = _repo_root(cwd)
    if root is None:
        return f"Error: {cwd!r} is not inside a git repository."
    args = ["checkout"]
    if create_branch:
        args.append("-b")
    args.append(ref)
    out, rc = _git(args, cwd=root)
    return out or f"Checked out {ref!r}." if rc == 0 else f"git checkout failed:\n{out}"
