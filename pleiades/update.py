"""Self-update — ONE implementation for the CLI and the web UI.

`check()` compares the local checkout to origin without touching it;
`run_update()` fetches + hard-resets to the remote branch (shallow-clone safe,
same strategy as the installer) and reinstalls the package so new dependencies
land too. The web UI calls these from its Update button and restarts itself.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def repo_root() -> Path:
    """The Pleiades git checkout this code runs from."""
    return Path(__file__).resolve().parent.parent


def is_git_checkout(root: Optional[Path] = None) -> bool:
    return ((root or repo_root()) / ".git").exists()


def _git(root: Path, *args: str, timeout: float = 60.0) -> tuple[int, str]:
    try:
        out = subprocess.run(["git", "-C", str(root), *args],
                             capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


@dataclass
class UpdateCheck:
    supported: bool          # is this a git checkout we can update?
    installed: str = ""      # local short sha
    latest: str = ""         # remote short sha
    behind: bool = False
    error: str = ""


def check(root: Optional[Path] = None) -> UpdateCheck:
    """Compare local HEAD against origin's branch tip (network: ls-remote only)."""
    root = root or repo_root()
    if not is_git_checkout(root):
        return UpdateCheck(supported=False,
                           error="not a git checkout — re-run the installer to update")
    code, local = _git(root, "rev-parse", "HEAD")
    if code != 0:
        return UpdateCheck(supported=False, error=local)
    _, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch if branch and branch != "HEAD" else "main"
    code, out = _git(root, "ls-remote", "origin", branch, timeout=20.0)
    if code != 0 or not out:
        return UpdateCheck(supported=True, installed=local[:7],
                           error=f"could not reach origin: {out or 'no output'}")
    remote = out.split()[0]
    return UpdateCheck(supported=True, installed=local[:7], latest=remote[:7],
                       behind=remote != local)


def run_update(root: Optional[Path] = None, *, reinstall: bool = True,
               log: Optional[Callable[[str], None]] = None) -> bool:
    """Fetch + hard-reset to origin, then reinstall. Returns True if code changed.

    Raises RuntimeError with a readable message on failure.
    """
    say = log or (lambda _msg: None)
    root = root or repo_root()
    if not is_git_checkout(root):
        raise RuntimeError("This install is not a git checkout; re-run the installer.")
    _, before = _git(root, "rev-parse", "HEAD")
    _, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch if branch and branch != "HEAD" else "main"

    say(f"fetching origin/{branch} …")
    code, out = _git(root, "fetch", "origin", branch, timeout=120.0)
    if code != 0:
        raise RuntimeError(f"git fetch failed: {out}")
    code, out = _git(root, "reset", "--hard", f"origin/{branch}")
    if code != 0:
        raise RuntimeError(f"git reset failed: {out}")
    _, after = _git(root, "rev-parse", "HEAD")
    changed = before != after
    say(f"now at {after[:7]}" + ("" if changed else " (already up to date)"))

    if changed and reinstall:
        say("reinstalling dependencies …")
        out = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                              "-e", f"{root}[all]"],
                             capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise RuntimeError(f"pip reinstall failed: {(out.stderr or '')[-400:]}")
    return changed
