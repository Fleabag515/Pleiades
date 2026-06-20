"""
sandbox.py -- additive safety rails under run_shell / run_python / start_process.

GOLDEN_BASELINE.md roadmap item #2: "run_shell / run_python currently execute
unsandboxed behind the permission gate. The native layer adds an opt-in sandbox
(resource limits, a scoped filesystem view, and configurable network egress)."
This module is that layer.

This is explicitly NOT a replacement for the approval gate (Agent._default_approve
/ cfg.exec_policy in harness/agent.py) -- that stays mandatory regardless of any
sandbox setting here. This module is defense-in-depth *underneath* approval: even
an approved call, or anything running under exec_policy="allow", still goes
through these rails, so a fat-fingered or misbehaving call can't take out the
host. Approval answers "should this run at all?"; the sandbox answers "if it
runs, how much damage can it do?".

What's real vs. best-effort here, stated plainly (don't oversell this):

  - Wall-clock timeout: real. Existing behavior, unchanged.
  - Memory ceiling: real on POSIX (RLIMIT_AS via preexec_fn, kernel-enforced).
    On every platform there's also a psutil watchdog thread that polls the
    process tree's RSS and kills it if it's over budget -- polling has a
    window, so a microsecond memory spike can slip through, but a runaway
    allocator or fork bomb gets caught within `watchdog_interval` seconds.
    Degrades to timeout-only if psutil isn't installed.
  - Hard destructive-command floor: real, unconditional, every platform, not
    configurable -- this is the one part of the sandbox that ignores
    `enabled`/exec_policy entirely. It blocks whole-system patterns (rm -rf /,
    mkfs, diskpart, fork bombs, reg delete HKLM, recursive-force-delete of a
    drive root). It is a denylist, not a guarantee -- it catches the obvious
    catastrophic cases, not every way to phrase them. The patterns specifically
    tolerate quoting/chaining around the dangerous target (e.g. `rm -rf "/"`,
    `rm -rf /; rest`, `rd /s /q C:\\ && more`) via a negative lookahead instead
    of a hard end-of-string anchor, since a plain end anchor is trivially
    defeated by anything trailing the command -- including a closing quote,
    which is unremarkable, not adversarial, shell style.
  - Filesystem scope: run_python gets a real in-process guard -- open() in a
    write/append/delete mode, os.remove, os.unlink, os.rmdir, shutil.rmtree,
    AND the pathlib.Path equivalents (open/write_text/write_bytes/unlink/
    rmdir, patched directly on the Path class rather than via builtins.open /
    io.open -- pathlib's accessor layer caches its own open reference per
    Python version in ways that make patching builtins.open/io.open
    unreliable across 3.10-3.12, confirmed by hand) are wrapped to deny paths
    outside the workspace root + extra_roots *before user code runs*. The
    containment check resolves symlinks (realpath, not abspath) so a symlink
    planted inside the workspace that points outside it doesn't fool the
    check. This is enforceable because we control the script. run_shell does
    NOT get a true filesystem jail -- Windows has no chroot/namespace
    primitive without containers (WSL2 / Docker), so beyond the hard-deny
    floor above, a shell command can still touch anything the OS account can.
    A real jail is a bigger lift, tracked as a follow-up, not pretended away
    here.
  - Network egress: opt-in via `network="deny"` (default stays "allow" --
    this does not change default behavior). run_python gets a real in-process
    socket block. run_shell gets a static pattern check against well-known
    network-capable commands (curl, git fetch, pip install, ssh, ...) -- this
    catches the obvious cases, not a determined adversary rewriting the
    command to dodge the regex. Same honesty caveat as filesystem scope.

A determined, deliberately adversarial model could still route around the
run_shell and run_python protections above -- e.g. shell out to a second
interpreter that never imports our guard (os.system, subprocess, ctypes), or
split a denylisted command across variables. Defending against that needs
OS-level isolation (a container or VM), which is GOLDEN_BASELINE.md's
documented harder follow-up, not this v1. What this DOES reliably stop:
accidental self-destructive actions (typo'd rm -rf, a careless cleanup
script, a runaway loop, a plain `Path(...).write_text(...)` outside the
workspace) and careless-but-not-adversarial autonomous runs -- which is the
actual failure mode this was built for.
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - degrades to timeout-only enforcement
    psutil = None

try:
    import resource  # type: ignore  # POSIX only
except ImportError:  # pragma: no cover - Windows
    resource = None


class SandboxViolation(Exception):
    """Raised by a hard, unconditional sandbox rule -- distinct from the
    approval gate. Callers turn this into an error string, never let it
    propagate past the tool boundary."""


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass
class SandboxPolicy:
    enabled: bool = True            # the *configurable* part; the hard-deny
                                     # floor below applies even if this is False
    mem_mb: int = 4096               # RSS ceiling for the process tree; 0 = unlimited
    fs_root: str = "."               # confinement root for run_python's write guard
    extra_roots: list = field(default_factory=list)
    network: str = "allow"           # "allow" | "deny"
    watchdog_interval: float = 0.5


def load_policy(cfg=None) -> SandboxPolicy:
    """Build a SandboxPolicy from Settings. Defaults preserve current
    behavior (network stays "allow", mem ceiling is generous) -- the sandbox
    is additive, not a silent behavior change."""
    if cfg is None:
        from .config import Config
        cfg = Config.load()
    return SandboxPolicy(
        enabled=bool(getattr(cfg, "sandbox_enabled", True)),
        mem_mb=int(getattr(cfg, "sandbox_mem_mb", 4096) or 0),
        fs_root=getattr(cfg, "workspace_root", ".") or ".",
        network=getattr(cfg, "sandbox_network", "allow") or "allow",
    )


# --------------------------------------------------------------------------- #
# Hard, unconditional floor (not gated by `enabled` -- see module docstring)
# --------------------------------------------------------------------------- #
# Note on the rm-rf-root / rd-s-q-root / remove-item-root patterns below:
# they deliberately do NOT end on a hard `$` (end-of-string) anchor. An
# end-of-string anchor is defeated by anything trailing the dangerous
# target -- a closing quote (`rm -rf "/"`), a chained command
# (`rm -rf /; rest`), redirection, etc. -- none of which is adversarial,
# just ordinary shell style. Instead they use a negative lookahead that
# only rejects the match when the "root" is actually the start of a longer
# path (e.g. `/home`, `C:\Users`), so legitimate subdirectory deletions
# still pass through untouched.
_HARD_DENY_PATTERNS = [
    r"rm\s+-rf\s+[\"']?/[\"']?(?![\w./-])",         # rm -rf / (+ quotes/chaining)
    r"rm\s+-rf\s+/\*",                             # rm -rf /*
    r":\(\)\{\s*:\|:&\s*\};:",                      # classic shell fork bomb
    r"\bmkfs\b",
    r"\bdd\s+[^\n]*of=/dev/(?:sd|nvme|disk|hd)",   # dd ... of=/dev/sdX
    r"\bdiskpart\b",
    r"\bformat\s+[a-zA-Z]:",                        # format C:
    r"rd\s+/s\s+/q\s+[a-zA-Z]:\\\\?(?![\w\\/-])",   # rd /s /q C:\ (+ chaining)
    r"remove-item\s+[^\n]*-recurse[^\n]*-force[^\n]*[a-zA-Z]:\\\\?(?![\w\\/-])",
    r"reg\s+delete\s+hklm",
]
_HARD_DENY_RE = re.compile("|".join(_HARD_DENY_PATTERNS), re.IGNORECASE)

_NETWORK_PATTERNS = [
    r"\bcurl\b", r"\bwget\b", r"invoke-webrequest\b", r"\biwr\b",
    r"\bssh\b", r"\bscp\b", r"\bsftp\b", r"\bftp\b", r"\bnc\b", r"\bncat\b",
    r"\btelnet\b", r"git\s+(?:clone|fetch|pull|push)", r"pip\s+install",
    r"npm\s+(?:install|i)\b", r"\bchoco\b", r"\bwinget\b",
    r"start-bitstransfer", r"\bnslookup\b",
]
_NETWORK_RE = re.compile("|".join(_NETWORK_PATTERNS), re.IGNORECASE)


def check_command_safety(text: str, policy: SandboxPolicy) -> None:
    """Raise SandboxViolation if `text` (a shell command line, or Python
    source that might shell out) matches a whole-system-destructive pattern.
    This check always runs, regardless of policy.enabled -- it is the floor,
    not the configurable sandbox. The network check IS gated by policy and
    only fires when network="deny" was explicitly requested.
    """
    if _HARD_DENY_RE.search(text):
        raise SandboxViolation(
            "blocked: matches a whole-system-destructive pattern (recursive "
            "delete of a drive root, mkfs/diskpart/format, fork bomb, or "
            "similar). If this is genuinely intended, run it manually outside "
            "Pleiades -- this floor is not configurable."
        )
    if policy.network == "deny" and _NETWORK_RE.search(text):
        raise SandboxViolation(
            "blocked by sandbox network policy (network=deny): this looks "
            "network-capable. Re-run with sandbox_network=allow if this "
            "command legitimately needs egress."
        )


# --------------------------------------------------------------------------- #
# Memory watchdog
# --------------------------------------------------------------------------- #
class _MemWatchdog:
    """Polls a process tree's RSS; kills it if it exceeds mem_mb. No-op if
    psutil isn't installed or mem_mb is 0 (unlimited) -- degrades to
    timeout-only enforcement, never raises."""

    def __init__(self, pid: int, mem_mb: int, interval: float = 0.5) -> None:
        self.pid = pid
        self.mem_mb = mem_mb
        self.interval = interval
        self.triggered = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if psutil is None or not self.mem_mb:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except Exception:
            return
        limit = self.mem_mb * 1024 * 1024
        while not self._stop.is_set():
            try:
                total = proc.memory_info().rss
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
                if total > limit:
                    self.triggered = True
                    for p in [proc] + children:
                        try:
                            p.kill()
                        except Exception:
                            pass
                    return
            except Exception:
                return
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def _preexec_rlimits(mem_mb: int):
    """POSIX-only real ceiling, applied in the child before exec. Returns
    None on Windows (no `resource` module) or when unlimited -- Popen treats
    preexec_fn=None as "no preexec function", which is fine on every
    platform including Windows."""
    if resource is None or not mem_mb:
        return None
    mem_bytes = mem_mb * 1024 * 1024

    def _set() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass

    return _set


# --------------------------------------------------------------------------- #
# Public entry point used by harness/builtins/shell.py and process.py
# --------------------------------------------------------------------------- #
def run_sandboxed(args, *, shell: bool, timeout: int,
                  policy: SandboxPolicy) -> subprocess.CompletedProcess:
    """Drop-in replacement for
        subprocess.run(args, shell=shell, capture_output=True, text=True, timeout=timeout)
    with a memory watchdog layered on top, plus the pre-flight hard-deny /
    network check. Raises subprocess.TimeoutExpired exactly like
    subprocess.run (same contract existing callers already handle) and
    SandboxViolation pre-flight (callers turn that into an error string).
    """
    command_str = args if isinstance(args, str) else " ".join(args)
    check_command_safety(command_str, policy)  # hard floor: always runs

    preexec = _preexec_rlimits(policy.mem_mb) if policy.enabled else None
    proc = subprocess.Popen(
        args, shell=shell,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        preexec_fn=preexec,
    )
    mem_budget = policy.mem_mb if policy.enabled else 0
    watchdog = _MemWatchdog(proc.pid, mem_budget, policy.watchdog_interval)
    watchdog.start()
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        watchdog.stop()
        raise
    finally:
        watchdog.stop()

    rc = proc.returncode
    if watchdog.triggered:
        out = (out or "") + f"\n[killed: exceeded sandbox memory ceiling ({policy.mem_mb} MiB)]"
        rc = -9
    return subprocess.CompletedProcess(args, rc, stdout=out, stderr="")


def attach_watchdog(pid: int, policy: SandboxPolicy) -> "_MemWatchdog":
    """For long-running background processes (process.py's start_process):
    attach a memory watchdog without blocking on communicate(). Caller should
    call .stop() when the process is reaped."""
    w = _MemWatchdog(pid, policy.mem_mb if policy.enabled else 0,
                     policy.watchdog_interval)
    w.start()
    return w


# --------------------------------------------------------------------------- #
# run_python in-process guard
# --------------------------------------------------------------------------- #
# Patches builtins.open/os.remove/os.unlink/os.rmdir/shutil.rmtree (the
# plain-function entry points) AND the equivalent pathlib.Path methods
# directly (open/write_text/write_bytes/unlink/rmdir). pathlib does NOT
# reliably route through builtins.open/io.open across Python versions --
# confirmed by hand on 3.10: Path.write_text/open bottom out through an
# accessor layer that, once io.open is reassigned to a plain Python function,
# triggers descriptor auto-binding and silently corrupts the call's
# positional arguments instead of cleanly dispatching to our wrapper. Patching
# Path's own methods sidesteps that whole hazard and is the standard, well
# understood way to monkeypatch a class.
_PY_GUARD = '''\\
# --- Pleiades sandbox guard (auto-injected, runs before your code) ---
import builtins as _b, os as _os, pathlib as _pathlib, shutil as _shutil, socket as _socket
_FS_ROOT = {fs_root!r}
_EXTRA = {extra_roots!r}
_NET_DENY = {net_deny!r}

def _inside(path):
    try:
        p = _os.path.realpath(_os.fspath(path))
    except Exception:
        return False
    roots = [_os.path.realpath(r) for r in ([_FS_ROOT] + _EXTRA)]
    return any(p == r or p.startswith(r + _os.sep) for r in roots)

_real_open = _b.open
def _guarded_open(file, mode="r", *a, **kw):
    if any(m in mode for m in ("w", "a", "x", "+")) and isinstance(file, (str, bytes, _os.PathLike)):
        if not _inside(file):
            raise PermissionError(f"sandbox: write outside workspace root denied: {{file}}")
    return _real_open(file, mode, *a, **kw)
_b.open = _guarded_open

def _guard_path_call(real, name):
    def _wrapped(path, *a, **kw):
        if not _inside(path):
            raise PermissionError(f"sandbox: {{name}} outside workspace root denied: {{path}}")
        return real(path, *a, **kw)
    return _wrapped

_os.remove = _guard_path_call(_os.remove, "os.remove")
_os.unlink = _guard_path_call(_os.unlink, "os.unlink")
_os.rmdir = _guard_path_call(_os.rmdir, "os.rmdir")
_shutil.rmtree = _guard_path_call(_shutil.rmtree, "shutil.rmtree")

_real_path_open = _pathlib.Path.open
def _guarded_path_open(self, mode="r", *a, **kw):
    if any(m in mode for m in ("w", "a", "x", "+")) and not _inside(self):
        raise PermissionError(f"sandbox: write outside workspace root denied: {{self}}")
    return _real_path_open(self, mode, *a, **kw)
_pathlib.Path.open = _guarded_path_open

_real_write_text = _pathlib.Path.write_text
def _guarded_write_text(self, *a, **kw):
    if not _inside(self):
        raise PermissionError(f"sandbox: write outside workspace root denied: {{self}}")
    return _real_write_text(self, *a, **kw)
_pathlib.Path.write_text = _guarded_write_text

_real_write_bytes = _pathlib.Path.write_bytes
def _guarded_write_bytes(self, *a, **kw):
    if not _inside(self):
        raise PermissionError(f"sandbox: write outside workspace root denied: {{self}}")
    return _real_write_bytes(self, *a, **kw)
_pathlib.Path.write_bytes = _guarded_write_bytes

_real_path_unlink = _pathlib.Path.unlink
def _guarded_path_unlink(self, *a, **kw):
    if not _inside(self):
        raise PermissionError(f"sandbox: unlink outside workspace root denied: {{self}}")
    return _real_path_unlink(self, *a, **kw)
_pathlib.Path.unlink = _guarded_path_unlink

_real_path_rmdir = _pathlib.Path.rmdir
def _guarded_path_rmdir(self, *a, **kw):
    if not _inside(self):
        raise PermissionError(f"sandbox: rmdir outside workspace root denied: {{self}}")
    return _real_path_rmdir(self, *a, **kw)
_pathlib.Path.rmdir = _guarded_path_rmdir

if _NET_DENY:
    def _blocked_connect(self, *a, **kw):
        raise PermissionError("sandbox: network access denied (network=deny)")
    _socket.socket.connect = _blocked_connect
    def _blocked_create_connection(*a, **kw):
        raise PermissionError("sandbox: network access denied (network=deny)")
    _socket.create_connection = _blocked_create_connection
# --- end guard ---

'''


def wrap_python_source(code: str, policy: SandboxPolicy) -> str:
    """Prepend the in-process guard to user code. No-op if disabled (the
    hard command-text check in run_sandboxed-equivalent callers still runs
    separately -- see harness/builtins/shell.py)."""
    if not policy.enabled:
        return code
    guard = _PY_GUARD.format(
        fs_root=str(Path(policy.fs_root).resolve()),
        extra_roots=[str(Path(r).resolve()) for r in policy.extra_roots],
        net_deny=(policy.network == "deny"),
    )
    return guard + "\n" + code
