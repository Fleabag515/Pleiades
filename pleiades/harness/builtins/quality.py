"""Code-quality tools — format, lint, test. The agent's feedback loop.

After writing code the agent should *know* whether it's correct, not guess.
Each tool auto-detects the right underlying program by file extension /
project layout, and reports what to install when nothing suitable is found.

    format_code   black/ruff (py), prettier (js/ts/json/css/md), gofmt, rustfmt
    lint_code     ruff/flake8/pyflakes (py), eslint (js/ts), go vet
    run_tests     pytest (py), npm test / jest (js), cargo test, go test
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..tools import tool

_OUT_CAP = 20_000


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return f"Error: '{cmd[0]}' timed out after {timeout}s."
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' not found."
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return f"[exit {proc.returncode}] {' '.join(cmd)}\n" + (out[:_OUT_CAP] or "(no output)")


def _first(*names: str) -> str | None:
    for n in names:
        if shutil.which(n):
            return n
    return None


_PRETTIER_EXT = {".js", ".jsx", ".ts", ".tsx", ".json", ".css", ".scss",
                 ".html", ".md", ".yaml", ".yml"}


# ---------------------------------------------------------------------------
@tool(tags=("quality",))
def format_code(path: str) -> str:
    """Auto-format a source file in place using the standard formatter for its language.

    path: the file to format (.py/.js/.ts/.json/.css/.md/.go/.rs/...).
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such file: {path}"
    ext = p.suffix.lower()

    if ext == ".py":
        if shutil.which("ruff"):
            return _run(["ruff", "format", str(p)])
        if shutil.which("black"):
            return _run(["black", "-q", str(p)])
        return "Error: no Python formatter — pip install ruff (or black)"
    if ext in _PRETTIER_EXT:
        if _first("prettier"):
            return _run(["prettier", "--write", str(p)])
        if shutil.which("npx"):
            return _run(["npx", "--yes", "prettier", "--write", str(p)], timeout=180)
        return "Error: prettier not found — npm install -g prettier"
    if ext == ".go" and shutil.which("gofmt"):
        return _run(["gofmt", "-w", str(p)])
    if ext == ".rs" and shutil.which("rustfmt"):
        return _run(["rustfmt", str(p)])
    return f"Error: no formatter known/installed for '{ext}' files."


# ---------------------------------------------------------------------------
@tool(safe=True, tags=("quality", "read"))
def lint_code(path: str) -> str:
    """Lint a source file or directory; returns the issues found (empty = clean).

    path: file or directory to lint.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such path: {path}"
    ext = p.suffix.lower()

    if ext == ".py" or (p.is_dir() and any(p.glob("**/*.py"))):
        linter = _first("ruff", "flake8", "pyflakes")
        if linter == "ruff":
            return _run(["ruff", "check", str(p)])
        if linter:
            return _run([linter, str(p)])
        return "Error: no Python linter — pip install ruff"
    if ext in (".js", ".jsx", ".ts", ".tsx"):
        if shutil.which("eslint"):
            return _run(["eslint", str(p)])
        if shutil.which("npx"):
            return _run(["npx", "--yes", "eslint", str(p)], timeout=180)
        return "Error: eslint not found — npm install -g eslint"
    if ext == ".go" and shutil.which("go"):
        return _run(["go", "vet", str(p)], cwd=str(p.parent))
    return f"Error: no linter known/installed for '{path}'."


# ---------------------------------------------------------------------------
@tool(tags=("quality",))
def run_tests(path: str = ".", pattern: str = "", timeout: int = 300) -> str:
    """Run the project's test suite and return the pass/fail summary plus failures.

    path: project directory (or a single test file).
    pattern: optional filter — pytest -k expression / jest test-name pattern.
    timeout: seconds before the run is killed.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such path: {path}"
    root = p if p.is_dir() else p.parent

    # Python
    if p.suffix == ".py" or (root / "pytest.ini").exists() \
            or (root / "pyproject.toml").exists() or list(root.glob("test*.py")) \
            or (root / "tests").exists():
        if shutil.which("pytest"):
            cmd = ["pytest", "-q", "--no-header", str(p)]
            if pattern:
                cmd += ["-k", pattern]
            return _summarize(_run(cmd, cwd=str(root), timeout=timeout))
        return "Error: pytest not found — pip install pytest"
    # Node
    if (root / "package.json").exists():
        if shutil.which("npm"):
            cmd = ["npm", "test", "--silent"]
            if pattern:
                cmd += ["--", "-t", pattern]
            return _summarize(_run(cmd, cwd=str(root), timeout=timeout))
        return "Error: npm not found."
    # Rust / Go
    if (root / "Cargo.toml").exists() and shutil.which("cargo"):
        cmd = ["cargo", "test"] + ([pattern] if pattern else [])
        return _summarize(_run(cmd, cwd=str(root), timeout=timeout))
    if list(root.glob("*_test.go")) and shutil.which("go"):
        cmd = ["go", "test", "./..."] + (["-run", pattern] if pattern else [])
        return _summarize(_run(cmd, cwd=str(root), timeout=timeout))
    return f"Error: no recognizable test setup under '{path}'."


def _summarize(output: str) -> str:
    """Keep the head (command/exit) and the tail (summary + failures)."""
    if len(output) <= _OUT_CAP:
        return output
    head, _, rest = output.partition("\n")
    return head + "\n... [output trimmed] ...\n" + rest[-(_OUT_CAP - 200):]
