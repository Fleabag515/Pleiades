"""File tools — read, write, edit, search, list, glob. Reads are safe; writes are gated."""

from __future__ import annotations

import difflib
import glob as _glob
import re
from pathlib import Path

from ..tools import tool


@tool(safe=True, tags=("file", "read"))
def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a UTF-8 text file, optionally restricted to a line range.

    path: absolute or workspace-relative file path.
    start_line: first line to return, 1-indexed (0 = from the beginning).
    end_line: last line to return, 1-indexed inclusive (0 = to the end).
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such file: {path}"
    if p.is_dir():
        return f"Error: {path} is a directory (use list_dir)."
    text = p.read_text(encoding="utf-8", errors="replace")

    if start_line > 0 or end_line > 0:
        lines = text.splitlines(keepends=True)
        total = len(lines)
        s = max(0, start_line - 1) if start_line > 0 else 0
        e = min(total, end_line) if end_line > 0 else total
        slice_lines = lines[s:e]
        header = f"[lines {s+1}–{min(e, total)} of {total} total]\n"
        text = header + "".join(slice_lines)
    else:
        if len(text) > 40_000:
            lines = text.splitlines()
            return (f"[showing lines 1–{len(lines)} — {len(text)} bytes total; "
                    f"use start_line/end_line to read sections]\n"
                    + text[:40_000] + "\n... [truncated]")

    return text


@tool(tags=("file",))
def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file with the given content.

    path: destination file path (parent dirs are created).
    content: full UTF-8 text to write.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {p}"


@tool(tags=("file",))
def append_file(path: str, content: str) -> str:
    """Append text to a file, creating it if it does not exist.

    path: destination file path.
    content: text to append (a newline is added between existing content and new).
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = p.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            content = "\n" + content
        p.write_text(existing + content, encoding="utf-8")
    else:
        p.write_text(content, encoding="utf-8")
    return f"Appended {len(content)} bytes to {p}"


@tool(tags=("file",))
def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace an exact substring in a file.

    path: file to edit.
    old: exact substring to find (must match literally, including whitespace).
    new: replacement text.
    replace_all: if True, replace every occurrence; otherwise only the first.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such file: {path}"
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        return "Error: `old` string not found — read the file and retry with the exact text."
    if not replace_all and count > 1:
        return (f"Error: `old` matches {count} locations. Either make `old` more specific "
                "(include surrounding lines) or pass replace_all=true.")
    new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    p.write_text(new_text, encoding="utf-8")
    n = count if replace_all else 1
    return f"Edited {p} ({n} replacement{'s' if n > 1 else ''})."


@tool(safe=True, tags=("file", "read"))
def preview_edit(path: str, old: str, new: str) -> str:
    """Show a unified diff of what edit_file would change, WITHOUT applying it. Always preview before editing.

    path: file to preview the edit on.
    old: exact substring to find.
    new: replacement text.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: no such file: {path}"
    text = p.read_text(encoding="utf-8")
    if old not in text:
        return "Error: `old` string not found — the edit would fail."
    new_text = text.replace(old, new, 1)
    diff = list(difflib.unified_diff(
        text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{p.name}",
        tofile=f"b/{p.name}",
        n=3,
    ))
    if not diff:
        return "(no change — old and new are identical)"
    return "".join(diff)


@tool(safe=True, tags=("file", "read"))
def grep_search(pattern: str, path: str = ".", glob_pattern: str = "**/*",
                context: int = 2, max_matches: int = 40) -> str:
    """Search files for a regex pattern, returning matches with line numbers and surrounding context.

    pattern: regex to find (e.g. "def authenticate" or "class.*Error").
    path: root directory to search in (default current directory).
    glob_pattern: file filter (e.g. "**/*.py"). Uses Python's glob syntax, which
        does NOT support brace expansion (no "**/*.{ts,js}") — call this tool
        once per extension instead, or match broadly and filter in the pattern.
        Default: all files.
    context: lines of context before/after each match (default 2).
    max_matches: cap on total matches returned (default 40).
    """
    base = Path(path).expanduser().resolve()
    if not base.exists():
        return f"Error: path does not exist: {path}"

    try:
        rx = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return f"Error: invalid regex — {e}"

    matches_found = 0
    out: list[str] = []

    candidate_files = sorted(base.rglob(
        glob_pattern.lstrip("/") if glob_pattern != "**/*" else "*"
    ) if "**" not in glob_pattern else base.glob(glob_pattern))

    # Skip binary files and unreadable paths
    for fpath in candidate_files:
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue

        lines = text.splitlines()
        file_header_added = False
        i = 0
        while i < len(lines):
            if rx.search(lines[i]):
                if matches_found >= max_matches:
                    out.append(f"... (max_matches={max_matches} reached)")
                    return "\n".join(out)
                if not file_header_added:
                    rel = fpath.relative_to(base)
                    out.append(f"\n### {rel}")
                    file_header_added = True
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                for j in range(start, end):
                    marker = ">" if j == i else " "
                    out.append(f"{marker} {j+1:4d}│ {lines[j]}")
                out.append("")
                matches_found += 1
                i = end  # skip past context to avoid duplicate blocks
            else:
                i += 1

    if not out:
        return f"(no matches for {pattern!r} in {path})"
    return "\n".join(out)


@tool(safe=True, tags=("file", "read"))
def get_symbols(path: str, kind: str = "all") -> str:
    """List functions, classes, and methods defined in a source file, with line numbers.

    Faster than reading the whole file when you just need to know what's in it.
    Supports Python (AST), and falls back to regex for other languages.

    path: source file to inspect.
    kind: filter to "functions", "classes", or "all" (default).
    """
    import ast as _ast

    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: not a file: {path}"

    src = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()

    # --- Python: use the AST for precision ---
    if ext == ".py":
        try:
            tree = _ast.parse(src, filename=str(p))
        except SyntaxError as e:
            return f"Parse error in {path}: {e}"

        out: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if kind in ("all", "functions"):
                    async_mark = "async " if isinstance(node, _ast.AsyncFunctionDef) else ""
                    args = [a.arg for a in node.args.args]
                    out.append(f"  {node.lineno:4d}  def {async_mark}{node.name}({', '.join(args)})")
            elif isinstance(node, _ast.ClassDef):
                if kind in ("all", "classes"):
                    bases = [_ast.unparse(b) for b in node.bases] if node.bases else []
                    base_str = f"({', '.join(bases)})" if bases else ""
                    out.append(f"  {node.lineno:4d}  class {node.name}{base_str}")
        if not out:
            return f"(no symbols found in {path})"
        header = f"Symbols in {p.name} ({len(out)} found):\n"
        return header + "\n".join(sorted(out, key=lambda x: int(x.strip().split()[0])))

    # --- Other languages: regex heuristics ---
    patterns: dict[str, list[tuple[str, str]]] = {
        ".js":  [("function", r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()"),
                 ("class",    r"class\s+(\w+)")],
        ".ts":  [("function", r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()"),
                 ("class",    r"class\s+(\w+)")],
        ".rb":  [("function", r"def\s+(\w+)"), ("class", r"class\s+(\w+)")],
        ".go":  [("function", r"func\s+(?:\(\w+\s+\*?\w+\)\s*)?(\w+)\s*\("),
                 ("class",    r"type\s+(\w+)\s+struct")],
        ".rs":  [("function", r"(?:pub\s+)?fn\s+(\w+)"), ("class", r"(?:pub\s+)?struct\s+(\w+)")],
        ".cpp": [("function", r"\b(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{"),
                 ("class",    r"class\s+(\w+)")],
        ".c":   [("function", r"\b(\w+)\s*\([^)]*\)\s*\{")],
        ".java":[("function", r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\("),
                 ("class",    r"class\s+(\w+)")],
    }

    pats = patterns.get(ext, [("symbol", r"(?:def|function|func|class|struct)\s+(\w+)")])
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        for sym_kind, pat in pats:
            if kind in ("all", "functions") and sym_kind == "function":
                m = re.search(pat, line)
                if m:
                    name = next((g for g in m.groups() if g), "?")
                    out.append(f"  {i:4d}  {sym_kind} {name}")
            elif kind in ("all", "classes") and sym_kind in ("class", "struct", "symbol"):
                m = re.search(pat, line)
                if m:
                    name = next((g for g in m.groups() if g), "?")
                    out.append(f"  {i:4d}  {sym_kind} {name}")
    if not out:
        return f"(no symbols found in {path} for kind={kind!r})"
    return f"Symbols in {p.name} ({len(out)} found):\n" + "\n".join(out)


@tool(safe=True, tags=("file", "read"))
def list_dir(path: str = ".") -> str:
    """List the entries in a directory (one per line, dirs marked with /).

    path: directory to list (default current directory).
    """
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    out = []
    for e in sorted(p.iterdir()):
        out.append(e.name + ("/" if e.is_dir() else ""))
    return "\n".join(out) or "(empty)"


@tool(safe=True, tags=("file", "read"))
def find_files(pattern: str, root: str = ".") -> str:
    """Find files matching a glob pattern (recursive).

    pattern: glob like "**/*.py" or "src/*.txt".
    root: base directory to search from.
    """
    base = Path(root).expanduser()
    matches = _glob.glob(str(base / pattern), recursive=True)
    return "\n".join(sorted(matches)[:200]) or "(no matches)"