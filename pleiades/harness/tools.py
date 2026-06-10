"""
tools.py — the tool registry, the heart of Pleiades.

A tool is a plain Python function decorated with `@tool`. Its JSON schema is
built automatically from the signature and docstring, so adding a capability is
just writing a function. The same registry is rendered for both the Anthropic
API (`input_schema`) and Ollama (OpenAI-style `parameters`).

    @tool(safe=True)                       # read-only, never gated
    def read_file(path: str) -> str:
        '''Read a UTF-8 text file.

        path: absolute or workspace-relative file path.
        '''
        ...

Side-effecting tools (safe=False, the default) pass through the agent's
permission policy before they run — this is how "the PC is their home, but
unsandboxed is the worst case" is enforced in practice.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


_PYTYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    schema: dict[str, Any]
    safe: bool = False          # read-only tools are never gated
    tags: tuple[str, ...] = ()  # e.g. ("web",) — used for subagent tool subsets

    def input_schema(self) -> dict[str, Any]:
        """Anthropic-style tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }

    def openai_schema(self) -> dict[str, Any]:
        """Ollama / OpenAI-style tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def select(self, names: list[str] | None = None,
               tags: list[str] | None = None) -> list[Tool]:
        """Subset of tools by explicit name or by tag (for subagents)."""
        out = []
        for t in self._tools.values():
            if names is not None and t.name not in names:
                continue
            if tags is not None and not (set(tags) & set(t.tags)):
                continue
            out.append(t)
        return out

    def __len__(self) -> int:
        return len(self._tools)


# the global registry every @tool lands in
registry = Registry()


def _build_schema(func: Callable) -> dict[str, Any]:
    """Derive a JSON Schema object from the function signature + type hints.

    Parameter descriptions are pulled from `name: desc` lines in the docstring.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    doc = inspect.getdoc(func) or ""
    param_docs: dict[str, str] = {}
    for line in doc.splitlines():
        s = line.strip()
        if ":" in s:
            key, _, rest = s.partition(":")
            key = key.strip()
            if key in sig.parameters and rest.strip():
                param_docs[key] = rest.strip()

    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if pname in ("self", "ctx", "_ctx"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = hints.get(pname, str)
        origin = getattr(ann, "__origin__", None)
        jtype = _PYTYPE_TO_JSON.get(origin or ann, "string")
        prop: dict[str, Any] = {"type": jtype}
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        if origin in (list,) or ann in (list,):
            prop["items"] = {"type": "string"}
        props[pname] = prop
        if p.default is inspect._empty:
            required.append(pname)

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def tool(_func: Callable | None = None, *, name: str | None = None,
         safe: bool = False, tags: tuple[str, ...] = ()) -> Callable:
    """Register a function as a tool. Use bare `@tool` or `@tool(safe=True)`.

    The first line of the docstring is the tool description shown to the model;
    `param: description` lines document individual arguments.
    """
    def wrap(func: Callable) -> Callable:
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n\n", 1)[0].strip() or func.__name__
        t = Tool(
            name=name or func.__name__,
            description=description,
            func=func,
            schema=_build_schema(func),
            safe=safe,
            tags=tags,
        )
        registry.add(t)
        func._pleiades_tool = t  # type: ignore[attr-defined]
        return func

    return wrap(_func) if _func is not None else wrap
