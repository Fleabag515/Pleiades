"""
llm.py — hybrid backend: Anthropic (Claude) + local Ollama, one interface.

`chat()` returns a normalized `Reply`:
    Reply.text          -> assistant prose (may be "")
    Reply.tool_calls    -> [ToolCall(id, name, args), ...]
    Reply.raw_content    -> backend-native assistant turn, appended verbatim to
                            history so multi-turn tool use round-trips correctly.

The agent loop is backend-agnostic: it reads .text / .tool_calls, executes
tools, and feeds results back via `format_tool_results()` for the right backend.

Anthropic uses content blocks (tool_use / tool_result). Ollama uses OpenAI-style
messages (tool_calls / role:"tool"). We translate at the edges so the loop sees
one shape.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import Config, Tier
from .tools import Tool


# ---------------------------------------------------------------------------
# Session-level token accumulator
# system.py's context_status() reads this to show the agent its budget.
# ---------------------------------------------------------------------------
_SESSION_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0,
                                  "calls": 0, "last_input_tokens": 0}


def get_session_usage() -> dict[str, int]:
    """Return a copy of accumulated token usage for this interpreter session."""
    return dict(_SESSION_USAGE)


def reset_session_usage() -> None:
    """Reset the session counters (e.g. at the start of a fresh task)."""
    _SESSION_USAGE.update(input_tokens=0, output_tokens=0, calls=0,
                          last_input_tokens=0)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None          # backend-native assistant turn
    backend: str = ""
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)   # input_tokens / output_tokens this call


class LLM:
    """Routes a chat request to the configured backend for a given tier."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._anthropic = None  # lazy

    # -- public ---------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[Tool],
             tier: Tier, system: str | None = None) -> Reply:
        if tier.backend == "anthropic":
            return self._chat_anthropic(messages, tools, tier, system)
        if tier.backend == "ollama":
            return self._chat_ollama(messages, tools, tier, system)
        if tier.backend == "openai":
            return self._chat_openai(messages, tools, tier, system)
        raise ValueError(f"unknown backend: {tier.backend}")

    def format_tool_results(self, backend: str,
                            results: list[tuple[ToolCall, str, bool]]) -> "dict | list":
        """Build the user-turn message carrying tool outputs back to the model.

        results: list of (call, output_text, is_error).
        """
        if backend == "anthropic":
            blocks = [{
                "type": "tool_result",
                "tool_use_id": c.id,
                "content": out,
                **({"is_error": True} if err else {}),
            } for c, out, err in results]
            return {"role": "user", "content": blocks}
        # ollama / openai: one message per tool result, each carrying the
        # tool_call_id required to bind it to its call (see agent loop, which
        # builds these inline). Returned as a list for callers that use this.
        return [{"role": "tool", "tool_call_id": c.id, "name": c.name,
                 "content": out} for c, out, err in results]

    # -- anthropic ------------------------------------------------------
    def _client(self):
        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(
                api_key=self.cfg.anthropic_api_key or None
            )
        return self._anthropic

    def _chat_anthropic(self, messages, tools, tier: Tier, system) -> Reply:
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": tier.model,
            "max_tokens": tier.max_tokens,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": tier.effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.input_schema() for t in tools]

        resp = client.messages.create(**kwargs)
        usage = {}
        if hasattr(resp, "usage") and resp.usage:
            usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0),
                     "output_tokens": getattr(resp.usage, "output_tokens", 0)}
        self._account(usage)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in resp.content if b.type == "tool_use"
        ]
        return Reply(text=text, tool_calls=calls, raw_content=resp.content,
                     backend="anthropic", stop_reason=resp.stop_reason or "",
                     usage=usage)

    @staticmethod
    def _account(usage: dict) -> None:
        _SESSION_USAGE["input_tokens"] += usage.get("input_tokens", 0)
        _SESSION_USAGE["output_tokens"] += usage.get("output_tokens", 0)
        _SESSION_USAGE["last_input_tokens"] = usage.get("input_tokens", 0)
        _SESSION_USAGE["calls"] += 1

    # -- ollama ---------------------------------------------------------
    def _chat_ollama(self, messages, tools, tier: Tier, system) -> Reply:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        body: dict[str, Any] = {
            "model": tier.model,
            "messages": msgs,
            "stream": False,
            "options": {"num_predict": tier.max_tokens},
        }
        if tools:
            body["tools"] = [t.openai_schema() for t in tools]

        data = self._post(f"{self.cfg.ollama_host}/api/chat", body)
        usage = {"input_tokens": data.get("prompt_eval_count", 0) or 0,
                 "output_tokens": data.get("eval_count", 0) or 0}
        self._account(usage)
        msg = data.get("message", {}) or {}
        text = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""
        if reasoning and not text:
            text = f"[Reasoning]: {reasoning}"
        calls = []
        for i, tc in enumerate(msg.get("tool_calls", []) or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append(ToolCall(id=f"ollama_{i}", name=fn.get("name", ""),
                                  args=args))
        return Reply(text=text, tool_calls=calls, raw_content=msg,
                     backend="ollama",
                     stop_reason="tool_use" if calls else "end_turn",
                     usage=usage)

    # -- openai (compatible proxies e.g. Anamnesis) ---------------------
    def _chat_openai(self, messages, tools, tier: Tier, system) -> Reply:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        body: dict[str, Any] = {
            "model": tier.model,
            "messages": msgs,
            "stream": False,
            "max_tokens": tier.max_tokens,
        }
        if tools:
            body["tools"] = [t.openai_schema() for t in tools]

        data = self._post(f"{self.cfg.openai_host}/chat/completions", body)
        u = data.get("usage", {}) or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0) or 0,
                 "output_tokens": u.get("completion_tokens", 0) or 0}
        self._account(usage)
        choices = data.get("choices", [])
        if not choices:
            return Reply(text="", tool_calls=[], backend="openai", usage=usage)
        msg = choices[0].get("message", {})
        text = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""
        if reasoning and not text:
            text = f"[Reasoning]: {reasoning}"
        calls = []
        for i, tc in enumerate(msg.get("tool_calls", []) or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append(ToolCall(id=tc.get("id") or f"openai_{i}", name=fn.get("name", ""),
                                  args=args))
        return Reply(text=text, tool_calls=calls, raw_content=msg,
                     backend="openai",
                     stop_reason="tool_use" if calls else "end_turn",
                     usage=usage)


    @staticmethod
    def _post(url: str, body: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))
