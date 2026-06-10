"""The agent loop: inference + tool calls.

Flow (see CLAUDE.md §4):
  1. Resolve the profile, ensure OUR inference server is up, and point the character's
     Anamnesis upstream at it.
  2. Ensure the Anamnesis character proxy is running; build an OpenAI client at its URL.
  3. Build the character's ToolBelt (search, email, browser, vault — gated by config).
  4. Send only the new user message + tool schemas. Anamnesis injects history & memory.
  5. Dispatch any tool_calls, append results, loop (with a hard cap).
  6. Return the final assistant message.

Memory is Anamnesis's job — we never re-send long histories or separately persist turns.
"""

from __future__ import annotations

from typing import Optional, Union

from . import config
from .anamnesis import Anamnesis
from .inference import ensure_inference
from .profiles import Profile, ProfileManager
from .tools import ToolBelt, ToolContext, build_default_belt

MAX_TOOL_ITERATIONS = 8


class Engine:
    def __init__(
        self,
        settings: Optional[config.Settings] = None,
        manager: Optional[ProfileManager] = None,
        anamnesis: Optional[Anamnesis] = None,
    ):
        self.settings = settings or config.Settings.load()
        self.anamnesis = anamnesis or Anamnesis(self.settings.anamnesis_control_url)
        self.manager = manager or ProfileManager(self.settings, self.anamnesis)

    # -- wiring ------------------------------------------------------------- #
    def _upstream_for(self, profile: Profile) -> str:
        """The OpenAI-compatible URL to use for this character.

        If the character has an assigned model (profile.model), run/return that model's
        dedicated server; otherwise use the default in-process engine (PLEIADES_MODEL_PATH).
        """
        model = getattr(profile, "model", "") or ""
        if model:
            from .models import ModelManager, ModelError
            mm = ModelManager()
            if mm.get(model):
                try:
                    if not mm.is_running(model):
                        mm.start(model)        # auto-start the assigned model
                    return mm.base_url(model)
                except ModelError:
                    pass  # fall back to the default engine below
        return ensure_inference(self.settings)

    def _client_for(self, profile: Profile):
        """Bring up inference + the character proxy and return an OpenAI client + ctx."""
        from openai import OpenAI

        # 1. Resolve the upstream the character should use, and point Anamnesis at it.
        upstream = self._upstream_for(profile)
        try:
            self.anamnesis.update_character(profile.name, upstream={"baseUrl": upstream})
        except Exception:
            # Non-fatal: the character may already be configured, or PATCH unsupported.
            pass

        # 2. Character proxy.
        proxy_url = self.anamnesis.ensure_running(
            profile.name, upstream={"baseUrl": upstream}
        )
        client = OpenAI(base_url=proxy_url, api_key="pleiades")

        # 3. Tools, scoped to this character — the chat IS the agent, so the
        # full harness tool registry (files, shell, git, web, browser, memory,
        # subagents, ...) is bridged in alongside the chat-native tools.
        vault = self.manager.open_vault(profile.name)
        ctx = ToolContext(profile=profile, vault=vault, settings=self.settings)
        belt = build_default_belt(ctx)
        self._bridge_harness_tools(belt, profile)
        return client, ctx, belt

    def _bridge_harness_tools(self, belt: ToolBelt, profile: Profile) -> None:
        """Expose every harness tool in chat (chat-native names win on clash)."""
        try:
            from .harness import builtins as _builtins      # noqa: F401  registers
            from .harness import identity as _identity
            from .harness.tools import registry
            from .tools import Tool

            try:  # bind identity so file/shell/memory tools live in the
                  # character's own workspace and memory
                _identity.bind_character(profile.name, cfg=self.settings)
            except Exception:
                pass

            class _Bridge(Tool):
                def __init__(self, ht):
                    self._ht = ht
                    self.name = ht.name
                    self.description = ht.description
                    self.parameters = ht.schema

                def run(self, ctx, **kwargs):
                    return str(self._ht.func(**kwargs))

            for ht in registry.all():
                if ht.name not in belt:
                    belt.add(_Bridge(ht))
        except Exception:
            # The harness is optional context — chat works without it.
            pass

    # -- main entry point --------------------------------------------------- #
    def run(self, profile: Union[str, Profile], user_message: str, *, system: Optional[str] = None) -> str:
        if isinstance(profile, str):
            profile = self.manager.get(profile)

        client, ctx, belt = self._client_for(profile)
        try:
            return self._loop(client, ctx, belt, profile, user_message, system)
        finally:
            ctx.close()

    @staticmethod
    def _base_messages(user_message: str, system: Optional[str]) -> list[dict]:
        """The new turn only (Anamnesis supplies memory/history) + local time."""
        import datetime
        now = datetime.datetime.now().astimezone()
        time_line = ("Current local date and time: "
                     + now.strftime("%A, %B %d %Y, %H:%M (%Z)"))
        sys_text = f"{system.rstrip()}\n\n{time_line}" if system else time_line
        return [{"role": "system", "content": sys_text},
                {"role": "user", "content": user_message}]

    def _loop(self, client, ctx: ToolContext, belt: ToolBelt, profile: Profile, user_message: str, system: Optional[str]) -> str:
        messages = self._base_messages(user_message, system)

        tools = belt.openai_schema()

        for _ in range(MAX_TOOL_ITERATIONS):
            resp = client.chat.completions.create(
                model=profile.name,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                return msg.content or ""

            # Echo the assistant's tool-call message, then each tool result.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                result = belt.dispatch(tc.function.name, tc.function.arguments, ctx)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        return "[engine] stopped after hitting the tool-call iteration cap."

    def stream_events(self, profile: Union[str, Profile], user_message: str,
                      *, system: Optional[str] = None):
        """The full turn as a stream of structured events.

        Yields dicts, in order:
            {"type": "token", "text": str}                       assistant text
            {"type": "tool_call", "name": str, "args": str}
            {"type": "tool_result", "name": str, "output": str, "ok": bool}
            {"type": "done", "tokens": int, "seconds": float, "tps": float}
        Tokens stream live from llama.cpp; each turn runs exactly once, so
        Anamnesis stores exactly what the caller sees. Falls back to a
        non-streamed request if the backend rejects stream+tools.
        """
        if isinstance(profile, str):
            profile = self.manager.get(profile)
        client, ctx, belt = self._client_for(profile)
        n_tokens = 0
        t_first = t_last = None
        try:
            messages = self._base_messages(user_message, system)
            tools = belt.openai_schema()
            import time as _time

            for _ in range(MAX_TOOL_ITERATIONS):
                content = ""
                calls: dict[int, dict] = {}
                streamed = True
                try:
                    stream = client.chat.completions.create(
                        model=profile.name, messages=messages,
                        tools=tools or None,
                        tool_choice="auto" if tools else None, stream=True,
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is None:
                            continue
                        if delta.content:
                            content += delta.content
                            n_tokens += 1
                            now = _time.time()
                            t_first = t_first or now
                            t_last = now
                            yield {"type": "token", "text": delta.content}
                        for tc in (delta.tool_calls or []):
                            slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id:
                                slot["id"] = tc.id
                            if tc.function and tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                slot["args"] += tc.function.arguments
                except Exception:
                    streamed = False
                if not streamed:
                    # Backend refused streaming (some llama.cpp builds reject
                    # stream+tools). One non-streamed request instead.
                    resp = client.chat.completions.create(
                        model=profile.name, messages=messages,
                        tools=tools or None,
                        tool_choice="auto" if tools else None,
                    )
                    msg = resp.choices[0].message
                    content = msg.content or ""
                    if content:
                        n_tokens += max(1, len(content) // 4)
                        now = _time.time()
                        t_first = t_first or now
                        t_last = now
                        yield {"type": "token", "text": content}
                    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
                        calls[i] = {"id": tc.id, "name": tc.function.name,
                                    "args": tc.function.arguments}

                if not calls:
                    break

                ordered = [calls[i] for i in sorted(calls)]
                messages.append({
                    "role": "assistant", "content": content,
                    "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                    "function": {"name": c["name"], "arguments": c["args"]}}
                                   for i, c in enumerate(ordered)],
                })
                for i, c in enumerate(ordered):
                    yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
                    result = belt.dispatch(c["name"], c["args"], ctx)
                    ok = not result.startswith("[error]")
                    yield {"type": "tool_result", "name": c["name"],
                           "output": result, "ok": ok}
                    messages.append({"role": "tool",
                                     "tool_call_id": c["id"] or f"call_{i}",
                                     "content": result})

            seconds = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
            tps = round(n_tokens / seconds, 1) if seconds > 0 else 0.0
            yield {"type": "done", "tokens": n_tokens,
                   "seconds": round(seconds, 2), "tps": tps}
        finally:
            ctx.close()

    def stream(self, profile: Union[str, Profile], user_message: str, *, system: Optional[str] = None):
        """Text-only view of stream_events (CLI REPL / Discord)."""
        for evt in self.stream_events(profile, user_message, system=system):
            if evt["type"] == "token":
                yield evt["text"]
