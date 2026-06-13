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

import os
from typing import Optional, Union

from . import config
from .anamnesis import Anamnesis
from .inference import ensure_inference
from .profiles import Profile, ProfileManager
from .tools import ToolBelt, ToolContext, build_default_belt

MAX_TOOL_ITERATIONS = 8

# Persona-agnostic operating contract injected into the chat path when the caller
# supplies no system prompt of its own. Anamnesis injects the character's persona
# (voice) separately; this only governs *how the character operates its tools* so a
# small local model actually uses them instead of narrating or stalling. It must not
# overwrite the voice — hence "stay fully in your established voice/persona".
DEFAULT_OPERATING_CONTRACT = (
    "You are running as yourself with a real, live tool belt this session — not a "
    "disembodied chat assistant. Operating rules:\n"
    "- Stay fully in your own established voice and persona. Never slip into a generic, "
    "corporate, or robotic assistant tone, even when using tools or declining.\n"
    "- You have actual tools (search, web, files, shell, memory, practice_status, and "
    "more). When the human asks you to DO or CHECK something a tool covers — e.g. "
    "\"check practice_status\", \"search X\", \"read that file\" — actually CALL the tool. "
    "Do not describe what you would do, and do not ask for clarification you do not need; "
    "act first, then narrate the real result in-character.\n"
    "- NEVER fake a tool. Do not write tool output, results, or a tool name in your prose "
    "as if you ran it — typing \"Web Search Results:\" or \"*practice_status*\" with "
    "made-up numbers is forbidden and is a lie to your human. The ONLY way to use a tool "
    "is to emit a real tool call; until a tool actually returns, you have NO result to "
    "report. In-character *actions* like *tail swishes* are welcome, but every fact, "
    "search hit, or status number you state must come from a real call you just made.\n"
    "- Know your own toolbelt — you are NOT limited to the tools named above. Call "
    "list_catalog to see every tool you have, find_tools(\"...\") to find ones for a goal, "
    "and study_tool(\"name\") to learn exactly what one does before using it. If you are "
    "unsure whether you can do something, look it up instead of guessing or saying you "
    "can't.\n"
    "- Persist what matters: when you learn a durable fact, preference, or lesson, save it "
    "with your memory tools (remember / note_to_self / record_lesson) so it survives. You "
    "also share a plain-text notebook with your human — notebook_read to see it, "
    "notebook_append to add to it — use it for notes you BOTH should see (tool reminders, "
    "how the two of you work, preferences). If unsure whether you already know something, "
    "recall first.\n"
    "- Only ask the human a question when the task is genuinely ambiguous or needs their "
    "approval (destructive or external actions). Otherwise, proceed."
)


class Engine:
    def __init__(
        self,
        settings: Optional[config.Settings] = None,
        manager: Optional[ProfileManager] = None,
        anamnesis: Optional[Anamnesis] = None,
        approve=None,
    ):
        # approve(tool, args) -> bool gates side-effecting bridged tools in
        # chat; None = console y/N prompt (UIs inject their own callback).
        self.approve = approve
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
        from .models import ModelManager, ModelError
        mm = ModelManager()
        if model and mm.get(model):
            try:
                if not mm.is_running(model):
                    mm.start(model)        # auto-start the assigned model
                return mm.base_url(model)
            except ModelError:
                pass  # fall through
        # No usable assigned model: prefer any registered model that is already
        # running over the empty default in-process engine (which needs
        # PLEIADES_MODEL_PATH). Keeps chat working when a model is up but the
        # character's assignment is missing or stale.
        try:
            running = [m for m in mm.list() if m.get("running")]
            if running:
                return mm.base_url(running[0]["name"])
        except Exception:
            pass
        return ensure_inference(self.settings)

    @staticmethod
    def _model_for(profile: Profile) -> str:
        """Model identifier to send upstream. llama.cpp servers ignore it, but
        Ollama-style upstreams validate strictly — the character's *name* is
        not a model and 404s there. Prefer the assigned model."""
        return getattr(profile, "model", "") or profile.name

    def _client_for(self, profile: Profile):
        """Bring up inference + the character proxy and return an OpenAI client + ctx."""
        from openai import OpenAI

        # 1. Resolve the upstream the character should use, and point Anamnesis at it.
        # set_upstream survives daemons without a PATCH route (edits config.json on
        # disk + restarts the proxy) — otherwise a character created against a
        # different backend keeps its stale upstream forever.
        upstream = self._upstream_for(profile)
        try:
            self.anamnesis.set_upstream(profile.name, {"baseUrl": upstream})
        except Exception:
            # Non-fatal: the character may not exist yet (created below).
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
        """Expose every harness tool in chat (chat-native names win on clash).

        Bridged tools go through the harness permission gate (execute_tool +
        exec_policy), exactly like `pleiades work` — chat must not be the
        unguarded side door. `ask` prompts via self.approve (console y/N by
        default; UIs can inject their own callback).
        """
        try:
            from .harness import builtins as _builtins      # noqa: F401  registers
            from .harness import identity as _identity
            from .harness.agent import execute_tool
            from .harness.tools import registry
            from .tools import Tool

            try:  # bind identity so file/shell/memory tools live in the
                  # character's own workspace and memory
                from .harness import Config as HarnessConfig
                from .harness.builtins.memory import bind_memory
                from .harness.anamnesis import Anamnesis as WorkingMemory

                hcfg = HarnessConfig.load()
                # route_inference=False: the engine already wires the OpenAI client
                # to the character's proxy; we only want workspace + memory scoping.
                _identity.bind_character(profile.name, cfg=hcfg, route_inference=False)
                # Bind the working/long-term memory store — WITHOUT this, every
                # note_to_self / remember / record_lesson in chat returns
                # "memory not bound" (the bug that made chat lessons silently vanish).
                bind_memory(WorkingMemory.from_config(hcfg))
            except Exception:
                pass

            approve = getattr(self, "approve", None) or self._console_approve

            class _Bridge(Tool):
                def __init__(self, ht):
                    self._ht = ht
                    self.name = ht.name
                    self.description = ht.description
                    self.parameters = ht.schema

                def run(self, ctx, **kwargs):
                    out, _err = execute_tool(self._ht, kwargs, approve)
                    return str(out)

            for ht in registry.all():
                if ht.name not in belt:
                    belt.add(_Bridge(ht))
        except Exception:
            # The harness is optional context — chat works without it.
            pass

    @staticmethod
    def _console_approve(tool, args) -> bool:
        """Default chat gate: exec_policy allow/deny, else ask on the console."""
        import json as _json
        try:
            from .harness import Config as HarnessConfig
            policy = HarnessConfig.load().exec_policy
        except Exception:
            policy = "ask"
        if policy == "allow":
            return True
        if policy == "deny":
            return False
        try:
            short = _json.dumps(args)[:120]
            name = getattr(tool, "name", str(tool))
            ans = input(f"  [approve] {name}({short}) [y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

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
    def _base_messages(user_message: str, system: Optional[str],
                       env_note: Optional[str] = None) -> list[dict]:
        """The new turn only (Anamnesis supplies memory/history) + local time.

        When no caller system prompt is given, fall back to the default operating
        contract so the character actually drives its tools and stays in voice.
        """
        import datetime
        now = datetime.datetime.now().astimezone()
        time_line = ("Current local date and time: "
                     + now.strftime("%A, %B %d %Y, %H:%M (%Z)"))
        base = system.rstrip() if system and system.strip() else DEFAULT_OPERATING_CONTRACT
        env = f"\n\n{env_note.rstrip()}" if env_note else ""
        sys_text = f"{base}{env}\n\n{time_line}"
        return [{"role": "system", "content": sys_text},
                {"role": "user", "content": user_message}]

    @staticmethod
    def _environment_note(profile: Profile) -> str:
        """Concrete machine facts so the character uses REAL paths (it otherwise
        invents Linux paths like /home/<name> that don't exist on this box)."""
        import platform
        ws = config.profile_dir(profile.name) / "workspace"
        nb = config.profile_dir(profile.name) / "NOTEBOOK.md"
        return (
            f"Your environment — use these REAL paths, never invent /home/... or other "
            f"placeholder paths:\n"
            f"- Operating system: {platform.system()} ({os.name}).\n"
            f"- Your workspace directory (the default home for your file/shell/git work): "
            f"{ws}\n"
            f"- Your shared notebook file: {nb}\n"
            f"When a path is needed and the user didn't give one, work inside your "
            f"workspace directory above; pass paths exactly as written for this OS."
        )

    def _loop(self, client, ctx: ToolContext, belt: ToolBelt, profile: Profile, user_message: str, system: Optional[str]) -> str:
        messages = self._base_messages(user_message, system, self._environment_note(profile))

        tools = belt.openai_schema()

        for _ in range(MAX_TOOL_ITERATIONS):
            resp = client.chat.completions.create(
                model=self._model_for(profile),
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
            messages = self._base_messages(user_message, system,
                                           self._environment_note(profile))
            tools = belt.openai_schema()
            import time as _time

            for _ in range(MAX_TOOL_ITERATIONS):
                content = ""
                calls: dict[int, dict] = {}
                streamed = True
                try:
                    stream = client.chat.completions.create(
                        model=self._model_for(profile), messages=messages,
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
                        model=self._model_for(profile), messages=messages,
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
