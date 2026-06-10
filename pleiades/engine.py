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

        # 3. Tools, scoped to this character.
        vault = self.manager.open_vault(profile.name)
        ctx = ToolContext(profile=profile, vault=vault, settings=self.settings)
        belt = build_default_belt(ctx)
        return client, ctx, belt

    # -- main entry point --------------------------------------------------- #
    def run(self, profile: Union[str, Profile], user_message: str, *, system: Optional[str] = None) -> str:
        if isinstance(profile, str):
            profile = self.manager.get(profile)

        client, ctx, belt = self._client_for(profile)
        try:
            return self._loop(client, ctx, belt, profile, user_message, system)
        finally:
            ctx.close()

    def _loop(self, client, ctx: ToolContext, belt: ToolBelt, profile: Profile, user_message: str, system: Optional[str]) -> str:
        # Send only the new turn — Anamnesis supplies system/memory/history.
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

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

    def stream(self, profile: Union[str, Profile], user_message: str, *, system: Optional[str] = None):
        """Yield the final assistant message (after any tool calls resolve).

        Yields text chunks. Each turn runs exactly once (no re-issued streaming
        request), so Anamnesis stores exactly what the caller sees.
        """
        if isinstance(profile, str):
            profile = self.manager.get(profile)
        client, ctx, belt = self._client_for(profile)
        try:
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_message})
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
                    # Yield the reply we already have. Re-issuing it as a true stream
                    # would run inference twice and store a duplicate (and possibly
                    # different) turn in Anamnesis memory.
                    if msg.content:
                        yield msg.content
                    return
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    result = belt.dispatch(tc.function.name, tc.function.arguments, ctx)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        finally:
            ctx.close()
