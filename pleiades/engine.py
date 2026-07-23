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
from typing import Callable, Optional, Union

from . import config
from .anamnesis import Anamnesis
from .inference import ensure_inference
from .profiles import Profile, ProfileManager
from .tools import ToolBelt, ToolContext, build_default_belt

# --- Cloud-provider key rotation (OpenRouter / Ollama Cloud) ---------------- #
# Process-local: resets on restart, which is fine — a fresh attempt on restart
# is the right behavior anyway. Anamnesis's set_upstream() restarts a
# character's proxy daemon whenever the upstream actually changes, so picking
# is "sticky" (keeps the last-used key while it's healthy) to avoid restarting
# the proxy on every single turn when nothing needs to change.
_KEY_COOLDOWN: dict[str, float] = {}   # raw key -> unix ts when it's usable again
_LAST_KEY: dict[str, str] = {}         # "provider:profile" -> last key picked

# The openai SDK's own default timeout is very long (read: effectively
# unbounded for a chat call) and stacks with its own retry/backoff. A stalled
# upstream (model wedged, Anamnesis proxy down, network hiccup) then blocks
# inside client.chat.completions.create() for ages before anything raises —
# "loading forever" in the UI even though server.py's NDJSON generator (and
# next.html's talkSend(), which already renders {"type":"error"} correctly)
# are both ready to surface the failure the moment it actually arrives.
#
# 120s was the first guess; a live trace on this machine showed it's too
# tight for a real (not stalled) chat turn — _client_for() bridges the FULL
# harness tool belt (60+ tools) into every chat message, which alone is a
# ~20k-token system prompt. On a 35B-A3B MoE model with experts offloaded to
# CPU (measured ~120 tok/s prompt-eval, ~20 tok/s generation here), that's
# 150-200s of prefill before generation even starts — a genuinely slow but
# *working* request, not a hang. 120s would abort it with a spurious timeout
# error. Generous but still bounded: a real stall (proxy down, wedged
# process) should never take 10 minutes to surface either way.
_UPSTREAM_TIMEOUT = 600.0


def _provider_for(model: str) -> Optional[str]:
    if model.startswith("openrouter:"):
        return "openrouter"
    if model.startswith("ollama-cloud:"):
        return "ollama-cloud"
    return None


def _pick_key(provider: str, profile_name: str, keys: list) -> str:
    """Pick a usable key for `provider`/`profile_name` from `keys`.

    Sticks with the last key used for this profile while it's still healthy.
    Skips keys currently in cooldown (recent 429/quota error). If every key is
    cooling down, returns whichever frees up soonest — better to retry too
    early than to refuse outright.
    """
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    import time as _time
    now = _time.time()
    slot = f"{provider}:{profile_name}"
    last = _LAST_KEY.get(slot)
    if last in keys and _KEY_COOLDOWN.get(last, 0) <= now:
        return last
    healthy = [k for k in keys if _KEY_COOLDOWN.get(k, 0) <= now]
    chosen = healthy[0] if healthy else min(keys, key=lambda k: _KEY_COOLDOWN.get(k, 0))
    _LAST_KEY[slot] = chosen
    return chosen


def _mark_cooldown(key: str, seconds: float) -> None:
    if key:
        import time as _time
        _KEY_COOLDOWN[key] = _time.time() + seconds


def _cooldown_for(status: Optional[int]) -> float:
    """How long to bench a key after an error, by HTTP status."""
    if status == 429:
        return 60.0          # rate limit — usually clears within a minute
    if status in (402, 403):
        return 6 * 3600.0    # quota exhausted / forbidden — won't clear itself soon
    return 30.0


def _retryable_key_error_status(exc: Exception) -> Optional[int]:
    """HTTP status if `exc` looks like a rate-limit/quota error worth rotating
    keys for, else None."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    return status if status in (429, 402, 403) else None


_REFLECTION = (
    "\n\n[Self-check after {n} steps] Before continuing, genuinely evaluate — this is "
    "not a formality:\n"
    "- Progress: what concrete, verifiable progress have you made since the last check? "
    "If you can't point to any, that's a real signal, not a reason to push harder the "
    "same way.\n"
    "- Looping: are you repeating the same actions/tool calls without new information? "
    "If so, you must change approach now — a different tool, a different angle, breaking "
    "the task into smaller pieces, or gathering missing info — before your next action.\n"
    "- Blocked/impossible: if you've made several genuinely different attempts and the "
    "goal is still not achievable (missing access or permissions, a contradiction in the "
    "request, a tool or resource that doesn't exist), STOP here. Say plainly that you're "
    "blocked, explain what you tried, and end your turn — do not keep repeating attempts "
    "that already failed.\n"
    "Otherwise, take one concrete next step toward the goal."
)


def _synthetic_tool_round(name: str, content: str, call_id: str) -> list[dict]:
    """A fabricated assistant tool_calls message + its role:"tool" result —
    used to splice a machine-generated notice (the self-reflection check
    above, a background-task-completion notice) into `messages` WITHOUT a
    role:"user" entry.

    Why not role:"user": Anamnesis's proxy (proxy.js) persists "the new user
    turn" by reversing the messages array and taking the first role=='user'
    entry it finds. A role:"user" injection is indistinguishable from real
    user speech to that scan and gets written into the character's
    long-term memory as if the user had actually said it — confirmed live by
    reading proxy.js's dedup logic (see fix/anamnesis-reflection-injection
    branch notes). role:"tool" messages are invisible to that scan (it
    filters on role=='user' only) and are never persisted or extracted by
    Anamnesis at all — history.js only ever inserts 'user'/'assistant' rows,
    and the extractor only ever reads rows already inserted that way — so a
    synthetic tool round is safe from both the dedup bug and any secondary
    leak through the extraction pipeline.

    The model never actually emitted this tool_calls entry; it's a
    fabricated round the engine constructs so the notice arrives through a
    channel the model already knows how to read (a tool result) instead of
    as fake user speech. engine.py only ever speaks the OpenAI wire format
    (see _client_for), so this shape is unconditional here — contrast with
    harness/agent.py's twin copy of this helper, which is backend-aware
    because that loop also supports the Anthropic backend.
    """
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


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
    "approval (destructive or external actions). Otherwise, proceed.\n"
    "- For a real multi-step job, use create_task/update_task/list_tasks (find_tools "
    "\"task list\" if you don't see them yet) to keep a live plan instead of only "
    "narrating steps -- mark each in_progress, then completed, as you actually do it.\n"
    "- You may occasionally see a tool_calls entry for system_reflection or "
    "system_notice in your own history that you don't remember emitting, followed by "
    "its result. That's expected: the runtime inserts these itself (a periodic "
    "self-check, or a heads-up that something running in the background finished) "
    "using the tool-result channel so it reaches you cleanly -- you did not call it "
    "and were not supposed to. Treat its content as a passive note from the runtime, "
    "not as your own prior action, and don't imitate it by calling those names yourself."
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
    def _resolve_upstream(self, profile: Profile) -> dict:
        """Resolve a character's brain -> upstream {baseUrl[, apiKey, model]}.

        profile.model conventions:
          ""                        -> default local in-process engine
          "<registered name>"       -> that local GGUF model's server
          "openrouter:<model_id>"   -> OpenRouter (OpenAI-compatible cloud)
          "ollama-cloud:<model_id>" -> Ollama Cloud (OpenAI-compatible)
        Cloud brains still flow THROUGH Anamnesis, so memory injection applies.
        """
        model = getattr(profile, "model", "") or ""
        s = self.settings
        if model.startswith("openrouter:"):
            key = _pick_key("openrouter", profile.name, s.openrouter_api_keys)
            return {"baseUrl": "https://openrouter.ai/api/v1",
                    "apiKey": key, "model": model.split(":", 1)[1]}
        if model.startswith("ollama-cloud:"):
            key = _pick_key("ollama-cloud", profile.name, s.ollama_cloud_api_keys)
            return {"baseUrl": (s.ollama_cloud_url or "https://ollama.com/v1"),
                    "apiKey": key, "model": model.split(":", 1)[1]}
        from .models import ModelManager, ModelError
        mm = ModelManager()
        if model and mm.get(model):
            try:
                if not mm.is_running(model):
                    mm.start(model)
                return {"baseUrl": mm.base_url(model), "model": profile.name}
            except ModelError:
                pass
        try:
            running = [m for m in mm.list() if m.get("running")]
            if running:
                return {"baseUrl": mm.base_url(running[0]["name"]), "model": profile.name}
        except Exception:
            pass
        # ponytail: legacy single-model bootstrap (PLEIADES_MODEL_PATH) used to launch
        # through the dumb python-only InferenceServer — no MoE offload, no flash-attn,
        # no kv-cache quant, no thread tuning. Register it with ModelManager instead so
        # it gets the exact same autofit/native-runtime treatment as every other model:
        # one optimized launch path, not two. ensure_inference now only backs `pleiades
        # serve` (manual foreground debug command) — ceiling: raise it the same way if
        # that command ever needs the smarter path too.
        if s.model_path:
            name = "default"
            existing = mm.get(name)
            if not existing or existing.get("path") != s.model_path:
                mm.add(name, s.model_path, n_ctx=s.n_ctx, chat_format=s.chat_format)
            mm.start(name)
            return {"baseUrl": mm.base_url(name), "model": profile.name}
        return {"baseUrl": ensure_inference(s), "model": profile.name}

    def _model_for(self, profile: Profile) -> str:
        """Model id sent upstream (cloud validates it; llama.cpp ignores it)."""
        return (self._resolve_upstream(profile).get("model")
                or getattr(profile, "model", "") or profile.name)

    def _point_upstream(self, profile: Profile) -> str:
        """Resolve the upstream the character should use, point Anamnesis at it,
        and return the character's proxy base URL.

        set_upstream() short-circuits (no restart) when the upstream hasn't
        actually changed, and survives daemons without a PATCH route (edits
        config.json on disk + restarts the proxy) — otherwise a character
        created against a different backend keeps its stale upstream forever.
        """
        up = self._resolve_upstream(profile)
        upstream = {"baseUrl": up["baseUrl"]}
        if up.get("apiKey"):
            upstream["apiKey"] = up["apiKey"]
        if up.get("model"):
            upstream["model"] = up["model"]
        try:
            self.anamnesis.set_upstream(profile.name, upstream)
        except Exception:
            pass
        return self.anamnesis.ensure_running(profile.name, upstream=upstream)

    def _session_id_for(self, profile: Profile) -> str:
        """Stable Anamnesis session key for this character's one continuous
        conversational thread.

        Every real conversational surface (webui, CLI, the Discord bot — all
        of them route through Engine) sends this same id, so they deliberately
        share Anamnesis's history/recency/scene recall: that's what makes a
        character feel like one continuous relationship instead of amnesiac
        per-window resets. Previously this was an ACCIDENT: every OpenAI
        client used the literal api_key="pleiades", and Anamnesis derives its
        session bucket from a hash of the bearer token — so everything
        collapsed onto the same bucket only because the token never varied.
        Fragile (any code path that ever passed a different key would silently
        fork onto a new, empty-feeling session) and it couldn't be told apart
        from noise.

        Non-conversational writers (cron health checks / watchdogs) must send
        their own distinct X-Session-Id so their pings don't eat slots in
        Anamnesis's small recencyTurns window mid-conversation — see
        ~/.openclaw/<character>/skills/context_watchdog.
        """
        return f"pleiades:{profile.name}"

    def _new_client(self, profile: Profile):
        """Re-resolve upstream (picks up any key rotation from a cooldown) and
        hand back a fresh client. Cheap when the key hasn't actually changed —
        _point_upstream's set_upstream() no-ops in that case."""
        from openai import OpenAI
        return OpenAI(base_url=self._point_upstream(profile), api_key="pleiades",
                      timeout=_UPSTREAM_TIMEOUT,
                      default_headers={"X-Session-Id": self._session_id_for(profile)})

    def _handle_upstream_error(self, profile: Profile, exc: Exception) -> bool:
        """On a rate-limit/quota error from a multi-key cloud provider, bench
        the key just used so the next pick skips it. Returns True if another
        key is available to retry with right now."""
        model = getattr(profile, "model", "") or ""
        provider = _provider_for(model)
        if not provider:
            return False
        status = _retryable_key_error_status(exc)
        if status is None:
            return False
        keys = (self.settings.openrouter_api_keys if provider == "openrouter"
                else self.settings.ollama_cloud_api_keys)
        slot = f"{provider}:{profile.name}"
        last = _LAST_KEY.get(slot)
        if last:
            _mark_cooldown(last, _cooldown_for(status))
            _LAST_KEY.pop(slot, None)
        import time as _time
        now = _time.time()
        return any(_KEY_COOLDOWN.get(k, 0) <= now for k in keys)

    def _chat_create(self, client, profile: Profile, **kwargs):
        """client.chat.completions.create with auto key-rotation: on a 429/402/403
        from a multi-key cloud provider, bench that key and retry with a fresh
        client bound to the next healthy one. Returns (result, client) — the
        client that ultimately succeeded, so callers keep using it for the rest
        of the turn."""
        attempts = 0
        while True:
            try:
                return client.chat.completions.create(**kwargs), client
            except Exception as exc:
                attempts += 1
                if attempts > 4 or not self._handle_upstream_error(profile, exc):
                    raise
                client = self._new_client(profile)

    def _client_for(self, profile: Profile):
        """Bring up inference + the character proxy and return an OpenAI client + ctx."""
        from openai import OpenAI

        proxy_url = self._point_upstream(profile)
        client = OpenAI(base_url=proxy_url, api_key="pleiades", timeout=_UPSTREAM_TIMEOUT,
                        default_headers={"X-Session-Id": self._session_id_for(profile)})

        # 3. Tools, scoped to this character — the chat IS the agent, so the
        # harness's lazy tool-search trio (find_tools/list_catalog/call_tool;
        # see _bridge_harness_tools) is bridged in alongside the chat-native
        # tools, giving on-demand reach into the full harness registry
        # (files, shell, git, web, browser, memory, subagents, ...) without
        # paying for 60+ full schemas on every turn.
        vault = self.manager.open_vault(profile.name)
        ctx = ToolContext(profile=profile, vault=vault, settings=self.settings)
        belt = build_default_belt(ctx)
        self._bridge_harness_tools(belt, profile)
        return client, ctx, belt

    def _bridge_harness_tools(self, belt: ToolBelt, profile: Profile) -> None:
        """Expose the harness's lazy tool-search trio in chat, not its full
        registry.

        This used to loop `registry.all()` and bridge a full schema for
        every harness tool (files, shell, git, web, browser, memory,
        subagents, ... 60+) into the chat belt on EVERY turn — a ~20k-token
        system prompt before the model produced a single reply token, even
        for "hi" (see _UPSTREAM_TIMEOUT's comment above for the measured
        prefill cost this caused). The harness already solved exactly this
        problem for `pleiades work` via pleiades/harness/toolsearch.py:
        find_tools/list_catalog discover tools on demand, call_tool invokes
        them by name. Bridging those 3 small dispatch tools instead of the
        raw registry gives chat the same reach for ~3 schemas instead of 60+,
        loaded lazily — the model calls find_tools only when a turn actually
        needs a capability beyond the chat-native belt.

        Bridged calls go through the harness permission gate (execute_tool +
        exec_policy / toolsearch.bind_dispatch), exactly like `pleiades work`
        — chat must not be the unguarded side door. `ask` prompts via
        self.approve (console y/N by default; UIs can inject their own
        callback).
        """
        try:
            from .harness import builtins as _builtins      # noqa: F401  registers
            from .harness import identity as _identity
            from .harness import toolsearch
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
            # find_tools/list_catalog/call_tool read this module-level binding
            # to apply the same approval policy call_tool dispatches into.
            #
            # agents_enabled gate (per-character, see profiles.Profile):
            # dispatch_subagent/dispatch_subagents_parallel (harness/
            # subagent.py) are tagged "meta", not "search", so registry.all()
            # already makes them reachable through call_tool/find_tools
            # whenever bind_dispatch's allow-list is unrestricted (None) --
            # that's the existing "backdoor" the owner flagged: the raw
            # dispatch capability was always live via chat's tool-search
            # bridge regardless of any UI toggle. When agents are OFF for
            # this character we now pass an explicit allow-list that is the
            # full registry MINUS those two tool names, so find_tools can't
            # surface them and call_tool refuses them outright ("not
            # available to this agent"). When ON, we leave the allow-list
            # unrestricted (None) exactly as before -- spawn_agents (the new,
            # managed/depth-capped/concurrency-capped path; see
            # pleiades/agents.py) is a separate, explicitly-bridged tool
            # added by build_default_belt, not this fallback.
            if getattr(profile, "agents_enabled", False):
                toolsearch.bind_dispatch(approve)
            else:
                blocked = {"dispatch_subagent", "dispatch_subagents_parallel"}
                allowed = {t.name for t in registry.all() if t.name not in blocked}
                toolsearch.bind_dispatch(approve, allowed)

            class _Bridge(Tool):
                def __init__(self, ht):
                    self._ht = ht
                    self.name = ht.name
                    self.description = ht.description
                    self.parameters = ht.schema
                    # Propagate safe=True so find_tools/list_catalog/call_tool
                    # (all dispatch plumbing, none unsafe by itself) skip
                    # ToolBelt.dispatch's own ask-gate — without this every
                    # lookup nags for approval under exec_policy=ask, defeating
                    # "lazy/on-demand" before the model even reaches a real
                    # tool. The actual target call_tool dispatches to still
                    # gets gated correctly: execute_tool() inside run() below
                    # checks THAT tool's own safe flag via toolsearch's bound
                    # approve callback (see toolsearch.bind_dispatch above).
                    self.safe = ht.safe

                def run(self, ctx, **kwargs):
                    out, _err = execute_tool(self._ht, kwargs, approve)
                    return str(out)

            # tags=["search"] is exactly the 3 toolsearch.py dispatch tools —
            # not the 60+ tools they discover/invoke on the model's behalf.
            for ht in registry.select(tags=["search"]):
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
                       env_note: Optional[str] = None,
                       history: Optional[list[dict]] = None) -> list[dict]:
        """The new turn (Anamnesis supplies deep memory/history) + local time.

        `history` is an optional short tail of recent real turns (see
        chats.recent_messages) resent alongside the new message so
        Anamnesis's own recency-window mechanism has something to work with
        (see the docstring on chats.recent_messages for why this matters).
        It is NOT a substitute for Anamnesis's memory — just enough raw
        context for pronoun/short-follow-up continuity within the current
        conversation.

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
        messages = [{"role": "system", "content": sys_text}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

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

        interval = max(0, getattr(self.settings, "eval_interval", 40))
        ceiling = max(1, getattr(self.settings, "max_rounds", 400))
        round_i = 0
        last_text = ""
        while True:
            if interval and round_i and round_i % interval == 0:
                messages.extend(_synthetic_tool_round(
                    "system_reflection", _REFLECTION.format(n=round_i), f"reflect_{round_i}"))
            if round_i >= ceiling:
                # Absolute safety net: the self-check above gets ~ceiling/interval
                # chances to resolve a loop/blocker first; this only fires if it never did.
                return last_text or (
                    f"(stopped: hit the {ceiling}-round safety ceiling without a final "
                    f"answer — the self-check every {interval} rounds never resolved it.)"
                )
            round_i += 1
            resp, client = self._chat_create(
                client, profile,
                model=self._model_for(profile),
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
            )
            msg = resp.choices[0].message
            if msg.content:
                last_text = msg.content
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
                result = belt.dispatch(tc.function.name, tc.function.arguments, ctx,
                                       approve=self.approve)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

    def stream_events(self, profile: Union[str, Profile], user_message: str,
                      *, system: Optional[str] = None,
                      history: Optional[list[dict]] = None,
                      should_stop: Optional[Callable[[], bool]] = None,
                      poll_injections: Optional[Callable[[], list[str]]] = None):
        """The full turn as a stream of structured events.

        Yields dicts, in order:
            {"type": "token", "text": str}                       assistant text
            {"type": "tool_call", "name": str, "args": str}
            {"type": "tool_result", "name": str, "output": str, "ok": bool}
            {"type": "user_injected", "text": str}     a poll_injections() message woven in
            {"type": "done", "tokens": int, "seconds": float, "tps": float}
            {"type": "stopped"}                      should_stop() fired early
        Tokens stream live from llama.cpp; each turn runs exactly once, so
        Anamnesis stores exactly what the caller sees. Falls back to a
        non-streamed request if the backend rejects stream+tools.

        `should_stop` is polled between rounds and between streamed chunks so
        an "emergency stop" button can cut a run short; it can't interrupt a
        single blocking tool call already in flight (e.g. a slow browser
        action) — ponytail: add cooperative cancellation to tools if that
        turns out to matter in practice.

        `poll_injections`, if given, returns any mid-turn user messages queued
        since it was last called (e.g. webui/server.py's chats_interject
        endpoint feeding a per-chat inbox Queue) — the substrate for "message
        an in-flight turn like a Claude Code agent" without aborting/restarting
        it. Re-prefilling the whole context to splice a message mid-token-
        generation costs 150-200s on the daily-driver model (see
        _UPSTREAM_TIMEOUT above), so this is ONLY ever drained and spliced at
        safe boundaries: the top of a round (before the next _chat_create) and
        between individual tool dispatches within a round. Even when polled
        mid-round (to catch a message that arrived during a slow tool call
        with low latency), anything collected is held in `pending_injections`
        and only actually appended to `messages` once ALL of that round's
        tool_call ids already have their tool results — splicing a "role":
        "user" message between an assistant's tool_calls message and its
        results would corrupt the transcript for chat templates that expect
        that pairing to be contiguous.

        This queue is role:"user" deliberately and only ever carries genuine
        user-authored text today — its one caller (webui/server.py's
        chats_interject endpoint) is only ever reached by a human via the
        desktop app's Composer, never by anything automated (confirmed: no
        background/scheduled/async mechanism in this repo posts to
        /interject or touches chat_inboxes as of this writing). That's the
        correct shape for real user speech — Anamnesis SHOULD persist it as
        a real user turn. If a future feature ever delivers a
        machine-generated notice (e.g. "your background tool call
        finished") through this same pending_injections channel, it must
        NOT be appended as role:"user" — route it through
        _synthetic_tool_round() instead, same as the self-check above, or
        it will reproduce the exact Anamnesis memory-pollution bug that
        mechanism was built to avoid (see fix/anamnesis-reflection-injection).
        """
        if isinstance(profile, str):
            profile = self.manager.get(profile)
        client, ctx, belt = self._client_for(profile)
        n_tokens = 0
        t_first = t_last = None
        try:
            messages = self._base_messages(user_message, system,
                                           self._environment_note(profile),
                                           history)
            tools = belt.openai_schema()
            import time as _time

            interval = max(0, getattr(self.settings, "eval_interval", 40))
            ceiling = max(1, getattr(self.settings, "max_rounds", 400))
            round_i = 0
            pending_injections: list[str] = []
            while True:
                if should_stop and should_stop():
                    yield {"type": "stopped"}
                    return
                # Safe boundary #1: top of the round, before the next
                # _chat_create. Whatever's in `pending_injections` (queued
                # mid-round below) plus anything newly available is applied
                # HERE, never mid-round — by construction, if we reached this
                # point via the bottom of the loop (not a fresh call), the
                # PREVIOUS round's tool_calls/tool results are already fully
                # paired in `messages`, so splicing a user message now cannot
                # land between them.
                if poll_injections:
                    pending_injections.extend(poll_injections())
                for _inj in pending_injections:
                    messages.append({"role": "user",
                                     "content": f"[message from the user, mid-task] {_inj}"})
                    yield {"type": "user_injected", "text": _inj}
                pending_injections = []
                if interval and round_i and round_i % interval == 0:
                    messages.extend(_synthetic_tool_round(
                        "system_reflection", _REFLECTION.format(n=round_i), f"reflect_{round_i}"))
                    yield {"type": "reasoning", "text": "[self-check] evaluating progress and checking for loops…"}
                if round_i >= ceiling:
                    # Absolute safety net: the self-check above gets ~ceiling/interval
                    # chances to resolve a loop/blocker first; this only fires if it never did.
                    msg_text = (
                        f"(stopped: hit the {ceiling}-round safety ceiling without a final "
                        f"answer — the self-check every {interval} rounds never resolved it.)"
                    )
                    yield {"type": "token", "text": msg_text}
                    n_tokens += max(1, len(msg_text) // 4)
                    now = _time.time()
                    t_first = t_first or now
                    t_last = now
                    break
                round_i += 1
                content = ""
                calls: dict[int, dict] = {}
                streamed = True
                try:
                    stream, client = self._chat_create(
                        client, profile,
                        model=self._model_for(profile), messages=messages,
                        tools=tools or None,
                        tool_choice="auto" if tools else None, stream=True,
                    )
                    for chunk in stream:
                        if should_stop and should_stop():
                            close = getattr(stream, "close", None)
                            if close:
                                try:
                                    close()
                                except Exception:
                                    pass
                            yield {"type": "stopped"}
                            return
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is None:
                            continue
                        rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        if rc:
                            t_first = t_first or _time.time()
                            yield {"type": "reasoning", "text": rc}
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
                    resp, client = self._chat_create(
                        client, profile,
                        model=self._model_for(profile), messages=messages,
                        tools=tools or None,
                        tool_choice="auto" if tools else None,
                    )
                    msg = resp.choices[0].message
                    content = msg.content or ""
                    rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                    if rc:
                        yield {"type": "reasoning", "text": rc}
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
                    # Safe boundary #2: between individual tool dispatches.
                    # Only COLLECT here -- never splice into `messages` yet,
                    # since we're sitting between the assistant's tool_calls
                    # message (already appended above) and this round's tool
                    # results (still being appended below). Anything collected
                    # is applied at the top of the NEXT round instead, once
                    # every tool_call id in `ordered` has its result appended.
                    if poll_injections:
                        pending_injections.extend(poll_injections())
                    yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
                    result = belt.dispatch(c["name"], c["args"], ctx, approve=self.approve)
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

    def stream(self, profile: Union[str, Profile], user_message: str, *, system: Optional[str] = None,
              history: Optional[list[dict]] = None):
        """Text-only view of stream_events (CLI REPL / Discord)."""
        for evt in self.stream_events(profile, user_message, system=system, history=history):
            if evt["type"] == "token":
                yield evt["text"]
