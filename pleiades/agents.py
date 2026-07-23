"""agents.py — ephemeral background sub-agents spawned from chat.

Owner-facing feature: a chat-belt tool (see pleiades/tools/agents_tool.py) lets
the model fan out one or more focused, throwaway agents that run in the
background while the conversation continues, each shown live in the desktop
app's right panel (task/status/history + a status light) and stoppable at any
time.

This is deliberately NOT the harness's existing dispatch_subagent /
dispatch_subagents_parallel (pleiades/harness/subagent.py) — those block the
calling agent's own turn until they finish and run inline in the same
request. spawn_agents here returns immediately with agent ids; the caller
polls GET /api/agents for status, exactly like the "pleiades work" background
job pattern in webui/server.py (work_jobs dict + threading.Thread + on_event
appending to a list of events).

Hard safety properties, all enforced here rather than left to convention:

  * Depth-capped at 1, unconditionally: a spawned agent's own tool set never
    includes dispatch_subagent / dispatch_subagents_parallel (or any other
    tool tagged "meta"), so it cannot itself spawn further agents. See
    _SAFE_TOOLS() for exactly how that set is built and why it's restricted
    to registry-safe (read-only) tools rather than "everything minus the two
    dispatch tools" — a wide non-search tool count (>18) flips harness
    Agent._active_tools() into tool-search mode, and that method's own
    `... or sm` fallback re-exposes the FULL search+meta tag set (dispatch
    tools included) with real schemas if the agent's tool set doesn't
    intersect at all with {find_tools, list_catalog, call_tool}. Deliberately
    including that trio (which happens automatically since they're
    safe=True) keeps the intersection non-empty and the fallback dead.

  * Ephemeral memory, never a character's real Anamnesis store: each spawned
    agent gets its own _EphemeralMemory instance (pure in-process attributes,
    nothing touches disk or the vector cache) passed explicitly into
    harness.agent.Agent(memory=...). This bypasses Agent's default fallback
    to builtins.memory._store() — a process-wide global that a concurrently
    running chat turn may have bound to the REAL character's memory via
    Engine._bridge_harness_tools. Passing memory explicitly is the only way
    to guarantee no leak; simply not calling bind_memory() would not be
    enough, since _store() reads whatever the global happens to hold at the
    moment this Agent is constructed.

  * exec_policy="deny" on top of the safe-tools-only set (defense in depth —
    belt AND suspenders): even if a non-safe tool somehow ended up in the
    set, it could not run without an interactive approval prompt, which a
    background thread must never block on.

  * should_stop wired to both a 5-minute wall clock and the /stop endpoint,
    via harness.agent.Agent.run's new should_stop parameter (polled once per
    step, mirroring Engine.stream_events).

  * A concurrency cap (default 3) on top of the harness's own
    max_subagent_depth guard: a BoundedSemaphore around the part of each
    worker that actually calls Agent.run(), so spawning more tasks than the
    cap just queues the extras instead of rejecting or over-subscribing the
    one shared llama-server process.

Points the spawned agent's Tier.base_url at the CURRENTLY RUNNING model's own
llama-server (pleiades.models.ModelManager.base_url) rather than through
Engine/Anamnesis — same process/port, zero extra VRAM, and explicitly not a
character-scoped proxy call (no persistent identity).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MAX_CONCURRENT = 3
STEP_CEILING = 40            # lower than the parent chat's max_rounds (400)
WALL_CLOCK_SECONDS = 300      # 5 minutes
BLOCKED_TOOL_NAMES = {"dispatch_subagent", "dispatch_subagents_parallel"}


# --------------------------------------------------------------------------- #
# Ephemeral, in-memory-only stand-in for harness.anamnesis.Anamnesis
# --------------------------------------------------------------------------- #
class _EphemeralMemory:
    """A throwaway memory store implementing the same surface the agent loop
    and the memory tools (harness/builtins/memory.py) expect: working(),
    scratch_*, checkpoint_*, write/forget/recall/list. Everything lives in
    plain instance attributes and is discarded when the agent/thread ends —
    nothing here ever touches disk, the embedding cache, or any character's
    real Anamnesis store. This is what makes a spawned agent "explicitly NOT
    an Anamnesis character": it has no persistent identity and no memory
    that outlives its own run.
    """

    def __init__(self) -> None:
        self._scratch = ""
        self._sections: dict[str, str] = {}
        self._checkpoints: dict[str, str] = {}
        self._notes: dict[str, str] = {}

    def _render(self) -> str:
        parts = [self._scratch] if self._scratch else []
        parts += [f"## {k}\n{v}" for k, v in self._sections.items()]
        return "\n\n".join(parts)

    def working(self) -> str:
        return self._render()

    def scratch_append(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._scratch = (self._scratch + f"\n[{ts}] {text}").strip()

    def scratch_set(self, text: str) -> None:
        self._scratch = text

    def scratch_clear(self) -> None:
        self._scratch = ""
        self._sections.clear()

    def scratch_set_section(self, section: str, content: str) -> None:
        self._sections[section] = content

    def scratch_append_section(self, section: str, text: str) -> None:
        cur = self._sections.get(section, "")
        self._sections[section] = (cur + f"\n- {text}").strip()

    def checkpoint_save(self, label: str, extra: str = "") -> str:
        slug = (label or "").strip().replace(" ", "-").lower() or f"cp-{len(self._checkpoints)}"
        self._checkpoints[slug] = self._render()
        return slug

    def checkpoint_load(self, label: str) -> Optional[str]:
        body = self._checkpoints.get(label)
        if body is None:
            return None
        self._scratch = body
        return body

    def checkpoint_list(self) -> list[tuple[str, str]]:
        return [(k, "") for k in self._checkpoints]

    def write(self, name: str, body: str, *, kind: str = "note") -> str:
        slug = (name or "").strip().replace(" ", "-").lower() or f"note-{len(self._notes)}"
        self._notes[slug] = body
        return slug

    def forget(self, name: str) -> bool:
        return self._notes.pop(name, None) is not None

    def recall(self, query: str, k: int = 3) -> str:
        if not self._notes:
            return ""
        q = set((query or "").lower().split())
        scored = []
        for slug, body in self._notes.items():
            score = len(q & set(body.lower().split()))
            if score:
                scored.append((score, slug, body))
        scored.sort(key=lambda x: -x[0])
        return "\n\n".join(f"[{slug}] {body}" for _s, slug, body in scored[:k])

    def list(self) -> list[str]:
        return list(self._notes)


def _safe_tools(extra_blocked: "set[str] | None" = None) -> list:
    """The tool set given to every spawned background agent: every tool tagged
    safe=True (read-only, never gated by exec_policy), minus anything tagged
    "meta" and minus BLOCKED_TOOL_NAMES by name (belt-and-suspenders — see
    module docstring for why both checks matter). Includes find_tools/
    list_catalog/call_tool (they're safe=True themselves), so a big safe
    catalog is still reachable via tool-search without ever being able to
    discover or call the two dispatch tools (toolsearch.call_tool's own
    allow-list, built from this exact list, refuses them by name too).
    """
    from .harness import builtins as _builtins  # noqa: F401  registers tools
    from .harness.tools import registry

    blocked = BLOCKED_TOOL_NAMES | (extra_blocked or set())
    return [t for t in registry.all()
            if t.safe and "meta" not in t.tags and t.name not in blocked]


@dataclass
class _AgentRecord:
    id: str
    task: str
    character: str = ""
    model: str = ""
    status: str = "queued"     # queued | running | done | error | stopped
    events: list = field(default_factory=list)
    result: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    steps: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "task": self.task, "character": self.character,
            "model": self.model, "status": self.status, "events": self.events,
            "result": self.result, "error": self.error, "started": self.started,
            "finished": self.finished, "steps": self.steps,
        }


def agent_view(rec: dict, since: int = 0) -> dict:
    """Mirror webui/server.py's _job_view: full record minus the raw events
    list, plus events since a cursor and a running count. Used by
    GET /api/agents/{id}?since=."""
    d = {k: v for k, v in rec.items() if k != "events"}
    d["events"] = rec["events"][since:]
    d["event_count"] = len(rec["events"])
    return d


class AgentManager:
    """Registry + fan-out for ephemeral background agents.

    Modeled on webui/server.py's work_jobs dict pattern (module-level
    registry of serializable state, one background thread per unit of work,
    on_event appends structured events the UI polls for) but self-contained
    here so both the chat-belt tool and the webui endpoints share exactly one
    instance/registry (see get_manager()).
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT) -> None:
        self.max_concurrent = max_concurrent
        self._agents: dict[str, _AgentRecord] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max_concurrent)

    # -- read side ------------------------------------------------------ #
    def list(self) -> list[dict]:
        with self._lock:
            recs = list(self._agents.values())
        return [r.to_dict() for r in sorted(recs, key=lambda r: -r.started)]

    def get(self, agent_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._agents.get(agent_id)
        return rec.to_dict() if rec else None

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._agents.values()
                       if r.status in ("queued", "running"))

    # -- control ---------------------------------------------------------- #
    def stop(self, agent_id: str) -> bool:
        evt = self._stop_events.get(agent_id)
        if not evt:
            return False
        evt.set()
        return True

    # -- spawn -------------------------------------------------------------- #
    def spawn_agents(self, tasks: list[str], *, character: str = "",
                     model: str = "", settings=None,
                     agent_factory: Optional[Callable[..., Any]] = None) -> list[str]:
        """Fan out `tasks` (list of task strings) to concurrent, ephemeral
        background agents. Returns the new agents' ids immediately; callers
        poll list()/get() for status.

        model: explicit model name to run on; blank = whichever model is
               currently running (first alive entry from ModelManager.running()).
        agent_factory: injection point for tests — a callable
               (task, base_url, model_name, settings) -> object with
               .run(task, on_event=, should_stop=) -> AgentResult-like (an
               object with .answer/.steps attributes is enough). Real callers
               leave this as the default (self._build_agent, a real
               harness.Agent pointed at base_url).
        """
        tasks = [t for t in (tasks or []) if isinstance(t, str) and t.strip()]
        if not tasks:
            raise ValueError("tasks must be a non-empty list of non-empty strings")

        if agent_factory is not None:
            # Test/injection path: whatever runs the agent doesn't need a real
            # llama-server base_url, so skip the ModelManager round-trip that
            # would otherwise require a registered+running model to exist.
            base_url, model_name = "", (model or "test")
        else:
            base_url, model_name = self._resolve_model(model)

        ids = []
        for task in tasks:
            agent_id = uuid.uuid4().hex[:12]
            rec = _AgentRecord(id=agent_id, task=task, character=character,
                               model=model_name)
            with self._lock:
                self._agents[agent_id] = rec
            self._stop_events[agent_id] = threading.Event()
            ids.append(agent_id)
            threading.Thread(
                target=self._run_one, args=(agent_id, task, base_url, model_name,
                                            settings, agent_factory),
                daemon=True,
            ).start()
        return ids

    # -- internals ----------------------------------------------------------- #
    def _resolve_model(self, model: str) -> tuple[str, str]:
        from .models import ModelManager

        mm = ModelManager()
        if model:
            if not mm.get(model):
                raise ValueError(f"unknown model '{model}'")
            if not mm.is_running(model):
                raise ValueError(f"model '{model}' is not running")
            return mm.base_url(model), model
        running = [r for r in mm.running() if r.get("alive")]
        if not running:
            raise ValueError(
                "no model is currently running -- start one (or pass an "
                "explicit model name) before spawning agents"
            )
        name = running[0]["name"]
        return mm.base_url(name), name

    def _build_agent(self, task: str, base_url: str, model_name: str, settings):
        """Real (non-test) construction path: a harness Agent pointed at the
        given llama-server, safe-tools-only, deny exec_policy, ephemeral
        memory, step ceiling well below the parent chat's."""
        from .harness import builtins as _builtins  # noqa: F401  registers tools
        from .harness.agent import Agent
        from .harness.config import Config as HarnessConfig, Tier

        cfg = HarnessConfig.load() if settings is None else settings
        import dataclasses
        tier = Tier(backend="openai", model=model_name, base_url=base_url,
                    max_tokens=4096)
        cfg = dataclasses.replace(cfg, exec_policy="deny", max_rounds=STEP_CEILING,
                                  tiers={**cfg.tiers, "__spawned__": tier},
                                  default_tier="__spawned__")

        def _deny_approve(_tool, _args) -> bool:
            return False  # no interactive prompts; safe tools bypass this anyway

        agent = Agent(cfg, tools=_safe_tools(), tier_name="__spawned__",
                      depth=1, approve=_deny_approve, memory=_EphemeralMemory())
        return agent

    def _run_one(self, agent_id: str, task: str, base_url: str, model_name: str,
                settings, agent_factory) -> None:
        rec = self._agents[agent_id]
        stop_evt = self._stop_events[agent_id]
        deadline = time.time() + WALL_CLOCK_SECONDS

        def should_stop() -> bool:
            return stop_evt.is_set() or time.time() >= deadline

        def emit(kind: str, payload: Any) -> None:
            if kind == "tool_call":
                rec.events.append({"kind": "tool_call", "ts": time.time(),
                                   "name": getattr(payload, "name", str(payload))})
            elif kind == "tool_result":
                call, out, is_err = payload
                rec.events.append({"kind": "tool_result", "ts": time.time(),
                                   "name": getattr(call, "name", ""),
                                   "ok": not is_err, "output": (out or "")[:2000]})
            elif kind == "text":
                rec.events.append({"kind": "text", "ts": time.time(), "text": str(payload)})
            # "reflect"/"stopped"/"done"/etc. are internal bookkeeping only;
            # deliberately not surfaced as UI events (mirrors work_jobs, which
            # only stores tool_call/tool_result/text too).

        rec.status = "queued"
        self._sem.acquire()
        try:
            rec.status = "running"
            factory = agent_factory or self._build_agent
            agent = factory(task, base_url, model_name, settings)
            result = agent.run(task, on_event=emit, should_stop=should_stop)
            rec.result = getattr(result, "answer", str(result))
            rec.steps = getattr(result, "steps", 0)
            rec.status = "stopped" if stop_evt.is_set() else "done"
        except Exception as e:
            rec.status = "error"
            rec.error = str(e)
        finally:
            rec.finished = time.time()
            self._sem.release()


# --------------------------------------------------------------------------- #
# Process-wide singleton — the chat-belt tool and the webui endpoints must
# share exactly one registry (mirrors module-level `work_jobs` in server.py).
# --------------------------------------------------------------------------- #
_MANAGER: Optional[AgentManager] = None


def get_manager() -> AgentManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AgentManager()
    return _MANAGER


def spawn_agents(tasks: list[str], *, character: str = "", model: str = "") -> list[str]:
    """Module-level convenience wrapper around get_manager().spawn_agents —
    the one the chat-belt tool calls."""
    return get_manager().spawn_agents(tasks, character=character, model=model)
