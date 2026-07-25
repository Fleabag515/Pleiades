"""Shared primitives between the two agent loops (engine.py's chat path and
harness/agent.py's work path). Currently just the self-reflection prompt and
the synthetic-tool-round helper -- hoisted here because they were byte-for-
byte duplicated between the two, not because this is meant to grow into a
merged loop (that's a separate, larger project -- see HANDOFF open item #7).
"""

from __future__ import annotations

REFLECTION = (
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


def synthetic_tool_round(backend: str, name: str, content: str, call_id: str) -> list[dict]:
    """A fabricated assistant tool_use/tool_calls message + its result, in
    whichever wire shape `backend` expects. Used to splice a machine-
    generated notice (a self-reflection check, a background-task-completion
    notice, a failed-model-call retry notice) into `messages` WITHOUT a
    role:"user" entry.

    Why not role:"user": when an agent loop is bound to a character
    (identity.bind_character / engine.py's Anamnesis routing), requests are
    routed through that character's Anamnesis proxy (memory injection +
    persistence). Anamnesis persists "the new user turn" by reversing
    `messages` and taking the first role=='user' entry it finds (proxy.js)
    -- a role:"user" injection here is indistinguishable from real user
    speech to that scan and gets written into the character's long-term
    memory as if the user had said it. role:"tool" (openai/ollama) and a
    tool_result content block (anthropic) are both invisible to that scan,
    and Anamnesis never persists or extracts either shape (history.js only
    ever inserts 'user'/'assistant' rows) -- confirmed against Anamnesis's
    src/proxy.js + src/extractor.js on fix/anamnesis-reflection-injection.

    The "anthropic" backend never routes through Anamnesis (it talks to the
    Anthropic API directly), so that branch exists for wire-format
    correctness (Anthropic has no "tool" role; a real tool result there is a
    role:"user" message with a tool_result content block), not because that
    backend can hit the Anamnesis bug. engine.py only ever speaks the OpenAI
    wire format, so it always calls this with backend="openai".
    """
    if backend == "anthropic":
        return [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": call_id, "name": name, "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content},
            ]},
        ]
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]
