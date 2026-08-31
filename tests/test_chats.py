"""Chat sessions + unified chat/agent wiring."""

from pleiades import chats
from pleiades.chats import _REPLAY_MAX_CALLS
from pleiades.engine import Engine
from pleiades.profiles import Profile
from pleiades.tools import ToolBelt, ToolContext


def test_chat_crud_roundtrip():
    c = chats.create("zoe")
    assert c["id"] and c["character"] == "zoe"
    chats.append_turn(c["id"], "hello world", [{"t": "text", "text": "hi"}],
                      {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    loaded = chats.load(c["id"])
    assert loaded["title"].startswith("hello world")
    assert loaded["messages"][0]["role"] == "user"
    assert loaded["messages"][1]["items"][0]["text"] == "hi"
    listing = chats.list_chats()
    assert any(x["id"] == c["id"] for x in listing)
    assert chats.delete(c["id"])
    assert not any(x["id"] == c["id"] for x in chats.list_chats())


def test_tool_usage_counts_completed_calls_and_scopes_by_character():
    # Unique-per-test character names -- PLEIADES_HOME/chats is a shared temp
    # dir for the whole test session (see conftest.py), so reusing "zoe"
    # across tool_usage tests would leak counts between them.
    c1 = chats.create("zoe-tu-count")
    chats.append_turn(c1["id"], "search something", [
        {"t": "tool", "name": "search", "args": "{}", "output": "...",
         "ok": True, "ts": 1000.0},
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})
    c2 = chats.create("zoe-tu-count")
    chats.append_turn(c2["id"], "search again", [
        {"t": "tool", "name": "search", "args": "{}", "output": "...",
         "ok": True, "ts": 2000.0},
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})
    c3 = chats.create("mark-tu-count")
    chats.append_turn(c3["id"], "vault get", [
        {"t": "tool", "name": "vault", "args": "{}", "output": "...",
         "ok": True, "ts": 3000.0},
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})

    zoe_usage = chats.tool_usage("zoe-tu-count")
    assert zoe_usage["search"]["count"] == 2
    assert zoe_usage["search"]["last_used"] == 2000.0
    assert "vault" not in zoe_usage  # scoped to zoe, not mark's tool use

    mark_usage = chats.tool_usage("mark-tu-count")
    assert mark_usage["vault"]["count"] == 1

    everyone = chats.tool_usage()
    assert everyone["search"]["count"] >= 2
    assert everyone["vault"]["count"] >= 1

    chats.delete(c1["id"])
    chats.delete(c2["id"])
    chats.delete(c3["id"])


def test_tool_usage_leaves_last_used_null_without_a_real_timestamp():
    """Historical transcripts written before per-call "ts" existed still
    count toward use_count, but must never get a fabricated last_used."""
    c = chats.create("zoe-tu-nots")
    chats.append_turn(c["id"], "old school call", [
        {"t": "tool", "name": "hardware", "args": "{}", "output": "...",
         "ok": True},  # no "ts" -- pre-tracking shape
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})

    usage = chats.tool_usage("zoe-tu-nots")
    assert usage["hardware"]["count"] == 1
    assert usage["hardware"]["last_used"] is None

    chats.delete(c["id"])


def test_tool_usage_ignores_pending_and_non_tool_items():
    c = chats.create("zoe-tu-pending")
    chats.append_turn(c["id"], "in flight", [
        {"t": "text", "text": "hi"},
        {"t": "tool", "name": "search", "args": "{}", "output": None,
         "ok": None, "ts": 5000.0},  # never completed -- must not count
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})

    assert chats.tool_usage("zoe-tu-pending") == {}

    chats.delete(c["id"])


def test_chat_bad_id_rejected():
    import pytest

    with pytest.raises((ValueError, FileNotFoundError)):
        chats.load("../escape")


def test_base_messages_inject_local_time():
    msgs = Engine._base_messages("hi", None)
    assert msgs[0]["role"] == "system"
    assert "Current local date and time" in msgs[0]["content"]
    msgs2 = Engine._base_messages("hi", "You are Zoe.")
    assert msgs2[0]["content"].startswith("You are Zoe.")
    assert "Current local date and time" in msgs2[0]["content"]


def test_harness_tools_bridge_into_chat_belt():
    """The chat IS the agent: harness tools are reachable from the chat tool
    belt — but lazily, via find_tools/list_catalog/call_tool, not as 60+
    full schemas bridged in directly on every turn (that was the ~20k-token
    prefill bloat described in engine.py's _UPSTREAM_TIMEOUT comment)."""
    e = Engine.__new__(Engine)
    from pleiades import config

    e.settings = config.Settings.load()
    belt = ToolBelt([])
    e._bridge_harness_tools(belt, Profile(name="bridge-test"))
    names = set(belt.names())
    assert {"find_tools", "list_catalog", "call_tool"} <= names, names
    # dispatch plumbing is safe (no-op by itself) so it never trips
    # ToolBelt's own ask-gate before the model even tries a real tool
    assert all(belt._tools[n].safe for n in ("find_tools", "list_catalog", "call_tool"))
    # individual harness tools are no longer bridged in directly
    assert not {"read_file", "run_shell", "web_search"} & names

    ctx = ToolContext(profile=Profile(name="bridge-test"), vault=None, settings=e.settings)
    catalog = belt.dispatch("list_catalog", {}, ctx)
    assert "read_file" in catalog and "run_shell" in catalog and "web_search" in catalog


def test_recent_messages_flattens_and_caps():
    c = chats.create("zoe")
    # 3 real turns, only the last max_turns should come back.
    chats.append_turn(c["id"], "what's the weather", [{"t": "text", "text": "sunny"}],
                      {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    chats.append_turn(c["id"], "message person X", [
        {"t": "tool", "name": "send_email", "args": "{}", "output": "sent", "ok": True},
        {"t": "text", "text": "done, sent it"},
    ], {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    chats.append_turn(c["id"], "that didn't work, send it again", [{"t": "text", "text": "resending now"}],
                      {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    loaded = chats.load(c["id"])

    hist = chats.recent_messages(loaded, max_turns=8)
    assert hist[0] == {"role": "user", "content": "what's the weather"}
    assert hist[1] == {"role": "assistant", "content": "sunny"}
    assert hist[2] == {"role": "user", "content": "message person X"}
    assert hist[3]["role"] == "assistant"
    # ONE tally line per turn, not one marker per tool item.
    assert hist[3]["content"].count("[memory of an earlier session") == 1
    assert "[memory of an earlier session — this turn used tools: send_email." in hist[3]["content"]
    assert "compact record" in hist[3]["content"].lower() or \
        "Compact record" in hist[3]["content"]
    assert "done, sent it" in hist[3]["content"]
    assert hist[-2] == {"role": "user", "content": "that didn't work, send it again"}
    assert hist[-1] == {"role": "assistant", "content": "resending now"}

    # max_turns=1 keeps only the LAST turn (last 2 raw entries).
    capped = chats.recent_messages(loaded, max_turns=1)
    assert capped == [
        {"role": "user", "content": "that didn't work, send it again"},
        {"role": "assistant", "content": "resending now"},
    ]
    chats.delete(c["id"])


def test_recent_messages_reconstructs_interleaved_user_injected():
    """A user_injected item (see chats.py's schema docstring) splits ONE
    persisted assistant turn back into the role sequence the live model
    actually saw: assistant-work-before, the interjection as its own
    user-role message, assistant-work-after."""
    c = chats.create("zoe")
    chats.append_turn(c["id"], "draft the report", [
        {"t": "text", "text": "Starting on the report now."},
        {"t": "tool", "name": "read_file", "args": "{}", "output": "...", "ok": True},
        {"t": "user_injected", "text": "actually make it shorter"},
        {"t": "text", "text": "Got it, trimming it down."},
    ], {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    loaded = chats.load(c["id"])

    hist = chats.recent_messages(loaded)
    assert hist[0] == {"role": "user", "content": "draft the report"}
    assert hist[1] == {"role": "assistant", "content": "Starting on the report now."}
    assert hist[2] == {"role": "user",
                       "content": "[message from the user, mid-task] actually make it shorter"}
    assert hist[3]["role"] == "assistant"
    # The single tally covers the whole persisted turn and lands on its
    # final segment, alongside whatever the assistant said last.
    assert "[memory of an earlier session — this turn used tools: read_file." in hist[3]["content"]
    assert "Got it, trimming it down." in hist[3]["content"]
    chats.delete(c["id"])


def test_recent_messages_user_injected_at_start_or_end_of_turn():
    """No assistant work before/after the interjection -> no empty
    role:assistant entries should be emitted around it."""
    c = chats.create("zoe")
    chats.append_turn(c["id"], "hi", [
        {"t": "user_injected", "text": "wait, different question"},
        {"t": "text", "text": "Sure, answering that instead."},
    ], {"tokens": 1, "seconds": 0.1, "tps": 10.0})
    loaded = chats.load(c["id"])
    hist = chats.recent_messages(loaded)
    assert hist == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "[message from the user, mid-task] wait, different question"},
        {"role": "assistant", "content": "Sure, answering that instead."},
    ]
    chats.delete(c["id"])


def test_recent_messages_collapses_tool_walls_into_one_tally():
    """A turn with dozens of tool calls (26- and 41-item turns seen live)
    must produce ONE tally line, never one marker per tool item — the
    per-item form flooded resent history with dozens of identical bracket
    lines and taught the model to imitate them."""
    c = chats.create("mark")
    items: list[dict] = [{"t": "text", "text": "digging in"}]
    for i in range(41):
        items.append({"t": "tool", "name": "call_tool", "args": "{}",
                      "output": "ok", "ok": True})
    items.append({"t": "tool", "name": "web_search", "args": "{}",
                  "output": "hit", "ok": False})
    items.append({"t": "text", "text": "found it"})
    chats.append_turn(c["id"], "go dig", items,
                      {"tokens": 2, "seconds": 1.0, "tps": 20.0})
    loaded = chats.load(c["id"])

    hist = chats.recent_messages(loaded)
    assert len(hist) == 2 and hist[1]["role"] == "assistant"
    content = hist[1]["content"]
    assert content.count("[memory of an earlier session") == 1
    assert "call_tool ×41" in content
    assert "web_search; some failed" in content
    # Real text survives around the single tally.
    assert "digging in" in content and "found it" in content

    # A turn that was ONLY tool calls still yields exactly one line.
    c2 = chats.create("mark")
    chats.append_turn(c2["id"], "check", [
        {"t": "tool", "name": "list_catalog", "args": "{}", "output": "...", "ok": True},
        {"t": "tool", "name": "list_catalog", "args": "{}", "output": "...", "ok": True},
    ], {"tokens": 1, "seconds": 0.5, "tps": 10.0})
    hist2 = chats.recent_messages(chats.load(c2["id"]))
    assert len(hist2) == 2
    assert hist2[1]["content"].count("[memory of an earlier session") == 1
    assert "list_catalog ×2" in hist2[1]["content"]
    chats.delete(c["id"])
    chats.delete(c2["id"])


def test_base_messages_inserts_history_before_new_message():
    history = [
        {"role": "user", "content": "message person X"},
        {"role": "assistant", "content": "done, sent it"},
    ]
    msgs = Engine._base_messages("send it again", None, None, history)
    assert msgs[0]["role"] == "system"
    assert msgs[1] == history[0]
    assert msgs[2] == history[1]
    assert msgs[-1] == {"role": "user", "content": "send it again"}

    # No history -> unchanged existing behavior (system + bare new message).
    msgs_no_hist = Engine._base_messages("hi", None, None, None)
    assert len(msgs_no_hist) == 2
    assert msgs_no_hist[-1] == {"role": "user", "content": "hi"}


def test_recent_messages_tool_replay_native_format():
    """Cloud brains get past tool calls replayed in NATIVE wire format:
    assistant message with real tool_calls array + one paired role:"tool"
    message per call, RESULTS LEFT OUT (stubbed). Native history is what
    models are trained on — nothing counterfeit to imitate."""
    c = chats.create("mark")
    chats.append_turn(c["id"], "dig in", [
        {"t": "text", "text": "On it."},
        {"t": "tool", "name": "web_search", "args": '{"q":"tor"}', "output": "10 hits", "ok": True},
        {"t": "text", "text": "Found a lead, going deeper."},
        {"t": "tool", "name": "call_tool", "args": "{}", "output": "ok1", "ok": True},
        {"t": "tool", "name": "call_tool", "args": "{}", "output": "ok2", "ok": True},
        {"t": "tool", "name": "call_tool", "args": "{}", "output": "ok3", "ok": True},
    ], {"tokens": 2, "seconds": 1.0, "tps": 20.0})
    loaded = chats.load(c["id"])

    hist = chats.recent_messages(loaded, tool_replay=True)
    # user, assistant(text+tool_calls), tool(stub), assistant(text), tool(stub), [user]
    roles = [(m["role"], "tool_calls" in m) for m in hist]
    assert roles[0] == ("user", False)

    a1 = hist[1]
    assert a1["role"] == "assistant" and a1["content"] == "On it."
    assert len(a1["tool_calls"]) == 1
    tc = a1["tool_calls"][0]
    assert tc["type"] == "function" and tc["function"]["name"] == "web_search"

    t1 = hist[2]
    assert t1["role"] == "tool"
    assert t1["tool_call_id"] == tc["id"]
    assert t1["content"].startswith("(result not stored in memory)")
    assert "10 hits" not in t1["content"]

    # consecutive identical calls collapse to ONE pair with a repeat note
    a2 = hist[3]
    assert a2["content"] == "Found a lead, going deeper."
    assert len(a2["tool_calls"]) == 1
    assert a2["tool_calls"][0]["function"]["name"] == "call_tool"
    t2 = hist[4]
    assert "identical repeats omitted" in t2["content"]
    assert t2["tool_call_id"] == a2["tool_calls"][0]["id"]

    # every tool_call id has exactly one paired tool message (template safety)
    ids_out = [tc["id"] for m in hist if m.get("tool_calls") for tc in m["tool_calls"]]
    ids_in = [m["tool_call_id"] for m in hist if m["role"] == "tool"]
    assert ids_out == ids_in and len(set(ids_out)) == len(ids_out)
    chats.delete(c["id"])


def test_recent_messages_tool_replay_wall_falls_back_to_tally():
    """Thrash runs collapse: call_tool ×29 identical replays as ONE
    representative pair. But more DISTINCT calls than _REPLAY_MAX_CALLS
    falls back to the single tally line — bounded worst case."""
    c = chats.create("mark")
    items = [{"t": "tool", "name": "call_tool", "args": "{}", "output": "x", "ok": True}
             for _ in range(29)]
    chats.append_turn(c["id"], "go", items,
                      {"tokens": 1, "seconds": 1.0, "tps": 10.0})
    hist = chats.recent_messages(chats.load(c["id"]), tool_replay=True)
    assert len(hist) == 3  # user + assistant(tool_calls, no content key) + tool stub
    assert "content" not in hist[1]
    assert len(hist[1]["tool_calls"]) == 1
    tool_msg = hist[2]
    assert tool_msg["role"] == "tool"
    assert "28 identical repeats omitted" in tool_msg["content"]
    chats.delete(c["id"])

    c2 = chats.create("mark")
    distinct = [{"t": "tool", "name": f"tool_{i}", "args": f'{{"n":{i}}}', "output": "x",
                 "ok": True} for i in range(_REPLAY_MAX_CALLS + 3)]
    chats.append_turn(c2["id"], "go wide", distinct,
                      {"tokens": 1, "seconds": 1.0, "tps": 10.0})
    hist2 = chats.recent_messages(chats.load(c2["id"]), tool_replay=True)
    assert len(hist2) == 2  # user + tally fallback
    assert "[memory of an earlier session — this turn used tools:" in hist2[1]["content"]
    assert not any(m.get("tool_calls") for m in hist2)
    chats.delete(c2["id"])


def test_recent_messages_tool_replay_interleaved_segments():
    """user_injected splits replayed turns exactly like the live model saw;
    tool pairs never cross an interjection boundary."""
    c = chats.create("mark")
    chats.append_turn(c["id"], "draft it", [
        {"t": "text", "text": "Starting."},
        {"t": "tool", "name": "read_file", "args": '{"p":"a.txt"}', "output": "...", "ok": True},
        {"t": "user_injected", "text": "make it shorter"},
        {"t": "text", "text": "Trimming."},
    ], {"tokens": 2, "seconds": 0.1, "tps": 20.0})
    hist = chats.recent_messages(chats.load(c["id"]), tool_replay=True)
    # [user, assistant+tool_calls, tool stub, user-injected, assistant]
    assert hist[0] == {"role": "user", "content": "draft it"}
    assert hist[1]["role"] == "assistant"
    assert hist[1]["content"] == "Starting."
    assert hist[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert hist[2]["role"] == "tool" and hist[2]["tool_call_id"] == hist[1]["tool_calls"][0]["id"]
    assert hist[3] == {"role": "user",
                       "content": "[message from the user, mid-task] make it shorter"}
    assert hist[4] == {"role": "assistant", "content": "Trimming."}
    chats.delete(c["id"])
