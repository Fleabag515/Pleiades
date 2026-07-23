"""Mid-turn user-message injection: Engine.stream_events' poll_injections
param only ever splices a queued message in at a SAFE round/tool boundary,
never mid-stream and never between an assistant's tool_calls message and its
tool results (see engine.py's stream_events docstring)."""

from pleiades import config
from pleiades.engine import Engine
from pleiades.profiles import Profile
from pleiades.tools import Tool, ToolBelt, ToolContext


class _NoopTool(Tool):
    name = "noop"
    description = "test-only tool that does nothing observable"
    parameters = {"type": "object", "properties": {}}
    safe = True

    def run(self, ctx, **kwargs):
        return "done"


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeFn(name, arguments)


class _FakeMsg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None
        self.reasoning = None


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResp:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _make_engine():
    eng = Engine(settings=config.Settings.load(), manager=object(), anamnesis=object())
    ctx = ToolContext(profile=Profile(name="inject-test"), vault=None, settings=eng.settings)
    belt = ToolBelt([_NoopTool()])

    def fake_client_for(self, profile):
        return object(), ctx, belt

    eng.__class__._client_for = fake_client_for  # patched per-test via monkeypatch below
    return eng


def test_injections_never_split_toolcalls_from_their_results(monkeypatch):
    """Round 1: assistant calls `noop`. An injection queued BEFORE the turn
    starts must land before round 1's request; an injection that arrives
    WHILE round 1's single tool is dispatching must NOT be spliced until
    round 2 -- i.e. it must appear strictly after round 1's assistant
    tool_calls message AND its paired tool-result message in `messages`."""
    from pleiades import engine as engine_mod

    ctx = ToolContext(profile=Profile(name="inject-test"), vault=None,
                      settings=config.Settings.load())
    belt = ToolBelt([_NoopTool()])

    def fake_client_for(self, profile):
        return object(), ctx, belt

    monkeypatch.setattr(engine_mod.Engine, "_client_for", fake_client_for)
    monkeypatch.setattr(engine_mod.Engine, "_model_for", lambda self, profile: "test-model")

    msg1 = _FakeMsg(content="", tool_calls=[_FakeToolCall("call_1", "noop", "{}")])
    msg2 = _FakeMsg(content="All done.", tool_calls=None)
    responses = [_FakeResp(msg1), _FakeResp(msg2)]
    captured_messages = []

    def fake_chat_create(self, client, profile, **kwargs):
        if kwargs.get("stream"):
            # Force the non-streaming fallback path (simpler to drive from a test).
            raise RuntimeError("no streaming in this test double")
        captured_messages.append(list(kwargs["messages"]))
        idx = len(captured_messages) - 1
        return responses[idx], client

    monkeypatch.setattr(engine_mod.Engine, "_chat_create", fake_chat_create)

    # poll #0 -> fires at the very top, before round 1's request.
    # poll #1 -> fires between round 1's (single) tool dispatch.
    # poll #2 -> fires at the top of round 2; nothing new.
    schedule = {
        0: ["fix the report format"],
        1: ["actually skip that, do Y instead"],
    }
    poll_calls = {"n": 0}

    def poll_injections():
        n = poll_calls["n"]
        poll_calls["n"] += 1
        return schedule.get(n, [])

    eng = Engine(settings=config.Settings.load(), manager=object(), anamnesis=object())
    events = list(eng.stream_events(Profile(name="inject-test"), "do the thing",
                                    poll_injections=poll_injections))

    # Both injections surfaced as user_injected events, in order.
    injected_events = [e for e in events if e["type"] == "user_injected"]
    assert [e["text"] for e in injected_events] == [
        "fix the report format", "actually skip that, do Y instead",
    ]

    assert len(captured_messages) == 2

    # Round 1's request: system, user, and the injection queued BEFORE the
    # turn started -- applied at the very first safe boundary.
    round1 = captured_messages[0]
    assert round1[0]["role"] == "system"
    assert round1[1] == {"role": "user", "content": "do the thing"}
    assert round1[2]["role"] == "user"
    assert "fix the report format" in round1[2]["content"]
    assert len(round1) == 3

    # Round 2's request must contain, in order: round 1's assistant
    # tool_calls message, its tool result, and ONLY THEN the injection that
    # arrived mid-round-1 -- never spliced between the two.
    round2 = captured_messages[1]
    assert round2[3]["role"] == "assistant"
    assert round2[3].get("tool_calls"), "assistant tool_calls message missing/misplaced"
    assert round2[4]["role"] == "tool", (
        "tool result must immediately follow its assistant tool_calls message"
    )
    assert round2[5]["role"] == "user"
    assert "actually skip that, do Y instead" in round2[5]["content"]
    assert round2[5]["content"].startswith("[message from the user, mid-task]")
    assert len(round2) == 6


def test_no_poll_injections_is_a_no_op(monkeypatch):
    """poll_injections=None (the default) must behave exactly as before --
    no user_injected events, no extra messages spliced in."""
    from pleiades import engine as engine_mod

    ctx = ToolContext(profile=Profile(name="inject-test"), vault=None,
                      settings=config.Settings.load())
    belt = ToolBelt([_NoopTool()])

    def fake_client_for(self, profile):
        return object(), ctx, belt

    monkeypatch.setattr(engine_mod.Engine, "_client_for", fake_client_for)
    monkeypatch.setattr(engine_mod.Engine, "_model_for", lambda self, profile: "test-model")

    msg = _FakeMsg(content="Hi there.", tool_calls=None)

    def fake_chat_create(self, client, profile, **kwargs):
        if kwargs.get("stream"):
            raise RuntimeError("no streaming in this test double")
        return _FakeResp(msg), client

    monkeypatch.setattr(engine_mod.Engine, "_chat_create", fake_chat_create)

    eng = Engine(settings=config.Settings.load(), manager=object(), anamnesis=object())
    events = list(eng.stream_events(Profile(name="inject-test"), "hello"))
    assert not any(e["type"] == "user_injected" for e in events)
