"""Chat sessions + unified chat/agent wiring."""

from pleiades import chats
from pleiades.engine import Engine
from pleiades.profiles import Profile
from pleiades.tools import ToolBelt


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
    """The chat IS the agent: harness tools appear in the chat tool belt."""
    e = Engine.__new__(Engine)
    from pleiades import config

    e.settings = config.Settings.load()
    belt = ToolBelt([])
    e._bridge_harness_tools(belt, Profile(name="bridge-test"))
    names = set(belt.names())
    assert {"read_file", "run_shell", "web_search"} <= names, names
