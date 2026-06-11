"""The chat engine must give the character real machine facts (so it stops
inventing Linux paths) and inject the operating contract. Pure, offline."""

import platform
from types import SimpleNamespace

from pleiades.engine import Engine, DEFAULT_OPERATING_CONTRACT


def test_environment_note_has_real_paths_and_os():
    note = Engine._environment_note(SimpleNamespace(name="TestChar"))
    assert "TestChar" in note            # the character's own dirs
    assert "workspace" in note.lower()
    assert platform.system() in note     # tells it which OS it is on
    assert "NOTEBOOK.md" in note         # points at the shared notebook


def test_base_messages_includes_env_and_contract():
    msgs = Engine._base_messages("hi", None, "ENV-MARKER-LINE")
    sys_text = msgs[0]["content"]
    assert "ENV-MARKER-LINE" in sys_text
    # default operating contract still present when no caller system prompt
    assert "NEVER fake a tool" in sys_text
    assert DEFAULT_OPERATING_CONTRACT.split("\n", 1)[0] in sys_text


def test_explicit_system_prompt_still_wins():
    msgs = Engine._base_messages("hi", "CUSTOM SYSTEM", "ENVX")
    sys_text = msgs[0]["content"]
    assert "CUSTOM SYSTEM" in sys_text
    assert "ENVX" in sys_text
