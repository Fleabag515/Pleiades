"""Self-directed practice: stats recording, status report, study guidance."""

import pleiades.harness.builtins.practice as practice
from pleiades.harness.builtins.practice import (practice_status, record_outcome,
                                                study_tool)


def _stats_to(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "STATS_PATH", tmp_path / "tool_stats.json")


def test_record_and_status(tmp_path, monkeypatch):
    _stats_to(tmp_path, monkeypatch)
    record_outcome("run_shell", True)
    record_outcome("run_shell", False, "missing 'command'")
    record_outcome("run_shell", False, "bad json")
    out = practice_status()
    assert "run_shell" in out and "1/3 ok" in out
    assert "practice this" in out


def test_status_empty(tmp_path, monkeypatch):
    _stats_to(tmp_path, monkeypatch)
    assert "No tool usage recorded yet" in practice_status()


def test_study_known_tool_includes_schema_and_mistakes(tmp_path, monkeypatch):
    _stats_to(tmp_path, monkeypatch)
    record_outcome("run_shell", False, "TypeError: missing 'command'")
    out = study_tool("run_shell")
    assert "Parameters (JSON schema)" in out
    assert "missing 'command'" in out
    assert "record_lesson" in out


def test_study_unknown_tool_suggests(tmp_path, monkeypatch):
    _stats_to(tmp_path, monkeypatch)
    out = study_tool("run_shel")
    assert "No tool named" in out and "run_shell" in out


def test_record_outcome_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "STATS_PATH", tmp_path / "no" / "such" / "x.json")
    record_outcome("x", True)  # parent dirs created, or swallowed — no raise
