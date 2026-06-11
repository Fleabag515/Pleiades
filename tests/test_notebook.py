"""Tests for the shared human-editable notebook tool. Pure file ops, offline."""

from pleiades.harness.builtins.notebook import (
    bind_notebook, notebook_read, notebook_append, notebook_write,
)


def test_empty_then_append_and_read(tmp_path):
    p = tmp_path / "NOTEBOOK.md"
    bind_notebook(str(p))
    assert "empty" in notebook_read().lower()

    notebook_append("kobolds drill browser_click before big runs")
    text = notebook_read()
    assert "kobolds drill browser_click" in text
    assert p.exists()


def test_append_preserves_history(tmp_path):
    p = tmp_path / "NOTEBOOK.md"
    bind_notebook(str(p))
    notebook_append("first")
    notebook_append("second")
    text = notebook_read()
    assert "first" in text and "second" in text


def test_write_replaces(tmp_path):
    p = tmp_path / "NOTEBOOK.md"
    bind_notebook(str(p))
    notebook_append("old line")
    notebook_write("# Fresh start\n")
    text = notebook_read()
    assert "Fresh start" in text
    assert "old line" not in text


def test_human_edits_are_visible(tmp_path):
    p = tmp_path / "NOTEBOOK.md"
    bind_notebook(str(p))
    # human edits the file directly in an editor
    p.write_text("# notes\n- remember to feed the kobold\n", encoding="utf-8")
    assert "feed the kobold" in notebook_read()
