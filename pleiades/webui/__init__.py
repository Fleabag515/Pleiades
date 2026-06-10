"""Pleiades Web UI — a local control panel for characters, models, and settings.

A FastAPI app that wraps the existing Pleiades managers (ProfileManager,
ModelManager, Vault, Anamnesis, Settings) and serves a polished single-page
frontend. Cross-platform: it runs in the browser, so it behaves identically on
Linux and Windows.

Launch it with::

    pleiades ui                 # added to the CLI
    python -m pleiades.webui    # or directly

Everything it shows is your own local state under ``~/.pleiades`` — no data
leaves the machine.
"""

from .server import create_app, run

__all__ = ["create_app", "run"]
