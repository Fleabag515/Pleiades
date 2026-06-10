"""Compatibility shim — harness config is now the unified project Settings.

The harness historically had its own Config/Tier. They are now the single,
project-wide `pleiades.config.Settings` (one source of truth: defaults + an
optional config.json + environment). This module re-exports them so existing
`from .config import Config, Tier` imports inside the harness keep working.
"""

from __future__ import annotations

from ..config import (  # noqa: F401
    Settings as Config,
    Tier,
    DEFAULT_TIERS,
    OPUS,
    SONNET,
    HAIKU,
)

__all__ = ["Config", "Tier", "DEFAULT_TIERS", "OPUS", "SONNET", "HAIKU"]
