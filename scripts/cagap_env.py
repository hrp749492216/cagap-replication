"""Shared environment loading for CAGAP runner scripts.

The project keeps local API credentials in ``.env`` and/or
``~/.cagap-secrets``. Scripts should call ``load_cagap_env()`` before reading
``os.environ`` so approved direct ``python3 scripts/...`` command prefixes do
not need a broad shell wrapper just to source secrets.
"""

from __future__ import annotations

from pathlib import Path

_LOADED = False


def load_cagap_env() -> None:
    """Load project-local env files once, without overriding real env vars."""
    global _LOADED
    if _LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        return

    repo_root = Path(__file__).resolve().parents[1]
    for path in (
        repo_root / ".env",
        Path.home() / ".cagap-secrets",
        Path.home() / ".cagap_secrets",
    ):
        if path.exists():
            load_dotenv(path, override=False)
    _LOADED = True
