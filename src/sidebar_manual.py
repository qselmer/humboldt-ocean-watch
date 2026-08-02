"""English-only Streamlit guidance and legacy interface-state migration."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import streamlit as st

from src.help_content import get_context_help


INTERFACE_LANGUAGE = "en"
INTERFACE_MODE = "interpretability_first"

LEGACY_INTERFACE_STATE_KEYS = (
    "interface_language",
    "interface_view_mode",
    "selected_language",
    "view_mode",
    "advanced_mode",
    "first_use_language",
    "first_use_language_variant",
)


def configured_interface(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the single supported Streamlit interface policy."""
    raw = dict(config.get("interface", {}))
    return {
        "language": str(raw.get("language", INTERFACE_LANGUAGE)),
        "mode": str(raw.get("mode", INTERFACE_MODE)),
        "progressive_disclosure": bool(raw.get("progressive_disclosure", True)),
        "show_internal_ids": bool(raw.get("show_internal_ids", False)),
        "show_language_selector": bool(raw.get("show_language_selector", False)),
        "show_view_mode_selector": bool(raw.get("show_view_mode_selector", False)),
    }


def migrate_legacy_interface_state(state: MutableMapping[str, Any]) -> tuple[str, ...]:
    """Remove obsolete language/mode keys left by older Streamlit sessions."""
    removed: list[str] = []
    for key in LEGACY_INTERFACE_STATE_KEYS:
        if key in state:
            del state[key]
            removed.append(key)
    return tuple(removed)


def render_dashboard_guide(active_tab: str = "Overview") -> None:
    """Render English contextual help without language or view-mode selectors."""
    with st.sidebar:
        st.divider()
        st.header("Help and interpretation guide", anchor=False)
        for section in get_context_help(active_tab, INTERFACE_LANGUAGE):
            with st.expander(section.title, expanded=False):
                st.write(section.body)


def render_first_use() -> None:
    """Render the concise, non-intrusive dashboard orientation guide."""
    with st.expander("How to use this dashboard", expanded=False):
        st.markdown(
            "1. Select an analysis date.\n\n"
            "2. Review the regional state.\n\n"
            "3. Compare maps and time series.\n\n"
            "4. Check representativeness and quality before interpretation."
        )
