"""Streamlit sidebar manual and progressive-disclosure helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import streamlit as st

from src.help_content import get_context_help
from src.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, language_name, normalize_language, tr


@dataclass(frozen=True)
class ViewVisibility:
    essential: bool
    technical_controls: bool
    detailed_tables: bool
    lineage_details: bool
    complete_catalogues: bool
    internal_ids: bool


def mode_visibility(mode: str) -> ViewVisibility:
    """Return the UI-only visibility policy; this never changes data filters."""
    advanced = str(mode).lower() == "advanced"
    return ViewVisibility(True, advanced, advanced, advanced, advanced, advanced)


def is_advanced(mode: str) -> bool:
    return mode_visibility(mode).technical_controls


def configured_interface(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(config.get("interface", {}))
    languages = [code for code in raw.get("available_languages", SUPPORTED_LANGUAGES) if code in SUPPORTED_LANGUAGES]
    return {
        "default_language": normalize_language(raw.get("default_language", DEFAULT_LANGUAGE)),
        "available_languages": languages or list(SUPPORTED_LANGUAGES),
        "default_view_mode": "advanced" if raw.get("default_view_mode") == "advanced" else "basic",
        "show_internal_ids": bool(raw.get("show_internal_ids", True)),
    }


def render_help_sidebar(
    config: Mapping[str, Any],
    active_tab: str = "Overview",
) -> tuple[str, str]:
    """Render the persistent one-language contextual manual."""
    interface = configured_interface(config)
    st.session_state.setdefault("interface_language", interface["default_language"])
    st.session_state.setdefault("interface_view_mode", interface["default_view_mode"])
    language = normalize_language(st.session_state["interface_language"])

    with st.sidebar:
        st.header(tr("help_title", language), anchor=False)
        language = st.selectbox(
            tr("language", language),
            interface["available_languages"],
            key="interface_language",
            format_func=lambda code: language_name(code, language),
        )
        language = normalize_language(language)
        mode = st.segmented_control(
            tr("view_mode", language),
            ["basic", "advanced"],
            key="interface_view_mode",
            format_func=lambda value: tr(str(value), language),
        ) or interface["default_view_mode"]
        st.divider()
        for section in get_context_help(active_tab, language):
            with st.expander(section.title, expanded=section.key == "quick_start"):
                st.write(section.body)
    return language, str(mode)


def render_first_use(language: str, view_mode: str) -> None:
    """Render a collapsible, session-only first-use guide in Basic mode."""
    if is_advanced(view_mode):
        return
    with st.expander(tr("getting_started", language), expanded=True):
        st.markdown(
            f"1. {tr('getting_started_1', language)}\n\n"
            f"2. {tr('getting_started_2', language)}\n\n"
            f"3. {tr('getting_started_3', language)}"
        )
