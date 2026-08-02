from __future__ import annotations

import inspect

from src.help_content import get_context_help
from src.sidebar_manual import (
    INTERFACE_LANGUAGE,
    INTERFACE_MODE,
    configured_interface,
    migrate_legacy_interface_state,
    render_dashboard_guide,
)


def test_interface_defaults_match_interpretability_first_contract() -> None:
    interface = configured_interface({})
    assert interface == {
        "language": "en",
        "mode": "interpretability_first",
        "progressive_disclosure": True,
        "show_internal_ids": False,
        "show_language_selector": False,
        "show_view_mode_selector": False,
    }
    assert INTERFACE_LANGUAGE == "en"
    assert INTERFACE_MODE == "interpretability_first"


def test_sidebar_has_no_language_or_view_mode_widget() -> None:
    source = inspect.getsource(render_dashboard_guide)
    assert "selectbox" not in source
    assert "segmented_control" not in source
    assert "interface_language" not in source
    assert "interface_view_mode" not in source


def test_legacy_interface_state_is_removed_defensively() -> None:
    state = {
        "interface_language": "es",
        "interface_view_mode": "advanced",
        "advanced_mode": True,
        "diagnosis_analysis_date": "2026-07-19",
    }
    removed = migrate_legacy_interface_state(state)
    assert set(removed) == {
        "interface_language", "interface_view_mode", "advanced_mode"
    }
    assert state == {"diagnosis_analysis_date": "2026-07-19"}


def test_manual_selection_falls_back_to_english_overview() -> None:
    unknown = get_context_help("not-a-tab", "en")
    overview = get_context_help("Overview", "en")
    assert unknown == overview
    assert unknown[0].title == "Quick start"
