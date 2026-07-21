from src.help_content import get_context_help
from src.sidebar_manual import configured_interface, mode_visibility


def test_interface_defaults_match_configuration_contract() -> None:
    interface = configured_interface({})
    assert interface["default_language"] == "es"
    assert interface["default_view_mode"] == "basic"
    assert interface["show_internal_ids"] is True


def test_basic_mode_prioritizes_essential_content() -> None:
    visibility = mode_visibility("basic")
    assert visibility.essential
    assert not visibility.technical_controls
    assert not visibility.detailed_tables
    assert not visibility.lineage_details
    assert not visibility.complete_catalogues
    assert not visibility.internal_ids


def test_advanced_mode_restores_complete_content() -> None:
    visibility = mode_visibility("advanced")
    assert all(vars(visibility).values())


def test_manual_selection_falls_back_to_overview() -> None:
    unknown = get_context_help("not-a-tab", "es")
    overview = get_context_help("Overview", "es")
    assert unknown == overview
