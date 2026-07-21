from src.i18n import format_identifier, human_label, internal_id_caption, tr


def test_spanish_and_english_translation_lookup() -> None:
    assert tr("help_title", "es") == "Ayuda y guía"
    assert tr("help_title", "en") == "Help and guide"


def test_missing_translation_has_deterministic_fallback() -> None:
    assert tr("missing_key", "en") == "missing_key"
    assert tr("missing_key", "es", default="Texto") == "Texto"


def test_human_readable_class_and_event_role_labels() -> None:
    assert human_label("strong_heterogeneous", "es") == "Calentamiento fuerte con alta variabilidad espacial"
    assert human_label("strong_heterogeneous", "en") == "Strong warming with high spatial variability"
    assert human_label("split_parent", "es") == "Parche que se divide"
    assert human_label("merge_child", "en") == "Merge-result patch"
    assert "_" not in human_label("daily_climatological", "en")


def test_internal_identifiers_keep_reproducible_values() -> None:
    assert format_identifier("FAM000002", "es") == "Familia 2"
    assert format_identifier("FAM000002", "en") == "Family 2"
    assert format_identifier("TRK000034", "es") == "Track 34"
    assert internal_id_caption("TRK000034", "en") == "Internal ID: TRK000034"
