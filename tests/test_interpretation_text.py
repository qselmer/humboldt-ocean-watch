from src.interpretation_text import interpret_representativeness, interpret_thermal_event


def _representative_metrics() -> dict:
    return {
        "class": "strong_heterogeneous",
        "weighted_mean_anomaly": 1.6,
        "spatial_standard_deviation": 0.9,
        "sign_coherence": 0.92,
        "positive_fraction": 0.91,
        "negative_fraction": 0.03,
        "neutral_fraction": 0.06,
        "valid_coverage": 0.97,
        "dominant_patch_fraction": 0.72,
    }


def test_representativeness_interpretation_is_deterministic_and_bilingual() -> None:
    spanish = interpret_representativeness("2026-07-19", _representative_metrics(), "es")
    english = interpret_representativeness("2026-07-19", _representative_metrics(), "en")
    assert "19 de julio de 2026" in spanish
    assert "anomalía fue positiva" in spanish
    assert "variación espacial importante" in spanish
    assert "19 July 2026" in english
    assert "anomalies were positive" in english
    assert "substantial spatial variation" in english


def test_insufficient_coverage_is_explicit() -> None:
    text = interpret_representativeness(
        "2026-07-19", {"class": "insufficient_coverage", "valid_coverage": 0.2}, "en"
    )
    assert "insufficient" in text.lower()


def test_event_interpretation_uses_cached_state_without_official_categories() -> None:
    overview = {
        "active_event": True,
        "event_day": 6,
        "current_intensity": 0.8,
        "patch_count": 3,
        "active_track_count": 2,
        "active_family_count": 1,
        "largest_patch_area_km2": 1200.0,
    }
    spanish = interpret_thermal_event("2026-07-19", overview, "es", split_count=1)
    english = interpret_thermal_event("2026-07-19", overview, "en", split_count=1)
    assert "día 6" in spanish and "3 parches" in spanish and "1 divisiones" in spanish
    assert "event day 6" in english and "3 patches" in english and "1 splits" in english
    forbidden = "el niño costero"
    assert forbidden not in spanish.lower()
    assert forbidden not in english.lower()


def test_inactive_event_remains_neutral() -> None:
    text = interpret_thermal_event(
        "2026-07-19",
        {"active_event": False, "patch_count": 0, "active_track_count": 0, "active_family_count": 0},
        "en",
    )
    assert "no univariate event is active" in text
