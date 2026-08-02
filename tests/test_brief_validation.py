from copy import deepcopy

import numpy as np

from src.brief_schema import new_context
from src.brief_validation import validate_brief_context


def valid_context() -> dict:
    context = new_context()
    context["product"] = {
        "product_name": "Humboldt Ocean Watch", "product_short_name": "HOW",
        "product_version": "0.1.0", "product_status": "experimental",
        "experimental_product": True, "region_name": "Niño 1+2", "region_code": "nino12",
        "spatial_bounds": {"longitude": [-90, -80], "latitude": [-10, 0]},
        "source_dataset": "cached", "source_variable": "analysed_sst", "institution": None,
        "scientific_scope": "Not an official Coastal El Niño magnitude classification.",
    }
    context["analysis"] = {
        "requested_analysis_date": "latest", "resolved_analysis_date": "2026-01-02",
        "generated_at": "2026-01-03T00:00:00+00:00", "available_start_date": "2026-01-01",
        "available_end_date": "2026-01-02", "data_latency_days": 1,
        "temporal_resolution": "1D", "spatial_resolution": None,
        "data_mode": "Copernicus cached data", "analysis_status": "valid",
    }
    context["availability"] = {
        key: {"available": True, "reason": None, "expected_path": f"outputs/{key}.parquet"}
        for key in (
            "regional_metrics", "quality_control", "climatology", "representativeness",
            "temporal_features", "spatial_features", "univariate_events", "daily_patches",
            "tracks", "event_families", "lineage_edges",
        )
    }
    context["representativeness"] = {"class_code": "unclassified", "evidence_score": None}
    context["provenance"] = {"sources": [{"source_id": "daily_metrics", "path": "data/metrics.parquet", "required": True}]}
    context["limitations"] = [{
        "limitation_id": "not_official", "category": "classification",
        "statement_es": "No es oficial.",
        "statement_en": "Not an official Coastal El Niño magnitude classification.",
        "affected_sections": ["product"], "severity": "warning",
    }]
    context["generation_constraints"] = {
        "prohibited_claims": [
            "official Coastal El Niño magnitude classification", "forecasts not present",
            "invented numerical values", "tracks are individual water masses",
        ],
        "numerical_fact_policy": "Every number requires a valid fact_id.",
        "missing_data_policy": "Unavailable facts must not be inferred.",
    }
    return context


def fact(value=0.5, unit="fraction", source_path="data/metrics.parquet") -> dict:
    return {
        "fact_id": "regional.value", "section": "regional_state", "name": "value",
        "value": value, "unit": unit, "status": "valid", "source_id": "daily_metrics",
        "source_path": source_path, "source_column": "value", "selected_date": "2026-01-02",
        "calculation_method": "cached", "precision": 3, "allowed_in_brief": True,
        "reason_when_unavailable": None,
    }


def test_valid_minimal_context_passes() -> None:
    assert validate_brief_context(valid_context()).result == "passed"


def test_nan_infinity_fraction_and_coordinate_rejection() -> None:
    for bad_fact in (
        fact(np.nan, "dimensionless"),
        fact(np.inf, "dimensionless"),
        fact(1.2, "fraction"),
        fact(95.0, "degree_latitude"),
    ):
        context = valid_context()
        context["fact_registry"] = [bad_fact]
        assert validate_brief_context(context).result == "failed"


def test_absolute_path_and_credential_like_provenance_are_rejected() -> None:
    context = valid_context()
    context["fact_registry"] = [fact(source_path="C:/Users/name/metrics.parquet")]
    assert validate_brief_context(context).result == "failed"
    context = valid_context()
    context["provenance"]["sources"][0]["api_key"] = "forbidden"
    assert validate_brief_context(context).result == "failed"


def test_duplicate_fact_missing_disclaimer_and_constraints_fail() -> None:
    context = valid_context()
    context["fact_registry"] = [fact(), deepcopy(fact())]
    assert validate_brief_context(context).result == "failed"
    context = valid_context()
    context["limitations"] = []
    assert validate_brief_context(context).result == "failed"
    context = valid_context()
    context["generation_constraints"] = {}
    assert validate_brief_context(context).result == "failed"


def test_payload_limit_is_enforced() -> None:
    context = valid_context()
    context["quality"]["large"] = "x" * 5000
    report = validate_brief_context(context, maximum_payload_kb=1)
    assert report.result == "failed"
    assert any(issue["code"] == "oversized_payload" for issue in report.errors)


def test_track_and_family_count_mismatches_are_rejected() -> None:
    context = valid_context()
    context["spatiotemporal_activity"] = {
        "active_track_count": {"value": 0},
        "active_family_count": {"value": 0},
        "top_tracks": [{"track_id": "TRK000001"}],
        "top_event_families": [{"event_family_id": "FAM000001"}],
    }
    report = validate_brief_context(context)
    codes = {issue["code"] for issue in report.errors}
    assert {"active_track_count_mismatch", "active_family_count_mismatch"} <= codes


def test_compact_fact_reference_must_match_registry() -> None:
    context = valid_context()
    context["fact_registry"] = [fact()]
    context["regional_state"]["value"] = {
        "fact_id": "regional.value", "value": 0.6, "unit": "fraction", "status": "valid",
    }
    report = validate_brief_context(context)
    assert any(issue["code"] == "fact_reference_mismatch" for issue in report.errors)


def test_error_quality_flag_fails_validation() -> None:
    context = valid_context()
    context["quality_flags"] = [{
        "flag_id": "broken_consistency", "severity": "error", "section": "quality",
        "message_code": "broken_consistency", "message_es": "Inconsistencia.",
        "message_en": "Inconsistency.", "affected_fact_ids": [],
        "suppress_dependent_claims": True,
    }]
    assert validate_brief_context(context).result == "failed"
