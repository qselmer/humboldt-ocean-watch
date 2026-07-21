"""Offline construction of the scientific-brief context package.

Only validated cached tables and summaries are read.  This module does not
open NetCDF cubes, call remote services, or rerun scientific calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src import __version__
from src.brief_fact_registry import FactRegistry
from src.brief_provenance import LoadedProduct, SourceSpec, is_stale, load_source, repository_relative
from src.brief_schema import REPRESENTATIVENESS_CLASSES, new_context
from src.export_utils import to_json_compatible
from src.i18n import human_label


CORE_COLUMNS = {
    "daily_metrics": frozenset(
        {
            "date", "mean_sst_c", "mean_sst_anomaly_c", "maximum_anomaly_c",
            "p90_anomaly_c", "mean_standardized_anomaly",
            "area_anomaly_ge_1c_percent", "area_anomaly_ge_2c_percent",
            "area_anomaly_ge_3c_percent", "area_above_climatological_p90_percent",
            "area_zscore_ge_2_percent", "warm_centroid_latitude",
            "warm_centroid_longitude",
            "valid_data_coverage_percent", "climatology_method",
            "climatology_reference_period", "climatology_fallback_used",
        }
    ),
    "representativeness": frozenset(
        {
            "date", "class", "status", "triggered_rule", "evidence_score",
            "valid_coverage", "weighted_mean_anomaly", "weighted_median_anomaly",
            "spatial_standard_deviation", "weighted_iqr", "positive_fraction",
            "negative_fraction", "neutral_fraction", "sign_coherence",
            "signal_heterogeneity_ratio", "mean_median_difference",
            "dominant_patch_fraction", "patch_density", "coexistence_index",
            "spatial_compensation_index", "climatology_method", "data_mode",
        }
    ),
}

QC_REQUIRED_KEYS = frozenset(
    {
        "data_mode", "units", "start_date", "end_date", "number_of_dates",
        "temporal_interval_summary", "expected_frequency", "spatial_dimensions",
        "spatial_bounds", "valid_data_fraction", "weighted_spatial_coverage",
        "warnings", "errors", "overall_status",
    }
)


@dataclass(frozen=True)
class BriefBuildInputs:
    daily_metrics: Path
    qc_report: Path
    representativeness: Path
    series_bank: Path
    temporal_features: Path
    spatial_features: Path
    univariate_events: Path
    univariate_flags: Path
    univariate_summary: Path
    daily_patches: Path
    daily_patch_summary: Path
    daily_patch_build_summary: Path
    patch_observations: Path
    lineage_edges: Path
    tracks: Path
    event_families: Path
    tracking_summary: Path


def input_paths_from_config(
    config: Mapping[str, Any],
    root: Path,
    overrides: Mapping[str, Path | None] | None = None,
) -> BriefBuildInputs:
    values = dict(overrides or {})

    def path(name: str, default: str) -> Path:
        candidate = values.get(name) or Path(default)
        return candidate if candidate.is_absolute() else root / candidate

    return BriefBuildInputs(
        daily_metrics=path("daily_metrics", config["data"]["metrics_path"]),
        qc_report=path("qc_report", config["quality_control"]["output"]),
        representativeness=path("representativeness", config["representativeness"]["output"]),
        series_bank=path("series_bank", config["series_bank"]["output"]),
        temporal_features=path("temporal_features", config["temporal_features"]["output"]),
        spatial_features=path("spatial_features", config["spatial_features"]["output"]),
        univariate_events=path("univariate_events", config["event_detection"]["events_output"]),
        univariate_flags=path("univariate_flags", config["event_detection"]["daily_flags_output"]),
        univariate_summary=path("univariate_summary", config["event_detection"]["summary_output"]),
        daily_patches=path("daily_patches", config["patch_detection"]["patches_output"]),
        daily_patch_summary=path("daily_patch_summary", config["patch_detection"]["daily_summary_output"]),
        daily_patch_build_summary=path("daily_patch_build_summary", config["patch_detection"]["json_summary_output"]),
        patch_observations=path("patch_observations", config["patch_tracking"]["observations_output"]),
        lineage_edges=path("lineage_edges", config["patch_tracking"]["edges_output"]),
        tracks=path("tracks", config["patch_tracking"]["tracks_output"]),
        event_families=path("event_families", config["patch_tracking"]["families_output"]),
        tracking_summary=path("tracking_summary", config["patch_tracking"]["json_summary_output"]),
    )


def _source_specs(inputs: BriefBuildInputs) -> list[SourceSpec]:
    return [
        SourceSpec("daily_metrics", inputs.daily_metrics, True, CORE_COLUMNS["daily_metrics"]),
        SourceSpec("qc_report", inputs.qc_report, True),
        SourceSpec("representativeness", inputs.representativeness, True, CORE_COLUMNS["representativeness"]),
        SourceSpec("series_bank", inputs.series_bank, False, frozenset({"date", "metric", "value", "status"})),
        SourceSpec("temporal_features", inputs.temporal_features, False, frozenset({"analysis_end_date", "window_days", "source_metric", "metric", "value", "status"})),
        SourceSpec("spatial_features", inputs.spatial_features, False, frozenset({"date", "variable", "metric", "value", "status"})),
        SourceSpec("univariate_events", inputs.univariate_events, False, frozenset({"event_id", "start_date", "end_date", "status"})),
        SourceSpec("univariate_flags", inputs.univariate_flags, False, frozenset({"date", "event_id", "status"})),
        SourceSpec("univariate_summary", inputs.univariate_summary, False),
        SourceSpec("daily_patches", inputs.daily_patches, False, frozenset({"date", "patch_id", "area_km2", "status"})),
        SourceSpec("daily_patch_summary", inputs.daily_patch_summary, False, frozenset({"date", "patch_count", "status"})),
        SourceSpec("daily_patch_build_summary", inputs.daily_patch_build_summary, False),
        SourceSpec("patch_observations", inputs.patch_observations, False, frozenset({"date", "local_patch_id", "track_id", "event_family_id", "status"})),
        SourceSpec("lineage_edges", inputs.lineage_edges, False, frozenset({"predecessor_date", "successor_date", "status"})),
        SourceSpec("tracks", inputs.tracks, False, frozenset({"track_id", "event_family_id", "start_date", "end_date", "status"})),
        SourceSpec("event_families", inputs.event_families, False, frozenset({"event_family_id", "start_date", "end_date", "status"})),
        SourceSpec("tracking_summary", inputs.tracking_summary, False),
    ]


def load_brief_products(inputs: BriefBuildInputs, root: Path) -> dict[str, LoadedProduct]:
    return {spec.source_id: load_source(spec, root) for spec in _source_specs(inputs)}


def resolve_analysis_date(daily_metrics: pd.DataFrame, requested: str) -> pd.Timestamp:
    """Resolve ``latest`` or require one exact valid ISO calendar date."""
    frame = daily_metrics.copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    except Exception as exc:
        raise ValueError("Daily metrics contain unreadable dates") from exc
    if frame.date.duplicated().any():
        duplicates = frame.loc[frame.date.duplicated(keep=False), "date"].dt.date.astype(str).tolist()
        raise ValueError(f"Daily metrics contain duplicate dates: {duplicates}")
    numeric_sst = pd.to_numeric(frame["mean_sst_c"], errors="coerce")
    valid = frame.loc[np.isfinite(numeric_sst.to_numpy(dtype=float))]
    if valid.empty:
        raise ValueError("Daily metrics contain no valid regional SST observations")
    if requested == "latest":
        return pd.Timestamp(valid.date.max()).normalize()
    try:
        parsed = datetime.strptime(requested, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Analysis date must be 'latest' or an ISO date in YYYY-MM-DD format") from exc
    selected = pd.Timestamp(parsed).normalize()
    rows = valid.loc[valid.date == selected]
    if rows.empty:
        raise ValueError(f"Explicit analysis date {requested} is unavailable or lacks valid regional SST")
    return selected


def _native(value: Any) -> Any:
    return to_json_compatible(value)


def _one_row(frame: pd.DataFrame, *, label: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one {label} row; found {len(frame)}")
    return frame.iloc[0]


def _availability(available: bool, reason: str | None, expected_path: str) -> dict[str, Any]:
    return {"available": bool(available), "reason": reason, "expected_path": expected_path}


def _add_flag(
    flags: list[dict[str, Any]],
    *,
    flag_id: str,
    severity: str,
    section: str,
    message_code: str,
    message_es: str,
    message_en: str,
    affected_fact_ids: list[str] | None = None,
    suppress: bool = False,
) -> None:
    if any(item["flag_id"] == flag_id for item in flags):
        return
    flags.append(
        {
            "flag_id": flag_id,
            "severity": severity,
            "section": section,
            "message_code": message_code,
            "message_es": message_es,
            "message_en": message_en,
            "affected_fact_ids": sorted(affected_fact_ids or []),
            "suppress_dependent_claims": suppress,
        }
    )


def _fact(
    registry: FactRegistry,
    products: Mapping[str, LoadedProduct],
    *,
    fact_id: str,
    section: str,
    name: str,
    value: Any,
    unit: str,
    source_id: str,
    source_column: str | None,
    selected_date: str,
    method: str,
    precision: int = 3,
    status: str = "valid",
    reason: str | None = None,
) -> dict[str, Any]:
    return registry.add(
        fact_id=fact_id,
        section=section,
        name=name,
        value=value,
        unit=unit,
        source_id=source_id,
        source_path=products[source_id].relative_path,
        source_column=source_column,
        selected_date=selected_date,
        calculation_method=method,
        precision=precision,
        status=status,
        reason_when_unavailable=reason,
    )


def _series_row(product: LoadedProduct, selected: pd.Timestamp, metric: str) -> pd.Series | None:
    rows = product.selected_rows(selected)
    if rows.empty or "metric" not in rows:
        return None
    rows = rows.loc[rows.metric.astype(str) == metric]
    return rows.iloc[0] if len(rows) else None


def _spatial_row(product: LoadedProduct, selected: pd.Timestamp, variable: str, metric: str) -> pd.Series | None:
    rows = product.selected_rows(selected)
    if rows.empty:
        return None
    rows = rows.loc[(rows.variable.astype(str) == variable) & (rows.metric.astype(str) == metric)]
    return rows.iloc[0] if len(rows) else None


def _temporal_row(product: LoadedProduct, selected: pd.Timestamp, window: int, metric: str) -> pd.Series | None:
    rows = product.selected_rows(selected)
    if rows.empty:
        return None
    rows = rows.loc[
        (pd.to_numeric(rows.window_days, errors="coerce") == window)
        & (rows.source_metric.astype(str) == "anomaly.weighted_mean")
        & (rows.metric.astype(str) == metric)
    ]
    return rows.iloc[0] if len(rows) else None


def _direction(value: Any, threshold: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "insufficient_data"
    if abs(float(value)) <= threshold:
        return "stable"
    return "increasing" if float(value) > 0 else "decreasing"


def _boundary(row: pd.Series) -> bool:
    return any(bool(row.get(column, False)) for column in (
        "touches_north_boundary", "touches_south_boundary",
        "touches_east_boundary", "touches_west_boundary",
        "touched_north_boundary", "touched_south_boundary",
        "touched_east_boundary", "touched_west_boundary",
    ))


def _limitations() -> list[dict[str, Any]]:
    rows = [
        ("experimental_product", "product", "Este producto es experimental.", "This product is experimental.", ["product"], "warning"),
        ("not_official_classification", "classification", "No constituye una clasificación oficial de la magnitud de El Niño Costero.", "It is not an official Coastal El Niño magnitude classification.", ["product", "regional_state", "univariate_event"], "warning"),
        ("method_dependence", "method", "Los resultados dependen de la climatología, los umbrales y la conectividad.", "Results depend on climatology, thresholds, and connectivity.", ["climatology", "daily_patches", "spatiotemporal_activity"], "info"),
        ("tracks_not_water_parcels", "interpretation", "Los tracks representan rasgos térmicos enlazados, no parcelas individuales de agua.", "Tracks represent linked thermal features, not individual water parcels.", ["spatiotemporal_activity"], "warning"),
        ("domain_boundaries", "spatial_domain", "Los rasgos que tocan los límites pueden extenderse fuera del dominio.", "Features touching domain boundaries may extend beyond the domain.", ["daily_patches", "spatiotemporal_activity"], "warning"),
        ("missing_data", "data_quality", "Los datos faltantes o inválidos pueden reducir la cobertura.", "Missing or invalid data may reduce coverage.", ["quality", "regional_state"], "info"),
        ("reference_period", "climatology", "La climatología diaria representa las condiciones de referencia de 1991–2020.", "The daily climatology represents 1991–2020 reference conditions.", ["climatology", "regional_state"], "info"),
        ("nominal_classes", "representativeness", "Las clases de representatividad son nominales, no ordinales.", "Representativeness classes are nominal, not ordinal.", ["representativeness"], "info"),
    ]
    return [
        {"limitation_id": identifier, "category": category, "statement_es": es, "statement_en": en, "affected_sections": sections, "severity": severity}
        for identifier, category, es, en, sections, severity in rows
    ]


def _generation_constraints() -> dict[str, Any]:
    return {
        "permitted_languages": ["es", "en"],
        "permitted_sections": ["regional_state", "recent_evolution", "spatial_structure", "representativeness", "univariate_event", "daily_patches", "spatiotemporal_activity", "quality", "limitations"],
        "prohibited_claims": [
            "official Coastal El Niño magnitude classification",
            "causal attribution not supported by inputs",
            "forecasts not present in the context",
            "biological or fisheries impacts not present in the context",
            "invented numerical values",
            "confidence probabilities derived from evidence_score",
            "claims that tracks are individual water masses or individual water parcels",
        ],
        "maximum_words_by_section": {"summary": 120, "regional_state": 100, "recent_evolution": 100, "spatial_structure": 100, "events": 120, "limitations": 80},
        "numerical_fact_policy": "Every numerical value included in the generated brief must correspond to a valid fact_id from fact_registry.",
        "uncertainty_policy": "Report quality flags and unavailable facts explicitly; do not convert evidence scores into probabilities.",
        "official_classification_policy": "Do not provide an official Coastal El Niño magnitude classification.",
        "citation_policy": "Cite the source_id and fact_id carried by the context; do not invent external citations.",
        "missing_data_policy": "Unavailable facts must not be inferred.",
    }


def build_brief_context(
    config: Mapping[str, Any],
    *,
    analysis_date: str = "latest",
    root: Path,
    inputs: BriefBuildInputs | None = None,
) -> tuple[dict[str, Any], dict[str, LoadedProduct]]:
    """Build a language-neutral context from local cached analytical products."""
    settings = config["brief_context"]
    products = load_brief_products(inputs or input_paths_from_config(config, root), root)
    metrics = products["daily_metrics"].frame.copy()
    metrics["date"] = pd.to_datetime(metrics.date, errors="raise").dt.normalize()
    selected = resolve_analysis_date(metrics, analysis_date)
    selected_iso = selected.date().isoformat()
    daily_row = _one_row(metrics.loc[metrics.date == selected], label="daily metrics")

    representative_rows = products["representativeness"].selected_rows(selected)
    representative_row = _one_row(representative_rows, label="representativeness")
    class_code = str(representative_row.get("class", "unclassified"))
    if class_code not in REPRESENTATIVENESS_CLASSES:
        raise ValueError(f"Unsupported representativeness class: {class_code}")

    qc = products["qc_report"].document
    missing_qc_keys = sorted(QC_REQUIRED_KEYS - set(qc))
    if missing_qc_keys:
        raise ValueError(f"QC report is structurally invalid; missing keys: {missing_qc_keys}")
    context = new_context(str(settings["schema_version"]))
    registry = FactRegistry()
    flags: list[dict[str, Any]] = []

    context["product"] = {
        "product_name": "Humboldt Ocean Watch",
        "product_short_name": "HOW",
        "product_version": __version__,
        "product_status": "experimental",
        "experimental_product": True,
        "region_name": "Niño 1+2",
        "region_code": "nino12",
        "spatial_bounds": {"longitude": list(config["region"]["longitude"]), "latitude": list(config["region"]["latitude"])},
        "source_dataset": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "source_variable": "analysed_sst",
        "institution": qc.get("institution"),
        "scientific_scope": (
            "Characterizes sea-surface temperature, temperature anomalies, spatial configuration, "
            "and thermal-event evolution. It is not an official Coastal El Niño magnitude classification."
        ),
    }
    available_start = pd.Timestamp(metrics.date.min()).date().isoformat()
    available_end = pd.Timestamp(metrics.date.max()).date().isoformat()
    generated = datetime.now(timezone.utc)
    latency = max(0, (generated.date() - selected.date()).days)
    spatial_dimensions = qc.get("spatial_dimensions", {})
    bounds = qc.get("spatial_bounds", {})
    spatial_resolution = None
    if all(key in spatial_dimensions for key in ("latitude", "longitude")) and all(key in bounds for key in ("latitude", "longitude")):
        spatial_resolution = {
            "latitude_degrees": abs(float(bounds["latitude"][1]) - float(bounds["latitude"][0])) / max(int(spatial_dimensions["latitude"]) - 1, 1),
            "longitude_degrees": abs(float(bounds["longitude"][1]) - float(bounds["longitude"][0])) / max(int(spatial_dimensions["longitude"]) - 1, 1),
            "method": "coordinate_range_divided_by_intervals",
        }
    context["analysis"] = {
        "requested_analysis_date": analysis_date,
        "resolved_analysis_date": selected_iso,
        "generated_at": generated.isoformat(),
        "available_start_date": available_start,
        "available_end_date": available_end,
        "data_latency_days": latency,
        "data_latency_reason": "generated date minus latest selected observation date",
        "temporal_resolution": qc.get("expected_frequency", config["quality_control"]["expected_frequency"]),
        "spatial_resolution": spatial_resolution,
        "data_mode": str(representative_row.get("data_mode", qc.get("data_mode", "unknown"))),
        "analysis_status": "valid",
    }

    availability_map = {
        "regional_metrics": "daily_metrics",
        "quality_control": "qc_report",
        "representativeness": "representativeness",
        "temporal_features": "temporal_features",
        "spatial_features": "spatial_features",
        "univariate_events": "univariate_events",
        "daily_patches": "daily_patches",
        "tracks": "tracks",
        "event_families": "event_families",
        "lineage_edges": "lineage_edges",
    }
    context["availability"] = {
        key: _availability(products[source].available, products[source].reason, products[source].relative_path)
        for key, source in availability_map.items()
    }
    climatology_method = str(daily_row.get("climatology_method"))
    context["availability"]["climatology"] = _availability(
        bool(climatology_method and climatology_method != "None"),
        None if climatology_method else "daily metrics do not identify a climatology",
        repository_relative(root / config["climatology"]["daily_path"], root),
    )
    for key, source in availability_map.items():
        if not products[source].available and not products[source].spec.required:
            _add_flag(
                flags, flag_id=f"missing_optional_product.{source}", severity="warning",
                section=key, message_code="missing_optional_product",
                message_es=f"El producto opcional {source} no está disponible.",
                message_en=f"Optional product {source} is unavailable.", suppress=True,
            )

    # Staleness is evaluated only for documented upstream relationships.  It is
    # intentionally based on file metadata and never triggers a recalculation.
    stale_relationships = {
        "representativeness": ("spatial_features", "representativeness", True),
        "temporal_features": ("series_bank", "recent_evolution", False),
        "univariate_events": ("series_bank", "univariate_event", False),
        "univariate_flags": ("series_bank", "univariate_event", False),
        "patch_observations": ("daily_patches", "spatiotemporal_activity", False),
        "lineage_edges": ("daily_patches", "spatiotemporal_activity", False),
        "tracks": ("daily_patches", "spatiotemporal_activity", False),
        "event_families": ("daily_patches", "spatiotemporal_activity", False),
        "tracking_summary": ("daily_patches", "spatiotemporal_activity", False),
    }
    for source, (upstream, section, is_core) in stale_relationships.items():
        if is_stale(products[source], products[upstream]):
            products[source].available = False
            products[source].reason = f"derived product is older than upstream product {upstream}"
            products[source].validation_status = "invalid"
            products[source].validation_messages.append(products[source].reason)
            _add_flag(
                flags,
                flag_id=f"stale_analytical_output.{source}",
                severity="error" if is_core else "warning",
                section=section,
                message_code="stale_analytical_output",
                message_es=f"El producto {source} parece desactualizado respecto de {upstream}.",
                message_en=f"Product {source} appears stale relative to {upstream}.",
                suppress=True,
            )

    # Refresh individual availability after staleness checks.
    for key, source in availability_map.items():
        context["availability"][key] = _availability(
            products[source].available,
            products[source].reason,
            products[source].relative_path,
        )

    combined_availability = {
        "univariate_events": products["univariate_events"].available and products["univariate_flags"].available,
        "daily_patches": products["daily_patches"].available and products["daily_patch_summary"].available,
        "tracks": products["patch_observations"].available and products["tracks"].available,
        "event_families": products["patch_observations"].available and products["event_families"].available,
        "lineage_edges": products["lineage_edges"].available,
    }
    for key, available in combined_availability.items():
        context["availability"][key]["available"] = bool(available)
        if not available and context["availability"][key]["reason"] is None:
            context["availability"][key]["reason"] = "one or more dependent cached products are unavailable or stale"
        if not available and not any(
            flag["section"] == key and flag["message_code"] in {"missing_optional_product", "stale_analytical_output"}
            for flag in flags
        ):
            _add_flag(
                flags,
                flag_id=f"missing_optional_dependency.{key}",
                severity="warning",
                section=key,
                message_code="missing_optional_product",
                message_es=f"Falta al menos un producto opcional necesario para {key}.",
                message_en=f"At least one optional product required by {key} is unavailable.",
                suppress=True,
            )

    context["provenance"] = {
        "repository_relative_paths": True,
        "sources": [products[key].provenance_record(selected) for key in sorted(products)],
    }

    coverage_percent = float(daily_row["valid_data_coverage_percent"])
    if not np.isfinite(coverage_percent) or not 0 <= coverage_percent <= 100:
        raise ValueError("Selected-date valid coverage must be finite and within [0, 100] percent")
    coverage = coverage_percent / 100.0
    qc_number_dates = int(qc.get("number_of_dates", len(metrics)))
    expected_dates = (pd.Timestamp(available_end) - pd.Timestamp(available_start)).days + 1
    temporal_completeness = min(1.0, qc_number_dates / expected_dates) if expected_dates else None
    coverage_ref = _fact(
        registry, products, fact_id="regional.valid_coverage", section="regional_state",
        name="valid_coverage", value=coverage, unit="fraction", source_id="daily_metrics",
        source_column="valid_data_coverage_percent", selected_date=selected_iso,
        method="percent_to_fraction", precision=4,
    )
    valid_cell_row = _series_row(products["series_bank"], selected, "sst.valid_cell_fraction")
    valid_cell_ref = _fact(
        registry, products, fact_id="quality.valid_cell_fraction", section="quality",
        name="valid_cell_fraction", value=valid_cell_row.get("value") if valid_cell_row is not None else None,
        unit="fraction", source_id="series_bank", source_column="value", selected_date=selected_iso,
        method="cached_series_bank_metric", precision=4,
        status=str(valid_cell_row.get("status", "unavailable")) if valid_cell_row is not None else "unavailable",
        reason=None if valid_cell_row is not None else "series-bank metric is unavailable",
    )
    completeness_ref = _fact(
        registry, products, fact_id="quality.temporal_completeness", section="quality",
        name="temporal_completeness", value=temporal_completeness, unit="fraction",
        source_id="qc_report", source_column="number_of_dates", selected_date=selected_iso,
        method="observed_dates_divided_by_calendar_dates", precision=4,
    )
    interval = qc.get("temporal_interval_summary", {})
    duplicated = qc.get("duplicated_dates", [])
    all_nan_dates = {str(value)[:10] for value in qc.get("all_nan_dates", [])}
    constant_dates = {str(value)[:10] for value in qc.get("constant_field_dates", [])}
    context["quality"] = {
        "overall_status": (
            qc.get("overall_status", "unavailable")
            if all(product.available for product in products.values() if product.spec.required)
            else "invalid"
        ),
        "valid_coverage": coverage_ref,
        "valid_cell_fraction": valid_cell_ref,
        "temporal_completeness": completeness_ref,
        "expected_frequency": qc.get("expected_frequency"),
        "irregular_interval_detected": not bool(interval.get("regular", False)),
        "duplicate_dates_detected": bool(duplicated),
        "all_nan_field": selected_iso in all_nan_dates,
        "constant_field": selected_iso in constant_dates,
        "core_inputs_valid": all(product.available for product in products.values() if product.spec.required),
        "optional_inputs_valid": all(product.available for product in products.values() if not product.spec.required),
        "warnings": list(qc.get("warnings", [])),
        "errors": list(qc.get("errors", [])),
    }
    if coverage < float(settings["minimum_valid_coverage"]):
        _add_flag(
            flags, flag_id="insufficient_coverage", severity="warning", section="quality",
            message_code="insufficient_coverage",
            message_es="La cobertura válida es inferior al mínimo configurado.",
            message_en="Valid coverage is below the configured minimum.",
            affected_fact_ids=[coverage_ref["fact_id"]], suppress=True,
        )
    if context["quality"]["irregular_interval_detected"]:
        _add_flag(
            flags, flag_id="irregular_temporal_sampling", severity="warning", section="quality",
            message_code="irregular_temporal_sampling", message_es="El muestreo temporal es irregular.",
            message_en="Temporal sampling is irregular.", suppress=True,
        )

    fallback_used = bool(daily_row.get("climatology_fallback_used", False))
    daily_path = repository_relative(root / config["climatology"]["daily_path"], root)
    context["climatology"] = {
        "available": True,
        "method": climatology_method,
        "reference_start": int(config["climatology"]["reference_start"]),
        "reference_end": int(config["climatology"]["reference_end"]),
        "reference_period": str(daily_row.get("climatology_reference_period")),
        "sampling_half_window_days": int(config["climatology"]["sampling_half_window_days"]),
        "smoothing_window_days": int(config["climatology"]["smoothing_window_days"]),
        "calendar_mapping": "stable month-day mapping on canonical leap year 2000",
        "leap_day_method": "February 29 is bin 60; non-leap years have no direct bin-60 sample but may contribute through the circular sampling window",
        "fallback_used": fallback_used,
        "fallback_method": config["climatology"]["fallback_method"],
        "climatology_file": daily_path if not fallback_used else repository_relative(root / config["climatology"]["monthly_path"], root),
        "compatibility_status": "warning" if fallback_used else "valid",
    }
    if fallback_used:
        _add_flag(
            flags, flag_id="monthly_climatology_fallback", severity="warning", section="climatology",
            message_code="monthly_climatology_fallback",
            message_es="Se utilizó la climatología mensual como alternativa explícita; no es equivalente a la referencia diaria.",
            message_en="The monthly climatology was used as an explicit fallback and is not equivalent to the daily reference.", suppress=False,
        )

    regional_mapping = [
        ("mean_sst_c", daily_row.get("mean_sst_c"), "degC", "daily_metrics", "mean_sst_c", "area_weighted_mean", 3),
        ("mean_anomaly_c", daily_row.get("mean_sst_anomaly_c"), "degC", "daily_metrics", "mean_sst_anomaly_c", "area_weighted_mean", 3),
        ("maximum_anomaly_c", daily_row.get("maximum_anomaly_c"), "degC", "daily_metrics", "maximum_anomaly_c", "spatial_maximum", 3),
        ("spatial_p90_anomaly_c", daily_row.get("p90_anomaly_c"), "degC", "daily_metrics", "p90_anomaly_c", "spatial_percentile", 3),
        ("mean_zscore", daily_row.get("mean_standardized_anomaly"), "dimensionless", "daily_metrics", "mean_standardized_anomaly", "area_weighted_mean", 3),
        ("area_anomaly_ge_1_fraction", float(daily_row.get("area_anomaly_ge_1c_percent")) / 100 if pd.notna(daily_row.get("area_anomaly_ge_1c_percent")) else None, "fraction", "daily_metrics", "area_anomaly_ge_1c_percent", "percent_to_fraction", 4),
        ("area_anomaly_ge_2_fraction", float(daily_row.get("area_anomaly_ge_2c_percent")) / 100 if pd.notna(daily_row.get("area_anomaly_ge_2c_percent")) else None, "fraction", "daily_metrics", "area_anomaly_ge_2c_percent", "percent_to_fraction", 4),
        ("area_anomaly_ge_3_fraction", float(daily_row.get("area_anomaly_ge_3c_percent")) / 100 if pd.notna(daily_row.get("area_anomaly_ge_3c_percent")) else None, "fraction", "daily_metrics", "area_anomaly_ge_3c_percent", "percent_to_fraction", 4),
        ("area_above_daily_p90_fraction", float(daily_row.get("area_above_climatological_p90_percent")) / 100 if pd.notna(daily_row.get("area_above_climatological_p90_percent")) else None, "fraction", "daily_metrics", "area_above_climatological_p90_percent", "percent_to_fraction", 4),
        ("area_zscore_ge_2_fraction", float(daily_row.get("area_zscore_ge_2_percent")) / 100 if pd.notna(daily_row.get("area_zscore_ge_2_percent")) else None, "fraction", "daily_metrics", "area_zscore_ge_2_percent", "percent_to_fraction", 4),
        ("warm_centroid_latitude", daily_row.get("warm_centroid_latitude"), "degree_latitude", "daily_metrics", "warm_centroid_latitude", "warm_anomaly_weighted_centroid", 4),
        ("warm_centroid_longitude", daily_row.get("warm_centroid_longitude"), "degree_longitude", "daily_metrics", "warm_centroid_longitude", "warm_anomaly_weighted_centroid", 4),
    ]
    median_row = _series_row(products["series_bank"], selected, "anomaly.weighted_median")
    p10_row = _series_row(products["series_bank"], selected, "anomaly.spatial_p10")
    max_z_row = _series_row(products["series_bank"], selected, "zscore.spatial_maximum")
    regional_mapping.extend([
        ("median_anomaly_c", median_row.get("value") if median_row is not None else None, "degC", "series_bank", "value", "weighted_median", 3),
        ("spatial_p10_anomaly_c", p10_row.get("value") if p10_row is not None else None, "degC", "series_bank", "value", "spatial_percentile", 3),
        ("maximum_zscore", max_z_row.get("value") if max_z_row is not None else None, "dimensionless", "series_bank", "value", "spatial_maximum", 3),
    ])
    context["regional_state"] = {"valid_coverage": coverage_ref}
    for name, value, unit, source, column, method, precision in regional_mapping:
        context["regional_state"][name] = _fact(
            registry, products, fact_id=f"regional.{name}", section="regional_state", name=name,
            value=value, unit=unit, source_id=source, source_column=column,
            selected_date=selected_iso, method=method, precision=precision,
            status="valid" if value is not None and pd.notna(value) else "unavailable",
        )

    # Deterministic recent differences use exact calendar dates; existing temporal features are reused.
    recent: dict[str, Any] = {"windows": []}
    thresholds = settings["stable_change_thresholds"]
    change_columns = {
        "mean_sst_c": ("mean_sst_c", 1.0, "degC"),
        "mean_anomaly_c": ("mean_sst_anomaly_c", 1.0, "degC"),
        "warm_area_fraction": ("area_anomaly_ge_2c_percent", 0.01, "fraction_change"),
        "valid_coverage": ("valid_data_coverage_percent", 0.01, "fraction_change"),
        "centroid_latitude": ("warm_centroid_latitude", 1.0, "degree_latitude"),
        "centroid_longitude": ("warm_centroid_longitude", 1.0, "degree_longitude"),
    }
    for window in settings["recent_windows_days"]:
        window = int(window)
        target = selected - pd.Timedelta(days=window)
        previous = metrics.loc[metrics.date == target]
        entry: dict[str, Any] = {
            "window_days": window,
            "window_start_date": target.date().isoformat(),
            "window_end_date": selected_iso,
            "differences": {},
        }
        for name, (column, scale, unit) in change_columns.items():
            value = None
            if len(previous) == 1 and pd.notna(previous.iloc[0].get(column)) and pd.notna(daily_row.get(column)):
                value = (float(daily_row[column]) - float(previous.iloc[0][column])) * scale
            ref = _fact(
                registry, products, fact_id=f"recent.{window}d.change_{name}", section="recent_evolution",
                name=f"change_{name}", value=value, unit=unit, source_id="daily_metrics",
                source_column=column, selected_date=selected_iso, method="exact_calendar_endpoint_difference",
                precision=4, status="valid" if value is not None else "insufficient",
                reason=None if value is not None else "exact window endpoint is unavailable",
            )
            entry["differences"][f"change_{name}"] = ref
        threshold = float(thresholds["mean_anomaly_c"])
        direction_thresholds = {
            "change_mean_sst_c": float(thresholds["mean_sst_c"]),
            "change_mean_anomaly_c": threshold,
            "change_warm_area_fraction": float(thresholds["warm_area_fraction"]),
            "change_valid_coverage": float(thresholds["valid_coverage"]),
            "change_centroid_latitude": float(thresholds["centroid_latitude_degrees"]),
            "change_centroid_longitude": float(thresholds["centroid_longitude_degrees"]),
        }
        entry["directions"] = {
            name: _direction(entry["differences"][name]["value"], stable_threshold)
            for name, stable_threshold in direction_thresholds.items()
        }
        # Backward-compatible window summary, explicitly anchored to mean anomaly.
        entry["direction"] = entry["directions"]["change_mean_anomaly_c"]
        entry["direction_metric"] = "change_mean_anomaly_c"
        temporal_metrics = {
            "temporal_slope": ("ols_slope", "slope_degC_per_day"),
            "variability": ("standard_deviation", "degC"),
            "instability": ("rmssd", "degC"),
            "persistence": ("lag1_autocorrelation", "dimensionless"),
        }
        for name, (metric_name, unit) in temporal_metrics.items():
            temporal_row = _temporal_row(products["temporal_features"], selected, window, metric_name)
            value = temporal_row.get("value") if temporal_row is not None else None
            entry[name] = _fact(
                registry, products, fact_id=f"recent.{window}d.{name}", section="recent_evolution",
                name=name, value=value, unit=unit, source_id="temporal_features", source_column="value",
                selected_date=selected_iso, method=f"cached_temporal_feature:{metric_name}", precision=4,
                status=str(temporal_row.get("status", "unavailable")) if temporal_row is not None else "unavailable",
                reason=None if temporal_row is not None else "cached temporal feature is unavailable",
            )
        recent["windows"].append(entry)
    context["recent_evolution"] = recent

    spatial_sources: list[tuple[str, Any, str, str, str, str]] = []
    for name, metric, unit in (
        ("spatial_mad_c", "anomaly.weighted_mad", "degC"),
        ("spatial_iqr_c", "anomaly.weighted_iqr", "degC"),
        ("p90_minus_p10_c", "anomaly.p90_minus_p10", "degC"),
        ("latitudinal_gradient", "anomaly.latitudinal_gradient", "degC_per_degree"),
        ("longitudinal_gradient", "anomaly.longitudinal_gradient", "degC_per_degree"),
    ):
        row = _series_row(products["series_bank"], selected, metric)
        spatial_sources.append((name, row.get("value") if row is not None else None, unit, "series_bank", "value", f"cached_series_bank_metric:{metric}"))
    for name, metric, unit in (
        ("moran_i", f"moran_i_{config['spatial_features']['connectivity']}", "index"),
        ("local_variability_summary", "local_spatial_standard_deviation", "degC"),
        ("thermal_gradient_summary", "thermal_gradient_magnitude", "degC_per_degree"),
    ):
        row = _spatial_row(products["spatial_features"], selected, "anomaly", metric)
        spatial_sources.append((name, row.get("value") if row is not None else None, unit, "spatial_features", "value", f"cached_spatial_feature:{metric}"))
    for name in (
        "positive_fraction", "negative_fraction", "neutral_fraction", "sign_coherence",
        "signal_heterogeneity_ratio", "mean_median_difference", "dominant_patch_fraction",
        "patch_density", "coexistence_index", "spatial_compensation_index",
    ):
        spatial_sources.append((name, representative_row.get(name), "fraction" if "fraction" in name or name in {"sign_coherence", "coexistence_index", "spatial_compensation_index"} else "index", "representativeness", name, "cached_representativeness_metric"))
    spatial_sources.insert(0, ("spatial_standard_deviation_c", representative_row.get("spatial_standard_deviation"), "degC", "representativeness", "spatial_standard_deviation", "cached_representativeness_metric"))
    context["spatial_structure"] = {"moran_connectivity": config["spatial_features"]["connectivity"]}
    for name, value, unit, source, column, method in spatial_sources:
        context["spatial_structure"][name] = _fact(
            registry, products, fact_id=f"spatial.{name}", section="spatial_structure", name=name,
            value=value, unit=unit, source_id=source, source_column=column,
            selected_date=selected_iso, method=method, precision=4,
            status="valid" if value is not None and pd.notna(value) else "unavailable",
        )

    evidence_ref = _fact(
        registry, products, fact_id="representativeness.evidence_score", section="representativeness",
        name="evidence_score", value=representative_row.get("evidence_score"), unit="fraction",
        source_id="representativeness", source_column="evidence_score", selected_date=selected_iso,
        method="ordered_rule_evidence", precision=3,
        status=str(representative_row.get("status", "unavailable")),
    )
    rep_coverage_ref = _fact(
        registry, products, fact_id="representativeness.valid_coverage", section="representativeness",
        name="valid_coverage", value=representative_row.get("valid_coverage"), unit="fraction",
        source_id="representativeness", source_column="valid_coverage", selected_date=selected_iso,
        method="area_weighted_valid_coverage", precision=4,
        status=str(representative_row.get("status", "unavailable")),
    )
    threshold_refs: dict[str, Any] = {}
    for name, value in config["representativeness"].items():
        if name in {"output", "epsilon"}:
            continue
        unit = "degC" if name.endswith("_c") else ("fraction" if "fraction" in name or name in {"minimum_coverage", "coherent_sign_fraction", "dominant_patch_fraction"} else "index")
        threshold_refs[name] = _fact(
            registry, products, fact_id=f"representativeness.threshold.{name}", section="representativeness",
            name=name, value=value, unit=unit, source_id="representativeness",
            source_column=None, selected_date=selected_iso, method="configured_classification_threshold", precision=3,
        )
    context["representativeness"] = {
        "available": products["representativeness"].available,
        "class_code": class_code,
        "class_label_es": human_label(class_code, "es"),
        "class_label_en": human_label(class_code, "en"),
        "evidence_score": evidence_ref,
        "triggered_rule": _native(representative_row.get("triggered_rule")),
        "valid_coverage": rep_coverage_ref,
        "input_metrics": {name: context["spatial_structure"][name] for name in (
            "spatial_standard_deviation_c", "positive_fraction", "negative_fraction", "neutral_fraction",
            "sign_coherence", "signal_heterogeneity_ratio", "mean_median_difference",
            "dominant_patch_fraction", "patch_density", "coexistence_index", "spatial_compensation_index",
        )},
        "threshold_values": threshold_refs,
        "interpretation_code": f"representativeness.{class_code}",
        "classification_metadata": {
            "nominal_classes": True,
            "ordinal_severity_ranking": False,
            "evidence_score_is_probability": False,
        },
        "status": str(representative_row.get("status")) if products["representativeness"].available else "invalid",
        "reason": (
            _native(representative_row.get("reason"))
            if products["representativeness"].available
            else products["representativeness"].reason
        ),
    }

    # Univariate event state comes only from cached daily flags and event table.
    flags_rows = products["univariate_flags"].selected_rows(selected)
    if len(flags_rows) > 1:
        raise ValueError(f"Expected at most one univariate event-flag row; found {len(flags_rows)}")
    flag_row = flags_rows.iloc[0] if len(flags_rows) else None
    active_event_id = None if flag_row is None or pd.isna(flag_row.get("event_id")) else str(flag_row.get("event_id"))
    event_rows = products["univariate_events"].frame
    event_match = event_rows.loc[event_rows.event_id.astype(str) == active_event_id] if active_event_id and not event_rows.empty else event_rows.iloc[0:0]
    if len(event_match) > 1:
        raise ValueError(f"Event ID {active_event_id} is duplicated in the event catalogue")
    event_row = event_match.iloc[0] if len(event_match) else None
    if active_event_id and event_row is None:
        products["univariate_events"].available = False
        products["univariate_events"].reason = "active event ID is missing from the event catalogue"
        products["univariate_events"].validation_status = "invalid"
        context["availability"]["univariate_events"] = _availability(
            False, products["univariate_events"].reason, products["univariate_events"].relative_path,
        )
        _add_flag(
            flags, flag_id="missing_referenced_event", severity="error", section="univariate_event",
            message_code="missing_referenced_event",
            message_es="El evento activo no existe en el catálogo de eventos.",
            message_en="The active event ID is missing from the event catalogue.", suppress=True,
        )
    active_event = event_row is not None
    event_section: dict[str, Any] = {
        "product_available": context["availability"]["univariate_events"]["available"],
        "active": active_event,
        "event_id": active_event_id,
        "source_variable": _native(flag_row.get("source_variable")) if flag_row is not None else None,
        "threshold_type": _native(event_row.get("threshold_type")) if event_row is not None else None,
        "direction": _native(event_row.get("direction")) if event_row is not None else None,
        "event_start_date": pd.Timestamp(event_row.get("start_date")).date().isoformat() if event_row is not None else None,
        "event_end_date": pd.Timestamp(event_row.get("end_date")).date().isoformat() if event_row is not None else None,
        "peak_date": pd.Timestamp(event_row.get("peak_date")).date().isoformat() if event_row is not None else None,
        "severity_class": _native(event_row.get("severity_class")) if event_row is not None else None,
        "status": str(flag_row.get("status")) if flag_row is not None else "unavailable",
        "reason": _native(flag_row.get("reason")) if flag_row is not None else products["univariate_flags"].reason,
    }
    event_values = [
        ("threshold_value_on_date", flag_row.get("threshold") if flag_row is not None else None, "degC", "univariate_flags", "threshold"),
        ("current_value", flag_row.get("value") if flag_row is not None else None, "degC", "univariate_flags", "value"),
        ("current_intensity", flag_row.get("intensity") if flag_row is not None else None, "degC", "univariate_flags", "intensity"),
        ("event_day", flag_row.get("event_day") if flag_row is not None else None, "day", "univariate_flags", "event_day"),
        ("valid_coverage", flag_row.get("valid_coverage") if flag_row is not None else None, "fraction", "univariate_flags", "valid_coverage"),
        ("duration_calendar_days", event_row.get("duration_calendar_days") if event_row is not None else None, "day", "univariate_events", "duration_calendar_days"),
        ("duration_observed_days", event_row.get("duration_observed_days") if event_row is not None else None, "day", "univariate_events", "duration_observed_days"),
        ("mean_intensity", event_row.get("mean_intensity") if event_row is not None else None, "degC", "univariate_events", "mean_intensity"),
        ("maximum_intensity", event_row.get("maximum_intensity") if event_row is not None else None, "degC", "univariate_events", "maximum_intensity"),
        ("cumulative_intensity", event_row.get("cumulative_intensity") if event_row is not None else None, "degC_day", "univariate_events", "cumulative_intensity"),
    ]
    for name, value, unit, source, column in event_values:
        event_section[name] = _fact(
            registry, products, fact_id=f"event.{name}", section="univariate_event", name=name,
            value=value, unit=unit, source_id=source, source_column=column, selected_date=selected_iso,
            method="cached_event_detection_output", precision=3,
            status="valid" if value is not None and pd.notna(value) else "unavailable",
            reason=None if active_event else "no active event on selected date",
        )
    event_section["threshold_value"] = (
        event_section["threshold_value_on_date"]
        if event_row is not None and str(event_row.get("threshold_type")) == "fixed"
        else None
    )
    context["univariate_event"] = event_section
    if not active_event:
        _add_flag(
            flags, flag_id="no_active_event", severity="info", section="univariate_event",
            message_code="no_active_event", message_es="No hay un evento univariado activo en la fecha seleccionada.",
            message_en="No univariate event is active on the selected date.", suppress=False,
        )

    # Daily patch summary and a bounded list of area-ranked patches.
    patch_rows = products["daily_patches"].selected_rows(selected)
    patch_summary_rows = products["daily_patch_summary"].selected_rows(selected)
    if len(patch_summary_rows) > 1:
        raise ValueError(f"Expected at most one daily patch-summary row; found {len(patch_summary_rows)}")
    patch_summary_row = patch_summary_rows.iloc[0] if len(patch_summary_rows) else None
    patch_count = int(patch_summary_row.get("patch_count", 0)) if patch_summary_row is not None else None
    if patch_count is not None and patch_count != len(patch_rows):
        products["daily_patches"].available = False
        products["daily_patches"].reason = "selected-date patch count disagrees with daily summary"
        products["daily_patches"].validation_status = "invalid"
        products["daily_patches"].validation_messages.append("selected-date patch count disagrees with daily summary")
        context["availability"]["daily_patches"] = _availability(
            False, products["daily_patches"].reason, products["daily_patches"].relative_path,
        )
        _add_flag(
            flags, flag_id="patch_count_mismatch", severity="error", section="daily_patches",
            message_code="patch_count_mismatch", message_es="El conteo de parches no coincide entre productos.",
            message_en="Patch count disagrees between cached products.", suppress=True,
        )
    daily_patch_section: dict[str, Any] = {
        "product_available": context["availability"]["daily_patches"]["available"] and patch_summary_row is not None,
        "status": (str(patch_summary_row.get("status")) if products["daily_patches"].available and patch_summary_row is not None else "invalid"),
        "reason": _native(patch_summary_row.get("reason")) if patch_summary_row is not None else products["daily_patch_summary"].reason,
        "top_patches": [],
    }
    patch_summary_values = [
        ("patch_count", patch_count, "count"), ("total_patch_area_km2", patch_summary_row.get("total_patch_area_km2") if patch_summary_row is not None else None, "km2"),
        ("threshold_area_fraction", patch_summary_row.get("threshold_area_fraction") if patch_summary_row is not None else None, "fraction"),
        ("largest_patch_area_km2", patch_summary_row.get("largest_patch_area_km2") if patch_summary_row is not None else None, "km2"),
        ("largest_patch_fraction", patch_summary_row.get("largest_patch_fraction") if patch_summary_row is not None else None, "fraction"),
        ("mean_patch_area_km2", patch_summary_row.get("mean_patch_area_km2") if patch_summary_row is not None else None, "km2"),
        ("fragmentation_index", patch_summary_row.get("fragmentation_index") if patch_summary_row is not None else None, "index"),
        ("dominant_patch_fraction", patch_summary_row.get("dominant_patch_fraction") if patch_summary_row is not None else None, "fraction"),
        ("maximum_patch_intensity", patch_summary_row.get("maximum_patch_intensity") if patch_summary_row is not None else None, "degC"),
        ("regional_patch_centroid_latitude", patch_summary_row.get("area_weighted_centroid_latitude") if patch_summary_row is not None else None, "degree_latitude"),
        ("regional_patch_centroid_longitude", patch_summary_row.get("area_weighted_centroid_longitude") if patch_summary_row is not None else None, "degree_longitude"),
        ("valid_coverage", patch_summary_row.get("valid_coverage") if patch_summary_row is not None else None, "fraction"),
    ]
    for name, value, unit in patch_summary_values:
        source_column = {"regional_patch_centroid_latitude": "area_weighted_centroid_latitude", "regional_patch_centroid_longitude": "area_weighted_centroid_longitude"}.get(name, name)
        daily_patch_section[name] = _fact(
            registry, products, fact_id=f"patches.{name}", section="daily_patches", name=name,
            value=value, unit=unit, source_id="daily_patch_summary", source_column=source_column,
            selected_date=selected_iso, method="cached_daily_patch_summary", precision=3,
            status="valid" if value is not None and pd.notna(value) else "unavailable",
        )
    observations_today = products["patch_observations"].selected_rows(selected)
    duplicated_assignments = (
        observations_today["local_patch_id"].duplicated().any()
        if not observations_today.empty and "local_patch_id" in observations_today
        else False
    )
    if duplicated_assignments:
        _add_flag(
            flags, flag_id="duplicate_patch_assignment", severity="error",
            section="spatiotemporal_activity", message_code="duplicate_patch_assignment",
            message_es="Un parche local tiene múltiples asignaciones de seguimiento.",
            message_en="A local patch has multiple tracking assignments.", suppress=True,
        )
    assignments = (
        observations_today.drop_duplicates("local_patch_id", keep="first").set_index("local_patch_id")
        if not observations_today.empty
        else pd.DataFrame()
    )
    ranked_patch_rows = (
        patch_rows.sort_values("area_km2", ascending=False)
        if "area_km2" in patch_rows
        else patch_rows
    )
    for rank, (_, row) in enumerate(ranked_patch_rows.head(int(settings["maximum_top_patches"])).iterrows(), start=1):
        patch_id = int(row.patch_id)
        assignment = assignments.loc[patch_id] if not assignments.empty and patch_id in assignments.index else None
        record: dict[str, Any] = {
            "patch_id": patch_id,
            "track_id": _native(assignment.get("track_id")) if assignment is not None else None,
            "event_family_id": _native(assignment.get("event_family_id")) if assignment is not None else None,
            "touches_domain_boundary": _boundary(row),
            "fact_ids": {},
        }
        top_fields = [
            ("area_km2", "km2"), ("area_fraction_of_threshold_total", "fraction"),
            ("centroid_latitude", "degree_latitude"), ("centroid_longitude", "degree_longitude"),
            ("mean_source_value", "degC"), ("maximum_source_value", "degC"),
            ("mean_exceedance", "degC"), ("maximum_exceedance", "degC"),
            ("compactness", "index"), ("elongation", "index"),
        ]
        for name, unit in top_fields:
            value = row.get(name)
            ref = _fact(
                registry, products, fact_id=f"patches.top{rank}.{name}", section="daily_patches",
                name=name, value=value, unit=unit, source_id="daily_patches", source_column=name,
                selected_date=selected_iso, method="area_ranked_patch_record", precision=3,
                status="valid" if value is not None and pd.notna(value) else "unavailable",
            )
            record[name] = ref["value"]
            record["fact_ids"][name] = ref["fact_id"]
        daily_patch_section["top_patches"].append(record)
    context["daily_patches"] = daily_patch_section
    if patch_count == 0:
        _add_flag(
            flags, flag_id="no_daily_patches", severity="info", section="daily_patches",
            message_code="no_daily_patches", message_es="No se retuvieron parches diarios en la fecha seleccionada.",
            message_en="No daily patches were retained on the selected date.", suppress=False,
        )

    # Active patch observations define track/family activity on the date.
    active_track_ids = sorted(observations_today.track_id.dropna().astype(str).unique()) if "track_id" in observations_today else []
    active_family_ids = sorted(observations_today.event_family_id.dropna().astype(str).unique()) if "event_family_id" in observations_today else []
    tracks = products["tracks"].frame
    families = products["event_families"].frame
    if not tracks.empty and tracks["track_id"].astype(str).duplicated().any():
        _add_flag(
            flags, flag_id="duplicate_track_id", severity="error",
            section="spatiotemporal_activity", message_code="duplicate_track_id",
            message_es="El catálogo contiene identificadores de track duplicados.",
            message_en="The track catalogue contains duplicate track IDs.", suppress=True,
        )
    if not families.empty and families["event_family_id"].astype(str).duplicated().any():
        _add_flag(
            flags, flag_id="duplicate_event_family_id", severity="error",
            section="spatiotemporal_activity", message_code="duplicate_event_family_id",
            message_es="El catálogo contiene identificadores de familia duplicados.",
            message_en="The event-family catalogue contains duplicate IDs.", suppress=True,
        )
    missing_tracks = sorted(set(active_track_ids) - set(tracks.get("track_id", pd.Series(dtype=str)).astype(str)))
    missing_families = sorted(set(active_family_ids) - set(families.get("event_family_id", pd.Series(dtype=str)).astype(str)))
    if missing_tracks:
        _add_flag(flags, flag_id="missing_referenced_track", severity="error", section="spatiotemporal_activity", message_code="missing_referenced_track", message_es="Hay tracks activos que no existen en el catálogo.", message_en="Active track IDs are missing from the track catalogue.", suppress=True)
    if missing_families:
        _add_flag(flags, flag_id="missing_referenced_family", severity="error", section="spatiotemporal_activity", message_code="missing_referenced_family", message_es="Hay familias activas que no existen en el catálogo.", message_en="Active family IDs are missing from the family catalogue.", suppress=True)
    if not observations_today.empty and not tracks.empty:
        track_family = (
            tracks.drop_duplicates("track_id", keep="first")
            .set_index(tracks.drop_duplicates("track_id", keep="first")["track_id"].astype(str))["event_family_id"]
            .astype(str)
        )
        inconsistent_assignments = [
            str(row.track_id)
            for _, row in observations_today.dropna(subset=["track_id", "event_family_id"]).iterrows()
            if str(row.track_id) in track_family.index
            and str(row.event_family_id) != str(track_family.loc[str(row.track_id)])
        ]
        if inconsistent_assignments:
            _add_flag(
                flags, flag_id="track_family_assignment_mismatch", severity="error",
                section="spatiotemporal_activity", message_code="track_family_assignment_mismatch",
                message_es="La familia asignada al parche no coincide con el catálogo de tracks.",
                message_en="A patch family assignment disagrees with the track catalogue.", suppress=True,
            )
    patch_ids = set(patch_rows.get("patch_id", pd.Series(dtype=int)).dropna().astype(int))
    observation_patch_ids = set(observations_today.get("local_patch_id", pd.Series(dtype=int)).dropna().astype(int))
    if products["patch_observations"].available and patch_ids != observation_patch_ids:
        _add_flag(
            flags, flag_id="patch_assignment_mismatch", severity="error",
            section="spatiotemporal_activity", message_code="patch_assignment_mismatch",
            message_es="Las asignaciones de track y familia no coinciden con los parches de la fecha.",
            message_en="Track and family assignments do not match selected-date patches.",
            suppress=True,
        )
    types = observations_today.get("node_event_type", pd.Series(dtype=str)).astype(str)
    type_names = ["appearance", "termination", "continuation", "split_parent", "split_child", "merge_parent", "merge_child", "complex_branch", "isolated_single_day"]
    activity_products_available = (
        products["patch_observations"].available
        and products["tracks"].available
        and products["event_families"].available
    )
    spatiotemporal: dict[str, Any] = {
        "product_available": activity_products_available,
        "active_split_event": bool(types.isin(["split_parent", "split_child"]).any()),
        "active_merge_event": bool(types.isin(["merge_parent", "merge_child"]).any()),
        "active_complex_branch": bool(types.eq("complex_branch").any()),
        "status": "valid" if activity_products_available else "unavailable",
        "reason": (
            None
            if activity_products_available
            else "one or more cached tracking products are unavailable or stale"
        ),
        "top_tracks": [],
        "top_event_families": [],
    }
    counts = {
        "active_track_count": len(active_track_ids),
        "active_family_count": len(active_family_ids),
        "active_patch_observation_count": len(observations_today),
        **{f"{name.replace('isolated_single_day', 'isolated_patch')}_count": int(types.eq(name).sum()) for name in type_names},
    }
    for name, value in counts.items():
        spatiotemporal[name] = _fact(
            registry, products, fact_id=f"activity.{name}", section="spatiotemporal_activity",
            name=name, value=value, unit="count", source_id="patch_observations",
            source_column="node_event_type" if name.endswith("_count") and not name.startswith("active_") else None,
            selected_date=selected_iso, method="selected_date_unique_count", precision=0,
        )
    active_tracks = tracks.loc[tracks.track_id.astype(str).isin(active_track_ids)].sort_values("maximum_area_km2", ascending=False) if active_track_ids and not tracks.empty else tracks.iloc[0:0]
    for rank, (_, row) in enumerate(active_tracks.head(int(settings["maximum_top_tracks"])).iterrows(), start=1):
        current = observations_today.loc[observations_today.track_id.astype(str) == str(row.track_id)].sort_values("area_km2", ascending=False).iloc[0]
        record: dict[str, Any] = {
            "track_id": str(row.track_id), "event_family_id": str(row.event_family_id),
            "start_date": pd.Timestamp(row.start_date).date().isoformat(), "end_date": pd.Timestamp(row.end_date).date().isoformat(),
            "current_node_event_type": str(current.node_event_type), "touches_domain_boundary": _boundary(row), "fact_ids": {},
        }
        track_fields = [
            ("observed_days", row.get("observed_days"), "day", "observed_days"),
            ("duration_calendar_days", row.get("duration_calendar_days"), "day", "duration_calendar_days"),
            ("current_area_km2", current.get("area_km2"), "km2", "area_km2"),
            ("maximum_area_km2", row.get("maximum_area_km2"), "km2", "maximum_area_km2"),
            ("cumulative_severity", row.get("cumulative_severity"), "degC_km2_day", "cumulative_severity"),
            ("trajectory_length_km", row.get("trajectory_length_km"), "km", "trajectory_length_km"),
            ("net_displacement_km", row.get("net_displacement_km"), "km", "net_displacement_km"),
            ("mean_speed_km_per_day", row.get("mean_speed_km_per_day"), "km_per_day", "mean_speed_km_per_day"),
            ("maximum_speed_km_per_day", row.get("maximum_speed_km_per_day"), "km_per_day", "maximum_speed_km_per_day"),
        ]
        for name, value, unit, column in track_fields:
            source = "patch_observations" if name == "current_area_km2" else "tracks"
            ref = _fact(registry, products, fact_id=f"activity.top_track{rank}.{name}", section="spatiotemporal_activity", name=name, value=value, unit=unit, source_id=source, source_column=column, selected_date=selected_iso, method="active_track_area_rank", precision=3, status="valid" if value is not None and pd.notna(value) else "unavailable")
            record[name] = ref["value"]
            record["fact_ids"][name] = ref["fact_id"]
        spatiotemporal["top_tracks"].append(record)
    active_families = families.loc[families.event_family_id.astype(str).isin(active_family_ids)].sort_values("maximum_total_daily_area_km2", ascending=False) if active_family_ids and not families.empty else families.iloc[0:0]
    for rank, (_, row) in enumerate(active_families.head(int(settings["maximum_top_families"])).iterrows(), start=1):
        record = {
            "event_family_id": str(row.event_family_id),
            "start_date": pd.Timestamp(row.start_date).date().isoformat(), "end_date": pd.Timestamp(row.end_date).date().isoformat(),
            "touches_domain_boundary": bool(float(row.get("boundary_contact_day_count", 0) or 0) > 0), "fact_ids": {},
        }
        family_fields = [
            ("duration_calendar_days", "day"), ("track_count", "count"),
            ("unique_patch_observation_count", "count"), ("split_count", "count"),
            ("merge_count", "count"), ("complex_branch_count", "count"),
            ("maximum_total_daily_area_km2", "km2"), ("family_area_days_km2_days", "km2_day"),
            ("cumulative_severity", "degC_km2_day"), ("family_trajectory_length_km", "km"),
            ("maximum_family_extent_km", "km"),
        ]
        rename = {"unique_patch_observation_count": "patch_observation_count", "maximum_total_daily_area_km2": "maximum_daily_area_km2"}
        for column, unit in family_fields:
            name = rename.get(column, column)
            value = row.get(column)
            ref = _fact(registry, products, fact_id=f"activity.top_family{rank}.{name}", section="spatiotemporal_activity", name=name, value=value, unit=unit, source_id="event_families", source_column=column, selected_date=selected_iso, method="active_family_area_rank", precision=3, status="valid" if value is not None and pd.notna(value) else "unavailable")
            record[name] = ref["value"]
            record["fact_ids"][name] = ref["fact_id"]
        spatiotemporal["top_event_families"].append(record)
    if any(_boundary(row) for _, row in active_tracks.iterrows()):
        _add_flag(flags, flag_id="track_touches_domain_boundary", severity="warning", section="spatiotemporal_activity", message_code="track_touches_domain_boundary", message_es="Al menos un track activo toca el límite del dominio.", message_en="At least one active track touches the domain boundary.", suppress=False)
    context["spatiotemporal_activity"] = spatiotemporal

    # Cross-product metadata consistency is explicit and never repaired silently.
    methods = {
        str(value)
        for value in [
            daily_row.get("climatology_method"), representative_row.get("climatology_method"),
            *(patch_rows.get("climatology_method", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(observations_today.get("climatology_method", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(active_families.get("climatology_method", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(event_match.get("climatology_method", pd.Series(dtype=str)).dropna().unique().tolist()),
        ]
        if pd.notna(value)
    }
    modes = {
        str(value)
        for value in [
            representative_row.get("data_mode"),
            *(patch_rows.get("data_mode", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(observations_today.get("data_mode", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(active_families.get("data_mode", pd.Series(dtype=str)).dropna().unique().tolist()),
            *(event_match.get("data_mode", pd.Series(dtype=str)).dropna().unique().tolist()),
        ]
        if pd.notna(value)
    }
    if len(methods) > 1:
        raise ValueError(f"Core products use incompatible climatology methods: {sorted(methods)}")
    if len(modes) > 1:
        raise ValueError(f"Products use incompatible data modes: {sorted(modes)}")

    context["fact_registry"] = registry.records()
    # Provenance reports the columns that actually support registered facts in
    # addition to structural columns needed to select and validate records.
    used_by_source: dict[str, set[str]] = {}
    for fact in context["fact_registry"]:
        source_column = fact.get("source_column")
        if source_column:
            used_by_source.setdefault(str(fact["source_id"]), set()).add(str(source_column))
    for source in context["provenance"]["sources"]:
        source_id = str(source["source_id"])
        structural = set(source.get("schema_columns_used", []))
        source["schema_columns_used"] = sorted(structural | used_by_source.get(source_id, set()))
    context["quality_flags"] = sorted(flags, key=lambda item: item["flag_id"])
    context["limitations"] = _limitations()
    context["generation_constraints"] = _generation_constraints()
    if any(flag["severity"] == "error" for flag in flags):
        context["analysis"]["analysis_status"] = "invalid"
    elif any(flag["severity"] == "warning" for flag in flags):
        context["analysis"]["analysis_status"] = "warning"
    else:
        context["analysis"]["analysis_status"] = "valid"
    return context, products
