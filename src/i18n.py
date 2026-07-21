"""Small, deterministic bilingual translation layer for the dashboard UI."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_LANGUAGE = "es"
SUPPORTED_LANGUAGES = ("es", "en")


TRANSLATIONS: dict[str, dict[str, str]] = {
    "language": {"es": "Idioma", "en": "Language"},
    "view_mode": {"es": "Modo de visualización", "en": "View mode"},
    "basic": {"es": "Básico", "en": "Basic"},
    "advanced": {"es": "Avanzado", "en": "Advanced"},
    "help_title": {"es": "Ayuda y guía", "en": "Help and guide"},
    "diagnosis_controls": {"es": "Controles del diagnóstico", "en": "Diagnosis controls"},
    "analysis_date": {"es": "Fecha de análisis", "en": "Analysis date"},
    "time_series_period": {"es": "Periodo de la serie temporal", "en": "Time-series period"},
    "anomaly_threshold": {"es": "Umbral de anomalía (°C)", "en": "Anomaly threshold (°C)"},
    "persistence_window": {"es": "Ventana de persistencia", "en": "Persistence window"},
    "active_mode": {"es": "Modo de datos activo", "en": "Active data mode"},
    "climatology": {"es": "Climatología", "en": "Climatology"},
    "getting_started": {"es": "Cómo empezar", "en": "Getting started"},
    "getting_started_1": {"es": "Seleccione una fecha.", "en": "Select a date."},
    "getting_started_2": {"es": "Lea la interpretación regional.", "en": "Read the regional interpretation."},
    "getting_started_3": {
        "es": "Explore parches, tracks y familias de eventos solo cuando necesite más detalle.",
        "en": "Explore patches, tracks, and event families only when more detail is needed.",
    },
    "regional_interpretation": {"es": "Interpretación regional", "en": "Regional interpretation"},
    "representativeness_class": {"es": "Clase de representatividad", "en": "Representativeness class"},
    "valid_coverage": {"es": "Cobertura válida", "en": "Valid coverage"},
    "mean_anomaly": {"es": "Anomalía media", "en": "Mean anomaly"},
    "warm_area_fraction": {"es": "Fracción de área cálida", "en": "Warm-area fraction"},
    "quality_representativeness": {"es": "Calidad y representatividad", "en": "Quality and representativeness"},
    "current_daily_class": {"es": "Clase diaria actual", "en": "Current daily class"},
    "evidence_score": {"es": "Puntaje de evidencia", "en": "Evidence score"},
    "triggered_rule": {"es": "Regla activada", "en": "Triggered rule"},
    "main_input_metrics": {"es": "Métricas principales de entrada", "en": "Main input metrics"},
    "advanced_diagnostics": {"es": "Diagnósticos detallados", "en": "Detailed diagnostics"},
    "thermal_events": {"es": "Eventos térmicos", "en": "Thermal events"},
    "event_overview": {"es": "Resumen de eventos", "en": "Event overview"},
    "daily_patches": {"es": "Parches diarios", "en": "Daily patches"},
    "tracks_trajectories": {"es": "Tracks y trayectorias", "en": "Tracks and trajectories"},
    "families_lineage": {"es": "Familias y linaje", "en": "Event families and lineage"},
    "active_univariate_event": {"es": "Evento univariado activo", "en": "Active univariate event"},
    "active": {"es": "Activo", "en": "Active"},
    "not_active": {"es": "No activo", "en": "Not active"},
    "current_event": {"es": "Evento actual", "en": "Current event"},
    "event_day": {"es": "Día del evento", "en": "Event day"},
    "event_duration": {"es": "Duración del evento", "en": "Event duration"},
    "current_intensity": {"es": "Intensidad actual", "en": "Current intensity"},
    "cumulative_intensity": {"es": "Intensidad acumulada", "en": "Cumulative intensity"},
    "daily_patch_count": {"es": "Número de parches diarios", "en": "Daily patch count"},
    "total_patch_area": {"es": "Área total de parches", "en": "Total patch area"},
    "largest_patch": {"es": "Parche más grande", "en": "Largest patch"},
    "active_tracks": {"es": "Tracks activos", "en": "Active tracks"},
    "active_families": {"es": "Familias activas", "en": "Active families"},
    "representativeness": {"es": "Representatividad", "en": "Representativeness"},
    "event_analysis_date": {"es": "Fecha del análisis de eventos", "en": "Event analysis date"},
    "source_variable": {"es": "Variable de origen", "en": "Source variable"},
    "threshold_type": {"es": "Tipo de umbral", "en": "Threshold type"},
    "direction": {"es": "Dirección", "en": "Direction"},
    "event_status": {"es": "Estado del evento", "en": "Event status"},
    "minimum_track_duration": {"es": "Duración mínima del track (días)", "en": "Minimum track duration (days)"},
    "minimum_track_area": {"es": "Área máxima mínima del track (km²)", "en": "Minimum track maximum area (km²)"},
    "family_selector": {"es": "Selector de familia de eventos", "en": "Event-family selector"},
    "track_selector": {"es": "Selector de track", "en": "Track selector"},
    "reset_event_filters": {"es": "Restablecer filtros", "en": "Reset event filters"},
    "all": {"es": "Todos", "en": "All"},
    "more_filters": {"es": "Más filtros", "en": "More filters"},
    "how_use_filters": {"es": "¿Cómo usar estos filtros?", "en": "How to use these filters"},
    "selected_event": {"es": "Evento univariado seleccionado", "en": "Selected univariate event"},
    "event_catalogue": {"es": "Catálogo de eventos", "en": "Event catalogue"},
    "technical_details": {"es": "Detalles técnicos", "en": "Technical details"},
    "cached_downloads": {"es": "Descargas de productos", "en": "Product downloads"},
    "essential_downloads": {"es": "Descargas esenciales", "en": "Essential downloads"},
    "data_methods": {"es": "Datos y métodos", "en": "Data and methods"},
    "glossary": {"es": "Glosario", "en": "Glossary"},
    "limitations": {"es": "Limitaciones", "en": "Limitations"},
    "internal_id": {"es": "ID interno", "en": "Internal ID"},
    "unavailable": {"es": "No disponible", "en": "Unavailable"},
    "days": {"es": "días", "en": "days"},
    "rows_after_filter": {"es": "{count} filas después del filtrado", "en": "{count} rows after filtering"},
}


VALUE_LABELS: dict[str, dict[str, str]] = {
    "insufficient_coverage": {"es": "Cobertura insuficiente", "en": "Insufficient coverage"},
    "strong_coherent": {"es": "Calentamiento fuerte y espacialmente coherente", "en": "Strong, spatially coherent warming"},
    "strong_heterogeneous": {"es": "Calentamiento fuerte con alta variabilidad espacial", "en": "Strong warming with high spatial variability"},
    "compensated_mixed": {"es": "Condiciones mixtas con compensación espacial", "en": "Spatially compensated mixed conditions"},
    "patch_distributed": {"es": "Señal distribuida entre varios parches", "en": "Signal distributed among multiple patches"},
    "weak_homogeneous": {"es": "Señal débil y espacialmente uniforme", "en": "Weak, spatially uniform signal"},
    "unclassified": {"es": "Sin clasificación", "en": "Unclassified"},
    "daily_climatological": {"es": "Umbral climatológico diario", "en": "Daily climatological threshold"},
    "fixed": {"es": "Umbral fijo", "en": "Fixed threshold"},
    "global_percentile": {"es": "Percentil global", "en": "Global percentile threshold"},
    "custom": {"es": "Umbral temporal personalizado", "en": "Custom time-varying threshold"},
    "above": {"es": "Por encima del umbral", "en": "Above threshold"},
    "below": {"es": "Por debajo del umbral", "en": "Below threshold"},
    "appearance": {"es": "Aparición", "en": "Appearance"},
    "continuation": {"es": "Continuación", "en": "Continuation"},
    "split_parent": {"es": "Parche que se divide", "en": "Splitting patch"},
    "split_child": {"es": "Rama originada por división", "en": "Split-derived branch"},
    "merge_parent": {"es": "Parche que entra en una fusión", "en": "Merging parent patch"},
    "merge_child": {"es": "Parche resultante de una fusión", "en": "Merge-result patch"},
    "complex_branch": {"es": "Ramificación compleja", "en": "Complex branch"},
    "termination": {"es": "Terminación", "en": "Termination"},
    "isolated_single_day": {"es": "Parche aislado de un día", "en": "Isolated one-day patch"},
    "valid": {"es": "Válido", "en": "Valid"},
    "warning": {"es": "Con advertencia", "en": "Warning"},
    "not_calculated": {"es": "No calculado", "en": "Not calculated"},
    "invalid": {"es": "Inválido", "en": "Invalid"},
    "anomaly": {"es": "Anomalía de TSM", "en": "SST anomaly"},
    "zscore": {"es": "Anomalía estandarizada", "en": "Standardized anomaly"},
    "sst": {"es": "Temperatura superficial del mar", "en": "Sea-surface temperature"},
    "regional_mean_sst": {"es": "TSM media regional", "en": "Regional mean SST"},
    "daily_smoothed_climatology": {"es": "Climatología diaria suavizada", "en": "Smoothed daily climatology"},
    "spatial_heterogeneity": {"es": "Heterogeneidad espacial", "en": "Spatial heterogeneity"},
    "patch": {"es": "Parche", "en": "Patch"},
    "split": {"es": "División", "en": "Split"},
    "merge": {"es": "Fusión", "en": "Merge"},
    "lineage": {"es": "Linaje", "en": "Lineage"},
    "iou": {"es": "Intersección sobre unión (IoU)", "en": "Intersection over union (IoU)"},
    "link_score": {"es": "Puntaje de enlace", "en": "Link score"},
    "centroid": {"es": "Centroide", "en": "Centroid"},
    "trajectory": {"es": "Trayectoria", "en": "Trajectory"},
    "cumulative_severity": {"es": "Severidad acumulada", "en": "Cumulative severity"},
}


def normalize_language(language: str | None) -> str:
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def tr(key: str, language: str = DEFAULT_LANGUAGE, default: str | None = None, **values: Any) -> str:
    """Translate a UI key, falling back to English and then a supplied default/key."""
    selected = normalize_language(language)
    options = TRANSLATIONS.get(key, {})
    text = options.get(selected) or options.get("en") or default or key
    return text.format(**values) if values else text


def human_label(value: Any, language: str = DEFAULT_LANGUAGE) -> str:
    """Return a primary human-readable label for a technical value."""
    if value is None:
        return tr("unavailable", language)
    key = str(value)
    translations = VALUE_LABELS.get(key)
    if translations:
        selected = normalize_language(language)
        return translations.get(selected) or translations.get("en") or key
    return key.replace("_", " ").strip().capitalize()


_IDENTIFIER_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)0*(?P<number>\d+)$")


def format_identifier(identifier: Any, language: str = DEFAULT_LANGUAGE) -> str:
    """Format stable internal identifiers for readers without losing reproducibility."""
    if identifier is None:
        return tr("unavailable", language)
    internal = str(identifier)
    match = _IDENTIFIER_PATTERN.match(internal)
    if not match:
        return internal
    prefix = match.group("prefix").upper()
    number = int(match.group("number"))
    entity = {
        "FAM": {"es": "Familia", "en": "Family"},
        "TRK": {"es": "Track", "en": "Track"},
        "E": {"es": "Evento", "en": "Event"},
        "EVT": {"es": "Evento", "en": "Event"},
        "P": {"es": "Parche", "en": "Patch"},
    }.get(prefix)
    if entity is None:
        return internal
    return f"{entity[normalize_language(language)]} {number}"


def internal_id_caption(identifier: Any, language: str = DEFAULT_LANGUAGE) -> str:
    return f"{tr('internal_id', language)}: {identifier}" if identifier else ""


def language_name(language: str, display_language: str = DEFAULT_LANGUAGE) -> str:
    names = {
        "es": {"es": "Español", "en": "Spanish"},
        "en": {"es": "Inglés", "en": "English"},
    }
    selected = normalize_language(display_language)
    return names.get(language, {}).get(selected, language)
