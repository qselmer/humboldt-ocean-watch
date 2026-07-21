"""Deterministic, bilingual interpretation text for cached diagnostics.

The functions in this module describe existing measurements.  They do not
classify official ENSO conditions and they never invoke a language model.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.i18n import normalize_language


_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)
_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def localized_date(value: str | date | datetime | pd.Timestamp, language: str = "es") -> str:
    """Format a date without relying on the host operating-system locale."""
    timestamp = pd.Timestamp(value)
    months = _MONTHS_ES if normalize_language(language) == "es" else _MONTHS_EN
    if normalize_language(language) == "es":
        return f"{timestamp.day} de {months[timestamp.month - 1]} de {timestamp.year}"
    return f"{timestamp.day} {months[timestamp.month - 1]} {timestamp.year}"


def interpret_representativeness(
    analysis_date: str | date | datetime | pd.Timestamp,
    metrics: Mapping[str, Any] | pd.Series,
    language: str = "es",
) -> str:
    """Describe spatial representativeness using transparent deterministic rules."""
    language = normalize_language(language)
    day = localized_date(analysis_date, language)
    classification = str(_get(metrics, "class", _get(metrics, "classification", "")))
    mean = _finite(_get(metrics, "weighted_mean_anomaly"))
    sd = _finite(_get(metrics, "spatial_standard_deviation"))
    positive = _finite(_get(metrics, "positive_fraction"))
    negative = _finite(_get(metrics, "negative_fraction"))
    coverage = _finite(_get(metrics, "valid_coverage", _get(metrics, "weighted_valid_coverage")))
    dominant = _finite(_get(metrics, "dominant_patch_fraction"))

    if coverage is None or classification == "insufficient_coverage":
        return (
            f"El {day}, la cobertura espacial válida fue insuficiente para una interpretación representativa."
            if language == "es"
            else f"On {day}, valid spatial coverage was insufficient for a representative interpretation."
        )

    if mean is None:
        signal = "no pudo resumirse con una anomalía media válida" if language == "es" else "could not be summarized with a valid mean anomaly"
    elif mean > 0.1:
        signal = "la anomalía fue positiva" if language == "es" else "anomalies were positive"
    elif mean < -0.1:
        signal = "la anomalía fue negativa" if language == "es" else "anomalies were negative"
    else:
        signal = "la anomalía media estuvo cerca de cero" if language == "es" else "the mean anomaly was near zero"

    if positive is not None and positive >= 0.75:
        extent = " en casi toda el área válida" if language == "es" else " across nearly all valid area"
    elif negative is not None and negative >= 0.75:
        extent = " en casi toda el área válida" if language == "es" else " across nearly all valid area"
    else:
        extent = " con una distribución espacial mixta" if language == "es" else " with a mixed spatial distribution"

    heterogeneous = classification in {"strong_heterogeneous", "compensated_mixed", "patch_distributed"} or (sd is not None and sd >= 0.75)
    if heterogeneous:
        structure = (
            "La señal regional presentó una variación espacial importante."
            if language == "es"
            else "The regional signal showed substantial spatial variation."
        )
    else:
        structure = (
            "La configuración espacial fue relativamente uniforme."
            if language == "es"
            else "The spatial pattern was relatively uniform."
        )
    if dominant is not None and dominant < 0.60 and classification == "patch_distributed":
        structure += (
            " No hubo un único parche dominante."
            if language == "es"
            else " No single patch was dominant."
        )
    prefix = f"El {day}, " if language == "es" else f"On {day}, "
    return f"{prefix}{signal}{extent}. {structure}"


def interpret_thermal_event(
    analysis_date: str | date | datetime | pd.Timestamp,
    overview: Any,
    language: str = "es",
    *,
    split_count: int = 0,
    merge_count: int = 0,
) -> str:
    """Describe the cached event state without adding scientific categories."""
    language = normalize_language(language)
    day = localized_date(analysis_date, language)
    active = bool(_get(overview, "active_event", False))
    event_day = _get(overview, "event_day")
    intensity = _finite(_get(overview, "current_intensity"))
    patches = int(_get(overview, "patch_count", 0) or 0)
    tracks = int(_get(overview, "active_track_count", 0) or 0)
    families = int(_get(overview, "active_family_count", 0) or 0)
    largest = _finite(_get(overview, "largest_patch_area_km2"))

    if language == "es":
        event_text = (
            f"Hay un evento univariado activo en su día {event_day}"
            if active and event_day is not None
            else ("Hay un evento univariado activo" if active else "No hay un evento univariado activo")
        )
        if intensity is not None and active:
            event_text += f", con una intensidad actual de {intensity:.2f}"
        spatial = f"Se identificaron {patches} parches, {tracks} tracks activos y {families} familias activas"
        if largest is not None and patches:
            spatial += f"; el parche mayor cubrió {largest:,.0f} km²"
        branch = ""
        if split_count or merge_count:
            branch = f" Se registraron {split_count} divisiones y {merge_count} fusiones en la fecha."
        return f"El {day}, {event_text}. {spatial}.{branch}"

    event_text = (
        f"a univariate event is active on event day {event_day}"
        if active and event_day is not None
        else ("a univariate event is active" if active else "no univariate event is active")
    )
    if intensity is not None and active:
        event_text += f", with current intensity {intensity:.2f}"
    spatial = f"There were {patches} patches, {tracks} active tracks, and {families} active families"
    if largest is not None and patches:
        spatial += f"; the largest patch covered {largest:,.0f} km²"
    branch = ""
    if split_count or merge_count:
        branch = f" The date included {split_count} splits and {merge_count} merges."
    return f"On {day}, {event_text}. {spatial}.{branch}"
