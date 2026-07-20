"""Non-official, descriptive thermal flags."""

from __future__ import annotations


def thermal_flag(mean_anomaly_c: float, threshold_c: float = 1.0) -> str:
    """Return an experimental threshold flag without classifying El Niño Costero."""
    return "above experimental threshold" if mean_anomaly_c > threshold_c else "below experimental threshold"
