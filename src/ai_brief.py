"""Deterministic brief text; optional AI integrations can be added later."""

from __future__ import annotations


def build_brief(date: str, mean_sst_c: float, mean_anomaly_c: float | None = None) -> str:
    """Build a factual experimental thermal summary."""
    anomaly_text = ""
    if mean_anomaly_c is not None:
        anomaly_text = f" The period-relative mean anomaly is {mean_anomaly_c:+.2f} °C."
    return (
        f"Experimental thermal monitoring for {date}: regional mean SST is "
        f"{mean_sst_c:.2f} °C.{anomaly_text} This product does not classify the "
        "official magnitude of El Niño Costero."
    )
