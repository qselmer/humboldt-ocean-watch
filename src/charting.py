"""Reusable, data-driven Altair chart helpers."""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import numpy as np
import pandas as pd
import xarray as xr

PROFILE_VALUE_COLUMN = "sst_anomaly_c"


def finite_numeric_domain(
    values: pd.Series | np.ndarray, *, padding_fraction: float = 0.0
) -> tuple[float, float] | None:
    """Return a finite domain, expanding constant series and adding padding."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    if span == 0:
        pad = max(abs(lower) * max(padding_fraction, 0.05), 0.1)
    else:
        pad = span * padding_fraction
    return lower - pad, upper + pad


def profile_dataframe(profile: xr.DataArray, coordinate: str) -> pd.DataFrame:
    """Build a stable, finite profile table independent of DataArray name."""
    if coordinate not in profile.coords:
        raise ValueError(f"Profile is missing the {coordinate!r} coordinate")
    frame = pd.DataFrame(
        {
            coordinate: np.asarray(profile[coordinate].values).reshape(-1),
            PROFILE_VALUE_COLUMN: np.asarray(profile.values).reshape(-1),
        }
    )
    coordinate_values = pd.to_numeric(frame[coordinate], errors="coerce")
    anomaly_values = pd.to_numeric(frame[PROFILE_VALUE_COLUMN], errors="coerce")
    finite = np.isfinite(coordinate_values) & np.isfinite(anomaly_values)
    return frame.loc[finite, [coordinate, PROFILE_VALUE_COLUMN]].reset_index(drop=True)


def profile_chart(
    profile: xr.DataArray, *, coordinate: str, title: str
) -> alt.Chart | None:
    """Create a profile chart with finite, data-driven axis domains."""
    frame = profile_dataframe(profile, coordinate)
    if frame.empty:
        return None
    x_domain = finite_numeric_domain(frame[coordinate])
    y_domain = finite_numeric_domain(frame[PROFILE_VALUE_COLUMN], padding_fraction=0.05)
    assert x_domain is not None and y_domain is not None
    return (
        alt.Chart(frame, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{coordinate}:Q", title=coordinate.capitalize(), scale=alt.Scale(domain=list(x_domain))),
            y=alt.Y(
                f"{PROFILE_VALUE_COLUMN}:Q",
                title="SST anomaly (°C)",
                scale=alt.Scale(domain=list(y_domain), zero=False),
            ),
            tooltip=[
                alt.Tooltip(f"{coordinate}:Q", format=".2f"),
                alt.Tooltip(f"{PROFILE_VALUE_COLUMN}:Q", title="Anomaly (°C)", format="+.3f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )


def temporal_chart(
    data: pd.DataFrame,
    *,
    x: str,
    columns: Sequence[str],
    y_title: str,
    title: str,
) -> alt.Chart | None:
    """Create a multi-series temporal chart using finite data-driven domains."""
    existing = [column for column in columns if column in data.columns]
    if x not in data.columns or not existing:
        return None
    frame = data[[x, *existing]].copy()
    frame[x] = pd.to_datetime(frame[x], errors="coerce")
    long = frame.melt(
        id_vars=x,
        value_vars=existing,
        var_name="_metric_series",
        value_name="_metric_value",
    )
    long["_metric_value"] = pd.to_numeric(long["_metric_value"], errors="coerce")
    long = long[long[x].notna() & np.isfinite(long["_metric_value"])].copy()
    if long.empty:
        return None
    y_domain = finite_numeric_domain(long["_metric_value"], padding_fraction=0.05)
    assert y_domain is not None
    x_min, x_max = long[x].min(), long[x].max()
    if x_min == x_max:
        x_min -= pd.Timedelta(hours=12)
        x_max += pd.Timedelta(hours=12)
    return (
        alt.Chart(long, title=title)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Date", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("_metric_value:Q", title=y_title, scale=alt.Scale(domain=list(y_domain), zero=False)),
            color=alt.Color("_metric_series:N", title="Series"),
            tooltip=[alt.Tooltip(f"{x}:T", title="Date"), alt.Tooltip("_metric_series:N", title="Series"), alt.Tooltip("_metric_value:Q", title="Value", format=".3f")],
        )
        .properties(height=320)
        .interactive()
    )


def centroid_temporal_chart(
    data: pd.DataFrame,
    *,
    coordinate: str,
    selected_date: str,
) -> alt.LayerChart | None:
    """Create one centroid-coordinate series with gaps and a selected-date rule."""
    column = f"warm_centroid_{coordinate}"
    if coordinate not in {"latitude", "longitude"}:
        raise ValueError("Centroid coordinate must be 'latitude' or 'longitude'")
    if "date" not in data.columns or column not in data.columns:
        return None
    frame = data[["date", column]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame[frame.date.notna()].copy()
    y_domain = finite_numeric_domain(frame[column], padding_fraction=0.05)
    if frame.empty or y_domain is None:
        return None
    date_min, date_max = frame.date.min(), frame.date.max()
    if date_min == date_max:
        date_min -= pd.Timedelta(hours=12)
        date_max += pd.Timedelta(hours=12)
    label = coordinate.capitalize()
    line = (
        alt.Chart(frame)
        .mark_line(point=True, invalid="break-paths-show-domains")
        .encode(
            x=alt.X("date:T", title="Date", scale=alt.Scale(domain=[date_min, date_max])),
            y=alt.Y(
                f"{column}:Q",
                title=f"Centroid {coordinate} (degrees)",
                scale=alt.Scale(domain=list(y_domain), zero=False),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{column}:Q", title=f"{label} (°)", format=".3f"),
            ],
        )
    )
    rule_data = pd.DataFrame({"selected_date": [pd.Timestamp(selected_date)]})
    rule = alt.Chart(rule_data).mark_rule(color="#d62728", strokeDash=[5, 4]).encode(
        x=alt.X("selected_date:T")
    )
    return (line + rule).properties(title=f"Centroid {coordinate} through time", height=280).interactive()
