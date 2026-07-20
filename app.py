"""Operational cached-data dashboard for Humboldt Ocean Watch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr

from src.charting import centroid_temporal_chart, profile_chart, temporal_chart
from src.daily_climatology import select_climatology
from src.daily_diagnosis import available_dates, diagnose_date
from src.data_loader import load_active_sst_dataset
from src.export_utils import dumps_json_safe
from src.plotting import plot_centroid_trajectory, plot_spatial_field
from src.temporal_metrics import build_metrics_table
from src.utils import configure_logging, load_config, resolve_project_path

st.set_page_config(page_title="Humboldt Ocean Watch", page_icon=":material/waves:", layout="wide")
config = load_config()
configure_logging(config["logging"]["level"])


@st.cache_data(show_spinner="Loading cached SST…", max_entries=2)
def load_operational_data() -> tuple[xr.Dataset, str, xr.Dataset | None, str | None, str | None]:
    data = config["data"]
    region = config["region"]
    dataset, mode = load_active_sst_dataset(
        resolve_project_path(data["live_path"]),
        resolve_project_path(data["demo_path"]),
        aliases=data["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": data["demo_days"],
            "seed": data["demo_seed"],
        },
    )
    method_config = config["climatology"]
    selection = select_climatology(
        resolve_project_path(method_config["daily_path"]),
        resolve_project_path(method_config["monthly_path"]),
        primary_method=method_config["primary_method"],
        fallback_method=method_config["fallback_method"],
        allow_monthly_fallback=method_config["allow_monthly_fallback"],
        data_mode=mode,
    )
    if selection.dataset is not None:
        selection.dataset.attrs["climatology_fallback_used"] = selection.fallback_used
    return dataset, mode, selection.dataset, selection.method, selection.warning


dataset, data_mode, climatology, climatology_method, climatology_warning = load_operational_data()
dates = available_dates(dataset)

with st.sidebar:
    st.header("Diagnosis controls")
    analysis_date = st.selectbox("Analysis date", dates, index=len(dates) - 1)
    period_choice = st.selectbox("Time-series period", [30, 90, 180, "All"], index=1)
    anomaly_threshold = st.slider("Anomaly threshold (°C)", 0.5, 5.0, 2.0, 0.5)
    persistence_window = st.segmented_control(
        "Persistence window", [7, 15, 30], default=7, format_func=lambda value: f"{value} days"
    )
    st.caption(f"Active mode: {data_mode}")
    st.caption(f"Climatology: {climatology_method or 'Unavailable'}")

fields, diagnosis = diagnose_date(
    dataset,
    analysis_date,
    climatology=climatology,
    data_mode=data_mode,
    threshold=anomaly_threshold,
    persistence_window=int(persistence_window or 7),
)
metrics_table = build_metrics_table(dataset, climatology)
selected_time = pd.Timestamp(analysis_date)
if period_choice != "All":
    series = metrics_table[
        (metrics_table.date >= selected_time - pd.Timedelta(days=int(period_choice) - 1))
        & (metrics_table.date <= selected_time)
    ]
else:
    series = metrics_table[metrics_table.date <= selected_time]
metrics = diagnosis["metrics"]

st.title("Humboldt Ocean Watch")
st.caption("Experimental daily thermal monitoring for the Niño 1+2 region")
if diagnosis["warning"]:
    st.warning(diagnosis["warning"], icon=":material/warning:")
if climatology_warning:
    st.warning(climatology_warning, icon=":material/info:")
st.caption(
    "Daily smoothed climatology 1991–2020"
    if climatology_method == "daily_smoothed"
    else f"Climatology status: {climatology_method or 'unavailable'}"
)

tabs = st.tabs(
    ["Overview", "Maps", "Time series", "Spatial behaviour", "Data and methods", "Export"]
)


def display_value(value: float | None, unit: str, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:+.2f} {unit}" if signed else f"{value:.2f} {unit}"


with tabs[0]:
    st.subheader(f"Regional overview · {analysis_date}")
    with st.container(horizontal=True):
        st.metric("Selected date", analysis_date, border=True)
        st.metric("Mean SST", display_value(metrics["mean_sst_c"], "°C"), border=True)
        st.metric("Mean anomaly", display_value(metrics["mean_sst_anomaly_c"], "°C", True), border=True)
        st.metric("Maximum anomaly", display_value(metrics["maximum_anomaly_c"], "°C", True), border=True)
    with st.container(horizontal=True):
        st.metric(
            f"Area ≥ +{anomaly_threshold:g} °C",
            display_value(metrics["area_anomaly_ge_threshold_percent"], "%"),
            border=True,
        )
        st.metric("Valid-data coverage", display_value(metrics["valid_data_coverage_percent"], "%"), border=True)
        st.metric("Active data mode", data_mode, border=True)

with tabs[1]:
    map_specs = [
        ("Current SST", "sst_current", "SST (°C)", "turbo", 18.0, 32.0),
        ("SST anomaly", "sst_anomaly", "Anomaly (°C)", "RdBu_r", -5.0, 5.0),
        ("Standardized anomaly", "sst_z_score", "Z-score", "RdBu_r", -3.0, 3.0),
        ("Daily SST change", "sst_daily_change", "Change (°C)", "RdBu_r", -2.0, 2.0),
        ("Anomaly persistence", "anomaly_persistence", "Persistence (fraction)", "YlOrRd", 0.0, 1.0),
    ]
    for row_start in range(0, len(map_specs), 2):
        columns = st.columns(2)
        for column, spec in zip(columns, map_specs[row_start : row_start + 2], strict=False):
            title, variable, label, cmap, vmin, vmax = spec
            with column.container(border=True):
                if variable not in fields:
                    st.info(f"{title} requires the real 1991–2020 climatology.")
                else:
                    map_field = fields[variable] / 100.0 if variable == "anomaly_persistence" else fields[variable]
                    fig = plot_spatial_field(
                        map_field,
                        title=f"{title} — {analysis_date}",
                        colorbar_label=label,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

with tabs[2]:
    st.subheader("Regional time series")
    sst_chart = temporal_chart(
        series, x="date", columns=["mean_sst_c", "seven_day_mean_sst_c"],
        y_title="SST (°C)", title="Mean SST",
    )
    if sst_chart is not None:
        st.altair_chart(sst_chart, width="stretch")
    if climatology is None:
        st.info("Anomaly-dependent time series are disabled until the real climatology is built.")
    else:
        temporal_specs = [
            (["mean_sst_anomaly_c", "seven_day_mean_anomaly_c"], "Anomaly (°C)", "Mean anomaly"),
            (["area_anomaly_ge_1c_percent", "area_anomaly_ge_2c_percent", "area_anomaly_ge_3c_percent"], "Area (%)", "Warm-anomaly area"),
            (["maximum_anomaly_c", "p90_anomaly_c"], "Anomaly (°C)", "Upper anomaly distribution"),
        ]
        for columns, y_title, title in temporal_specs:
            chart = temporal_chart(series, x="date", columns=columns, y_title=y_title, title=title)
            if chart is None:
                st.info(f"No finite values are available for {title.lower()}.")
            else:
                st.altair_chart(chart, width="stretch")
        centroid_columns = st.columns(2)
        for column, coordinate in zip(centroid_columns, ("latitude", "longitude"), strict=True):
            with column.container(border=True):
                chart = centroid_temporal_chart(
                    series, coordinate=coordinate, selected_date=analysis_date
                )
                if chart is None:
                    st.info(f"No finite centroid {coordinate} values are available.")
                else:
                    st.altair_chart(chart, width="stretch")

with tabs[3]:
    if climatology is None:
        st.info("Spatial anomaly behaviour is disabled until the real climatology is built.")
    else:
        left, right = st.columns(2)
        with left.container(border=True):
            fig = plot_spatial_field(
                fields.anomaly_persistence / 100.0,
                title=f"Warm-anomaly persistence — {analysis_date}",
                colorbar_label="Persistence (fraction)",
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right.container(border=True):
            st.subheader("Warm-anomaly centroid trajectory")
            try:
                fig = plot_centroid_trajectory(series, analysis_date)
            except ValueError as exc:
                st.info(str(exc))
            else:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        profile_left, profile_right = st.columns(2)
        with profile_left.container(border=True):
            st.subheader("Latitude anomaly profile")
            chart = profile_chart(
                fields.latitude_anomaly_profile,
                coordinate="latitude",
                title="Zonal-mean SST anomaly",
            )
            if chart is None:
                st.info("No finite latitude-profile values are available.")
            else:
                st.altair_chart(chart, width="stretch")
        with profile_right.container(border=True):
            st.subheader("Longitude anomaly profile")
            chart = profile_chart(
                fields.longitude_anomaly_profile,
                coordinate="longitude",
                title="Meridional-mean SST anomaly",
            )
            if chart is None:
                st.info("No finite longitude-profile values are available.")
            else:
                st.altair_chart(chart, width="stretch")

with tabs[4]:
    st.subheader("Data and methods")
    st.write({
        "active_mode": data_mode,
        "source": dataset.attrs.get("source_path"),
        "available_period": f"{dates[0]} to {dates[-1]}",
        "study_region": "90–80°W, 10°S–0°",
        "temperature_units": dataset.sst.attrs.get("units"),
        "climatology_method": climatology_method or "Unavailable",
        "climatology_reference_period": (
            f"{config['climatology']['reference_start']}–{config['climatology']['reference_end']}"
        ),
        "daily_sampling_half_window_days": config["climatology"]["sampling_half_window_days"],
        "daily_smoothing_window_days": config["climatology"]["smoothing_window_days"],
        "fallback_method": config["climatology"]["fallback_method"],
        "leap_day_method": (
            climatology.attrs.get("leap_day_method", "Not applicable to monthly fallback")
            if climatology is not None else "Unavailable"
        ),
        "active_climatology_file": diagnosis.get("climatology_file"),
        "climatology_fallback_used": diagnosis.get("climatology_fallback_used", False),
        "percentile_calculation_state": (
            climatology.attrs.get("percentile_calculation_state", "not applicable")
            if climatology is not None else "not_calculated"
        ),
    })
    st.warning("Experimental product. Official ENSO and El Niño Costero classification is not provided.")
    st.caption("Dates are read from the NetCDF time coordinate, never from global time-coverage attributes.")

with tabs[5]:
    st.subheader("Export selected diagnosis")
    st.download_button("Download temporal metrics CSV", series.to_csv(index=False), "nino12_daily_metrics.csv", "text/csv")
    st.download_button("Download diagnosis NetCDF", bytes(fields.to_netcdf()), f"nino12_diagnosis_{analysis_date}.nc", "application/x-netcdf")
    diagnosis_json = dumps_json_safe(diagnosis)
    st.download_button(
        label="Download diagnosis JSON",
        data=diagnosis_json,
        file_name=f"nino12_diagnosis_{analysis_date}.json",
        mime="application/json",
    )
