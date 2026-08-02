"""Operational cached-data dashboard for Humboldt Ocean Watch."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr

from src.charting import (
    centroid_temporal_chart,
    profile_chart,
    representativeness_class_chart,
    representativeness_temporal_chart,
    temporal_chart,
)
from src.daily_climatology import select_climatology
from src.daily_diagnosis import available_dates, diagnose_date
from src.data_loader import load_active_sst_dataset
from src.event_dashboard import render_thermal_events_tab
from src.event_data_loader import load_event_products
from src.export_utils import dumps_json_safe
from src.geographic_foundation import prepare_geographic_foundation
from src.geographic_overlays import plot_geographic_foundation
from src.geography import load_geography_registry
from src.help_content import (
    glossary,
    metric_tooltip,
    representativeness_definition,
    representativeness_notes,
    scientific_disclaimer,
)
from src.i18n import human_label, tr
from src.interpretation_text import interpret_representativeness
from src.plotting import plot_centroid_trajectory, plot_spatial_field
from src.representativeness import (
    RepresentativenessThresholds,
    calculate_daily_metrics,
    classify_representativeness,
)
from src.temporal_metrics import build_metrics_table
from src.sidebar_manual import is_advanced, render_first_use, render_help_sidebar
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


@st.cache_data(show_spinner=False, max_entries=2)
def load_analytics_products() -> tuple[pd.DataFrame, dict]:
    representativeness_path = resolve_project_path(config["representativeness"]["output"])
    qc_path = resolve_project_path(config["quality_control"]["output"])
    representativeness = (
        pd.read_parquet(representativeness_path)
        if representativeness_path.exists()
        else pd.DataFrame()
    )
    if not representativeness.empty:
        representativeness["date"] = pd.to_datetime(
            representativeness["date"], errors="coerce"
        )
    qc_report = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    return representativeness, qc_report


@st.cache_data(show_spinner=False, max_entries=10)
def load_download_file(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_resource(show_spinner="Preparing local geographic foundation…", max_entries=1)
def load_geographic_foundation_resource():
    """Build local coastline geometry once without permitting downloads."""
    return prepare_geographic_foundation(config)


dataset, data_mode, climatology, climatology_method, climatology_warning = load_operational_data()
representativeness_table, qc_report = load_analytics_products()
event_products = load_event_products(config, root=Path.cwd())
representativeness_thresholds = RepresentativenessThresholds.from_mapping(
    config["representativeness"]
)
dates = available_dates(dataset)

active_main_tab = st.session_state.get("main_dashboard_tab", "Overview")
language, view_mode = render_help_sidebar(config, active_main_tab)
advanced_mode = is_advanced(view_mode)

with st.sidebar:
    st.header(tr("diagnosis_controls", language), anchor=False)
    analysis_date = st.selectbox(
        tr("analysis_date", language), dates, index=len(dates) - 1,
        key="diagnosis_analysis_date",
    )
    detail_context = (
        st.container()
        if advanced_mode
        else st.expander(tr("advanced_diagnostics", language), expanded=False)
    )
    with detail_context:
        period_choice = st.selectbox(
            tr("time_series_period", language), [30, 90, 180, "All"],
            index=1, key="diagnosis_period",
        )
        anomaly_threshold = st.slider(
            tr("anomaly_threshold", language), 0.5, 5.0, 2.0, 0.5,
            key="diagnosis_anomaly_threshold",
        )
        persistence_window = st.segmented_control(
            tr("persistence_window", language), [7, 15, 30],
            default=7, key="diagnosis_persistence_window",
            format_func=lambda value: f"{value} {'días' if language == 'es' else 'days'}",
        )
    st.caption(f"{tr('active_mode', language)}: {data_mode}")
    st.caption(f"{tr('climatology', language)}: {climatology_method or 'Unavailable'}")

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
if not representativeness_table.empty:
    if period_choice != "All":
        representativeness_series = representativeness_table[
            (representativeness_table.date >= selected_time - pd.Timedelta(days=int(period_choice) - 1))
            & (representativeness_table.date <= selected_time)
        ].copy()
    else:
        representativeness_series = representativeness_table[
            representativeness_table.date <= selected_time
        ].copy()
    selected_representativeness = representativeness_table[
        representativeness_table.date == selected_time
    ]
else:
    representativeness_series = pd.DataFrame()
    selected_representativeness = pd.DataFrame()

current_representativeness_metrics: dict[str, float] = {}
current_representativeness_classification = None
if "sst_anomaly" in fields:
    current_representativeness_metrics = calculate_daily_metrics(
        fields.sst_anomaly,
        representativeness_thresholds,
        connectivity=config["spatial_features"]["connectivity"],
        minimum_patch_cells=int(config["spatial_features"]["minimum_patch_cells"]),
    )
    current_representativeness_classification = classify_representativeness(
        current_representativeness_metrics,
        representativeness_thresholds,
    )

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

render_first_use(language, view_mode)
tab_labels = [
    "Overview", "Maps", "Time series", "Spatial behaviour",
    "Quality and representativeness", "Thermal events", "Data and methods", "Export",
]
if st.session_state.get("main_dashboard_tab") not in (None, *tab_labels):
    del st.session_state["main_dashboard_tab"]
tabs = st.tabs(
    tab_labels, default="Overview", key="main_dashboard_tab", on_change="rerun"
)


def display_value(value: float | None, unit: str, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:+.2f} {unit}" if signed else f"{value:.2f} {unit}"


def display_fraction(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{100.0 * value:.1f}%"


def display_class(value: str | None) -> str:
    return human_label(value, language) if value else tr("unavailable", language)


with tabs[0]:
    st.subheader(f"Regional overview · {analysis_date}")
    if not selected_representativeness.empty:
        current_row = selected_representativeness.iloc[0]
        current_class = str(current_row["class"])
        current_coverage = float(current_row["valid_coverage"])
        current_mean_anomaly = float(current_row["weighted_mean_anomaly"])
    elif current_representativeness_classification is not None:
        current_class = current_representativeness_classification.classification
        current_coverage = current_representativeness_metrics["weighted_valid_coverage"]
        current_mean_anomaly = current_representativeness_metrics["weighted_mean_anomaly"]
    else:
        current_class = None
        current_coverage = None
        current_mean_anomaly = metrics["mean_sst_anomaly_c"]
    warm_area_fraction = current_representativeness_metrics.get("warm_area_fraction")
    with st.container(horizontal=True):
        st.metric(tr("representativeness_class", language), display_class(current_class), border=True)
        st.metric(
            tr("valid_coverage", language), display_fraction(current_coverage),
            help=metric_tooltip("valid_coverage", language), border=True,
        )
        st.metric(
            tr("mean_anomaly", language), display_value(current_mean_anomaly, "°C", True),
            help=metric_tooltip("mean_anomaly", language), border=True,
        )
        st.metric(
            f"Warm-area fraction (≥ +{representativeness_thresholds.strong_signal_c:g} °C)",
            display_fraction(warm_area_fraction),
            border=True,
        )
    interpretation_source = (
        selected_representativeness.iloc[0]
        if not selected_representativeness.empty
        else {**current_representativeness_metrics, "class": current_class}
    )
    st.markdown(f"**{tr('regional_interpretation', language)}**")
    st.info(interpret_representativeness(analysis_date, interpretation_source, language))
    st.caption(representativeness_notes(language))

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
    st.subheader(tr("quality_representativeness", language))
    st.caption(representativeness_notes(language))
    if selected_representativeness.empty:
        st.info(
            "No representativeness row is available for the selected date. Run "
            "`uv run python scripts/build_representativeness.py` to refresh the cached table."
        )
    else:
        representative_row = selected_representativeness.iloc[0]
        with st.container(horizontal=True):
            st.metric(
                tr("current_daily_class", language),
                display_class(str(representative_row["class"])),
                border=True,
            )
            st.metric(
                tr("evidence_score", language),
                display_value(float(representative_row["evidence_score"]), ""),
                help=metric_tooltip("evidence_score", language),
                border=True,
            )
            st.metric(
                tr("valid_coverage", language),
                display_fraction(float(representative_row["valid_coverage"])),
                help=metric_tooltip("valid_coverage", language),
                border=True,
            )
        st.info(interpret_representativeness(analysis_date, representative_row, language))
        st.caption(representativeness_definition(str(representative_row["class"]), language))
        with st.container(border=True):
            st.markdown(f"**{tr('triggered_rule', language)}**")
            triggered_rule = str(representative_row["triggered_rule"])
            st.write(human_label(triggered_rule, language))
            if advanced_mode:
                st.caption(f"{tr('internal_id', language)}: {triggered_rule}")

        st.markdown("**Main input metrics**")
        main_metrics = pd.DataFrame([
            {"Metric": "Mean anomaly", "Value": representative_row["weighted_mean_anomaly"], "Unit": "°C"},
            {"Metric": "Spatial standard deviation", "Value": representative_row["spatial_standard_deviation"], "Unit": "°C"},
            {"Metric": "Sign coherence", "Value": representative_row["sign_coherence"], "Unit": "fraction"},
            {"Metric": "Signal-to-heterogeneity ratio", "Value": representative_row["signal_heterogeneity_ratio"], "Unit": "ratio"},
            {"Metric": "Positive fraction", "Value": representative_row["positive_fraction"], "Unit": "fraction"},
            {"Metric": "Negative fraction", "Value": representative_row["negative_fraction"], "Unit": "fraction"},
            {"Metric": "Neutral fraction", "Value": representative_row["neutral_fraction"], "Unit": "fraction"},
            {"Metric": "Dominant patch fraction", "Value": representative_row["dominant_patch_fraction"], "Unit": "fraction"},
        ])
        st.dataframe(
            main_metrics,
            hide_index=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.3f")},
        )

        class_chart = representativeness_class_chart(
            representativeness_series,
            selected_date=analysis_date,
        )
        if class_chart is not None:
            st.altair_chart(class_chart, width="stretch")

        continuous_columns = st.columns(2)
        continuous_specs = [
            (
                continuous_columns[0],
                ["sign_coherence"],
                "Fraction",
                "Sign coherence",
                {"sign_coherence": "Sign coherence"},
            ),
            (
                continuous_columns[1],
                ["signal_heterogeneity_ratio"],
                "Ratio",
                "Signal-to-heterogeneity ratio",
                {"signal_heterogeneity_ratio": "Signal-to-heterogeneity ratio"},
            ),
        ]
        for column, columns, y_title, title, labels in continuous_specs:
            with column.container(border=True):
                chart = representativeness_temporal_chart(
                    representativeness_series,
                    columns=columns,
                    selected_date=analysis_date,
                    y_title=y_title,
                    title=title,
                    labels=labels,
                )
                if chart is None:
                    st.info(f"No finite values are available for {title.lower()}.")
                else:
                    st.altair_chart(chart, width="stretch")

        fraction_chart = representativeness_temporal_chart(
            representativeness_series,
            columns=["positive_fraction", "negative_fraction", "neutral_fraction"],
            selected_date=analysis_date,
            y_title="Weighted fraction",
            title="Positive, negative, and neutral area fractions",
            labels={
                "positive_fraction": "Positive fraction",
                "negative_fraction": "Negative fraction",
                "neutral_fraction": "Neutral fraction",
            },
        )
        if fraction_chart is not None:
            st.altair_chart(fraction_chart, width="stretch")
        dominant_chart = representativeness_temporal_chart(
            representativeness_series,
            columns=["dominant_patch_fraction"],
            selected_date=analysis_date,
            y_title="Weighted fraction",
            title="Dominant patch fraction",
            labels={"dominant_patch_fraction": "Dominant patch fraction"},
        )
        if dominant_chart is not None:
            st.altair_chart(dominant_chart, width="stretch")

    with st.container(border=True):
        st.markdown("**QC summary**")
        if qc_report:
            with st.container(horizontal=True):
                st.metric("QC status", str(qc_report.get("overall_status", "unknown")).capitalize())
                st.metric("Dates checked", str(qc_report.get("number_of_dates", "Unavailable")))
                st.metric(
                    "Weighted coverage",
                    display_fraction(qc_report.get("weighted_spatial_coverage")),
                )
                regular = qc_report.get("temporal_interval_summary", {}).get("regular")
                st.metric("Daily intervals", "Regular" if regular else "Irregular")
            qc_messages = [*qc_report.get("warnings", []), *qc_report.get("errors", [])]
            if qc_messages:
                st.warning("; ".join(map(str, qc_messages)))
            else:
                st.caption("No QC warnings or structural errors were reported.")
        else:
            st.info("The cached QC report is unavailable.")

    st.markdown("**Analytics downloads**")
    download_specs = [
        ("QC report JSON", config["quality_control"]["output"], "qc_report.json", "application/json"),
        ("Series bank", config["series_bank"]["output"], "daily_series_bank.parquet", "application/octet-stream"),
        ("Temporal features", config["temporal_features"]["output"], "temporal_features.parquet", "application/octet-stream"),
        ("Spatial features", config["spatial_features"]["output"], "spatial_features.parquet", "application/octet-stream"),
        ("Representativeness table", config["representativeness"]["output"], "representativeness.parquet", "application/octet-stream"),
    ]
    with st.container(horizontal=True):
        for label, configured_path, file_name, mime in download_specs:
            local_path = resolve_project_path(configured_path)
            if local_path.exists():
                st.download_button(
                    label=label,
                    data=load_download_file(str(local_path)),
                    file_name=file_name,
                    mime=mime,
                    key=f"analytics_download_{file_name}",
                )
            else:
                st.button(
                    label,
                    disabled=True,
                    key=f"analytics_missing_{file_name}",
                    help=f"Cached file is unavailable: {local_path}",
                )

with tabs[5]:
    render_thermal_events_tab(
        event_products,
        config,
        analysis_date=analysis_date,
        representativeness=representativeness_table,
        anomaly=fields["sst_anomaly"] if "sst_anomaly" in fields else None,
        project_root=Path.cwd(),
        language=language,
        view_mode=view_mode,
        show_internal_ids=bool(config["interface"]["show_internal_ids"]),
    )

with tabs[6]:
    st.subheader(tr("data_methods", language))
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
    st.markdown("#### Geographic foundation")
    geography_registry = load_geography_registry(config)
    st.info(
        "The geographic foundation defines the future multidomain scope. Current "
        "scientific calculations remain restricted to the existing Niño 1+2 domain "
        "until the multidomain SST increment is implemented."
    )
    st.caption(
        "60-nautical-mile corridor measured from the continental Pacific coastline "
        "of Ecuador, Peru, and Chile. Islands do not generate independent buffers."
    )
    st.write({
        "Pacific context bounds": geography_registry.display_domain.bounds.to_dict(),
        "Humboldt coastal bounds": geography_registry.analysis_domains[
            "humboldt_coastal"
        ].bounds.to_dict(),
        "South Pacific High diagnostic domain bounds": geography_registry.analysis_domains[
            "south_pacific_high"
        ].bounds.to_dict(),
        "60 nm corridor specification": geography_registry.coastal_corridors[
            "humboldt_60nm"
        ].to_dict(),
    })
    nino_region_rows = [
        {
            "Region": region.label,
            "Internal ID": region.id,
            "Longitude": f"{region.bounds.west:g} to {region.bounds.east:g}",
            "Latitude": f"{region.bounds.south:g} to {region.bounds.north:g}",
            "Geometry": region.geometry_type,
        }
        for region in geography_registry.standard_regions.values()
    ]
    st.dataframe(pd.DataFrame(nino_region_rows), hide_index=True, width="stretch")
    if tabs[6].open:
        try:
            geographic_foundation = load_geographic_foundation_resource()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            st.warning(
                "The local geographic foundation could not be prepared. No download "
                f"was attempted. Details: {exc}"
            )
        else:
            st.write({
                "Coastline source status": geographic_foundation.source_status.to_dict(),
                "Coastal corridor geometry status": geographic_foundation.corridor_status,
            })
            for warning in geographic_foundation.warnings:
                st.warning(warning)
            geographic_figure = plot_geographic_foundation(
                geographic_foundation.registry,
                local_geometries=geographic_foundation.local_geometries,
                corridor=geographic_foundation.corridor,
                source_status=geographic_foundation.source_status,
            )
            st.pyplot(geographic_figure, width="stretch")
            plt.close(geographic_figure)
    else:
        st.caption(
            "Open Data and methods to load the cached local coastline and geographic preview."
        )
    st.markdown("#### Increment 4 event definitions")
    st.write({
        "univariate_events": {
            "source_series": config["event_detection"].get("source_metric"),
            "threshold_type": config["event_detection"]["threshold_type"],
            "fixed_threshold": config["event_detection"].get("fixed_anomaly_threshold_c"),
            "minimum_duration_days": config["event_detection"]["minimum_duration_days"],
            "allowed_interruptions_days": config["event_detection"]["allowed_gap_days"],
            "maximum_calendar_gap_days": config["event_detection"]["maximum_calendar_gap_days"],
        },
        "daily_patches": {
            "source_variable": config["patch_detection"]["source_variable"],
            "direction": config["patch_detection"]["direction"],
            "threshold_type": config["patch_detection"]["threshold_type"],
            "connectivity": config["patch_detection"]["connectivity"],
            "minimum_patch_cells": config["patch_detection"]["minimum_patch_cells"],
            "minimum_patch_area_km2": config["patch_detection"]["minimum_patch_area_km2"],
            "area_method": "Spherical latitude-longitude cell quadrilaterals",
        },
        "tracking": {
            "maximum_calendar_gap_days": config["patch_tracking"]["maximum_calendar_gap_days"],
            "minimum_iou": config["patch_tracking"]["minimum_iou"],
            "minimum_predecessor_overlap": config["patch_tracking"]["minimum_predecessor_overlap"],
            "minimum_successor_overlap": config["patch_tracking"]["minimum_successor_overlap"],
            "distance_fallback": config["patch_tracking"]["allow_distance_fallback"],
            "minimum_link_score": config["patch_tracking"]["minimum_link_score"],
            "score_weights": config["patch_tracking"]["score_weights"],
            "continuation_backbone": "Deterministic maximum-score one-to-one assignment",
            "track_definition": "Maximal non-branching path through the lineage DAG",
            "event_family_definition": "Weakly connected component including splits and merges",
        },
    })
    st.markdown("#### Cached event-product status")
    st.write({
        "daily_patch_label_cube": str(resolve_project_path(config["patch_detection"]["labels_output"])),
        "track_and_family_label_cube": str(resolve_project_path(config["patch_tracking"]["track_labels_output"])),
        "tracking_summary_status": event_products.tracking_summary.status,
    })
    st.markdown("#### Event-tracking limitations")
    st.info(
        "Local daily patch IDs are not persistent. Tracks are maximal non-branching paths, while event "
        "families may contain splits and merges. Results depend on threshold and connectivity, and small "
        "threshold changes can alter lineage structure. Boundary-touching tracks may be incomplete. "
        "Tracking describes thermal features, not individual water parcels. This experimental product "
        "is not an official El Niño Costero classification."
    )
    st.warning("Experimental product. Official ENSO and El Niño Costero classification is not provided.")
    st.markdown(f"#### {tr('glossary', language)}")
    glossary_rows = [
        {
            ("Término" if language == "es" else "Term"): human_label(term, language),
            ("Definición" if language == "es" else "Definition"): definition,
        }
        for term, definition in glossary(language).items()
    ]
    st.dataframe(pd.DataFrame(glossary_rows), hide_index=True, width="stretch")
    st.warning(scientific_disclaimer(language), icon=":material/science:")
    st.caption("Dates are read from the NetCDF time coordinate, never from global time-coverage attributes.")

with tabs[7]:
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
