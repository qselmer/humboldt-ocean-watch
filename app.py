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
from src.sidebar_manual import (
    INTERFACE_LANGUAGE,
    migrate_legacy_interface_state,
    render_dashboard_guide,
    render_first_use,
)
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


@st.cache_data(show_spinner="Loading cached multidomain SST products…", max_entries=3)
def load_multidomain_products(
    status_path: str,
    snapshot_path: str,
    indices_path: str,
    preview_path: str,
    freshness: tuple[int, ...],
) -> tuple[dict, dict, pd.DataFrame, bytes]:
    """Read only pre-built local products; freshness participates in the cache key."""
    del freshness
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    indices = pd.read_parquet(indices_path)
    preview = Path(preview_path).read_bytes()
    return status, snapshot, indices, preview


@st.cache_data(show_spinner="Loading cached multidomain anomaly products…", max_entries=3)
def load_multidomain_anomaly_products(
    status_path: str,
    indices_path: str,
    preview_path: str,
    freshness: tuple[int, ...],
) -> tuple[dict, pd.DataFrame, bytes]:
    """Read only prepared anomaly products; never build, download, or write."""
    del freshness
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    indices = pd.read_parquet(indices_path)
    preview = Path(preview_path).read_bytes()
    return status, indices, preview


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

migrate_legacy_interface_state(st.session_state)
active_main_tab = st.session_state.get("main_dashboard_tab", "Overview")
language = INTERFACE_LANGUAGE

with st.sidebar:
    st.header("Analysis controls", anchor=False)
    analysis_date = st.selectbox(
        "Analysis date", dates, index=len(dates) - 1,
        key="diagnosis_analysis_date",
    )
    with st.expander("Advanced calculation settings", expanded=False):
        period_choice = st.selectbox(
            "Time-series window", [30, 90, 180, "All"],
            index=1, key="diagnosis_period",
        )
        anomaly_threshold = st.slider(
            "Warm-anomaly threshold (°C)", 0.5, 5.0, 2.0, 0.5,
            key="diagnosis_anomaly_threshold",
        )
        persistence_window = st.segmented_control(
            "Persistence window", [7, 15, 30],
            default=7, key="diagnosis_persistence_window",
            format_func=lambda value: f"{value} days",
        )
    st.markdown("**Data status**")
    st.caption(f"Active data source: {data_mode}")
    st.caption(f"Climatology method: {climatology_method or 'Unavailable'}")

render_dashboard_guide(active_main_tab)

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

render_first_use()
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


def render_cached_download(
    label: str,
    configured_path: str | Path,
    *,
    key: str,
) -> None:
    """Render a local cached-product download or a clear unavailable state."""
    local_path = resolve_project_path(configured_path)
    mime_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".nc": "application/x-netcdf",
        ".parquet": "application/octet-stream",
        ".md": "text/markdown",
    }
    if local_path.exists() and local_path.is_file():
        st.download_button(
            label,
            data=load_download_file(str(local_path)),
            file_name=local_path.name,
            mime=mime_types.get(local_path.suffix.lower(), "application/octet-stream"),
            key=key,
        )
    else:
        st.button(
            label,
            disabled=True,
            key=f"{key}_unavailable",
            help=f"Cached product is unavailable: {local_path}",
        )


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
        st.metric("Analysis date", str(analysis_date), border=True)
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
    st.caption(
        "This deterministic interpretation summarizes cached measurements; it is not a forecast, "
        "causal attribution, or official Coastal El Niño classification."
    )
    with st.expander("Evidence and supporting diagnostics", expanded=False):
        st.write({
            "Mean SST": display_value(metrics.get("mean_sst_c"), "°C"),
            "Maximum anomaly": display_value(metrics.get("maximum_anomaly_c"), "°C", True),
            "90th-percentile anomaly": display_value(metrics.get("p90_anomaly_c"), "°C", True),
            "Data mode": data_mode,
            "Climatology method": climatology_method or "Unavailable",
        })
    with st.expander("How to interpret these indicators", expanded=False):
        st.write(representativeness_notes(language))
        st.write(
            "Read the anomaly together with valid coverage and representativeness. A regional mean "
            "can conceal spatial contrasts even when its numerical value is valid."
        )

with tabs[1]:
    st.info(
        "Maps show complementary dimensions of the current thermal state. They should be "
        "interpreted together rather than as interchangeable indicators."
    )
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
    with st.expander("Map interpretation and limitations", expanded=False):
        st.write(
            "All maps use the fixed Niño 1+2 extent and comparable color limits between dates. "
            "White cells may represent land or unavailable data. Persistence is a fraction, while "
            "SST, anomaly, z-score, and daily change describe different quantities."
        )

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
        with st.expander("Centroid diagnostics", expanded=False):
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
        with st.expander("Detailed temporal metrics and window definitions", expanded=False):
            st.write(
                f"Visible window: {period_choice}. Seven-day means use cached daily observations; "
                "missing dates are not treated as valid observations."
            )

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
        st.info(
            "Persistence indicates repeated threshold exceedance; the centroid summarizes the "
            "location of warm-anomaly area and does not track an individual water parcel."
        )
        with st.expander("Spatial diagnostic details", expanded=False):
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
            with st.expander("Technical identifiers", expanded=False):
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
    )

with tabs[6]:
    st.subheader("Data and methods")
    st.warning(
        "Experimental thermal-monitoring product. It is not an official ENSO or Coastal El Niño "
        "classification, a forecast, or a causal attribution.",
        icon=":material/science:",
    )

    with st.expander("Active data and coverage", expanded=True):
        st.write({
            "Active data source": data_mode,
            "Source file": dataset.attrs.get("source_path"),
            "Available period": f"{dates[0]} to {dates[-1]}",
            "Active study region": "90–80°W, 10°S–0° (Niño 1+2)",
            "Temperature units": dataset.sst.attrs.get("units"),
            "Valid coverage on selected date": display_value(
                diagnosis["metrics"].get("valid_data_coverage_percent"), "%"
            ),
        })
        st.caption(
            "Dates are read from the NetCDF time coordinate, never from global "
            "time-coverage attributes."
        )

    with st.expander("Climatology", expanded=False):
        st.write({
            "Active method": climatology_method or "Unavailable",
            "Reference period": (
                f"{config['climatology']['reference_start']}–"
                f"{config['climatology']['reference_end']}"
            ),
            "Sampling half-window": (
                f"±{config['climatology']['sampling_half_window_days']} calendar days"
            ),
            "Smoothing window": (
                f"{config['climatology']['smoothing_window_days']} circular days"
            ),
            "Fallback method": config["climatology"]["fallback_method"],
            "Leap-day method": (
                climatology.attrs.get("leap_day_method", "Not applicable to monthly fallback")
                if climatology is not None else "Unavailable"
            ),
            "Active climatology file": diagnosis.get("climatology_file"),
            "Fallback used": diagnosis.get("climatology_fallback_used", False),
            "Percentile calculation state": (
                climatology.attrs.get("percentile_calculation_state", "not applicable")
                if climatology is not None else "not_calculated"
            ),
        })

    with st.expander("Scientific calculations", expanded=False):
        st.write({
            "Univariate events": {
                "Source series": config["event_detection"].get("source_metric"),
                "Threshold type": config["event_detection"]["threshold_type"],
                "Fixed threshold": config["event_detection"].get("fixed_anomaly_threshold_c"),
                "Minimum duration (days)": config["event_detection"]["minimum_duration_days"],
                "Allowed interruptions (days)": config["event_detection"]["allowed_gap_days"],
                "Maximum calendar gap (days)": config["event_detection"]["maximum_calendar_gap_days"],
            },
            "Daily patches": {
                "Source variable": config["patch_detection"]["source_variable"],
                "Direction": config["patch_detection"]["direction"],
                "Threshold type": config["patch_detection"]["threshold_type"],
                "Connectivity": config["patch_detection"]["connectivity"],
                "Minimum patch cells": config["patch_detection"]["minimum_patch_cells"],
                "Minimum patch area (km²)": config["patch_detection"]["minimum_patch_area_km2"],
                "Area method": "Spherical latitude-longitude cell quadrilaterals",
            },
            "Tracking": {
                "Maximum calendar gap (days)": config["patch_tracking"]["maximum_calendar_gap_days"],
                "Minimum IoU": config["patch_tracking"]["minimum_iou"],
                "Minimum predecessor overlap": config["patch_tracking"]["minimum_predecessor_overlap"],
                "Minimum successor overlap": config["patch_tracking"]["minimum_successor_overlap"],
                "Distance fallback": config["patch_tracking"]["allow_distance_fallback"],
                "Minimum link score": config["patch_tracking"]["minimum_link_score"],
                "Score weights": config["patch_tracking"]["score_weights"],
                "Continuation backbone": "Deterministic maximum-score one-to-one assignment",
                "Track definition": "Maximal non-branching path through the lineage DAG",
                "Event-family definition": (
                    "Weakly connected component including continuations, splits, and merges"
                ),
            },
        })
        st.write({
            "Daily patch label cube": str(
                resolve_project_path(config["patch_detection"]["labels_output"])
            ),
            "Track and family label cube": str(
                resolve_project_path(config["patch_tracking"]["track_labels_output"])
            ),
            "Tracking summary status": event_products.tracking_summary.status,
        })

    geographic_details = st.expander(
        "Geographic foundation", expanded=False, on_change="rerun"
    )
    if tabs[6].open and geographic_details.open:
        with geographic_details:
            geography_registry = load_geography_registry(config)
            st.info(
                "Niño 3.4, Niño 3, and Niño 1+2 are geographic overlays for context. "
                "Current scientific calculations remain restricted to Niño 1+2."
            )
            st.caption(
                "The 60-nautical-mile corridor is measured only from the continental "
                "Pacific coastline of Ecuador, Peru, and Chile. Islands do not generate buffers."
            )
            st.write({
                "Pacific context bounds": geography_registry.display_domain.bounds.to_dict(),
                "Humboldt coastal bounds": geography_registry.analysis_domains[
                    "humboldt_coastal"
                ].bounds.to_dict(),
                "South Pacific High diagnostic bounds": geography_registry.analysis_domains[
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

    multidomain_details = st.expander(
        "Multidomain SST foundation", expanded=False, on_change="rerun"
    )
    if tabs[6].open and multidomain_details.open:
        with multidomain_details:
            sst_outputs = config["sst"]["outputs"]
            multidomain_paths = {
                "status": resolve_project_path(sst_outputs["status"]),
                "snapshot": resolve_project_path(sst_outputs["snapshot"]),
                "indices": resolve_project_path(
                    config["sst"]["regional_indices"]["output"]
                ),
                "preview": resolve_project_path(sst_outputs["preview_figure"]),
            }
            missing_products = [
                str(path) for path in multidomain_paths.values() if not path.exists()
            ]
            if missing_products:
                st.info(
                    "The multidomain SST cache has not been built yet. Missing local products: "
                    + ", ".join(missing_products)
                )
                st.code(
                    "uv run python scripts\\generate_multidomain_demo_data.py --all\n"
                    "uv run python scripts\\build_multidomain_sst_snapshot.py --allow-demo\n"
                    "uv run python scripts\\preview_multidomain_sst.py",
                    language="powershell",
                )
            else:
                freshness = tuple(
                    path.stat().st_mtime_ns for path in multidomain_paths.values()
                )
                try:
                    multidomain_status, multidomain_snapshot, nino_indices, preview_bytes = (
                        load_multidomain_products(
                            str(multidomain_paths["status"]),
                            str(multidomain_paths["snapshot"]),
                            str(multidomain_paths["indices"]),
                            str(multidomain_paths["preview"]),
                            freshness,
                        )
                    )
                except (OSError, ValueError, KeyError) as exc:
                    st.warning(
                        "Cached multidomain SST products could not be read. No rebuild or "
                        f"download was attempted. Details: {exc}"
                    )
                else:
                    domain_rows = []
                    for domain_id, details in multidomain_status.get("domains", {}).items():
                        dataset_details = details.get("dataset") or {}
                        domain_rows.append(
                            {
                                "Domain": details.get("label", domain_id),
                                "Status": details.get("status", "unknown"),
                                "Latest date": details.get("latest_date"),
                                "Source mode": details.get("source_mode"),
                                "Source resolution (°)": dataset_details.get(
                                    "longitude_resolution_degrees"
                                ),
                                "Target resolution (°)": details.get(
                                    "target_resolution_degrees"
                                ),
                                "Climatology": (
                                    "available"
                                    if details.get("climatology_available")
                                    else details.get("climatology_status", "unavailable")
                                ),
                            }
                        )
                    st.dataframe(pd.DataFrame(domain_rows), hide_index=True, width="stretch")
                    st.image(
                        preview_bytes,
                        caption=(
                            "Pre-built offline preview: Pacific context, Humboldt coastal SST, "
                            "and daily Niño-region mean SST."
                        ),
                        width="stretch",
                    )
                    display_columns = [
                        "date",
                        "region_label",
                        "mean_sst_c",
                        "valid_coverage",
                        "source_mode",
                        "anomaly_status",
                    ]
                    st.dataframe(
                        nino_indices.loc[:, [
                            column for column in display_columns if column in nino_indices
                        ]],
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "mean_sst_c": st.column_config.NumberColumn(
                                "Mean SST (°C)", format="%.2f"
                            ),
                            "valid_coverage": st.column_config.NumberColumn(
                                "Valid coverage", format="percent"
                            ),
                        },
                    )
                    st.warning(
                        "Niño 3.4 and Niño 3 anomalies are unavailable until compatible spatial "
                        "and temporal climatologies are built. No official ENSO classification "
                        "or ONI is calculated.",
                        icon=":material/info:",
                    )
                    st.info(
                        "The detailed operational calculations, events, patches, tracks, "
                        "representativeness, and scientific briefs continue to use Niño 1+2."
                    )
                    if multidomain_snapshot.get("warnings"):
                        st.caption("Cached warnings: " + "; ".join(multidomain_snapshot["warnings"]))

            st.markdown("#### Multidomain climatology and anomaly status")
            anomaly_outputs = config["sst"]["anomaly_outputs"]
            anomaly_paths = {
                "status": resolve_project_path(anomaly_outputs["status"]),
                "indices": resolve_project_path(anomaly_outputs["regional_indices"]),
                "preview": resolve_project_path(anomaly_outputs["preview_figure"]),
            }
            missing_anomaly_products = [
                str(path) for path in anomaly_paths.values() if not path.exists()
            ]
            if missing_anomaly_products:
                st.info(
                    "Prepared multidomain climatology and anomaly products are unavailable. "
                    "No calculation or download was attempted. Missing: "
                    + ", ".join(missing_anomaly_products)
                )
                st.code(
                    "uv run python scripts\\generate_multidomain_demo_climatology.py --all --overwrite\n\n"
                    "uv run python scripts\\build_multidomain_sst_anomalies.py `\n"
                    "  --allow-demo `\n"
                    "  --overwrite\n\n"
                    "uv run python scripts\\preview_multidomain_sst_anomalies.py `\n"
                    "  --overwrite",
                    language="powershell",
                )
            else:
                anomaly_freshness = tuple(
                    path.stat().st_mtime_ns for path in anomaly_paths.values()
                )
                try:
                    anomaly_status, regional_anomalies, anomaly_preview = (
                        load_multidomain_anomaly_products(
                            str(anomaly_paths["status"]),
                            str(anomaly_paths["indices"]),
                            str(anomaly_paths["preview"]),
                            anomaly_freshness,
                        )
                    )
                except (OSError, ValueError, KeyError) as exc:
                    st.warning(
                        "Prepared anomaly products could not be read. No rebuild, write, or "
                        f"download was attempted. Details: {exc}"
                    )
                else:
                    anomaly_rows = []
                    for domain_id, details in anomaly_status.get("domains", {}).items():
                        compatibility = details.get("climatology", {}).get(
                            "compatibility", {}
                        )
                        anomaly = details.get("anomaly", {})
                        anomaly_rows.append(
                            {
                                "Domain": domain_id,
                                "SST type": details.get("sst_mode"),
                                "Climatology type": compatibility.get(
                                    "climatology_mode"
                                ),
                                "Compatibility": compatibility.get("status"),
                                "Method": compatibility.get("method"),
                                "Reference period": compatibility.get(
                                    "reference_period"
                                ),
                                "Resolution (°)": compatibility.get(
                                    "climatology_resolution"
                                ),
                                "Coverage": compatibility.get("coverage"),
                                "Anomaly availability": anomaly.get("status"),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(anomaly_rows),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Coverage": st.column_config.NumberColumn(
                                "Coverage", format="percent"
                            )
                        },
                    )
                    st.image(
                        anomaly_preview,
                        caption=(
                            "Prepared offline anomaly preview — synthetic demonstration "
                            "products are not observations."
                        ),
                        width="stretch",
                    )
                    anomaly_display_columns = [
                        "date",
                        "region_label",
                        "mean_sst_c",
                        "climatological_mean_c",
                        "anomaly_c",
                        "standardized_anomaly",
                        "threshold_p90_c",
                        "exceeds_p90",
                        "valid_coverage",
                        "anomaly_status",
                    ]
                    st.dataframe(
                        regional_anomalies.loc[
                            :,
                            [
                                column
                                for column in anomaly_display_columns
                                if column in regional_anomalies
                            ],
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                    for warning in anomaly_status.get("warnings", []):
                        st.warning(str(warning), icon=":material/info:")
            st.caption(
                "Demo products are not observations. The active detailed operational "
                "pipeline remains Niño 1+2. No official ENSO classification or ONI is produced."
            )

    with st.expander("Quality control", expanded=False):
        if qc_report:
            st.json(qc_report)
        else:
            st.info("The cached quality-control report is unavailable.")

    with st.expander("Limitations and glossary", expanded=False):
        st.info(
            "Local daily patch IDs are not persistent. Tracks are maximal non-branching paths, "
            "while event families may contain splits and merges. Results depend on threshold and "
            "connectivity. Boundary-touching tracks may be incomplete. Tracking describes thermal "
            "features, not individual water parcels."
        )
        glossary_rows = [
            {"Term": human_label(term, language), "Definition": definition}
            for term, definition in glossary(language).items()
        ]
        st.dataframe(pd.DataFrame(glossary_rows), hide_index=True, width="stretch")
        st.warning(scientific_disclaimer(language), icon=":material/science:")


with tabs[7]:
    st.subheader("Export")
    st.caption(
        "Downloads use cached local products and safe serialization. Missing products are "
        "disabled explicitly; rendering this page does not create new analytical outputs."
    )

    with st.expander("Daily diagnosis", expanded=True):
        st.download_button(
            "Download temporal metrics CSV",
            series.to_csv(index=False),
            "nino12_daily_metrics.csv",
            "text/csv",
        )
        st.download_button(
            "Download diagnosis NetCDF",
            bytes(fields.to_netcdf()),
            f"nino12_diagnosis_{analysis_date}.nc",
            "application/x-netcdf",
        )
        diagnosis_json = dumps_json_safe(diagnosis)
        st.download_button(
            label="Download diagnosis JSON",
            data=diagnosis_json,
            file_name=f"nino12_diagnosis_{analysis_date}.json",
            mime="application/json",
        )

    with st.expander("Analytical tables", expanded=False):
        analytical_products = [
            ("Series bank", config["series_bank"]["output"]),
            ("Temporal features", config["temporal_features"]["output"]),
            ("Spatial features", config["spatial_features"]["output"]),
            ("Representativeness", config["representativeness"]["output"]),
        ]
        for index, (label, path) in enumerate(analytical_products):
            render_cached_download(label, path, key=f"export_analytics_{index}")

    with st.expander("Thermal-event products", expanded=False):
        event_paths = [
            ("Univariate events", config["event_detection"]["events_output"]),
            ("Univariate daily flags", config["event_detection"]["daily_flags_output"]),
            ("Daily patches", config["patch_detection"]["patches_output"]),
            ("Daily patch summary", config["patch_detection"]["daily_summary_output"]),
            ("Patch observations", config["patch_tracking"]["observations_output"]),
            ("Lineage edges", config["patch_tracking"]["edges_output"]),
            ("Tracks", config["patch_tracking"]["tracks_output"]),
            ("Event families", config["patch_tracking"]["families_output"]),
        ]
        for index, (label, path) in enumerate(event_paths):
            render_cached_download(label, path, key=f"export_events_{index}")

    with st.expander("Scientific brief products", expanded=False):
        brief_directory = resolve_project_path(config["brief_generation"]["output_directory"])
        brief_files = (
            sorted(
                path
                for path in brief_directory.glob("*")
                if path.is_file() and path.suffix.lower() in {".json", ".md", ".parquet"}
            )
            if brief_directory.exists()
            else []
        )
        if brief_files:
            for index, path in enumerate(brief_files):
                render_cached_download(
                    path.name,
                    path,
                    key=f"export_brief_{index}",
                )
        else:
            st.info("No cached scientific brief products are available.")

    with st.expander("Metadata and validation reports", expanded=False):
        report_paths = [
            ("Quality-control report", config["quality_control"]["output"]),
            (
                "Daily climatology validation",
                "outputs/reports/daily_climatology_validation.json",
            ),
            ("Univariate event summary", config["event_detection"]["summary_output"]),
            ("Daily patch build summary", config["patch_detection"]["json_summary_output"]),
            ("Tracking summary", config["patch_tracking"]["json_summary_output"]),
        ]
        for index, (label, path) in enumerate(report_paths):
            render_cached_download(label, path, key=f"export_reports_{index}")
