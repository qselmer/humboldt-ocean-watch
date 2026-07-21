from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd

from src.event_charts import (
    lineage_chart,
    plot_track_trajectory,
    temporal_metric_chart,
    univariate_event_timeline,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": ["2026-01-01_P0001", "2026-01-02_P0001", "2026-01-04_P0001"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]),
        "local_patch_id": [1, 1, 1], "track_id": ["TRK000001"] * 3,
        "event_family_id": ["FAM000001"] * 3,
        "node_event_type": ["appearance", "continuation", "termination"],
        "area_km2": [10.0, 12.0, 11.0], "mean_exceedance": [1.0, 1.2, 1.1],
        "centroid_longitude": [-86.0, -85.8, -85.5], "centroid_latitude": [-5.0, -4.8, -4.7],
    })


def test_univariate_timeline_has_event_regions_and_selected_date() -> None:
    flags = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4), "value": [0.0, 2.0, 2.5, 0.0],
        "threshold": [1.0] * 4, "intensity": [np.nan, 1.0, 1.5, np.nan],
        "event_id": [None, "EVT001", "EVT001", None], "event_day": [np.nan, 1, 2, np.nan],
    })
    events = pd.DataFrame({"event_id": ["EVT001"], "start_date": [pd.Timestamp("2026-01-02")], "end_date": [pd.Timestamp("2026-01-03")], "peak_date": [pd.Timestamp("2026-01-03")]})
    chart = univariate_event_timeline(flags, events, "2026-01-02")
    assert chart is not None
    spec = chart.to_dict()
    assert len(spec["layer"]) >= 5


def test_temporal_chart_handles_one_day_and_marks_selected_date() -> None:
    chart = temporal_metric_chart(pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "area": [10.0]}), value_column="area", selected_date="2026-01-01", title="Area", unit="km²")
    assert chart is not None
    spec = chart.to_dict()
    assert len(spec["layer"]) == 2
    domain = spec["layer"][0]["encoding"]["y"]["scale"]["domain"]
    assert domain[0] < 10 < domain[1]


def test_track_trajectory_dynamic_bounds_arrows_and_missing_gap() -> None:
    figure = plot_track_trajectory(_observations(), "2026-01-02", maximum_arrows=25)
    assert isinstance(figure, matplotlib.figure.Figure)
    axis = figure.axes[0]
    assert axis.get_xlim()[0] < -86 and axis.get_xlim()[1] > -85.5
    assert axis.get_ylim()[0] < -5 and axis.get_ylim()[1] > -4.7
    assert len(axis.patches) == 1  # The two-day missing gap is not bridged.


def test_constant_one_day_track_has_minimum_padding_and_no_arrows() -> None:
    figure = plot_track_trajectory(_observations().iloc[[0]], "2026-01-01")
    axis = figure.axes[0]
    assert np.isclose(np.diff(axis.get_xlim())[0], 0.2)
    assert np.isclose(np.diff(axis.get_ylim())[0], 0.2)
    assert len(axis.patches) == 0


def test_separate_centroid_charts_use_their_own_domains() -> None:
    data = _observations()
    latitude = temporal_metric_chart(data, value_column="centroid_latitude", selected_date="2026-01-02", title="Latitude", unit="degrees north")
    longitude = temporal_metric_chart(data, value_column="centroid_longitude", selected_date="2026-01-02", title="Longitude", unit="degrees east")
    assert latitude is not None and longitude is not None
    assert latitude.to_dict()["layer"][0]["encoding"]["y"]["scale"]["domain"] != longitude.to_dict()["layer"][0]["encoding"]["y"]["scale"]["domain"]


def test_lineage_view_split_merge_and_maximum_node_simplification() -> None:
    nodes = pd.concat([_observations(), _observations().assign(
        node_id=["2026-01-01_P0002", "2026-01-02_P0002", "2026-01-04_P0002"],
        local_patch_id=2, track_id="TRK000002", node_event_type="complex_branch",
    )], ignore_index=True)
    edges = pd.DataFrame({
        "predecessor_node_id": [nodes.node_id.iloc[0], nodes.node_id.iloc[0], nodes.node_id.iloc[4]],
        "successor_node_id": [nodes.node_id.iloc[1], nodes.node_id.iloc[4], nodes.node_id.iloc[2]],
        "lineage_relation": ["continuation", "split", "merge"],
        "is_continuation_backbone": [True, False, False],
    })
    result = lineage_chart(nodes, edges, selected_date="2026-01-02", selected_track="TRK000001", maximum_nodes=4)
    assert result.chart is not None
    assert result.simplified
    assert result.displayed_node_count <= 4
    assert result.chart.to_dict()["layer"]


def test_lineage_empty_input_has_explicit_empty_result() -> None:
    result = lineage_chart(pd.DataFrame(), pd.DataFrame(), selected_date="2026-01-01")
    assert result.chart is None and result.displayed_node_count == 0

