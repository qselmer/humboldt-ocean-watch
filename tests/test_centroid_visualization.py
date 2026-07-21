"""Tests for centroid trajectory and independent temporal charts."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from src.charting import centroid_temporal_chart
from src.plotting import plot_centroid_trajectory


def metrics(longitude, latitude) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(longitude)),
            "warm_centroid_longitude": longitude,
            "warm_centroid_latitude": latitude,
        }
    )


def test_trajectory_uses_dynamic_bounds_with_eight_percent_padding() -> None:
    figure = plot_centroid_trajectory(metrics([-85.0, -84.0], [-5.0, -3.0]), "2026-01-02")
    axis = figure.axes[0]
    np.testing.assert_allclose(axis.get_xlim(), [-85.1, -83.9])
    np.testing.assert_allclose(axis.get_ylim(), [-5.16, -2.84])
    plt.close(figure)


def test_constant_coordinate_uses_minimum_padding() -> None:
    figure = plot_centroid_trajectory(metrics([-84.0, -84.0], [-5.0, -4.0]), "2026-01-02")
    np.testing.assert_allclose(figure.axes[0].get_xlim(), [-84.1, -83.9])
    plt.close(figure)


def test_trajectory_never_exceeds_maximum_arrow_count() -> None:
    frame = metrics(np.linspace(-85, -80, 100), np.linspace(-8, -2, 100))
    figure = plot_centroid_trajectory(frame, "2026-02-01", maximum_arrows=7)
    arrows = [patch for patch in figure.axes[0].patches if isinstance(patch, FancyArrowPatch)]
    assert len(arrows) <= 7
    plt.close(figure)


def test_missing_centroid_dates_are_removed_without_failure() -> None:
    frame = metrics([-85.0, np.nan, -83.0], [-5.0, -4.0, -3.0])
    figure = plot_centroid_trajectory(frame, "2026-01-02")
    line = figure.axes[0].lines[0]
    assert len(line.get_xdata()) == 2
    plt.close(figure)


def test_latitude_and_longitude_series_are_separate_and_break_gaps() -> None:
    frame = metrics([-85.0, np.nan, -83.0], [-5.0, -4.0, -3.0])
    latitude = centroid_temporal_chart(frame, coordinate="latitude", selected_date="2026-01-03")
    longitude = centroid_temporal_chart(frame, coordinate="longitude", selected_date="2026-01-03")
    assert latitude is not None and longitude is not None
    latitude_spec = latitude.to_dict()
    longitude_spec = longitude.to_dict()
    assert latitude_spec["layer"][0]["encoding"]["y"]["field"] == "warm_centroid_latitude"
    assert longitude_spec["layer"][0]["encoding"]["y"]["field"] == "warm_centroid_longitude"
    assert longitude_spec["layer"][0]["mark"]["invalid"] == "break-paths-show-domains"
    assert latitude_spec["layer"][1]["mark"]["type"] == "rule"
