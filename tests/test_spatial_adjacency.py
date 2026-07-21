"""Tests for sparse grid adjacency, Moran I, and daily patches."""

from __future__ import annotations

import numpy as np
import pytest

from src.spatial_adjacency import (
    build_adjacency,
    calculate_moran_i,
    calculate_patch_features,
    label_patches,
)


def _metric_map(results):
    return {result.metric: result for result in results}


def test_rook_and_queen_adjacency_have_expected_edges() -> None:
    rook = build_adjacency((2, 2), connectivity="rook")
    queen = build_adjacency((2, 2), connectivity="queen")
    assert rook.n_edges == 8
    assert queen.n_edges == 12
    assert queen.n_edges > rook.n_edges


def test_moran_i_distinguishes_clustered_and_alternating_fields() -> None:
    adjacency = build_adjacency((4, 4), connectivity="rook")
    clustered = np.array([
        [1, 1, -1, -1], [1, 1, -1, -1],
        [1, 1, -1, -1], [1, 1, -1, -1],
    ], dtype=float)
    alternating = np.indices((4, 4)).sum(axis=0) % 2
    alternating = np.where(alternating, 1.0, -1.0)
    clustered_result = calculate_moran_i(clustered, adjacency, minimum_valid_coverage=0.5)
    alternating_result = calculate_moran_i(alternating, adjacency, minimum_valid_coverage=0.5)
    assert clustered_result.status == "valid"
    assert alternating_result.status == "valid"
    assert clustered_result.value > 0
    assert alternating_result.value < 0


def test_moran_i_rejects_zero_variance_and_insufficient_coverage() -> None:
    adjacency = build_adjacency((3, 3), connectivity="queen")
    constant = calculate_moran_i(np.ones((3, 3)), adjacency)
    assert constant.status == "not_calculated"
    assert "variance" in (constant.reason or "").lower()
    sparse = np.full((3, 3), np.nan)
    sparse[0, 0] = 1.0
    insufficient = calculate_moran_i(sparse, adjacency, minimum_valid_coverage=0.8)
    assert insufficient.status == "not_calculated"
    assert "coverage" in (insufficient.reason or "").lower()


def test_rook_and_queen_connectivity_change_diagonal_patch_count() -> None:
    mask = np.array([[True, False], [False, True]])
    _, rook_count = label_patches(mask, connectivity="rook")
    _, queen_count = label_patches(mask, connectivity="queen")
    assert rook_count == 2
    assert queen_count == 1


def test_one_patch_metrics_and_geometry() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    results = _metric_map(calculate_patch_features(
        mask, np.arange(4.0), np.arange(4.0), connectivity="queen"
    ))
    assert results["patch_count"].value == 1
    assert results["total_threshold_area"].value == 4
    assert results["largest_patch_fraction"].value == 1
    assert results["fragmentation_index"].value == pytest.approx(0.0)
    assert results["centroid_dispersion"].status == "valid"
    assert results["orientation"].status == "valid"
    assert results["elongation"].status == "valid"


def test_multiple_patches_fragmentation_and_filtering() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4, 4] = True
    unfiltered = _metric_map(calculate_patch_features(
        mask, np.arange(5.0), np.arange(5.0), connectivity="rook", minimum_patch_cells=1
    ))
    filtered = _metric_map(calculate_patch_features(
        mask, np.arange(5.0), np.arange(5.0), connectivity="rook", minimum_patch_cells=2
    ))
    assert unfiltered["patch_count"].value == 2
    assert unfiltered["fragmentation_index"].value > 0
    assert filtered["patch_count"].value == 1
    assert filtered["total_threshold_area"].value == 4


def test_zero_threshold_area_returns_explicit_not_calculated_geometry() -> None:
    results = _metric_map(calculate_patch_features(
        np.zeros((3, 3), dtype=bool), np.arange(3.0), np.arange(3.0)
    ))
    assert results["patch_count"].value == 0
    assert results["total_threshold_area"].value == 0
    assert results["fragmentation_index"].value == 0
    assert results["centroid_dispersion"].status == "not_calculated"
    assert results["orientation"].status == "not_calculated"
