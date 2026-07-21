"""Deterministic tests for spherical pairwise patch linking."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.patch_linking import (
    PatchTrackingConfig,
    calculate_pair_metrics,
    link_date_pair,
)


def _row(date: str, patch_id: int, cells: np.ndarray, *, latitude=0.0, longitude=0.0):
    return {
        "date": pd.Timestamp(date),
        "patch_id": patch_id,
        "area_km2": float(cells.sum()),
        "centroid_latitude": latitude,
        "centroid_longitude": longitude,
        "mean_source_value": 3.0,
        "maximum_source_value": 4.0,
        "touches_north_boundary": False,
        "touches_south_boundary": False,
        "touches_east_boundary": False,
        "touches_west_boundary": False,
    }


def _labels(identifier: int, cells: list[tuple[int, int]], shape=(4, 4)) -> np.ndarray:
    values = np.zeros(shape, dtype=np.int32)
    for row, column in cells:
        values[row, column] = identifier
    return values


def test_stationary_continuation_has_unit_overlap_and_score_bounds() -> None:
    labels = _labels(1, [(1, 1), (1, 2)])
    result = calculate_pair_metrics(
        _row("2024-01-01", 1, labels == 1),
        _row("2024-01-02", 1, labels == 1),
        labels,
        labels,
        np.ones(labels.shape),
        elapsed_days=1,
        config=PatchTrackingConfig(),
    )
    assert result["intersection_cell_count"] == 2
    assert result["area_weighted_iou"] == pytest.approx(1.0)
    assert result["candidate_status"] == "accepted"
    assert result["link_basis"] == "overlap"
    assert 0 <= result["link_score"] <= 1


def test_translating_patch_with_overlap_uses_spherical_area() -> None:
    before = _labels(1, [(1, 0), (1, 1)])
    after = _labels(2, [(1, 1), (1, 2)])
    areas = np.arange(1, 17, dtype=float).reshape(4, 4)
    result = calculate_pair_metrics(
        _row("2024-01-01", 1, before == 1),
        _row("2024-01-02", 2, after == 2, longitude=0.5),
        before,
        after,
        areas,
        elapsed_days=1,
        config=PatchTrackingConfig(),
    )
    assert result["intersection_area_km2"] == areas[1, 1]
    assert result["union_area_km2"] == areas[1, 0] + areas[1, 1] + areas[1, 2]
    assert result["candidate_status"] == "accepted"


def test_nonoverlap_requires_explicit_distance_fallback() -> None:
    before = _labels(1, [(1, 0)])
    after = _labels(2, [(1, 2)])
    rows_before = pd.DataFrame([_row("2024-01-01", 1, before == 1, longitude=0.0)])
    rows_after = pd.DataFrame([_row("2024-01-02", 2, after == 2, longitude=0.1)])
    disabled = link_date_pair(
        rows_before, rows_after, before, after, np.ones(before.shape),
        predecessor_date="2024-01-01", successor_date="2024-01-02",
        config=PatchTrackingConfig(),
    )
    assert disabled.empty
    enabled = link_date_pair(
        rows_before, rows_after, before, after, np.ones(before.shape),
        predecessor_date="2024-01-01", successor_date="2024-01-02",
        config=replace(PatchTrackingConfig(), allow_distance_fallback=True),
    )
    assert len(enabled) == 1
    assert enabled.iloc[0].link_basis == "distance_fallback"


@pytest.mark.parametrize(
    ("iou", "predecessor", "successor"),
    [(0.2, 0.99, 0.99), (0.99, 0.4, 0.99), (0.99, 0.99, 0.4)],
)
def test_any_overlap_criterion_can_make_a_candidate(iou, predecessor, successor) -> None:
    before = _labels(1, [(1, 0), (1, 1)])
    after = _labels(2, [(1, 1), (1, 2)])
    config = replace(
        PatchTrackingConfig(), minimum_iou=iou,
        minimum_predecessor_overlap=predecessor,
        minimum_successor_overlap=successor,
        minimum_link_score=0.0,
    )
    result = calculate_pair_metrics(
        _row("2024-01-01", 1, before == 1),
        _row("2024-01-02", 2, after == 2),
        before, after, np.ones(before.shape), elapsed_days=1, config=config,
    )
    assert result["candidate_status"] == "accepted"


def test_distance_fallback_rejects_area_ratio_outside_bounds() -> None:
    before = _labels(1, [(0, 0), (0, 1), (1, 0), (1, 1)])
    after = _labels(2, [(3, 3)])
    config = replace(
        PatchTrackingConfig(), allow_distance_fallback=True,
        maximum_centroid_distance_km=1000.0, minimum_area_ratio=0.5,
    )
    result = calculate_pair_metrics(
        _row("2024-01-01", 1, before == 1),
        _row("2024-01-02", 2, after == 2),
        before, after, np.ones(before.shape), elapsed_days=1, config=config,
    )
    assert result["candidate_status"] == "rejected"


def test_score_weights_are_validated() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        PatchTrackingConfig(score_weights={
            "iou": 1.0, "predecessor_overlap": 1.0,
            "successor_overlap": 0.0, "centroid_proximity": 0.0,
            "area_similarity": 0.0,
        })
    with pytest.raises(ValueError, match="non-negative"):
        PatchTrackingConfig(score_weights={
            "iou": 1.1, "predecessor_overlap": -0.1,
            "successor_overlap": 0.0, "centroid_proximity": 0.0,
            "area_similarity": 0.0,
        })


def test_hungarian_backbone_is_one_to_one_and_ties_are_deterministic() -> None:
    before = np.asarray([[1, 1, 2, 2]], dtype=np.int32)
    after = np.asarray([[1, 2, 1, 2]], dtype=np.int32)
    before_rows = pd.DataFrame([
        _row("2024-01-01", 1, before == 1),
        _row("2024-01-01", 2, before == 2),
    ])
    after_rows = pd.DataFrame([
        _row("2024-01-02", 1, after == 1),
        _row("2024-01-02", 2, after == 2),
    ])
    result = link_date_pair(
        before_rows, after_rows, before, after, np.ones(before.shape),
        predecessor_date="2024-01-01", successor_date="2024-01-02",
        config=replace(PatchTrackingConfig(), minimum_link_score=0.0),
    )
    backbone = result.loc[result.is_continuation_backbone]
    assert len(backbone) == 2
    assert backbone.predecessor_patch_id.is_unique
    assert backbone.successor_patch_id.is_unique
    repeated = link_date_pair(
        before_rows, after_rows, before, after, np.ones(before.shape),
        predecessor_date="2024-01-01", successor_date="2024-01-02",
        config=replace(PatchTrackingConfig(), minimum_link_score=0.0),
    )
    assert backbone[["predecessor_patch_id", "successor_patch_id"]].values.tolist() == repeated.loc[
        repeated.is_continuation_backbone,
        ["predecessor_patch_id", "successor_patch_id"],
    ].values.tolist()


def test_temporal_gap_is_not_bridged_unless_explicitly_enabled() -> None:
    labels = _labels(1, [(1, 1)])
    row_before = _row("2024-01-01", 1, labels == 1)
    row_after = _row("2024-01-03", 1, labels == 1)
    rejected = calculate_pair_metrics(
        row_before, row_after, labels, labels, np.ones(labels.shape),
        elapsed_days=2, config=PatchTrackingConfig(),
    )
    assert rejected["candidate_status"] == "rejected"
    bridged = calculate_pair_metrics(
        row_before, row_after, labels, labels, np.ones(labels.shape),
        elapsed_days=2,
        config=replace(
            PatchTrackingConfig(), allow_gap_bridging=True,
            maximum_bridge_gap_days=2,
        ),
    )
    assert bridged["candidate_status"] == "accepted"
    assert bridged["crosses_temporal_gap"] is True
