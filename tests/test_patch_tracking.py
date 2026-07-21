"""Lineage-event, track-segment, family, and DAG tests."""

import pandas as pd
import pytest

from src.patch_linking import PAIR_METRIC_COLUMNS, node_identifier
from src.patch_tracking import build_lineage, validate_lineage_dag


def _patches(specification):
    rows = []
    for date, patch_id, latitude, longitude in specification:
        rows.append({
            "date": pd.Timestamp(date), "patch_id": patch_id,
            "source_variable": "anomaly", "threshold_type": "fixed",
            "direction": "above", "area_km2": 10.0 + patch_id,
            "centroid_latitude": latitude, "centroid_longitude": longitude,
            "mean_source_value": 3.0, "maximum_source_value": 4.0,
            "mean_exceedance": 1.0, "maximum_exceedance": 2.0,
            "valid_coverage": 1.0, "climatology_method": "daily_smoothed",
            "data_mode": "Copernicus cached data", "status": "valid", "reason": None,
            "touches_north_boundary": False, "touches_south_boundary": False,
            "touches_east_boundary": False, "touches_west_boundary": False,
        })
    return pd.DataFrame(rows)


def _edge(before_date, before_id, after_date, after_id, *, backbone=True, gap=False):
    return {
        "predecessor_date": pd.Timestamp(before_date),
        "predecessor_patch_id": before_id,
        "successor_date": pd.Timestamp(after_date),
        "successor_patch_id": after_id,
        "elapsed_days": int((pd.Timestamp(after_date) - pd.Timestamp(before_date)).days),
        "intersection_cell_count": 1,
        "intersection_area_km2": 1.0,
        "union_area_km2": 1.0,
        "area_weighted_iou": 1.0,
        "predecessor_overlap_fraction": 1.0,
        "successor_overlap_fraction": 1.0,
        "centroid_distance_km": 0.0,
        "area_ratio": 1.0,
        "log_area_ratio": 0.0,
        "mean_intensity_difference": 0.0,
        "maximum_intensity_difference": 0.0,
        "boundary_touching_status": False,
        "candidate_status": "accepted", "rejection_reason": None,
        "link_score": 1.0, "link_basis": "both",
        "crosses_temporal_gap": gap,
        "is_continuation_backbone": backbone,
    }


def _edges(*rows):
    return pd.DataFrame(rows, columns=PAIR_METRIC_COLUMNS)


def test_appearance_continuation_and_termination_form_one_track() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.0, -85.0),
        ("2024-01-02", 1, -5.0, -85.0),
        ("2024-01-03", 1, -5.0, -85.0),
    ])
    observations, edges = build_lineage(patches, _edges(
        _edge("2024-01-01", 1, "2024-01-02", 1),
        _edge("2024-01-02", 1, "2024-01-03", 1),
    ))
    assert observations.node_event_type.tolist() == ["appearance", "continuation", "termination"]
    assert observations.track_id.nunique() == 1
    assert observations.event_family_id.nunique() == 1
    assert set(edges.lineage_relation) == {"continuation"}


def test_isolated_patch_receives_own_one_day_track_and_family() -> None:
    observations, edges = build_lineage(
        _patches([("2024-01-01", 7, -4.0, -84.0)]), _edges()
    )
    assert edges.empty
    assert observations.iloc[0].node_event_type == "isolated_single_day"
    assert observations.iloc[0].track_id == "TRK000001"
    assert observations.iloc[0].event_family_id == "FAM000001"


def test_one_to_two_split_starts_new_child_tracks() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.0, -85.0),
        ("2024-01-02", 1, -5.1, -85.1),
        ("2024-01-02", 2, -4.9, -84.9),
    ])
    observations, edges = build_lineage(patches, _edges(
        _edge("2024-01-01", 1, "2024-01-02", 1, backbone=True),
        _edge("2024-01-01", 1, "2024-01-02", 2, backbone=False),
    ))
    assert observations.loc[observations.date == pd.Timestamp("2024-01-01"), "node_event_type"].item() == "split_parent"
    assert set(observations.loc[observations.date == pd.Timestamp("2024-01-02"), "node_event_type"]) == {"split_child"}
    assert observations.track_id.nunique() == 3
    assert observations.event_family_id.nunique() == 1
    assert set(edges.lineage_relation) == {"split"}


def test_two_to_one_merge_terminates_parent_tracks() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.1, -85.1),
        ("2024-01-01", 2, -4.9, -84.9),
        ("2024-01-02", 1, -5.0, -85.0),
    ])
    observations, edges = build_lineage(patches, _edges(
        _edge("2024-01-01", 1, "2024-01-02", 1, backbone=True),
        _edge("2024-01-01", 2, "2024-01-02", 1, backbone=False),
    ))
    assert set(observations.loc[observations.date == pd.Timestamp("2024-01-01"), "node_event_type"]) == {"merge_parent"}
    assert observations.loc[observations.date == pd.Timestamp("2024-01-02"), "node_event_type"].item() == "merge_child"
    assert observations.track_id.nunique() == 3
    assert set(edges.lineage_relation) == {"merge"}


def test_many_to_many_branch_is_complex() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.2, -85.2), ("2024-01-01", 2, -4.8, -84.8),
        ("2024-01-02", 1, -5.1, -85.1), ("2024-01-02", 2, -4.9, -84.9),
    ])
    edge_rows = [
        _edge("2024-01-01", before, "2024-01-02", after, backbone=before == after)
        for before in (1, 2) for after in (1, 2)
    ]
    observations, edges = build_lineage(patches, _edges(*edge_rows))
    assert set(observations.node_event_type) == {"complex_branch"}
    assert set(edges.lineage_relation) == {"complex"}
    assert observations.track_id.nunique() == 4


def test_separate_components_get_deterministic_families_and_tracks() -> None:
    patches = _patches([
        ("2024-01-01", 2, -4.0, -84.0), ("2024-01-01", 1, -6.0, -86.0),
        ("2024-01-02", 2, -4.0, -84.0), ("2024-01-02", 1, -6.0, -86.0),
    ])
    edges = _edges(
        _edge("2024-01-01", 1, "2024-01-02", 1),
        _edge("2024-01-01", 2, "2024-01-02", 2),
    )
    first, _ = build_lineage(patches.sample(frac=1, random_state=4), edges)
    second, _ = build_lineage(patches.sample(frac=1, random_state=9), edges)
    key = ["node_id", "track_id", "event_family_id"]
    assert first[key].sort_values("node_id").reset_index(drop=True).equals(
        second[key].sort_values("node_id").reset_index(drop=True)
    )
    assert first.event_family_id.nunique() == 2


def test_split_branches_that_later_merge_share_one_family() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.0, -85.0),
        ("2024-01-02", 1, -5.2, -85.2), ("2024-01-02", 2, -4.8, -84.8),
        ("2024-01-03", 1, -5.0, -85.0),
    ])
    observations, _ = build_lineage(patches, _edges(
        _edge("2024-01-01", 1, "2024-01-02", 1),
        _edge("2024-01-01", 1, "2024-01-02", 2, backbone=False),
        _edge("2024-01-02", 1, "2024-01-03", 1),
        _edge("2024-01-02", 2, "2024-01-03", 1, backbone=False),
    ))
    assert observations.event_family_id.nunique() == 1
    assert observations.track_id.nunique() == 4


def test_gap_metadata_is_retained_on_both_nodes() -> None:
    patches = _patches([
        ("2024-01-01", 1, -5.0, -85.0),
        ("2024-01-03", 1, -5.0, -85.0),
    ])
    observations, _ = build_lineage(
        patches, _edges(_edge("2024-01-01", 1, "2024-01-03", 1, gap=True))
    )
    assert observations.link_crossed_temporal_gap.all()


def test_directed_cycle_and_backward_edges_are_rejected() -> None:
    first = node_identifier("2024-01-01", 1)
    second = node_identifier("2024-01-02", 1)
    backward = pd.DataFrame([{
        "predecessor_node_id": second, "successor_node_id": first,
        "predecessor_date": pd.Timestamp("2024-01-02"),
        "successor_date": pd.Timestamp("2024-01-01"),
    }])
    with pytest.raises(ValueError, match="forward"):
        validate_lineage_dag([first, second], backward)
