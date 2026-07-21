"""Pairwise spatial-patch linking and deterministic continuation assignment.

Daily patch identifiers are deliberately treated as date-local labels.  This
module compares patches only between chronologically adjacent *available*
dates, calculates overlap with spherical cell areas, and retains two related
but distinct products:

* all accepted candidate edges, used to describe splits and merges; and
* a maximum-score one-to-one Hungarian assignment, used as the continuation
  backbone.

Centroid distances use a haversine approximation with the IUGG mean Earth
radius from :mod:`src.grid_geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.grid_geometry import EARTH_RADIUS_KM


PAIR_METRIC_COLUMNS = [
    "predecessor_date", "predecessor_patch_id", "successor_date",
    "successor_patch_id", "elapsed_days", "intersection_cell_count",
    "intersection_area_km2", "union_area_km2", "area_weighted_iou",
    "predecessor_overlap_fraction", "successor_overlap_fraction",
    "centroid_distance_km", "area_ratio", "log_area_ratio",
    "mean_intensity_difference", "maximum_intensity_difference",
    "boundary_touching_status", "candidate_status", "rejection_reason",
    "link_score", "link_basis", "crosses_temporal_gap",
    "is_continuation_backbone",
]


@dataclass(frozen=True)
class PatchTrackingConfig:
    """Validated configuration for temporal patch links."""

    maximum_calendar_gap_days: int = 1
    allow_gap_bridging: bool = False
    maximum_bridge_gap_days: int = 0
    minimum_iou: float = 0.05
    minimum_predecessor_overlap: float = 0.15
    minimum_successor_overlap: float = 0.15
    allow_distance_fallback: bool = False
    maximum_centroid_distance_km: float = 100.0
    minimum_area_ratio: float = 0.25
    maximum_area_ratio: float = 4.0
    minimum_link_score: float = 0.10
    score_weights: Mapping[str, float] | None = None
    preserve_all_candidate_edges: bool = True

    def __post_init__(self) -> None:
        if self.maximum_calendar_gap_days < 1:
            raise ValueError("maximum_calendar_gap_days must be at least one")
        if self.maximum_bridge_gap_days < 0:
            raise ValueError("maximum_bridge_gap_days cannot be negative")
        if self.allow_gap_bridging and (
            self.maximum_bridge_gap_days <= self.maximum_calendar_gap_days
        ):
            raise ValueError(
                "Gap bridging requires maximum_bridge_gap_days greater than "
                "maximum_calendar_gap_days"
            )
        for name in (
            "minimum_iou", "minimum_predecessor_overlap",
            "minimum_successor_overlap", "minimum_link_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not np.isfinite(self.maximum_centroid_distance_km) or (
            self.maximum_centroid_distance_km <= 0
        ):
            raise ValueError("maximum_centroid_distance_km must be finite and positive")
        if not np.isfinite(self.minimum_area_ratio) or self.minimum_area_ratio <= 0:
            raise ValueError("minimum_area_ratio must be finite and positive")
        if not np.isfinite(self.maximum_area_ratio) or (
            self.maximum_area_ratio < self.minimum_area_ratio
        ):
            raise ValueError("maximum_area_ratio must be at least minimum_area_ratio")
        expected = {
            "iou", "predecessor_overlap", "successor_overlap",
            "centroid_proximity", "area_similarity",
        }
        weights = dict(self.score_weights or {
            "iou": 0.40,
            "predecessor_overlap": 0.20,
            "successor_overlap": 0.20,
            "centroid_proximity": 0.10,
            "area_similarity": 0.10,
        })
        if set(weights) != expected:
            raise ValueError(
                "score_weights must contain exactly: " + ", ".join(sorted(expected))
            )
        values = np.asarray(list(weights.values()), dtype=float)
        if not np.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("Link score weights must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-9):
            raise ValueError("Link score weights must sum to one")
        object.__setattr__(self, "score_weights", weights)


def tracking_config(values: Mapping[str, Any]) -> PatchTrackingConfig:
    """Construct a validated tracking configuration from YAML values."""
    required = {
        "maximum_calendar_gap_days", "allow_gap_bridging",
        "maximum_bridge_gap_days", "minimum_iou",
        "minimum_predecessor_overlap", "minimum_successor_overlap",
        "allow_distance_fallback", "maximum_centroid_distance_km",
        "minimum_area_ratio", "maximum_area_ratio", "minimum_link_score",
        "score_weights", "preserve_all_candidate_edges",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError("patch_tracking configuration is missing: " + ", ".join(missing))
    return PatchTrackingConfig(**{name: values[name] for name in required})


def node_identifier(date: Any, local_patch_id: int) -> str:
    """Return a stable node identifier without implying patch persistence."""
    stamp = pd.Timestamp(date).normalize()
    if pd.isna(stamp) or int(local_patch_id) < 1:
        raise ValueError("A node requires a readable date and positive local patch ID")
    return f"{stamp.date().isoformat()}::P{int(local_patch_id):06d}"


def haversine_distance_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return great-circle distance in kilometres for two finite centroids."""
    coordinates = np.asarray(
        [latitude_a, longitude_a, latitude_b, longitude_b], dtype=float
    )
    if not np.isfinite(coordinates).all():
        return np.nan
    phi_a, phi_b = np.deg2rad([latitude_a, latitude_b])
    delta_phi = phi_b - phi_a
    delta_lambda = np.deg2rad((longitude_b - longitude_a + 180.0) % 360.0 - 180.0)
    haversine = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi_a) * np.cos(phi_b) * np.sin(delta_lambda / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))))


def initial_bearing_degrees(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the initial bearing clockwise from true north in [0, 360)."""
    coordinates = np.asarray(
        [latitude_a, longitude_a, latitude_b, longitude_b], dtype=float
    )
    if not np.isfinite(coordinates).all():
        return np.nan
    phi_a, phi_b = np.deg2rad([latitude_a, latitude_b])
    delta_lambda = np.deg2rad((longitude_b - longitude_a + 180.0) % 360.0 - 180.0)
    y = np.sin(delta_lambda) * np.cos(phi_b)
    x = np.cos(phi_a) * np.sin(phi_b) - np.sin(phi_a) * np.cos(phi_b) * np.cos(delta_lambda)
    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        return np.nan
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def _row_value(row: Mapping[str, Any] | pd.Series, name: str) -> Any:
    if name not in row:
        raise ValueError(f"Patch record is missing required field {name!r}")
    return row[name]


def _boundary_status(row: Mapping[str, Any] | pd.Series) -> bool:
    names = (
        "touches_north_boundary", "touches_south_boundary",
        "touches_east_boundary", "touches_west_boundary",
    )
    return bool(any(bool(row.get(name, False)) for name in names))


def _score_components(
    *,
    iou: float,
    predecessor_overlap: float,
    successor_overlap: float,
    centroid_distance: float,
    area_ratio: float,
    config: PatchTrackingConfig,
) -> tuple[float, float, float]:
    proximity = (
        max(0.0, 1.0 - centroid_distance / config.maximum_centroid_distance_km)
        if np.isfinite(centroid_distance)
        else 0.0
    )
    similarity = (
        min(area_ratio, 1.0 / area_ratio)
        if np.isfinite(area_ratio) and area_ratio > 0
        else 0.0
    )
    weights = config.score_weights
    assert weights is not None
    score = (
        weights["iou"] * iou
        + weights["predecessor_overlap"] * predecessor_overlap
        + weights["successor_overlap"] * successor_overlap
        + weights["centroid_proximity"] * proximity
        + weights["area_similarity"] * similarity
    )
    return float(np.clip(score, 0.0, 1.0)), proximity, similarity


def calculate_pair_metrics(
    predecessor: Mapping[str, Any] | pd.Series,
    successor: Mapping[str, Any] | pd.Series,
    predecessor_labels: np.ndarray,
    successor_labels: np.ndarray,
    cell_area_km2: np.ndarray,
    *,
    elapsed_days: int,
    config: PatchTrackingConfig,
) -> dict[str, Any]:
    """Calculate spherical-overlap diagnostics and candidate status for a pair."""
    before = np.asarray(predecessor_labels)
    after = np.asarray(successor_labels)
    areas = np.asarray(cell_area_km2, dtype=float)
    if before.ndim != 2 or after.shape != before.shape or areas.shape != before.shape:
        raise ValueError("Pair labels and cell areas must share one two-dimensional grid")
    if not np.isfinite(areas).all() or bool((areas <= 0).any()):
        raise ValueError("Pairwise cell areas must be finite and positive")
    if elapsed_days < 1:
        raise ValueError("elapsed_days must be positive")
    predecessor_id = int(_row_value(predecessor, "patch_id"))
    successor_id = int(_row_value(successor, "patch_id"))
    predecessor_cells = before == predecessor_id
    successor_cells = after == successor_id
    if not predecessor_cells.any() or not successor_cells.any():
        raise ValueError("Every linked local patch must have at least one labelled cell")
    intersection = predecessor_cells & successor_cells
    union = predecessor_cells | successor_cells
    intersection_area = float(areas[intersection].sum())
    union_area = float(areas[union].sum())
    predecessor_area = float(areas[predecessor_cells].sum())
    successor_area = float(areas[successor_cells].sum())
    if min(union_area, predecessor_area, successor_area) <= 0:
        raise ValueError("Patch overlap denominators must be positive")
    iou = intersection_area / union_area
    predecessor_overlap = intersection_area / predecessor_area
    successor_overlap = intersection_area / successor_area
    area_ratio = successor_area / predecessor_area
    distance = haversine_distance_km(
        float(_row_value(predecessor, "centroid_latitude")),
        float(_row_value(predecessor, "centroid_longitude")),
        float(_row_value(successor, "centroid_latitude")),
        float(_row_value(successor, "centroid_longitude")),
    )
    score, _, _ = _score_components(
        iou=iou,
        predecessor_overlap=predecessor_overlap,
        successor_overlap=successor_overlap,
        centroid_distance=distance,
        area_ratio=area_ratio,
        config=config,
    )
    overlap_candidate = bool(
        iou >= config.minimum_iou
        or predecessor_overlap >= config.minimum_predecessor_overlap
        or successor_overlap >= config.minimum_successor_overlap
    )
    distance_eligible = bool(
        np.isfinite(distance)
        and distance <= config.maximum_centroid_distance_km
        and config.minimum_area_ratio <= area_ratio <= config.maximum_area_ratio
    )
    crosses_gap = elapsed_days > config.maximum_calendar_gap_days
    temporal_allowed = elapsed_days <= config.maximum_calendar_gap_days
    if crosses_gap:
        temporal_allowed = bool(
            config.allow_gap_bridging
            and elapsed_days <= config.maximum_bridge_gap_days
            and distance_eligible
        )
    fallback_candidate = bool(
        distance_eligible
        and (
            (config.allow_distance_fallback and not intersection.any())
            or crosses_gap
        )
    )
    candidate = temporal_allowed and (overlap_candidate or fallback_candidate)
    distance_basis_enabled = bool(
        config.allow_distance_fallback
        or (crosses_gap and config.allow_gap_bridging)
    )
    if overlap_candidate and distance_eligible and distance_basis_enabled:
        basis = "both"
    elif overlap_candidate:
        basis = "overlap"
    elif fallback_candidate:
        basis = "distance_fallback"
    else:
        basis = None
    reasons: list[str] = []
    if not temporal_allowed:
        reasons.append("Temporal gap exceeds the configured linking allowance")
    if not overlap_candidate and not fallback_candidate:
        reasons.append("Pair satisfies neither overlap nor enabled distance criteria")
    if candidate and score < config.minimum_link_score:
        candidate = False
        reasons.append("Link score is below minimum_link_score")
    mean_difference = float(
        float(_row_value(successor, "mean_source_value"))
        - float(_row_value(predecessor, "mean_source_value"))
    )
    maximum_difference = float(
        float(_row_value(successor, "maximum_source_value"))
        - float(_row_value(predecessor, "maximum_source_value"))
    )
    return {
        "predecessor_date": pd.Timestamp(_row_value(predecessor, "date")).normalize(),
        "predecessor_patch_id": predecessor_id,
        "successor_date": pd.Timestamp(_row_value(successor, "date")).normalize(),
        "successor_patch_id": successor_id,
        "elapsed_days": int(elapsed_days),
        "intersection_cell_count": int(intersection.sum()),
        "intersection_area_km2": intersection_area,
        "union_area_km2": union_area,
        "area_weighted_iou": iou,
        "predecessor_overlap_fraction": predecessor_overlap,
        "successor_overlap_fraction": successor_overlap,
        "centroid_distance_km": distance,
        "area_ratio": area_ratio,
        "log_area_ratio": float(np.log(area_ratio)),
        "mean_intensity_difference": mean_difference,
        "maximum_intensity_difference": maximum_difference,
        "boundary_touching_status": bool(
            _boundary_status(predecessor) or _boundary_status(successor)
        ),
        "candidate_status": "accepted" if candidate else "rejected",
        "rejection_reason": "; ".join(reasons) if reasons else None,
        "link_score": score,
        "link_basis": basis,
        "crosses_temporal_gap": crosses_gap,
        "is_continuation_backbone": False,
    }


def _label_bounding_boxes(labels: np.ndarray, identifiers: list[int]) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for identifier in identifiers:
        rows, columns = np.where(labels == identifier)
        if rows.size:
            boxes[identifier] = (
                int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())
            )
    return boxes


def _boxes_intersect(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[1] < second[0] or second[1] < first[0]
        or first[3] < second[2] or second[3] < first[2]
    )


def _assign_backbone(candidates: list[dict[str, Any]]) -> set[tuple[int, int]]:
    if not candidates:
        return set()
    predecessor_ids = sorted({int(row["predecessor_patch_id"]) for row in candidates})
    successor_ids = sorted({int(row["successor_patch_id"]) for row in candidates})
    predecessor_index = {value: index for index, value in enumerate(predecessor_ids)}
    successor_index = {value: index for index, value in enumerate(successor_ids)}
    score = np.zeros((len(predecessor_ids), len(successor_ids)), dtype=float)
    valid = np.zeros_like(score, dtype=bool)
    for row in candidates:
        i = predecessor_index[int(row["predecessor_patch_id"])]
        j = successor_index[int(row["successor_patch_id"])]
        # Lexicographic diagnostics are represented below the precision of the
        # published score. Sorted IDs make the final tie-break deterministic.
        identifier_tie = (
            (len(predecessor_ids) - i) * (len(successor_ids) - j) * 1e-14
        )
        tie = (
            float(row["area_weighted_iou"]) * 1e-10
            + min(float(row["intersection_area_km2"]), 1e6) * 1e-18
            + max(0.0, 1.0 - float(row["centroid_distance_km"]) / 20050.0) * 1e-12
            + identifier_tie
        )
        score[i, j] = float(row["link_score"]) + tie
        valid[i, j] = True
    rows, columns = linear_sum_assignment(-score)
    accepted: set[tuple[int, int]] = set()
    for i, j in zip(rows, columns, strict=True):
        if valid[i, j]:
            accepted.add((predecessor_ids[i], successor_ids[j]))
    return accepted


def link_date_pair(
    predecessor_patches: pd.DataFrame,
    successor_patches: pd.DataFrame,
    predecessor_labels: np.ndarray,
    successor_labels: np.ndarray,
    cell_area_km2: np.ndarray,
    *,
    predecessor_date: Any,
    successor_date: Any,
    config: PatchTrackingConfig,
) -> pd.DataFrame:
    """Return all accepted edges and mark a deterministic Hungarian backbone.

    Bounding boxes prefilter overlap work.  When distance fallback or explicit
    temporal-gap bridging is enabled, vectorized centroid distances add only
    geometrically eligible non-overlap pairs to the comparison set.
    """
    columns = PAIR_METRIC_COLUMNS
    if predecessor_patches.empty or successor_patches.empty:
        return pd.DataFrame(columns=columns)
    before_date = pd.Timestamp(predecessor_date).normalize()
    after_date = pd.Timestamp(successor_date).normalize()
    elapsed = int((after_date - before_date).days)
    if elapsed < 1:
        raise ValueError("Successor date must be later than predecessor date")
    before = predecessor_patches.sort_values("patch_id").reset_index(drop=True)
    after = successor_patches.sort_values("patch_id").reset_index(drop=True)
    before_ids = before.patch_id.astype(int).tolist()
    after_ids = after.patch_id.astype(int).tolist()
    before_boxes = _label_bounding_boxes(np.asarray(predecessor_labels), before_ids)
    after_boxes = _label_bounding_boxes(np.asarray(successor_labels), after_ids)
    if set(before_boxes) != set(before_ids) or set(after_boxes) != set(after_ids):
        raise ValueError("Patch table identifiers do not match the supplied label fields")

    pair_indices: set[tuple[int, int]] = set()
    for i, predecessor_id in enumerate(before_ids):
        for j, successor_id in enumerate(after_ids):
            if _boxes_intersect(before_boxes[predecessor_id], after_boxes[successor_id]):
                pair_indices.add((i, j))
    needs_distance = config.allow_distance_fallback or (
        config.allow_gap_bridging and elapsed > config.maximum_calendar_gap_days
    )
    if needs_distance:
        before_lat = before.centroid_latitude.to_numpy(dtype=float)[:, None]
        before_lon = before.centroid_longitude.to_numpy(dtype=float)[:, None]
        after_lat = after.centroid_latitude.to_numpy(dtype=float)[None, :]
        after_lon = after.centroid_longitude.to_numpy(dtype=float)[None, :]
        phi_before = np.deg2rad(before_lat)
        phi_after = np.deg2rad(after_lat)
        delta_phi = phi_after - phi_before
        delta_lon = np.deg2rad((after_lon - before_lon + 180.0) % 360.0 - 180.0)
        haversine = np.sin(delta_phi / 2) ** 2 + np.cos(phi_before) * np.cos(phi_after) * np.sin(delta_lon / 2) ** 2
        distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        ratios = after.area_km2.to_numpy(dtype=float)[None, :] / before.area_km2.to_numpy(dtype=float)[:, None]
        eligible = (
            (distances <= config.maximum_centroid_distance_km)
            & (ratios >= config.minimum_area_ratio)
            & (ratios <= config.maximum_area_ratio)
        )
        for i, j in np.argwhere(eligible):
            pair_indices.add((int(i), int(j)))

    candidates: list[dict[str, Any]] = []
    for i, j in sorted(pair_indices, key=lambda value: (before_ids[value[0]], after_ids[value[1]])):
        result = calculate_pair_metrics(
            before.iloc[i], after.iloc[j], predecessor_labels, successor_labels,
            cell_area_km2, elapsed_days=elapsed, config=config,
        )
        if result["candidate_status"] == "accepted":
            candidates.append(result)
    backbone = _assign_backbone(candidates)
    for row in candidates:
        row["is_continuation_backbone"] = (
            int(row["predecessor_patch_id"]), int(row["successor_patch_id"])
        ) in backbone
    return pd.DataFrame(candidates, columns=columns)
