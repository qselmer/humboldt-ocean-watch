"""Deterministic daily spatial thermal-patch identification.

Labels are local to one analysis date. They are deliberately not persistent
track identifiers and are never linked across dates in Increment 4B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr
from scipy import ndimage

from src.patch_validation import ValidatedPatchInput, validate_patch_inputs
from src.spatial_adjacency import Connectivity

PatchDirection = Literal["above", "below"]
PatchComparison = Literal["inclusive", "exclusive"]
PatchThresholdType = Literal["fixed", "standardized", "daily_climatological"]


@dataclass(frozen=True)
class PatchDetectionConfig:
    """Configuration for a single-date threshold mask and component filter."""

    source_variable: str
    direction: PatchDirection
    threshold_type: PatchThresholdType
    comparison: PatchComparison = "inclusive"
    connectivity: Connectivity = "queen"
    minimum_patch_cells: int = 4
    minimum_patch_area_km2: float = 0.0
    remove_boundary_only_artifacts: bool = False

    def __post_init__(self) -> None:
        if not self.source_variable.strip():
            raise ValueError("source_variable must be non-empty")
        if self.direction not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
        if self.threshold_type not in {"fixed", "standardized", "daily_climatological"}:
            raise ValueError(
                "threshold_type must be fixed, standardized, or daily_climatological"
            )
        if self.comparison not in {"inclusive", "exclusive"}:
            raise ValueError("comparison must be 'inclusive' or 'exclusive'")
        if self.connectivity not in {"rook", "queen"}:
            raise ValueError("connectivity must be 'rook' or 'queen'")
        if self.minimum_patch_cells < 1:
            raise ValueError("minimum_patch_cells must be positive")
        if not np.isfinite(self.minimum_patch_area_km2) or self.minimum_patch_area_km2 < 0:
            raise ValueError("minimum_patch_area_km2 must be finite and non-negative")


@dataclass(frozen=True)
class PatchDetectionResult:
    """A retained label field plus unfiltered threshold and validity masks."""

    patch_id: xr.DataArray
    threshold_mask: xr.DataArray
    valid_ocean_mask: xr.DataArray
    exceedance: xr.DataArray
    validated: ValidatedPatchInput
    raw_component_count: int
    retained_patch_count: int
    removed_by_cell_count: int
    removed_by_area: int
    removed_boundary_only: int
    status: str
    reason: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def comparison_expression(direction: PatchDirection, comparison: PatchComparison) -> str:
    return {
        ("above", "inclusive"): "source >= threshold",
        ("above", "exclusive"): "source > threshold",
        ("below", "inclusive"): "source <= threshold",
        ("below", "exclusive"): "source < threshold",
    }[(direction, comparison)]


def threshold_definition(
    config: PatchDetectionConfig,
    *,
    threshold_name: str | None = None,
    scalar_threshold: float | None = None,
) -> str:
    """Return a stable human-readable patch definition."""
    if config.threshold_type == "daily_climatological":
        basis = f"daily smoothed climatological {threshold_name or 'threshold'}"
    elif config.threshold_type == "standardized":
        basis = f"standardized anomaly threshold {scalar_threshold:g}"
    else:
        basis = f"fixed {config.source_variable} threshold {scalar_threshold:g}"
    return f"{basis}; {comparison_expression(config.direction, config.comparison)}"


def _threshold_comparison(
    source: np.ndarray,
    threshold: np.ndarray,
    *,
    direction: PatchDirection,
    comparison: PatchComparison,
) -> np.ndarray:
    if direction == "above":
        return source >= threshold if comparison == "inclusive" else source > threshold
    return source <= threshold if comparison == "inclusive" else source < threshold


def _relative_exceedance(
    source: np.ndarray,
    threshold: np.ndarray,
    direction: PatchDirection,
) -> np.ndarray:
    return source - threshold if direction == "above" else threshold - source


def _boundary_only(cells: np.ndarray) -> bool:
    boundary = np.zeros(cells.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return bool(cells.any() and np.all(boundary[cells]))


def identify_patches(
    field: xr.DataArray,
    threshold: float | xr.DataArray | np.ndarray,
    *,
    config: PatchDetectionConfig,
    analysis_date: object | None = None,
    ocean_mask: xr.DataArray | np.ndarray | None = None,
    cell_areas: xr.DataArray | np.ndarray | None = None,
) -> PatchDetectionResult:
    """Identify, filter, and consecutively relabel components for one date."""
    validated = validate_patch_inputs(
        field,
        threshold,
        analysis_date=analysis_date,
        ocean_mask=ocean_mask,
        cell_areas=cell_areas,
    )
    source_values = np.asarray(validated.field.values, dtype=float)
    threshold_values = np.asarray(validated.threshold.values, dtype=float)
    valid = np.asarray(validated.valid_ocean_mask.values, dtype=bool)
    raw_mask = valid & _threshold_comparison(
        source_values,
        threshold_values,
        direction=config.direction,
        comparison=config.comparison,
    )
    exceedance_values = np.where(
        valid,
        _relative_exceedance(source_values, threshold_values, config.direction),
        np.nan,
    )
    structure = ndimage.generate_binary_structure(
        2, 1 if config.connectivity == "rook" else 2
    )
    preliminary, raw_count = ndimage.label(raw_mask, structure=structure)
    areas = np.asarray(validated.cell_area_km2.values, dtype=float)
    labels = np.zeros(raw_mask.shape, dtype=np.int32)
    next_label = 0
    removed_by_cells = 0
    removed_by_area = 0
    removed_boundary = 0
    for component in range(1, int(raw_count) + 1):
        cells = preliminary == component
        if int(cells.sum()) < config.minimum_patch_cells:
            removed_by_cells += 1
            continue
        component_area = float(areas[cells].sum())
        if component_area < config.minimum_patch_area_km2:
            removed_by_area += 1
            continue
        if config.remove_boundary_only_artifacts and _boundary_only(cells):
            removed_boundary += 1
            continue
        next_label += 1
        labels[cells] = next_label

    coordinates = validated.field.coords
    patch_array = xr.DataArray(
        labels,
        dims=validated.field.dims,
        coords=coordinates,
        name="patch_id",
        attrs={
            "long_name": "local daily thermal patch identifier",
            "valid_min": 0,
            "patch_id_scope": "local to each date; not a persistent track identifier",
        },
    )
    mask_array = xr.DataArray(
        raw_mask,
        dims=validated.field.dims,
        coords=coordinates,
        name="threshold_mask",
    )
    exceedance_array = xr.DataArray(
        exceedance_values,
        dims=validated.field.dims,
        coords=coordinates,
        name="exceedance",
        attrs={"long_name": "source exceedance relative to active threshold"},
    )
    if next_label == 0:
        if raw_count == 0:
            reason = "No valid ocean cells exceeded the configured threshold"
        else:
            reason = "Threshold components existed but none passed the configured patch filters"
        status = "not_calculated"
    else:
        reason = None
        status = "valid"
    return PatchDetectionResult(
        patch_id=patch_array,
        threshold_mask=mask_array,
        valid_ocean_mask=validated.valid_ocean_mask,
        exceedance=exceedance_array,
        validated=validated,
        raw_component_count=int(raw_count),
        retained_patch_count=next_label,
        removed_by_cell_count=removed_by_cells,
        removed_by_area=removed_by_area,
        removed_boundary_only=removed_boundary,
        status=status,
        reason=reason,
        warnings=validated.warnings,
    )
