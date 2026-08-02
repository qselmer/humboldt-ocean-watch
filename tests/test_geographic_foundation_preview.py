"""Integration tests for offline geographic preview products."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from scripts.preview_geographic_foundation import (
    _contains_absolute_path,
    build_foundation_artifacts,
)
from src.coastline_source import LocalCoastlineSource, LocalGeometries
from src.geographic_foundation import prepare_geographic_foundation
from src.utils import load_config


def test_preview_writes_png_and_safe_json_with_synthetic_coast(tmp_path) -> None:
    source = LocalCoastlineSource(
        resolution="synthetic",
        root=tmp_path,
        coastline_path=tmp_path / "synthetic_coastline.shp",
        land_path=tmp_path / "synthetic_land.shp",
        borders_path=None,
    )
    geometries = LocalGeometries(
        display_coastline=LineString([(-75.0, -44.0), (-75.0, 1.0)]),
        all_land=box(-75.0, -45.0, -70.0, 2.0),
        borders=None,
        mainland_land=box(-75.0, -45.0, -70.0, 2.0),
        mainland_source_coastline=LineString([(-75.0, -44.0), (-75.0, 1.0)]),
        island_land=None,
    )
    output = tmp_path / "foundation.png"
    report = tmp_path / "foundation.json"
    artifacts = build_foundation_artifacts(
        config_path=Path("config.yaml"),
        output_path=output,
        report_path=report,
        overwrite=True,
        source=source,
        local_geometries=geometries,
    )
    assert artifacts.output_path == output
    assert output.is_file() and output.stat().st_size > 0
    image = mpimg.imread(output)
    assert image.shape[0] >= 500
    assert image.shape[1] >= 900
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "passed"
    assert payload["coastal_corridor_geometry_status"]["available"] is True
    corridor_status = payload["coastal_corridor_geometry_status"]
    assert corridor_status["coastline_type"] == "continental_mainland_only"
    assert corridor_status["include_islands"] is False
    assert corridor_status["islands_excluded_as_buffer_sources"] is True
    assert corridor_status["all_land_retained_as_exclusion_mask"] is True
    assert corridor_status["buffer_distance_nm"] == 60.0
    assert corridor_status["buffer_distance_km"] == 111.12
    serialized = report.read_text(encoding="utf-8")
    assert "NaN" not in serialized and "Infinity" not in serialized
    assert _contains_absolute_path(payload) is False


def test_preview_preserves_existing_output_without_overwrite(tmp_path) -> None:
    output = tmp_path / "foundation.png"
    report = tmp_path / "foundation.json"
    output.write_bytes(b"previous")
    try:
        build_foundation_artifacts(
            config_path=Path("config.yaml"),
            output_path=output,
            report_path=report,
            overwrite=False,
        )
    except FileExistsError:
        pass
    else:  # pragma: no cover - explicit assertion message.
        raise AssertionError("Existing preview should require --overwrite")
    assert output.read_bytes() == b"previous"


def test_foundation_without_local_coastline_is_available_with_warning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.geographic_foundation.find_local_coastline_source", lambda: None
    )
    foundation = prepare_geographic_foundation(load_config(Path("config.yaml")))
    assert foundation.source_status.available is False
    assert foundation.corridor is None
    assert foundation.corridor_status["status"] == "geometry_source_unavailable"
    assert foundation.warnings


def test_preview_foundation_does_not_buffer_a_disconnected_island(tmp_path) -> None:
    source = LocalCoastlineSource(
        resolution="synthetic",
        root=tmp_path,
        coastline_path=tmp_path / "synthetic_coastline.shp",
        land_path=tmp_path / "synthetic_land.shp",
        borders_path=None,
    )
    mainland = box(-75.0, -45.0, -70.0, 2.0)
    island = box(-78.3, -20.2, -78.1, -20.0)
    mainland_coast = LineString([(-75.0, -44.0), (-75.0, 1.0)])
    geometries = LocalGeometries(
        display_coastline=unary_union([mainland_coast, island.boundary]),
        all_land=unary_union([mainland, island]),
        borders=None,
        mainland_land=mainland,
        mainland_source_coastline=mainland_coast,
        island_land=island,
    )
    foundation = prepare_geographic_foundation(
        load_config(Path("config.yaml")),
        source=source,
        local_geometries=geometries,
    )
    assert foundation.corridor is not None
    assert not foundation.corridor.buffer(1.0e-12).contains(island.centroid)
    assert foundation.corridor_status["islands_excluded_as_buffer_sources"] is True


def test_app_imports_without_starting_streamlit_server() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import app; print('APP_IMPORT_OK')"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "APP_IMPORT_OK" in result.stdout
