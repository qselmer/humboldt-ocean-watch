"""Sanitized Copernicus Marine preflight and the sole download adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import inspect
import multiprocessing
import os
from pathlib import Path
import queue as queue_module
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import xarray as xr

from src.geography import GeographicBounds

PREFLIGHT_STATUSES = {
    "valid", "authentication_unavailable", "dataset_not_found", "variable_not_found",
    "temporal_coverage_insufficient", "spatial_coverage_insufficient", "api_incompatible",
    "metadata_unavailable", "network_error",
}
REQUIRED_SUBSET_PARAMETERS = {
    "dataset_id", "variables", "minimum_longitude", "maximum_longitude",
    "minimum_latitude", "maximum_latitude", "start_datetime", "end_datetime",
    "output_filename", "output_directory", "file_format", "netcdf_compression_level",
}
REQUIRED_DESCRIBE_PARAMETERS = {"dataset_id", "show_all_versions", "raise_on_error"}
REQUIRED_OPEN_PARAMETERS = {
    "dataset_id", "variables", "minimum_longitude", "maximum_longitude",
    "minimum_latitude", "maximum_latitude", "start_datetime", "end_datetime",
    "chunk_size_limit",
}


@dataclass(frozen=True)
class SourceMetadata:
    dataset_id: str
    variable: str
    product_id: str | None
    dataset_version: str | None
    coordinate_names: tuple[str, ...]
    longitude_bounds: tuple[float, float] | None
    latitude_bounds: tuple[float, float] | None
    temporal_bounds: tuple[str, str] | None
    longitude_resolution_degrees: float | None
    latitude_resolution_degrees: float | None
    units: str | None


@dataclass(frozen=True)
class SourcePreflight:
    status: str
    offline: bool
    dataset_id: str
    source_variable: str
    toolbox_version: str | None
    selected_metadata_api: str
    selected_download_api: str
    api_signatures_valid: bool
    authentication_checked: bool
    metadata: SourceMetadata | None
    domains_checked: tuple[str, ...]
    checked_at: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Unsupported catalogue response")


def _api_compatible(module: Any) -> bool:
    for name, required in (
        ("describe", REQUIRED_DESCRIBE_PARAMETERS),
        ("subset", REQUIRED_SUBSET_PARAMETERS),
        ("open_dataset", REQUIRED_OPEN_PARAMETERS),
    ):
        function = getattr(module, name, None)
        if function is None or not required <= set(inspect.signature(function).parameters):
            return False
    return True


def _coordinate(variable: Mapping[str, Any], *names: str) -> Mapping[str, Any] | None:
    for coordinate in variable.get("coordinates", []):
        if str(coordinate.get("coordinate_id", "")).lower() in names:
            return coordinate
    return None


def _float_pair(coordinate: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if coordinate is None:
        return None
    try:
        return float(coordinate["minimum_value"]), float(coordinate["maximum_value"])
    except (KeyError, TypeError, ValueError):
        return None


def _temporal_pair(coordinate: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if coordinate is None:
        return None
    lower, upper = coordinate.get("minimum_value"), coordinate.get("maximum_value")
    return None if lower is None or upper is None else (str(lower), str(upper))


def extract_source_metadata(catalogue: Any, dataset_id: str, variable_name: str) -> tuple[str, SourceMetadata | None]:
    """Extract the latest official catalogue contract without endpoint details."""
    root = _mapping(catalogue)
    dataset: Mapping[str, Any] | None = None
    product: Mapping[str, Any] | None = None
    for candidate_product in root.get("products", []):
        for candidate in candidate_product.get("datasets", []):
            if candidate.get("dataset_id") == dataset_id:
                dataset, product = candidate, candidate_product
                break
    if dataset is None or product is None:
        return "dataset_not_found", None
    versions = dataset.get("versions", [])
    if not versions:
        return "metadata_unavailable", None
    version = versions[0]
    variable: Mapping[str, Any] | None = None
    for part in version.get("parts", []):
        for service in part.get("services", []):
            for candidate in service.get("variables", []):
                if candidate.get("short_name") == variable_name:
                    variable = candidate
                    break
    if variable is None:
        return "variable_not_found", None
    longitude = _coordinate(variable, "longitude", "lon", "x")
    latitude = _coordinate(variable, "latitude", "lat", "y")
    temporal = _coordinate(variable, "time", "t")
    names = tuple(str(item.get("coordinate_id")) for item in variable.get("coordinates", []))
    lon_step = None if longitude is None else longitude.get("step")
    lat_step = None if latitude is None else latitude.get("step")
    metadata = SourceMetadata(
        dataset_id=dataset_id, variable=variable_name,
        product_id=str(product.get("product_id")) if product.get("product_id") else None,
        dataset_version=str(version.get("label")) if version.get("label") else None,
        coordinate_names=names,
        longitude_bounds=_float_pair(longitude), latitude_bounds=_float_pair(latitude),
        temporal_bounds=_temporal_pair(temporal),
        longitude_resolution_degrees=None if lon_step is None else abs(float(lon_step)),
        latitude_resolution_degrees=None if lat_step is None else abs(float(lat_step)),
        units=None if variable.get("units") is None else str(variable.get("units")),
    )
    return "valid", metadata


def validate_source_metadata(
    metadata: SourceMetadata, domains: Mapping[str, GeographicBounds],
    reference_start: str, reference_end: str,
) -> str:
    if metadata.temporal_bounds is None:
        return "metadata_unavailable"
    def parse_remote_datetime(value: str) -> datetime:
        try:
            numeric = float(value)
        except ValueError:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Copernicus Marine Toolbox 2.4 catalogue coordinates serialize time
        # as Unix epoch milliseconds (for example 370742400000.0).
        if abs(numeric) > 10_000_000_000:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)

    try:
        available_start = parse_remote_datetime(metadata.temporal_bounds[0])
        available_end = parse_remote_datetime(metadata.temporal_bounds[1])
        requested_start = datetime.fromisoformat(reference_start).replace(tzinfo=available_start.tzinfo)
        requested_end = datetime.fromisoformat(reference_end).replace(tzinfo=available_end.tzinfo)
    except ValueError:
        return "metadata_unavailable"
    if available_start > requested_start or available_end < requested_end:
        return "temporal_coverage_insufficient"
    if metadata.longitude_bounds is None or metadata.latitude_bounds is None:
        return "metadata_unavailable"
    west, east = metadata.longitude_bounds
    south, north = metadata.latitude_bounds
    if any(
        bounds.west < west or bounds.east > east or bounds.south < south or bounds.north > north
        for bounds in domains.values()
    ):
        return "spatial_coverage_insufficient"
    if not {"longitude", "latitude", "time"} <= {name.lower() for name in metadata.coordinate_names}:
        return "metadata_unavailable"
    return "valid"


def run_source_preflight(
    config: Mapping[str, Any], domains: Mapping[str, GeographicBounds], *, offline: bool,
    module: Any | None = None, catalogue: Any | None = None,
) -> SourcePreflight:
    """Validate local API and, online, catalogue plus one tiny authenticated read."""
    build = config["real_climatology_build"]
    dataset_id, variable = str(build["dataset_id"]), str(build["source_variable"])
    checked = datetime.now(timezone.utc).isoformat()
    if "NRT" in dataset_id.upper() or dataset_id != "METOFFICE-GLO-SST-L4-REP-OBS-SST":
        return SourcePreflight("dataset_not_found", offline, dataset_id, variable, None,
            "describe", "subset", False, False, None, tuple(domains), checked,
            "Configured source is not the approved OSTIA reprocessed dataset")
    try:
        if module is None:
            import copernicusmarine as module
        version = str(getattr(module, "__version__", "unknown"))
        compatible = _api_compatible(module)
    except (ImportError, TypeError, ValueError):
        version, compatible = None, False
    if not compatible:
        return SourcePreflight("api_incompatible", offline, dataset_id, variable, version,
            "describe", "subset", False, False, None, tuple(domains), checked,
            "Installed Copernicus Marine Toolbox API is incompatible")
    if offline:
        return SourcePreflight("valid", True, dataset_id, variable, version, "describe", "subset",
            True, False, None, tuple(domains), checked,
            "Local configuration and API contracts are valid; remote metadata was not checked")
    try:
        response = catalogue if catalogue is not None else module.describe(
            dataset_id=dataset_id, show_all_versions=True, raise_on_error=True,
            disable_progress_bar=True,
        )
        status, metadata = extract_source_metadata(response, dataset_id, variable)
        if status == "valid" and metadata is not None:
            status = validate_source_metadata(
                metadata, domains,
                f"{int(build['start_year'])}-01-01T00:00:00",
                f"{int(build['end_year'])}-12-31T23:59:59",
            )
        if status != "valid" or metadata is None:
            return SourcePreflight(status, False, dataset_id, variable, version, "describe", "subset",
                True, False, metadata, tuple(domains), checked, "Remote source metadata did not satisfy the contract")
        # A single grid-cell/day lazy read proves the Toolbox can authenticate.
        probe = module.open_dataset(
            dataset_id=dataset_id, variables=[variable], minimum_longitude=-85.0,
            maximum_longitude=-85.0, minimum_latitude=-5.0, maximum_latitude=-5.0,
            start_datetime=f"{int(build['start_year'])}-01-01T00:00:00",
            end_datetime=f"{int(build['start_year'])}-01-01T23:59:59", chunk_size_limit=-1,
        )
        try:
            probe[variable].isel(time=0, latitude=0, longitude=0).load()
        finally:
            probe.close()
        return SourcePreflight("valid", False, dataset_id, variable, version, "describe", "subset",
            True, True, metadata, tuple(domains), checked,
            "Remote metadata and a minimal authenticated read are valid")
    except Exception as exc:  # Remote library exceptions vary between releases.
        name = type(exc).__name__.lower()
        status = "authentication_unavailable" if any(token in name for token in ("credential", "auth", "login")) else "network_error"
        return SourcePreflight(status, False, dataset_id, variable, version, "describe", "subset",
            True, False, None, tuple(domains), checked,
            f"Remote preflight failed ({type(exc).__name__}); details were sanitized")


class CopernicusSSTAdapter:
    """Single controlled adapter for metadata, dry-run, pilot, and full modes."""

    def __init__(
        self, module: Any, *, attempts: int = 3,
        initial_backoff_seconds: float = 5.0, timeout_seconds: float = 600.0,
    ):
        self.module = module
        self.attempts = attempts
        self.initial_backoff_seconds = initial_backoff_seconds
        self.timeout_seconds = timeout_seconds

    def subset_to_netcdf(
        self, *, dataset_id: str, variable: str, bounds: GeographicBounds,
        start_datetime: str, end_datetime: str, target: Path, mode: str,
        compression_level: int,
    ) -> Path:
        if mode not in {"pilot", "full"}:
            raise ValueError("Network subset is allowed only in pilot or full mode")
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            with tempfile.TemporaryDirectory(prefix="cmems-subset-", dir=target.parent) as temporary:
                try:
                    kwargs = {
                        "dataset_id": dataset_id, "variables": [variable],
                        "minimum_longitude": bounds.west, "maximum_longitude": bounds.east,
                        "minimum_latitude": bounds.south, "maximum_latitude": bounds.north,
                        "start_datetime": start_datetime, "end_datetime": end_datetime,
                        "output_filename": target.name, "output_directory": temporary,
                        "file_format": "netcdf", "netcdf_compression_level": compression_level,
                        "overwrite": True, "disable_progress_bar": True,
                    }
                    if getattr(self.module, "__name__", "") == "copernicusmarine":
                        _timed_copernicus_subset(kwargs, self.timeout_seconds)
                        response = None
                    else:  # Deterministic injected adapter for offline tests.
                        response = self.module.subset(**kwargs)
                    downloaded = Path(getattr(response, "file_path", Path(temporary) / target.name))
                    if not downloaded.exists():
                        candidates = tuple(Path(temporary).glob("*.nc"))
                        if len(candidates) != 1:
                            raise RuntimeError("Copernicus subset did not create one NetCDF file")
                        downloaded = candidates[0]
                    os.replace(downloaded, target)
                    return target
                except Exception as exc:
                    last_error = exc
            if attempt < self.attempts:
                time.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"Copernicus subset failed after {self.attempts} attempts") from last_error


def _subset_process(kwargs: dict[str, Any], queue: Any) -> None:
    """Child entry point: keep credentials within the official Toolbox process."""
    try:
        import copernicusmarine
        copernicusmarine.subset(**kwargs)
        queue.put(("ok", None))
    except Exception as exc:
        queue.put(("error", type(exc).__name__))


def _timed_copernicus_subset(kwargs: dict[str, Any], timeout_seconds: float) -> None:
    """Run one subset in a killable child so timeout cancellation is real."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_subset_process, args=(kwargs, queue), daemon=False)
    process.start()
    try:
        process.join(timeout_seconds)
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(10)
        raise
    if process.is_alive():
        process.terminate()
        process.join(10)
        raise TimeoutError(f"Copernicus subset exceeded the {timeout_seconds:g}-second policy")
    try:
        status, error_type = queue.get(timeout=2)
    except queue_module.Empty:
        raise RuntimeError("Copernicus subset child exited without a result")
    if status != "ok":
        raise RuntimeError(f"Copernicus subset failed ({error_type})")
