"""Atomic NetCDF checkpoints with contract fingerprints and SHA-256 sidecars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class CheckpointContract:
    domain_id: str
    tile_id: str
    processing_stage: str
    source_dataset: str
    source_variable: str
    target_resolution: float
    checkpoint_year: int | None = None
    source_version: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointValidation:
    valid: bool
    reason: str
    checksum: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def checkpoint_metadata(
    contract: CheckpointContract, *, requested_bounds: tuple[float, float, float, float],
    effective_bounds: tuple[float, float, float, float], source_resolution: float,
    valid_coverage: float, time_range: tuple[str, str], aggregation_applied: bool,
    created_at: str, software_revision: str, network_source: str = "Copernicus Marine",
) -> dict[str, Any]:
    values = {
        **asdict(contract),
        "source_product_family": "ostia",
        "requested_bounds": ",".join(map(str, requested_bounds)),
        "effective_bounds": ",".join(map(str, effective_bounds)),
        "source_resolution": float(source_resolution),
        "target_resolution": float(contract.target_resolution),
        "time_range": " to ".join(time_range),
        "units": "degrees_Celsius",
        "valid_coverage": float(valid_coverage),
        "created_at": created_at,
        "software_revision": software_revision,
        "contract_fingerprint": contract.fingerprint,
        "aggregation_applied": "true" if aggregation_applied else "false",
        "network_source": network_source,
    }
    return {key: value for key, value in values.items() if value is not None}


def validate_checkpoint(path: Path, contract: CheckpointContract) -> CheckpointValidation:
    if not path.exists():
        return CheckpointValidation(False, "missing checkpoint")
    sidecar = checksum_path(path)
    if not sidecar.exists():
        return CheckpointValidation(False, "missing checksum sidecar")
    try:
        expected_checksum = sidecar.read_text(encoding="ascii").strip()
        actual_checksum = sha256_file(path)
        if expected_checksum != actual_checksum:
            return CheckpointValidation(False, "checksum mismatch", actual_checksum)
        with xr.open_dataset(path) as dataset:
            if dataset.attrs.get("contract_fingerprint") != contract.fingerprint:
                return CheckpointValidation(False, "contract fingerprint mismatch", actual_checksum)
            if dataset.attrs.get("domain_id") != contract.domain_id:
                return CheckpointValidation(False, "domain mismatch", actual_checksum)
            if dataset.attrs.get("tile_id") != contract.tile_id:
                return CheckpointValidation(False, "tile mismatch", actual_checksum)
            if dataset.attrs.get("processing_stage") != contract.processing_stage:
                return CheckpointValidation(False, "processing stage mismatch", actual_checksum)
            if not np.isclose(float(dataset.attrs.get("target_resolution", -1)), contract.target_resolution):
                return CheckpointValidation(False, "target resolution mismatch", actual_checksum)
            if contract.checkpoint_year is not None:
                if int(dataset.attrs.get("checkpoint_year", -1)) != contract.checkpoint_year:
                    return CheckpointValidation(False, "checkpoint year mismatch", actual_checksum)
                dates = pd.DatetimeIndex(dataset.time.values)
                if dates.empty or set(dates.year) != {contract.checkpoint_year}:
                    return CheckpointValidation(False, "time coordinate year mismatch", actual_checksum)
            variable = "sst" if "sst" in dataset else "climatology_mean" if "climatology_mean" in dataset else None
            if variable is None:
                return CheckpointValidation(False, "missing scientific variable", actual_checksum)
            if dataset[variable].attrs.get("units") != "degrees_Celsius":
                return CheckpointValidation(False, "units mismatch", actual_checksum)
            if np.isinf(dataset[variable].values).any():
                return CheckpointValidation(False, "infinite values", actual_checksum)
    except (OSError, ValueError, TypeError, UnicodeError, KeyError) as exc:
        return CheckpointValidation(False, f"unreadable checkpoint ({type(exc).__name__})")
    return CheckpointValidation(True, "valid", actual_checksum)


def atomic_write_checkpoint(
    dataset: xr.Dataset, path: Path, contract: CheckpointContract,
    *, encoding: Mapping[str, Mapping[str, Any]] | None = None,
    extra_validator: Callable[[xr.Dataset], list[str]] | None = None,
) -> CheckpointValidation:
    """Write, validate, and replace both file and checksum without touching a valid target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = dataset.copy()

    def safe_attributes(values: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)):
                result[key] = "true" if bool(value) else "false"
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Mapping):
                result[key] = json.dumps(value, sort_keys=True)
            else:
                result[key] = value
        return result

    candidate.attrs = safe_attributes(candidate.attrs)
    for name in candidate.variables:
        candidate[name].attrs = safe_attributes(candidate[name].attrs)
    candidate.attrs["contract_fingerprint"] = contract.fingerprint
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".nc", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    temporary_sidecar = checksum_path(temporary)
    try:
        candidate.to_netcdf(temporary, encoding=dict(encoding or {}))
        if extra_validator is not None:
            with xr.open_dataset(temporary) as opened:
                errors = extra_validator(opened)
            if errors:
                raise ValueError("; ".join(errors))
        checksum = sha256_file(temporary)
        temporary_sidecar.write_text(checksum, encoding="ascii")
        os.replace(temporary, path)
        os.replace(temporary_sidecar, checksum_path(path))
        result = validate_checkpoint(path, contract)
        if not result.valid:
            raise ValueError(f"Published checkpoint failed validation: {result.reason}")
        return result
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
