"""Build the resumable 1991-2020 daily smoothed OSTIA climatology."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_climatology import build_daily_statistics, validate_daily_climatology
from src.data_loader import find_sst_variable, normalize_sst, subset_nino12

DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
SOURCE_VARIABLE = "analysed_sst"
LONGITUDE_BOUNDS = (-90.0, -80.0)
LATITUDE_BOUNDS = (-10.0, 0.0)
SOFTWARE_VERSION = "Humboldt Ocean Watch Increment 2"
LOGGER = logging.getLogger(__name__)


def copernicus_open_kwargs(open_dataset: Callable[..., xr.Dataset]) -> dict[str, int]:
    """Preflight the installed API and return its supported lazy-chunk option."""
    parameters = inspect.signature(open_dataset).parameters
    required = {
        "dataset_id", "variables", "minimum_longitude", "maximum_longitude",
        "minimum_latitude", "maximum_latitude", "start_datetime", "end_datetime",
    }
    missing = required - set(parameters)
    if missing:
        raise RuntimeError(
            "Installed copernicusmarine.open_dataset is incompatible; missing parameters: "
            f"{sorted(missing)}"
        )
    if "chunk_size_limit" not in parameters:
        raise RuntimeError(
            "Installed copernicusmarine.open_dataset is incompatible: "
            "chunk_size_limit is not supported"
        )
    if "chunks" in parameters:
        LOGGER.debug("API exposes chunks, but this builder intentionally uses chunk_size_limit")
    return {"chunk_size_limit": -1}


def prepare_year(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Normalize, subset, and annotate one yearly checkpoint."""
    normalized = normalize_sst(dataset, find_sst_variable(dataset))
    regional = subset_nino12(normalized)
    assert isinstance(regional, xr.Dataset)
    prepared = regional[["sst"]].sortby("time")
    prepared.attrs.update(
        checkpoint_year=year,
        source_dataset=DATASET_ID,
        source_variable=SOURCE_VARIABLE,
        longitude_bounds="-90,-80",
        latitude_bounds="-10,0",
    )
    return prepared


def validate_year_checkpoint(path: Path, year: int) -> tuple[bool, str]:
    """Validate readability, provenance, coordinates, coverage, units, and dimensions."""
    if not path.exists():
        return False, "file does not exist"
    try:
        with xr.open_dataset(path) as dataset:
            if "sst" not in dataset:
                return False, "missing sst variable"
            if set(("time", "latitude", "longitude")) - set(dataset.coords):
                return False, "missing time/latitude/longitude coordinates"
            if dataset.sst.dims != ("time", "latitude", "longitude"):
                return False, f"unexpected SST dimensions {dataset.sst.dims}"
            if int(dataset.attrs.get("checkpoint_year", -1)) != year:
                return False, "checkpoint year metadata mismatch"
            if dataset.attrs.get("source_variable") != SOURCE_VARIABLE:
                return False, "source variable metadata mismatch"
            if dataset.attrs.get("longitude_bounds") != "-90,-80" or dataset.attrs.get("latitude_bounds") != "-10,0":
                return False, "requested spatial bounds metadata mismatch"
            dates = pd.DatetimeIndex(dataset.time.values)
            if dates.empty or set(dates.year) != {year}:
                return False, "time coordinate does not belong exclusively to requested year"
            if (dates[0].month, dates[0].day) != (1, 1) or (dates[-1].month, dates[-1].day) != (12, 31):
                return False, "checkpoint does not cover January 1 through December 31"
            if dataset.sst.attrs.get("units") != "degrees_Celsius":
                return False, "SST units are not degrees_Celsius"
            if not np.isfinite(dataset.sst.values).any():
                return False, "SST contains no finite values"
    except (OSError, ValueError, TypeError) as exc:
        return False, f"unreadable checkpoint: {exc}"
    return True, "valid"


def _write_checkpoint(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".nc", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        dataset.to_netcdf(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_metadata(
    result: xr.Dataset, start_year: int, end_year: int,
    sampling_half_window_days: int, smoothing_window_days: int,
) -> xr.Dataset:
    reference = f"{start_year}-01-01 to {end_year}-12-31"
    result.attrs.update(
        title="Niño 1+2 smoothed daily SST climatology",
        source_dataset=DATASET_ID,
        source_variable=SOURCE_VARIABLE,
        region="Niño 1+2",
        longitude_bounds="-90,-80",
        latitude_bounds="-10,0",
        reference_start=start_year,
        reference_end=end_year,
        reference_period=reference,
        climatology_method="daily_smoothed",
        sampling_half_window_days=sampling_half_window_days,
        smoothing_window_days=smoothing_window_days,
        created_at=datetime.now(timezone.utc).isoformat(),
        software_version=SOFTWARE_VERSION,
        experimental_product="true",
    )
    total_window = 2 * sampling_half_window_days + 1
    for name in result.data_vars:
        result[name].attrs.update(
            reference_period=reference,
            sampling_window_days=total_window,
            smoothing_window_days=(
                smoothing_window_days if name in {"climatology_mean", "threshold_p10", "threshold_p90"} else 0
            ),
        )
    return result


def build_daily_climatology(
    start_year: int, end_year: int, output: Path, *, checkpoint_dir: Path | None = None,
    resume: bool = False, overwrite: bool = False, sampling_half_window_days: int = 5,
    smoothing_window_days: int = 31,
) -> Path:
    """Build from validated yearly checkpoints and atomically publish the result."""
    if start_year > end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")
    if output.exists() and resume:
        try:
            with xr.open_dataset(output) as existing:
                errors = validate_daily_climatology(existing)
        except (OSError, ValueError):
            errors = ["existing output is unreadable"]
        if not errors:
            LOGGER.info("Valid final output already exists: %s", output)
            return output
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output}")

    import copernicusmarine
    chunk_options = copernicus_open_kwargs(copernicusmarine.open_dataset)
    checkpoints_root = checkpoint_dir or output.parent / "checkpoints_daily"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []
    for year in range(start_year, end_year + 1):
        checkpoint = checkpoints_root / f"ostia_nino12_{year}.nc"
        valid, reason = validate_year_checkpoint(checkpoint, year)
        if resume and valid:
            LOGGER.info("Reusing validated checkpoint %s", checkpoint)
            checkpoints.append(checkpoint)
            continue
        if checkpoint.exists():
            LOGGER.warning("Replacing invalid or non-resumed checkpoint %s: %s", checkpoint, reason)
        source = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID, variables=[SOURCE_VARIABLE],
            minimum_longitude=LONGITUDE_BOUNDS[0], maximum_longitude=LONGITUDE_BOUNDS[1],
            minimum_latitude=LATITUDE_BOUNDS[0], maximum_latitude=LATITUDE_BOUNDS[1],
            start_datetime=f"{year}-01-01T00:00:00", end_datetime=f"{year}-12-31T23:59:59",
            **chunk_options,
        )
        try:
            prepared = prepare_year(source, year)
            _write_checkpoint(prepared, checkpoint)
        finally:
            source.close()
        valid, reason = validate_year_checkpoint(checkpoint, year)
        if not valid:
            raise RuntimeError(f"Yearly checkpoint validation failed for {year}: {reason}")
        checkpoints.append(checkpoint)

    samples = xr.open_mfdataset(checkpoints, combine="by_coords", chunks={"time": 366})
    try:
        # Exact quantiles require one pooled time chunk. The ~200x200 spatial
        # subset retains its automatic spatial layout; no needless spatial rechunk is applied.
        result = build_daily_statistics(
            samples.sst.chunk({"time": -1}),
            sampling_half_window_days=sampling_half_window_days,
            smoothing_window_days=smoothing_window_days,
            calculate_percentiles=True,
        )
        result = _finalize_metadata(
            result, start_year, end_year, sampling_half_window_days, smoothing_window_days
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".nc", dir=output.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            result.to_netcdf(temporary)
            with xr.open_dataset(temporary) as candidate:
                errors = validate_daily_climatology(candidate)
            if errors:
                raise RuntimeError("Final daily climatology validation failed: " + "; ".join(errors))
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    except (ValueError, NotImplementedError) as exc:
        raise RuntimeError(
            "Exact P10/P90 calculation failed; percentile_calculation_state=not_calculated. "
            "No approximation or partial output was published."
        ) from exc
    finally:
        samples.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--output", type=Path, default=Path("data/climatology/nino12_daily_climatology_1991_2020.nc"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/climatology/checkpoints_daily"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sampling-half-window-days", type=int, default=5)
    parser.add_argument("--smoothing-window-days", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    print(build_daily_climatology(
        args.start_year, args.end_year, args.output, checkpoint_dir=args.checkpoint_dir,
        resume=args.resume, overwrite=args.overwrite,
        sampling_half_window_days=args.sampling_half_window_days,
        smoothing_window_days=args.smoothing_window_days,
    ))
