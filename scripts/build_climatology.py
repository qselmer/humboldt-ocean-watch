"""Build resumable 1991–2020 monthly OSTIA climatology year by year."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import find_sst_variable, normalize_sst, subset_nino12

DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
LOGGER = logging.getLogger(__name__)


def yearly_accumulator(dataset: xr.Dataset) -> xr.Dataset:
    """Reduce one year's daily SST to monthly sum, sum of squares, and count."""
    normalized = normalize_sst(dataset, find_sst_variable(dataset))
    regional = subset_nino12(normalized)
    assert isinstance(regional, xr.Dataset)
    sst = regional.sst
    return xr.Dataset({
        "sst_sum": sst.groupby("time.month").sum("time", skipna=True),
        "sst_sum_of_squares": (sst**2).groupby("time.month").sum("time", skipna=True),
        "valid_observation_count": sst.groupby("time.month").count("time"),
    })


def finalize_accumulator(accumulator: xr.Dataset) -> xr.Dataset:
    """Convert sufficient statistics into mean, population std, and count."""
    count = accumulator.valid_observation_count
    mean = (accumulator.sst_sum / count).where(count > 0)
    variance = (accumulator.sst_sum_of_squares / count - mean**2).clip(min=0).where(count > 0)
    return xr.Dataset(
        {
            "climatological_mean": mean.assign_attrs(units="degrees_Celsius"),
            "climatological_standard_deviation": np.sqrt(variance).assign_attrs(units="degrees_Celsius"),
            "valid_observation_count": count.astype("int64"),
        },
        attrs={
            "dataset_id": DATASET_ID,
            "reference_period": "1991-01-01 to 2020-12-31",
            "region": "Niño 1+2: longitude -90 to -80, latitude -10 to 0",
        },
    )


def build_climatology(start_year: int, end_year: int, output: Path) -> Path:
    """Download/reuse yearly checkpoints and write the combined climatology."""
    if start_year > end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")
    import copernicusmarine

    checkpoint_dir = output.parent / f".{output.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    yearly: list[xr.Dataset] = []
    for year in range(start_year, end_year + 1):
        checkpoint = checkpoint_dir / f"{year}.nc"
        if checkpoint.exists():
            LOGGER.info("Resuming from checkpoint %s", checkpoint)
            with xr.open_dataset(checkpoint) as opened:
                yearly.append(opened.load())
            continue
        LOGGER.info("Opening OSTIA reprocessed SST for %s", year)
        source = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            variables=["analysed_sst"],
            minimum_longitude=-90.0,
            maximum_longitude=-80.0,
            minimum_latitude=-10.0,
            maximum_latitude=0.0,
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
        )
        statistics = yearly_accumulator(source).load()
        source.close()
        statistics.to_netcdf(checkpoint)
        yearly.append(statistics)
    accumulator = yearly[0]
    for statistics in yearly[1:]:
        accumulator = accumulator.fillna(0) + statistics.fillna(0)
    result = finalize_accumulator(accumulator)
    result.attrs["reference_period"] = f"{start_year}-01-01 to {end_year}-12-31"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/climatology/nino12_monthly_climatology_1991_2020.nc"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = parse_args()
    print(build_climatology(options.start_year, options.end_year, options.output))
