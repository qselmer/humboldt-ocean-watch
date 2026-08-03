from types import SimpleNamespace

import numpy as np
import xarray as xr

from src.copernicus_sst_source import run_source_preflight
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def _catalogue(*, dataset=True, variable=True, temporal=("1981-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
               longitude=(-180.0, 180.0), latitude=(-90.0, 90.0)):
    variables = []
    if variable:
        variables.append({
            "short_name": "analysed_sst", "units": "kelvin",
            "coordinates": [
                {"coordinate_id": "time", "minimum_value": temporal[0], "maximum_value": temporal[1]},
                {"coordinate_id": "latitude", "minimum_value": latitude[0], "maximum_value": latitude[1], "step": 0.05},
                {"coordinate_id": "longitude", "minimum_value": longitude[0], "maximum_value": longitude[1], "step": 0.05},
            ],
        })
    datasets = [{"dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST", "versions": [
        {"label": "202601", "parts": [{"services": [{"variables": variables}]}]}
    ]}] if dataset else []
    return {"products": [{"product_id": "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011", "datasets": datasets}]}


class FakeToolbox:
    __version__ = "2.4.1"

    @staticmethod
    def describe(dataset_id, show_all_versions=False, raise_on_error=False, disable_progress_bar=False):
        return _catalogue()

    @staticmethod
    def subset(dataset_id, variables, minimum_longitude, maximum_longitude,
               minimum_latitude, maximum_latitude, start_datetime, end_datetime,
               output_filename, output_directory, file_format, netcdf_compression_level,
               overwrite=False, disable_progress_bar=False):
        raise AssertionError("preflight must not call subset")

    @staticmethod
    def open_dataset(dataset_id, variables, minimum_longitude, maximum_longitude,
                     minimum_latitude, maximum_latitude, start_datetime, end_datetime,
                     chunk_size_limit=-1):
        return xr.Dataset(
            {"analysed_sst": (("time", "latitude", "longitude"), np.array([[[293.15]]]))},
            coords={"time": ["1991-01-01"], "latitude": [-5.0], "longitude": [-85.0]},
        )


def _domains(config):
    specs = load_sst_domain_specs(config).domains
    return {key: specs[key].bounds for key in ("pacific_context", "humboldt_coastal")}


def test_offline_preflight_checks_api_without_authentication_or_network() -> None:
    config = load_config()
    result = run_source_preflight(config, _domains(config), offline=True, module=FakeToolbox)
    assert result.status == "valid"
    assert result.api_signatures_valid is True
    assert result.authentication_checked is False
    assert result.metadata is None


def test_online_preflight_confirms_catalogue_and_minimal_authenticated_read() -> None:
    config = load_config()
    result = run_source_preflight(
        config, _domains(config), offline=False, module=FakeToolbox, catalogue=_catalogue()
    )
    assert result.status == "valid"
    assert result.authentication_checked is True
    assert result.metadata.dataset_version == "202601"
    assert result.metadata.longitude_resolution_degrees == 0.05


def test_online_preflight_accepts_toolbox_epoch_millisecond_time_coordinates() -> None:
    config = load_config()
    catalogue = _catalogue(temporal=("370742400000.0", "1774915200000.0"))
    result = run_source_preflight(
        config, _domains(config), offline=False, module=FakeToolbox, catalogue=catalogue
    )
    assert result.status == "valid"
    assert result.authentication_checked is True


def test_preflight_classifies_dataset_variable_time_and_space_failures() -> None:
    config = load_config()
    domains = _domains(config)
    cases = [
        (_catalogue(dataset=False), "dataset_not_found"),
        (_catalogue(variable=False), "variable_not_found"),
        (_catalogue(temporal=("2000-01-01T00:00:00Z", "2020-12-31T23:59:59Z")), "temporal_coverage_insufficient"),
        (_catalogue(longitude=(-100.0, -60.0)), "spatial_coverage_insufficient"),
    ]
    for catalogue, expected in cases:
        result = run_source_preflight(
            config, domains, offline=False, module=FakeToolbox, catalogue=catalogue
        )
        assert result.status == expected
        assert result.authentication_checked is False


def test_incompatible_api_and_sanitized_remote_error() -> None:
    config = load_config()
    domains = _domains(config)
    incompatible = SimpleNamespace(__version__="0", describe=lambda: None)
    assert run_source_preflight(config, domains, offline=True, module=incompatible).status == "api_incompatible"

    class Broken(FakeToolbox):
        @staticmethod
        def open_dataset(dataset_id, variables, minimum_longitude, maximum_longitude,
                         minimum_latitude, maximum_latitude, start_datetime, end_datetime,
                         chunk_size_limit=-1):
            raise RuntimeError("password=never-print-this C:/private/credentials")

    result = run_source_preflight(
        config, domains, offline=False, module=Broken, catalogue=_catalogue()
    )
    assert result.status == "network_error"
    assert "never-print-this" not in result.message
    assert "credentials" not in result.message.lower()
