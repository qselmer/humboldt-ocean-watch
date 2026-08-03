import sys

import pytest

from scripts import build_real_multidomain_climatology as cli
from src.climatology_build_plan import build_climatology_plan
from src.copernicus_sst_source import CopernicusSSTAdapter
from src.real_climatology_builder import assemble_domain_climatology
from src.utils import load_config


def test_dry_run_is_offline_and_succeeds(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["builder", "--all", "--dry-run"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert "METOFFICE-GLO-SST-L4-REP-OBS-SST" in output


def test_full_build_requires_execute_and_separate_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["builder", "--all"])
    with pytest.raises(SystemExit, match="explicitly pass --execute"):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["builder", "--all", "--execute"])
    with pytest.raises(SystemExit, match="--confirm-full-build"):
        cli.main()


def test_adapter_rejects_network_in_metadata_and_dry_modes() -> None:
    adapter = CopernicusSSTAdapter(object())
    with pytest.raises(ValueError, match="pilot or full"):
        adapter.subset_to_netcdf(
            dataset_id="dataset", variable="sst", bounds=None,
            start_datetime="1991-01-01", end_datetime="1991-01-02",
            target=None, mode="dry_run", compression_level=4,
        )


def test_assembly_rejects_missing_or_duplicate_tiles(tmp_path) -> None:
    config = load_config()
    domain = build_climatology_plan(config, ("humboldt_coastal",)).domains[0]
    with pytest.raises(ValueError, match="one unique"):
        assemble_domain_climatology([], tmp_path / "pilot.nc", domain, config, pilot=True, source_version="v1")
    repeated = [tmp_path / "same.nc"] * domain.number_of_tiles
    with pytest.raises(ValueError, match="one unique"):
        assemble_domain_climatology(repeated, tmp_path / "pilot.nc", domain, config, pilot=True, source_version="v1")
