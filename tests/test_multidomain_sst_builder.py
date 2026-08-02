from copy import deepcopy

import pytest

from scripts.build_multidomain_sst_snapshot import (
    _promote_atomically,
    build_multidomain_snapshot,
)
from src.utils import load_config


def test_dry_run_writes_nothing(tmp_path) -> None:
    config = deepcopy(load_config())
    report = build_multidomain_snapshot(
        config,
        allow_demo=True,
        output_directory=tmp_path,
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["network_access"] is False
    assert not list(tmp_path.rglob("*"))


def test_staging_failure_preserves_existing_valid_product(tmp_path) -> None:
    target = tmp_path / "status.json"
    target.write_text("valid", encoding="utf-8")

    def good(path):
        path.write_text("replacement", encoding="utf-8")

    def fail(path):
        raise RuntimeError("planned failure")

    with pytest.raises(RuntimeError, match="planned failure"):
        _promote_atomically(
            [(target, good), (tmp_path / "snapshot.json", fail)], overwrite=True
        )
    assert target.read_text(encoding="utf-8") == "valid"
