import json

import numpy as np
import pytest

from src.brief_fact_registry import FactRegistry
from src.export_utils import dumps_json_safe


def _add(registry: FactRegistry, fact_id: str, value=1.2345, **overrides):
    values = {
        "fact_id": fact_id,
        "section": "regional_state",
        "name": fact_id.split(".")[-1],
        "value": value,
        "unit": "degC",
        "source_id": "daily_metrics",
        "source_path": "data/processed/metrics.parquet",
        "source_column": "value",
        "selected_date": "2026-01-02",
        "calculation_method": "cached_metric",
        "precision": 3,
    }
    values.update(overrides)
    return registry.add(**values)


def test_registry_is_unique_and_deterministically_sorted() -> None:
    registry = FactRegistry()
    _add(registry, "regional.z")
    _add(registry, "regional.a")
    assert [row["fact_id"] for row in registry.records()] == ["regional.a", "regional.z"]
    with pytest.raises(ValueError, match="Duplicate fact_id"):
        _add(registry, "regional.a")


def test_unavailable_nonfinite_fact_becomes_null() -> None:
    registry = FactRegistry()
    reference = _add(registry, "regional.missing", np.nan)
    fact = registry.records()[0]
    assert reference["value"] is None
    assert fact["status"] == "unavailable"
    assert fact["reason_when_unavailable"]
    assert json.loads(dumps_json_safe(registry.records()))[0]["value"] is None


def test_precision_allowed_policy_and_provenance_are_retained() -> None:
    registry = FactRegistry()
    _add(registry, "regional.mean", precision=4, allowed_in_brief=False)
    fact = registry.records()[0]
    assert fact["precision"] == 4
    assert fact["allowed_in_brief"] is False
    assert fact["source_column"] == "value"


def test_controlled_units_and_relative_paths_are_enforced() -> None:
    registry = FactRegistry()
    with pytest.raises(ValueError, match="controlled unit"):
        _add(registry, "regional.bad_unit", unit="degrees C")
    with pytest.raises(ValueError, match="repository-relative"):
        _add(registry, "regional.absolute", source_path="C:/Users/person/metrics.parquet")
