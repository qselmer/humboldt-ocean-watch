"""Canonical, deterministic registry of numerical scientific facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from src.brief_schema import CONTEXT_STATUSES, CONTROLLED_UNITS, metric_reference
from src.export_utils import to_json_compatible


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def validate_repository_relative_path(value: str) -> None:
    path = str(value)
    if not path or Path(path).is_absolute() or _WINDOWS_ABSOLUTE.match(path):
        raise ValueError(f"Fact source path must be repository-relative: {value!r}")
    if ".." in PurePosixPath(path.replace("\\", "/")).parts:
        raise ValueError(f"Fact source path cannot escape the repository: {value!r}")


@dataclass(frozen=True)
class BriefFact:
    fact_id: str
    section: str
    name: str
    value: Any
    unit: str
    status: str
    source_id: str
    source_path: str
    source_column: str | None
    selected_date: str
    calculation_method: str
    precision: int
    allowed_in_brief: bool
    reason_when_unavailable: str | None

    def __post_init__(self) -> None:
        if not self.fact_id or "." not in self.fact_id:
            raise ValueError("fact_id must be a stable dotted identifier")
        if self.unit not in CONTROLLED_UNITS:
            raise ValueError(f"Unsupported controlled unit: {self.unit!r}")
        if self.status not in CONTEXT_STATUSES:
            raise ValueError(f"Unsupported fact status: {self.status!r}")
        if self.precision < 0:
            raise ValueError("Fact precision cannot be negative")
        validate_repository_relative_path(self.source_path)
        if self.value is None and self.status == "valid":
            raise ValueError("A valid fact cannot have a null value")
        if self.value is not None and isinstance(self.value, (float, np.floating)):
            if not math.isfinite(float(self.value)):
                raise ValueError("Fact values cannot contain NaN or Infinity")

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


class FactRegistry:
    """Accumulate unique facts and emit a stable order."""

    def __init__(self) -> None:
        self._facts: dict[str, BriefFact] = {}

    def add(
        self,
        *,
        fact_id: str,
        section: str,
        name: str,
        value: Any,
        unit: str,
        source_id: str,
        source_path: str,
        source_column: str | None,
        selected_date: str,
        calculation_method: str,
        precision: int = 3,
        status: str = "valid",
        allowed_in_brief: bool = True,
        reason_when_unavailable: str | None = None,
    ) -> dict[str, Any]:
        if fact_id in self._facts:
            raise ValueError(f"Duplicate fact_id: {fact_id}")
        native = to_json_compatible(value)
        if native is None:
            status = "unavailable" if status == "valid" else status
            reason_when_unavailable = reason_when_unavailable or "source value is missing or non-finite"
        fact = BriefFact(
            fact_id=fact_id,
            section=section,
            name=name,
            value=native,
            unit=unit,
            status=status,
            source_id=source_id,
            source_path=source_path.replace("\\", "/"),
            source_column=source_column,
            selected_date=selected_date,
            calculation_method=calculation_method,
            precision=precision,
            allowed_in_brief=allowed_in_brief,
            reason_when_unavailable=reason_when_unavailable,
        )
        self._facts[fact_id] = fact
        return metric_reference(fact.fact_id, fact.value, fact.unit, fact.status)

    def records(self) -> list[dict[str, Any]]:
        return [self._facts[key].to_dict() for key in sorted(self._facts)]

    def __len__(self) -> int:
        return len(self._facts)
