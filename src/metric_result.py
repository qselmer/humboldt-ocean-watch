"""Standard result model for recoverable analytics outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.export_utils import to_json_compatible

MetricStatus = Literal["valid", "warning", "not_calculated", "invalid"]
ALLOWED_STATUSES = frozenset({"valid", "warning", "not_calculated", "invalid"})


@dataclass(frozen=True)
class MetricResult:
    """A numerical metric plus its calculation and quality context."""

    metric: str
    value: float = np.nan
    status: MetricStatus = "valid"
    reason: str | None = None
    n_observations: int = 0
    valid_coverage: float = np.nan
    unit: str = "1"
    family: str = "unspecified"
    window_days: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("MetricResult.metric must be a non-empty string")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported metric status: {self.status!r}")
        if self.n_observations < 0:
            raise ValueError("MetricResult.n_observations cannot be negative")
        coverage = float(self.valid_coverage)
        if np.isfinite(coverage) and not 0.0 <= coverage <= 1.0:
            raise ValueError("MetricResult.valid_coverage must be between 0 and 1")
        if self.window_days is not None and self.window_days < 1:
            raise ValueError("MetricResult.window_days must be positive when supplied")
        if not isinstance(self.metadata, dict):
            raise TypeError("MetricResult.metadata must be a dictionary")
        if self.value is None:
            object.__setattr__(self, "value", np.nan)

    def to_dict(self) -> dict[str, Any]:
        """Return native fields, retaining NaN internally for undefined values."""
        return {
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "reason": self.reason,
            "n_observations": int(self.n_observations),
            "valid_coverage": self.valid_coverage,
            "unit": self.unit,
            "family": self.family,
            "window_days": self.window_days,
            "metadata": self.metadata,
        }

    def to_json_compatible(self) -> dict[str, Any]:
        """Return a structure accepted by the project's strict JSON exporter."""
        converted = to_json_compatible(self.to_dict())
        assert isinstance(converted, dict)
        return converted


def not_calculated(
    metric: str,
    reason: str,
    *,
    unit: str,
    family: str,
    n_observations: int = 0,
    valid_coverage: float = np.nan,
    window_days: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MetricResult:
    """Create a standard undefined result without a sentinel value."""
    return MetricResult(
        metric=metric,
        value=np.nan,
        status="not_calculated",
        reason=reason,
        n_observations=n_observations,
        valid_coverage=valid_coverage,
        unit=unit,
        family=family,
        window_days=window_days,
        metadata=metadata or {},
    )
