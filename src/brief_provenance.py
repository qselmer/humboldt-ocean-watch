"""Offline product loading and repository-relative provenance records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


DATE_COLUMNS = (
    "date",
    "analysis_end_date",
    "start_date",
    "predecessor_date",
)


def repository_relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source product is outside the repository: {path}") from exc
    return relative.as_posix()


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    required: bool
    required_columns: frozenset[str] = frozenset()


@dataclass
class LoadedProduct:
    spec: SourceSpec
    relative_path: str
    available: bool
    value: pd.DataFrame | dict[str, Any] | None = None
    reason: str | None = None
    validation_status: str = "unavailable"
    validation_messages: list[str] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        return self.value if isinstance(self.value, pd.DataFrame) else pd.DataFrame()

    @property
    def document(self) -> dict[str, Any]:
        return self.value if isinstance(self.value, dict) else {}

    def selected_rows(self, selected_date: pd.Timestamp) -> pd.DataFrame:
        frame = self.frame
        if frame.empty:
            return frame.copy()
        for column in ("date", "analysis_end_date"):
            if column in frame:
                values = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
                return frame.loc[values == selected_date.normalize()].copy()
        return frame.iloc[0:0].copy()

    def provenance_record(self, selected_date: pd.Timestamp) -> dict[str, Any]:
        path = self.spec.path
        modified = None
        if path.exists():
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        frame = self.frame
        date_start = date_end = None
        for column in DATE_COLUMNS:
            if column in frame:
                dates = pd.to_datetime(frame[column], errors="coerce").dropna()
                if not dates.empty:
                    date_start = dates.min().date().isoformat()
                    date_end = dates.max().date().isoformat()
                break
        return {
            "source_id": self.spec.source_id,
            "path": self.relative_path,
            "required": self.spec.required,
            "available": self.available,
            "file_modified_time": modified,
            "row_count": int(len(frame)) if isinstance(self.value, pd.DataFrame) else None,
            "dimensions": ({"rows": int(frame.shape[0]), "columns": int(frame.shape[1])} if isinstance(self.value, pd.DataFrame) else None),
            "date_start": date_start,
            "date_end": date_end,
            "selected_date_rows": int(len(self.selected_rows(selected_date))),
            "schema_columns_used": sorted(self.spec.required_columns),
            "validation_status": self.validation_status,
            "validation_messages": list(self.validation_messages),
        }


def load_source(spec: SourceSpec, root: Path) -> LoadedProduct:
    """Read one local Parquet or JSON product without external access."""
    relative = repository_relative(spec.path, root)
    if not spec.path.exists():
        reason = f"cached product not found: {relative}"
        if spec.required:
            raise FileNotFoundError(reason)
        return LoadedProduct(spec, relative, False, reason=reason)
    try:
        if spec.path.suffix.lower() == ".parquet":
            value: pd.DataFrame | dict[str, Any] = pd.read_parquet(spec.path)
            missing = sorted(spec.required_columns - set(value.columns))
            if missing:
                raise ValueError(f"missing required columns: {missing}")
        elif spec.path.suffix.lower() == ".json":
            parsed = json.loads(spec.path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("JSON product must contain an object")
            value = parsed
        else:
            raise ValueError(f"unsupported cached product type: {spec.path.suffix}")
    except Exception as exc:
        reason = f"could not read {relative}: {exc}"
        if spec.required:
            raise ValueError(reason) from exc
        return LoadedProduct(
            spec, relative, False, reason=reason,
            validation_status="invalid", validation_messages=[reason],
        )
    return LoadedProduct(spec, relative, True, value=value, validation_status="valid")


def is_stale(derived: LoadedProduct, upstream: LoadedProduct) -> bool:
    if not derived.available or not upstream.available:
        return False
    return derived.spec.path.stat().st_mtime < upstream.spec.path.stat().st_mtime
