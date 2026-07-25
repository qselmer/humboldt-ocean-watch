"""Fact-grounded validation of numerical tokens in scientific prose."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Any, Iterable

from src.brief_output_schema import NARRATIVE_SECTION_IDS


_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*%)?")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_IDENTIFIER_PATTERNS = (
    re.compile(r"\b(?:TRK|FAM|E)\d+\b", re.IGNORECASE),
    re.compile(r"\bP(?:10|25|75|90)\b", re.IGNORECASE),
    re.compile(r"\b(?:patch|parche|track|family|familia)\s+(?:ID\s*)?\d+\b", re.IGNORECASE),
    re.compile(r"Niño\s+1\s*\+\s*2", re.IGNORECASE),
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_LOCALIZED_DATE_RE = re.compile(
    r"\b(?:(\d{1,2})\s+de\s+([A-Za-záéíóúñ]+)\s+de\s+(\d{4})|"
    r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})|"
    r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4}))\b",
    re.IGNORECASE,
)
_REFERENCE_RANGE_RE = re.compile(r"\b(\d{4})\s*[–—-]\s*(\d{4})\b")
_WINDOW_RE = re.compile(r"\b(\d+)\s*[- ]?(?:day|days|día|días)\b", re.IGNORECASE)


@dataclass(frozen=True)
class NarrativeRecord:
    path: str
    text: str
    supporting_fact_ids: tuple[str, ...]
    quality_flag_ids: tuple[str, ...]
    claim_types: tuple[str, ...]


def iter_narrative_records(payload: dict[str, Any]) -> Iterable[NarrativeRecord]:
    for section_id in NARRATIVE_SECTION_IDS:
        section = payload.get(section_id, {})
        yield NarrativeRecord(
            f"{section_id}.heading", str(section.get("heading", "")), (), (), ("methodological_note",)
        )
        if section.get("reason"):
            yield NarrativeRecord(
                f"{section_id}.reason", str(section["reason"]), (), (), ("limitation",)
            )
        for index, paragraph in enumerate(section.get("paragraphs", [])):
            yield NarrativeRecord(
                path=f"{section_id}.paragraphs[{index}]",
                text=str(paragraph.get("text", "")),
                supporting_fact_ids=tuple(map(str, paragraph.get("supporting_fact_ids", []))),
                quality_flag_ids=tuple(map(str, paragraph.get("quality_flag_ids", []))),
                claim_types=tuple(map(str, paragraph.get("claim_types", []))),
            )
    for index, message in enumerate(payload.get("key_messages", [])):
        yield NarrativeRecord(
            path=f"key_messages[{index}]",
            text=str(message.get("text", "")),
            supporting_fact_ids=tuple(map(str, message.get("supporting_fact_ids", []))),
            quality_flag_ids=tuple(map(str, message.get("quality_flag_ids", []))),
            claim_types=tuple(map(str, message.get("claim_types", []))),
        )
    for index, limitation in enumerate(payload.get("limitations", [])):
        yield NarrativeRecord(
            path=f"limitations[{index}]",
            text=str(limitation.get("text", "")),
            supporting_fact_ids=tuple(map(str, limitation.get("supporting_fact_ids", []))),
            quality_flag_ids=tuple(map(str, limitation.get("quality_flag_ids", []))),
            claim_types=("limitation",),
        )
    yield NarrativeRecord("title", str(payload.get("title", "")), (), (), ("methodological_note",))
    yield NarrativeRecord("disclaimer", str(payload.get("disclaimer", "")), (), (), ("limitation",))
    for index, note in enumerate(payload.get("generation_notes", [])):
        yield NarrativeRecord(f"generation_notes[{index}]", str(note), (), (), ("methodological_note",))


def _all_context_dates(context: dict[str, Any]) -> set[str]:
    dates: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            dates.update(_ISO_DATE_RE.findall(value))

    walk(context)
    return dates


def _allowed_spans(text: str, context: dict[str, Any]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    allowed_dates = _all_context_dates(context)
    for match in _ISO_DATE_RE.finditer(text):
        if match.group(0) in allowed_dates:
            spans.append(match.span())
    for pattern in _IDENTIFIER_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))

    reference_start = context.get("climatology", {}).get("reference_start")
    reference_end = context.get("climatology", {}).get("reference_end")
    for match in _REFERENCE_RANGE_RE.finditer(text):
        if (int(match.group(1)), int(match.group(2))) == (reference_start, reference_end):
            spans.append(match.span())

    allowed_windows = {
        int(item.get("window_days"))
        for item in context.get("recent_evolution", {}).get("windows", [])
        if item.get("window_days") is not None
    }
    for match in _WINDOW_RE.finditer(text):
        if int(match.group(1)) in allowed_windows:
            spans.append(match.span())

    for match in _LOCALIZED_DATE_RE.finditer(text):
        groups = match.groups()
        if groups[0]:
            day_value, month_name, year_value = int(groups[0]), groups[1], int(groups[2])
        elif groups[3]:
            day_value, month_name, year_value = int(groups[3]), groups[4], int(groups[5])
        else:
            day_value, month_name, year_value = int(groups[7]), groups[6], int(groups[8])
        month_value = _MONTHS.get(month_name.casefold())
        try:
            iso = date(year_value, int(month_value), day_value).isoformat() if month_value else None
        except ValueError:
            iso = None
        if iso in allowed_dates:
            spans.append(match.span())
    return spans


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _decimals(token: str) -> int:
    clean = token.strip().rstrip("%").strip().replace(",", ".")
    return len(clean.split(".", 1)[1]) if "." in clean else 0


def _explicit_unit(text: str, match: re.Match[str]) -> str | None:
    token = match.group(0)
    around = text[max(0, match.start() - 8): min(len(text), match.end() + 24)].casefold()
    compact = re.sub(r"\s+", " ", around)
    if "%" in token:
        return "percent"
    day_power = r"(?:/\s*(?:day|día)|(?:·|x|\*)?\s*(?:per\s+)?(?:day|día)(?:\s*(?:\^-?1|[-−⁻]1|⁻¹))?)"
    if re.search(rf"°c\s*(?:·|x|\*)?\s*km(?:²|2)\s*{day_power}", compact):
        return "degC_km2_day"
    if re.search(rf"°c\s*{day_power}", compact):
        return "degC_day"
    if "°c" in around:
        return "degC"
    if re.search(rf"km(?:²|2)\s*{day_power}", compact):
        return "km2_day"
    if "km²" in around or "km2" in around:
        return "km2"
    if re.search(rf"km\s*{day_power}", compact):
        return "km_per_day"
    if "km" in around:
        return "km"
    if re.search(r"\b(?:day|days|día|días)\b", around):
        return "day"
    if "°s" in around or "latitude" in around or "latitud" in around:
        return "degree_latitude"
    if "°w" in around or "longitude" in around or "longitud" in around:
        return "degree_longitude"
    return None


def _unit_compatible(explicit: str | None, fact_unit: str) -> bool:
    if explicit is None:
        return True
    groups = {
        "percent": {"fraction", "fraction_change", "percent"},
        "degC": {"degC"},
        "degC_day": {"degC_day", "slope_degC_per_day"},
        "degC_km2_day": {"degC_km2_day"},
        "km2": {"km2"},
        "km2_day": {"km2_day"},
        "km": {"km"},
        "km_per_day": {"km_per_day"},
        "day": {"day"},
        "degree_latitude": {"degree_latitude"},
        "degree_longitude": {"degree_longitude"},
    }
    return fact_unit in groups[explicit]


def _matches_fact(number: float, token: str, explicit_unit: str | None, fact: dict[str, Any]) -> bool:
    if fact.get("status") != "valid" or not fact.get("allowed_in_brief", False):
        return False
    value = fact.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return False
    unit = str(fact.get("unit"))
    if not _unit_compatible(explicit_unit, unit):
        return False
    precision = int(fact.get("precision", 0))
    decimals = _decimals(token)
    if decimals > precision:
        return False
    candidates = [float(value)]
    if explicit_unit == "percent" and unit in {"fraction", "fraction_change"}:
        if str(fact.get("name")) == "evidence_score" or "evidence_score" in str(fact.get("fact_id")):
            return False
        candidates = [float(value) * 100.0]
    elif explicit_unit == "percent":
        candidates = [float(value)]
    elif unit in {"fraction", "fraction_change"} and decimals == 0 and abs(value) < 1:
        return False
    return any(math.isclose(number, round(candidate, decimals), abs_tol=10 ** (-(decimals + 2))) for candidate in candidates)


def validate_numeric_claims(payload: dict[str, Any], context: dict[str, Any]) -> list[dict[str, str]]:
    facts = {str(fact.get("fact_id")): fact for fact in context.get("fact_registry", [])}
    suppressed = {
        str(fact_id)
        for flag in context.get("quality_flags", [])
        if flag.get("suppress_dependent_claims")
        for fact_id in flag.get("affected_fact_ids", [])
    }
    issues: list[dict[str, str]] = []
    for record in iter_narrative_records(payload):
        allowed_spans = _allowed_spans(record.text, context)
        supporting = [facts[fact_id] for fact_id in record.supporting_fact_ids if fact_id in facts and fact_id not in suppressed]
        for match in _NUMBER_RE.finditer(record.text):
            if _inside(match.start(), allowed_spans):
                continue
            token = match.group(0)
            try:
                number = float(token.strip().rstrip("%").strip().replace(",", "."))
            except ValueError:
                continue
            explicit_unit = _explicit_unit(record.text, match)
            if not any(_matches_fact(number, token, explicit_unit, fact) for fact in supporting):
                issues.append(
                    {
                        "code": "unsupported_numeric_claim",
                        "path": record.path,
                        "message": f"Numeric token {token!r} is not supported by a cited valid fact with compatible units",
                    }
                )
    return issues
