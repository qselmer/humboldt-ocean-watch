"""Deterministic fact, claim, quality, and cross-language validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import re
import unicodedata
from typing import Any, Mapping

from src.brief_disclaimer import get_mandatory_disclaimer
from src.brief_fact_coverage import (
    REQUIRED_REPRESENTATION,
    claim_support_locations,
    derive_mandatory_fact_coverage_plan,
)
from src.brief_language_validation import brief_human_text, validate_brief_language
from src.brief_numeric_validation import iter_narrative_records, validate_numeric_claims
from src.brief_output_schema import NARRATIVE_SECTION_IDS, validate_output_schema
from src.export_utils import dumps_json_safe, to_json_compatible


FINAL_VALIDATION_STATUSES = {"passed", "passed_with_warnings", "failed"}


@dataclass(frozen=True)
class BriefValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class OperationalFactCoverageIssue:
    code: str
    path: str
    message: str
    fact_id: str
    available: bool
    allowed_in_brief: bool
    value_available: bool
    authoritative_value_available: bool
    valid_zero: bool
    required_representation: str
    claim_support_field: str
    suggested_sections: list[str]
    present_in_facts_used: bool
    present_in_claim_support: bool
    claim_support_locations_inspected: list[str]
    present_in_each_inspected_location: dict[str, bool]
    derived_facts_used_membership: bool
    repair_instruction_generated: bool


@dataclass
class ScientificBriefValidationReport:
    final_status: str
    language: str
    analysis_date: str | None
    context_sha256: str
    checks: dict[str, bool]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def context_sha256(context: Mapping[str, Any]) -> str:
    canonical = dumps_json_safe(context, indent=None)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _words(text: str) -> int:
    return len(re.findall(r"\b[^\W_]+(?:[-’'][^\W_]+)*\b", text, flags=re.UNICODE))


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _negated(text: str, start: int, end: int) -> bool:
    around = _normalize(text[max(0, start - 35): min(len(text), end + 15)])
    return bool(re.search(r"\b(no|not|never|without|sin|ningun[oa]?|does not|do not|no se)\b", around))


_DEFAULT_PROHIBITED_PATTERNS: dict[str, tuple[str, ...]] = {
    "official_classification": (
        r"clasificacion oficial", r"official classification", r"el nino costero (?:moderado|fuerte|extremo)",
        r"coastal el nino (?:moderate|strong|extreme)",
    ),
    "forecast": (
        r"\bforecast", r"\bpredict", r"\bwill\b", r"expected to", r"\bpronostic", r"\bpredec",
        r"se espera que", r"(?:aument|disminu)ira\b",
    ),
    "causal_attribution": (
        r"caused by", r"due to", r"driven by", r"causad[oa] por", r"debido a", r"impulsad[oa] por",
    ),
    "biological_fisheries_impact": (
        r"(?:biological|fishery|fisheries|ecosystem) impact", r"impacto (?:biologico|pesquero|ecosistemico)",
        r"fish (?:mortality|catch|landings)", r"mortalidad|capturas|desembarques",
    ),
    "economic_social_impact": (
        r"economic impact", r"social impact", r"impacto economico", r"impacto social",
    ),
    "confidence_probability": (
        r"confidence probability", r"probability of", r"probabilidad de", r"confianza de",
    ),
    "track_water_parcel": (
        r"tracks? (?:are|represent) (?:individual )?(?:water parcels?|water masses?)",
        r"tracks? (?:son|representan) (?:parcelas|masas) (?:individuales )?de agua",
    ),
    "ordinal_representativeness": (
        r"representativeness (?:severity|ranking)", r"more severe representativeness",
        r"severidad de representatividad", r"representatividad mas severa",
    ),
    "external_reference": (
        r"https?://", r"\bdoi\s*:", r"\[[0-9]+\]", r"according to (?:a|the) study",
        r"segun (?:un|el) estudio",
    ),
}


def _claim_issues(payload: dict[str, Any], settings: Mapping[str, Any]) -> list[BriefValidationIssue]:
    text_records = list(iter_narrative_records(payload))
    configured = settings.get("prohibited_claim_patterns", {})
    issues: list[BriefValidationIssue] = []
    for category, defaults in _DEFAULT_PROHIBITED_PATTERNS.items():
        patterns = tuple(configured.get(category, defaults))
        for record in text_records:
            for pattern in patterns:
                normalized = _normalize(record.text)
                for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                    if category != "external_reference" and _negated(normalized, match.start(), match.end()):
                        continue
                    issues.append(
                        BriefValidationIssue(
                            f"prohibited_{category}", record.path,
                            f"Prohibited claim pattern detected ({category})",
                        )
                    )
                    break
                else:
                    continue
                break
    return issues


def _fact_and_quality_issues(
    payload: dict[str, Any],
    context: dict[str, Any],
    settings: Mapping[str, Any],
) -> tuple[
    list[BriefValidationIssue | OperationalFactCoverageIssue],
    list[BriefValidationIssue],
]:
    errors: list[BriefValidationIssue | OperationalFactCoverageIssue] = []
    warnings: list[BriefValidationIssue] = []
    facts = {str(item.get("fact_id")): item for item in context.get("fact_registry", [])}
    flags = {str(item.get("flag_id")): item for item in context.get("quality_flags", [])}
    suppressed = {
        str(fact_id)
        for flag in flags.values()
        if flag.get("suppress_dependent_claims")
        for fact_id in flag.get("affected_fact_ids", [])
    }
    referenced_facts: set[str] = set()
    substantive_locations = claim_support_locations(
        payload, substantive_only=True
    )
    claim_supported_facts = {
        fact_id
        for fact_ids in substantive_locations.values()
        for fact_id in fact_ids
    }
    referenced_flags: set[str] = set()
    for record in iter_narrative_records(payload):
        fact_ids = set(record.supporting_fact_ids)
        flag_ids = set(record.quality_flag_ids)
        referenced_facts.update(fact_ids)
        factual = not (
            set(record.claim_types) <= {"limitation", "quality_caveat", "methodological_note"}
        )
        referenced_flags.update(flag_ids)
        if factual and record.path not in {"title", "disclaimer"} and not fact_ids:
            errors.append(
                BriefValidationIssue(
                    "missing_supporting_fact", record.path,
                    "A factual narrative record must cite at least one supporting fact",
                )
            )
        for fact_id in fact_ids:
            fact = facts.get(fact_id)
            if fact is None:
                errors.append(BriefValidationIssue("unknown_fact_id", record.path, f"Unknown fact ID {fact_id}"))
            elif fact.get("status") != "valid" or fact.get("value") is None:
                errors.append(BriefValidationIssue("unavailable_fact_used", record.path, f"Fact {fact_id} is unavailable"))
            elif not fact.get("allowed_in_brief", False):
                errors.append(BriefValidationIssue("disallowed_fact_used", record.path, f"Fact {fact_id} is not allowed in briefs"))
            elif fact_id in suppressed:
                errors.append(BriefValidationIssue("suppressed_fact_used", record.path, f"Fact {fact_id} is suppressed by quality control"))
        for flag_id in flag_ids:
            if flag_id not in flags:
                errors.append(BriefValidationIssue("unknown_quality_flag", record.path, f"Unknown quality flag {flag_id}"))

    declared_facts = set(map(str, payload.get("facts_used", [])))
    if declared_facts != referenced_facts:
        errors.append(
            BriefValidationIssue(
                "facts_used_mismatch", "facts_used",
                "facts_used must exactly equal the set cited by narrative records",
            )
        )

    coverage_plan = derive_mandatory_fact_coverage_plan(context)
    for entry in coverage_plan.required_facts:
        if entry.fact_id in claim_supported_facts:
            continue
        errors.append(
            OperationalFactCoverageIssue(
                code="required_operational_fact_not_represented",
                path="facts_used",
                message="A required operational fact is not represented in paragraph or key-message claim support",
                fact_id=entry.fact_id,
                available=entry.available,
                allowed_in_brief=entry.allowed_in_brief,
                value_available=entry.value_available,
                authoritative_value_available=entry.value_available,
                valid_zero=entry.valid_zero,
                required_representation=REQUIRED_REPRESENTATION,
                claim_support_field=entry.claim_support_field,
                suggested_sections=list(entry.suggested_sections),
                present_in_facts_used=entry.fact_id in declared_facts,
                present_in_claim_support=entry.fact_id in claim_supported_facts,
                claim_support_locations_inspected=list(substantive_locations),
                present_in_each_inspected_location={
                    path: entry.fact_id in fact_ids
                    for path, fact_ids in substantive_locations.items()
                },
                derived_facts_used_membership=entry.fact_id in referenced_facts,
                repair_instruction_generated=False,
            )
        )

    required_limitations = {
        str(item.get("limitation_id"))
        for item in context.get("limitations", [])
        if item.get("limitation_id")
    }
    output_limitations = {
        str(item.get("limitation_id")) for item in payload.get("limitations", [])
    }
    missing_limitations = sorted(required_limitations - output_limitations)
    if missing_limitations:
        errors.append(
            BriefValidationIssue(
                "required_limitations_missing",
                "limitations",
                "Required context limitations are missing: " + ", ".join(missing_limitations),
            )
        )

    output_section_for_flag = {
        "regional_state": "regional_state",
        "recent_evolution": "recent_evolution",
        "spatial_structure": "spatial_structure",
        "representativeness": "spatial_structure",
        "univariate_event": "event_status",
        "daily_patches": "event_status",
        "spatiotemporal_activity": "event_status",
        "quality": "data_quality",
        "climatology": "data_quality",
    }
    for flag_id, flag in flags.items():
        severity = flag.get("severity")
        if severity in {"warning", "error"} and flag_id not in referenced_flags:
            errors.append(
                BriefValidationIssue(
                    "material_quality_flag_omitted", "data_quality",
                    f"Material quality flag {flag_id} must be represented",
                )
            )
        if flag.get("suppress_dependent_claims"):
            output_section = output_section_for_flag.get(str(flag.get("section")))
            if output_section and payload.get(output_section, {}).get("status") == "available":
                errors.append(
                    BriefValidationIssue(
                        "suppressed_section_not_limited", output_section,
                        f"Section must be limited or unavailable because of {flag_id}",
                    )
                )
            if flag_id not in referenced_flags:
                errors.append(
                    BriefValidationIssue(
                        "suppression_not_disclosed", output_section or "data_quality",
                        f"Suppressed-claim flag {flag_id} must be disclosed",
                    )
                )

    coverage = context.get("quality", {}).get("valid_coverage", {}).get("value")
    if isinstance(coverage, (int, float)) and coverage < float(settings.get("minimum_valid_coverage", 0.8)):
        if payload.get("spatial_structure", {}).get("status") == "available":
            errors.append(
                BriefValidationIssue(
                    "insufficient_coverage_not_limited", "spatial_structure",
                    "Spatial structure must be limited when valid coverage is insufficient",
                )
            )
    if context.get("climatology", {}).get("fallback_used"):
        fallback_flags = {
            key for key, value in flags.items() if value.get("message_code") == "monthly_climatology_fallback"
        }
        if not fallback_flags <= referenced_flags or payload.get("data_quality", {}).get("status") == "available":
            errors.append(
                BriefValidationIssue(
                    "monthly_fallback_not_disclosed", "data_quality",
                    "Monthly climatology fallback must be explicit and data quality must be limited",
                )
            )
        quality_text = _normalize(
            " ".join(
                str(item.get("text", ""))
                for item in payload.get("data_quality", {}).get("paragraphs", [])
            )
        )
        if "monthly" not in quality_text and "mensual" not in quality_text:
            errors.append(
                BriefValidationIssue(
                    "monthly_fallback_not_named",
                    "data_quality",
                    "Monthly climatology fallback must be named explicitly",
                )
            )
    return errors, warnings


def _section_and_word_issues(
    payload: dict[str, Any], settings: Mapping[str, Any]
) -> list[BriefValidationIssue]:
    issues: list[BriefValidationIssue] = []
    for section_id in NARRATIVE_SECTION_IDS:
        section = payload.get(section_id, {})
        paragraphs = section.get("paragraphs", [])
        status = section.get("status")
        reason = section.get("reason")
        if status == "unavailable" and (paragraphs or not reason):
            issues.append(
                BriefValidationIssue(
                    "invalid_unavailable_section", section_id,
                    "An unavailable section must have no paragraphs and a reason",
                )
            )
        if status == "available" and reason not in {None, ""}:
            issues.append(
                BriefValidationIssue(
                    "available_section_has_reason", section_id,
                    "An available section must not carry an unavailability reason",
                )
            )
        if status in {"available", "limited"} and not paragraphs:
            issues.append(
                BriefValidationIssue(
                    "section_has_no_paragraphs", section_id,
                    "An available or limited section must contain narrative content",
                )
            )
        if status == "limited" and not reason:
            issues.append(
                BriefValidationIssue(
                    "limited_section_missing_reason", section_id,
                    "A limited section must state the reason for the limitation",
                )
            )
        if section_id == "executive_summary" and status != "unavailable" and not 1 <= len(paragraphs) <= 3:
            issues.append(
                BriefValidationIssue(
                    "executive_summary_paragraph_count", section_id,
                    "Executive summary must contain one to three paragraphs",
                )
            )
        limit = int(
            settings["executive_summary_max_words"]
            if section_id == "executive_summary"
            else settings["section_max_words"]
        )
        count = sum(_words(str(paragraph.get("text", ""))) for paragraph in paragraphs)
        if count > limit:
            issues.append(
                BriefValidationIssue(
                    "section_word_limit", section_id,
                    f"Section contains {count} words; maximum is {limit}",
                )
            )
    maximum_messages = int(settings["maximum_key_messages"])
    if len(payload.get("key_messages", [])) > maximum_messages:
        issues.append(BriefValidationIssue("too_many_key_messages", "key_messages", "Too many key messages"))
    for index, item in enumerate(payload.get("key_messages", [])):
        count = _words(str(item.get("text", "")))
        if count > int(settings["key_message_max_words"]):
            issues.append(
                BriefValidationIssue(
                    "key_message_word_limit", f"key_messages[{index}]",
                    f"Key message contains {count} words",
                )
            )
    return issues


def validate_scientific_brief(
    payload: Any,
    context: dict[str, Any],
    *,
    language: str,
    settings: Mapping[str, Any],
    generation_metadata: Mapping[str, Any] | None = None,
) -> ScientificBriefValidationReport:
    """Run the complete deterministic validation stack."""
    errors: list[BriefValidationIssue] = []
    warnings: list[BriefValidationIssue] = []
    checks: dict[str, bool] = {}

    schema_issues = validate_output_schema(payload)
    checks["schema_validation"] = not schema_issues

    # The JSON Schema is the authoritative structural gate. Semantic validators
    # intentionally do not traverse a payload whose nested types are invalid.
    if schema_issues:
        checks["semantic_validation"] = False
        context_hash = context_sha256(context)
        analysis_date = payload.get("analysis_date") if isinstance(payload, Mapping) else None
        return ScientificBriefValidationReport(
            final_status="failed",
            language=language,
            analysis_date=analysis_date if isinstance(analysis_date, str) else None,
            context_sha256=context_hash,
            checks=checks,
            warnings=[],
            errors=schema_issues,
        )

    assert isinstance(payload, dict)
    checks["semantic_validation"] = True

    expected_date = context.get("analysis", {}).get("resolved_analysis_date")
    try:
        date.fromisoformat(str(payload.get("analysis_date")))
        date_valid = payload.get("analysis_date") == expected_date
    except ValueError:
        date_valid = False
    if not date_valid:
        errors.append(BriefValidationIssue("analysis_date_mismatch", "analysis_date", "Brief analysis date does not match context"))
    checks["analysis_date_validation"] = date_valid

    language_issues = [BriefValidationIssue(item["code"], item["path"], item["message"]) for item in validate_brief_language(payload, language)]
    errors.extend(language_issues)
    checks["language_validation"] = not language_issues
    checks["disclaimer_validation"] = payload.get("disclaimer") == get_mandatory_disclaimer(language)

    section_issues = _section_and_word_issues(payload, settings)
    errors.extend(section_issues)
    checks["section_validation"] = not any(item.code not in {"section_word_limit", "key_message_word_limit", "too_many_key_messages"} for item in section_issues)
    checks["word_limit_validation"] = not any(item.code in {"section_word_limit", "key_message_word_limit", "too_many_key_messages"} for item in section_issues)

    fact_issues, quality_warnings = _fact_and_quality_issues(payload, context, settings)
    errors.extend(fact_issues)
    warnings.extend(quality_warnings)
    checks["fact_id_validation"] = not any("fact" in item.code for item in fact_issues)
    checks["quality_flag_enforcement"] = not any(
        item.code in {
            "suppressed_fact_used", "material_quality_flag_omitted", "suppressed_section_not_limited",
            "suppression_not_disclosed", "insufficient_coverage_not_limited", "monthly_fallback_not_disclosed",
            "monthly_fallback_not_named",
        }
        for item in fact_issues
    )

    numeric_issues = [BriefValidationIssue(item["code"], item["path"], item["message"]) for item in validate_numeric_claims(payload, context)]
    errors.extend(numeric_issues)
    checks["numerical_claim_validation"] = not numeric_issues
    checks["unit_validation"] = not numeric_issues

    claim_issues = _claim_issues(payload, settings)
    errors.extend(claim_issues)
    checks["prohibited_claim_validation"] = not claim_issues

    context_hash = context_sha256(context)
    metadata_hash = generation_metadata.get("context_sha256") if generation_metadata else context_hash
    hash_valid = metadata_hash == context_hash
    if not hash_valid:
        errors.append(BriefValidationIssue("context_hash_mismatch", "generation_metadata", "Generation metadata does not match context"))
    checks["context_hash_validation"] = hash_valid
    checks["cross_language_validation"] = True

    status = "failed" if errors else ("passed_with_warnings" if warnings else "passed")
    return ScientificBriefValidationReport(
        final_status=status,
        language=language,
        analysis_date=payload.get("analysis_date"),
        context_sha256=context_hash,
        checks=checks,
        warnings=[asdict(item) for item in warnings],
        errors=[asdict(item) for item in errors],
    )


def _direction_code(payload: dict[str, Any], language: str) -> str:
    text = _normalize(" ".join(
        paragraph.get("text", "")
        for paragraph in payload.get("recent_evolution", {}).get("paragraphs", [])
    ))
    groups = {
        "es": {
            "increasing": ("aument", "increment", "ascend"),
            "decreasing": ("dismin", "descend", "reduccion"),
            "stable": ("estable", "sin cambio"),
        },
        "en": {
            "increasing": ("increas", "rising"),
            "decreasing": ("decreas", "declin", "falling"),
            "stable": ("stable", "little change"),
        },
    }
    detected = [code for code, terms in groups[language].items() if any(term in text for term in terms)]
    return detected[0] if len(detected) == 1 else "unspecified"


def _representativeness_code(payload: dict[str, Any], language: str) -> str:
    text = _normalize(" ".join(
        paragraph.get("text", "")
        for paragraph in payload.get("spatial_structure", {}).get("paragraphs", [])
    ))
    groups = {
        "es": {
            "coherent": ("coherente", "uniforme"),
            "heterogeneous": ("heterogene", "variabilidad espacial"),
            "mixed": ("mixta", "signos opuestos", "compens"),
            "distributed": ("distribuid", "varios parches"),
            "weak": ("debil",),
        },
        "en": {
            "coherent": ("coherent", "uniform"),
            "heterogeneous": ("heterogene", "spatial variability"),
            "mixed": ("mixed", "opposite signs", "compens"),
            "distributed": ("distributed", "several patches"),
            "weak": ("weak",),
        },
    }
    detected = [code for code, terms in groups[language].items() if any(term in text for term in terms)]
    return detected[0] if len(detected) == 1 else "unspecified"


def validate_cross_language(
    spanish: dict[str, Any], english: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "analysis_date": spanish.get("analysis_date") == english.get("analysis_date"),
        "section_availability": all(
            spanish.get(section, {}).get("status") == english.get(section, {}).get("status")
            for section in NARRATIVE_SECTION_IDS
        ),
        "facts_used": set(spanish.get("facts_used", [])) == set(english.get("facts_used", [])),
        "event_active_state": ("event.event_day" in set(spanish.get("facts_used", []))) == ("event.event_day" in set(english.get("facts_used", []))),
        "patch_count": ("patches.patch_count" in set(spanish.get("facts_used", []))) == ("patches.patch_count" in set(english.get("facts_used", []))),
        "track_count": ("activity.active_track_count" in set(spanish.get("facts_used", []))) == ("activity.active_track_count" in set(english.get("facts_used", []))),
        "family_count": ("activity.active_family_count" in set(spanish.get("facts_used", []))) == ("activity.active_family_count" in set(english.get("facts_used", []))),
        "limitations": {item.get("limitation_id") for item in spanish.get("limitations", [])} == {item.get("limitation_id") for item in english.get("limitations", [])},
        "disclaimers": spanish.get("disclaimer") == get_mandatory_disclaimer("es") and english.get("disclaimer") == get_mandatory_disclaimer("en"),
        "recent_direction": _direction_code(spanish, "es") == _direction_code(english, "en"),
        "representativeness": _representativeness_code(spanish, "es") == _representativeness_code(english, "en"),
    }
    errors = [
        {"code": f"cross_language_{name}", "path": name, "message": f"Cross-language check failed: {name}"}
        for name, passed in checks.items() if not passed
    ]
    return {
        "final_status": "passed" if not errors else "failed",
        "checks": checks,
        "warnings": [],
        "errors": errors,
    }
