"""Deterministic mandatory operational-fact coverage planning.

The plan is derived from validated context state and compact fact references.
It never creates scientific prose or mutates a generated brief.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Mapping

from src.brief_output_schema import NARRATIVE_SECTION_IDS
from src.export_utils import to_json_compatible


CLAIM_SUPPORT_FIELD = "supporting_fact_ids"
REQUIRED_REPRESENTATION = "paragraph_or_key_message_claim_support"
FACTS_USED_SOURCE = "derived_from_claim_support"
FINAL_COVERAGE_CHECKLIST = (
    "verify_each_mandatory_fact_occurs_in_at_least_one_claim_support_field",
    "verify_each_supported_sentence_explicitly_communicates_its_fact",
    "verify_no_mandatory_valid_zero_fact_is_omitted",
    "verify_no_fact_id_is_attached_to_an_unrelated_claim",
    "verify_all_claim_support_ids_exist_in_the_fact_registry",
)

_CATEGORY_SECTION_GUIDANCE: dict[str, tuple[str, ...]] = {
    "regional_state": ("regional_state",),
    "recent_evolution": ("recent_evolution",),
    "spatial_structure": ("spatial_structure",),
    "representativeness": ("spatial_structure", "regional_state"),
    "univariate_event": ("event_status",),
    "daily_patches": ("event_status", "spatial_structure"),
    "spatiotemporal_activity": ("event_status", "spatial_structure"),
    "quality": ("data_quality",),
}


@dataclass(frozen=True)
class MandatoryFactCoverageEntry:
    fact_id: str
    source_context_path: str
    available: bool
    availability_status: str
    allowed_in_brief: bool
    value_available: bool
    value: Any
    unit: str | None
    precision: int | None
    fact_category: str
    suggested_sections: tuple[str, ...]
    valid_zero: bool
    represented_in_claim_support: str = "required"
    required_representation: str = REQUIRED_REPRESENTATION
    claim_support_field: str = CLAIM_SUPPORT_FIELD
    required: bool = True
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


@dataclass(frozen=True)
class MandatoryFactCoveragePlan:
    required_facts: tuple[MandatoryFactCoverageEntry, ...]
    excluded_candidates: tuple[MandatoryFactCoverageEntry, ...]

    @property
    def required_fact_ids(self) -> tuple[str, ...]:
        return tuple(entry.fact_id for entry in self.required_facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_type": "mandatory_operational_fact_coverage",
            "required_representation": REQUIRED_REPRESENTATION,
            "claim_support_field": CLAIM_SUPPORT_FIELD,
            "required_fact_count": len(self.required_facts),
            "required_fact_ids": list(self.required_fact_ids),
            "required_facts": [entry.to_dict() for entry in self.required_facts],
            "excluded_candidates": [
                entry.to_dict() for entry in self.excluded_candidates
            ],
        }


@dataclass(frozen=True)
class FactsUsedDerivationAudit:
    facts_used_source: str
    fact_ids: tuple[str, ...]
    claim_support_locations_inspected: tuple[str, ...]
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def _fact_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _claim_types(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def claim_support_locations(
    payload: Mapping[str, Any],
    *,
    substantive_only: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return every schema-authorized claim-support location.

    Paragraph and key-message support can satisfy mandatory substantive
    coverage. Limitation support remains authoritative for the root
    ``facts_used`` index but cannot satisfy that mandatory coverage rule.
    """
    locations: dict[str, tuple[str, ...]] = {}
    non_substantive_types = {
        "limitation",
        "quality_caveat",
        "methodological_note",
    }
    for section_id in NARRATIVE_SECTION_IDS:
        section = payload.get(section_id)
        if not isinstance(section, Mapping):
            continue
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue
        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, Mapping):
                continue
            claim_types = _claim_types(paragraph.get("claim_types"))
            if substantive_only and claim_types <= non_substantive_types:
                continue
            path = f"{section_id}.paragraphs[{index}].{CLAIM_SUPPORT_FIELD}"
            locations[path] = _fact_ids(paragraph.get(CLAIM_SUPPORT_FIELD))

    messages = payload.get("key_messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            claim_types = _claim_types(message.get("claim_types"))
            if substantive_only and claim_types <= non_substantive_types:
                continue
            path = f"key_messages[{index}].{CLAIM_SUPPORT_FIELD}"
            locations[path] = _fact_ids(message.get(CLAIM_SUPPORT_FIELD))

    if not substantive_only:
        limitations = payload.get("limitations")
        if isinstance(limitations, list):
            for index, limitation in enumerate(limitations):
                if not isinstance(limitation, Mapping):
                    continue
                path = f"limitations[{index}].{CLAIM_SUPPORT_FIELD}"
                locations[path] = _fact_ids(limitation.get(CLAIM_SUPPORT_FIELD))
    return locations


def derive_facts_used_from_claim_support(
    payload: dict[str, Any],
) -> FactsUsedDerivationAudit:
    """Replace the root index using only explicit model-authored support IDs.

    No prose, claim type, support assignment, status, value, or interpretation
    is created or changed. Unknown IDs remain in the derived index so normal
    fact-registry validation can reject them rather than silently dropping them.
    """
    locations = claim_support_locations(payload)
    derived = tuple(
        sorted({fact_id for fact_ids in locations.values() for fact_id in fact_ids})
    )
    original = payload.get("facts_used")
    replacement = list(derived)
    changed = original != replacement
    payload["facts_used"] = replacement
    return FactsUsedDerivationAudit(
        facts_used_source=FACTS_USED_SOURCE,
        fact_ids=derived,
        claim_support_locations_inspected=tuple(locations),
        changed=changed,
    )


def _nested(context: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = context
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _active_candidate_paths(context: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Return current conditional rules as paths to compact fact references."""
    paths: list[tuple[str, ...]] = [("quality", "valid_coverage")]
    if bool(_nested(context, ("univariate_event", "active"))):
        paths.append(("univariate_event", "event_day"))
    if bool(_nested(context, ("daily_patches", "product_available"))):
        paths.append(("daily_patches", "patch_count"))
    if bool(_nested(context, ("spatiotemporal_activity", "product_available"))):
        paths.extend(
            [
                ("spatiotemporal_activity", "active_track_count"),
                ("spatiotemporal_activity", "active_family_count"),
            ]
        )
    return paths


def _suppressed_fact_ids(context: Mapping[str, Any]) -> set[str]:
    return {
        str(fact_id)
        for flag in context.get("quality_flags", [])
        if isinstance(flag, Mapping) and flag.get("suppress_dependent_claims")
        for fact_id in flag.get("affected_fact_ids", [])
    }


def derive_mandatory_fact_coverage_plan(
    context: Mapping[str, Any],
) -> MandatoryFactCoveragePlan:
    """Derive eligible requirements from context state and the fact registry."""
    registry = {
        str(item.get("fact_id")): item
        for item in context.get("fact_registry", [])
        if isinstance(item, Mapping) and item.get("fact_id")
    }
    suppressed = _suppressed_fact_ids(context)
    required: list[MandatoryFactCoverageEntry] = []
    excluded: list[MandatoryFactCoverageEntry] = []
    seen: set[str] = set()

    for path in _active_candidate_paths(context):
        compact_reference = _nested(context, path)
        fact_id = (
            str(compact_reference.get("fact_id"))
            if isinstance(compact_reference, Mapping)
            and compact_reference.get("fact_id")
            else ""
        )
        if not fact_id or fact_id in seen:
            continue
        seen.add(fact_id)
        fact = registry.get(fact_id, {})
        status = str(fact.get("status", "unavailable"))
        value = fact.get("value")
        value_available = value is not None
        available = status == "valid" and value_available
        allowed = bool(fact.get("allowed_in_brief", False))
        category = str(fact.get("section", path[0]))
        exclusion_reason: str | None = None
        if fact_id not in registry:
            exclusion_reason = "not_in_fact_registry"
        elif not available:
            exclusion_reason = "unavailable"
        elif not allowed:
            exclusion_reason = "not_allowed_in_brief"
        elif fact_id in suppressed:
            exclusion_reason = "suppressed_by_quality_control"
        entry = MandatoryFactCoverageEntry(
            fact_id=fact_id,
            source_context_path=".".join(path),
            available=available,
            availability_status=status,
            allowed_in_brief=allowed,
            value_available=value_available,
            value=value,
            unit=str(fact.get("unit")) if fact.get("unit") is not None else None,
            precision=(
                int(fact["precision"])
                if fact.get("precision") is not None
                else None
            ),
            fact_category=category,
            suggested_sections=_CATEGORY_SECTION_GUIDANCE.get(
                category, ("event_status",)
            ),
            valid_zero=(
                available
                and isinstance(value, Real)
                and not isinstance(value, bool)
                and float(value) == 0.0
            ),
            required=exclusion_reason is None,
            exclusion_reason=exclusion_reason,
        )
        (required if entry.required else excluded).append(entry)

    return MandatoryFactCoveragePlan(
        required_facts=tuple(sorted(required, key=lambda item: item.fact_id)),
        excluded_candidates=tuple(sorted(excluded, key=lambda item: item.fact_id)),
    )


def required_fact_repair_details(
    plan: MandatoryFactCoveragePlan,
    missing_fact_ids: set[str],
    validation_errors: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return only authoritative, content-free details for missing facts."""
    issues = {
        str(issue.get("fact_id")): issue
        for issue in (validation_errors or [])
        if issue.get("code") == "required_operational_fact_not_represented"
        and issue.get("fact_id")
    }
    details: list[dict[str, Any]] = []
    for entry in plan.required_facts:
        if entry.fact_id not in missing_fact_ids:
            continue
        detail = entry.to_dict()
        issue = issues.get(entry.fact_id, {})
        detail["currently_present_in_claim_support"] = bool(
            issue.get("present_in_claim_support", False)
        )
        detail["currently_present_in_facts_used"] = bool(
            issue.get("present_in_facts_used", False)
        )
        details.append(detail)
    return details
