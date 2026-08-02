"""Application-owned mandatory disclaimer for scientific briefs.

The disclaimer is operational metadata, not model-authored scientific content.
This module is the single source used by insertion, validation, rendering, and
tests.  No fuzzy or normalized matching is permitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, MutableMapping


DISCLAIMER_SOURCE = "application_constant"
SUPPORTED_DISCLAIMER_LANGUAGES = ("es", "en")

_DISCLAIMER_VALUES = {
    "es": (
        "Este producto es experimental y no constituye una clasificación oficial "
        "de la magnitud de El Niño Costero."
    ),
    "en": (
        "This is an experimental product and does not constitute an official "
        "classification of Coastal El Niño magnitude."
    ),
}

# Read-only compatibility export.  The literals themselves exist only above.
MANDATORY_DISCLAIMERS = MappingProxyType(_DISCLAIMER_VALUES)


class MandatoryDisclaimerConfigurationError(RuntimeError):
    """Stable local configuration error raised before any provider request."""

    code = "mandatory_disclaimer_configuration_error"


@dataclass(frozen=True)
class DisclaimerInsertionAudit:
    disclaimer_source: str
    disclaimer_inserted: bool
    disclaimer_language: str
    raw_model_disclaimer_present: bool
    raw_model_disclaimer_matched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_mandatory_disclaimer(language: str) -> str:
    """Return the one exact disclaimer for a supported language."""
    if language not in SUPPORTED_DISCLAIMER_LANGUAGES:
        raise MandatoryDisclaimerConfigurationError(
            "No mandatory disclaimer is configured for the requested language"
        )
    value = _DISCLAIMER_VALUES.get(language)
    if not isinstance(value, str) or not value:
        raise MandatoryDisclaimerConfigurationError(
            "The mandatory disclaimer configuration is unavailable"
        )
    return value


def insert_mandatory_disclaimer(
    payload: MutableMapping[str, Any], language: str,
) -> DisclaimerInsertionAudit:
    """Set only the application-owned disclaimer and return content-free audit data."""
    if not isinstance(payload, MutableMapping):
        raise TypeError("Scientific brief payload must be a mutable mapping")
    expected = get_mandatory_disclaimer(language)
    raw_present = "disclaimer" in payload
    raw_matched = raw_present and payload.get("disclaimer") == expected
    payload["disclaimer"] = expected
    return DisclaimerInsertionAudit(
        disclaimer_source=DISCLAIMER_SOURCE,
        disclaimer_inserted=True,
        disclaimer_language=language,
        raw_model_disclaimer_present=raw_present,
        raw_model_disclaimer_matched=raw_matched,
    )

