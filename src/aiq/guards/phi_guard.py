"""Detect and redact PHI-like values before text reaches a model.

The patterns are deliberately conservative (they over-redact rather than under-redact):
in a regulated setting a false positive costs a slightly worse prompt, while a false
negative is a reportable disclosure. Only the *type* and *count* of what was blocked is
logged, never the value itself.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Order matters: more specific identifiers are redacted before broader ones.
PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\bMRN[:#\s]*\d{6,10}\b", re.IGNORECASE),
    "MEMBER_ID": re.compile(r"\bMBR-\d{6}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?:\+1[\s.-]?)?(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"),
    "DOB": re.compile(
        r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b(?!T)"
    ),
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b\.?"
    ),
}


@dataclass(frozen=True)
class Finding:
    phi_type: str
    value: str


@dataclass
class Redaction:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(f.phi_type for f in self.findings))


def scan(text: str) -> list[Finding]:
    """Return every PHI-like value found in ``text`` (no redaction)."""
    return _apply(text)[1]


def _apply(text: str) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    for phi_type, pattern in PHI_PATTERNS.items():

        def _sub(match: re.Match[str], phi_type: str = phi_type) -> str:
            findings.append(Finding(phi_type, match.group(0)))
            return f"[REDACTED_{phi_type}]"

        text = pattern.sub(_sub, text)
    return text, findings


class PhiGuard:
    """Redacts PHI and keeps a value-free audit trail of what it blocked."""

    def __init__(self) -> None:
        self.audit: list[dict[str, int]] = []

    def redact(self, text: str) -> Redaction:
        redacted, findings = _apply(text)
        result = Redaction(redacted, findings)
        if findings:
            self.audit.append(result.counts)
            logger.info("phi_guard redacted %s", result.counts)
        return result

    @property
    def total_blocked(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        for entry in self.audit:
            total.update(entry)
        return dict(total)
