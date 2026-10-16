"""Failure-Triage agent: a failed test -> category, confidence, evidence and owner.

Hybrid by design: cheap, explainable regex rules handle the failures they are sure
about; only the rest (no rule, or rules that disagree) goes to the LLM. That keeps
cost and latency down and makes most decisions auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from aiq.llm import LLM, extract_json, wrap_input

Category = Literal["product_bug", "automation_issue", "environment", "test_data"]
CATEGORIES: list[str] = ["product_bug", "automation_issue", "environment", "test_data"]

OWNERS = {
    "product_bug": "dev-team",
    "automation_issue": "qa-automation",
    "environment": "platform-devops",
    "test_data": "qa-test-data",
    "unclassified": "qa-triage-rotation",
}

SYSTEM_PROMPT = """You triage failed automated tests. Classify the failure as exactly one of:
product_bug (the system under test behaves wrongly), automation_issue (the test code,
locators or waits are wrong), environment (infrastructure, network, dependencies down),
test_data (missing, stale or conflicting test data).
Return JSON only: {"category": str, "confidence": number 0-1, "evidence": [short quotes]}"""


@dataclass(frozen=True)
class Rule:
    category: str
    pattern: re.Pattern[str]
    confidence: float
    reason: str


RULES: list[Rule] = [
    Rule(
        "environment",
        re.compile(r"\b(?:502|503|504)\b|Bad Gateway|Service Unavailable|Gateway Timeout"),
        0.95,
        "gateway / upstream 5xx",
    ),
    Rule(
        "environment",
        re.compile(
            r"Connection ?Refused|ECONNREFUSED|ECONNRESET|Name or service not known"
            r"|getaddrinfo|Read timed out|connect timeout",
            re.I,
        ),
        0.9,
        "network or connectivity failure",
    ),
    Rule(
        "automation_issue",
        re.compile(
            r"NoSuchElement|StaleElementReference|ElementNotInteractable|ElementClickIntercepted"
            r"|InvalidSelector|waiting for (?:selector|locator)|locator\.\w+: Timeout"
        ),
        0.9,
        "UI locator or synchronisation error",
    ),
    Rule(
        "test_data",
        re.compile(
            r"seed (?:data|job|file)|fixture\b.*\bnot found|no test member"
            r"|duplicate key value|UniqueViolation|FOREIGN KEY constraint",
            re.I,
        ),
        0.9,
        "missing or conflicting test data",
    ),
    Rule(
        "product_bug",
        re.compile(r"Internal Server Error|HTTP 500"),
        0.85,
        "server-side 500 from the system under test",
    ),
]


class Failure(BaseModel):
    id: str
    test_name: str
    error: str
    stack: str = ""
    logs: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join([self.test_name, self.error, self.stack, *self.logs])


class Triage(BaseModel):
    failure_id: str
    category: Literal["product_bug", "automation_issue", "environment", "test_data", "unclassified"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    suggested_owner: str
    routed_to: Literal["rules", "llm"]
    route_reason: str


class _LLMAnswer(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class FailureTriageAgent:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, failure: Failure) -> Triage:
        hits = [(rule, m.group(0)) for rule in RULES if (m := rule.pattern.search(failure.text))]
        categories = {rule.category for rule, _ in hits}
        if len(categories) == 1:
            category = categories.pop()
            return Triage(
                failure_id=failure.id,
                category=category,
                confidence=max(rule.confidence for rule, _ in hits),
                evidence=[f"{rule.reason}: {match!r}" for rule, match in hits],
                suggested_owner=OWNERS[category],
                routed_to="rules",
                route_reason="single rule category matched",
            )
        reason = "no rule matched" if not hits else f"rules disagree: {sorted(categories)}"
        return self._ask_llm(failure, reason)

    def _ask_llm(self, failure: Failure, reason: str) -> Triage:
        payload = failure.model_dump(exclude={"id"})
        raw = self.llm.complete(SYSTEM_PROMPT, wrap_input(payload), task="failure_triage")
        try:
            answer = _LLMAnswer.model_validate(extract_json(raw))
        except (ValueError, ValidationError):
            return Triage(
                failure_id=failure.id,
                category="unclassified",
                confidence=0.0,
                evidence=["model answer was not valid JSON for the schema"],
                suggested_owner=OWNERS["unclassified"],
                routed_to="llm",
                route_reason=reason,
            )
        return Triage(
            failure_id=failure.id,
            category=answer.category,
            confidence=answer.confidence,
            evidence=answer.evidence,
            suggested_owner=OWNERS[answer.category],
            routed_to="llm",
            route_reason=reason,
        )
