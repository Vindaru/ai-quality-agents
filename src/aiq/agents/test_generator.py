"""Requirements-to-Test agent: user story + acceptance criteria -> Gherkin scenarios."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiq.evals.scorers import near_duplicates, schema_valid
from aiq.llm import LLM, extract_json

SYSTEM_PROMPT = """You are a senior QA engineer for a member outreach and referral platform.
Turn the user story into Gherkin test scenarios. Return JSON only, shaped as:
{"scenarios": [{"feature": str, "scenario": str, "steps": [str, ...],
  "test_layer": "UI" | "API" | "DB", "source_criterion_id": str,
  "tags": ["positive" | "negative" | "edge", ...]}]}
Rules:
- Every scenario traces to exactly one acceptance criterion id from the input.
- Steps start with Given, When, Then, And or But.
- Choose the lowest test layer that can prove the criterion.
- Add negative and boundary scenarios where they apply (opt-out, invalid phone numbers,
  unsupported languages, limits).
- Never include real personal data; use obviously synthetic values."""


class Criterion(BaseModel):
    id: str
    text: str


class Story(BaseModel):
    id: str
    title: str
    narrative: str
    acceptance_criteria: list[Criterion]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    steps: list[str] = Field(min_length=2)
    test_layer: Literal["UI", "API", "DB"]
    source_criterion_id: str
    tags: list[Literal["positive", "negative", "edge"]] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def _gherkin_keywords(cls, steps: list[str]) -> list[str]:
        for step in steps:
            if step.split(" ", 1)[0] not in {"Given", "When", "Then", "And", "But"}:
                raise ValueError(f"step does not start with a Gherkin keyword: {step!r}")
        return steps

    @property
    def text(self) -> str:
        return " ".join([self.scenario, *self.steps])


# Domain risks a reviewer would ask about. (story trigger, risk name, evidence in scenarios)
RISK_CHECKLIST: list[tuple[re.Pattern[str], str, re.Pattern[str]]] = [
    (re.compile(r"\bsms\b|text message", re.I), "opt-out", re.compile(r"opt.?out|\bstop\b", re.I)),
    (
        re.compile(r"\bsms\b|\bphone\b|\bcall", re.I),
        "invalid phone",
        re.compile(
            r"invalid (?:phone|number)|phone numbers? (?:is|are) invalid|fewer than \d+ digits",
            re.I,
        ),
    ),
    (
        re.compile(r"language|spanish|translat", re.I),
        "unsupported language",
        re.compile(r"unsupported language|not supported|missing for the member's language", re.I),
    ),
]


@dataclass
class Issue:
    code: str
    detail: str


@dataclass
class GenerationResult:
    story_id: str
    scenarios: list[Scenario]
    invalid: list[dict] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def raw_count(self) -> int:
        return len(self.scenarios) + len(self.invalid)


class RequirementsToTestAgent:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, story: Story) -> GenerationResult:
        prompt = f"<input>{story.model_dump_json()}</input>"
        raw = self.llm.complete(SYSTEM_PROMPT, prompt, task="test_generation")
        result = self._parse(story.id, raw)
        self._check(story, result)
        return result

    @staticmethod
    def _parse(story_id: str, raw: str) -> GenerationResult:
        try:
            payload = extract_json(raw)
        except ValueError as exc:
            return GenerationResult(story_id, [], [{"error": str(exc), "raw": raw[:500]}])
        items = payload.get("scenarios", []) if isinstance(payload, dict) else payload
        result = GenerationResult(story_id, [])
        for item in items if isinstance(items, list) else []:
            ok, error = schema_valid(item, Scenario)
            if ok:
                result.scenarios.append(Scenario.model_validate(item))
            else:
                result.invalid.append({"error": error, "raw": json.dumps(item)[:500]})
        return result

    @staticmethod
    def _check(story: Story, result: GenerationResult) -> None:
        ids = {c.id for c in story.acceptance_criteria}
        scenarios = result.scenarios
        for bad in result.invalid:
            result.issues.append(Issue("INVALID_SCENARIO", bad["error"]))
        for s in scenarios:
            if s.source_criterion_id not in ids:
                result.issues.append(
                    Issue("UNTRACED", f"{s.scenario!r} -> {s.source_criterion_id}")
                )
        covered = {s.source_criterion_id for s in scenarios}
        for c in story.acceptance_criteria:
            if c.id not in covered:
                result.issues.append(Issue("UNCOVERED_CRITERION", c.id))
        result.duplicate_pairs = near_duplicates([s.text for s in scenarios])
        for i, j in result.duplicate_pairs:
            dup = f"{scenarios[i].source_criterion_id} ~ {scenarios[j].source_criterion_id}"
            result.issues.append(Issue("NEAR_DUPLICATE", dup))
        if not any({"negative", "edge"} & set(s.tags) for s in scenarios):
            result.issues.append(Issue("NO_NEGATIVE_CASE", story.id))
        story_text = " ".join(
            [story.title, story.narrative, *(c.text for c in story.acceptance_criteria)]
        )
        all_scenarios = " ".join(s.text for s in scenarios)
        for trigger, risk, evidence in RISK_CHECKLIST:
            if trigger.search(story_text) and not evidence.search(all_scenarios):
                result.issues.append(Issue("MISSING_RISK_CASE", risk))
