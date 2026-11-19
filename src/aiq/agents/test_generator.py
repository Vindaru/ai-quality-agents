"""Requirements-to-Test agent: user story + acceptance criteria -> Gherkin scenarios."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aiq.evals.scorers import near_duplicates, schema_valid
from aiq.llm import LLM, extract_json, wrap_input

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
- The When step is a concrete action a tester can perform, with a synthetic value where it
  matters ("When a member replies STOP"), never a placeholder such as "When the workflow runs".
- The Then step states a checkable result: a value, status, record, message or HTTP code.
  Never "as specified" or "works correctly".
- Tag "negative" only for invalid input or an error path, "edge" only for a boundary or limit,
  "positive" for the normal path, even when the normal path is an opt-out or a refusal.
- Never include real personal data; use obviously synthetic values."""

REPAIR_PROMPT = """You are a senior QA engineer fixing ONE Gherkin scenario that failed review.
The input has the acceptance criterion, the scenario, and the review problems. Fix exactly the
listed problems, keep the same source_criterion_id and test_layer, and change nothing else.
Return JSON only: one scenario object with the same fields as the input scenario."""

# Deterministic review rules, run on every scenario before it leaves the agent. They catch the
# problems the scenario judge found most often, cheaply and the same way every time. They are
# deliberately separate from the judge: the agent fixes what the lint finds, and the judge, which
# never sees the lint, measures whether the fix made the test better.
PLACEHOLDER_WHEN = re.compile(
    r"^When (?:the workflow runs|it runs|the system processes it|the input sits exactly on)",
    re.I,
)
VAGUE_THEN = re.compile(r"as specified|works correctly|\bit works\b|behaves as expected", re.I)
ERROR_PATH = re.compile(
    r"\b(?:invalid|reject\w*|errors?|unsupported|not supported|missing|expired|unknown|fails?"
    r"|without|never|no further|no outreach|denied|unauthori\w*|validation|[45]\d\d)\b",
    re.I,
)
STRONG_ERROR = re.compile(r"\b(?:invalid|reject\w*|unknown|expired|unauthori\w*|4\d\d)\b", re.I)


def lint_scenario(criterion: str, scenario: Scenario) -> list[str]:
    """Review problems for one scenario, as short codes with the evidence."""
    problems = []
    whens = [s for s in scenario.steps if s.startswith("When")]
    thens = [s for s in scenario.steps if s.startswith(("Then", "And", "But"))]
    if not whens or any(PLACEHOLDER_WHEN.search(w) for w in whens):
        problems.append("PLACEHOLDER_WHEN: the When step is not a concrete action")
    if not thens or any(VAGUE_THEN.search(t) for t in thens):
        problems.append("VAGUE_THEN: the Then step does not state a checkable result")
    if "negative" in scenario.tags and not ERROR_PATH.search(criterion):
        problems.append("TAG_NEGATIVE_ON_NORMAL_PATH: the criterion describes a normal path")
    if "positive" in scenario.tags and STRONG_ERROR.search(criterion):
        problems.append("TAG_POSITIVE_ON_ERROR_PATH: the criterion describes an error path")
    return problems


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
    repairs: list[dict] = field(default_factory=list)
    model_calls: int = 0

    @property
    def raw_count(self) -> int:
        return len(self.scenarios) + len(self.invalid)


class RequirementsToTestAgent:
    def __init__(self, llm: LLM, *, repair: bool = True) -> None:
        self.llm = llm
        self.repair = repair

    def run(self, story: Story) -> GenerationResult:
        prompt = wrap_input(story.model_dump())
        raw = self.llm.complete(SYSTEM_PROMPT, prompt, task="test_generation")
        result = self._parse(story.id, raw)
        result.model_calls = 1
        if self.repair:
            self._repair(story, result)
        self._check(story, result)
        return result

    def _repair(self, story: Story, result: GenerationResult) -> None:
        """One review-and-fix round: lint every scenario, and send each one with problems back
        to the model once, with the problems listed. A repair is kept only if it is schema
        valid, still traces to the same criterion, and has fewer problems than before. One
        round only, so cost is bounded at one extra call per flagged scenario."""
        criteria = {c.id: c.text for c in story.acceptance_criteria}
        for i, scenario in enumerate(result.scenarios):
            criterion = criteria.get(scenario.source_criterion_id)
            if criterion is None:
                continue  # untraced: reported by _check, nothing to repair against
            problems = lint_scenario(criterion, scenario)
            if not problems:
                continue
            raw = self.llm.complete(
                REPAIR_PROMPT,
                wrap_input(
                    {
                        "acceptance_criterion": criterion,
                        "scenario": scenario.model_dump(),
                        "problems": problems,
                    }
                ),
                task="test_repair",
            )
            result.model_calls += 1
            record = {"criterion_id": scenario.source_criterion_id, "before": problems}
            try:
                fixed = Scenario.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                result.repairs.append(
                    {
                        **record,
                        "after": problems,
                        "accepted": False,
                        "reason": f"invalid repair: {exc}"[:200],
                    }
                )
                continue
            after = lint_scenario(criterion, fixed)
            accepted = fixed.source_criterion_id == scenario.source_criterion_id and len(
                after
            ) < len(problems)
            if accepted:
                result.scenarios[i] = fixed
            result.repairs.append(
                {
                    **record,
                    "after": after if accepted else problems,
                    "accepted": accepted,
                    "reason": "",
                }
            )

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
        criteria = {c.id: c.text for c in story.acceptance_criteria}
        for s in scenarios:
            if s.source_criterion_id in criteria:
                for problem in lint_scenario(criteria[s.source_criterion_id], s):
                    code, _, detail = problem.partition(": ")
                    result.issues.append(Issue(code, f"{s.source_criterion_id}: {detail}"))
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
