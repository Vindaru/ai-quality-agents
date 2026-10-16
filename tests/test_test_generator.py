import json
from pathlib import Path

import yaml

from aiq.agents.test_generator import RequirementsToTestAgent, Scenario, Story
from aiq.llm import MockLLM

GOLDEN = Path(__file__).parents[1] / "data/golden/test_generator.yaml"

STORY = Story(
    id="US-T",
    title="Member opts out of SMS",
    narrative="As a member I want to stop text messages.",
    acceptance_criteria=[
        {"id": "AC1", "text": "When a member replies STOP, the system sets sms_opt_out to true."},
        {"id": "AC2", "text": "A phone number with fewer than 10 digits is rejected."},
    ],
)


class _FixedLLM:
    name = "fixed"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def complete(self, system, prompt, *, task):
        return self.answer


def _scenario(**overrides):
    base = {
        "feature": "f",
        "scenario": "s",
        "steps": ["Given a member", "Then it works"],
        "test_layer": "API",
        "source_criterion_id": "AC1",
        "tags": ["positive"],
    }
    return {**base, **overrides}


def test_mock_output_is_schema_valid_and_traced():
    result = RequirementsToTestAgent(MockLLM()).run(STORY)
    assert result.invalid == []
    assert {s.source_criterion_id for s in result.scenarios} == {"AC1", "AC2"}
    assert all(isinstance(s, Scenario) for s in result.scenarios)


def test_mock_is_deterministic():
    agent = RequirementsToTestAgent(MockLLM())
    assert agent.run(STORY).scenarios == agent.run(STORY).scenarios


def test_invalid_layer_and_bad_gherkin_are_rejected():
    answer = json.dumps(
        {"scenarios": [_scenario(test_layer="E2E"), _scenario(steps=["Click it", "Then ok"])]}
    )
    result = RequirementsToTestAgent(_FixedLLM(answer)).run(STORY)
    assert result.scenarios == [] and len(result.invalid) == 2
    codes = [i.code for i in result.issues]
    assert codes.count("INVALID_SCENARIO") == 2 and "UNCOVERED_CRITERION" in codes


def test_checks_flag_untraced_duplicates_and_missing_negatives():
    answer = json.dumps(
        [
            _scenario(scenario="reply STOP disables sms"),
            _scenario(scenario="reply STOP disables sms"),
            _scenario(source_criterion_id="AC9"),
        ]
    )
    codes = {i.code for i in RequirementsToTestAgent(_FixedLLM(answer)).run(STORY).issues}
    assert {"UNTRACED", "NEAR_DUPLICATE", "NO_NEGATIVE_CASE", "UNCOVERED_CRITERION"} <= codes


def test_risk_checklist_flags_missing_invalid_phone_case():
    answer = json.dumps([_scenario(scenario="member replies STOP", tags=["negative"])])
    issues = RequirementsToTestAgent(_FixedLLM(answer)).run(STORY).issues
    assert ("MISSING_RISK_CASE", "invalid phone") in [(i.code, i.detail) for i in issues]


def test_non_json_answer_is_reported_not_raised():
    result = RequirementsToTestAgent(_FixedLLM("I cannot help")).run(STORY)
    assert result.scenarios == [] and result.invalid


def test_golden_set_is_well_formed():
    stories = yaml.safe_load(GOLDEN.read_text())["stories"]
    assert 12 <= len(stories) <= 15
    for raw in stories:
        story = Story.model_validate(raw)
        ids = {c.id for c in story.acceptance_criteria}
        assert {e["criterion"] for e in raw["expected"]} <= ids
        for exp in raw["expected"]:
            assert exp["kind"] in {"positive", "negative", "edge"}
            assert all(isinstance(k, str) for k in exp["keywords"]), exp
