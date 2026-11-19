import json
from pathlib import Path

import yaml

from aiq.agents.test_generator import RequirementsToTestAgent, Scenario, Story, lint_scenario
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


# --- review-and-fix loop ---------------------------------------------------------------------

STOP = "When a member replies STOP, the system sets sms_opt_out to true."
SHORT_PHONE = "A phone number with fewer than 10 digits is rejected."


def _s(steps, tags=("positive",), cid="AC1"):
    return Scenario.model_validate(
        _scenario(steps=list(steps), tags=list(tags), source_criterion_id=cid)
    )


def test_lint_flags_placeholder_when_and_vague_then():
    problems = lint_scenario(
        SHORT_PHONE,
        _s(
            ["Given a member", "When the workflow runs", "Then the limit is applied as specified"],
            tags=["negative"],
        ),
    )
    assert [p.split(":")[0] for p in problems] == ["PLACEHOLDER_WHEN", "VAGUE_THEN"]


def test_lint_flags_wrong_tags_both_ways():
    good_steps = ["Given a member", "When a member replies STOP", "Then sms_opt_out is true"]
    assert "TAG_NEGATIVE_ON_NORMAL_PATH" in lint_scenario(STOP, _s(good_steps, ["negative"]))[0]
    error = "A status update for an unknown referral id returns HTTP 404."
    steps = ["Given a member", "When an update for id R-0 is sent", "Then HTTP 404 is returned"]
    assert "TAG_POSITIVE_ON_ERROR_PATH" in lint_scenario(error, _s(steps, ["positive"]))[0]


def test_a_clean_scenario_passes_lint():
    steps = ["Given a member", "When a member replies STOP", "Then sms_opt_out is set to true"]
    assert lint_scenario(STOP, _s(steps)) == []


class _ScriptedLLM:
    """First call drafts scenarios; later calls answer repairs with a fixed reply."""

    name = "scripted"

    def __init__(self, draft: list[dict], repair_reply: str) -> None:
        self.draft, self.repair_reply, self.tasks = draft, repair_reply, []

    def complete(self, system, prompt, *, task):
        self.tasks.append(task)
        if task == "test_generation":
            return json.dumps({"scenarios": self.draft})
        return self.repair_reply


PLACEHOLDER = _scenario(
    steps=["Given a member", "When the workflow runs", "Then sms_opt_out is set to true"]
)


def test_repair_is_kept_when_it_removes_problems():
    fixed = _scenario(
        steps=["Given a member", "When a member replies STOP", "Then sms_opt_out is set to true"]
    )
    llm = _ScriptedLLM([PLACEHOLDER], json.dumps(fixed))
    result = RequirementsToTestAgent(llm).run(STORY)
    assert llm.tasks == ["test_generation", "test_repair"] and result.model_calls == 2
    assert result.scenarios[0].steps[1] == "When a member replies STOP"
    assert result.repairs[0]["accepted"] and result.repairs[0]["after"] == []


def test_repair_that_changes_the_criterion_or_breaks_the_schema_is_rejected():
    moved = _scenario(
        source_criterion_id="AC2",
        steps=["Given a member", "When a member replies STOP", "Then x is set"],
    )
    for reply in (json.dumps(moved), "not json", json.dumps({**PLACEHOLDER, "test_layer": "E2E"})):
        result = RequirementsToTestAgent(_ScriptedLLM([PLACEHOLDER], reply)).run(STORY)
        assert result.scenarios[0].steps[1] == "When the workflow runs"
        assert result.repairs[0]["accepted"] is False
        assert any(i.code == "PLACEHOLDER_WHEN" for i in result.issues)


def test_repair_off_makes_one_call_and_still_reports_lint():
    llm = _ScriptedLLM([PLACEHOLDER], "unused")
    result = RequirementsToTestAgent(llm, repair=False).run(STORY)
    assert llm.tasks == ["test_generation"] and result.repairs == []
    assert any(i.code == "PLACEHOLDER_WHEN" for i in result.issues)
