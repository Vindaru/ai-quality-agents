import json
from pathlib import Path

import pytest
import yaml

from aiq.evals import runner, scorers
from aiq.evals.judge import CHECKS, ScenarioJudge
from aiq.llm import MockLLM, input_payload

DATA = Path(__file__).parents[1] / "data"
CALIBRATION = yaml.safe_load((DATA / "judge" / "calibration.yaml").read_text())["items"]


def _answer(checks: dict[str, bool], verdict: str | None = None) -> str:
    return json.dumps(
        {
            "checks": {k: {"pass": v, "reason": "r"} for k, v in checks.items()},
            "verdict": verdict or ("pass" if all(checks.values()) else "fail"),
        }
    )


class OracleJudgeLLM:
    """A judge that always agrees with the human labels, to test the trust logic."""

    name = "oracle"

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        assert task == "scenario_judge"
        data = input_payload(prompt)
        for item in CALIBRATION:
            if item["criterion"] == data["acceptance_criterion"] and (
                item["scenario"]["steps"] == data["scenario"]["steps"]
            ):
                return _answer(item["human"])
        return _answer(dict.fromkeys(CHECKS, True))


def test_calibration_set_is_labeled_on_every_check_and_roughly_balanced():
    ids = [i["id"] for i in CALIBRATION]
    assert len(ids) == len(set(ids)) >= 30
    for item in CALIBRATION:
        assert set(item["human"]) == set(CHECKS), item["id"]
        assert item["source"] in {"mock", "human"}
    passes = sum(all(i["human"].values()) for i in CALIBRATION)
    assert 0.35 <= passes / len(CALIBRATION) <= 0.65


def test_verdict_is_derived_from_the_checks_not_trusted_from_the_model():
    checks = dict.fromkeys(CHECKS, True) | {"observable_outcome": False}
    result = ScenarioJudge.parse(_answer(checks, verdict="pass"))
    assert result.verdict == "fail" and not result.checks["observable_outcome"]


@pytest.mark.parametrize("raw", ["no json here", '{"checks": {}}', '{"verdict": "pass"}'])
def test_unparseable_judge_output_is_an_error_not_a_pass(raw):
    assert ScenarioJudge.parse(raw).verdict == "error"


def test_mock_judge_fails_a_placeholder_when_step():
    scenario = {
        "scenario": "x",
        "tags": ["positive"],
        "steps": ["Given a member", "When the workflow runs", "Then status SENT is stored"],
    }
    result = ScenarioJudge(MockLLM()).grade("The referral is stored with status SENT.", scenario)
    assert result.verdict == "fail" and result.checks["concrete_action"] is False


def test_judge_is_not_fooled_by_an_injected_pass_request():
    item = next(i for i in CALIBRATION if i["id"] == "J25")
    assert ScenarioJudge(MockLLM()).grade(item["criterion"], item["scenario"]).verdict == "fail"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (["p", "f", "p", "f"], ["p", "f", "p", "f"], 1.0),
        (["p", "p", "f", "f"], ["p", "f", "p", "f"], 0.0),
        (["p", "p", "p", "f"], ["p", "p", "f", "f"], 0.5),
    ],
)
def test_cohen_kappa(a, b, expected):
    assert scorers.cohen_kappa(a, b) == pytest.approx(expected)


def test_kappa_rejects_mismatched_raters():
    with pytest.raises(ValueError):
        scorers.cohen_kappa(["p"], ["p", "f"])


@pytest.fixture(scope="module")
def generated():
    return runner.eval_test_generator(MockLLM(), DATA / "golden")["cases"]


def test_mock_judge_is_reported_but_not_trusted(generated):
    judge = runner.eval_judge(MockLLM(), generated, DATA / "golden", DATA / "judge")
    thresholds = yaml.safe_load((DATA / "thresholds.yaml").read_text())
    rows, trusted = runner.gate(
        {"judge": judge}, {"judge": thresholds["judge"], "judge_trust": thresholds["judge_trust"]}
    )
    assert not trusted
    assert "scenario_pass_rate" not in {r["metric"] for r in rows}
    cm = judge["confusion_matrix"]
    assert sum(sum(r.values()) for r in cm.values()) == len(CALIBRATION)


def test_a_calibrated_judge_becomes_part_of_the_gate(generated):
    judge = runner.eval_judge(OracleJudgeLLM(), generated, DATA / "golden", DATA / "judge")
    assert judge["metrics"]["agreement"] == judge["metrics"]["cohen_kappa"] == 1.0
    assert judge["findings"] == []
    thresholds = yaml.safe_load((DATA / "thresholds.yaml").read_text())
    rows, trusted = runner.gate(
        {"judge": judge}, {"judge": thresholds["judge"], "judge_trust": thresholds["judge_trust"]}
    )
    assert trusted
    assert "scenario_pass_rate" in {r["metric"] for r in rows}


def test_judge_sees_redacted_text_only():
    """Judge calls go through the PHI guard like every other model call."""
    recorder = runner.RecordingLLM(MockLLM())
    from aiq.llm import GuardedLLM

    scenario = {
        "scenario": "x",
        "tags": ["positive"],
        "steps": ["Given SSN 123-45-6789", "When saved", "Then status SENT"],
    }
    ScenarioJudge(GuardedLLM(recorder)).grade("Status is SENT.", scenario)
    assert "123-45-6789" not in recorder.prompts[0]
