import json
from pathlib import Path

import pytest

from aiq.agents.test_to_code import GeneratedTest
from aiq.evals import runner
from aiq.execution import execute, release_decision, to_failure
from aiq.llm import MockLLM

DATA = Path(__file__).parents[1] / "data"

GOOD = GeneratedTest(
    test_name="test_unknown_referral_404",
    code=(
        "def test_unknown_referral_404(client):\n"
        '    r = client.post("/referrals/R-404/status", json={"status": "X"},\n'
        '                    headers={"Authorization": "tok-valid"})\n'
        "    assert r.status == 404\n"
    ),
)
WRONG = GeneratedTest(
    test_name="test_member_is_opted_out",
    code=(
        "def test_member_is_opted_out(client):\n"
        '    assert client.get("/members/M-1").json["sms_opt_out"]\n'
    ),
)


def test_regression_is_a_product_bug_and_blocks():
    [run] = execute([GOOD], bug="unknown_referral_200")
    assert run.outcome == "fail" and run.expected_category == "product_bug"
    failure = to_failure(run, "R", None)
    assert "assert r.status == 404" in failure.error
    assert failure.logs == ["passes on the baseline build B-good"]


def test_a_test_wrong_on_the_baseline_is_an_automation_issue():
    [run] = execute([WRONG])
    assert run.fails_on_baseline and run.expected_category == "automation_issue"
    assert "also fails on the baseline build" in to_failure(run, "R", None).logs[-1]


@pytest.mark.parametrize(
    ("fault", "category", "error"),
    [
        ("gateway_down", "environment", "ConnectionRefusedError"),
        ("seed_missing", "test_data", "LookupError"),
    ],
)
def test_environment_faults_get_their_own_category(fault, category, error):
    [run] = execute([WRONG], fault=fault)
    assert run.outcome == "error" and run.message.startswith(error)
    assert run.expected_category == category
    assert to_failure(run, "R", fault).logs  # the fault's log lines travel with the failure


def test_passing_tests_have_no_category():
    [run] = execute([GOOD])
    assert run.outcome == "pass" and run.expected_category is None


@pytest.mark.parametrize(
    ("categories", "decision"),
    [
        ([], "ship"),
        (["automation_issue"], "ship_with_test_fix"),
        (["automation_issue", "environment"], "rerun"),
        (["test_data", "product_bug"], "block"),
    ],
)
def test_release_decision(categories, decision):
    assert release_decision(categories) == decision


class _BlameTheEnvironment:
    """A triage model that calls everything an environment problem."""

    name = "blame-env"

    def complete(self, system, prompt, *, task):
        return json.dumps({"category": "environment", "confidence": 0.9, "evidence": []})


def test_bad_triage_shows_up_as_wrong_release_decisions():
    code_cases = runner.eval_test_to_code(MockLLM(), DATA / "golden")["cases"]
    bad = runner.eval_run_and_triage(_BlameTheEnvironment(), code_cases, DATA / "golden")
    # Product bugs reach the model (no rule matches them), get called environment, and the
    # release decision turns from block into rerun: a regression would be retried, not stopped.
    assert bad["metrics"]["decision_accuracy"] < 1.0
    wrong = {r["id"] for r in bad["runs"] if r["decision"] != r["expected_decision"]}
    assert wrong == {"R2", "R3", "R4"}


def test_escaped_defect_is_counted():
    code_cases = runner.eval_test_to_code(MockLLM(), DATA / "golden")["cases"]
    result = runner.eval_run_and_triage(MockLLM(), code_cases, DATA / "golden")
    assert result["metrics"]["escaped_defect_count"] == 1
    assert any(f["type"] == "escaped_defect" and f["case"] == "R7" for f in result["findings"])
