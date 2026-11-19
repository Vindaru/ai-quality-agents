"""Stage 3: run a generated suite against a release candidate and triage what fails.

A *release run* is one candidate service build (correct, or with one planted bug) in one test
environment (healthy, gateway down, or seed data missing). Every generated test runs on it and
on the known-good baseline build in a healthy environment. Each failure becomes a ``Failure``
record, written the way a CI system would (error, failing line, log lines), and goes to the
Failure-Triage agent.

Ground truth comes from what was injected, so it needs no labeling:

- the environment fault broke the test        -> that fault's category (environment, test_data)
- the assertion also fails on the baseline    -> automation_issue (the test itself is wrong)
- the assertion fails only on the candidate   -> product_bug (the build regressed)
- the test crashed in its own code            -> automation_issue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiq.agents.failure_triage import Failure
from aiq.agents.test_to_code import GeneratedTest, run_test
from aiq.demo.referral_service import FAULTS, ReferralService

FAULT_ERRORS = {"gateway_down": "ConnectionRefusedError", "seed_missing": "LookupError"}


@dataclass
class TestRun:
    __test__ = False  # not a pytest class

    test_name: str
    outcome: str  # pass | fail | error
    message: str
    fails_on_baseline: bool
    expected_category: str | None = None  # None when the test passed


def execute(
    tests: list[GeneratedTest], bug: str | None = None, fault: str | None = None
) -> list[TestRun]:
    runs = []
    for test in tests:
        baseline_outcome, _ = run_test(test, ReferralService())
        outcome, message = run_test(test, ReferralService(bug), fault)
        run = TestRun(test.test_name, outcome, message, baseline_outcome != "pass")
        if outcome != "pass":
            run.expected_category = _ground_truth(run, fault)
        runs.append(run)
    return runs


def _ground_truth(run: TestRun, fault: str | None) -> str:
    if fault and run.outcome == "error" and run.message.startswith(FAULT_ERRORS[fault]):
        return FAULTS[fault]["category"]
    if run.outcome == "error" or run.fails_on_baseline:
        return "automation_issue"
    return "product_bug"


def to_failure(run: TestRun, run_id: str, fault: str | None) -> Failure:
    """What CI would hand the triage agent: the error, plus the log lines it would have."""
    logs = list(FAULTS[fault]["logs"]) if fault else []
    if run.outcome == "fail":
        logs.append(
            "also fails on the baseline build B-good"
            if run.fails_on_baseline
            else "passes on the baseline build B-good"
        )
    kind = "AssertionError: " if run.outcome == "fail" else ""
    return Failure(
        id=f"{run_id}/{run.test_name}",
        test_name=run.test_name,
        error=f"{kind}{run.message}",
        logs=logs,
    )


def release_decision(categories: list[str]) -> str:
    """What the run means for the release. Product bugs block it. Environment and test-data
    failures mean the run proved nothing, so rerun. Broken tests alone don't block a release,
    but they get a ticket."""
    if "product_bug" in categories:
        return "block"
    if {"environment", "test_data"} & set(categories):
        return "rerun"
    if "automation_issue" in categories:
        return "ship_with_test_fix"
    return "ship"


def summarize(runs: list[TestRun]) -> dict[str, Any]:
    return {
        "tests": len(runs),
        "failed": sum(r.outcome != "pass" for r in runs),
        "expected": [r.expected_category for r in runs if r.expected_category],
    }
