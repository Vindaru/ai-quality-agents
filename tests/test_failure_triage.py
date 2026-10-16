from collections import Counter
from pathlib import Path

import yaml

from aiq.agents.failure_triage import CATEGORIES, Failure, FailureTriageAgent
from aiq.llm import MockLLM

GOLDEN = Path(__file__).parents[1] / "data/golden/triage.yaml"


class _CountingLLM(MockLLM):
    calls = 0

    def complete(self, system, prompt, *, task):
        type(self).calls += 1
        return super().complete(system, prompt, task=task)


class _BrokenLLM:
    name = "broken"

    def complete(self, system, prompt, *, task):
        return "not json"


def _failure(error, logs=()):
    return Failure(id="X", test_name="test_x", error=error, logs=list(logs))


def test_confident_rule_skips_the_llm():
    llm = _CountingLLM()
    result = FailureTriageAgent(llm).run(_failure("HTTPError: 503 Service Unavailable"))
    assert (result.category, result.routed_to, llm.calls) == ("environment", "rules", 0)
    assert result.suggested_owner == "platform-devops"


def test_locator_error_is_automation():
    result = FailureTriageAgent(MockLLM()).run(_failure("NoSuchElementException: #submit"))
    assert result.category == "automation_issue"


def test_conflicting_rules_go_to_llm():
    failure = _failure("HTTP 500 Internal Server Error", logs=["ECONNRESET upstream"])
    result = FailureTriageAgent(MockLLM()).run(failure)
    assert result.routed_to == "llm" and "disagree" in result.route_reason


def test_unknown_failure_goes_to_llm():
    result = FailureTriageAgent(MockLLM()).run(
        _failure("AssertionError: expected SENT but was FAILED")
    )
    assert (result.routed_to, result.category) == ("llm", "product_bug")


def test_invalid_llm_answer_is_unclassified_not_a_crash():
    result = FailureTriageAgent(_BrokenLLM()).run(_failure("something odd happened"))
    assert result.category == "unclassified" and result.confidence == 0.0


def test_golden_set_is_balanced_and_labeled():
    failures = yaml.safe_load(GOLDEN.read_text())["failures"]
    assert 40 <= len(failures) <= 60
    assert set(Counter(f["label"] for f in failures).values()) == {12}
    assert {f["label"] for f in failures} == set(CATEGORIES)
    assert len({f["id"] for f in failures}) == len(failures)
