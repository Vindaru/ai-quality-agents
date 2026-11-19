import json
from pathlib import Path

import pytest

from aiq.agents.test_fixer import (
    Problem,
    SuiteTest,
    TestFixAgent,
    find_problems,
    mutation_report,
    restore_placeholders,
    score,
    verify,
)
from aiq.agents.test_to_code import GeneratedTest
from aiq.evals import runner
from aiq.llm import GuardedLLM, MockLLM

DATA = Path(__file__).parents[1] / "data"

WEAK_E164 = """def test_phone_normalised(client):
    r = client.put("/members/M-1/phone", json={"phone": "(555) 010-0199"})
    assert r.status == 200
"""
STRONG_E164 = WEAK_E164 + '    assert client.get("/members/M-1").json["phone"] == "+15550100199"\n'
BROKEN = """def test_invalid_phone_rejected(client):
    r = client.put("/members/M-1/phone", json={"phone": "555-CALL-NOW"})
    assert r.status == 400
"""


def _suite(*codes):
    out = []
    for i, code in enumerate(codes):
        name = code.split("(")[0].removeprefix("def ")
        test = GeneratedTest(test_name=name, code=code)
        passes, kills = score(test)
        out.append(SuiteTest(f"C{i}", "criterion", test, kills, passes))
    return out


def test_find_problems_lists_broken_tests_then_survivors():
    problems = find_problems(_suite(WEAK_E164, BROKEN))
    assert problems[0].kind == "broken_test" and "400" in problems[0].evidence
    assert {p.target for p in problems[1:]} >= {"no_e164", "wrong_invalid_code"}


def test_verify_accepts_a_real_fix_and_rejects_the_shortcuts():
    suite = _suite(WEAK_E164)
    survivor = Problem("surviving_mutant", "no_e164", mutation_report("no_e164"))
    strong = GeneratedTest(test_name="test_phone_normalised", code=STRONG_E164)
    assert verify(strong, suite, survivor)[0] == ""
    # unchanged test: passes, loses nothing, but still misses the bug
    assert "still passes" in verify(suite[0].test, suite, survivor)[0]
    # a new name is not a fix of an existing test
    renamed = GeneratedTest(test_name="test_other", code=STRONG_E164.replace("test_phone_n", "x"))
    assert "no test named" in verify(renamed, suite, survivor)[0]
    # unsafe code never runs
    unsafe = GeneratedTest(test_name="test_phone_normalised", code=STRONG_E164 + "    open('x')\n")
    assert "safety" in verify(unsafe, suite, survivor)[0]


def test_a_patch_that_weakens_a_test_is_rejected():
    strong_suite = _suite(STRONG_E164)
    weakened = GeneratedTest(test_name="test_phone_normalised", code=WEAK_E164)
    problem = Problem("surviving_mutant", "stop_case_sensitive", "x")
    assert "stops catching" in verify(weakened, strong_suite, problem)[0]


def test_a_broken_test_cannot_be_fixed_by_asserting_nothing():
    suite = _suite(BROKEN)
    empty = GeneratedTest(
        test_name="test_invalid_phone_rejected",
        code='def test_invalid_phone_rejected(client):\n    client.put("/members/M-1/phone", '
        'json={"phone": "555-CALL-NOW"})\n',
    )
    reason, _ = verify(empty, suite, Problem("broken_test", "C0", "assert r.status == 400"))
    assert "catches no bug" in reason


def test_rejected_patch_is_retried_with_feedback_and_kept_only_when_verified():
    suite = _suite(WEAK_E164)
    agent = TestFixAgent(GuardedLLM(MockLLM()))
    attempts = agent.fix(Problem("surviving_mutant", "no_e164", mutation_report("no_e164")), suite)
    assert [a.accepted for a in attempts] == [False, True]
    assert "fails on the correct service" in attempts[0].reason
    assert "no_e164" in suite[0].kills and agent.model_calls == 2


def test_placeholders_from_the_phi_guard_are_restored_locally():
    patched = 'x = "[REDACTED_PHONE]"\ny = "[REDACTED_PHONE]"\n'
    original = 'a = "(555) 010-0199"\nb = "(555) 010-0100"\n'
    assert restore_placeholders(patched, original) == (
        'x = "(555) 010-0199"\ny = "(555) 010-0100"\n'
    )


class _Lazy:
    """A fix model that always hands back the test it was given."""

    name = "lazy"

    def complete(self, system, prompt, *, task):
        tests = json.loads(prompt.removeprefix("<input>").removesuffix("</input>"))["tests"]
        return json.dumps({"test_name": tests[0]["test_name"], "code": tests[0]["code"]})


@pytest.fixture(scope="module")
def code_cases():
    return runner.eval_test_to_code(MockLLM(), DATA / "golden")["cases"]


def test_fix_stage_closes_every_gap_on_the_mock(code_cases):
    m = runner.eval_fix(GuardedLLM(MockLLM()), code_cases, DATA / "golden")["metrics"]
    assert m["fix_rate"] == 1.0 and m["kills_lost"] == 0
    assert m["mutation_kill_rate_after"] == 1.0 and m["escaped_defect_count_after"] == 0


def test_a_fixer_that_changes_nothing_fails_the_gate(code_cases):
    """Mutation test of the stage: a do-nothing fixer must not look like a working one."""
    m = runner.eval_fix(_Lazy(), code_cases, DATA / "golden")["metrics"]
    assert m["fix_rate"] == 0 and m["escaped_defect_count_after"] == 1
    assert m["patches_rejected"] == 2 * m["problems"]
