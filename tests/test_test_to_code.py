import json
from pathlib import Path

import pytest

from aiq.agents.test_to_code import GeneratedTest, TestToCodeAgent, run_test, safety_problem
from aiq.demo.referral_service import BUGS, ReferralService, ServiceClient
from aiq.evals import runner

DATA = Path(__file__).parents[1] / "data"


def _test(body: str, name: str = "test_x") -> GeneratedTest:
    return GeneratedTest(test_name=name, code=f"def {name}(client):\n{body}\n")


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("import os\ndef test_x(client):\n    pass\n", "exactly one function"),
        ("def test_x(client):\n    import os\n", "imports"),
        ("def test_x(client):\n    open('/etc/passwd')\n", "'open'"),
        ("def test_x(client):\n    eval('1')\n", "'eval'"),
        ("def test_x(client):\n    client._service.bug = None\n", "private attribute"),
        ("def test_x(client):\n    global y\n", "global"),
        ("def test_x(client):\n    __builtins__['open']\n", "dunder name"),
        ("def test_x():\n    pass\n", "exactly one argument"),
        ("def test_y(client):\n    pass\n", "named test_x"),
        ("def test_x(client)\n    pass\n", "syntax error"),
    ],
)
def test_safety_allowlist_refuses_unsafe_or_malformed_code(code, reason):
    assert reason in safety_problem(code, "test_x")


def test_safe_code_is_allowed():
    assert (
        safety_problem("def test_x(client):\n    assert client.get('/x').status\n", "test_x") == ""
    )


def test_run_test_reports_pass_fail_and_error():
    ok = _test('    assert client.get("/members/M-1").status == 200')
    bad = _test('    assert client.get("/members/M-1").status == 500')
    crash = _test('    client.get("/members/M-1").json["missing"]')
    assert run_test(ok, ReferralService())[0] == "pass"
    assert run_test(bad, ReferralService())[0] == "fail"
    assert run_test(crash, ReferralService())[0] == "error"


def test_generated_code_runs_without_dangerous_builtins():
    # run_test is called directly here, skipping the AST check, to test the second layer alone
    sneaky = _test("    __builtins__['print']('x')")
    assert run_test(sneaky, ReferralService())[0] == "error"


@pytest.mark.parametrize("bug", sorted(BUGS))
def test_every_mutant_really_changes_behaviour(bug):
    """A mutant identical to the correct service would be unkillable and unfair to the agent."""
    probes = [
        ("POST", "/members/M-1/replies", {"text": "stop"}, None),
        ("GET", "/audit/consent_audit", None, None),
        ("POST", "/outreach/sms", {"member_id": "M-1"}, None),
        ("PUT", "/members/M-1/phone", {"phone": "555010012"}, None),
        ("PUT", "/members/M-1/phone", {"phone": "(555) 010-0199"}, None),
        ("POST", "/referrals/R-404/status", {"status": "X"}, {"Authorization": "tok-valid"}),
        ("POST", "/referrals/R-1/status", {"status": "X"}, {"Authorization": "tok-expired"}),
        ("POST", "/referrals/R-1/status", {"status": "Y"}, {"Authorization": "tok-valid"}),
        ("GET", "/audit/referral_audit", None, None),
    ]

    def trace(service):
        return [vars(service.request(*p)) for p in probes]

    assert trace(ReferralService(bug)) != trace(ReferralService())


def test_contract_examples_hold_on_the_correct_service():
    client = ServiceClient(ReferralService())
    assert client.post("/members/M-1/replies", json={"text": " Stop "}).json["sms_opt_out"]
    assert client.post("/outreach/sms", json={"member_id": "M-1"}).status == 409
    assert client.put("/members/M-1/phone", json={"phone": "555-CALL-NOW"}).json == {
        "error": "phone_invalid"
    }
    assert client.put("/members/M-1/phone", json={"phone": "(555) 010-0199"}).json == {
        "phone": "+15550100199"
    }


class _AlwaysFailing:
    name = "always-failing"

    def complete(self, system, prompt, *, task):
        return json.dumps(
            {"test_name": "test_x", "code": "def test_x(client):\n    assert False\n"}
        )


def test_a_test_that_always_fails_kills_nothing():
    result = runner.eval_test_to_code(_AlwaysFailing(), DATA / "golden")
    assert result["metrics"]["pass_on_correct_rate"] == 0.0
    assert result["metrics"]["mutation_kill_rate"] == 0.0


def test_unsafe_answer_is_refused_and_counted():
    class _Unsafe:
        name = "unsafe"

        def complete(self, system, prompt, *, task):
            code = "def test_x(client):\n    import os\n"
            return json.dumps({"test_name": "test_x", "code": code})

    result = TestToCodeAgent(_Unsafe()).run("C", "criterion", {})
    assert not result.runnable and "imports" in result.problem
    metrics = runner.eval_test_to_code(_Unsafe(), DATA / "golden")["metrics"]
    assert metrics["unsafe_code_count"] == 11 and metrics["runnable_rate"] == 0.0


def test_mock_suite_shows_its_blind_spots():
    result = runner.eval_test_to_code(runner.GuardedLLM(runner.get_provider()), DATA / "golden")
    survivors = {m for m, v in result["mutants"].items() if not v["killed_by"]}
    assert survivors == {"stop_case_sensitive", "no_e164"}
