"""Fix agent: repair broken generated tests and strengthen weak ones.

Two kinds of problem come out of the earlier stages:

* **broken_test**: a generated test fails on the correct service, so it is wrong (stage 2 finds
  these; stage 3 triages them as automation issues).
* **surviving_mutant**: a planted bug that no test catches, so the suite has a blind spot
  (stage 2's mutation score; stage 3 shows the cost as an escaped defect).

The agent proposes a replacement for one existing test. It never decides whether its own patch
is good: ``verify`` does, by running it. A patch is kept only if

1. it passes the same safety allowlist as any generated code,
2. it replaces a test that exists (same name), and passes on the correct service,
3. it still catches every bug the old test caught (a fix may never weaken the suite),
4. it solves the problem: a repaired test must catch at least one bug (a test that asserts
   nothing would also "pass"), and a strengthened test must catch the surviving bug.

A rejected patch goes back to the agent once with the reason, the same review-and-retry shape
as the Requirements-to-Test repair loop.

Test code goes through the PHI guard like any other prompt, so a phone number in a test reaches
the model as ``[REDACTED_PHONE]`` and may come back that way. ``restore_placeholders`` puts the
original values back locally, from the test being replaced, so no value is ever sent to the model
and the patch still runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from aiq.agents.test_to_code import GeneratedTest, run_test, safety_problem
from aiq.demo.referral_service import API_CONTRACT, BUGS, ReferralService
from aiq.guards.phi_guard import scan
from aiq.llm import LLM, extract_json, wrap_input

SYSTEM_PROMPT = f"""You are a senior SDET fixing an automated API test suite.
You get one problem and the current tests. Problem kinds:
- broken_test: the named test fails on a correct build. Fix the test, not the expectation the
  scenario states; read the contract below for the right status codes and fields.
- surviving_mutant: mutation testing found a bug that no test catches. Strengthen the ONE
  existing test that should have caught it, so it fails on that bug.
Rewrite exactly one existing test and keep its name. Never delete or loosen an assertion: the
new test must still catch everything the old one caught. Use only the `client` argument, no
imports. If feedback from a previous attempt is given, it says why that patch was rejected.

{API_CONTRACT}
Return JSON only: {{"test_name": str, "code": str, "explanation": str}}"""


@dataclass
class Problem:
    kind: str  # "broken_test" | "surviving_mutant"
    target: str  # the scenario id of a broken test, or the bug id of a surviving mutant
    evidence: str  # the failure message, or the mutation report


@dataclass
class SuiteTest:
    scenario_id: str
    criterion: str
    test: GeneratedTest
    kills: set[str] = field(default_factory=set)
    passes: bool = True


@dataclass
class Attempt:
    problem: Problem
    test_name: str
    accepted: bool
    reason: str
    kills: list[str] = field(default_factory=list)


class _Patch(BaseModel):
    test_name: str = Field(pattern=r"^test_[a-z0-9_]+$")
    code: str = Field(min_length=10)
    explanation: str = ""


def mutation_report(bug: str) -> str:
    return f"mutation testing: a build where {BUGS[bug]} passes every test in the suite"


def score(test: GeneratedTest) -> tuple[bool, set[str]]:
    """Whether the test passes on the correct service, and which bugs it catches there."""
    if run_test(test, ReferralService())[0] != "pass":
        return False, set()
    return True, {bug for bug in BUGS if run_test(test, ReferralService(bug))[0] != "pass"}


def restore_placeholders(code: str, original: str) -> str:
    """Replace each ``[REDACTED_<TYPE>]`` in ``code``, in order, with the values of that type found
    in ``original``. A placeholder with no value left to restore is left as it is, and the
    patch then fails verification."""
    values: dict[str, list[str]] = {}
    for finding in scan(original):
        values.setdefault(finding.phi_type, []).append(finding.value)
    for phi_type, found in values.items():
        token = f"[REDACTED_{phi_type}]"
        for value in found:
            code = code.replace(token, value, 1)
    return code


def verify(patch: GeneratedTest, suite: list[SuiteTest], problem: Problem) -> tuple[str, set]:
    """Empty reason if the patch may replace its test, otherwise why not."""
    old = next((t for t in suite if t.test.test_name == patch.test_name), None)
    if old is None:
        return f"no test named {patch.test_name} in the suite; rewrite an existing one", set()
    unsafe = safety_problem(patch.code, patch.test_name)
    if unsafe:
        return f"refused by the safety check: {unsafe}", set()
    passes, kills = score(patch)
    if not passes:
        outcome, message = run_test(patch, ReferralService())
        return f"the patched test {outcome}s on the correct service: {message}", set()
    lost = old.kills - kills
    if lost:
        return f"the patch stops catching {sorted(lost)}", kills
    if problem.kind == "broken_test" and not kills:
        return "the patched test passes but catches no bug: it no longer tests anything", kills
    if problem.kind == "surviving_mutant" and problem.target not in kills:
        return f"the patched test still passes when {BUGS[problem.target]}", kills
    return "", kills


class TestFixAgent:
    __test__ = False  # not a pytest class

    def __init__(self, llm: LLM, max_attempts: int = 2) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.model_calls = 0

    def propose(self, problem: Problem, suite: list[SuiteTest], feedback: str) -> Any:
        payload = {
            "problem": {"kind": problem.kind, "evidence": problem.evidence},
            "tests": [
                {"criterion": t.criterion, "test_name": t.test.test_name, "code": t.test.code}
                for t in suite
            ],
            "feedback": feedback,
        }
        self.model_calls += 1
        raw = self.llm.complete(SYSTEM_PROMPT, wrap_input(payload), task="test_fix")
        try:
            patch = _Patch.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            return f"invalid answer: {exc}"[:200]
        return GeneratedTest(test_name=patch.test_name, code=patch.code)

    def fix(self, problem: Problem, suite: list[SuiteTest]) -> list[Attempt]:
        """Try to solve one problem, editing ``suite`` in place when a patch is accepted."""
        attempts: list[Attempt] = []
        feedback = ""
        for _ in range(self.max_attempts):
            patch = self.propose(problem, suite, feedback)
            if isinstance(patch, str):
                attempts.append(Attempt(problem, "", False, patch))
                feedback = patch
                continue
            old = next((t for t in suite if t.test.test_name == patch.test_name), None)
            if old is not None:
                patch.code = restore_placeholders(patch.code, old.test.code)
            reason, kills = verify(patch, suite, problem)
            attempts.append(Attempt(problem, patch.test_name, not reason, reason, sorted(kills)))
            if not reason:
                for item in suite:
                    if item.test.test_name == patch.test_name:
                        item.test, item.kills, item.passes = patch, kills, True
                return attempts
            feedback = reason
        return attempts


def find_problems(suite: list[SuiteTest]) -> list[Problem]:
    """Broken tests first (a broken test cannot catch anything), then surviving mutants."""
    problems = [
        Problem("broken_test", t.scenario_id, run_test(t.test, ReferralService())[1])
        for t in suite
        if not t.passes
    ]
    caught = set().union(*(t.kills for t in suite)) if suite else set()
    problems += [
        Problem("surviving_mutant", b, mutation_report(b)) for b in BUGS if b not in caught
    ]
    return problems
