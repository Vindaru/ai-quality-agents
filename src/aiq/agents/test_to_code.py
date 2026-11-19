"""Test-to-Code agent: one approved Gherkin scenario -> one runnable pytest-style test.

The generated code is untrusted model output that we are about to execute, so it passes three
gates before it runs:

1. **Shape:** valid Python with exactly one ``def test_*(client)`` function.
2. **Safety:** an AST allowlist. No imports, no ``exec``/``eval``/``open``/``__import__``,
   no dunder attribute access, no ``global``. A test only needs the ``client`` it is given.
3. **Sandbox:** it runs with a minimal set of builtins, against an in-process service, with no
   file system or network to reach.

These layers are defence in depth for an offline demo, not a hard security boundary: Python's
``exec`` cannot be fully sandboxed. Code from a real model should also run in a separate,
locked-down process or container with no credentials, as any CI job running untrusted code would.

How good the test is gets measured outside the agent (see ``eval_test_to_code``): it must pass
on the correct service and fail on deliberately broken versions of it (mutation testing).
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from aiq.demo.referral_service import API_CONTRACT, ReferralService, ServiceClient
from aiq.llm import LLM, extract_json, wrap_input

SYSTEM_PROMPT = f"""You are a senior SDET. Turn ONE approved Gherkin scenario into ONE Python test.
Write a single function `def test_<name>(client):` using plain `assert` statements. Do not
import anything; the `client` argument is all you need. Assert the exact status code AND the
data the scenario's Then step promises (a field value, an audit row, a normalised phone), not
only the status code: a test that cannot fail on a wrong value is not a test.

{API_CONTRACT}
Return JSON only: {{"test_name": str, "code": str}}"""

FORBIDDEN_CALLS = {
    "exec",
    "eval",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "input",
}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "len",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "set",
        "tuple",
        "sorted",
        "any",
        "all",
        "min",
        "max",
        "sum",
        "range",
        "enumerate",
        "zip",
        "isinstance",
        "AssertionError",
        "KeyError",
        "ValueError",
        "True",
        "False",
        "None",
    )  # fmt: skip
}


class GeneratedTest(BaseModel):
    test_name: str = Field(pattern=r"^test_[a-z0-9_]+$")
    code: str = Field(min_length=10)


@dataclass
class CodeResult:
    scenario_id: str
    test: GeneratedTest | None
    problem: str = ""  # why the code was refused; empty when it may run

    @property
    def runnable(self) -> bool:
        return self.test is not None and not self.problem


def safety_problem(code: str, test_name: str) -> str:
    """Empty string if the code may run, otherwise the first reason it may not."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != test_name:
        return f"code must be exactly one function named {test_name}"
    if [a.arg for a in functions[0].args.args] != ["client"]:
        return "the test function must take exactly one argument, client"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            return "imports are not allowed"
        if isinstance(node, ast.Global | ast.Nonlocal):
            return "global and nonlocal are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return f"access to private attribute {node.attr!r} is not allowed"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"use of dunder name {node.id!r} is not allowed"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return f"use of {node.id!r} is not allowed"
    return ""


def run_test(test: GeneratedTest, service: ReferralService) -> tuple[str, str]:
    """Run one generated test against one service. Returns (outcome, message) with outcome
    pass, fail (an assertion failed) or error (the test itself crashed)."""
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(test.code, f"<{test.test_name}>", "exec"), namespace)  # noqa: S102 - vetted
        namespace[test.test_name](ServiceClient(service))
    except AssertionError as exc:
        return "fail", str(exc)[:200] or "assertion failed"
    except Exception as exc:  # noqa: BLE001 - any crash in generated code is a test error
        return "error", f"{type(exc).__name__}: {exc}"[:200]
    return "pass", ""


class TestToCodeAgent:
    __test__ = False  # not a pytest test class, despite the name

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, scenario_id: str, criterion: str, scenario: dict[str, Any]) -> CodeResult:
        prompt = wrap_input({"acceptance_criterion": criterion, "scenario": scenario})
        raw = self.llm.complete(SYSTEM_PROMPT, prompt, task="test_codegen")
        try:
            test = GeneratedTest.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            return CodeResult(scenario_id, None, f"invalid answer: {exc}"[:200])
        return CodeResult(scenario_id, test, safety_problem(test.code, test.test_name))
