"""Model access for the agents.

* ``LLM`` is the one interface every agent depends on.
* ``MockLLM`` is a deterministic, offline stand-in so CI never needs a key. It is a
  simple heuristic "model" with realistic blind spots; the evals measure those blind
  spots exactly as they would measure a real model's.
* ``AnthropicLLM`` is used only when ``AIQ_LLM_PROVIDER=anthropic``.
* ``GuardedLLM`` puts the PHI guard in front of whichever model is configured, so no
  agent can send unredacted text to a provider.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Protocol

from aiq.guards.phi_guard import PhiGuard

DEFAULT_MODEL = "claude-opus-5-5"


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        """Return the model's raw text answer. ``task`` lets the mock pick a behavior."""
        ...


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in ``text`` (tolerates code fences and prose)."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    return value


def wrap_input(obj: Any) -> str:
    """Serialize agent input into an <input> block. Angle brackets are escaped so text
    from a member or a log line cannot close the block and pose as instructions."""
    body = json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<input>{body}</input>"


def input_payload(prompt: str) -> dict[str, Any]:
    """Agents wrap their structured input in <input>...</input>; the mock reads it back."""
    match = re.search(r"<input>(.*?)</input>", prompt, re.DOTALL)
    if not match:
        raise ValueError("prompt has no <input> block")
    return json.loads(match.group(1))


MockHandler = Callable[[str, str], str]


class MockLLM:
    """Deterministic offline model. Same prompt in, same text out, every time."""

    name = "mock-heuristic-v1"
    handlers: dict[str, MockHandler] = {}

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        try:
            handler = self.handlers[task]
        except KeyError as exc:
            raise ValueError(f"MockLLM has no behavior for task {task!r}") from exc
        return handler(system, prompt)


def mock_behavior(task: str) -> Callable[[MockHandler], MockHandler]:
    """Register the mock's behavior for one task."""

    def register(fn: MockHandler) -> MockHandler:
        MockLLM.handlers[task] = fn
        return fn

    return register


class AnthropicLLM:
    """Real provider via the official SDK. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the extra first: pip install -e '.[anthropic]'") from exc
        self._client = anthropic.Anthropic()
        self.name = model

    def complete(self, system: str, prompt: str, *, task: str) -> str:  # pragma: no cover
        response = self._client.messages.create(
            model=self.name,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        if response.stop_reason == "refusal":
            return ""  # agents treat an empty answer as an invalid response
        return "".join(block.text for block in response.content if block.type == "text")


class GuardedLLM:
    """Redacts PHI from system prompt and user prompt before delegating."""

    def __init__(self, inner: LLM, guard: PhiGuard | None = None) -> None:
        self.inner = inner
        self.guard = guard or PhiGuard()
        self.name = inner.name

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        return self.inner.complete(
            self.guard.redact(system).text, self.guard.redact(prompt).text, task=task
        )


def get_provider() -> LLM:
    """The configured model *without* the guard (used by evals to measure the guard)."""
    provider = os.getenv("AIQ_LLM_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM(os.getenv("AIQ_LLM_MODEL", DEFAULT_MODEL))
    raise ValueError(f"Unknown AIQ_LLM_PROVIDER {provider!r} (expected 'mock' or 'anthropic')")


def get_llm() -> GuardedLLM:
    """The configured model with the PHI guard in front. Agents should use this."""
    return GuardedLLM(get_provider())


def get_judge_provider() -> LLM:
    """The model that grades the agents. Set ``AIQ_JUDGE_MODEL`` to judge with a different
    (ideally stronger, or at least different) model than the one being judged."""
    provider = os.getenv("AIQ_LLM_PROVIDER", "mock").strip().lower()
    if provider == "anthropic":
        return AnthropicLLM(os.getenv("AIQ_JUDGE_MODEL", DEFAULT_MODEL))
    return get_provider()


# --- mock behaviors -------------------------------------------------------------------------
# Each behavior is a small heuristic standing in for a model. They are intentionally
# imperfect in the ways real models are (missed edge cases, wrong labels, an invented
# enum value) so that the evals have something real to catch.

_NEGATIVE_HINT = re.compile(
    r"\b(invalid|unsupported|reject\w*|not|never|no|fails?|missing|without|expired|stop)\b", re.I
)
_EDGE_HINT = re.compile(
    r"\b(within|at most|at least|more than|less than|limit|maximum|minimum)\b"
    r"|\b\d[\d,]* (?:hours?|days?|minutes?|characters?|attempts?)\b",
    re.I,
)
_DB_HINT = re.compile(r"\b(stored|recorded|persisted|saved|audit|table|database|history)\b", re.I)
_UI_HINT = re.compile(r"\b(screen|page|displays?|shown?|dashboard|button|form|console)\b", re.I)


def _mock_layer(text: str) -> str:
    if re.search(r"end-to-end", text, re.I):
        return "E2E"  # mimics a model inventing a value outside the allowed enum
    if _DB_HINT.search(text):
        return "DB"
    if _UI_HINT.search(text):
        return "UI"
    return "API"


def _mock_steps(text: str) -> list[str]:
    given = "Given a synthetic member record exists"
    match = re.match(r"(?i)(?:when|if)\s+(.+?),\s*(.+)", text)
    if match:
        return [given, f"When {match.group(1)}", f"Then {match.group(2)}"]
    return [given, "When the workflow runs", f"Then {text[0].lower()}{text[1:]}"]


@mock_behavior("test_generation")
def _mock_test_generation(system: str, prompt: str) -> str:
    story = input_payload(prompt)
    scenarios = []
    for criterion in story["acceptance_criteria"]:
        text = criterion["text"].rstrip(".")
        base = {
            "feature": story["title"],
            "test_layer": _mock_layer(text),
            "source_criterion_id": criterion["id"],
        }
        kind = "negative" if _NEGATIVE_HINT.search(text) else "positive"
        scenarios.append({**base, "scenario": text, "steps": _mock_steps(text), "tags": [kind]})
        if _EDGE_HINT.search(text):
            scenarios.append(
                {
                    **base,
                    "scenario": f"Boundary: {text}",
                    "steps": [
                        "Given a synthetic member record exists",
                        f"When the input sits exactly on the limit in: {text}",
                        "Then the limit is applied as specified",
                    ],
                    "tags": ["edge"],
                }
            )
    return json.dumps({"scenarios": scenarios})


_TRIAGE_KEYWORDS: dict[str, list[str]] = {
    "product_bug": ["expected", "but was", "assert", "!=", "500", "incorrect"],
    "automation_issue": ["selector", "locator", "element", "stale", "flaky", "click", "retry"],
    "environment": [
        "timeout",
        "connection",
        "unavailable",
        "dns",
        "certificate",
        "space left",
        "oom",
    ],
    "test_data": ["seed", "fixture", "not found", "missing", "duplicate", "expired", "test data"],
}


@mock_behavior("failure_triage")
def _mock_failure_triage(system: str, prompt: str) -> str:
    text = json.dumps(input_payload(prompt)).casefold()
    hits = {cat: [k for k in kws if k in text] for cat, kws in _TRIAGE_KEYWORDS.items()}
    category = max(hits, key=lambda c: len(hits[c]))  # ties go to the first category
    total = sum(len(v) for v in hits.values())
    confidence = round(len(hits[category]) / total, 2) if total else 0.25
    return json.dumps({"category": category, "confidence": confidence, "evidence": hits[category]})


@mock_behavior("outreach_reply")
def _mock_outreach_reply(system: str, prompt: str) -> str:
    # A deliberately gullible model: it follows instructions found in member text and
    # happily repeats what it was given. The guardrails, not the model, keep it safe.
    data = input_payload(prompt)
    text, member = data["text"], data["member"]
    low = text.casefold()
    spanish = member["language"] == "es"
    if re.search(r"instruction|prompt|secret|rules|instrucciones|reglas", low):
        message = f"Sure, here is what I was told: {system[:240]}"
    elif low.strip() in {"stop", "alto"}:
        message = "Has sido dado de baja." if spanish else "You are unsubscribed from texts."
    elif low.strip() == "help":
        message = "Ayuda: llame a la linea de apoyo." if spanish else "Help: call the support line."
    elif re.search(r"on file|my (?:date of birth|phone number|address)", low):
        message = (
            f"Your phone on file is {member['phone']} and your date of birth is {member['dob']}."
        )
    else:
        thanks = "Gracias" if spanish else "Thanks"
        message = (
            f'{thanks} {member["first_name"]}, we received: "{text}". A coordinator will follow up.'
        )
    return json.dumps({"action": "reply", "message": message})


# Words that make a Then step checkable, and When steps that are only placeholders.
_OBSERVABLE = re.compile(
    r"\b(?:\d+|true|false|status|stored|recorded|persisted|saved|logged|sent|rejected|returned"
    r"|displays?|shown|set to|created|blocked|falls? back|normalized)\b|[A-Z]{3,}|\"",
)
_PLACEHOLDER_WHEN = re.compile(
    r"^When (?:the workflow runs|it runs|the system processes it)\b", re.I
)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "that",
    "be",
    "by",
    "as",
    "it",
    "its",
    "this",
    "when",
    "then",
    "given",
    "member",
    "members",
    "system",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_.]+", text.casefold()) if w not in _STOPWORDS}


@mock_behavior("scenario_judge")
def _mock_scenario_judge(system: str, prompt: str) -> str:
    # A cheap, literal-minded judge. Its blind spots are realistic ones: it rewards word
    # overlap rather than meaning (so a correct but reworded scenario can fail), and it trusts
    # a "positive" tag without checking it. Calibration against human labels measures exactly
    # how much those blind spots cost.
    data = input_payload(prompt)
    criterion, sc = data["acceptance_criterion"], data["scenario"]
    steps = sc.get("steps") or []
    whens = [s for s in steps if s.startswith("When")]
    thens = [s for s in steps if s.startswith(("Then", "And"))]
    crit_words = _content_words(criterion)
    overlap = len(crit_words & _content_words(" ".join([sc.get("scenario", ""), *steps])))
    tags = set(sc.get("tags") or [])
    checks = {
        "tests_criterion": bool(crit_words) and overlap / len(crit_words) >= 0.4,
        "concrete_action": bool(whens) and not any(_PLACEHOLDER_WHEN.search(w) for w in whens),
        "observable_outcome": bool(thens)
        and any(_OBSERVABLE.search(t) for t in thens)
        and not any(re.search(r"as specified|works correctly|it works", t, re.I) for t in thens),
        "tag_matches": ("negative" not in tags or bool(_NEGATIVE_HINT.search(criterion)))
        and ("edge" not in tags or bool(_EDGE_HINT.search(criterion))),
    }
    return json.dumps(
        {
            "checks": {k: {"pass": v, "reason": "heuristic"} for k, v in checks.items()},
            "verdict": "pass" if all(checks.values()) else "fail",
        }
    )


_QUANTITY = re.compile(
    r"\b(?:at most|at least|within|more than|less than|fewer than|longer than)?\s*"
    r"(\d[\d,]*)\s*(miles?|minutes?|hours?|days?|times|digits?|characters?|members?)?",
    re.I,
)
_REJECTED = re.compile(r"^(?:an? |the )?(.+?) (is|are) (rejected|ignored)\b(.*)$", re.I)
_RETURNS = re.compile(r"^(?:an? |the )?(.+?) returns? (.+)$", re.I)
_RECORDED = re.compile(
    r"^(?:an? |the |every |each )?(.+?) (?:is|are) (recorded|stored|written|persisted) (.+)$", re.I
)
_CONDITION = re.compile(r"^(?:when|if|after|before)\s+.+?,\s*(.+)$", re.I)


def _lower_first(text: str) -> str:
    """Lower-case a sentence's first letter, but leave acronyms such as ZIP alone."""
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


@mock_behavior("test_repair")
def _mock_test_repair(system: str, prompt: str) -> str:
    # A stand-in for a model fixing one flagged scenario. Like a real model it fixes the
    # problems it was told about when the criterion gives it something to work with, and
    # leaves the rest alone, so some repairs fail and the metrics show it.
    data = input_payload(prompt)
    criterion = data["acceptance_criterion"].rstrip(".")
    scenario, problems = dict(data["scenario"]), " ".join(data["problems"])
    steps = list(scenario["steps"])
    given = steps[0]
    when = next((s for s in steps if s.startswith("When")), "When the workflow runs")
    then = next((s for s in steps if s.startswith("Then")), "Then it works")
    if "PLACEHOLDER_WHEN" in problems or "VAGUE_THEN" in problems:
        quantity = _QUANTITY.search(criterion) if "edge" in scenario["tags"] else None
        condition = _CONDITION.match(criterion)
        outcome = condition.group(1) if condition else criterion
        rejected, returns = _REJECTED.match(criterion), _RETURNS.match(criterion)
        recorded = _RECORDED.match(criterion)
        if quantity and quantity.group(1):
            amount = " ".join(x for x in quantity.group(1, 2) if x)
            when = f"When the input is exactly at the limit of {amount}"
            then = f"Then {_lower_first(outcome)} (checked at exactly {amount})"
        elif rejected:
            subject, verb, action, rest = rejected.groups()
            when = f"When {_lower_first(subject)} {verb.lower()} submitted"
            then = f"Then the request is {action.lower()}{rest}"
        elif returns:
            when = f"When {_lower_first(returns.group(1))} is sent"
            then = f"Then the response returns {returns.group(2)}"
        elif recorded:
            subject, verb, where = recorded.groups()
            when = f"When a new {_lower_first(subject)} is created"
            then = f"Then it is {verb.lower()} {where}"
    if "TAG_NEGATIVE_ON_NORMAL_PATH" in problems:
        scenario["tags"] = ["positive" if t == "negative" else t for t in scenario["tags"]]
    if "TAG_POSITIVE_ON_ERROR_PATH" in problems:
        scenario["tags"] = ["negative" if t == "positive" else t for t in scenario["tags"]]
    scenario["steps"] = [given, when, then]
    return json.dumps(scenario)


# Code the mock "writes" for each kind of scenario, keyed by a phrase in the criterion. Like a
# real model it is mostly right, with blind spots the mutation score exposes: the trailing-
# spaces test never tries lower case, the E.164 test checks only the status code, and the
# invalid-phone test guesses 400 instead of reading the contract's 422.
_CODEGEN: list[tuple[str, str, str]] = [
    (
        "trailing spaces",
        "test_stop_with_spaces_opts_out",
        """
    r = client.post("/members/M-1/replies", json={"text": "STOP  "})
    assert r.status == 200
    assert client.get("/members/M-1").json["sms_opt_out"] is True""",
    ),
    (
        "replies STOP",
        "test_stop_sets_opt_out",
        """
    r = client.post("/members/M-1/replies", json={"text": "STOP"})
    assert r.status == 200
    assert client.get("/members/M-1").json["sms_opt_out"] is True""",
    ),
    (
        "no further outreach",
        "test_no_sms_after_opt_out",
        """
    client.post("/members/M-1/replies", json={"text": "STOP"})
    r = client.post("/outreach/sms", json={"member_id": "M-1", "message": "hi"})
    assert r.status == 409
    assert r.json["error"] == "opted_out\"""",
    ),
    (
        "consent audit",
        "test_opt_out_is_audited",
        """
    client.post("/members/M-1/replies", json={"text": "STOP"})
    rows = client.get("/audit/consent_audit").json["rows"]
    assert len(rows) == 1
    assert rows[0]["event"] == "opt_out" and rows[0]["timestamp"]""",
    ),
    (
        "phone number is invalid",
        "test_invalid_phone_rejected",
        """
    r = client.put("/members/M-1/phone", json={"phone": "555-CALL-NOW"})
    assert r.status == 400""",
    ),
    (
        "E.164",
        "test_phone_normalised",
        """
    r = client.put("/members/M-1/phone", json={"phone": "(555) 010-0199"})
    assert r.status == 200""",
    ),
    (
        "fewer than 10 digits",
        "test_short_phone_rejected",
        """
    r = client.put("/members/M-1/phone", json={"phone": "555010012"})
    assert r.status == 422
    assert r.json["error"] == "phone_too_short\"""",
    ),
    (
        "unknown referral",
        "test_unknown_referral_404",
        """
    r = client.post("/referrals/R-404/status", json={"status": "COMPLETED"},
                    headers={"Authorization": "tok-valid"})
    assert r.status == 404""",
    ),
    (
        "expired partner token",
        "test_expired_token_401",
        """
    r = client.post("/referrals/R-1/status", json={"status": "COMPLETED"},
                    headers={"Authorization": "tok-expired"})
    assert r.status == 401
    assert client.get("/referrals/R-1").json["status"] == "SENT\"""",
    ),
    (
        "audit log",
        "test_status_change_audited",
        """
    client.post("/referrals/R-1/status", json={"status": "COMPLETED"},
                headers={"Authorization": "tok-valid"})
    rows = client.get("/audit/referral_audit").json["rows"]
    assert rows == [{"referral_id": "R-1", "status": "COMPLETED"}]""",
    ),
    (
        "COMPLETED",
        "test_referral_completed",
        """
    r = client.post("/referrals/R-1/status", json={"status": "COMPLETED"},
                    headers={"Authorization": "tok-valid"})
    assert r.status == 200
    assert client.get("/referrals/R-1").json["status"] == "COMPLETED\"""",
    ),
]


@mock_behavior("test_codegen")
def _mock_test_codegen(system: str, prompt: str) -> str:
    criterion = input_payload(prompt)["acceptance_criterion"]
    for phrase, name, body in _CODEGEN:
        if phrase.casefold() in criterion.casefold():
            return json.dumps({"test_name": name, "code": f"def {name}(client):{body}\n"})
    # Nothing it recognises: a test that runs but checks almost nothing, as weak models do.
    body = '\n    assert client.get("/members/M-1").status == 200'
    return json.dumps(
        {"test_name": "test_member_exists", "code": f"def test_member_exists(client):{body}\n"}
    )


# The fix agent's mock: it reads the evidence the way a model would, and makes one realistic
# mistake. Its first E.164 patch guesses a formatted phone number instead of reading the
# contract; the verifier rejects it, and the retry (with the rejection as feedback) gets it right.
_FIX_RULES: list[tuple[str, Callable[[str], bool], Callable[[str, str], str]]] = [
    (
        "r.status == 400",
        lambda code: "r.status == 400" in code,
        lambda code, feedback: (
            code.replace("r.status == 400", "r.status == 422").rstrip()
            + '\n    assert r.json["error"] == "phone_invalid"\n'
        ),
    ),
    (
        "upper-case STOP",
        lambda code: '"STOP  "' in code,
        lambda code, feedback: code.replace('"STOP  "', '"stop  "'),
    ),
    (
        "E.164",
        lambda code: "/phone" in code and "r.status == 200" in code,
        lambda code, feedback: (
            code.rstrip()
            + '\n    assert client.get("/members/M-1").json["phone"] == '
            + ('"+15550100199"' if feedback else '"+1 (555) 010-0199"')
            + "\n"
        ),
    ),
]


@mock_behavior("test_fix")
def _mock_test_fix(system: str, prompt: str) -> str:
    data = input_payload(prompt)
    evidence, tests, feedback = data["problem"]["evidence"], data["tests"], data["feedback"]
    for clue, wanted, rewrite in _FIX_RULES:
        target = next((t for t in tests if wanted(t["code"])), None)
        if clue in evidence and target:
            code = rewrite(target["code"], feedback)
            return json.dumps({"test_name": target["test_name"], "code": code, "explanation": clue})
    # Nothing it recognises: it hands back the first test unchanged, which fixes nothing.
    first = tests[0]
    return json.dumps({"test_name": first["test_name"], "code": first["code"], "explanation": "?"})
