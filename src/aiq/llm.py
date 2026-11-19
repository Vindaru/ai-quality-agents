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
