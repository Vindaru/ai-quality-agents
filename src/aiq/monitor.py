"""Monitoring: is the agent still behaving like it did when it passed its evals?

Two questions, answered without new labels:

1. **Production drift.** In production nobody labels each failure, so accuracy cannot be
   measured day to day. What can be measured is how the agent *behaves*: how often the rules
   could not decide and the model had to, how often the model was unsure, how often it gave up,
   and which categories it is choosing. ``monitor_stream`` compares each day with the profile the
   agent had on its golden set (the reference) and raises an alert when a signal moves past its
   limit. A new kind of failure shows up as more model routing and lower confidence; a bad deploy
   shows up as a shift in the category mix, measured with the population stability index (PSI).
   Alerts send a sample of that day's failures to a person, whose labels then become new golden
   cases.

2. **Regression between runs.** ``compare_runs`` checks each gated metric against the previous
   run in ``reports/history.jsonl``. The gate only asks "is it above the floor?"; this asks "did
   it get worse?", which catches a slide long before it reaches the floor.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from aiq.agents.failure_triage import CATEGORIES, Failure, FailureTriageAgent, Triage

LABELS = [*CATEGORIES, "unclassified"]


def psi(reference: dict[str, float], current: dict[str, float], floor: float = 0.01) -> float:
    """Population stability index between two category distributions (shares that sum to 1).
    Rule of thumb: < 0.10 stable, 0.10-0.25 worth a look, > 0.25 a real shift. A floor keeps an
    empty bucket from dividing by zero."""
    total = 0.0
    for key in set(reference) | set(current):
        ref = max(reference.get(key, 0.0), floor)
        cur = max(current.get(key, 0.0), floor)
        total += (cur - ref) * math.log(cur / ref)
    return round(total, 4)


def profile(triages: list[Triage], low_confidence: float) -> dict[str, Any]:
    """The label-free signals for a batch of triaged failures."""
    n = len(triages)
    counts = Counter(t.category for t in triages)
    return {
        "failures": n,
        "llm_routed_rate": round(sum(t.routed_to == "llm" for t in triages) / n, 4),
        "low_confidence_rate": round(sum(t.confidence < low_confidence for t in triages) / n, 4),
        "unclassified_rate": round(counts["unclassified"] / n, 4),
        "mix": {label: round(counts[label] / n, 4) for label in LABELS},
    }


def alerts_for(day: dict[str, Any], reference: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    alerts = []
    rise = day["llm_routed_rate"] - reference["llm_routed_rate"]
    if rise > rules["llm_routed_rate_rise"]:
        alerts.append(
            f"model routing {day['llm_routed_rate']:.0%} vs {reference['llm_routed_rate']:.0%} "
            "on the golden set: failures the rules do not recognise"
        )
    if day["low_confidence_rate"] > rules["max_low_confidence_rate"]:
        alerts.append(f"{day['low_confidence_rate']:.0%} of answers below confidence threshold")
    if day["unclassified_rate"] > rules["max_unclassified_rate"]:
        alerts.append(f"{day['unclassified_rate']:.0%} unclassified")
    shift = psi(reference["mix"], day["mix"])
    if shift > rules["max_category_psi"]:
        top = max(day["mix"], key=lambda k: day["mix"][k] - reference["mix"].get(k, 0.0))
        alerts.append(f"category mix shifted (PSI {shift:.2f}), mostly toward {top}")
    return alerts


def monitor_stream(
    agent: FailureTriageAgent,
    days: list[dict[str, Any]],
    reference: dict[str, Any],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    """Triage each day's failures and check its signals. Only ``id``, ``test_name``, ``error``,
    ``stack`` and ``logs`` reach the agent and the monitor; the answer key stays out."""
    fields = set(Failure.model_fields)
    out = []
    for day in days:
        triages = [
            agent.run(Failure.model_validate({k: v for k, v in f.items() if k in fields}))
            for f in day["failures"]
        ]
        signals = profile(triages, rules["low_confidence"])
        signals["psi"] = psi(reference["mix"], signals["mix"])
        alerts = alerts_for(signals, reference, rules)
        review = [t.failure_id for t in triages if t.confidence < rules["low_confidence"]]
        out.append(
            {
                "day": day["day"],
                "signals": signals,
                "alerts": alerts,
                "review_queue": review[: rules["review_sample"]] if alerts else [],
                "triages": [t.model_dump() for t in triages],
            }
        )
    return out


def compare_runs(
    current: dict[str, Any],
    previous: dict[str, Any],
    thresholds: dict[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Gated metrics that moved the wrong way by more than ``tolerance`` since ``previous``.
    The direction comes from the threshold: a ``min`` metric should not fall, a ``max`` metric
    should not rise."""
    out = []
    for suite, rules in thresholds.items():
        for metric, bound in rules.items():
            if not isinstance(bound, dict):
                continue
            now = current.get(suite, {}).get(metric)
            before = previous.get(suite, {}).get(metric)
            if now is None or before is None:
                continue
            delta = round(now - before, 4)
            if (delta < -tolerance) if "min" in bound else (delta > tolerance):
                out.append(
                    {"suite": suite, "metric": metric, "before": before, "now": now, "delta": delta}
                )
    return out
