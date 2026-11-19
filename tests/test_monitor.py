import json
from pathlib import Path

import pytest
import yaml

from aiq.agents.failure_triage import Triage
from aiq.evals import runner
from aiq.llm import MockLLM
from aiq.monitor import alerts_for, compare_runs, profile, psi
from aiq.tracing import Tracer, TracingLLM, summarize

DATA = Path(__file__).parents[1] / "data"
RULES = yaml.safe_load((DATA / "monitor" / "alerts.yaml").read_text())


def _triage(category="environment", routed="rules", confidence=0.9):
    return Triage(
        failure_id="X",
        category=category,
        confidence=confidence,
        evidence=[],
        suggested_owner="o",
        routed_to=routed,
        route_reason="r",
    )


def test_psi_is_zero_for_the_same_mix_and_grows_with_the_shift():
    mix = {"a": 0.5, "b": 0.5}
    assert psi(mix, mix) == 0
    small = psi(mix, {"a": 0.6, "b": 0.4})
    large = psi(mix, {"a": 0.9, "b": 0.1})
    assert 0 < small < 0.1 < 0.25 < large


def test_profile_reads_behaviour_not_labels():
    p = profile(
        [_triage(), _triage(routed="llm", confidence=0.3), _triage("unclassified", "llm", 0.0)],
        low_confidence=0.6,
    )
    assert p["llm_routed_rate"] == pytest.approx(0.6667)
    assert p["low_confidence_rate"] == pytest.approx(0.6667)
    assert p["unclassified_rate"] == pytest.approx(0.3333)


def test_alerts_fire_on_each_signal_and_stay_quiet_on_the_reference():
    reference = profile([_triage(c) for c in ["environment", "test_data"] * 5], 0.6)
    assert alerts_for(reference, reference, RULES) == []
    routed = profile([_triage(routed="llm", confidence=0.3)] * 10, 0.6)
    texts = " ".join(alerts_for(routed, reference, RULES))
    assert "model routing" in texts and "confidence" in texts
    shifted = profile([_triage("product_bug")] * 10, 0.6)
    assert "category mix shifted" in " ".join(alerts_for(shifted, reference, RULES))


def test_compare_runs_follows_the_direction_of_each_threshold():
    thresholds = {"s": {"recall": {"min": 0.8}, "leaks": {"max": 0}}}
    before = {"s": {"recall": 0.90, "leaks": 0}}
    assert compare_runs({"s": {"recall": 0.89, "leaks": 0}}, before, thresholds, 0.02) == []
    worse = compare_runs({"s": {"recall": 0.85, "leaks": 1}}, before, thresholds, 0.02)
    assert {r["metric"] for r in worse} == {"recall", "leaks"}
    better = compare_runs({"s": {"recall": 0.99, "leaks": 0}}, before, thresholds, 0.02)
    assert better == []


def test_history_keeps_one_line_per_commit_and_model(tmp_path):
    path = tmp_path / "history.jsonl"
    for value in (0.5, 0.6):
        runner.append_history(path, {"git_sha": "abc", "model": "m", "metrics": {"v": value}})
    runner.append_history(path, {"git_sha": "def", "model": "m", "metrics": {"v": 0.7}})
    rows = runner.read_history(path)
    assert [(r["git_sha"], r["metrics"]["v"]) for r in rows] == [("abc", 0.6), ("def", 0.7)]


class _Echo:
    name = "echo-model"

    def complete(self, system, prompt, *, task):
        return '{"ok": true}'


class _Broken:
    name = "broken"

    def complete(self, system, prompt, *, task):
        raise TimeoutError("provider timed out")


def test_spans_record_usage_and_errors_but_never_text():
    tracer = Tracer()
    secret = "member phone 813-555-0142"
    TracingLLM(_Echo(), tracer).complete("system v1", secret, task="x")
    with pytest.raises(TimeoutError):
        TracingLLM(_Broken(), tracer).complete("system v1", secret, task="y")
    dumped = json.dumps([s.__dict__ for s in tracer.spans])
    assert "813-555-0142" not in dumped and "system v1" not in dumped
    ok, failed = tracer.spans
    assert ok.attributes["gen_ai.request.model"] == "echo-model"
    assert ok.attributes["aiq.output_json_valid"] is True
    assert failed.attributes["error.type"] == "TimeoutError"
    summary = summarize(tracer.spans)
    assert summary["y"]["errors"] == 1
    assert summary["x"]["prompt_versions"] == summary["y"]["prompt_versions"]


def test_monitor_alerts_on_every_drift_night_and_no_normal_night():
    results = runner.run_all(MockLLM(), DATA)
    days = results["suites"]["monitoring"]["days"]
    assert [d["day"] for d in days if d["alerts"]] == [4, 5, 6]
    assert all(d["review_queue"] == [] for d in days if not d["alerts"])
    assert results["history"]["regressions"] == []


def test_a_deaf_monitor_fails_the_gate(tmp_path):
    """Mutation test: limits so loose that nothing alerts must fail drift_day_recall."""
    import shutil

    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    rules = dict(RULES, llm_routed_rate_rise=1, max_low_confidence_rate=1, max_category_psi=99)
    (data / "monitor" / "alerts.yaml").write_text(yaml.safe_dump(rules))
    results = runner.run_all(MockLLM(), data, history_path=tmp_path / "h.jsonl")
    failed = {r["metric"] for r in results["gate"] if not r["passed"]}
    assert failed == {"drift_day_recall"}
