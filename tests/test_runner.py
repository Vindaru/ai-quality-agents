import json
import shutil
from pathlib import Path

import pytest
import yaml

from aiq.evals import runner
from aiq.guards import phi_guard
from aiq.llm import MockLLM

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def results():
    return runner.run_all(MockLLM(), DATA)


def test_runs_every_suite_and_passes_the_gate_on_the_mock(results):
    assert set(results["suites"]) == {
        "test_generator",
        "triage",
        "safety",
        "judge",
        "test_to_code",
        "run_and_triage",
        "monitoring",
        "fix",
        "pipeline",
        "rag",
    }
    assert results["passed"], [r for r in results["gate"] if not r["passed"]]
    assert results["run"]["offline_mock"] is True


def test_mock_runs_are_deterministic(results):
    again = runner.run_all(MockLLM(), DATA)
    for suite in results["suites"]:
        assert again["suites"][suite]["metrics"] == results["suites"][suite]["metrics"]


def test_guardrails_matter_ablation(results):
    guarded = results["suites"]["safety"]["metrics"]
    unguarded = results["suites"]["safety"]["ablation_unguarded"]
    assert guarded["phi_leak_count"] == 0 < unguarded["phi_leak_count"]
    assert guarded["phi_in_prompt_count"] == 0 < unguarded["phi_in_prompt_count"]
    assert guarded["injection_resistance"] > unguarded["injection_resistance"]
    assert results["suites"]["safety"]["redactions"]  # the guard actually redacted something


def test_triage_confusion_matrix_adds_up(results):
    triage = results["suites"]["triage"]
    total = sum(sum(row.values()) for row in triage["confusion_matrix"].values())
    assert total == triage["counts"]["failures"] == 48
    m = triage["metrics"]
    assert m["accuracy_ci95_low"] <= m["accuracy"] <= m["accuracy_ci95_high"]


@pytest.mark.parametrize(
    ("bound", "value", "passed"),
    [({"min": 0.8}, 0.8, True), ({"min": 0.8}, 0.79, False), ({"max": 0}, 1, False)],
)
def test_apply_gate_bounds(bound, value, passed):
    rows = runner.apply_gate({"s": {"metrics": {"m": value}}}, {"s": {"m": bound}})
    assert rows[0]["passed"] is passed


def test_gate_command_fails_when_a_threshold_is_missed(results, tmp_path, monkeypatch):
    reports, outputs, data = tmp_path / "reports", tmp_path / "outputs", tmp_path / "data"
    runner.write_outputs(results, reports, outputs)
    shutil.copytree(DATA, data)
    monkeypatch.setattr(runner, "REPORTS", reports)
    monkeypatch.setattr(runner, "DATA", data)
    assert runner.main(["gate"]) == 0

    thresholds = yaml.safe_load((data / "thresholds.yaml").read_text())
    thresholds["triage"]["accuracy"] = {"min": 0.99}
    (data / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    assert runner.main(["gate"]) == 1


def test_outputs_keep_case_detail_out_of_the_committed_summary(results, tmp_path):
    runner.write_outputs(results, tmp_path / "r", tmp_path / "o")
    summary = json.loads((tmp_path / "r" / "eval_results.json").read_text())
    assert all("cases" not in suite for suite in summary["suites"].values())
    assert (tmp_path / "o" / "triage_cases.jsonl").read_text().count("\n") == 48
    assert "tracer" not in summary
    assert (tmp_path / "o" / "traces.jsonl").read_text().count("\n") == summary["suites"][
        "monitoring"
    ]["metrics"]["model_calls"]
    history = runner.read_history(tmp_path / "r" / "history.jsonl")
    assert [h["git_sha"] for h in history] == [summary["run"]["git_sha"]]


@pytest.mark.parametrize("disabled", ["PHONE", "SSN", "MEMBER_ID"])
def test_safety_eval_catches_a_broken_guard(monkeypatch, disabled):
    """Mutation test: switch off one redaction pattern and the gate must notice."""
    monkeypatch.delitem(phi_guard.PHI_PATTERNS, disabled)
    metrics = runner.eval_safety(MockLLM(), DATA / "golden")["metrics"]
    assert metrics["phi_in_prompt_count"] > 0


def test_repair_loop_improves_judged_quality_without_hurting_recall(results):
    judge = results["suites"]["judge"]["metrics"]
    assert judge["scenario_pass_rate"] > judge["scenario_pass_rate_without_repair"]
    without = runner.eval_test_generator(MockLLM(), DATA / "golden", repair=False)["metrics"]
    with_repair = results["suites"]["test_generator"]["metrics"]
    assert with_repair["expected_scenario_recall"] >= without["expected_scenario_recall"]
    assert with_repair["lint_issue_rate"] < with_repair["lint_flagged_rate"]
