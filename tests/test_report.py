from pathlib import Path

import pytest

from aiq import report
from aiq.evals import runner
from aiq.llm import MockLLM

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def markdown():
    return report.render(runner.run_all(MockLLM(), DATA))


def test_report_has_every_required_section(markdown):
    for heading in [
        "# Eval report",
        "## Results at a glance",
        "## 1. Requirements-to-Test agent",
        "## 2. Failure-Triage agent",
        "**Confusion matrix**",
        "## 3. Outreach safety",
        "## What the evals caught",
        "## Known limitations",
    ]:
        assert heading in markdown


def test_report_names_the_model_and_gate_result(markdown):
    assert "mock-heuristic-v1" in markdown and "deterministic offline mock" in markdown
    assert "**PASSED** (11/11 checks)" in markdown


def test_report_lists_three_to_five_examples(markdown):
    section = markdown.split("## What the evals caught")[1].split("<details>")[0]
    assert 3 <= section.count("\n- **") <= 5


def test_formatting():
    assert report._fmt("accuracy", 0.85416) == "85.4%"
    assert report._fmt("phi_leak_count", 0) == "0"
    assert report._bound("accuracy", {"min": 0.8}) == "≥ 80.0%"


def test_write_reports_creates_both_files(tmp_path):
    runner.write_outputs(runner.run_all(MockLLM(), DATA), tmp_path / "r", tmp_path / "o")
    assert {p.name for p in (tmp_path / "r").iterdir()} == {"eval_results.json", "eval_report.md"}
