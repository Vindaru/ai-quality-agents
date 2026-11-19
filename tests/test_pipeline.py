from pathlib import Path

import pytest
import yaml

from aiq import pipeline as pl
from aiq.llm import MockLLM

DATA = Path(__file__).parents[1] / "data"
APPROVALS = pl.load_approvals(DATA / "pipeline" / "approvals.yaml")
RUNS = {
    r["id"]: r for r in yaml.safe_load((DATA / "golden" / "release_runs.yaml").read_text())["runs"]
}


@pytest.fixture(scope="module")
def built():
    p = pl.Pipeline(MockLLM(), APPROVALS)
    return p, p.build()


def test_no_approvals_stops_before_any_code_is_written():
    p = pl.Pipeline(MockLLM(), {})
    build = p.build()
    assert build.status == "awaiting_scenario_approval"
    assert build.suite == [] and "test_codegen" not in p.tracing()
    assert all(i.lint for i in build.pending)  # only flagged scenarios wait for a person
    with pytest.raises(ValueError):
        p.release(build, RUNS["R1"])


def test_policy_never_approves_a_flagged_scenario():
    build = pl.Pipeline(MockLLM(), {}, auto_approve_clean=True).build()
    assert all(not i.lint for i in build.scenarios if i.status == "approved")
    strict = pl.Pipeline(MockLLM(), {}, auto_approve_clean=False).build()
    assert {i.status for i in strict.scenarios} == {"pending"}


def test_rejected_scenarios_get_no_test(built):
    _, build = built
    rejected = {i.key for i in build.scenarios if i.status == "rejected"}
    assert rejected == {"US-04/AC4/1"}
    assert not rejected & {t.scenario_id for t in build.suite}


def test_fix_stage_runs_inside_the_pipeline(built):
    _, build = built
    assert build.before["survivors"] and build.after["survivors"] == []
    assert build.after["pass_on_correct"] == build.after["tests"]


@pytest.mark.parametrize(
    ("candidate", "status"),
    [("R1", "shipped"), ("R2", "blocked"), ("R5", "rerun"), ("R6", "rerun"), ("R7", "blocked")],
)
def test_each_candidate_gets_the_right_outcome(built, candidate, status):
    p, build = built
    assert p.release(build, RUNS[candidate]).status == status


def test_ship_waits_for_a_sign_off_for_that_exact_decision(built):
    _, build = built
    no_signoff = pl.Pipeline(MockLLM(), {"scenarios": APPROVALS["scenarios"]})
    assert no_signoff.release(build, RUNS["R1"]).status == "awaiting_signoff"
    wrong = dict(APPROVALS, release={"R1": {"decision": "ship_with_test_fix"}})
    assert pl.Pipeline(MockLLM(), wrong).release(build, RUNS["R1"]).status == "awaiting_signoff"


def test_cli_writes_the_release_readiness_report(tmp_path):
    out = tmp_path / "rr.md"
    assert pl.main(["run", "--candidate", "R2", "--out", str(out)]) == 1  # blocked
    text = out.read_text()
    assert "**BLOCKED**" in text and "product_bug" in text and "## Open risks" in text
    assert pl.main(["run", "--candidate", "R1", "--out", str(out)]) == 0
    assert "signed off by qa-lead" in out.read_text()
    empty = tmp_path / "none.yaml"
    empty.write_text("{}")
    assert pl.main(["run", "--approvals", str(empty), "--out", str(out)]) == 1
    assert "STOPPED at scenario approval" in out.read_text()
