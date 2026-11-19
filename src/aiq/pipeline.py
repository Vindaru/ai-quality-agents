"""Stage 6: the orchestrator. Requirements in, a release decision out, with people at the two
points where a machine should not decide alone.

    python -m aiq.pipeline run --candidate R1      # writes reports/release_readiness.md

The chain, all agents from the earlier stages:

1. **Requirements -> scenarios.** Requirements-to-Test agent with its review-and-fix loop.
2. **Gate 1, scenario approval.** A scenario the lint passes may be approved by policy; any
   scenario still flagged after repair needs a named reviewer in ``data/pipeline/approvals.yaml``.
   If anything is still pending, the pipeline stops here and lists what needs review. Nothing
   downstream runs on an unapproved scenario.
3. **Scenarios -> code.** Test-to-Code agent, behind the safety allowlist.
4. **Evaluate the suite.** Pass on the correct service, mutation kills.
5. **Fix.** Fix agent, every patch verified by running it.
6. **Run and triage.** The suite runs on the release candidate; Failure-Triage labels every
   failure; the labels become block, rerun or ship.
7. **Gate 2, release sign-off.** Block and rerun take effect at once, because they are the safe
   direction. Ship (with or without a test fix) needs a person's sign-off for that candidate.

Every model call is traced, and the release-readiness report lists what each stage found, what
is still open, and who approved what.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiq.agents.failure_triage import FailureTriageAgent
from aiq.agents.test_fixer import SuiteTest, TestFixAgent, find_problems, score
from aiq.agents.test_generator import RequirementsToTestAgent, Story, lint_scenario
from aiq.agents.test_to_code import TestToCodeAgent
from aiq.demo.referral_service import BUGS
from aiq.execution import execute, release_decision, to_failure
from aiq.llm import LLM, GuardedLLM, get_provider
from aiq.tracing import Tracer, TracingLLM, summarize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
# The stories the demo referral service implements, so their tests can actually run.
DEMO_STORIES = ["US-01", "US-04", "US-07"]


@dataclass
class ScenarioItem:
    key: str  # story/criterion/index, the id a reviewer approves
    story_id: str
    criterion: str
    scenario: dict[str, Any]
    lint: list[str]
    status: str = "pending"  # approved | rejected | pending
    approved_by: str = ""
    note: str = ""


@dataclass
class BuildResult:
    status: str  # "ready" | "awaiting_scenario_approval"
    scenarios: list[ScenarioItem]
    suite: list[SuiteTest] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    fixes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pending(self) -> list[ScenarioItem]:
        return [s for s in self.scenarios if s.status == "pending"]


@dataclass
class ReleaseResult:
    candidate: str
    bug: str | None
    fault: str | None
    decision: str  # block | rerun | ship_with_test_fix | ship
    status: str  # blocked | rerun | shipped | awaiting_signoff
    signed_off_by: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)


def _suite_metrics(suite: list[SuiteTest]) -> dict[str, Any]:
    caught = set().union(*(t.kills for t in suite)) if suite else set()
    return {
        "tests": len(suite),
        "pass_on_correct": sum(t.passes for t in suite),
        "mutants_caught": len(caught),
        "mutants": len(BUGS),
        "survivors": sorted(set(BUGS) - caught),
    }


class Pipeline:
    def __init__(
        self,
        llm: LLM,
        approvals: dict[str, Any] | None = None,
        *,
        auto_approve_clean: bool = True,
        data_dir: Path = DATA,
    ) -> None:
        self.tracer = Tracer()
        self.llm = GuardedLLM(TracingLLM(llm, self.tracer))
        self.model = llm.name
        self.approvals = approvals or {}
        self.auto_approve_clean = auto_approve_clean
        self.data_dir = data_dir

    # --- stages 1 to 5 ---------------------------------------------------------------------

    def build(self, story_ids: list[str] = DEMO_STORIES) -> BuildResult:
        golden = self.data_dir / "golden"
        raw = yaml.safe_load((golden / "test_generator.yaml").read_text(encoding="utf-8"))
        stories = {s["id"]: s for s in raw["stories"]}
        generator = RequirementsToTestAgent(self.llm)
        items: list[ScenarioItem] = []
        for sid in story_ids:
            story = Story.model_validate(
                {k: v for k, v in stories[sid].items() if k in Story.model_fields}
            )
            criteria = {c.id: c.text for c in story.acceptance_criteria}
            seen: dict[str, int] = {}
            for sc in generator.run(story).scenarios:
                cid = sc.source_criterion_id
                seen[cid] = seen.get(cid, 0) + 1
                items.append(
                    ScenarioItem(
                        key=f"{sid}/{cid}/{seen[cid]}",
                        story_id=sid,
                        criterion=criteria.get(cid, ""),
                        scenario=sc.model_dump(),
                        lint=[p.split(":")[0] for p in lint_scenario(criteria.get(cid, ""), sc)],
                    )
                )
        self._approve(items)
        result = BuildResult("ready", items)
        if result.pending:
            result.status = "awaiting_scenario_approval"
            return result

        coder = TestToCodeAgent(self.llm)
        names: set[str] = set()
        for item in (i for i in items if i.status == "approved"):
            code = coder.run(item.key, item.criterion, item.scenario)
            if not code.runnable:
                result.refused.append(f"{item.key}: {code.problem}")
                continue
            if code.test.test_name in names:  # the same test from two scenarios of one criterion
                continue
            names.add(code.test.test_name)
            passes, kills = score(code.test)
            result.suite.append(SuiteTest(item.key, item.criterion, code.test, kills, passes))
        result.before = _suite_metrics(result.suite)

        fixer = TestFixAgent(self.llm)
        for problem in find_problems(result.suite):
            caught = set().union(*(t.kills for t in result.suite))
            if problem.kind == "surviving_mutant" and problem.target in caught:
                continue
            tries = fixer.fix(problem, result.suite)
            result.fixes.append(
                {
                    "problem": problem.kind,
                    "target": problem.target,
                    "attempts": len(tries),
                    "fixed": tries[-1].accepted,
                    "test_name": tries[-1].test_name,
                    "reason": tries[-1].reason,
                }
            )
        result.after = _suite_metrics(result.suite)
        return result

    def _approve(self, items: list[ScenarioItem]) -> None:
        decisions = self.approvals.get("scenarios") or {}
        reviewer = self.approvals.get("reviewer", "")
        for item in items:
            human = decisions.get(item.key)
            if human:
                item.status = "approved" if human["decision"] == "approve" else "rejected"
                item.approved_by, item.note = reviewer, human.get("note", "")
            elif self.auto_approve_clean and not item.lint:
                item.status, item.approved_by = "approved", "policy: lint clean"

    # --- stages 6 and 7 --------------------------------------------------------------------

    def release(self, build: BuildResult, spec: dict[str, Any]) -> ReleaseResult:
        if build.status != "ready":
            raise ValueError("cannot release: scenarios are still awaiting approval")
        triage = FailureTriageAgent(self.llm)
        tests = [t.test for t in build.suite]
        failed = [r for r in execute(tests, spec["bug"], spec["fault"]) if r.outcome != "pass"]
        triaged = [triage.run(to_failure(r, spec["id"], spec["fault"])) for r in failed]
        decision = release_decision([t.category for t in triaged])
        result = ReleaseResult(
            spec["id"],
            spec["bug"],
            spec["fault"],
            decision,
            {"block": "blocked", "rerun": "rerun"}.get(decision, "awaiting_signoff"),
            failures=[
                {
                    "test": r.test_name,
                    "triaged": t.category,
                    "owner": t.suggested_owner,
                    "message": r.message,
                    "expected": r.expected_category,
                }
                for r, t in zip(failed, triaged, strict=True)
            ],
        )
        if result.status == "awaiting_signoff":
            sign = (self.approvals.get("release") or {}).get(spec["id"])
            if sign and sign.get("decision") == decision:  # sign-off is for this exact decision
                result.status = "shipped"
                result.signed_off_by = self.approvals.get("reviewer", "")
        return result

    def tracing(self) -> dict[str, Any]:
        return summarize(self.tracer.spans)


# --- release-readiness report ----------------------------------------------------------------


def render(
    pipeline: Pipeline, build: BuildResult, release: ReleaseResult | None, mock: bool
) -> str:
    def table(header: list[str], rows: list[list[str]]) -> list[str]:
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return out + ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]

    verdict = {
        "shipped": "**SHIP** (signed off)",
        "awaiting_signoff": "**READY TO SHIP, awaiting sign-off**",
        "blocked": "**BLOCKED**",
        "rerun": "**RERUN** (environment or test data, not the build)",
    }
    status = (
        verdict[release.status]
        if release
        else f"**STOPPED at scenario approval** ({len(build.pending)} scenarios to review)"
    )
    out = [
        "# Release readiness",
        "",
        "Generated by `python -m aiq.pipeline run`. Do not edit by hand.",
        "",
        *table(
            ["", ""],
            [
                ["Candidate", release.candidate if release else "-"],
                ["Model", f"`{pipeline.model}`" + (" (offline mock)" if mock else "")],
                ["Status", status],
            ],
        ),
        "",
        "## 1-2. Scenarios and approval",
        "",
    ]
    counts = {s: sum(i.status == s for i in build.scenarios) for s in ("approved", "rejected")}
    out += [
        f"{len(build.scenarios)} scenarios from {len(DEMO_STORIES)} stories: {counts['approved']} "
        f"approved, {counts['rejected']} rejected, {len(build.pending)} pending. Lint-clean "
        "scenarios are approved by policy; flagged ones need a reviewer.",
        "",
        *table(
            ["Scenario", "Lint", "Status", "By", "Note"],
            [
                [i.key, ", ".join(i.lint) or "clean", i.status, i.approved_by or "-", i.note or "-"]
                for i in build.scenarios
            ],
        ),
    ]
    if build.status != "ready":
        out += ["", "Add a decision for each pending scenario to `data/pipeline/approvals.yaml`."]
        return "\n".join(out) + "\n"
    b, a = build.before, build.after
    out += [
        "",
        "## 3-5. Tests, evaluation and fixes",
        "",
        f"{b['tests']} tests generated"
        + (f"; {len(build.refused)} refused by the safety check" if build.refused else "")
        + ".",
        "",
        *table(
            ["", "As generated", "After the Fix agent"],
            [
                [
                    "Pass on the correct service",
                    f"{b['pass_on_correct']}/{b['tests']}",
                    f"{a['pass_on_correct']}/{a['tests']}",
                ],
                [
                    "Planted bugs caught",
                    f"{b['mutants_caught']}/{b['mutants']}",
                    f"{a['mutants_caught']}/{a['mutants']}",
                ],
                [
                    "Bugs no test catches",
                    ", ".join(b["survivors"]) or "none",
                    ", ".join(a["survivors"]) or "none",
                ],
            ],
        ),  # fmt: skip
    ]
    if build.fixes:
        out += [
            "",
            *table(
                ["Problem", "Target", "Test", "Attempts", "Fixed"],
                [
                    [
                        f["problem"],
                        f["target"],
                        f["test_name"] or "-",
                        str(f["attempts"]),
                        "yes" if f["fixed"] else f"no: {f['reason']}",
                    ]
                    for f in build.fixes
                ],
            ),  # fmt: skip
        ]
    if release:
        out += [
            "",
            "## 6-7. Run, triage and decision",
            "",
            f"Candidate `{release.candidate}`: planted bug `{release.bug or 'none'}`, environment "
            f"`{release.fault or 'healthy'}`. {len(release.failures)} failed tests; decision "
            f"`{release.decision}`; status {verdict[release.status]}"
            + (f", signed off by {release.signed_off_by}." if release.signed_off_by else "."),
            "",
        ]
        if release.failures:
            out += table(
                ["Test", "Triage", "Owner", "Error"],
                [[f["test"], f["triaged"], f["owner"], f["message"]] for f in release.failures],
            )
    spans = pipeline.tracing()
    out += [
        "",
        "## Model usage",
        "",
        *table(
            ["Task", "Calls", "Input tokens", "Output tokens", "Errors", "Invalid JSON"],
            [
                [
                    k,
                    str(v["calls"]),
                    f"{v['input_tokens']:,}",
                    f"{v['output_tokens']:,}",
                    str(v["errors"]),
                    str(v["invalid_json"]),
                ]
                for k, v in spans.items()
            ],
        ),  # fmt: skip
        "",
        "## Open risks",
        "",
    ]
    risks = []
    if mock:
        risks.append("Results come from the offline mock model, not a real one.")
    if a["survivors"]:
        risks.append(f"No test catches: {', '.join(a['survivors'])}.")
    rejected = [i.key for i in build.scenarios if i.status == "rejected"]
    if rejected:
        risks.append(f"Rejected scenarios, not covered by this suite: {', '.join(rejected)}.")
    if build.refused:
        risks.append(f"Generated code refused by the safety check: {'; '.join(build.refused)}.")
    risks.append(
        "Token counts are estimates; the scenario judge is not yet trusted to gate (see the eval "
        "report)."
    )
    out += [f"- {r}" for r in risks]
    return "\n".join(out) + "\n"


def load_approvals(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aiq.pipeline")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--candidate", default="R1", help="release run id from release_runs.yaml")
    parser.add_argument("--approvals", type=Path, default=DATA / "pipeline" / "approvals.yaml")
    parser.add_argument("--out", type=Path, default=REPORTS / "release_readiness.md")
    args = parser.parse_args(argv)
    provider = get_provider()
    pipeline = Pipeline(provider, load_approvals(args.approvals))
    build = pipeline.build()
    release = None
    if build.status == "ready":
        runs = yaml.safe_load((DATA / "golden" / "release_runs.yaml").read_text())["runs"]
        spec = next((r for r in runs if r["id"] == args.candidate), None)
        if spec is None:
            parser.error(f"unknown candidate {args.candidate!r}")
        release = pipeline.release(build, spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(pipeline, build, release, provider.name.startswith("mock")), encoding="utf-8"
    )
    print(f"{release.status if release else build.status}: wrote {args.out}")
    return 0 if release and release.status in {"shipped", "awaiting_signoff"} else 1


if __name__ == "__main__":
    sys.exit(main())
