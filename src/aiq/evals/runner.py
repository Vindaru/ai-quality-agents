"""Run every golden set, score it, and enforce the quality gate.

python -m aiq.evals.runner run    # run evals, write reports/ and outputs/
python -m aiq.evals.runner gate   # exit 1 if any threshold in data/thresholds.yaml is missed
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from aiq.agents.failure_triage import CATEGORIES, Failure, FailureTriageAgent, Triage
from aiq.agents.outreach_assistant import CANARY, Member, OutreachAssistant
from aiq.agents.test_fixer import SuiteTest, TestFixAgent, find_problems, score
from aiq.agents.test_generator import RequirementsToTestAgent, Story, lint_scenario
from aiq.agents.test_to_code import GeneratedTest, TestToCodeAgent, run_test
from aiq.demo.referral_service import BUGS, ReferralService
from aiq.evals import scorers
from aiq.evals.judge import CHECKS, JUDGE_VERSION, ScenarioJudge
from aiq.evals.rag_judge import CHECKS as RAG_CHECKS
from aiq.evals.rag_judge import RAG_JUDGE_VERSION, RagJudge
from aiq.execution import execute, release_decision, to_failure
from aiq.guards.phi_guard import scan
from aiq.llm import LLM, GuardedLLM, get_judge_provider, get_provider
from aiq.monitor import compare_runs, monitor_stream, profile
from aiq.pipeline import Pipeline, load_approvals
from aiq.rag.bot import RagBot
from aiq.rag.retriever import BM25Retriever, chunk_docs
from aiq.report import write_reports
from aiq.tracing import Tracer, TracingLLM, summarize

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


class RecordingLLM:
    """Wraps a model and keeps every prompt it receives, so evals can prove what the
    model did (and did not) see."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.name = inner.name
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        self.prompts.append(f"{system}\n{prompt}")
        return self.inner.complete(system, prompt, task=task)


def _load(name: str, data_dir: Path) -> Any:
    return yaml.safe_load((data_dir / name).read_text(encoding="utf-8"))


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


# --- test generator --------------------------------------------------------------------------


def eval_test_generator(
    llm: LLM, data_dir: Path = DATA / "golden", *, repair: bool = True
) -> dict[str, Any]:
    agent = RequirementsToTestAgent(llm, repair=repair)
    stories = _load("test_generator.yaml", data_dir)["stories"]
    n = dict.fromkeys(
        [
            "raw",
            "valid",
            "traced",
            "redundant",
            "criteria",
            "covered",
            "expected",
            "matched",
            "layer_ok",
            "with_negative",
            "risk_gaps",
            "lint_flagged",
            "repairs_accepted",
            "repairs_clean",
            "lint_left",
            "model_calls",
        ],
        0,
    )
    findings, cases = [], []
    for raw_story in stories:
        story = Story.model_validate(raw_story)
        result = agent.run(story)
        ids = [c.id for c in story.acceptance_criteria]
        sources = [s.source_criterion_id for s in result.scenarios]
        n["raw"] += result.raw_count
        n["valid"] += len(result.scenarios)
        n["traced"] += sum(s in ids for s in sources)
        n["redundant"] += len({j for _, j in result.duplicate_pairs})
        n["criteria"] += len(ids)
        n["covered"] += sum(c in set(sources) for c in ids)
        n["with_negative"] += any({"negative", "edge"} & set(s.tags) for s in result.scenarios)
        n["risk_gaps"] += sum(i.code == "MISSING_RISK_CASE" for i in result.issues)
        n["model_calls"] += result.model_calls
        n["lint_flagged"] += len(result.repairs)
        n["repairs_accepted"] += sum(r["accepted"] for r in result.repairs)
        n["repairs_clean"] += sum(r["accepted"] and not r["after"] for r in result.repairs)
        texts = {c.id: c.text for c in story.acceptance_criteria}
        n["lint_left"] += sum(
            bool(lint_scenario(texts[sc.source_criterion_id], sc))
            for sc in result.scenarios
            if sc.source_criterion_id in texts
        )
        for r in result.repairs:
            if not (r["accepted"] and not r["after"]):
                findings.append(
                    {
                        "case": f"{story.id}/{r['criterion_id']}",
                        "type": "repair_incomplete",
                        "detail": f"still {', '.join(p.split(':')[0] for p in r['after'])}",
                    }
                )
        for bad in result.invalid:
            findings.append({"case": story.id, "type": "schema", "detail": bad["error"]})
        for i, j in result.duplicate_pairs:
            a, b = result.scenarios[i], result.scenarios[j]
            findings.append(
                {
                    "case": story.id,
                    "type": "near_duplicate",
                    "detail": f"{a.source_criterion_id} and {b.source_criterion_id} "
                    "produce the same test",
                }
            )
        for exp in raw_story["expected"]:
            n["expected"] += 1
            hits = [
                s
                for s in result.scenarios
                if s.source_criterion_id == exp["criterion"]
                and scorers.contains_all(s.text, exp["keywords"])
                and (exp["kind"] == "positive" or exp["kind"] in s.tags)
            ]
            where = f"{story.id}/{exp['criterion']}"
            if not hits:
                findings.append(
                    {
                        "case": where,
                        "type": "missed_scenario",
                        "detail": f"no {exp['kind']} scenario covering {exp['keywords']}",
                    }
                )
                continue
            n["matched"] += 1
            if any(s.test_layer == exp["layer"] for s in hits):
                n["layer_ok"] += 1
            else:
                findings.append(
                    {
                        "case": where,
                        "type": "wrong_layer",
                        "detail": f"expected {exp['layer']}, "
                        f"got {sorted({s.test_layer for s in hits})}",
                    }
                )
        cases.append(
            {
                "story": story.id,
                "scenarios": [s.model_dump() for s in result.scenarios],
                "invalid": result.invalid,
                "issues": [vars(i) for i in result.issues],
            }
        )
    metrics = {
        "schema_validity": _ratio(n["valid"], n["raw"]),
        "criteria_coverage": _ratio(n["covered"], n["criteria"]),
        "traceability": _ratio(n["traced"], n["valid"]),
        "duplicate_rate": _ratio(n["redundant"], n["valid"]),
        "expected_scenario_recall": _ratio(n["matched"], n["expected"]),
        "layer_accuracy": _ratio(n["layer_ok"], n["matched"]),
        "stories_with_negative_case": _ratio(n["with_negative"], len(stories)),
        "risk_checklist_gaps": n["risk_gaps"],
        "lint_flagged_rate": _ratio(n["lint_flagged"], n["valid"]),
        "repair_success_rate": _ratio(n["repairs_clean"], n["lint_flagged"]),
        "lint_issue_rate": _ratio(n["lint_left"], n["valid"]),
        "model_calls": n["model_calls"],
    }
    counts = {
        "stories": len(stories),
        "scenarios": n["raw"],
        "expected": n["expected"],
        "matched": n["matched"],
    }
    return {"metrics": metrics, "counts": counts, "findings": findings, "cases": cases}


# --- triage ----------------------------------------------------------------------------------


def eval_triage(llm: LLM, data_dir: Path = DATA / "golden") -> dict[str, Any]:
    agent = FailureTriageAgent(llm)
    failures = _load("triage.yaml", data_dir)["failures"]
    y_true, y_pred, cases, findings = [], [], [], []
    for raw in failures:
        label = raw["label"]
        triage = agent.run(Failure.model_validate({k: v for k, v in raw.items() if k != "label"}))
        y_true.append(label)
        y_pred.append(triage.category)
        cases.append({"label": label, **triage.model_dump()})
        if triage.category != label:
            findings.append(
                {
                    "case": raw["id"],
                    "type": "misclassified",
                    "detail": f"{label} predicted as {triage.category} via {triage.routed_to} "
                    f"({triage.route_reason}); evidence {triage.evidence}",
                }
            )
    cm = scorers.classification_metrics(y_true, y_pred, CATEGORIES)
    low, high = scorers.wilson_interval(cm["correct"], cm["total"])

    def _route_accuracy(route: str) -> float:
        picked = [c for c in cases if c["routed_to"] == route]
        return _ratio(sum(c["label"] == c["category"] for c in picked), len(picked))

    routed = sum(c["routed_to"] == "llm" for c in cases)
    metrics = {
        "accuracy": round(cm["accuracy"], 4),
        "accuracy_ci95_low": round(low, 4),
        "accuracy_ci95_high": round(high, 4),
        "macro_f1": round(sum(p["f1"] for p in cm["per_class"].values()) / len(CATEGORIES), 4),
        "llm_routed_rate": _ratio(routed, len(cases)),
        "rules_accuracy": _route_accuracy("rules"),
        "llm_accuracy": _route_accuracy("llm"),
    }
    return {
        "metrics": metrics,
        "counts": {"failures": len(cases), "routed_to_llm": routed, "correct": cm["correct"]},
        "per_class": cm["per_class"],
        "confusion_matrix": cm["confusion_matrix"],
        "findings": findings,
        "cases": cases,
    }


# --- safety ----------------------------------------------------------------------------------


def _safety_pass(provider: LLM, guarded: bool, data: dict[str, Any]) -> dict[str, Any]:
    recorder = RecordingLLM(provider)
    guarded_llm = GuardedLLM(recorder)
    agent = OutreachAssistant(guarded_llm if guarded else recorder, guarded=guarded)
    # Planted PHI comes from the golden data, not from the guard's regexes, so the eval
    # still sees a leak when the guard itself is broken.
    identifiers = ("phone", "dob", "member_id")
    all_planted = {m[k] for m in data["members"].values() for k in identifiers}
    all_planted |= {v for c in data["cases"] for v in c.get("phi", [])}
    cases, findings = [], []
    for case in data["cases"]:
        member = Member.model_validate(data["members"][case["member"]])
        decision = agent.run(member, case["text"])
        planted = [member.phone, member.dob, member.member_id, *case.get("phi", [])]
        leaks = scorers.phi_leaks(decision.message, planted)
        resisted = scorers.injection_resisted(
            decision.message, [CANARY, *case.get("forbidden", [])]
        )
        correct = scorers.exact_match(case["expected_action"], decision.action)
        cases.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "expected": case["expected_action"],
                "action": decision.action,
                "flags": decision.flags,
                "correct": correct,
                "phi_leaks": sorted({f.phi_type for f in leaks}),
                "resisted": resisted,
                "message": decision.message if guarded else "(unguarded output not stored)",
            }
        )
        if not correct or leaks or not resisted:
            detail = f"expected {case['expected_action']}, got {decision.action}"
            if leaks:
                detail += f"; PHI types leaked {sorted({f.phi_type for f in leaks})}"
            if not resisted:
                detail += "; confidential or forbidden text in reply"
            findings.append(
                {"case": case["id"], "type": f"safety_{case['kind']}", "detail": detail}
            )
        elif guarded and any(f.endswith("_blocked") for f in decision.flags):
            findings.append(
                {
                    "case": case["id"],
                    "type": "caught_by_output_guard",
                    "detail": f"input checks missed it; output guard blocked it ({decision.flags})",
                }
            )
    injections = [c for c in cases if c["kind"] == "injection"]
    should_reply = [c for c in cases if c["expected"] == "reply"]
    metrics = {
        "injection_resistance": _ratio(sum(c["resisted"] for c in injections), len(injections)),
        "phi_leak_count": sum(bool(c["phi_leaks"]) for c in cases),
        "phi_in_prompt_count": sum(
            bool(scan(p)) or any(v in p for v in all_planted) for p in recorder.prompts
        ),
        "refusal_correctness": _ratio(sum(c["correct"] for c in cases), len(cases)),
        "false_refusals": sum(c["action"] != "reply" for c in should_reply),
        "model_calls": len(recorder.prompts),
    }
    redactions = guarded_llm.guard.total_blocked
    return {"metrics": metrics, "redactions": redactions, "findings": findings, "cases": cases}


def eval_safety(provider: LLM, data_dir: Path = DATA / "golden") -> dict[str, Any]:
    data = _load("outreach_safety.yaml", data_dir)
    guarded = _safety_pass(provider, True, data)
    unguarded = _safety_pass(provider, False, data)
    return {
        "metrics": guarded["metrics"],
        "ablation_unguarded": unguarded["metrics"],
        "counts": {"cases": len(data["cases"])},
        "redactions": guarded["redactions"],
        "findings": guarded["findings"],
        "cases": guarded["cases"],
    }


# --- test to code ----------------------------------------------------------------------------


def eval_test_to_code(llm: LLM, data_dir: Path = DATA / "golden") -> dict[str, Any]:
    """Generate a test per approved scenario, run each against the correct demo service and
    against every mutant. A test only counts toward the kill score if it passes on the correct
    service: a test that always fails would "kill" every mutant and prove nothing."""
    agent = TestToCodeAgent(llm)
    scenarios = _load("test_to_code.yaml", data_dir)["scenarios"]
    cases, findings, good_tests = [], [], []
    for item in scenarios:
        result = agent.run(item["id"], item["criterion"], item["scenario"])
        case: dict[str, Any] = {
            "id": item["id"],
            "test_name": result.test.test_name if result.test else None,
            "code": result.test.code if result.test else None,
            "refused": result.problem,
        }
        if not result.runnable:
            findings.append({"case": item["id"], "type": "code_refused", "detail": result.problem})
            cases.append(case)
            continue
        outcome, message = run_test(result.test, ReferralService())
        case["on_correct_service"] = outcome
        if outcome != "pass":
            findings.append(
                {
                    "case": item["id"],
                    "type": "test_fails_on_correct_service",
                    "detail": f"{result.test.test_name} {outcome}s on the correct service"
                    + (f": {message}" if message else ""),
                }
            )
        else:
            case["kills"] = [
                bug for bug in BUGS if run_test(result.test, ReferralService(bug))[0] != "pass"
            ]
            good_tests.append(case)
        cases.append(case)

    killed_by = {bug: [c["id"] for c in good_tests if bug in c["kills"]] for bug in BUGS}
    for bug, killers in killed_by.items():
        if not killers:
            findings.append(
                {
                    "case": bug,
                    "type": "mutant_survived",
                    "detail": f"no generated test fails when {BUGS[bug]}",
                }
            )
    runnable = [c for c in cases if not c["refused"]]
    metrics = {
        "runnable_rate": _ratio(len(runnable), len(cases)),
        "unsafe_code_count": sum(
            c["refused"].startswith(("imports", "use of", "access to", "global")) for c in cases
        ),
        "pass_on_correct_rate": _ratio(len(good_tests), len(runnable)),
        "mutation_kill_rate": _ratio(sum(bool(k) for k in killed_by.values()), len(BUGS)),
    }
    return {
        "metrics": metrics,
        "counts": {"scenarios": len(cases), "mutants": len(BUGS), "good_tests": len(good_tests)},
        "mutants": {bug: {"bug": BUGS[bug], "killed_by": k} for bug, k in killed_by.items()},
        "findings": findings,
        "cases": cases,
    }


# --- run and triage --------------------------------------------------------------------------


def eval_run_and_triage(
    llm: LLM, code_cases: list[dict[str, Any]], data_dir: Path = DATA / "golden"
) -> dict[str, Any]:
    """Stage 3: run the stage-2 suite on each release candidate, triage every failure, and turn
    the triage into a release decision. Scores the triage against the injected cause and the
    decision against the decision the true causes call for."""
    agent = FailureTriageAgent(llm)
    tests = [
        GeneratedTest(test_name=c["test_name"], code=c["code"])
        for c in code_cases
        if c.get("code") and not c.get("refused")
    ]
    runs = _load("release_runs.yaml", data_dir)["runs"]
    y_true, y_pred, cases, findings = [], [], [], []
    decisions_ok = escaped = 0
    for spec in runs:
        results = execute(tests, spec["bug"], spec["fault"])
        failed = [r for r in results if r.outcome != "pass"]
        triaged = [agent.run(to_failure(r, spec["id"], spec["fault"])) for r in failed]
        for r, t in zip(failed, triaged, strict=True):
            y_true.append(r.expected_category)
            y_pred.append(t.category)
            if t.category != r.expected_category:
                findings.append(
                    {
                        "case": f"{spec['id']}/{r.test_name}",
                        "type": "pipeline_misrouted",
                        "detail": f"{r.expected_category} triaged as {t.category} via "
                        f"{t.routed_to} ({t.route_reason}); error {r.message!r}",
                    }
                )
        decision = release_decision([t.category for t in triaged])
        expected = release_decision([r.expected_category for r in failed])
        decisions_ok += decision == expected
        if spec["bug"] and decision.startswith("ship"):
            escaped += 1
            findings.append(
                {
                    "case": spec["id"],
                    "type": "escaped_defect",
                    "detail": f"build with bug {spec['bug']} would ship: no generated test failed",
                }
            )
        cases.append(
            {
                "id": spec["id"],
                "bug": spec["bug"],
                "fault": spec["fault"],
                "failed": len(failed),
                "decision": decision,
                "expected_decision": expected,
                "failures": [
                    {
                        "test": r.test_name,
                        "expected": r.expected_category,
                        "triaged": t.category,
                        "routed_to": t.routed_to,
                    }
                    for r, t in zip(failed, triaged, strict=True)
                ],
            }
        )
    cm = scorers.classification_metrics(y_true, y_pred, CATEGORIES)
    metrics = {
        "triage_accuracy": round(cm["accuracy"], 4),
        "decision_accuracy": _ratio(decisions_ok, len(runs)),
        "escaped_defect_count": escaped,
        "failures_triaged": len(y_true),
    }
    return {
        "metrics": metrics,
        "counts": {"runs": len(runs), "tests": len(tests), "failures": len(y_true)},
        "confusion_matrix": cm["confusion_matrix"],
        "runs": [{k: v for k, v in c.items() if k != "failures"} for c in cases],
        "findings": findings,
        "cases": cases,
    }


# --- LLM judge -------------------------------------------------------------------------------


def eval_judge(
    judge_llm: LLM,
    generated: list[dict[str, Any]],
    data_dir: Path = DATA / "golden",
    judge_dir: Path = DATA / "judge",
    baseline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate the scenario judge against human labels, then use it to grade every scenario
    the Requirements-to-Test agent produced. The second number only means something while the
    first one passes its gate."""
    judge = ScenarioJudge(judge_llm)
    calibration = _load("calibration.yaml", judge_dir)["items"]
    human, machine, cases, findings = [], [], [], []
    per_check_agree = dict.fromkeys(CHECKS, 0)
    errors = 0
    for item in calibration:
        result = judge.grade(item["criterion"], item["scenario"])
        truth = "pass" if all(item["human"][c] for c in CHECKS) else "fail"
        errors += result.verdict == "error"
        verdict = "fail" if result.verdict == "error" else result.verdict
        human.append(truth)
        machine.append(verdict)
        disagreed = [c for c in CHECKS if result.checks.get(c) != item["human"][c]]
        for c in CHECKS:
            per_check_agree[c] += c not in disagreed
        cases.append(
            {
                "id": item["id"],
                "source": item["source"],
                "human": truth,
                "judge": result.verdict,
                "checks_disagreed": disagreed,
                "reasons": result.reasons,
                "error": result.error,
            }
        )
        if verdict != truth:
            findings.append(
                {
                    "case": item["id"],
                    "type": "judge_false_pass" if verdict == "pass" else "judge_false_fail",
                    "detail": f"human {truth}, judge {verdict}; checks disagreed: "
                    f"{', '.join(disagreed) or 'none'}"
                    + (f" ({item['note']})" if item.get("note") else ""),
                }
            )

    n = len(calibration)
    agree = sum(h == m for h, m in zip(human, machine, strict=True))
    low, high = scorers.wilson_interval(agree, n)
    # The judge's job in a gate is to catch bad scenarios, so "fail" is the positive class.
    cm = scorers.classification_metrics(human, machine, ["pass", "fail"])
    fail = cm["per_class"]["fail"]

    stories = {s["id"]: s for s in _load("test_generator.yaml", data_dir)["stories"]}

    def grade_all(cases_to_grade: list[dict[str, Any]]) -> tuple[int, int, dict[str, int]]:
        graded = passed = 0
        check_fails = dict.fromkeys(CHECKS, 0)
        for case in cases_to_grade:
            criteria = {c["id"]: c["text"] for c in stories[case["story"]]["acceptance_criteria"]}
            for scenario in case["scenarios"]:
                criterion = criteria.get(scenario["source_criterion_id"])
                if criterion is None:  # untraced scenarios are already counted by traceability
                    continue
                result = judge.grade(criterion, scenario)
                graded += 1
                passed += result.passed
                for c, ok in result.checks.items():
                    check_fails[c] += not ok
                scenario["judge"] = {"verdict": result.verdict, "checks": result.checks}
        return graded, passed, check_fails

    graded, passed, check_fails = grade_all(generated)
    base_graded, base_passed, _ = grade_all(baseline) if baseline is not None else (0, 0, {})

    metrics = {
        "agreement": _ratio(agree, n),
        "agreement_ci95_low": round(low, 4),
        "agreement_ci95_high": round(high, 4),
        "cohen_kappa": round(scorers.cohen_kappa(human, machine), 4),
        "fail_precision": round(fail["precision"], 4),
        "fail_recall": round(fail["recall"], 4),
        "judge_error_count": errors,
        "scenario_pass_rate": _ratio(passed, graded),
        "scenario_pass_rate_without_repair": _ratio(base_passed, base_graded),
    }
    return {
        "metrics": metrics,
        "counts": {
            "calibration_items": n,
            "human_pass": human.count("pass"),
            "scenarios_graded": graded,
            "scenarios_passed": passed,
            "scenarios_passed_without_repair": base_passed,
            "scenarios_graded_without_repair": base_graded,
        },
        "judge": {"version": JUDGE_VERSION, "model": judge_llm.name},
        "per_check_agreement": {c: _ratio(v, n) for c, v in per_check_agree.items()},
        "scenario_check_fail_rate": {c: _ratio(v, graded) for c, v in check_fails.items()},
        "confusion_matrix": cm["confusion_matrix"],
        "findings": findings,
        "cases": cases,
    }


# --- fix ----------------------------------------------------------------------------------


def eval_fix(
    llm: LLM, code_cases: list[dict[str, Any]], data_dir: Path = DATA / "golden"
) -> dict[str, Any]:
    """Stage 5: hand every broken test and every surviving mutant from stage 2 to the Fix agent,
    keep only patches the verifier accepts, then measure the suite again: on the correct service,
    against every mutant, and on the stage-3 release runs."""
    criteria = {s["id"]: s["criterion"] for s in _load("test_to_code.yaml", data_dir)["scenarios"]}
    suite = []
    for c in code_cases:
        if not c.get("code") or c.get("refused"):
            continue
        test = GeneratedTest(test_name=c["test_name"], code=c["code"])
        passes, kills = score(test)
        suite.append(SuiteTest(c["id"], criteria[c["id"]], test, kills, passes))
    before = {t.test.test_name: set(t.kills) for t in suite}
    problems = find_problems(suite)
    agent = TestFixAgent(llm)
    attempts, findings, fixed = [], [], 0
    for problem in problems:
        caught = set().union(*(t.kills for t in suite))
        if problem.kind == "surviving_mutant" and problem.target in caught:
            fixed += 1  # an earlier fix already catches it
            continue
        tries = agent.fix(problem, suite)
        attempts += tries
        fixed += tries[-1].accepted
        for t in tries:
            if not t.accepted:
                findings.append(
                    {
                        "case": problem.target,
                        "type": "patch_rejected",
                        "detail": f"{t.test_name or 'no test'}: {t.reason}",
                    }
                )
        if not tries[-1].accepted:
            findings.append(
                {"case": problem.target, "type": "fix_failed", "detail": problem.evidence}
            )
    # Re-measure from scratch rather than trusting the verifier's own bookkeeping.
    rescored = {t.test.test_name: score(t.test) for t in suite}
    caught_after = set().union(*(k for _, k in rescored.values()))
    kills_lost = sum(len(before[name] - kills) for name, (_, kills) in rescored.items())
    tests = [t.test for t in suite]
    escaped = 0
    for spec in _load("release_runs.yaml", data_dir)["runs"]:
        failed = [r for r in execute(tests, spec["bug"], spec["fault"]) if r.outcome != "pass"]
        decision = release_decision([r.expected_category for r in failed])
        escaped += bool(spec["bug"]) and decision.startswith("ship")
    metrics = {
        "problems": len(problems),
        "fix_rate": _ratio(fixed, len(problems)),
        "patches_rejected": sum(not a.accepted for a in attempts),
        "kills_lost": kills_lost,
        "pass_on_correct_rate_after": _ratio(sum(p for p, _ in rescored.values()), len(suite)),
        "mutation_kill_rate_after": _ratio(len(caught_after), len(BUGS)),
        "escaped_defect_count_after": escaped,
        "model_calls": agent.model_calls,
    }
    return {
        "metrics": metrics,
        "counts": {"tests": len(suite), "problems": len(problems), "attempts": len(attempts)},
        "attempts": [
            {
                "problem": a.problem.kind,
                "target": a.problem.target,
                "test_name": a.test_name,
                "accepted": a.accepted,
                "reason": a.reason,
            }
            for a in attempts
        ],
        "findings": findings,
        "cases": [
            {"id": t.scenario_id, "test_name": t.test.test_name, "code": t.test.code} for t in suite
        ],
    }


# --- pipeline -------------------------------------------------------------------------------


def eval_pipeline(llm: LLM, data_dir: Path = DATA) -> dict[str, Any]:
    """Stage 6: run the whole chain on the demo stories and every release candidate. Scores the
    final decision against the one the injected causes call for, and checks that both human
    gates hold: no approvals means the pipeline stops before writing code, and no sign-off means
    nothing ships."""
    approvals = load_approvals(data_dir / "pipeline" / "approvals.yaml")
    runs = _load("release_runs.yaml", data_dir / "golden")["runs"]
    pipeline = Pipeline(llm, approvals, data_dir=data_dir)
    build = pipeline.build()
    findings: list[dict[str, Any]] = []
    rows = []
    for spec in runs:
        release = pipeline.release(build, spec)
        expected = release_decision([f["expected"] for f in release.failures])
        rows.append(
            {
                "id": spec["id"],
                "bug": spec["bug"],
                "fault": spec["fault"],
                "decision": release.decision,
                "expected_decision": expected,
                "status": release.status,
                "signed_off_by": release.signed_off_by,
            }
        )
        if release.decision != expected:
            findings.append(
                {
                    "case": spec["id"],
                    "type": "pipeline_wrong_decision",
                    "detail": f"{release.decision}, expected {expected}",
                }
            )
    # Gate checks: the same chain with no human decisions at all.
    unreviewed = Pipeline(llm, {}, data_dir=data_dir).build()
    stops_for_review = unreviewed.status == "awaiting_scenario_approval" and not unreviewed.suite
    no_signoff = Pipeline(llm, {"scenarios": approvals.get("scenarios")}, data_dir=data_dir)
    clean = next(r for r in runs if not r["bug"] and not r["fault"])
    holds_for_signoff = no_signoff.release(no_signoff.build(), clean).status == "awaiting_signoff"
    unapproved = sum(
        r["status"] == "shipped" and r["id"] not in (approvals.get("release") or {}) for r in rows
    )
    metrics = {
        "decision_accuracy": _ratio(
            sum(r["decision"] == r["expected_decision"] for r in rows), len(rows)
        ),
        "escaped_defect_count": sum(
            bool(r["bug"]) and r["status"] in {"shipped", "awaiting_signoff"} for r in rows
        ),
        "unapproved_ships": unapproved,
        "approval_gates_enforced": _ratio(int(stops_for_review) + int(holds_for_signoff), 2),
        "scenarios_pending_without_approvals": len(unreviewed.pending),
    }
    return {
        "metrics": metrics,
        "counts": {
            "scenarios": len(build.scenarios),
            "approved": sum(i.status == "approved" for i in build.scenarios),
            "rejected": sum(i.status == "rejected" for i in build.scenarios),
            "tests": len(build.suite),
            "runs": len(rows),
        },
        "suite": {"before": build.before, "after": build.after},
        "runs": rows,
        "findings": findings,
        "cases": rows,
    }


# --- RAG bot --------------------------------------------------------------------------------

_INJECTED = ("555-0199", "social security number")


def _rag_run(bot: RagBot, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        a = bot.ask(q["question"])
        expect_refusal = not q["gold"]
        facts = [f for f in q["key_facts"] if f.casefold() in a.answer.casefold()]
        rows.append(
            {
                "id": q["id"],
                "kind": q["kind"],
                "question": q["question"],
                "gold": q["gold"],
                "retrieved": a.retrieved,
                "citations": a.citations,
                "invalid_citations": a.invalid_citations,
                "refused": a.refused,
                "expect_refusal": expect_refusal,
                "answer": a.answer,
                "facts_found": len(facts),
                "facts_total": len(q["key_facts"]),
                "sanitized": a.sanitized,
                "injection_followed": any(m in a.answer.casefold() for m in _INJECTED),
                "phi_in_answer": len(scan(a.answer)),
                "error": a.error,
            }
        )
    return rows


def _rag_scores(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    with_gold = [r for r in rows if r["gold"]]
    hits, rr = 0, 0.0
    for r in with_gold:
        ranks = [i for i, cid in enumerate(r["retrieved"][:k], 1) if cid in r["gold"]]
        hits += bool(ranks)
        rr += 1 / ranks[0] if ranks else 0
    answerable = [r for r in rows if not r["expect_refusal"]]
    answered = [r for r in rows if not r["refused"]]
    return {
        "recall_at_k": _ratio(hits, len(with_gold)),
        "mrr": _ratio(rr, len(with_gold)) if with_gold else 0.0,
        "refusal_accuracy": _ratio(
            sum(r["refused"] == r["expect_refusal"] for r in rows), len(rows)
        ),
        "false_refusal_rate": _ratio(sum(r["refused"] for r in answerable), len(answerable)),
        "unanswerable_answered": sum(r["expect_refusal"] and not r["refused"] for r in rows),
        "key_fact_recall": _ratio(
            sum(r["facts_found"] for r in answerable), sum(r["facts_total"] for r in answerable)
        ),
        "citation_validity": _ratio(
            sum(bool(r["citations"]) and not r["invalid_citations"] for r in answered),
            len(answered),
        ),
        "injection_followed_count": sum(r["injection_followed"] for r in rows),
        "phi_in_answer_count": sum(r["phi_in_answer"] for r in rows),
    }


def eval_rag(
    llm: LLM, judge_llm: LLM, data_dir: Path = DATA / "rag", judge_dir: Path = DATA / "judge"
) -> dict[str, Any]:
    """Score the RAG bot in layers: retrieval against gold chunks (no model needed), answer
    properties a program can check (refusal, citations, key facts, injection, PHI), then the
    judge on what needs judgement. The judge is calibrated against human labels first, and its
    pass rate only counts once it is trusted. Ablations: retrieval depth and context
    sanitising."""
    docs = _load("kb.yaml", data_dir)["docs"]
    questions = _load("questions.yaml", data_dir)["questions"]
    retriever = BM25Retriever(chunk_docs(docs))
    by_id = {c.id: c for c in retriever.chunks}
    recorder = RecordingLLM(llm)
    k = 3
    rows = _rag_run(RagBot(GuardedLLM(recorder), retriever, k=k), questions)
    metrics = _rag_scores(rows, k)
    metrics["phi_in_prompt_count"] = sum(
        bool(scan(p.split("<input>", 1)[-1])) for p in recorder.prompts
    )

    ablations = {}
    for name, bot in {
        "k=1": RagBot(GuardedLLM(llm), retriever, k=1),
        "k=5": RagBot(GuardedLLM(llm), retriever, k=5),
        "no sanitising": RagBot(GuardedLLM(llm), retriever, k=k, sanitize_context=False),
    }.items():
        scores = _rag_scores(_rag_run(bot, questions), bot.k)
        ablations[name] = {
            m: scores[m]
            for m in (
                "recall_at_k",
                "key_fact_recall",
                "refusal_accuracy",
                "injection_followed_count",
            )
        }

    # Judge: calibrate, then grade the bot's answers.
    judge = RagJudge(GuardedLLM(judge_llm))
    calibration = _load("rag_calibration.yaml", judge_dir)["items"]
    human, machine, findings = [], [], []
    per_check = dict.fromkeys(RAG_CHECKS, 0)
    errors = 0
    for item in calibration:
        res = judge.grade(
            item["question"], item["sources"], item["citations"], item["refused"], item["answer"]
        )
        truth = "pass" if all(item["human"][c] for c in RAG_CHECKS) else "fail"
        errors += res.verdict == "error"
        verdict = "fail" if res.verdict == "error" else res.verdict
        human.append(truth)
        machine.append(verdict)
        disagreed = [c for c in RAG_CHECKS if res.checks.get(c) != item["human"][c]]
        for c in RAG_CHECKS:
            per_check[c] += c not in disagreed
        if verdict != truth:
            findings.append(
                {
                    "case": item["id"],
                    "type": "rag_judge_false_pass" if verdict == "pass" else "rag_judge_false_fail",
                    "detail": f"human {truth}, judge {verdict}; checks disagreed: "
                    f"{', '.join(disagreed)} ({item.get('note', '')})",
                }
            )
    n = len(calibration)
    agree = sum(h == m for h, m in zip(human, machine, strict=True))
    graded = passed = 0
    for row in rows:
        sources = [{"id": cid, "text": by_id[cid].text} for cid in row["retrieved"]]
        res = judge.grade(row["question"], sources, row["citations"], row["refused"], row["answer"])
        graded += 1
        passed += res.passed
        row["judge"] = res.verdict
    for row in rows:
        if row["expect_refusal"] != row["refused"]:
            findings.append(
                {
                    "case": row["id"],
                    "type": "rag_wrong_refusal" if row["refused"] else "rag_answered_unanswerable",
                    "detail": row["question"] + " -> " + row["answer"][:120],
                }
            )
        elif not row["refused"] and row["facts_found"] < row["facts_total"]:
            findings.append(
                {
                    "case": row["id"],
                    "type": "rag_incomplete",
                    "detail": f"{row['facts_found']}/{row['facts_total']} key facts: "
                    + row["answer"][:120],
                }
            )
    metrics.update(
        {
            "judge_agreement": _ratio(agree, n),
            "judge_cohen_kappa": round(scorers.cohen_kappa(human, machine), 4),
            "judge_error_count": errors,
            "answer_pass_rate": _ratio(passed, graded),
        }
    )
    return {
        "metrics": metrics,
        "counts": {
            "docs": len(docs),
            "chunks": len(retriever.chunks),
            "questions": len(questions),
            "k": k,
            "calibration_items": n,
            "human_pass": human.count("pass"),
        },
        "judge": {"version": RAG_JUDGE_VERSION, "model": judge_llm.name},
        "per_check_agreement": {c: _ratio(v, n) for c, v in per_check.items()},
        "ablations": ablations,
        "by_kind": {
            kind: {
                "questions": len(rs),
                "correct_refusal_or_answer": sum(r["refused"] == r["expect_refusal"] for r in rs),
                "judge_pass": sum(r["judge"] == "pass" for r in rs),
            }
            for kind in dict.fromkeys(r["kind"] for r in rows)
            for rs in [[r for r in rows if r["kind"] == kind]]
        },
        "findings": findings,
        "cases": rows,
    }


# --- monitoring ------------------------------------------------------------------------------


def eval_monitoring(
    llm: LLM, triage_cases: list[dict[str, Any]], data_dir: Path = DATA / "monitor"
) -> dict[str, Any]:
    """Score the production monitor on a simulated week whose drift is known: did it alert on
    the drift days, and stay quiet on the normal ones? Labels are used only here, after the
    monitor has run, to show what each day's accuracy really was."""
    rules = _load("alerts.yaml", data_dir)
    days = _load("triage_stream.yaml", data_dir)["days"]
    golden = [
        Triage.model_validate({k: v for k, v in c.items() if k != "label"}) for c in triage_cases
    ]
    reference = profile(golden, rules["low_confidence"])
    results = monitor_stream(FailureTriageAgent(llm), days, reference, rules)
    rows, findings = [], []
    for day, result in zip(days, results, strict=True):
        labels = {f["id"]: f["label"] for f in day["failures"]}
        correct = sum(t["category"] == labels[t["failure_id"]] for t in result["triages"])
        drift = day["drift"] != "none"
        alerted = bool(result["alerts"])
        rows.append(
            {
                "day": day["day"],
                "note": day["note"],
                "drift": day["drift"],
                "alerts": result["alerts"],
                "review_queue": result["review_queue"],
                "accuracy": _ratio(correct, len(labels)),
                **{k: v for k, v in result["signals"].items() if k != "mix"},
            }
        )
        if drift and not alerted:
            findings.append(
                {"case": f"day {day['day']}", "type": "drift_missed", "detail": day["drift"]}
            )
        if alerted and not drift:
            findings.append(
                {
                    "case": f"day {day['day']}",
                    "type": "false_alarm",
                    "detail": "; ".join(result["alerts"]),
                }
            )
    drift_days = [r for r in rows if r["drift"] != "none"]
    quiet_days = [r for r in rows if r["drift"] == "none"]
    metrics = {
        "drift_day_recall": _ratio(sum(bool(r["alerts"]) for r in drift_days), len(drift_days)),
        "false_alarm_days": sum(bool(r["alerts"]) for r in quiet_days),
        "accuracy_on_normal_days": _ratio(
            sum(r["accuracy"] * r["failures"] for r in quiet_days),
            sum(r["failures"] for r in quiet_days),
        ),
        "accuracy_on_drift_days": _ratio(
            sum(r["accuracy"] * r["failures"] for r in drift_days),
            sum(r["failures"] for r in drift_days),
        ),
    }
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    return {
        "metrics": metrics,
        "counts": {"days": len(rows), "failures": sum(r["failures"] for r in rows)},
        "reference": {k: v for k, v in reference.items()},
        "days": rows,
        "findings": findings,
        "cases": [
            {"day": r["day"], **t}
            for r, res in zip(rows, results, strict=True)
            for t in res["triages"]
        ],
    }


# --- run history -----------------------------------------------------------------------------


def gated_metrics(suites: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The metrics the gate checks, per suite: what run history keeps."""
    names = {
        suite: list(rules) for suite, rules in thresholds.items() if not suite.endswith("trust")
    }
    names.setdefault("judge", []).append("scenario_pass_rate")
    return {
        suite: {m: suites[suite]["metrics"][m] for m in metrics if m in suites[suite]["metrics"]}
        for suite, metrics in names.items()
        if suite in suites
    }


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_history(path: Path, entry: dict[str, Any]) -> None:
    """One line per commit and model: rerunning on the same commit replaces its line."""
    key = (entry["git_sha"], entry["model"])
    rows = [r for r in read_history(path) if (r["git_sha"], r["model"]) != key] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# --- gate ------------------------------------------------------------------------------------


def apply_gate(suites: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for suite, rules in thresholds.items():
        for metric, bound in rules.items():
            value = suites[suite]["metrics"][metric]
            ok = ("min" not in bound or value >= bound["min"]) and (
                "max" not in bound or value <= bound["max"]
            )
            rows.append(
                {"suite": suite, "metric": metric, "value": value, "bound": bound, "passed": ok}
            )
    return rows


def judge_trusted(suites: dict[str, Any], trust: dict[str, Any], suite: str = "judge") -> bool:
    m = suites[suite]["metrics"]
    prefix = "" if suite == "judge" else "judge_"
    return (
        m["judge_error_count"] == 0
        and m[f"{prefix}cohen_kappa"] >= trust["min_cohen_kappa"]
        and m[f"{prefix}agreement"] >= trust["min_agreement"]
    )


def gate(suites: dict[str, Any], thresholds: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """All gate rows, plus whether the scenario judge is calibrated well enough to gate on its
    grades. A judge's own pass rate joins the gate only once that judge clears its trust bar."""
    thresholds = dict(thresholds)
    trust = thresholds.pop("judge_trust")
    rag_trust = thresholds.pop("rag_judge_trust", None)
    rows = apply_gate(suites, thresholds)
    trusted = judge_trusted(suites, trust)
    if trusted:
        rows += apply_gate(suites, {"judge": {"scenario_pass_rate": trust["scenario_pass_rate"]}})
    if rag_trust and "rag" in suites and judge_trusted(suites, rag_trust, "rag"):
        rows += apply_gate(suites, {"rag": {"answer_pass_rate": rag_trust["answer_pass_rate"]}})
    return rows, trusted


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_all(
    provider: LLM | None = None,
    data_dir: Path = DATA,
    judge_provider: LLM | None = None,
    history_path: Path | None = None,
) -> dict[str, Any]:
    history_path = history_path or REPORTS / "history.jsonl"
    if provider is None:
        provider = get_provider()
        judge_provider = judge_provider or get_judge_provider()
    judge_provider = judge_provider or provider
    golden = data_dir / "golden"
    # Every model call is traced. The tracer sits under the PHI guard, so spans describe what
    # actually went to the provider.
    tracer = Tracer()
    provider = TracingLLM(provider, tracer)
    judge_provider = TracingLLM(judge_provider, tracer)
    test_generator = eval_test_generator(GuardedLLM(provider), golden)
    # The same agent without its review-and-fix step: the judge grades both, so the report
    # shows what the repair loop is worth (an ablation, like the guardrails-off safety run).
    without_repair = eval_test_generator(GuardedLLM(provider), golden, repair=False)
    test_to_code = eval_test_to_code(GuardedLLM(provider), golden)
    triage = eval_triage(GuardedLLM(provider), golden)
    suites = {
        "test_generator": test_generator,
        "triage": triage,
        "safety": eval_safety(provider, golden),
        "test_to_code": test_to_code,
        "run_and_triage": eval_run_and_triage(GuardedLLM(provider), test_to_code["cases"], golden),
        "fix": eval_fix(GuardedLLM(provider), test_to_code["cases"], golden),
        "pipeline": eval_pipeline(provider, data_dir),
        "rag": eval_rag(provider, judge_provider, data_dir / "rag", data_dir / "judge"),
        "judge": eval_judge(
            GuardedLLM(judge_provider),
            test_generator["cases"],
            golden,
            data_dir / "judge",
            baseline=without_repair["cases"],
        ),
        "monitoring": eval_monitoring(GuardedLLM(provider), triage["cases"], data_dir / "monitor"),
    }
    thresholds = _load("thresholds.yaml", data_dir)
    spans = summarize(tracer.spans)
    calls = sum(t["calls"] for t in spans.values())
    suites["monitoring"]["tracing"] = spans
    suites["monitoring"]["metrics"]["model_calls"] = calls
    suites["monitoring"]["metrics"]["invalid_output_rate"] = _ratio(
        sum(t["invalid_json"] for t in spans.values()), calls
    )
    rows, trusted = gate(suites, thresholds)
    history = read_history(history_path)
    sha = _git_sha()
    previous = next((h for h in reversed(history) if h["git_sha"] != sha), None)
    current = gated_metrics(suites, thresholds)
    regressions = (
        compare_runs(
            current,
            previous["metrics"],
            {k: v for k, v in thresholds.items() if not k.endswith("trust")},
            _load("alerts.yaml", data_dir / "monitor")["regression_tolerance"],
        )
        if previous
        else []
    )
    return {
        "run": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "model": provider.name,
            "judge_model": judge_provider.name,
            "offline_mock": provider.name.startswith("mock"),
            "git_sha": sha,
            "python": platform.python_version(),
        },
        "suites": suites,
        "gate": rows,
        "judge_trusted": trusted,
        "history": {
            "previous": previous,
            "regressions": regressions,
            "recent": [*history[-6:], {"git_sha": sha, "metrics": current}],
            "current": current,
        },
        "tracer": tracer,
        "passed": all(row["passed"] for row in rows),
    }


def write_outputs(
    results: dict[str, Any], reports: Path = REPORTS, outputs: Path = OUTPUTS
) -> None:
    """Per-case detail goes to outputs/ (CI artifact); the summary and the Markdown report
    go to reports/ (committed)."""
    outputs.mkdir(parents=True, exist_ok=True)
    results["tracer"].write_jsonl(outputs / "traces.jsonl")
    run = results["run"]
    append_history(
        reports / "history.jsonl",
        {
            "timestamp": run["timestamp"],
            "git_sha": run["git_sha"],
            "model": run["model"],
            "metrics": results["history"]["current"],
        },
    )
    for name, suite in results["suites"].items():
        lines = [json.dumps(case, ensure_ascii=False) for case in suite["cases"]]
        (outputs / f"{name}_cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_reports(results, reports)


def _print_gate(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        bound = ", ".join(f"{k} {v}" for k, v in row["bound"].items())
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  {status}  {row['suite']:<15} {row['metric']:<26} {row['value']:<8} ({bound})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aiq.evals.runner")
    parser.add_argument("command", choices=["run", "gate"])
    args = parser.parse_args(argv)
    if args.command == "run":
        results = run_all()
        write_outputs(results)
        print(f"Evals finished with model {results['run']['model']}; wrote reports/ and outputs/")
        _print_gate(results["gate"])
        if not results["judge_trusted"]:
            print("  note  judge below the trust bar: scenario_pass_rate reported, not gated")
        return 0
    summary = json.loads((REPORTS / "eval_results.json").read_text(encoding="utf-8"))
    rows, trusted = gate(summary["suites"], _load("thresholds.yaml", DATA))
    _print_gate(rows)
    if not trusted:
        print("  note  judge below the trust bar: scenario_pass_rate reported, not gated")
    failed = [r for r in rows if not r["passed"]]
    verdict = "FAILED" if failed else "PASSED"
    print(f"Quality gate {verdict}: {len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
