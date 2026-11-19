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

from aiq.agents.failure_triage import CATEGORIES, Failure, FailureTriageAgent
from aiq.agents.outreach_assistant import CANARY, Member, OutreachAssistant
from aiq.agents.test_generator import RequirementsToTestAgent, Story, lint_scenario
from aiq.agents.test_to_code import TestToCodeAgent, run_test
from aiq.demo.referral_service import BUGS, ReferralService
from aiq.evals import scorers
from aiq.evals.judge import CHECKS, JUDGE_VERSION, ScenarioJudge
from aiq.guards.phi_guard import scan
from aiq.llm import LLM, GuardedLLM, get_judge_provider, get_provider
from aiq.report import write_reports

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


def judge_trusted(suites: dict[str, Any], trust: dict[str, Any]) -> bool:
    m = suites["judge"]["metrics"]
    return (
        m["judge_error_count"] == 0
        and m["cohen_kappa"] >= trust["min_cohen_kappa"]
        and m["agreement"] >= trust["min_agreement"]
    )


def gate(suites: dict[str, Any], thresholds: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """All gate rows, plus whether the judge is calibrated well enough to gate on its grades."""
    thresholds = dict(thresholds)
    trust = thresholds.pop("judge_trust")
    rows = apply_gate(suites, thresholds)
    trusted = judge_trusted(suites, trust)
    if trusted:
        rows += apply_gate(suites, {"judge": {"scenario_pass_rate": trust["scenario_pass_rate"]}})
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
    provider: LLM | None = None, data_dir: Path = DATA, judge_provider: LLM | None = None
) -> dict[str, Any]:
    if provider is None:
        provider = get_provider()
        judge_provider = judge_provider or get_judge_provider()
    judge_provider = judge_provider or provider
    golden = data_dir / "golden"
    test_generator = eval_test_generator(GuardedLLM(provider), golden)
    # The same agent without its review-and-fix step: the judge grades both, so the report
    # shows what the repair loop is worth (an ablation, like the guardrails-off safety run).
    without_repair = eval_test_generator(GuardedLLM(provider), golden, repair=False)
    suites = {
        "test_generator": test_generator,
        "triage": eval_triage(GuardedLLM(provider), golden),
        "safety": eval_safety(provider, golden),
        "test_to_code": eval_test_to_code(GuardedLLM(provider), golden),
        "judge": eval_judge(
            GuardedLLM(judge_provider),
            test_generator["cases"],
            golden,
            data_dir / "judge",
            baseline=without_repair["cases"],
        ),
    }
    rows, trusted = gate(suites, _load("thresholds.yaml", data_dir))
    return {
        "run": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "model": provider.name,
            "judge_model": judge_provider.name,
            "offline_mock": provider.name.startswith("mock"),
            "git_sha": _git_sha(),
            "python": platform.python_version(),
        },
        "suites": suites,
        "gate": rows,
        "judge_trusted": trusted,
        "passed": all(row["passed"] for row in rows),
    }


def write_outputs(
    results: dict[str, Any], reports: Path = REPORTS, outputs: Path = OUTPUTS
) -> None:
    """Per-case detail goes to outputs/ (CI artifact); the summary and the Markdown report
    go to reports/ (committed)."""
    outputs.mkdir(parents=True, exist_ok=True)
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
