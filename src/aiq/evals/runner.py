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
from aiq.agents.test_generator import RequirementsToTestAgent, Story
from aiq.evals import scorers
from aiq.guards.phi_guard import scan
from aiq.llm import LLM, GuardedLLM, get_provider
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


def eval_test_generator(llm: LLM, data_dir: Path = DATA / "golden") -> dict[str, Any]:
    agent = RequirementsToTestAgent(llm)
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
    agent = OutreachAssistant(GuardedLLM(recorder) if guarded else recorder, guarded=guarded)
    cases, findings = [], []
    for case in data["cases"]:
        member = Member.model_validate(data["members"][case["member"]])
        decision = agent.run(member, case["text"])
        planted = [
            member.phone,
            member.dob,
            member.member_id,
            *(f.value for f in scan(case["text"])),
        ]
        leaks = scorers.phi_leaks(decision.message, planted)
        resisted = scorers.injection_resisted(
            decision.message, [CANARY, *case.get("forbidden", [])]
        )
        correct = decision.action == case["expected_action"]
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
        "phi_in_prompt_count": sum(bool(scan(p)) for p in recorder.prompts),
        "refusal_correctness": _ratio(sum(c["correct"] for c in cases), len(cases)),
        "false_refusals": sum(c["action"] != "reply" for c in should_reply),
        "model_calls": len(recorder.prompts),
    }
    return {"metrics": metrics, "findings": findings, "cases": cases}


def eval_safety(provider: LLM, data_dir: Path = DATA / "golden") -> dict[str, Any]:
    data = _load("outreach_safety.yaml", data_dir)
    guarded = _safety_pass(provider, True, data)
    unguarded = _safety_pass(provider, False, data)
    return {
        "metrics": guarded["metrics"],
        "ablation_unguarded": unguarded["metrics"],
        "counts": {"cases": len(data["cases"])},
        "findings": guarded["findings"],
        "cases": guarded["cases"],
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


def run_all(provider: LLM | None = None, data_dir: Path = DATA) -> dict[str, Any]:
    provider = provider or get_provider()
    golden = data_dir / "golden"
    suites = {
        "test_generator": eval_test_generator(GuardedLLM(provider), golden),
        "triage": eval_triage(GuardedLLM(provider), golden),
        "safety": eval_safety(provider, golden),
    }
    gate = apply_gate(suites, _load("thresholds.yaml", data_dir))
    return {
        "run": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "model": provider.name,
            "offline_mock": provider.name.startswith("mock"),
            "git_sha": _git_sha(),
            "python": platform.python_version(),
        },
        "suites": suites,
        "gate": gate,
        "passed": all(row["passed"] for row in gate),
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
        return 0
    summary = json.loads((REPORTS / "eval_results.json").read_text(encoding="utf-8"))
    rows = apply_gate(summary["suites"], _load("thresholds.yaml", DATA))
    _print_gate(rows)
    failed = [r for r in rows if not r["passed"]]
    verdict = "FAILED" if failed else "PASSED"
    print(f"Quality gate {verdict}: {len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
