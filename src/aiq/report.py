"""Render the eval results as Markdown. The report is generated, never hand-written.

python -m aiq.report    # re-render reports/eval_report.md from reports/eval_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPORTS = Path(__file__).resolve().parents[2] / "reports"

SUITE_TITLES = {
    "test_generator": "Requirements-to-Test",
    "triage": "Failure-Triage",
    "safety": "Outreach safety",
}
COUNT_METRICS = {
    "phi_leak_count",
    "phi_in_prompt_count",
    "risk_checklist_gaps",
    "false_refusals",
    "model_calls",
}

# Which findings make the best examples, and what each one means.
EXAMPLE_ORDER = [
    "caught_by_output_guard",
    "safety_injection",
    "misclassified:rules",
    "misclassified:llm",
    "schema",
    "missed_scenario",
    "near_duplicate",
    "wrong_layer",
]
EXPLANATIONS = {
    "caught_by_output_guard": "The input filter missed a reworded injection and the model "
    "repeated its confidential instructions. The output guard blocked the reply before it "
    "was sent. This is why there are two layers.",
    "safety_injection": "This injection was not in a language the input filter covers. The "
    "reply was harmless only because the model did not act on it, which is luck, not a "
    "control. Next step: multilingual injection detection.",
    "misclassified:rules": "A regex rule fired with high confidence on the wrong category. "
    "Rules are fast and auditable, but they need their own regression cases too.",
    "misclassified:llm": "No rule matched, and the fallback model picked a class from surface "
    "keywords. A better prompt, a stronger model or a new rule would fix this case.",
    "schema": "The model invented a value outside the allowed enum. Schema validation dropped "
    "the scenario instead of letting it reach a test plan, and coverage dropped with it.",
    "missed_scenario": "A case a reviewer would expect (usually a boundary) was not generated. "
    "This is the most common gap and the main reason recall is below 100%.",
    "near_duplicate": "Two acceptance criteria say the same thing, so the agent wrote the same "
    "test twice. Flagging it lets the story be fixed before tests are written.",
    "wrong_layer": "The scenario is right but sits on a different test layer than a reviewer "
    "would choose, which matters for test pyramid cost.",
}
LIMITATIONS = [
    "**Small golden sets** ({stories} stories, {failures} failures, {cases} safety prompts). "
    "Look at the triage confidence interval before reading much into a 2-point change.",
    "**Mock model.** These numbers come from a deterministic heuristic mock so CI is free and "
    "repeatable. Its mistakes are real outputs of that mock, but a real model will make "
    "different ones. Set `AIQ_LLM_PROVIDER=anthropic` to measure a real model.",
    "**Keyword matching for expected scenarios.** A correct scenario worded differently counts "
    "as a miss. An LLM-as-judge, calibrated against human labels, is the planned upgrade.",
    "**Regex PHI detection** over-redacts (for example ISO dates in logs) and cannot see "
    "free-text PHI such as diagnoses or names. It is a backstop, not a de-identification service.",
    "**Injection detection is pattern based** and covers English and Spanish only.",
    "**Thresholds were set from the first baseline run** and should be tightened as the agents "
    "improve.",
]


def _fmt(metric: str, value: Any) -> str:
    if metric in COUNT_METRICS or isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value * 100:.1f}%"


def _bound(metric: str, bound: dict[str, Any]) -> str:
    return " and ".join(f"{'≥' if k == 'min' else '≤'} {_fmt(metric, v)}" for k, v in bound.items())


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def _gate_rows(results: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["suite"], r["metric"]): r for r in results["gate"]}


def _metric_table(results: dict[str, Any], suite: str) -> list[str]:
    gate = _gate_rows(results)
    rows = []
    for metric, value in results["suites"][suite]["metrics"].items():
        g = gate.get((suite, metric))
        rows.append(
            [
                metric,
                _fmt(metric, value),
                _bound(metric, g["bound"]) if g else "not gated",
                ("pass" if g["passed"] else "**FAIL**") if g else "",
            ]
        )
    return _table(["Metric", "Result", "Threshold", "Gate"], rows)


def _kind(finding: dict[str, Any]) -> str:
    if finding["type"] == "misclassified":
        return "misclassified:rules" if "via rules" in finding["detail"] else "misclassified:llm"
    return finding["type"]


def _examples(results: dict[str, Any]) -> list[dict[str, Any]]:
    findings = [f for s in results["suites"].values() for f in s["findings"]]
    chosen = []
    for kind in EXAMPLE_ORDER:
        match = next((f for f in findings if _kind(f) == kind), None)
        if match:
            chosen.append(match)
    return chosen[:5]


def render(results: dict[str, Any]) -> str:
    run = results["run"]
    passed = sum(r["passed"] for r in results["gate"])
    model = f"`{run['model']}`" + (" (deterministic offline mock)" if run["offline_mock"] else "")
    out = [
        "# Eval report",
        "",
        "Generated by `python -m aiq.evals.runner run` (`make eval`). Do not edit by hand.",
        "",
        *_table(
            ["Run", ""],
            [
                ["Date", run["timestamp"]],
                ["Model", model],
                ["Commit", f"`{run['git_sha']}`"],
                ["Python", run["python"]],
                [
                    "Quality gate",
                    f"**{'PASSED' if results['passed'] else 'FAILED'}** "
                    f"({passed}/{len(results['gate'])} checks)",
                ],
            ],
        ),
        "",
        "## Results at a glance",
        "",
        *_table(
            ["Agent", "Metric", "Result", "Threshold", "Gate"],
            [
                [
                    SUITE_TITLES[r["suite"]],
                    r["metric"],
                    _fmt(r["metric"], r["value"]),
                    _bound(r["metric"], r["bound"]),
                    "pass" if r["passed"] else "**FAIL**",
                ]
                for r in results["gate"]
            ],
        ),
    ]

    tg = results["suites"]["test_generator"]
    c = tg["counts"]
    out += [
        "",
        "## 1. Requirements-to-Test agent",
        "",
        f"{c['stories']} user stories produced {c['scenarios']} scenarios; "
        f"{c['matched']} of {c['expected']} reviewer-expected scenarios were found.",
        "",
        *_metric_table(results, "test_generator"),
    ]

    tr = results["suites"]["triage"]
    m = tr["metrics"]
    labels = list(tr["confusion_matrix"])
    out += [
        "",
        "## 2. Failure-Triage agent",
        "",
        f"Accuracy {_fmt('accuracy', m['accuracy'])} on {tr['counts']['failures']} failures "
        f"(95% CI {_fmt('a', m['accuracy_ci95_low'])} to {_fmt('a', m['accuracy_ci95_high'])}). "
        f"Rules decided {tr['counts']['failures'] - tr['counts']['routed_to_llm']} cases at "
        f"{_fmt('a', m['rules_accuracy'])} accuracy; the LLM handled "
        f"{tr['counts']['routed_to_llm']} at {_fmt('a', m['llm_accuracy'])}.",
        "",
        *_metric_table(results, "triage"),
        "",
        "**Per class**",
        "",
        *_table(
            ["Class", "Precision", "Recall", "F1", "Support"],
            [
                [
                    k,
                    _fmt("p", v["precision"]),
                    _fmt("r", v["recall"]),
                    _fmt("f", v["f1"]),
                    str(v["support"]),
                ]
                for k, v in tr["per_class"].items()
            ],
        ),
        "",
        "**Confusion matrix** (rows = true label, columns = predicted)",
        "",
        *_table(
            ["true \\ predicted", *labels],
            [[t, *(str(tr["confusion_matrix"][t][p]) for p in labels)] for t in labels],
        ),
    ]

    sf = results["suites"]["safety"]
    ab = sf["ablation_unguarded"]
    out += [
        "",
        "## 3. Outreach safety",
        "",
        f"{sf['counts']['cases']} adversarial and normal member messages: prompt injection "
        "(English, Spanish, Vietnamese), requests for another member's data, PHI in the input, "
        "requests to send PHI by SMS, and a crisis message. The same set is run with the "
        "guardrails off to show what they prevent.",
        "",
        *_table(
            ["Metric", "With guardrails", "Without guardrails"],
            [[k, _fmt(k, v), _fmt(k, ab[k])] for k, v in sf["metrics"].items()],
        ),
        "",
        *_metric_table(results, "safety"),
        "",
        "## What the evals caught",
        "",
    ]
    for f in _examples(results):
        out += [f"- **{f['case']}** ({f['type']}): {f['detail']}  ", f"  {EXPLANATIONS[_kind(f)]}"]

    sizes = {
        "stories": tg["counts"]["stories"],
        "failures": tr["counts"]["failures"],
        "cases": sf["counts"]["cases"],
    }
    all_findings = [
        [SUITE_TITLES[name], f["case"], f["type"], f["detail"].replace("|", "\\|")]
        for name, s in results["suites"].items()
        for f in s["findings"]
    ]
    out += [
        "",
        f"<details><summary>All {len(all_findings)} findings</summary>",
        "",
        *_table(["Agent", "Case", "Type", "Detail"], all_findings),
        "",
        "</details>",
        "",
        "## Known limitations",
        "",
        *[f"- {item}".format(**sizes) for item in LIMITATIONS],
        "",
    ]
    return "\n".join(out)


def write_reports(results: dict[str, Any], reports: Path = REPORTS) -> None:
    """Write reports/eval_results.json (summary, no per-case detail) and eval_report.md."""
    summary = {
        **results,
        "suites": {
            name: {k: v for k, v in suite.items() if k != "cases"}
            for name, suite in results["suites"].items()
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "eval_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports / "eval_report.md").write_text(render(summary), encoding="utf-8")


def main() -> int:
    results = json.loads((REPORTS / "eval_results.json").read_text(encoding="utf-8"))
    (REPORTS / "eval_report.md").write_text(render(results), encoding="utf-8")
    print("Rendered reports/eval_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
