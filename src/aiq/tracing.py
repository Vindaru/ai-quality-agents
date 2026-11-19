"""Tracing for every model call, shaped like OpenTelemetry GenAI spans.

``TracingLLM`` wraps a model and records one span per call: which agent task made it, the model,
token counts, latency, whether the answer was parseable JSON, and the version of the system
prompt. Attribute names follow the OpenTelemetry GenAI semantic conventions (``gen_ai.*``), so the
same spans could be exported to Langfuse, Phoenix or any OTLP backend without renaming.

Spans never hold prompt or answer text. Prompts can carry member data, and a trace store is one
more place for PHI to leak, so a span keeps sizes and a hash of the *system* prompt only. The
system prompt is fixed per agent version, so its hash identifies the prompt version without
exposing any input.

Token counts use a four-characters-per-token estimate, because the offline mock has no
tokenizer. A real provider's usage numbers should replace the estimate when available.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aiq.llm import LLM, extract_json


def _tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def prompt_version(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_unix_ms: int
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects spans for one run (one trace)."""

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.spans: list[Span] = []

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(s), ensure_ascii=False) for s in self.spans]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TracingLLM:
    def __init__(self, inner: LLM, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer
        self.name = inner.name

    def complete(self, system: str, prompt: str, *, task: str) -> str:
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self.inner.name,
            "gen_ai.usage.input_tokens": _tokens(system) + _tokens(prompt),
            "aiq.task": task,
            "aiq.prompt_version": prompt_version(system),
        }
        start = time.time()
        clock = time.perf_counter()
        try:
            answer = self.inner.complete(system, prompt, task=task)
        except Exception as exc:
            attributes["error.type"] = type(exc).__name__
            self._record(task, start, clock, attributes)
            raise
        attributes["gen_ai.usage.output_tokens"] = _tokens(answer)
        try:
            extract_json(answer)
            attributes["aiq.output_json_valid"] = True
        except ValueError:
            attributes["aiq.output_json_valid"] = False
        self._record(task, start, clock, attributes)
        return answer

    def _record(self, task: str, start: float, clock: float, attributes: dict[str, Any]) -> None:
        self.tracer.spans.append(
            Span(
                trace_id=self.tracer.trace_id,
                span_id=uuid.uuid4().hex[:16],
                name=f"chat {task}",
                start_unix_ms=int(start * 1000),
                duration_ms=round((time.perf_counter() - clock) * 1000, 3),
                attributes=attributes,
            )
        )


def summarize(spans: list[Span]) -> dict[str, dict[str, Any]]:
    """Per-task totals: calls, tokens, errors, invalid JSON answers and prompt versions.
    Latency is left out on purpose: it is in the span file, but it would make the committed
    report change on every run."""
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "errors": 0,
            "invalid_json": 0,
            "prompt_versions": set(),
        }
    )
    for span in spans:
        a = span.attributes
        row = by_task[a["aiq.task"]]
        row["calls"] += 1
        row["input_tokens"] += a["gen_ai.usage.input_tokens"]
        row["output_tokens"] += a.get("gen_ai.usage.output_tokens", 0)
        row["errors"] += "error.type" in a
        row["invalid_json"] += a.get("aiq.output_json_valid") is False
        row["prompt_versions"].add(a["aiq.prompt_version"])
    return {
        task: {**row, "prompt_versions": sorted(row["prompt_versions"])}
        for task, row in sorted(by_task.items())
    }
