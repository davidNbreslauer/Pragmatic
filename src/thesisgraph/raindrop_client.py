from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from thesisgraph.eval_workshop import build_eval_workshop
from thesisgraph.schemas import ObservabilityRecord, ResearchState


ObservabilityMode = Literal["local", "raindrop", "off"]

DEFAULT_TRACE_DIR = Path(".thesisgraph") / "traces"


def record_research_run(
    state: ResearchState,
    *,
    mode: ObservabilityMode = "local",
    trace_dir: str | Path | None = None,
    fallback_to_local: bool = True,
) -> ObservabilityRecord:
    trace_id = f"tg_{uuid.uuid4().hex[:12]}"

    if mode == "off":
        return ObservabilityRecord(
            trace_id=trace_id,
            backend="off",
            status="skipped",
            message="Observability disabled for this run.",
        )

    if mode == "local":
        return _record_local(state, trace_id=trace_id, trace_dir=trace_dir)

    try:
        return _record_raindrop_sdk(state, trace_id=trace_id)
    except Exception as exc:
        if not fallback_to_local:
            return ObservabilityRecord(
                trace_id=trace_id,
                backend="raindrop",
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
        record = _record_local(state, trace_id=trace_id, trace_dir=trace_dir)
        record.message = f"Raindrop unavailable; wrote local trace instead. {type(exc).__name__}: {exc}"
        return record


def build_trace_payload(state: ResearchState, *, trace_id: str) -> dict:
    eval_workshop = state.eval_workshop or build_eval_workshop(state)
    return {
        "trace_id": trace_id,
        "created_at": datetime.now(UTC).isoformat(),
        "thesis": state.thesis.model_dump(),
        "summary": {
            "assumptions": len(state.assumptions),
            "sources": len(state.sources),
            "evidence_items": len(state.evidence_items),
            "evidence_conflicts": len(state.evidence_conflicts),
            "invalid_leaps": len(state.invalid_leaps),
            "verifier_results": len(state.verifier_results),
            "generated_evals": len(state.generated_evals),
            "task_spans": len(eval_workshop.task_spans),
            "failure_eval_links": len(eval_workshop.failure_eval_links),
            "replay_outcomes": len(eval_workshop.replay_outcomes),
        },
        "trace_events": [event.model_dump() for event in state.trace_events],
        "research_task_results": [result.model_dump() for result in state.research_task_results],
        "task_spans": [span.model_dump() for span in eval_workshop.task_spans],
        "evidence_conflicts": [conflict.model_dump() for conflict in state.evidence_conflicts],
        "invalid_leaps": [leap.model_dump() for leap in state.invalid_leaps],
        "belief_updates": [update.model_dump() for update in state.belief_updates],
        "verifier_results": [result.model_dump() for result in state.verifier_results],
        "generated_evals": [generated_eval.model_dump() for generated_eval in state.generated_evals],
        "decisive_tests": [test.model_dump() for test in state.decisive_tests],
        "eval_workshop": eval_workshop.model_dump(),
        "failure_eval_links": [
            link.model_dump()
            for link in eval_workshop.failure_eval_links
        ],
        "replay_outcomes": [
            outcome.model_dump()
            for outcome in eval_workshop.replay_outcomes
        ],
    }


def _record_local(
    state: ResearchState,
    *,
    trace_id: str,
    trace_dir: str | Path | None = None,
) -> ObservabilityRecord:
    directory = Path(trace_dir) if trace_dir is not None else DEFAULT_TRACE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace_id}.json"
    payload = build_trace_payload(state, trace_id=trace_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ObservabilityRecord(
        trace_id=trace_id,
        backend="local",
        status="recorded",
        trace_path=str(path),
        event_id=trace_id,
        eval_artifact_ids=[generated_eval.id for generated_eval in state.generated_evals],
        workshop_artifact_ids=_workshop_artifact_ids(state),
        message="Recorded local Raindrop-compatible trace artifact.",
    )


def _record_raindrop_sdk(state: ResearchState, *, trace_id: str) -> ObservabilityRecord:
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        raise RuntimeError("RAINDROP_WRITE_KEY is not set.")

    import raindrop.analytics as raindrop

    raindrop.init(write_key, tracing_enabled=True, bypass_otel_for_tools=True)
    eval_workshop = state.eval_workshop or build_eval_workshop(state)
    interaction = raindrop.begin(
        user_id="thesisgraph-local",
        event="thesisgraph_research_run",
        event_id=trace_id,
        input=state.thesis.text,
        convo_id=trace_id,
        properties={
            "assumptions": len(state.assumptions),
            "evidence_items": len(state.evidence_items),
            "evidence_conflicts": len(state.evidence_conflicts),
            "invalid_leaps": len(state.invalid_leaps),
            "verifier_results": len(state.verifier_results),
            "generated_evals": len(state.generated_evals),
            "task_spans": len(eval_workshop.task_spans),
            "failure_eval_links": len(eval_workshop.failure_eval_links),
            "replay_outcomes": len(eval_workshop.replay_outcomes),
        },
        attachments=[
            {
                "type": "text",
                "name": "ResearchState summary",
                "value": json.dumps(build_trace_payload(state, trace_id=trace_id), indent=2),
                "role": "output",
            }
        ],
    )

    for event in state.trace_events:
        interaction.track_tool(
            name=f"thesisgraph.{event.stage}",
            input=event.metadata,
            output=event.message,
            duration_ms=0,
            properties={"trace_event_id": event.id},
        )

    for span in eval_workshop.task_spans:
        interaction.track_tool(
            name=f"thesisgraph.task.{span.task_type}",
            input={
                "task_id": span.task_id,
                "source_ids": span.source_ids,
                "backend": span.backend,
            },
            output={
                "status": span.status,
                "evidence_item_count": span.evidence_item_count,
                "evidence_conflict_count": span.evidence_conflict_count,
                "verifier_result_count": span.verifier_result_count,
                "error": span.error,
            },
            duration_ms=0,
            properties={"span_id": span.id},
        )

    for generated_eval in state.generated_evals:
        interaction.track_tool(
            name="thesisgraph.generated_eval",
            input={
                "failure_observed": generated_eval.failure_observed,
                "root_cause": generated_eval.root_cause,
            },
            output={
                "eval_rule": generated_eval.eval_rule,
                "expected_behavior": generated_eval.expected_behavior,
            },
            duration_ms=0,
            properties={"eval_id": generated_eval.id},
        )

    for link in eval_workshop.failure_eval_links:
        interaction.track_tool(
            name=f"thesisgraph.eval_workshop.{link.link_type}",
            input={
                "source_id": link.source_id,
                "affected_assumption_ids": link.affected_assumption_ids,
                "source_ids": link.source_ids,
            },
            output={"target_id": link.target_id, "summary": link.summary},
            duration_ms=0,
            properties={"link_id": link.id},
        )

    for outcome in eval_workshop.replay_outcomes:
        interaction.track_tool(
            name="thesisgraph.eval_workshop.replay_outcome",
            input={
                "assumption_id": outcome.assumption_id,
                "before_confidence": outcome.before_confidence,
                "applied_eval_rule": outcome.applied_eval_rule,
            },
            output={
                "after_confidence": outcome.after_confidence,
                "passed": outcome.passed,
                "summary": outcome.summary,
            },
            duration_ms=0,
            properties={"replay_outcome_id": outcome.id},
        )

    interaction.finish(
        output=(
            f"Detected {len(state.invalid_leaps)} invalid leaps and generated "
            f"{len(state.generated_evals)} evals."
        )
    )
    raindrop.flush()
    return ObservabilityRecord(
        trace_id=trace_id,
        backend="raindrop",
        status="recorded",
        event_id=trace_id,
        eval_artifact_ids=[generated_eval.id for generated_eval in state.generated_evals],
        workshop_artifact_ids=_workshop_artifact_ids(state),
        message="Recorded run through Raindrop SDK.",
    )


def _workshop_artifact_ids(state: ResearchState) -> list[str]:
    eval_workshop = state.eval_workshop or build_eval_workshop(state)
    return [
        *[span.id for span in eval_workshop.task_spans],
        *[link.id for link in eval_workshop.failure_eval_links],
        *[outcome.id for outcome in eval_workshop.replay_outcomes],
    ]
