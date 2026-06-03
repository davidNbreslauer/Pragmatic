from __future__ import annotations

from pathlib import Path

from pragmatic.belief_update import apply_belief_updates, update_beliefs
from pragmatic.decisive_tests import propose_decisive_tests
from pragmatic.eval_writer import generate_evals_from_failures
from pragmatic.eval_workshop import build_eval_workshop
from pragmatic.extractors import ExtractionMode
from pragmatic.invalid_leaps import detect_invalid_leaps
from pragmatic.raindrop_client import ObservabilityMode, record_research_run
from pragmatic.research_loop import DEFAULT_THESIS, run_research_loop
from pragmatic.schemas import (
    ExecutionBackend,
    ReplayComparison,
    ReplayResult,
    ResearchState,
    TraceEvent,
)


REPLAY_ASSUMPTION_IDS = ["A5", "A6"]


def run_replay_demo(
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
    corpus_path: str | Path | None = None,
    execution_backend: ExecutionBackend | None = None,
    extraction_mode: ExtractionMode = "local",
    modal_fallback: bool = True,
    observability_mode: ObservabilityMode = "local",
    observability_dir: str | Path | None = None,
    raindrop_fallback: bool = True,
) -> ReplayResult:
    first_pass_base = run_research_loop(
        thesis_text,
        max_iterations=max_iterations,
        corpus_path=corpus_path,
        execution_backend=execution_backend,
        extraction_mode=extraction_mode,
        modal_fallback=modal_fallback,
        observability_mode="off",
    )
    first_pass = simulate_overcredited_first_pass(first_pass_base)
    applied_eval_rules = _replay_eval_rules(first_pass)

    replay_pass = run_research_loop(
        thesis_text,
        max_iterations=max_iterations,
        corpus_path=corpus_path,
        execution_backend=execution_backend,
        extraction_mode=extraction_mode,
        modal_fallback=modal_fallback,
        observability_mode="off",
    )
    _append_trace(
        replay_pass,
        "replay",
        "Replayed the same thesis after applying generated benchmark-proxy eval rules.",
        metadata={"applied_eval_rules": str(len(applied_eval_rules))},
    )

    comparisons = compare_replay_passes(first_pass, replay_pass)
    summary = _replay_summary(first_pass)
    replay_pass.eval_workshop = build_eval_workshop(
        replay_pass,
        replay_first_pass=first_pass,
        replay_comparisons=comparisons,
        applied_eval_rules=applied_eval_rules,
    )
    _append_trace(
        replay_pass,
        "eval_workshop",
        replay_pass.eval_workshop.summary,
        metadata={
            "task_spans": str(len(replay_pass.eval_workshop.task_spans)),
            "failure_eval_links": str(len(replay_pass.eval_workshop.failure_eval_links)),
            "replay_outcomes": str(len(replay_pass.eval_workshop.replay_outcomes)),
        },
    )
    replay_pass.observability = record_research_run(
        replay_pass,
        mode=observability_mode,
        trace_dir=observability_dir,
        fallback_to_local=raindrop_fallback,
    )
    _append_trace(
        replay_pass,
        "observability",
        f"Recorded replay observability artifact via {replay_pass.observability.backend}.",
        metadata={
            "trace_id": replay_pass.observability.trace_id,
            "backend": replay_pass.observability.backend,
            "status": replay_pass.observability.status,
        },
    )
    return ReplayResult(
        first_pass=first_pass,
        replay_pass=replay_pass,
        applied_eval_rules=applied_eval_rules,
        comparisons=comparisons,
        eval_workshop=replay_pass.eval_workshop,
        summary=summary,
    )


def simulate_overcredited_first_pass(state: ResearchState) -> ResearchState:
    simulated = state.model_copy(deep=True)
    simulated.observability = None
    simulated.evidence_conflicts = []
    overcredited_source_ids = _overcredited_source_ids(simulated)

    for item in simulated.evidence_items:
        if item.source_id not in overcredited_source_ids:
            continue
        item.evidence_type = "direct"
        item.confidence = max(item.confidence, 0.82)
        item.claim_supported = (
            "Proxy evidence is over-credited as direct evidence for the claimed real-world application."
        )
        item.limitation = (
            "First-pass failure: the source was not bounded as proxy or indirect evidence."
        )

    for assumption in simulated.assumptions:
        assumption.support_level = "unknown"
        assumption.confidence = 0.0
        assumption.latest_update = None

    simulated.invalid_leaps = detect_invalid_leaps(simulated)
    simulated.belief_updates = update_beliefs(simulated)
    apply_belief_updates(simulated, simulated.belief_updates)
    simulated.decisive_tests = propose_decisive_tests(simulated)
    simulated.generated_evals = generate_evals_from_failures(simulated.invalid_leaps)
    simulated.eval_workshop = build_eval_workshop(simulated)
    _append_trace(
        simulated,
        "replay_first_pass",
        "Simulated a first-pass failure that treated proxy evidence as direct support.",
        metadata={"overcredited_sources": ", ".join(sorted(overcredited_source_ids))},
    )
    return simulated


def compare_replay_passes(
    first_pass: ResearchState,
    replay_pass: ResearchState,
) -> list[ReplayComparison]:
    first_assumptions = {assumption.id: assumption for assumption in first_pass.assumptions}
    replay_assumptions = {assumption.id: assumption for assumption in replay_pass.assumptions}
    comparisons: list[ReplayComparison] = []

    for assumption_id in REPLAY_ASSUMPTION_IDS:
        before = first_assumptions[assumption_id]
        after = replay_assumptions[assumption_id]
        delta = round(after.confidence - before.confidence, 2)
        comparisons.append(
            ReplayComparison(
                assumption_id=assumption_id,
                before_support=before.support_level,
                after_support=after.support_level,
                before_confidence=before.confidence,
                after_confidence=after.confidence,
                change_summary=f"{delta:+.2f} confidence after replay",
                rationale=_comparison_rationale(assumption_id, before.confidence, after.confidence),
            )
        )
    return comparisons


def _replay_eval_rules(state: ResearchState) -> list[str]:
    return [
        generated_eval.eval_rule
        for generated_eval in state.generated_evals
        if "benchmark" in generated_eval.eval_rule.lower()
        or "directly measures" in generated_eval.eval_rule.lower()
    ]


def _overcredited_source_ids(state: ResearchState) -> set[str]:
    return {
        item.source_id
        for item in state.evidence_items
        if item.evidence_type in {"proxy", "indirect", "anecdotal"}
        and any(assumption_id in REPLAY_ASSUMPTION_IDS for assumption_id in item.assumption_ids)
    }


def _replay_summary(first_pass: ResearchState) -> str:
    return (
        "First pass over-credited proxy evidence as direct application evidence. "
        "The generated eval rule forces the replay to classify evidence by what it directly "
        "measures, lowering confidence in standards-relevant validation."
    )


def _comparison_rationale(
    assumption_id: str,
    before_confidence: float,
    after_confidence: float,
) -> str:
    if assumption_id == "A5":
        return (
            "The replay keeps system-integration evidence visible but stops treating "
            "proxy evidence as direct application proof."
        )
    if after_confidence < before_confidence:
        return (
            "The replay downgrades standards-relevant validation because the evidence "
            "is reclassified by what it directly measures."
        )
    return "The replay preserves the stricter evidence boundary for this assumption."


def _append_trace(
    state: ResearchState,
    stage: str,
    message: str,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    state.trace_events.append(
        TraceEvent(
            id=f"trace_{len(state.trace_events) + 1:03d}",
            stage=stage,
            message=message,
            metadata=metadata or {},
        )
    )
