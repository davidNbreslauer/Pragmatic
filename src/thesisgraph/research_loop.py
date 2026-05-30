from __future__ import annotations

from pathlib import Path

from thesisgraph.belief_update import apply_belief_updates, update_beliefs
from thesisgraph.corpus import load_corpus, rank_sources_for_questions
from thesisgraph.decisive_tests import propose_decisive_tests
from thesisgraph.eval_writer import generate_evals_from_failures
from thesisgraph.eval_workshop import build_eval_workshop
from thesisgraph.execution import (
    build_cross_check_task,
    build_source_extraction_tasks,
    execute_research_tasks,
)
from thesisgraph.extractors import ExtractionMode
from thesisgraph.invalid_leaps import detect_invalid_leaps
from thesisgraph.raindrop_client import ObservabilityMode, record_research_run
from thesisgraph.verifiers import build_verifier_tasks
from thesisgraph.schemas import (
    Assumption,
    ExecutionBackend,
    ResearchQuestion,
    RetrievalScore,
    ResearchState,
    Source,
    Thesis,
    TraceEvent,
)


DEFAULT_THESIS = "Graph-based AI scientist systems can accelerate real materials discovery."


def run_research_loop(
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
    corpus_path: str | Path | None = None,
    execution_backend: ExecutionBackend | None = None,
    extraction_mode: ExtractionMode | None = None,
    modal_fallback: bool = True,
    observability_mode: ObservabilityMode = "local",
    observability_dir: str | Path | None = None,
    raindrop_fallback: bool = True,
) -> ResearchState:
    resolved_backend = _resolve_execution_backend(execution_backend, extraction_mode)
    state = ResearchState(thesis=Thesis(text=thesis_text, domain="materials discovery"))
    _trace(state, "initialize", "Initialized ResearchState from thesis.")

    state.assumptions = decompose_thesis(thesis_text)
    _trace(state, "decompose", f"Generated {len(state.assumptions)} assumptions.")

    state.research_questions = generate_initial_questions(state)
    _trace(state, "plan", f"Generated {len(state.research_questions)} research questions.")

    corpus = load_corpus(corpus_path)

    for iteration in range(1, max_iterations + 1):
        state.iteration = iteration
        open_questions = [question for question in state.research_questions if question.status == "open"]
        if not open_questions:
            _trace(state, "stop", "No open questions remain.")
            break

        retrieved_sources = retrieve_sources(open_questions, corpus)
        retrieval_scores = score_retrieval(open_questions, corpus)
        _append_unique(state.retrieval_scores, retrieval_scores)
        _append_unique(state.sources, retrieved_sources)
        for question in open_questions:
            question.status = "answered"
        top_score = max((score.score for score in retrieval_scores), default=0.0)
        _trace(
            state,
            "retrieve",
            f"Retrieved {len(retrieved_sources)} local corpus sources with deterministic scoring.",
            metadata={
                "retrieval_score_count": str(len(retrieval_scores)),
                "top_score": f"{top_score:.3f}",
                "top_source_id": _top_source_id(retrieval_scores),
            },
        )

        extraction_tasks = build_source_extraction_tasks(
            retrieved_sources,
            state.assumptions,
            open_questions,
        )
        execution = execute_research_tasks(
            extraction_tasks,
            backend=resolved_backend,
            fallback_to_local=modal_fallback,
        )
        _append_unique(state.research_task_results, execution.results)
        extracted_items = [
            item
            for result in execution.results
            for item in result.evidence_items
        ]
        _append_unique(state.evidence_items, sorted(extracted_items, key=lambda item: item.id))
        _trace(
            state,
            "execute",
            (
                f"Executed {len(extraction_tasks)} source research tasks via "
                f"{execution.backend} backend."
            ),
            metadata={
                "backend": execution.backend,
                "attempted_backend": execution.attempted_backend,
                "task_count": str(len(extraction_tasks)),
                "source_count": str(len(retrieved_sources)),
                **({"fallback_reason": execution.fallback_reason} if execution.fallback_reason else {}),
            },
        )
        for result in execution.results:
            _trace(
                state,
                "task",
                f"{result.task_type} {result.task_id} {result.status} via {result.backend}.",
                metadata={
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "backend": result.backend,
                    "status": result.status,
                    "source_ids": ",".join(result.source_ids),
                },
            )
        _trace(
            state,
            "extract",
            (
                f"Extracted {len(extracted_items)} typed evidence items via "
                f"{execution.backend} execution backend."
            ),
            metadata={
                "backend": execution.backend,
                "attempted_backend": execution.attempted_backend,
                "task_count": str(len(extraction_tasks)),
                "source_count": str(len(retrieved_sources)),
                **({"fallback_reason": execution.fallback_reason} if execution.fallback_reason else {}),
            },
        )

        cross_check_task = build_cross_check_task(
            state.sources,
            state.evidence_items,
            state.assumptions,
        )
        cross_check_execution = execute_research_tasks(
            [cross_check_task],
            backend=resolved_backend,
            fallback_to_local=modal_fallback,
        )
        _append_unique(state.research_task_results, cross_check_execution.results)
        conflicts = [
            conflict
            for result in cross_check_execution.results
            for conflict in result.evidence_conflicts
        ]
        state.evidence_conflicts = []
        _append_unique(state.evidence_conflicts, sorted(conflicts, key=lambda conflict: conflict.id))
        for result in cross_check_execution.results:
            _trace(
                state,
                "task",
                f"{result.task_type} {result.task_id} {result.status} via {result.backend}.",
                metadata={
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "backend": result.backend,
                    "status": result.status,
                    "source_ids": ",".join(result.source_ids),
                },
            )
        _trace(
            state,
            "cross_check",
            f"Detected {len(state.evidence_conflicts)} cross-source evidence conflicts.",
            metadata={
                "backend": cross_check_execution.backend,
                "attempted_backend": cross_check_execution.attempted_backend,
                "conflict_count": str(len(state.evidence_conflicts)),
                **(
                    {"fallback_reason": cross_check_execution.fallback_reason}
                    if cross_check_execution.fallback_reason
                    else {}
                ),
            },
        )

        leaps = detect_invalid_leaps(state)
        _append_unique(state.invalid_leaps, leaps)
        _trace(state, "skeptic", f"Detected {len(leaps)} invalid inference leaps.")

        updates = update_beliefs(state)
        state.belief_updates = updates
        apply_belief_updates(state, updates)
        _trace(state, "belief_update", f"Updated {len(updates)} assumption beliefs.")

    state.decisive_tests = propose_decisive_tests(state)
    verifier_tasks = build_verifier_tasks(
        state.decisive_tests,
        state.sources,
        state.evidence_items,
        state.evidence_conflicts,
    )
    if verifier_tasks:
        verifier_execution = execute_research_tasks(
            verifier_tasks,
            backend=resolved_backend,
            fallback_to_local=modal_fallback,
        )
        _append_unique(state.research_task_results, verifier_execution.results)
        state.verifier_results = [
            verifier_result
            for result in verifier_execution.results
            for verifier_result in result.verifier_results
        ]
        for result in verifier_execution.results:
            _trace(
                state,
                "task",
                f"{result.task_type} {result.task_id} {result.status} via {result.backend}.",
                metadata={
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "backend": result.backend,
                    "status": result.status,
                    "source_ids": ",".join(result.source_ids),
                },
            )
        _trace(
            state,
            "verifier",
            f"Ran {len(state.verifier_results)} decisive-test verifier results.",
            metadata={
                "backend": verifier_execution.backend,
                "attempted_backend": verifier_execution.attempted_backend,
                "verifier_result_count": str(len(state.verifier_results)),
                **(
                    {"fallback_reason": verifier_execution.fallback_reason}
                    if verifier_execution.fallback_reason
                    else {}
                ),
            },
        )
        updates = update_beliefs(state)
        state.belief_updates = updates
        apply_belief_updates(state, updates)
        _trace(
            state,
            "belief_update",
            f"Applied verifier results to {len(updates)} assumption beliefs.",
        )

    state.generated_evals = generate_evals_from_failures(state.invalid_leaps)
    _trace(state, "eval_writer", f"Generated {len(state.generated_evals)} evals from failures.")
    state.eval_workshop = build_eval_workshop(state)
    _trace(
        state,
        "eval_workshop",
        state.eval_workshop.summary,
        metadata={
            "task_spans": str(len(state.eval_workshop.task_spans)),
            "failure_eval_links": str(len(state.eval_workshop.failure_eval_links)),
            "replay_outcomes": str(len(state.eval_workshop.replay_outcomes)),
        },
    )
    state.observability = record_research_run(
        state,
        mode=observability_mode,
        trace_dir=observability_dir,
        fallback_to_local=raindrop_fallback,
    )
    _trace(
        state,
        "observability",
        f"Recorded observability artifact via {state.observability.backend}.",
        metadata={
            "trace_id": state.observability.trace_id,
            "backend": state.observability.backend,
            "status": state.observability.status,
        },
    )
    return state


def decompose_thesis(thesis_text: str) -> list[Assumption]:
    return [
        Assumption(
            id="A1",
            text="Graph memory captures useful scientific structure.",
            why_it_matters="The thesis depends on graph state representing more than a bag of retrieved text.",
            evidence_needed=[
                "Ablations showing graph structure improves scientific reasoning.",
                "Examples where graph relations preserve useful domain structure.",
            ],
        ),
        Assumption(
            id="A2",
            text="The system retrieves better context than standard RAG.",
            why_it_matters="Discovery claims need an advantage over simpler retrieval baselines.",
            evidence_needed=[
                "Head-to-head comparison against strong non-graph RAG.",
                "Recall or answer-quality gains on scientific corpora.",
            ],
        ),
        Assumption(
            id="A3",
            text="The system generates non-obvious hypotheses.",
            why_it_matters="Acceleration requires more than restating known literature.",
            evidence_needed=[
                "Blinded expert novelty ratings.",
                "Time-split evidence that generated hypotheses were not already known.",
            ],
        ),
        Assumption(
            id="A4",
            text="The hypotheses are testable.",
            why_it_matters="Untestable hypotheses cannot drive real discovery.",
            evidence_needed=[
                "Operational predictions with measurable outcomes.",
                "Experimental or simulator protocols tied to hypotheses.",
            ],
        ),
        Assumption(
            id="A5",
            text="Benchmarks correlate with real scientific value.",
            why_it_matters="Benchmark gains are only useful if they predict discovery outcomes.",
            evidence_needed=[
                "Evidence connecting benchmark performance to downstream scientific validation.",
                "Analysis of benchmark construct validity.",
            ],
        ),
        Assumption(
            id="A6",
            text="There is prospective validation.",
            why_it_matters="The strongest discovery claim requires forward-looking validation.",
            evidence_needed=[
                "Predictions made before validation.",
                "Independent experimental or simulator confirmation.",
            ],
        ),
        Assumption(
            id="A7",
            text="The system beats strong baselines.",
            why_it_matters="A complex graph-agent system must outperform simpler alternatives.",
            evidence_needed=[
                "Comparisons against standard RAG, expert search, and human baselines.",
                "Matched budget and tool access.",
            ],
        ),
        Assumption(
            id="A8",
            text="Results generalize beyond cherry-picked examples.",
            why_it_matters="A few demos do not establish broad discovery acceleration.",
            evidence_needed=[
                "Multiple domains or held-out tasks.",
                "Negative controls and failure analysis.",
            ],
        ),
    ]


def generate_initial_questions(state: ResearchState) -> list[ResearchQuestion]:
    return [
        ResearchQuestion(
            id="Q1",
            assumption_ids=["A1", "A2"],
            question="Do graph-based systems retrieve or organize scientific context better than standard RAG?",
            query="graph RAG scientific context retrieval ablation",
            priority=2,
        ),
        ResearchQuestion(
            id="Q2",
            assumption_ids=["A3", "A4"],
            question="Do AI-scientist systems generate novel, testable materials hypotheses?",
            query="AI scientist generated hypotheses novelty testability materials",
            priority=1,
        ),
        ResearchQuestion(
            id="Q3",
            assumption_ids=["A5", "A6"],
            question="Does benchmark performance imply prospective materials discovery?",
            query="scientific discovery benchmark prospective validation proxy evidence",
            priority=1,
        ),
        ResearchQuestion(
            id="Q4",
            assumption_ids=["A7", "A8"],
            question="Do graph-agent results beat strong baselines and generalize beyond selected examples?",
            query="graph agent baseline ablation generalization scientific discovery",
            priority=2,
        ),
    ]


def retrieve_sources(questions: list[ResearchQuestion], corpus: list[Source]) -> list[Source]:
    if not questions:
        return []
    ranked_sources, _scores = rank_sources_for_questions(questions, corpus)
    return ranked_sources


def score_retrieval(
    questions: list[ResearchQuestion],
    corpus: list[Source],
) -> list[RetrievalScore]:
    if not questions:
        return []
    _ranked_sources, scores = rank_sources_for_questions(questions, corpus)
    return scores


def _resolve_execution_backend(
    execution_backend: ExecutionBackend | None,
    extraction_mode: ExtractionMode | None,
) -> ExecutionBackend:
    if execution_backend is not None:
        return execution_backend
    if extraction_mode == "modal":
        return "modal"
    return "local"


def _append_unique(existing: list, additions: list) -> None:
    existing_ids = {_unique_id(item) for item in existing}
    for item in additions:
        item_id = _unique_id(item)
        if item_id not in existing_ids:
            existing.append(item)
            existing_ids.add(item_id)


def _unique_id(item) -> str:
    if hasattr(item, "id"):
        return item.id
    if hasattr(item, "task_id"):
        return item.task_id
    raise AttributeError(f"Cannot append unique item without id: {item!r}")


def _top_source_id(scores: list[RetrievalScore]) -> str:
    if not scores:
        return ""
    top_score = sorted(scores, key=lambda score: (-score.score, score.source_id))[0]
    return top_score.source_id


def _trace(
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
