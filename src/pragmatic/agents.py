from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pragmatic.belief_update import apply_belief_updates, update_beliefs
from pragmatic.corpus import load_corpus, score_corpus_for_questions
from pragmatic.decisive_tests import propose_decisive_tests
from pragmatic.eval_writer import generate_evals_from_failures
from pragmatic.eval_workshop import build_eval_workshop
from pragmatic.execution import (
    build_cross_check_task,
    build_source_extraction_tasks,
    build_source_parse_tasks,
    execute_research_tasks,
)
from pragmatic.extractors import ExtractionMode, extract_evidence
from pragmatic.invalid_leaps import detect_invalid_leaps
from pragmatic.raindrop_client import ObservabilityMode, record_research_run
from pragmatic.replay import run_replay_demo
from pragmatic.research_loop import (
    DEFAULT_THESIS,
    decompose_thesis,
    generate_initial_questions,
    observe_research_loop,
    retrieve_sources,
    run_research_loop,
    score_retrieval,
)
from pragmatic.source_search import build_web_corpus
from pragmatic.schemas import (
    AgentRunRecord,
    AgentRunStep,
    Assumption,
    BeliefUpdate,
    DecisiveTest,
    EvidenceConflict,
    EvidenceItem,
    EvalWorkshopRecord,
    ExecutionBackend,
    GeneratedEval,
    InvalidLeap,
    ObservabilityRecord,
    ResearchBatchResult,
    ResearchQuestion,
    RetrievalScore,
    SourceAcquisitionMode,
    ResearchState,
    Source,
    Thesis,
    TraceEvent,
)
from pragmatic.verifiers import build_verifier_tasks

try:
    from agents import Agent, ModelSettings, Runner, function_tool
    from agents.agent_output import AgentOutputSchema
except ImportError:  # pragma: no cover - exercised only when the SDK is absent.
    Agent = None  # type: ignore[assignment]
    AgentOutputSchema = None  # type: ignore[assignment]
    ModelSettings = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]

try:
    from openai.types.shared import Reasoning
except ImportError:  # pragma: no cover - OpenAI is a required dependency in normal installs.
    Reasoning = None  # type: ignore[assignment]


class AgentsSDKUnavailable(RuntimeError):
    """Raised when live OpenAI Agents SDK objects are requested without the SDK."""


class LiveAgentsSDKNotEnabled(RuntimeError):
    """Raised when a live SDK run is requested without explicit opt-in."""


class AgentsSDKCredentialsError(RuntimeError):
    """Raised when live SDK execution is requested without API credentials."""


class AgentOutputValidationError(RuntimeError):
    """Raised when an SDK run does not produce a valid ResearchState."""


ProgressCallback = Callable[[dict[str, Any]], None]
PartialStateCallback = Callable[[ResearchState], None]
_PROGRESS_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "pragmatic_progress_callback",
    default=None,
)
_PARTIAL_STATE_HOLDER: ContextVar[dict[str, Any] | None] = ContextVar(
    "pragmatic_partial_state_holder",
    default=None,
)

_PARTIAL_LIST_FIELDS = (
    "assumptions",
    "research_questions",
    "sources",
    "retrieval_scores",
    "evidence_items",
    "evidence_conflicts",
    "invalid_leaps",
    "belief_updates",
    "decisive_tests",
    "verifier_results",
    "generated_evals",
    "research_task_results",
    "trace_events",
)


def _emit_progress(
    stage: str,
    status: str,
    message: str,
    **metadata: Any,
) -> None:
    callback = _PROGRESS_CALLBACK.get()
    if callback is None:
        return
    kind = metadata.pop("kind", None)
    callback(
        {
            "stage": stage,
            "status": status,
            "message": message,
            "kind": kind,
            "metadata": metadata,
        }
    )


def _emit_tool_progress(
    tool_name: str,
    status: str,
    message: str,
    **metadata: Any,
) -> None:
    _emit_progress(f"tool.{tool_name}", status, message, tool_name=tool_name, **metadata)


def _emit_cockpit_event(
    kind: str,
    *,
    stage: str,
    status: str = "running",
    message: str,
    **metadata: Any,
) -> None:
    _emit_progress(stage, status, message, kind=kind, **metadata)


def _emit_node_add(
    node_id: str,
    node_kind: str,
    label: str,
    *,
    confidence: float | None = None,
    stage: str = "graph",
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "id": node_id,
        "node_kind": node_kind,
        "label": label[:140],
    }
    if confidence is not None:
        payload["confidence"] = confidence
    payload.update({key: value for key, value in extra.items() if value not in (None, "")})
    _emit_cockpit_event(
        "node.add",
        stage=stage,
        status="created",
        message=f"Added {node_kind} node.",
        **payload,
    )


def _emit_edge_add(
    from_id: str,
    to_id: str,
    relation: str,
    *,
    stage: str = "graph",
) -> None:
    _emit_cockpit_event(
        "edge.add",
        stage=stage,
        status="created",
        message=f"Linked {from_id} to {to_id}.",
        **{"from": from_id, "to": to_id, "relation": relation},
    )


def _emit_counter(
    *,
    sources: int = 0,
    evidence: int = 0,
    leaps: int = 0,
    conflicts: int = 0,
    tests: int = 0,
    stage: str = "counter",
) -> None:
    _emit_cockpit_event(
        "counter",
        stage=stage,
        status="updated",
        message="Updated research counters.",
        sources=sources,
        evidence=evidence,
        leaps=leaps,
        conflicts=conflicts,
        tests=tests,
    )


def _emit_confidence(
    assumption_id: str,
    previous: float,
    current: float,
    reason: str,
    *,
    stage: str = "tool.update_beliefs_tool",
) -> None:
    _emit_cockpit_event(
        "node.confidence",
        stage=stage,
        status="updated",
        message=f"Updated confidence for {assumption_id}.",
        id=assumption_id,
        **{"from": previous, "to": current},
        reason=reason[:220],
    )


def _emit_state_graph_snapshot(state: ResearchState, *, stage: str = "graph") -> None:
    for assumption in state.assumptions:
        _emit_node_add(
            assumption.id,
            "assumption",
            assumption.text,
            confidence=assumption.confidence,
            stage=stage,
        )
    for source in state.sources:
        _emit_node_add(
            source.id,
            "source",
            source.title,
            stage=stage,
            source_type=source.source_type,
            url=source.url,
            citation=source.citation,
        )
    for item in state.evidence_items:
        _emit_node_add(
            item.id,
            "evidence",
            item.claim_supported,
            confidence=item.confidence,
            stage=stage,
        )
        _emit_edge_add(
            item.source_id,
            item.id,
            "contradicts" if item.evidence_type == "contradictory" else "supports",
            stage=stage,
        )
        for assumption_id in item.assumption_ids:
            relation = "proxy_only" if item.evidence_type in {"proxy", "indirect"} else "supports"
            if item.evidence_type == "contradictory":
                relation = "contradicts"
            _emit_edge_add(item.id, assumption_id, relation, stage=stage)
    for test in state.decisive_tests:
        _emit_node_add(test.id, "test", test.test, stage=stage)
        for assumption_id in test.would_resolve:
            _emit_edge_add(assumption_id, test.id, "tests", stage=stage)
    _emit_counter(
        sources=len(state.sources),
        evidence=len(state.evidence_items),
        leaps=len(state.invalid_leaps),
        conflicts=len(state.evidence_conflicts),
        tests=len(state.decisive_tests),
        stage=stage,
    )


def _partial_state() -> ResearchState | None:
    holder = _PARTIAL_STATE_HOLDER.get()
    if holder is None:
        return None
    state = holder.get("state")
    return state if isinstance(state, ResearchState) else None


def _publish_partial_state() -> None:
    holder = _PARTIAL_STATE_HOLDER.get()
    if holder is None:
        return
    state = holder.get("state")
    callback = holder.get("callback")
    if isinstance(state, ResearchState) and callback is not None:
        callback(state.model_copy(deep=True))


def _merge_partial_state(candidate: ResearchState) -> None:
    state = _partial_state()
    if state is None:
        return
    if candidate.thesis.text and state.thesis.text != candidate.thesis.text:
        state.thesis = candidate.thesis
    state.iteration = max(state.iteration, candidate.iteration)
    for field in _PARTIAL_LIST_FIELDS:
        additions = getattr(candidate, field)
        if additions:
            _append_unique(getattr(state, field), additions)
    if candidate.observability is not None:
        state.observability = candidate.observability
    if candidate.eval_workshop is not None:
        state.eval_workshop = candidate.eval_workshop
    if candidate.agent_run is not None:
        state.agent_run = candidate.agent_run
    _publish_partial_state()


def _record_partial_list(field: str, values: list[Any]) -> None:
    state = _partial_state()
    if state is None or not values:
        return
    _append_unique(getattr(state, field), values)
    _publish_partial_state()


def _record_partial_batch(result: ResearchBatchResult) -> None:
    state = _partial_state()
    if state is None:
        return
    _append_unique(state.research_task_results, result.results)
    _append_unique(
        state.sources,
        [source for task_result in result.results for source in task_result.sources],
    )
    _append_unique(
        state.evidence_items,
        [item for task_result in result.results for item in task_result.evidence_items],
    )
    _append_unique(
        state.evidence_conflicts,
        [conflict for task_result in result.results for conflict in task_result.evidence_conflicts],
    )
    _append_unique(
        state.verifier_results,
        [verifier for task_result in result.results for verifier in task_result.verifier_results],
    )
    _publish_partial_state()


RESEARCH_MANAGER_INSTRUCTIONS = """
You are ResearchManager for Pragmatic.

Own the research loop and keep the output grounded in typed ResearchState artifacts.
Use tools to decompose the thesis, plan questions, retrieve local corpus sources,
execute local or Modal research tasks, extract evidence, cross-check evidence,
run decisive-test verifiers, detect invalid inference leaps, update beliefs,
propose decisive tests, and generate evals from reasoning failures.

Do not present proxy benchmark evidence as direct evidence of real-world discovery.
Keep retrieval/reasoning support separate from actual autonomous-discovery support.
Return only a valid ResearchState object. Do not invent fields outside the schema.
""".strip()


@dataclass(frozen=True)
class ResearchManager:
    """Facade for deterministic and live SDK orchestration."""

    model: str | None = None
    max_turns: int = 3

    def run_deterministic(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        modal_fallback: bool = True,
        source_mode: SourceAcquisitionMode = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str | None = None,
        max_web_sources: int = 8,
        observability_mode: ObservabilityMode = "local",
    ) -> ResearchState:
        return run_research_loop(
            thesis_text,
            max_iterations=max_iterations,
            corpus_path=corpus_path,
            execution_backend=execution_backend,
            extraction_mode=extraction_mode,
            modal_fallback=modal_fallback,
            source_mode=source_mode,
            allow_live_web_search=allow_live_web_search,
            web_search_model=web_search_model,
            max_web_sources=max_web_sources,
            observability_mode=observability_mode,
        )

    def finalize_partial_state(
        self,
        state: ResearchState,
        *,
        execution_backend: ExecutionBackend,
        source_mode: SourceAcquisitionMode = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str | None = None,
        max_web_sources: int = 8,
        observability_mode: ObservabilityMode = "off",
    ) -> ResearchState:
        return _finalize_live_research_state(
            state,
            execution_backend=execution_backend,
            source_mode=source_mode,
            allow_live_web_search=allow_live_web_search,
            web_search_model=web_search_model,
            max_web_sources=max_web_sources,
            observability_mode=observability_mode,
        )

    def run_sdk_orchestrated(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        source_mode: SourceAcquisitionMode = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str | None = None,
        max_web_sources: int = 8,
        observability_mode: ObservabilityMode = "local",
    ) -> ResearchState:
        _require_agents_sdk()
        agent = self.build_agent()
        tools = {tool.name: tool for tool in agent.tools}
        state = ResearchState(thesis=Thesis(text=thesis_text, domain=thesis_text[:96]))
        _append_agent_trace(
            state,
            "OpenAI Agents SDK ResearchManager initialized specialist orchestration.",
            metadata={"mode": "scripted_sdk", "agent": agent.name},
        )

        step_records: list[AgentRunStep] = []
        corpus_path_value = str(corpus_path) if corpus_path else ""
        execution_backend_value = execution_backend or ("modal" if extraction_mode == "modal" else "")
        resolved_backend = _validate_execution_backend(execution_backend_value)
        search_model_value = web_search_model or self.model or ""

        assumptions_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="AssumptionDecomposer",
            tool_name="decompose_thesis_tool",
            arguments={"thesis_text": thesis_text},
            summary="Decomposed thesis into explicit assumptions.",
        )
        state.assumptions = [
            Assumption.model_validate(assumption)
            for assumption in json.loads(assumptions_json)
        ]

        questions_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="QueryPlanner",
            tool_name="plan_questions_tool",
            arguments={"state_json": state.model_dump_json()},
            summary="Planned initial research questions from assumptions.",
        )
        state.research_questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(questions_json)
        ]

        for iteration in range(1, max_iterations + 1):
            state.iteration = iteration
            open_questions = [
                question
                for question in state.research_questions
                if question.status == "open"
            ]
            if not open_questions:
                _append_agent_trace(
                    state,
                    "ResearchManager stopped because no open questions remain.",
                    metadata={"iteration": str(iteration)},
                )
                break

            open_questions_json = _to_json([question.model_dump() for question in open_questions])
            sources_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="RetrieverTool",
                tool_name="retrieve_sources_tool",
                arguments={
                    "research_questions_json": open_questions_json,
                    "corpus_path": corpus_path_value,
                    "thesis_text": thesis_text,
                    "source_mode": source_mode,
                    "allow_live_web_search": allow_live_web_search,
                    "web_search_model": search_model_value,
                    "max_web_sources": max_web_sources,
                },
                summary="Retrieved sources for open questions.",
            )
            retrieved_sources = [
                Source.model_validate(source)
                for source in json.loads(sources_json)
            ]
            _append_unique(state.sources, retrieved_sources)

            scores_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="RetrievalScorer",
                tool_name="score_retrieval_tool",
                arguments={
                    "research_questions_json": open_questions_json,
                    "corpus_path": corpus_path_value,
                    "sources_json": sources_json,
                },
                summary="Scored retrieval matches.",
            )
            retrieval_scores = [
                RetrievalScore.model_validate(score)
                for score in json.loads(scores_json)
            ]
            _append_unique(state.retrieval_scores, retrieval_scores)
            for question in open_questions:
                question.status = "answered"

            extraction_result_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="EvidenceExtractor",
                tool_name="execute_source_research_tasks_tool",
                arguments={
                    "sources_json": sources_json,
                    "assumptions_json": assumptions_json,
                    "research_questions_json": open_questions_json,
                    "execution_backend": execution_backend_value,
                    "fallback_to_local": True,
                },
                summary="Executed source extraction tasks through the selected backend.",
            )
            extraction_result = ResearchBatchResult.model_validate_json(extraction_result_json)
            _append_unique(state.research_task_results, extraction_result.results)
            extracted_items = [
                item
                for result in extraction_result.results
                for item in result.evidence_items
            ]
            _append_unique(state.evidence_items, sorted(extracted_items, key=lambda item: item.id))

            cross_check_result_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="Skeptic",
                tool_name="cross_check_evidence_tool",
                arguments={
                    "sources_json": _to_json([source.model_dump() for source in state.sources]),
                    "evidence_items_json": _to_json([item.model_dump() for item in state.evidence_items]),
                    "assumptions_json": assumptions_json,
                    "execution_backend": execution_backend_value,
                    "fallback_to_local": True,
                },
                summary="Cross-checked evidence across sources before belief updates.",
            )
            cross_check_result = ResearchBatchResult.model_validate_json(cross_check_result_json)
            _append_unique(state.research_task_results, cross_check_result.results)
            state.evidence_conflicts = [
                conflict
                for result in cross_check_result.results
                for conflict in result.evidence_conflicts
            ]

            leaps_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="Skeptic",
                tool_name="detect_invalid_leaps_tool",
                arguments={"state_json": state.model_dump_json()},
                summary="Detected invalid inference leaps in the current research state.",
            )
            _append_unique(
                state.invalid_leaps,
                [InvalidLeap.model_validate(leap) for leap in json.loads(leaps_json)],
            )

            updates_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="BeliefUpdater",
                tool_name="update_beliefs_tool",
                arguments={"state_json": state.model_dump_json()},
                summary="Updated assumption support and confidence from typed evidence.",
            )
            updates = [
                BeliefUpdate.model_validate(update)
                for update in json.loads(updates_json)
            ]
            state.belief_updates = updates
            apply_belief_updates(state, updates)

        decisive_tests_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="DecisiveTestWriter",
            tool_name="propose_decisive_tests_tool",
            arguments={"state_json": state.model_dump_json()},
            summary="Proposed decisive tests for the remaining uncertainty.",
        )
        state.decisive_tests = [
            DecisiveTest.model_validate(test)
            for test in json.loads(decisive_tests_json)
        ]

        verifier_result_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="DecisiveTestVerifier",
            tool_name="run_decisive_test_verifiers_tool",
            arguments={
                "decisive_tests_json": decisive_tests_json,
                "sources_json": _to_json([source.model_dump() for source in state.sources]),
                "evidence_items_json": _to_json([item.model_dump() for item in state.evidence_items]),
                "evidence_conflicts_json": _to_json([conflict.model_dump() for conflict in state.evidence_conflicts]),
                "execution_backend": execution_backend_value,
                "fallback_to_local": True,
            },
            summary="Ran decisive-test verifier tasks through the selected backend.",
        )
        verifier_result = ResearchBatchResult.model_validate_json(verifier_result_json)
        _append_unique(state.research_task_results, verifier_result.results)
        state.verifier_results = [
            result
            for task_result in verifier_result.results
            for result in task_result.verifier_results
        ]
        if state.verifier_results:
            verifier_updates_json = _invoke_specialist_tool(
                tools,
                step_records,
                agent_name="BeliefUpdater",
                tool_name="update_beliefs_tool",
                arguments={"state_json": state.model_dump_json()},
                summary="Applied decisive-test verifier results to belief confidence.",
            )
            verifier_updates = [
                BeliefUpdate.model_validate(update)
                for update in json.loads(verifier_updates_json)
            ]
            state.belief_updates = verifier_updates
            apply_belief_updates(state, verifier_updates)

        evals_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="EvalWriter",
            tool_name="generate_evals_from_failures_tool",
            arguments={
                "invalid_leaps_json": _to_json([leap.model_dump() for leap in state.invalid_leaps])
            },
            summary="Generated eval artifacts from detected reasoning failures.",
        )
        state.generated_evals = [
            GeneratedEval.model_validate(generated_eval)
            for generated_eval in json.loads(evals_json)
        ]

        state.agent_run = AgentRunRecord(
            mode="scripted_sdk",
            status="succeeded",
            agent_name=agent.name,
            model=self.model,
            tool_names=sorted(tools),
            steps=list(step_records),
            final_output_validated=False,
            message="Offline SDK-scripted orchestration is building workshop artifacts.",
        )
        workshop_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="RaindropWorkshop",
            tool_name="build_eval_workshop_tool",
            arguments={"state_json": state.model_dump_json()},
            summary="Built the failure-to-eval workshop links and task spans.",
        )
        state.eval_workshop = EvalWorkshopRecord.model_validate_json(workshop_json)
        state.agent_run = state.agent_run.model_copy(update={"steps": list(step_records)})

        observability_json = _invoke_specialist_tool(
            tools,
            step_records,
            agent_name="RaindropWorkshop",
            tool_name="record_observability_tool",
            arguments={
                "state_json": state.model_dump_json(),
                "observability_mode": observability_mode,
            },
            summary="Recorded trace and workshop artifacts through the observability adapter.",
        )
        state.observability = ObservabilityRecord.model_validate_json(observability_json)
        state.agent_run = AgentRunRecord(
            mode="scripted_sdk",
            status="succeeded",
            agent_name=agent.name,
            model=self.model,
            tool_names=sorted(tools),
            steps=step_records,
            final_output_validated=True,
            message="Offline SDK-scripted orchestration ran specialist agents/tools instead of the canonical loop tool.",
        )
        state.eval_workshop = build_eval_workshop(state)
        _append_agent_trace(
            state,
            "OpenAI Agents SDK specialist orchestration validated the final ResearchState.",
            metadata={
                "mode": "scripted_sdk",
                "step_count": str(len(step_records)),
                "execution_backend": resolved_backend,
            },
        )
        return state

    def build_agent(self) -> Any:
        _require_agents_sdk()
        model_settings = _reasoning_summary_model_settings()
        agent_kwargs: dict[str, Any] = {}
        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings
        return Agent(
            name="ResearchManager",
            instructions=RESEARCH_MANAGER_INSTRUCTIONS,
            model=self.model,
            tools=build_research_tools(),
            output_type=AgentOutputSchema(ResearchState, strict_json_schema=False),
            **agent_kwargs,
        )

    def run_live_sync(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        source_mode: SourceAcquisitionMode = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str | None = None,
        max_web_sources: int = 8,
        observability_mode: ObservabilityMode = "off",
        allow_live_sdk: bool = False,
        progress_callback: ProgressCallback | None = None,
        partial_state_callback: PartialStateCallback | None = None,
    ) -> ResearchState:
        return _run_awaitable(
            self.run_live(
                thesis_text,
                max_iterations=max_iterations,
                corpus_path=corpus_path,
                execution_backend=execution_backend,
                extraction_mode=extraction_mode,
                source_mode=source_mode,
                allow_live_web_search=allow_live_web_search,
                web_search_model=web_search_model,
                max_web_sources=max_web_sources,
                observability_mode=observability_mode,
                allow_live_sdk=allow_live_sdk,
                progress_callback=progress_callback,
                partial_state_callback=partial_state_callback,
            )
        )

    async def run_live(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        source_mode: SourceAcquisitionMode = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str | None = None,
        max_web_sources: int = 8,
        observability_mode: ObservabilityMode = "off",
        allow_live_sdk: bool = False,
        progress_callback: ProgressCallback | None = None,
        partial_state_callback: PartialStateCallback | None = None,
    ) -> ResearchState:
        _require_live_sdk_enabled(allow_live_sdk)
        _require_agents_sdk()
        agent = self.build_agent()
        execution_backend_value = execution_backend or ""
        corpus_path_value = str(corpus_path) if corpus_path else ""
        search_policy = (
            "Use retrieve_sources_tool with source_mode='web' when acquiring sources. "
            "Live web search is explicitly allowed for this run. "
            if source_mode == "web" and allow_live_web_search
            else "Do not perform live web search or add sources outside the prepared corpus. "
        )
        prompt = (
            "Run the Pragmatic research loop for this thesis. "
            "Make one brief strategy decision, then call run_deterministic_research_loop_tool once "
            "with the run settings below and return the schema-valid ResearchState it produces. "
            "The individual specialist tools remain available for exceptional cases, but do not "
            "orchestrate them sequentially by default. "
            "If any invalid_leaps are present in the final state, generated_evals must be non-empty. "
            f"{search_policy}"
            "Return the final schema-valid ResearchState object.\n\n"
            f"max_iterations: {max_iterations}\n"
            f"corpus_path: {corpus_path_value}\n"
            f"execution_backend: {execution_backend_value}\n"
            f"extraction_mode: {extraction_mode}\n"
            f"source_mode: {source_mode}\n"
            f"allow_live_web_search: {allow_live_web_search}\n"
            f"web_search_model: {web_search_model or self.model or ''}\n"
            f"max_web_sources: {max_web_sources}\n"
            f"observability_mode: {observability_mode}\n\n"
            f"Thesis: {thesis_text}"
        )
        partial_holder: dict[str, Any] = {
            "state": ResearchState(thesis=Thesis(text=thesis_text, domain=thesis_text[:96])),
            "callback": partial_state_callback,
        }
        token = _PROGRESS_CALLBACK.set(progress_callback)
        partial_token = _PARTIAL_STATE_HOLDER.set(partial_holder)
        try:
            _publish_partial_state()
            _emit_progress(
                "live_sdk",
                "running",
                "OpenAI Agents SDK runner started.",
                model=self.model or "",
                max_turns=self.max_turns,
            )
            result = Runner.run_streamed(agent, input=prompt, max_turns=self.max_turns)
            delta_buffer: dict[str, Any] = {"chunks": [], "last_flush": time.monotonic()}
            async for event in result.stream_events():
                _emit_stream_event(event, delta_buffer)
            _flush_reasoning_delta(delta_buffer)
            _emit_progress(
                "live_sdk",
                "succeeded",
                "OpenAI Agents SDK runner returned a final output.",
            )
        finally:
            _PARTIAL_STATE_HOLDER.reset(partial_token)
            _PROGRESS_CALLBACK.reset(token)
        state = _coerce_research_state(result)
        state.agent_run = AgentRunRecord(
            mode="live_sdk",
            status="succeeded",
            agent_name=agent.name,
            model=self.model,
            tool_names=sorted(tool.name for tool in agent.tools),
            steps=[
                AgentRunStep(
                    id="agent_step_001",
                    tool_name="Runner.run",
                    status="succeeded",
                    summary="Live OpenAI Agents SDK runner returned a valid ResearchState.",
                )
            ],
            final_output_validated=True,
            message="Live OpenAI Agents SDK run returned a schema-validated ResearchState.",
        )
        state = _finalize_live_research_state(
            state,
            execution_backend=_validate_execution_backend(execution_backend_value),
            source_mode=source_mode,
            allow_live_web_search=allow_live_web_search,
            web_search_model=web_search_model or self.model,
            max_web_sources=max_web_sources,
            observability_mode=observability_mode,
        )
        _append_agent_trace(
            state,
            "OpenAI Agents SDK live orchestration validated the final ResearchState.",
            metadata={"mode": "live_sdk"},
        )
        return state


def build_research_tools() -> list[Any]:
    _require_agents_sdk()
    return [
        decompose_thesis_tool,
        plan_questions_tool,
        retrieve_sources_tool,
        score_retrieval_tool,
        extract_evidence_tool,
        execute_source_research_tasks_tool,
        cross_check_evidence_tool,
        detect_invalid_leaps_tool,
        update_beliefs_tool,
        propose_decisive_tests_tool,
        run_decisive_test_verifiers_tool,
        generate_evals_from_failures_tool,
        build_eval_workshop_tool,
        record_observability_tool,
        run_deterministic_research_loop_tool,
        run_replay_demo_tool,
    ]


def _require_agents_sdk() -> None:
    if Agent is None or AgentOutputSchema is None or Runner is None or function_tool is None:
        raise AgentsSDKUnavailable(
            "Install the OpenAI Agents SDK with `pip install openai-agents` to use live orchestration."
        )


def _reasoning_summary_model_settings() -> Any | None:
    if ModelSettings is None or Reasoning is None:
        return None
    try:
        return ModelSettings(reasoning=Reasoning(summary="auto"))
    except Exception:
        try:
            return ModelSettings(reasoning=Reasoning(generate_summary="auto"))
        except Exception:
            return None


def _require_live_sdk_enabled(allow_live_sdk: bool) -> None:
    if not allow_live_sdk:
        raise LiveAgentsSDKNotEnabled(
            "Live OpenAI Agents SDK execution requires explicit opt-in with allow_live_sdk=True."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise AgentsSDKCredentialsError(
            "Live OpenAI Agents SDK execution requires OPENAI_API_KEY."
        )


if function_tool is not None:

    @function_tool
    def decompose_thesis_tool(thesis_text: str) -> str:
        """Decompose a thesis into Pragmatic assumption objects as JSON."""

        _emit_tool_progress("decompose_thesis_tool", "running", "Decomposing the question into assumptions.")
        assumptions = decompose_thesis(thesis_text)
        _record_partial_list("assumptions", assumptions)
        for assumption in assumptions:
            _emit_node_add(
                assumption.id,
                "assumption",
                assumption.text,
                confidence=assumption.confidence,
                stage="tool.decompose_thesis_tool",
            )
        _emit_counter(stage="tool.decompose_thesis_tool")
        _emit_tool_progress(
            "decompose_thesis_tool",
            "succeeded",
            f"Created {len(assumptions)} assumptions.",
            assumption_count=len(assumptions),
        )
        return _to_json([assumption.model_dump() for assumption in assumptions])

    @function_tool
    def plan_questions_tool(state_json: str) -> str:
        """Generate initial research questions from a ResearchState JSON string."""

        _emit_tool_progress("plan_questions_tool", "running", "Planning evidence questions.")
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        questions = generate_initial_questions(state)
        _record_partial_list("research_questions", questions)
        for question in questions:
            for assumption_id in question.assumption_ids:
                _emit_edge_add(
                    assumption_id,
                    question.id,
                    "asks",
                    stage="tool.plan_questions_tool",
                )
        _emit_tool_progress(
            "plan_questions_tool",
            "succeeded",
            f"Planned {len(questions)} research questions.",
            question_count=len(questions),
        )
        return _to_json([question.model_dump() for question in questions])

    @function_tool
    def retrieve_sources_tool(
        research_questions_json: str,
        corpus_path: str = "",
        thesis_text: str = "",
        source_mode: str = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str = "",
        max_web_sources: int = 8,
    ) -> str:
        """Retrieve prepared or live-web sources for research questions and return Source JSON."""

        _emit_tool_progress(
            "retrieve_sources_tool",
            "running",
            "Retrieving evidence sources.",
            source_mode=source_mode,
            allow_live_web_search=allow_live_web_search,
        )
        questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(research_questions_json)
        ]
        if _validate_source_mode(source_mode) == "web":
            if not allow_live_web_search:
                raise ValueError("source_mode='web' requires allow_live_web_search=True.")
            corpus = build_web_corpus(
                thesis_text,
                questions,
                model=web_search_model or None,
                max_sources=max_web_sources,
            )
        else:
            from pragmatic.corpus import resolve_corpus_path

            corpus = load_corpus(resolve_corpus_path(thesis_text, corpus_path or None))
        sources = retrieve_sources(questions, corpus)
        _record_partial_list("sources", sources)
        for source in sources:
            _emit_node_add(
                source.id,
                "source",
                source.title,
                stage="tool.retrieve_sources_tool",
                source_type=source.source_type,
                url=source.url,
                citation=source.citation,
            )
        _emit_counter(sources=len(sources), stage="tool.retrieve_sources_tool")
        _emit_tool_progress(
            "retrieve_sources_tool",
            "succeeded",
            f"Retrieved {len(sources)} sources.",
            source_count=len(sources),
        )
        return _to_json([source.model_dump() for source in sources])

    @function_tool
    def score_retrieval_tool(
        research_questions_json: str,
        corpus_path: str = "",
        sources_json: str = "",
    ) -> str:
        """Score retrieval matches and return RetrievalScore JSON."""

        _emit_tool_progress("score_retrieval_tool", "running", "Scoring source relevance.")
        questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(research_questions_json)
        ]
        if sources_json:
            corpus = [Source.model_validate(source) for source in json.loads(sources_json)]
            scores = score_corpus_for_questions(questions, corpus)
        else:
            corpus = load_corpus(corpus_path or None)
            scores = score_retrieval(questions, corpus)
        _record_partial_list("retrieval_scores", scores)
        _emit_tool_progress(
            "score_retrieval_tool",
            "succeeded",
            f"Recorded {len(scores)} retrieval scores.",
            score_count=len(scores),
        )
        return _to_json([score.model_dump() for score in scores])

    @function_tool
    def extract_evidence_tool(sources_json: str, assumptions_json: str) -> str:
        """Extract typed evidence items from source and assumption JSON."""

        _emit_tool_progress("extract_evidence_tool", "running", "Extracting typed evidence.")
        sources = [Source.model_validate(source) for source in json.loads(sources_json)]
        assumptions = [
            Assumption.model_validate(assumption)
            for assumption in json.loads(assumptions_json)
        ]
        evidence = extract_evidence(sources, assumptions)
        _record_partial_list("evidence_items", evidence)
        _emit_tool_progress(
            "extract_evidence_tool",
            "succeeded",
            f"Extracted {len(evidence)} evidence items.",
            evidence_count=len(evidence),
        )
        return _to_json([item.model_dump() for item in evidence])

    @function_tool
    def execute_source_research_tasks_tool(
        sources_json: str,
        assumptions_json: str,
        research_questions_json: str,
        execution_backend: str = "",
        fallback_to_local: bool = True,
    ) -> str:
        """Fan out source extraction ResearchTask objects and return ResearchBatchResult JSON."""

        _emit_tool_progress(
            "execute_source_research_tasks_tool",
            "running",
            "Fanning out parse and extraction tasks.",
            execution_backend=execution_backend,
        )
        sources = [Source.model_validate(source) for source in json.loads(sources_json)]
        assumptions = [
            Assumption.model_validate(assumption)
            for assumption in json.loads(assumptions_json)
        ]
        questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(research_questions_json)
        ]
        parse_tasks = build_source_parse_tasks(sources, questions)
        parse_result = execute_research_tasks(
            parse_tasks,
            backend=_validate_execution_backend(execution_backend),
            fallback_to_local=fallback_to_local,
        )
        parsed_sources = [
            source
            for task_result in parse_result.results
            if task_result.status == "succeeded"
            for source in task_result.sources
        ] or sources
        extraction_tasks = build_source_extraction_tasks(parsed_sources, assumptions, questions)
        extraction_result = execute_research_tasks(
            extraction_tasks,
            backend=_validate_execution_backend(execution_backend),
            fallback_to_local=fallback_to_local,
        )
        fallback_reasons = [
            reason
            for reason in [parse_result.fallback_reason, extraction_result.fallback_reason]
            if reason
        ]
        combined_results = [*parse_result.results, *extraction_result.results]
        source_title_by_id = {source.id: source.title for source in parsed_sources}
        result = ResearchBatchResult(
            backend=extraction_result.backend,
            attempted_backend=extraction_result.attempted_backend,
            results=combined_results,
            fallback_reason="; ".join(fallback_reasons) or None,
            metadata={
                "task_count": str(len(combined_results)),
                "parse_task_count": str(len(parse_tasks)),
                "extract_task_count": str(len(extraction_tasks)),
                "source_count": str(len(parsed_sources)),
                "backend_sequence": f"{parse_result.backend}->{extraction_result.backend}",
            },
        )
        _record_partial_batch(result)
        _emit_cockpit_event(
            "fanout.spawn",
            stage="tool.execute_source_research_tasks_tool",
            status="running",
            message=f"Spawned {len(combined_results)} research workers.",
            tasks=len(combined_results),
            backend=result.attempted_backend,
        )
        for task_result in combined_results:
            source_id = task_result.source_ids[0] if task_result.source_ids else ""
            _emit_cockpit_event(
                "fanout.task",
                stage="tool.execute_source_research_tasks_tool",
                status="succeeded" if task_result.status == "succeeded" else "failed",
                message=f"{task_result.task_id} {task_result.status}.",
                task_id=task_result.task_id,
                task_type=task_result.task_type,
                source_id=source_id,
                source_title=source_title_by_id.get(source_id, ""),
                task_status=task_result.status,
                evidence_count=len(task_result.evidence_items),
            )
            for source in task_result.sources:
                _emit_node_add(
                    source.id,
                    "source",
                    source.title,
                    stage="tool.execute_source_research_tasks_tool",
                    source_type=source.source_type,
                    url=source.url,
                    citation=source.citation,
                )
            for item in task_result.evidence_items:
                _emit_node_add(
                    item.id,
                    "evidence",
                    item.claim_supported,
                    confidence=item.confidence,
                    stage="tool.execute_source_research_tasks_tool",
                )
                _emit_edge_add(
                    item.source_id,
                    item.id,
                    "contradicts" if item.evidence_type == "contradictory" else "supports",
                    stage="tool.execute_source_research_tasks_tool",
                )
                for assumption_id in item.assumption_ids:
                    relation = "proxy_only" if item.evidence_type in {"proxy", "indirect"} else "supports"
                    if item.evidence_type == "contradictory":
                        relation = "contradicts"
                    _emit_edge_add(item.id, assumption_id, relation, stage="tool.execute_source_research_tasks_tool")
        _emit_counter(
            sources=len(parsed_sources),
            evidence=sum(len(task_result.evidence_items) for task_result in combined_results),
            stage="tool.execute_source_research_tasks_tool",
        )
        _emit_tool_progress(
            "execute_source_research_tasks_tool",
            "succeeded",
            f"Completed {len(combined_results)} research tasks.",
            task_count=len(combined_results),
            backend=result.backend,
        )
        return result.model_dump_json()

    @function_tool
    def cross_check_evidence_tool(
        sources_json: str,
        evidence_items_json: str,
        assumptions_json: str,
        execution_backend: str = "",
        fallback_to_local: bool = True,
    ) -> str:
        """Cross-check extracted evidence and return ResearchBatchResult JSON."""

        _emit_tool_progress(
            "cross_check_evidence_tool",
            "running",
            "Checking evidence across sources.",
            execution_backend=execution_backend,
        )
        sources = [Source.model_validate(source) for source in json.loads(sources_json)]
        evidence_items = [
            EvidenceItem.model_validate(item)
            for item in json.loads(evidence_items_json)
        ]
        assumptions = [
            Assumption.model_validate(assumption)
            for assumption in json.loads(assumptions_json)
        ]
        task = build_cross_check_task(sources, evidence_items, assumptions)
        result = execute_research_tasks(
            [task],
            backend=_validate_execution_backend(execution_backend),
            fallback_to_local=fallback_to_local,
        )
        _record_partial_batch(result)
        conflicts = [
            conflict
            for task_result in result.results
            for conflict in task_result.evidence_conflicts
        ]
        _emit_counter(
            sources=len(sources),
            evidence=len(evidence_items),
            conflicts=len(conflicts),
            stage="tool.cross_check_evidence_tool",
        )
        _emit_tool_progress(
            "cross_check_evidence_tool",
            "succeeded",
            "Cross-check task completed.",
            task_count=len(result.results),
        )
        return result.model_dump_json()

    @function_tool
    def detect_invalid_leaps_tool(state_json: str) -> str:
        """Detect invalid inference leaps from a ResearchState JSON string."""

        _emit_tool_progress("detect_invalid_leaps_tool", "running", "Looking for invalid inference leaps.")
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        leaps = detect_invalid_leaps(state)
        _record_partial_list("invalid_leaps", leaps)
        _emit_counter(
            sources=len(state.sources),
            evidence=len(state.evidence_items),
            leaps=len(leaps),
            conflicts=len(state.evidence_conflicts),
            tests=len(state.decisive_tests),
            stage="tool.detect_invalid_leaps_tool",
        )
        _emit_tool_progress(
            "detect_invalid_leaps_tool",
            "succeeded",
            f"Detected {len(leaps)} invalid leaps.",
            invalid_leap_count=len(leaps),
        )
        return _to_json([leap.model_dump() for leap in leaps])

    @function_tool
    def update_beliefs_tool(state_json: str) -> str:
        """Compute belief updates from a ResearchState JSON string."""

        _emit_tool_progress("update_beliefs_tool", "running", "Updating the belief graph.")
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        updates = update_beliefs(state)
        partial = _partial_state()
        if partial is not None:
            partial.belief_updates = updates
            apply_belief_updates(partial, updates)
            _publish_partial_state()
        for update in updates:
            _emit_confidence(
                update.assumption_id,
                update.previous_confidence,
                update.new_confidence,
                update.rationale,
            )
        _emit_tool_progress(
            "update_beliefs_tool",
            "succeeded",
            f"Applied {len(updates)} belief updates.",
            belief_update_count=len(updates),
        )
        return _to_json([update.model_dump() for update in updates])

    @function_tool
    def propose_decisive_tests_tool(state_json: str) -> str:
        """Propose decisive tests from a ResearchState JSON string."""

        _emit_tool_progress("propose_decisive_tests_tool", "running", "Proposing decisive follow-up tests.")
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        tests = propose_decisive_tests(state)
        _record_partial_list("decisive_tests", tests)
        for test in tests:
            _emit_node_add(test.id, "test", test.test, stage="tool.propose_decisive_tests_tool")
            for assumption_id in test.would_resolve:
                _emit_edge_add(assumption_id, test.id, "tests", stage="tool.propose_decisive_tests_tool")
        _emit_counter(
            sources=len(state.sources),
            evidence=len(state.evidence_items),
            leaps=len(state.invalid_leaps),
            conflicts=len(state.evidence_conflicts),
            tests=len(tests),
            stage="tool.propose_decisive_tests_tool",
        )
        _emit_tool_progress(
            "propose_decisive_tests_tool",
            "succeeded",
            f"Proposed {len(tests)} decisive tests.",
            decisive_test_count=len(tests),
        )
        return _to_json([test.model_dump() for test in tests])

    @function_tool
    def run_decisive_test_verifiers_tool(
        decisive_tests_json: str,
        sources_json: str,
        evidence_items_json: str,
        evidence_conflicts_json: str,
        execution_backend: str = "",
        fallback_to_local: bool = True,
    ) -> str:
        """Run decisive-test verifier tasks and return ResearchBatchResult JSON."""

        _emit_tool_progress(
            "run_decisive_test_verifiers_tool",
            "running",
            "Running verifier tasks.",
            execution_backend=execution_backend,
        )
        tests = [
            DecisiveTest.model_validate(test)
            for test in json.loads(decisive_tests_json)
        ]
        sources = [Source.model_validate(source) for source in json.loads(sources_json)]
        evidence_items = [
            EvidenceItem.model_validate(item)
            for item in json.loads(evidence_items_json)
        ]
        evidence_conflicts = [
            EvidenceConflict.model_validate(conflict)
            for conflict in json.loads(evidence_conflicts_json)
        ]
        tasks = build_verifier_tasks(
            tests,
            sources,
            evidence_items,
            evidence_conflicts,
        )
        result = execute_research_tasks(
            tasks,
            backend=_validate_execution_backend(execution_backend),
            fallback_to_local=fallback_to_local,
        )
        _record_partial_batch(result)
        _emit_tool_progress(
            "run_decisive_test_verifiers_tool",
            "succeeded",
            f"Verifier batch completed with {len(result.results)} task results.",
            task_count=len(result.results),
        )
        return result.model_dump_json()

    @function_tool
    def generate_evals_from_failures_tool(invalid_leaps_json: str) -> str:
        """Generate eval artifacts from invalid leap JSON."""

        _emit_tool_progress("generate_evals_from_failures_tool", "running", "Generating evals from failures.")
        invalid_leaps = [
            InvalidLeap.model_validate(invalid_leap)
            for invalid_leap in json.loads(invalid_leaps_json)
        ]
        evals = generate_evals_from_failures(invalid_leaps)
        _record_partial_list("generated_evals", evals)
        for generated_eval in evals:
            _emit_node_add(
                generated_eval.id,
                "eval",
                generated_eval.eval_rule,
                stage="tool.generate_evals_from_failures_tool",
            )
            if generated_eval.source_failure_id:
                _emit_edge_add(
                    generated_eval.source_failure_id,
                    generated_eval.id,
                    "becomes_eval",
                    stage="tool.generate_evals_from_failures_tool",
                )
        _emit_tool_progress(
            "generate_evals_from_failures_tool",
            "succeeded",
            f"Generated {len(evals)} evals.",
            generated_eval_count=len(evals),
        )
        return _to_json([generated_eval.model_dump() for generated_eval in evals])

    @function_tool
    def build_eval_workshop_tool(state_json: str) -> str:
        """Build failure-to-eval workshop links and task-span artifacts."""

        _emit_tool_progress("build_eval_workshop_tool", "running", "Building the Workshop connection graph.")
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        workshop = build_eval_workshop(state)
        partial = _partial_state()
        if partial is not None:
            partial.eval_workshop = workshop
            _publish_partial_state()
        _emit_tool_progress(
            "build_eval_workshop_tool",
            "succeeded",
            "Workshop connection graph ready.",
            connection_count=len(workshop.connection_rows),
        )
        return workshop.model_dump_json()

    @function_tool
    def record_observability_tool(
        state_json: str,
        observability_mode: str = "local",
    ) -> str:
        """Record trace/workshop observability artifacts and return ObservabilityRecord JSON."""

        _emit_tool_progress(
            "record_observability_tool",
            "running",
            "Recording observability artifacts.",
            observability_mode=observability_mode,
        )
        state = ResearchState.model_validate_json(state_json)
        _merge_partial_state(state)
        record = record_research_run(
            state,
            mode=_validate_observability_mode(observability_mode),
        )
        partial = _partial_state()
        if partial is not None:
            partial.observability = record
            _publish_partial_state()
        _emit_tool_progress(
            "record_observability_tool",
            "succeeded",
            "Observability artifacts recorded.",
            trace_id=record.trace_id,
            backend=record.backend,
        )
        return record.model_dump_json()

    @function_tool
    def run_deterministic_research_loop_tool(
        thesis_text: str = DEFAULT_THESIS,
        max_iterations: int = 1,
        corpus_path: str = "",
        extraction_mode: str = "local",
        execution_backend: str = "",
        observability_mode: str = "local",
        source_mode: str = "prepared",
        allow_live_web_search: bool = False,
        web_search_model: str = "",
        max_web_sources: int = 8,
    ) -> str:
        """Run the deterministic Pragmatic loop and return ResearchState JSON."""

        _emit_tool_progress(
            "run_deterministic_research_loop_tool",
            "running",
            "Running the canonical research loop.",
            source_mode=source_mode,
            execution_backend=execution_backend,
        )
        def observe_progress(state: ResearchState, stage: str) -> None:
            _merge_partial_state(state)
            _emit_state_graph_snapshot(state, stage=f"loop.{stage}")

        with observe_research_loop(observe_progress):
            state = run_research_loop(
                thesis_text,
                max_iterations=max_iterations,
                corpus_path=corpus_path or None,
                execution_backend=_validate_execution_backend(execution_backend) if execution_backend else None,
                extraction_mode=_validate_extraction_mode(extraction_mode),
                source_mode=_validate_source_mode(source_mode),
                allow_live_web_search=allow_live_web_search,
                web_search_model=web_search_model or None,
                max_web_sources=max_web_sources,
                observability_mode=_validate_observability_mode(observability_mode),
            )
        _merge_partial_state(state)
        _emit_state_graph_snapshot(state, stage="tool.run_deterministic_research_loop_tool")
        _emit_tool_progress(
            "run_deterministic_research_loop_tool",
            "succeeded",
            "Canonical research loop completed.",
            assumption_count=len(state.assumptions),
            evidence_count=len(state.evidence_items),
            invalid_leap_count=len(state.invalid_leaps),
        )
        return state.model_dump_json()

    @function_tool
    def run_replay_demo_tool(
        thesis_text: str = DEFAULT_THESIS,
        max_iterations: int = 1,
        corpus_path: str = "",
        extraction_mode: str = "local",
        execution_backend: str = "",
        observability_mode: str = "local",
    ) -> str:
        """Run the failure-to-eval replay demo and return ReplayResult JSON."""

        _emit_tool_progress("run_replay_demo_tool", "running", "Running failure-to-eval replay.")
        replay = run_replay_demo(
            thesis_text,
            max_iterations=max_iterations,
            corpus_path=corpus_path or None,
            execution_backend=_validate_execution_backend(execution_backend) if execution_backend else None,
            extraction_mode=_validate_extraction_mode(extraction_mode),
            observability_mode=_validate_observability_mode(observability_mode),
        )
        _merge_partial_state(replay.replay_pass)
        _emit_state_graph_snapshot(replay.replay_pass, stage="tool.run_replay_demo_tool")
        for comparison in replay.comparisons:
            _emit_cockpit_event(
                "node.confidence",
                stage="tool.run_replay_demo_tool",
                status="updated",
                message=f"Replay changed confidence for {comparison.assumption_id}.",
                id=comparison.assumption_id,
                **{"from": comparison.before_confidence, "to": comparison.after_confidence},
                reason=comparison.change_summary,
            )
        _emit_tool_progress(
            "run_replay_demo_tool",
            "succeeded",
            "Replay completed.",
            comparison_count=len(replay.comparisons),
        )
        return replay.model_dump_json()

else:
    decompose_thesis_tool = None
    plan_questions_tool = None
    retrieve_sources_tool = None
    score_retrieval_tool = None
    extract_evidence_tool = None
    execute_source_research_tasks_tool = None
    cross_check_evidence_tool = None
    detect_invalid_leaps_tool = None
    update_beliefs_tool = None
    propose_decisive_tests_tool = None
    run_decisive_test_verifiers_tool = None
    generate_evals_from_failures_tool = None
    build_eval_workshop_tool = None
    record_observability_tool = None
    run_deterministic_research_loop_tool = None
    run_replay_demo_tool = None


def _to_json(value: Any) -> str:
    return json.dumps(value, indent=2)


def _invoke_function_tool(tool: Any, arguments: dict[str, Any]) -> str:
    context = SimpleNamespace(tool_name=tool.name, run_config=None)
    result = tool.on_invoke_tool(context, json.dumps(arguments))
    if inspect.isawaitable(result):
        result = _run_awaitable(result)
    if not isinstance(result, str):
        return json.dumps(result)
    return result


def _invoke_specialist_tool(
    tools: dict[str, Any],
    step_records: list[AgentRunStep],
    *,
    agent_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    summary: str,
) -> str:
    result = _invoke_function_tool(tools[tool_name], arguments)
    step_records.append(
        AgentRunStep(
            id=f"agent_step_{len(step_records) + 1:03d}",
            agent_name=agent_name,
            tool_name=tool_name,
            status="succeeded",
            summary=summary,
        )
    )
    return result


def _run_awaitable(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Use the async SDK path when an event loop is already running.")


def _coerce_research_state(value: Any) -> ResearchState:
    if hasattr(value, "final_output_as"):
        try:
            return value.final_output_as(ResearchState)
        except Exception:
            value = getattr(value, "final_output", value)
    elif hasattr(value, "final_output"):
        value = value.final_output

    if isinstance(value, ResearchState):
        return value
    if isinstance(value, str):
        try:
            return ResearchState.model_validate_json(value)
        except Exception as exc:
            raise AgentOutputValidationError(
                "Agent output was a string but not valid ResearchState JSON."
            ) from exc
    if isinstance(value, dict):
        try:
            return ResearchState.model_validate(value)
        except Exception as exc:
            raise AgentOutputValidationError(
                "Agent output was a mapping but not a valid ResearchState."
            ) from exc
    raise AgentOutputValidationError(
        f"Agent output could not be converted to ResearchState: {type(value).__name__}"
    )


def _emit_stream_event(event: Any, delta_buffer: dict[str, Any]) -> None:
    event_type = getattr(event, "type", "")
    if event_type == "raw_response_event":
        data = getattr(event, "data", None)
        data_type = getattr(data, "type", "")
        delta = getattr(data, "delta", None)
        if _should_forward_reasoning_delta(data_type, delta):
            _buffer_reasoning_delta(delta_buffer, delta)
        return

    if event_type == "agent_updated_stream_event":
        agent = getattr(event, "new_agent", None)
        name = getattr(agent, "name", "agent")
        _emit_cockpit_event(
            "agent.update",
            stage="live_sdk",
            status="running",
            message=f"Agent switched to {name}.",
            name=name,
        )
        return

    if event_type != "run_item_stream_event":
        return

    name = getattr(event, "name", "")
    item = getattr(event, "item", None)
    tool_name = _stream_tool_name(item)
    if name == "tool_called":
        _emit_cockpit_event(
            "tool.call",
            stage=f"tool.{tool_name}" if tool_name else "tool",
            status="running",
            message=f"Calling {tool_name or 'tool'}.",
            name=tool_name,
            args_preview=_stream_tool_args_preview(item),
        )
    elif name == "tool_output":
        _emit_cockpit_event(
            "tool.output",
            stage=f"tool.{tool_name}" if tool_name else "tool",
            status="succeeded",
            message=f"{tool_name or 'Tool'} returned output.",
            name=tool_name,
            summary=_stream_tool_output_summary(item),
        )
    elif name == "reasoning_item_created":
        _emit_cockpit_event(
            "reasoning.delta",
            stage="live_sdk",
            status="running",
            message="Reasoning item created.",
            text="Thinking through the next research step. ",
        )


def _should_forward_reasoning_delta(data_type: str, delta: Any) -> bool:
    if not isinstance(delta, str) or not delta:
        return False
    if data_type == "response.reasoning_summary_text.delta":
        return True
    if data_type != "response.output_text.delta":
        return False
    return _looks_like_prose_delta(delta)


def _looks_like_prose_delta(delta: str) -> bool:
    stripped = delta.strip()
    if not stripped:
        return False
    if stripped[0] in "{}[]\":,":
        return False
    punctuation_count = sum(delta.count(char) for char in "{}[]\":")
    return punctuation_count <= max(2, len(delta) // 6)


def _buffer_reasoning_delta(delta_buffer: dict[str, Any], delta: str) -> None:
    chunks = delta_buffer.setdefault("chunks", [])
    chunks.append(delta)
    text = "".join(chunks)
    now = time.monotonic()
    last_flush = float(delta_buffer.get("last_flush", now))
    if (
        now - last_flush >= 0.05
        or len(text) >= 160
        or text.endswith((".", "?", "!", "\n", "; "))
    ):
        _flush_reasoning_delta(delta_buffer)


def _flush_reasoning_delta(delta_buffer: dict[str, Any]) -> None:
    chunks = delta_buffer.get("chunks") or []
    if not chunks:
        return
    text = "".join(chunks)
    delta_buffer["chunks"] = []
    delta_buffer["last_flush"] = time.monotonic()
    if not text.strip():
        return
    _emit_cockpit_event(
        "reasoning.delta",
        stage="live_sdk",
        status="running",
        message="Streaming model reasoning.",
        text=text,
    )


def _stream_tool_name(item: Any) -> str:
    raw_item = getattr(item, "raw_item", None)
    for candidate in (
        getattr(item, "tool_name", None),
        getattr(item, "name", None),
        getattr(raw_item, "name", None),
        getattr(raw_item, "function", None) and getattr(raw_item.function, "name", None),
    ):
        if candidate:
            return str(candidate)
    return ""


def _stream_tool_args_preview(item: Any) -> str:
    raw_item = getattr(item, "raw_item", None)
    value = (
        getattr(item, "arguments", None)
        or getattr(raw_item, "arguments", None)
        or getattr(raw_item, "input", None)
        or ""
    )
    return str(value)[:180]


def _stream_tool_output_summary(item: Any) -> str:
    value = getattr(item, "output", None)
    if value is None:
        raw_item = getattr(item, "raw_item", None)
        value = getattr(raw_item, "output", None)
    text = str(value or "")
    return text[:220]


def _finalize_live_research_state(
    state: ResearchState,
    *,
    execution_backend: ExecutionBackend,
    observability_mode: ObservabilityMode,
    source_mode: SourceAcquisitionMode = "prepared",
    allow_live_web_search: bool = False,
    web_search_model: str | None = None,
    max_web_sources: int = 8,
) -> ResearchState:
    """Close gaps left by a live model while preserving its schema-valid state."""

    if not state.assumptions:
        state.assumptions = decompose_thesis(state.thesis.text)
        _append_agent_trace(
            state,
            "Finalized live state with deterministic assumption decomposition.",
            metadata={"assumptions": str(len(state.assumptions))},
        )

    if not state.research_questions:
        state.research_questions = generate_initial_questions(state)
        _append_agent_trace(
            state,
            "Finalized live state with deterministic research questions.",
            metadata={"questions": str(len(state.research_questions))},
        )

    if not state.sources:
        if source_mode == "web":
            if not allow_live_web_search:
                raise ValueError("source_mode='web' requires allow_live_web_search=True.")
            sources = build_web_corpus(
                state.thesis.text,
                state.research_questions,
                model=web_search_model,
                max_sources=max_web_sources,
            )
        else:
            sources = retrieve_sources(state.research_questions, load_corpus(None))
        _append_unique(state.sources, sources)
        _append_agent_trace(
            state,
            "Finalized live state with source acquisition.",
            metadata={
                "source_mode": source_mode,
                "sources": str(len(state.sources)),
            },
        )

    if state.research_questions and state.sources and not state.retrieval_scores:
        state.retrieval_scores = score_corpus_for_questions(state.research_questions, state.sources)

    if state.sources and state.assumptions and not state.evidence_items:
        parse_tasks = build_source_parse_tasks(state.sources, state.research_questions)
        parse_execution = execute_research_tasks(
            parse_tasks,
            backend=execution_backend,
            fallback_to_local=True,
        )
        _append_unique(state.research_task_results, parse_execution.results)
        parsed_sources = [
            source
            for result in parse_execution.results
            if result.status == "succeeded"
            for source in result.sources
        ] or state.sources
        extraction_tasks = build_source_extraction_tasks(
            parsed_sources,
            state.assumptions,
            state.research_questions,
        )
        execution = execute_research_tasks(
            extraction_tasks,
            backend=execution_backend,
            fallback_to_local=True,
        )
        _append_unique(state.research_task_results, execution.results)
        _append_unique(
            state.evidence_items,
            [
                item
                for result in execution.results
                for item in result.evidence_items
            ],
        )
        _append_agent_trace(
            state,
            "Finalized live state with typed evidence extraction.",
            metadata={
                "backend": execution.backend,
                "attempted_backend": execution.attempted_backend,
                "evidence_items": str(len(state.evidence_items)),
            },
        )

    if state.sources and state.evidence_items and not state.evidence_conflicts:
        task = build_cross_check_task(state.sources, state.evidence_items, state.assumptions)
        execution = execute_research_tasks(
            [task],
            backend=execution_backend,
            fallback_to_local=True,
        )
        _append_unique(state.research_task_results, execution.results)
        state.evidence_conflicts = [
            conflict
            for result in execution.results
            for conflict in result.evidence_conflicts
        ]
        _append_agent_trace(
            state,
            "Finalized live state with deterministic cross-source evidence checking.",
            metadata={
                "backend": execution.backend,
                "attempted_backend": execution.attempted_backend,
                "conflict_count": str(len(state.evidence_conflicts)),
            },
        )

    if not state.invalid_leaps:
        _append_unique(state.invalid_leaps, detect_invalid_leaps(state))
        _append_agent_trace(
            state,
            "Finalized live state with deterministic invalid-leap detection.",
            metadata={"invalid_leaps": str(len(state.invalid_leaps))},
        )

    if not state.decisive_tests:
        state.decisive_tests = propose_decisive_tests(state)
        _append_agent_trace(
            state,
            "Finalized live state with decisive-test proposals.",
            metadata={"decisive_tests": str(len(state.decisive_tests))},
        )

    if state.decisive_tests and not state.verifier_results:
        tasks = build_verifier_tasks(
            state.decisive_tests,
            state.sources,
            state.evidence_items,
            state.evidence_conflicts,
        )
        if tasks:
            execution = execute_research_tasks(
                tasks,
                backend=execution_backend,
                fallback_to_local=True,
            )
            _append_unique(state.research_task_results, execution.results)
            state.verifier_results = [
                verifier_result
                for result in execution.results
                for verifier_result in result.verifier_results
            ]
            _append_agent_trace(
                state,
                "Finalized live state with decisive-test verifier tasks.",
                metadata={
                    "backend": execution.backend,
                    "attempted_backend": execution.attempted_backend,
                    "verifier_results": str(len(state.verifier_results)),
                },
            )

    if state.evidence_items:
        updates = update_beliefs(state)
        state.belief_updates = updates
        apply_belief_updates(state, updates)

    if state.invalid_leaps and not state.generated_evals:
        state.generated_evals = generate_evals_from_failures(state.invalid_leaps)
        _append_agent_trace(
            state,
            "Finalized live state by generating evals from detected failures.",
            metadata={"generated_evals": str(len(state.generated_evals))},
        )

    state.eval_workshop = build_eval_workshop(state)
    if observability_mode != "off":
        state.observability = record_research_run(
            state,
            mode=observability_mode,
        )
    return state


def _append_agent_trace(
    state: ResearchState,
    message: str,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    state.trace_events.append(
        TraceEvent(
            id=f"trace_{len(state.trace_events) + 1:03d}",
            stage="agent",
            message=message,
            metadata=metadata or {},
        )
    )


def _append_unique(existing: list, additions: list) -> None:
    existing_ids = {_unique_id(item) for item in existing}
    for item in additions:
        item_id = _unique_id(item)
        if item_id not in existing_ids:
            existing.append(item)
            existing_ids.add(item_id)


def _unique_id(item: Any) -> str:
    if hasattr(item, "id"):
        return item.id
    if hasattr(item, "task_id"):
        return item.task_id
    if hasattr(item, "assumption_id"):
        return item.assumption_id
    raise AttributeError(f"Cannot append unique item without id: {item!r}")


def _validate_extraction_mode(value: str) -> ExtractionMode:
    if value == "modal":
        return "modal"
    return "local"


def _validate_execution_backend(value: str) -> ExecutionBackend:
    if value == "modal":
        return "modal"
    return "local"


def _validate_source_mode(value: str) -> SourceAcquisitionMode:
    if value == "web":
        return "web"
    return "prepared"


def _validate_observability_mode(value: str) -> ObservabilityMode:
    if value == "raindrop":
        return "raindrop"
    if value == "off":
        return "off"
    return "local"
