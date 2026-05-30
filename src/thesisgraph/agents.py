from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from thesisgraph.belief_update import apply_belief_updates, update_beliefs
from thesisgraph.corpus import load_corpus
from thesisgraph.decisive_tests import propose_decisive_tests
from thesisgraph.eval_writer import generate_evals_from_failures
from thesisgraph.eval_workshop import build_eval_workshop
from thesisgraph.execution import (
    build_cross_check_task,
    build_source_extraction_tasks,
    build_source_parse_tasks,
    execute_research_tasks,
)
from thesisgraph.extractors import ExtractionMode, extract_evidence
from thesisgraph.invalid_leaps import detect_invalid_leaps
from thesisgraph.raindrop_client import ObservabilityMode, record_research_run
from thesisgraph.replay import run_replay_demo
from thesisgraph.research_loop import (
    DEFAULT_THESIS,
    decompose_thesis,
    generate_initial_questions,
    retrieve_sources,
    run_research_loop,
    score_retrieval,
)
from thesisgraph.schemas import (
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
    ResearchState,
    Source,
    Thesis,
    TraceEvent,
)
from thesisgraph.verifiers import build_verifier_tasks

try:
    from agents import Agent, Runner, function_tool
    from agents.agent_output import AgentOutputSchema
except ImportError:  # pragma: no cover - exercised only when the SDK is absent.
    Agent = None  # type: ignore[assignment]
    AgentOutputSchema = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]


class AgentsSDKUnavailable(RuntimeError):
    """Raised when live OpenAI Agents SDK objects are requested without the SDK."""


class LiveAgentsSDKNotEnabled(RuntimeError):
    """Raised when a live SDK run is requested without explicit opt-in."""


class AgentsSDKCredentialsError(RuntimeError):
    """Raised when live SDK execution is requested without API credentials."""


class AgentOutputValidationError(RuntimeError):
    """Raised when an SDK run does not produce a valid ResearchState."""


RESEARCH_MANAGER_INSTRUCTIONS = """
You are ResearchManager for ThesisGraph.

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
    max_turns: int = 10

    def run_deterministic(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        modal_fallback: bool = True,
        observability_mode: ObservabilityMode = "local",
    ) -> ResearchState:
        return run_research_loop(
            thesis_text,
            max_iterations=max_iterations,
            corpus_path=corpus_path,
            execution_backend=execution_backend,
            extraction_mode=extraction_mode,
            modal_fallback=modal_fallback,
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
        observability_mode: ObservabilityMode = "local",
    ) -> ResearchState:
        _require_agents_sdk()
        agent = self.build_agent()
        tools = {tool.name: tool for tool in agent.tools}
        state = ResearchState(thesis=Thesis(text=thesis_text, domain="materials discovery"))
        _append_agent_trace(
            state,
            "OpenAI Agents SDK ResearchManager initialized specialist orchestration.",
            metadata={"mode": "scripted_sdk", "agent": agent.name},
        )

        step_records: list[AgentRunStep] = []
        corpus_path_value = str(corpus_path) if corpus_path else ""
        execution_backend_value = execution_backend or ("modal" if extraction_mode == "modal" else "")
        resolved_backend = _validate_execution_backend(execution_backend_value)

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
                },
                summary="Retrieved prepared-corpus sources for open questions.",
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
                },
                summary="Scored prepared-corpus retrieval matches.",
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
        return Agent(
            name="ResearchManager",
            instructions=RESEARCH_MANAGER_INSTRUCTIONS,
            model=self.model,
            tools=build_research_tools(),
            output_type=AgentOutputSchema(ResearchState, strict_json_schema=False),
        )

    def run_live_sync(
        self,
        thesis_text: str = DEFAULT_THESIS,
        *,
        max_iterations: int = 1,
        corpus_path: str | Path | None = None,
        execution_backend: ExecutionBackend | None = None,
        extraction_mode: ExtractionMode = "local",
        observability_mode: ObservabilityMode = "off",
        allow_live_sdk: bool = False,
    ) -> ResearchState:
        return _run_awaitable(
            self.run_live(
                thesis_text,
                max_iterations=max_iterations,
                corpus_path=corpus_path,
                execution_backend=execution_backend,
                extraction_mode=extraction_mode,
                observability_mode=observability_mode,
                allow_live_sdk=allow_live_sdk,
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
        observability_mode: ObservabilityMode = "off",
        allow_live_sdk: bool = False,
    ) -> ResearchState:
        _require_live_sdk_enabled(allow_live_sdk)
        _require_agents_sdk()
        agent = self.build_agent()
        execution_backend_value = execution_backend or ""
        corpus_path_value = str(corpus_path) if corpus_path else ""
        prompt = (
            "Run the ThesisGraph research loop for this thesis using specialist tools. "
            "Prefer this order: decompose_thesis_tool, plan_questions_tool, "
            "retrieve_sources_tool, score_retrieval_tool, execute_source_research_tasks_tool, "
            "cross_check_evidence_tool, detect_invalid_leaps_tool, update_beliefs_tool, "
            "propose_decisive_tests_tool, run_decisive_test_verifiers_tool, "
            "generate_evals_from_failures_tool, build_eval_workshop_tool, "
            "record_observability_tool. "
            "If any invalid_leaps are present, generated_evals must be non-empty. "
            "If generated_evals are missing, call generate_evals_from_failures_tool before returning. "
            "Do not perform live web search or add sources outside the prepared corpus. "
            "Return the final schema-valid ResearchState object.\n\n"
            f"max_iterations: {max_iterations}\n"
            f"corpus_path: {corpus_path_value}\n"
            f"execution_backend: {execution_backend_value}\n"
            f"extraction_mode: {extraction_mode}\n"
            f"observability_mode: {observability_mode}\n\n"
            f"Thesis: {thesis_text}"
        )
        result = await Runner.run(agent, prompt, max_turns=self.max_turns)
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
        """Decompose a thesis into ThesisGraph assumption objects as JSON."""

        assumptions = decompose_thesis(thesis_text)
        return _to_json([assumption.model_dump() for assumption in assumptions])

    @function_tool
    def plan_questions_tool(state_json: str) -> str:
        """Generate initial research questions from a ResearchState JSON string."""

        state = ResearchState.model_validate_json(state_json)
        questions = generate_initial_questions(state)
        return _to_json([question.model_dump() for question in questions])

    @function_tool
    def retrieve_sources_tool(research_questions_json: str, corpus_path: str = "") -> str:
        """Retrieve local corpus sources for research questions and return Source JSON."""

        questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(research_questions_json)
        ]
        corpus = load_corpus(corpus_path or None)
        sources = retrieve_sources(questions, corpus)
        return _to_json([source.model_dump() for source in sources])

    @function_tool
    def score_retrieval_tool(research_questions_json: str, corpus_path: str = "") -> str:
        """Score prepared-corpus retrieval matches and return RetrievalScore JSON."""

        questions = [
            ResearchQuestion.model_validate(question)
            for question in json.loads(research_questions_json)
        ]
        corpus = load_corpus(corpus_path or None)
        scores = score_retrieval(questions, corpus)
        return _to_json([score.model_dump() for score in scores])

    @function_tool
    def extract_evidence_tool(sources_json: str, assumptions_json: str) -> str:
        """Extract typed evidence items from source and assumption JSON."""

        sources = [Source.model_validate(source) for source in json.loads(sources_json)]
        assumptions = [
            Assumption.model_validate(assumption)
            for assumption in json.loads(assumptions_json)
        ]
        evidence = extract_evidence(sources, assumptions)
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
        return result.model_dump_json()

    @function_tool
    def detect_invalid_leaps_tool(state_json: str) -> str:
        """Detect invalid inference leaps from a ResearchState JSON string."""

        state = ResearchState.model_validate_json(state_json)
        leaps = detect_invalid_leaps(state)
        return _to_json([leap.model_dump() for leap in leaps])

    @function_tool
    def update_beliefs_tool(state_json: str) -> str:
        """Compute belief updates from a ResearchState JSON string."""

        state = ResearchState.model_validate_json(state_json)
        updates = update_beliefs(state)
        return _to_json([update.model_dump() for update in updates])

    @function_tool
    def propose_decisive_tests_tool(state_json: str) -> str:
        """Propose decisive tests from a ResearchState JSON string."""

        state = ResearchState.model_validate_json(state_json)
        tests = propose_decisive_tests(state)
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
        return result.model_dump_json()

    @function_tool
    def generate_evals_from_failures_tool(invalid_leaps_json: str) -> str:
        """Generate eval artifacts from invalid leap JSON."""

        invalid_leaps = [
            InvalidLeap.model_validate(invalid_leap)
            for invalid_leap in json.loads(invalid_leaps_json)
        ]
        evals = generate_evals_from_failures(invalid_leaps)
        return _to_json([generated_eval.model_dump() for generated_eval in evals])

    @function_tool
    def build_eval_workshop_tool(state_json: str) -> str:
        """Build failure-to-eval workshop links and task-span artifacts."""

        state = ResearchState.model_validate_json(state_json)
        workshop = build_eval_workshop(state)
        return workshop.model_dump_json()

    @function_tool
    def record_observability_tool(
        state_json: str,
        observability_mode: str = "local",
    ) -> str:
        """Record trace/workshop observability artifacts and return ObservabilityRecord JSON."""

        state = ResearchState.model_validate_json(state_json)
        record = record_research_run(
            state,
            mode=_validate_observability_mode(observability_mode),
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
    ) -> str:
        """Run the deterministic ThesisGraph loop and return ResearchState JSON."""

        state = run_research_loop(
            thesis_text,
            max_iterations=max_iterations,
            corpus_path=corpus_path or None,
            execution_backend=_validate_execution_backend(execution_backend) if execution_backend else None,
            extraction_mode=_validate_extraction_mode(extraction_mode),
            observability_mode=_validate_observability_mode(observability_mode),
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

        replay = run_replay_demo(
            thesis_text,
            max_iterations=max_iterations,
            corpus_path=corpus_path or None,
            execution_backend=_validate_execution_backend(execution_backend) if execution_backend else None,
            extraction_mode=_validate_extraction_mode(extraction_mode),
            observability_mode=_validate_observability_mode(observability_mode),
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


def _finalize_live_research_state(
    state: ResearchState,
    *,
    execution_backend: ExecutionBackend,
    observability_mode: ObservabilityMode,
) -> ResearchState:
    """Close gaps left by a live model while preserving its schema-valid state."""

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
    raise AttributeError(f"Cannot append unique item without id: {item!r}")


def _validate_extraction_mode(value: str) -> ExtractionMode:
    if value == "modal":
        return "modal"
    return "local"


def _validate_execution_backend(value: str) -> ExecutionBackend:
    if value == "modal":
        return "modal"
    return "local"


def _validate_observability_mode(value: str) -> ObservabilityMode:
    if value == "raindrop":
        return "raindrop"
    if value == "off":
        return "off"
    return "local"
