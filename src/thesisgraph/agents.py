from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from thesisgraph.belief_update import update_beliefs
from thesisgraph.corpus import load_corpus
from thesisgraph.decisive_tests import propose_decisive_tests
from thesisgraph.eval_writer import generate_evals_from_failures
from thesisgraph.extractors import ExtractionMode, extract_evidence
from thesisgraph.invalid_leaps import detect_invalid_leaps
from thesisgraph.raindrop_client import ObservabilityMode
from thesisgraph.replay import run_replay_demo
from thesisgraph.research_loop import (
    DEFAULT_THESIS,
    decompose_thesis,
    generate_initial_questions,
    retrieve_sources,
    run_research_loop,
)
from thesisgraph.schemas import (
    AgentRunRecord,
    AgentRunStep,
    Assumption,
    ExecutionBackend,
    InvalidLeap,
    ResearchQuestion,
    ResearchState,
    Source,
    TraceEvent,
)

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
        tool = tools["run_deterministic_research_loop_tool"]
        state_json = _invoke_function_tool(
            tool,
            {
                "thesis_text": thesis_text,
                "max_iterations": max_iterations,
                "corpus_path": str(corpus_path) if corpus_path else "",
                "execution_backend": execution_backend or "",
                "extraction_mode": extraction_mode,
                "observability_mode": observability_mode,
            },
        )
        state = _coerce_research_state(state_json)
        state.agent_run = AgentRunRecord(
            mode="scripted_sdk",
            status="succeeded",
            agent_name=agent.name,
            model=self.model,
            tool_names=sorted(tools),
            steps=[
                AgentRunStep(
                    id="agent_step_001",
                    tool_name=tool.name,
                    status="succeeded",
                    summary=(
                        "SDK tool invoked canonical research loop and returned a schema-validated ResearchState."
                    ),
                )
            ],
            final_output_validated=True,
            message="Offline SDK-scripted orchestration used the same tool boundary as the live agent.",
        )
        _append_agent_trace(
            state,
            "OpenAI Agents SDK scripted orchestration validated the final ResearchState.",
            metadata={"mode": "scripted_sdk", "tool": tool.name},
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
            "Run the ThesisGraph research loop for this thesis. Use the available "
            "run_deterministic_research_loop_tool exactly once with these arguments. "
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
        extract_evidence_tool,
        detect_invalid_leaps_tool,
        update_beliefs_tool,
        propose_decisive_tests_tool,
        generate_evals_from_failures_tool,
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
    def generate_evals_from_failures_tool(invalid_leaps_json: str) -> str:
        """Generate eval artifacts from invalid leap JSON."""

        invalid_leaps = [
            InvalidLeap.model_validate(invalid_leap)
            for invalid_leap in json.loads(invalid_leaps_json)
        ]
        evals = generate_evals_from_failures(invalid_leaps)
        return _to_json([generated_eval.model_dump() for generated_eval in evals])

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
    extract_evidence_tool = None
    detect_invalid_leaps_tool = None
    update_beliefs_tool = None
    propose_decisive_tests_tool = None
    generate_evals_from_failures_tool = None
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
