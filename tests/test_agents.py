import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from thesisgraph import DEFAULT_THESIS
from thesisgraph.agents import ResearchManager, _coerce_research_state, build_research_tools


def test_research_manager_deterministic_path_matches_core_loop():
    manager = ResearchManager()

    state = manager.run_deterministic(DEFAULT_THESIS, observability_mode="off")

    assert state.thesis.text == DEFAULT_THESIS
    assert len(state.assumptions) == 8
    assert any("benchmark" in leap.leap.lower() for leap in state.invalid_leaps)
    assert state.generated_evals


def test_research_manager_builds_openai_agent_with_expected_tools():
    pytest.importorskip("agents")
    manager = ResearchManager(model=None)

    agent = manager.build_agent()

    assert agent.name == "ResearchManager"
    tool_names = {tool.name for tool in agent.tools}
    assert {
        "decompose_thesis_tool",
        "retrieve_sources_tool",
        "detect_invalid_leaps_tool",
        "run_deterministic_research_loop_tool",
    }.issubset(tool_names)
    assert agent.output_type is not None


def test_openai_tool_wrapper_can_run_deterministic_loop_without_api_call():
    pytest.importorskip("agents")
    tools = {tool.name: tool for tool in build_research_tools()}
    loop_tool = tools["run_deterministic_research_loop_tool"]
    context = SimpleNamespace(tool_name=loop_tool.name, run_config=None)

    result_json = loop_tool.on_invoke_tool(
        context,
        json.dumps({
            "thesis_text": DEFAULT_THESIS,
            "max_iterations": 1,
            "corpus_path": "",
            "observability_mode": "off",
        }),
    )

    if inspect.isawaitable(result_json):
        result_json = asyncio.run(result_json)

    assert "generated_evals" in result_json
    assert "Benchmark QA performance" in result_json


def test_sdk_scripted_orchestration_returns_valid_research_state():
    pytest.importorskip("agents")
    manager = ResearchManager(model=None)

    state = manager.run_sdk_orchestrated(DEFAULT_THESIS, observability_mode="off")

    assert state.thesis.text == DEFAULT_THESIS
    assert state.agent_run is not None
    assert state.agent_run.mode == "scripted_sdk"
    assert state.agent_run.final_output_validated is True
    assert any(step.tool_name == "run_deterministic_research_loop_tool" for step in state.agent_run.steps)
    assert any(event.stage == "agent" for event in state.trace_events)


def test_agent_output_coercion_accepts_run_result_like_object():
    state = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    class FakeRunResult:
        final_output = state.model_dump()

    parsed = _coerce_research_state(FakeRunResult())

    assert parsed.thesis.text == DEFAULT_THESIS
