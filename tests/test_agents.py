import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from thesisgraph import DEFAULT_THESIS
from thesisgraph.agents import (
    AgentsSDKCredentialsError,
    LiveAgentsSDKNotEnabled,
    ResearchManager,
    _coerce_research_state,
    build_research_tools,
)
from thesisgraph.cli import main


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
        "execute_source_research_tasks_tool",
        "cross_check_evidence_tool",
        "detect_invalid_leaps_tool",
        "run_decisive_test_verifiers_tool",
        "record_observability_tool",
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
    tool_steps = [step.tool_name for step in state.agent_run.steps]
    specialist_agents = {step.agent_name for step in state.agent_run.steps}
    assert "run_deterministic_research_loop_tool" not in tool_steps
    assert "decompose_thesis_tool" in tool_steps
    assert "execute_source_research_tasks_tool" in tool_steps
    assert "run_decisive_test_verifiers_tool" in tool_steps
    assert "record_observability_tool" in tool_steps
    assert {"AssumptionDecomposer", "EvidenceExtractor", "BeliefUpdater", "EvalWriter"}.issubset(
        specialist_agents
    )
    assert any(event.stage == "agent" for event in state.trace_events)


def test_agent_output_coercion_accepts_run_result_like_object():
    state = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    class FakeRunResult:
        final_output = state.model_dump()

    parsed = _coerce_research_state(FakeRunResult())

    assert parsed.thesis.text == DEFAULT_THESIS


def test_live_sdk_requires_explicit_opt_in():
    pytest.importorskip("agents")
    manager = ResearchManager(model=None)

    with pytest.raises(LiveAgentsSDKNotEnabled):
        manager.run_live_sync(DEFAULT_THESIS, allow_live_sdk=False)


def test_live_sdk_requires_api_key_when_opted_in(monkeypatch):
    pytest.importorskip("agents")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = ResearchManager(model=None)

    with pytest.raises(AgentsSDKCredentialsError):
        manager.run_live_sync(DEFAULT_THESIS, allow_live_sdk=True)


def test_live_sdk_uses_runner_and_validates_state(monkeypatch):
    pytest.importorskip("agents")
    from thesisgraph import agents as agents_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")
    captured = {}

    async def fake_run(agent, prompt, **kwargs):
        captured["agent_name"] = agent.name
        captured["prompt"] = prompt
        captured["max_turns"] = kwargs["max_turns"]
        return SimpleNamespace(final_output=expected.model_dump())

    monkeypatch.setattr(agents_module.Runner, "run", fake_run)

    state = ResearchManager(model="test-model", max_turns=4).run_live_sync(
        DEFAULT_THESIS,
        observability_mode="off",
        allow_live_sdk=True,
    )

    assert state.thesis.text == DEFAULT_THESIS
    assert state.agent_run is not None
    assert state.agent_run.mode == "live_sdk"
    assert state.agent_run.model == "test-model"
    assert state.agent_run.final_output_validated is True
    assert state.agent_run.steps[0].tool_name == "Runner.run"
    assert captured["agent_name"] == "ResearchManager"
    assert captured["max_turns"] == 4
    assert "Do not perform live web search" in captured["prompt"]
    assert "execute_source_research_tasks_tool" in captured["prompt"]


def test_live_sdk_finalizes_missing_generated_evals(monkeypatch):
    pytest.importorskip("agents")
    from thesisgraph import agents as agents_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    partial = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")
    partial.generated_evals = []
    partial.eval_workshop = None
    partial.observability = None

    async def fake_run(agent, prompt, **kwargs):
        del agent, prompt, kwargs
        return SimpleNamespace(final_output=partial.model_dump())

    monkeypatch.setattr(agents_module.Runner, "run", fake_run)

    state = ResearchManager().run_live_sync(
        DEFAULT_THESIS,
        observability_mode="off",
        allow_live_sdk=True,
    )

    assert state.invalid_leaps
    assert state.generated_evals
    assert state.eval_workshop is not None
    assert any("generating evals" in event.message for event in state.trace_events)


def test_cli_live_sdk_requires_explicit_allow_flag(capsys):
    pytest.importorskip("agents")

    exit_code = main(["run-live-sdk", "--thesis", DEFAULT_THESIS])

    assert exit_code == 2
    assert "explicit opt-in" in capsys.readouterr().err


def test_cli_live_sdk_writes_schema_valid_output(monkeypatch, tmp_path):
    pytest.importorskip("agents")
    from thesisgraph import agents as agents_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    async def fake_run(agent, prompt, **kwargs):
        del agent, prompt, kwargs
        return SimpleNamespace(final_output=expected.model_dump())

    monkeypatch.setattr(agents_module.Runner, "run", fake_run)
    output_path = tmp_path / "live_state.json"

    exit_code = main(
        [
            "run-live-sdk",
            "--allow-live-sdk",
            "--observability",
            "off",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["agent_run"]["mode"] == "live_sdk"
    assert payload["agent_run"]["final_output_validated"] is True
