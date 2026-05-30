import json
import os
from types import SimpleNamespace

from pragmatic import DEFAULT_THESIS
from pragmatic.agents import ResearchManager
from pragmatic.cli import main
from pragmatic.live_harness import load_latest_live_run, run_live_harness_sync


class FakeStreamedRun:
    def __init__(self, final_output):
        self.final_output = final_output

    async def stream_events(self):
        if False:
            yield None


def test_live_harness_dry_run_does_not_call_sdk(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry run should not call the live SDK")

    monkeypatch.setattr(ResearchManager, "run_live", fail_if_called)

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="dry_run",
        output_dir=tmp_path,
    )

    assert result.status == "ready"
    assert result.mode == "dry_run"
    assert result.state is None
    assert result.guardrails.prepared_corpus_only is True
    assert result.guardrails.allow_live_web_search is False
    assert result.output_path is not None
    assert "No OpenAI API call" in result.message


def test_live_harness_blocks_live_without_explicit_allow(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=False,
        output_dir=tmp_path,
    )

    assert result.status == "blocked"
    assert result.error_type == "LiveAgentsSDKNotEnabled"
    assert result.credentials_available is True
    assert result.state is None


def test_live_harness_success_writes_schema_valid_artifact(monkeypatch, tmp_path):
    from pragmatic import agents as agents_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    def fake_run_streamed(agent, input, **kwargs):
        assert agent.name == "ResearchManager"
        assert "Do not perform live web search" in input
        assert kwargs["max_turns"] == 3
        return FakeStreamedRun(expected.model_dump())

    monkeypatch.setattr(agents_module.Runner, "run_streamed", fake_run_streamed)

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=True,
        max_turns=3,
        timeout_seconds=10,
        observability_mode="off",
        output_dir=tmp_path,
    )

    assert result.status == "succeeded"
    assert result.state is not None
    assert result.state.agent_run is not None
    assert result.state.agent_run.mode == "live_sdk"
    assert result.proof is not None
    assert result.proof.final_output_validated is True
    assert result.proof.demo_ready is True
    assert result.output_path is not None
    payload = json.loads((tmp_path / f"{result.id}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["state"]["agent_run"]["mode"] == "live_sdk"
    assert payload["proof"]["demo_ready"] is True


def test_live_harness_passes_live_web_search_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    async def fake_run_live(self, *args, **kwargs):
        del self, args
        captured.update(kwargs)
        return ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    monkeypatch.setattr(ResearchManager, "run_live", fake_run_live)

    result = run_live_harness_sync(
        "Can spider silk make a bullet proof vest?",
        mode="live",
        allow_live_sdk=True,
        source_mode="web",
        allow_live_web_search=True,
        web_search_model="test-search-model",
        max_web_sources=5,
        observability_mode="off",
        output_dir=tmp_path,
    )

    assert result.status == "succeeded"
    assert result.guardrails.source_mode == "web"
    assert result.guardrails.prepared_corpus_only is False
    assert result.guardrails.allow_live_web_search is True
    assert captured["source_mode"] == "web"
    assert captured["allow_live_web_search"] is True
    assert captured["web_search_model"] == "test-search-model"
    assert captured["max_web_sources"] == 5


def test_live_harness_can_require_demo_proof(monkeypatch, tmp_path):
    from pragmatic import agents as agents_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="local")

    def fake_run_streamed(agent, input, **kwargs):
        del agent, input, kwargs
        return FakeStreamedRun(expected.model_dump())

    monkeypatch.setattr(agents_module.Runner, "run_streamed", fake_run_streamed)

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=True,
        execution_backend="modal",
        observability_mode="local",
        require_demo_proof=True,
        output_dir=tmp_path,
    )

    assert result.status == "failed"
    assert result.error_type == "DemoProofIncomplete"
    assert result.proof is not None
    assert result.proof.demo_ready is False
    assert result.proof.remote_modal_task_count == 0


def test_live_harness_times_out_live_run(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def slow_run(self, *args, **kwargs):
        del self, args, kwargs
        await asyncio.sleep(0.05)
        return ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    monkeypatch.setattr(ResearchManager, "run_live", slow_run)

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=True,
        timeout_seconds=0.001,
        output_dir=tmp_path,
    )

    assert result.status == "timed_out"
    assert result.error_type == "TimeoutError"
    assert result.state is None


def test_live_harness_recovers_partial_state_on_timeout(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    partial = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")
    partial.generated_evals = []
    partial.eval_workshop = None
    partial.observability = None

    async def slow_run(self, *args, **kwargs):
        del self, args
        kwargs["partial_state_callback"](partial)
        await asyncio.sleep(0.05)
        return partial

    monkeypatch.setattr(ResearchManager, "run_live", slow_run)

    result = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=True,
        timeout_seconds=0.001,
        observability_mode="off",
        output_dir=tmp_path,
    )

    assert result.status == "timed_out"
    assert result.state is not None
    assert result.state.evidence_items
    assert result.state.agent_run is not None
    assert result.state.agent_run.final_output_validated is False
    assert result.state.generated_evals
    assert result.proof is not None
    assert result.proof.generated_eval_count == len(result.state.generated_evals)


def test_load_latest_live_run_can_require_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_run_live(self, *args, **kwargs):
        del self, args, kwargs
        return ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    monkeypatch.setattr(ResearchManager, "run_live", fake_run_live)

    ready = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="dry_run",
        output_dir=tmp_path,
    )
    succeeded = run_live_harness_sync(
        DEFAULT_THESIS,
        mode="live",
        allow_live_sdk=True,
        observability_mode="off",
        output_dir=tmp_path,
    )
    ready_path = tmp_path / f"{ready.id}.json"
    succeeded_path = tmp_path / f"{succeeded.id}.json"
    os.utime(succeeded_path, (1, 1))
    os.utime(ready_path, (2, 2))

    latest = load_latest_live_run(tmp_path, require_state=True)

    assert latest is not None
    assert latest.id == succeeded.id
    assert latest.state is not None


def test_cli_live_harness_dry_run_writes_output(tmp_path):
    output_path = tmp_path / "harness.json"

    exit_code = main(
        [
            "live-run-harness",
            "--no-artifact",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["mode"] == "dry_run"
