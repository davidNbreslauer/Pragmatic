import json

from thesisgraph import DEFAULT_THESIS
from thesisgraph.raindrop_client import (
    build_trace_payload,
    build_workshop_payload,
    record_research_run,
)
from thesisgraph.research_loop import run_research_loop


def test_local_observability_writes_trace_artifact(tmp_path):
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    record = record_research_run(state, mode="local", trace_dir=tmp_path)

    assert record.backend == "local"
    assert record.status == "recorded"
    assert record.trace_path is not None
    assert record.workshop_path is not None
    payload = json.loads((tmp_path / f"{record.trace_id}.json").read_text())
    workshop_payload = json.loads(
        (tmp_path / "workshops" / f"{record.trace_id}.json").read_text()
    )
    assert payload["trace_id"] == record.trace_id
    assert payload["summary"]["generated_evals"] == len(state.generated_evals)
    assert payload["summary"]["task_spans"] == len(state.eval_workshop.task_spans)
    assert payload["generated_evals"]
    assert payload["task_spans"]
    assert payload["failure_eval_links"]
    assert payload["eval_workshop"]["summary"]
    assert workshop_payload["trace_id"] == record.trace_id
    assert workshop_payload["failure_artifacts"]
    assert workshop_payload["eval_artifacts"]
    assert workshop_payload["raindrop_event_plan"]


def test_research_loop_records_observability_on_state(tmp_path):
    state = run_research_loop(
        DEFAULT_THESIS,
        observability_mode="local",
        observability_dir=tmp_path,
    )

    assert state.observability is not None
    assert state.observability.backend == "local"
    assert state.observability.trace_path is not None
    assert state.observability.workshop_path is not None
    assert state.observability.eval_artifact_ids == [
        generated_eval.id for generated_eval in state.generated_evals
    ]
    assert state.observability.failure_artifact_ids
    assert state.observability.workshop_artifact_ids
    assert any(event.stage == "observability" for event in state.trace_events)


def test_observability_can_be_disabled():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    assert state.observability is not None
    assert state.observability.backend == "off"
    assert state.observability.status == "skipped"


def test_raindrop_mode_falls_back_to_local_without_write_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RAINDROP_WRITE_KEY", raising=False)
    state = run_research_loop(
        DEFAULT_THESIS,
        observability_mode="raindrop",
        observability_dir=tmp_path,
        raindrop_fallback=True,
    )

    assert state.observability is not None
    assert state.observability.backend == "local"
    assert "Raindrop unavailable" in (state.observability.message or "")
    assert state.observability.trace_path is not None
    assert state.observability.workshop_path is not None


def test_trace_payload_links_failures_to_evals_and_tasks():
    from thesisgraph import ResearchManager

    state = ResearchManager().run_sdk_orchestrated(DEFAULT_THESIS, observability_mode="off")

    payload = build_trace_payload(state, trace_id="tg_test")
    link_types = {link["link_type"] for link in payload["failure_eval_links"]}

    assert payload["agent_steps"]
    assert "invalid_leap_to_eval" in link_types
    assert "evidence_conflict_to_invalid_leap" in link_types
    assert "verifier_failure_to_eval" in link_types
    assert any(span["task_id"].startswith("task_parse") for span in payload["task_spans"])
    assert any(span["task_id"].startswith("task_extract") for span in payload["task_spans"])
    assert any(span["agent_name"] == "EvidenceExtractor" for span in payload["task_spans"])
    assert any(span["worker_status"] == "completed" for span in payload["task_spans"])


def test_workshop_payload_is_failure_eval_replay_bundle():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    payload = build_workshop_payload(state, trace_id="tg_test")
    eval_artifact = payload["eval_artifacts"][0]
    failure_ids = {artifact["artifact_id"] for artifact in payload["failure_artifacts"]}

    assert payload["schema_version"] == "1"
    assert payload["summary"]["eval_artifacts"] == len(state.generated_evals)
    assert payload["summary"]["failure_artifacts"] == len(failure_ids)
    assert eval_artifact["artifact_type"] == "generated_eval"
    assert eval_artifact["source_failure"]["failure_artifact_id"] in failure_ids
    assert any(
        event["event"] == "thesisgraph.failure_to_eval"
        for event in payload["raindrop_event_plan"]
    )


def test_workshop_payload_connects_sdk_modal_failure_eval_replay():
    from thesisgraph import ResearchManager
    from thesisgraph.replay import run_replay_demo

    state = ResearchManager().run_sdk_orchestrated(DEFAULT_THESIS, observability_mode="off")
    payload = build_workshop_payload(state, trace_id="tg_test")

    assert payload["specialist_step_artifacts"]
    assert payload["task_artifacts"]
    assert payload["connection_rows"]
    assert any(
        row["specialist"] == "EvidenceExtractor" and row["task_id"]
        for row in payload["connection_rows"]
    )
    assert any(
        event["event"] == "thesisgraph.agent_step"
        for event in payload["raindrop_event_plan"]
    )
    assert any(
        event["event"] == "thesisgraph.failure_artifact"
        for event in payload["raindrop_event_plan"]
    )

    replay = run_replay_demo(DEFAULT_THESIS, observability_mode="off")
    replay_payload = build_workshop_payload(replay.replay_pass, trace_id="tg_replay")
    replay_events = {
        event["event"]
        for event in replay_payload["raindrop_event_plan"]
    }
    assert "thesisgraph.replay_outcome" in replay_events
    assert any(row["replay_id"] for row in replay_payload["connection_rows"])
