import json

from thesisgraph import DEFAULT_THESIS
from thesisgraph.raindrop_client import build_trace_payload, record_research_run
from thesisgraph.research_loop import run_research_loop


def test_local_observability_writes_trace_artifact(tmp_path):
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    record = record_research_run(state, mode="local", trace_dir=tmp_path)

    assert record.backend == "local"
    assert record.status == "recorded"
    assert record.trace_path is not None
    payload = json.loads((tmp_path / f"{record.trace_id}.json").read_text())
    assert payload["trace_id"] == record.trace_id
    assert payload["summary"]["generated_evals"] == len(state.generated_evals)
    assert payload["summary"]["task_spans"] == len(state.eval_workshop.task_spans)
    assert payload["generated_evals"]
    assert payload["task_spans"]
    assert payload["failure_eval_links"]
    assert payload["eval_workshop"]["summary"]


def test_research_loop_records_observability_on_state(tmp_path):
    state = run_research_loop(
        DEFAULT_THESIS,
        observability_mode="local",
        observability_dir=tmp_path,
    )

    assert state.observability is not None
    assert state.observability.backend == "local"
    assert state.observability.trace_path is not None
    assert state.observability.eval_artifact_ids == [
        generated_eval.id for generated_eval in state.generated_evals
    ]
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


def test_trace_payload_links_failures_to_evals_and_tasks():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    payload = build_trace_payload(state, trace_id="tg_test")
    link_types = {link["link_type"] for link in payload["failure_eval_links"]}

    assert "invalid_leap_to_eval" in link_types
    assert "evidence_conflict_to_invalid_leap" in link_types
    assert "verifier_failure_to_eval" in link_types
    assert payload["task_spans"][0]["task_id"].startswith("task_extract")
