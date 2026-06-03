from pragmatic import DEFAULT_THESIS
from pragmatic.research_loop import run_research_loop
from realtime_app import (
    MAX_STORED_EVENTS_PER_JOB,
    RUNS,
    _append_event,
    _build_bottom_line,
    _new_job,
    _normalize_config,
)


def test_append_event_preserves_kind_and_caps_events():
    RUNS.clear()
    job_id = _new_job("test thesis", {})

    for index in range(MAX_STORED_EVENTS_PER_JOB + 3):
        _append_event(
            job_id,
            {
                "stage": "graph",
                "status": "created",
                "message": f"event {index}",
                "kind": "node.add",
                "metadata": {"id": f"A{index}", "node_kind": "assumption"},
            },
        )

    events = RUNS[job_id]["events"]

    assert len(events) == MAX_STORED_EVENTS_PER_JOB
    assert events[-1]["kind"] == "node.add"
    assert events[-1]["metadata"]["id"] == f"A{MAX_STORED_EVENTS_PER_JOB + 2}"
    assert events[0]["message"] == "event 3"


def test_realtime_defaults_are_offline_playground_safe():
    config = _normalize_config({})

    assert config["orchestration"] == "scripted_sdk"
    assert config["execution_backend"] == "local"
    assert config["source_mode"] == "prepared"
    assert config["allow_live_web_search"] is False
    assert config["live_sdk_enabled"] is False
    assert config["live_dry_run"] is True
    assert config["require_demo_proof"] is False
    assert config["timeout_seconds"] == 60


def test_build_bottom_line_populates_verdict_and_next_test():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    bottom_line = _build_bottom_line(state)

    assert bottom_line["verdict"]
    assert bottom_line["confidence_band"] in {"low", "mid", "high"}
    assert bottom_line["one_liner"]
    assert bottom_line["one_liner_source"] == "deterministic"
    assert bottom_line["because"]
    assert bottom_line["biggest_risk"]["text"]
    assert bottom_line["decisive_next_test"] != "No decisive next test was generated."
