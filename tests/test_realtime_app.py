from realtime_app import MAX_STORED_EVENTS_PER_JOB, RUNS, _append_event, _new_job


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
