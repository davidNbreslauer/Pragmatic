from pragmatic import DEFAULT_THESIS
from pragmatic.agents import ResearchManager
from pragmatic.ui_flow import (
    build_orchestration_flow_snapshot,
    render_orchestration_flow_html,
)


def test_orchestration_flow_snapshot_counts_research_state():
    state = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")

    snapshot = build_orchestration_flow_snapshot(
        state,
        scenario_name="Test Scenario",
        current_run_source="Test run",
    )

    assert snapshot["scenario"] == "Test Scenario"
    assert snapshot["counts"]["agents"] == 1
    assert snapshot["counts"]["sources"] == len(state.sources)
    assert snapshot["counts"]["evidence"] == len(state.evidence_items)
    assert snapshot["counts"]["invalid_leaps"] == len(state.invalid_leaps)
    assert snapshot["pipeline_nodes"][0]["id"] == "thesis"
    assert snapshot["pipeline_nodes"][-1]["id"] == "belief"
    assert snapshot["data_nodes"]
    assert snapshot["data_edges"]
    assert snapshot["flow_events"]
    event_kinds = {event["kind"] for event in snapshot["flow_events"]}
    assert {"evidence_item", "invalid_leap", "belief_update"}.issubset(event_kinds)


def test_orchestration_flow_html_embeds_animation_payload():
    state = ResearchManager().run_deterministic(DEFAULT_THESIS, observability_mode="off")
    snapshot = build_orchestration_flow_snapshot(
        state,
        scenario_name="Live Full System",
        current_run_source="Loaded latest live run",
    )

    html = render_orchestration_flow_html(snapshot)

    assert "Orchestration Flow" in html
    assert "Real event replay" in html
    assert "flow_events" in html
    assert "flow-data" in html
    assert "requestAnimationFrame" in html
    assert "active-node" in html
    assert "animateMotion" not in html
    assert "Live Full System" in html
    assert "Loaded latest live run" in html
