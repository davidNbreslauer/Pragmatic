from thesisgraph import DEFAULT_THESIS
from thesisgraph.cross_checking import detect_evidence_conflicts
from thesisgraph.research_loop import run_research_loop


def test_cross_source_checker_detects_validation_conflicts():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    conflicts = detect_evidence_conflicts(state.sources, state.evidence_items)

    assert conflicts
    assert any(conflict.id == "conflict_a6_sparse_prospective_validation" for conflict in conflicts)
    assert any(conflict.severity == "high" for conflict in conflicts)


def test_research_loop_records_cross_source_conflicts_and_invalid_leap():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    assumptions = {assumption.id: assumption for assumption in state.assumptions}

    assert state.evidence_conflicts
    assert any(event.stage == "cross_check" for event in state.trace_events)
    assert any(
        leap.id == "leap_mixed_sources_to_stable_belief"
        for leap in state.invalid_leaps
    )
    assert "cross-source conflict penalty" in (assumptions["A6"].latest_update or "")


def test_cross_check_task_is_recorded_in_task_results():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    cross_check_results = [
        result for result in state.research_task_results if result.task_type == "cross_check"
    ]

    assert len(cross_check_results) == 1
    assert cross_check_results[0].evidence_conflicts
