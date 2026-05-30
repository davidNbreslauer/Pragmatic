from thesisgraph import DEFAULT_THESIS
from thesisgraph.research_loop import run_research_loop
from thesisgraph.verifiers import build_verifier_tasks, run_mock_verifier


def test_mock_verifier_fails_without_direct_prospective_validation():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    result = run_mock_verifier(
        state.decisive_tests[0],
        state.sources,
        state.evidence_items,
        state.evidence_conflicts,
    )

    assert result.status == "fail"
    assert result.confidence_delta < 0
    assert "A6" in result.affected_assumption_ids
    assert result.failed_criteria


def test_verifier_tasks_carry_decisive_test_context():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    tasks = build_verifier_tasks(
        state.decisive_tests,
        state.sources,
        state.evidence_items,
        state.evidence_conflicts,
    )

    assert tasks
    assert tasks[0].task_type == "verify_decisive_test"
    assert tasks[0].decisive_test is not None
    assert tasks[0].evidence_conflicts


def test_research_loop_records_verifier_results_and_belief_adjustment():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    assumptions = {assumption.id: assumption for assumption in state.assumptions}

    assert state.verifier_results
    assert state.verifier_results[0].status == "fail"
    assert any(event.stage == "verifier" for event in state.trace_events)
    assert any(result.task_type == "verify_decisive_test" for result in state.research_task_results)
    assert "verifier result adjustment" in (assumptions["A6"].latest_update or "")
