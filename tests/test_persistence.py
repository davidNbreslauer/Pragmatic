from thesisgraph import DEFAULT_THESIS
from thesisgraph.persistence import compare_runs, list_runs, load_run, save_run
from thesisgraph.research_loop import run_research_loop


def test_save_and_load_run_round_trips_research_state(tmp_path):
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    summary = save_run(state, run_dir=tmp_path, run_id="run_test")
    loaded = load_run("run_test", run_dir=tmp_path)
    runs = list_runs(run_dir=tmp_path)

    assert summary.run_id == "run_test"
    assert summary.assumption_count == len(state.assumptions)
    assert loaded.thesis.text == state.thesis.text
    assert loaded.generated_evals[0].id == state.generated_evals[0].id
    assert runs[0].run_id == "run_test"


def test_save_run_updates_index_for_existing_run_id(tmp_path):
    first_state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    second_state = first_state.model_copy(deep=True)
    second_state.thesis.text = "Updated thesis text"

    save_run(first_state, run_dir=tmp_path, run_id="run_same")
    save_run(second_state, run_dir=tmp_path, run_id="run_same")
    runs = list_runs(run_dir=tmp_path)
    loaded = load_run("run_same", run_dir=tmp_path)

    assert len(runs) == 1
    assert runs[0].thesis_text == "Updated thesis text"
    assert loaded.thesis.text == "Updated thesis text"


def test_compare_runs_reports_belief_delta_for_a6():
    baseline = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    current = baseline.model_copy(deep=True)
    for assumption in current.assumptions:
        if assumption.id == "A6":
            assumption.confidence = 0.25
            assumption.support_level = "weak"
            assumption.latest_update = "Manual comparison fixture."

    comparison = compare_runs(
        baseline,
        current,
        baseline_run_id="baseline",
        current_run_id="current",
    )
    deltas = {delta.assumption_id: delta for delta in comparison.deltas}

    assert deltas["A6"].delta == 0.25
    assert deltas["A6"].current_support == "weak"
    assert "1 of 8 assumptions changed" in comparison.summary
