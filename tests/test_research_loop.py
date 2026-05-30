from thesisgraph import DEFAULT_THESIS, run_research_loop
from thesisgraph.corpus import load_corpus


def test_load_corpus_validates_sources():
    sources = load_corpus()

    assert len(sources) >= 8
    assert sources[0].id == "source_001"


def test_research_loop_produces_core_state():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    assert state.thesis.text == DEFAULT_THESIS
    assert len(state.assumptions) == 8
    assert len(state.research_questions) == 4
    assert len(state.sources) >= 8
    assert len(state.evidence_items) >= 8
    assert state.invalid_leaps
    assert state.belief_updates
    assert state.generated_evals
    assert state.decisive_tests


def test_proxy_benchmark_generates_failure_eval():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    benchmark_evidence = [
        item for item in state.evidence_items if item.source_id in {"source_004", "source_005"}
    ]
    assert benchmark_evidence
    assert all(item.evidence_type == "proxy" for item in benchmark_evidence)
    assert any("benchmark" in leap.leap.lower() for leap in state.invalid_leaps)
    assert any("proxy evidence" in generated_eval.eval_rule for generated_eval in state.generated_evals)


def test_belief_update_downgrades_prospective_validation():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    assumptions_by_id = {assumption.id: assumption for assumption in state.assumptions}

    assert assumptions_by_id["A6"].support_level == "unsupported"
    assert assumptions_by_id["A6"].confidence <= 0.2


def test_research_state_serializes_to_json():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    serialized = state.model_dump_json()

    assert "Graph-based AI scientist systems" in serialized
    assert "generated_evals" in serialized
