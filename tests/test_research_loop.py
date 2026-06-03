from pragmatic import DEFAULT_THESIS, run_research_loop
from pragmatic.corpus import load_corpus, score_source_for_question
from pragmatic.research_loop import decompose_thesis, generate_initial_questions
from pragmatic.schemas import ResearchState, Thesis


def test_load_corpus_validates_sources():
    sources = load_corpus()

    assert len(sources) >= 8
    assert sources[0].id == "spider_001"
    assert sources[0].published_year == 2026
    assert "spider-silk" in sources[0].tags
    assert sources[0].evidence_scope == "material-property support, not ballistic vest validation"


def test_retrieval_scoring_matches_query_terms():
    sources = {source.id: source for source in load_corpus()}
    state = ResearchState(
        thesis=Thesis(text=DEFAULT_THESIS),
        assumptions=decompose_thesis(DEFAULT_THESIS),
    )
    questions = generate_initial_questions(state)
    application_question = next(question for question in questions if question.id == "Q3")

    standard_score = score_source_for_question(sources["spider_006"], application_question)
    property_score = score_source_for_question(sources["spider_001"], application_question)

    assert standard_score.score > 0
    assert property_score.score > 0
    assert "level" in standard_score.matched_terms


def test_research_loop_produces_core_state():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    assert state.thesis.text == DEFAULT_THESIS
    assert len(state.assumptions) == 8
    assert len(state.research_questions) == 8
    assert len(state.sources) >= 8
    assert len(state.retrieval_scores) == len(state.research_questions) * len(state.sources)
    assert len(state.evidence_items) >= 8
    assert state.invalid_leaps
    assert state.belief_updates
    assert state.generated_evals
    assert state.decisive_tests


def test_proxy_evidence_generates_failure_eval():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    proxy_evidence = [
        item for item in state.evidence_items if item.evidence_type in {"proxy", "indirect", "anecdotal"}
    ]
    assert proxy_evidence
    assert any("application-ready" in leap.leap.lower() for leap in state.invalid_leaps)
    assert any("directly measures" in generated_eval.eval_rule for generated_eval in state.generated_evals)


def test_belief_update_downgrades_prospective_validation():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    assumptions_by_id = {assumption.id: assumption for assumption in state.assumptions}

    assert assumptions_by_id["A6"].support_level == "unsupported"
    assert assumptions_by_id["A6"].confidence <= 0.2


def test_research_state_serializes_to_json():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    serialized = state.model_dump_json()

    assert "Spider silk for bullet proof vests" in serialized
    assert "retrieval_scores" in serialized
    assert "generated_evals" in serialized
