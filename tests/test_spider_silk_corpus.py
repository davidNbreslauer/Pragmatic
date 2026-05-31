from pathlib import Path

from pragmatic.corpus import SPIDER_SILK_CORPUS_PATH, load_corpus, resolve_corpus_path
from pragmatic.replay import run_replay_demo
from pragmatic.research_loop import run_research_loop
from pragmatic.schemas import Source


THESIS = "Spider silk for bullet proof vests"


def test_spider_silk_corpus_loads_and_validates_sources():
    sources = load_corpus(SPIDER_SILK_CORPUS_PATH)

    assert len(sources) == 8
    for source in sources:
        Source.model_validate(source.model_dump())
        assert source.id.startswith("spider_")
        assert len(source.text) <= 900


def test_spider_silk_thesis_auto_routes_to_prepared_corpus():
    assert resolve_corpus_path(THESIS) == SPIDER_SILK_CORPUS_PATH
    assert resolve_corpus_path(THESIS, Path("custom.json")) == Path("custom.json")


def test_spider_silk_loop_produces_offline_demo_artifacts():
    state = run_research_loop(
        THESIS,
        source_mode="prepared",
        allow_live_web_search=False,
        observability_mode="off",
    )

    assert state.sources
    assert all(source.id.startswith("spider_") for source in state.sources)
    assert any(assumption.confidence > 0 for assumption in state.assumptions)
    assert any(item.evidence_type == "direct" for item in state.evidence_items)
    assert any(item.evidence_type == "contradictory" for item in state.evidence_items)
    assert any(item.evidence_type == "proxy" for item in state.evidence_items)
    assert state.invalid_leaps
    assert state.generated_evals
    assert state.decisive_tests
    assert any("NIJ Level IIIA" in test.test and "V50" in test.test for test in state.decisive_tests)

    bottom_line_source = " ".join(
        [update.rationale for update in state.belief_updates]
        + [leap.why_invalid for leap in state.invalid_leaps]
    )
    assert bottom_line_source


def test_spider_silk_replay_lowers_overcredited_confidence():
    replay = run_replay_demo(
        THESIS,
        corpus_path=SPIDER_SILK_CORPUS_PATH,
        observability_mode="off",
    )

    assert replay.applied_eval_rules
    deltas = [
        comparison.after_confidence - comparison.before_confidence
        for comparison in replay.comparisons
    ]
    assert min(deltas) < 0
    assert max(deltas) <= 0
