import json
from types import SimpleNamespace

from thesisgraph.research_loop import (
    decompose_thesis,
    generate_initial_questions,
    run_research_loop,
)
from thesisgraph.schemas import ResearchState, Source, Thesis
from thesisgraph.source_search import build_web_corpus


SPIDER_THESIS = "Can spider silk make a bullet proof vest?"


def test_generic_decomposition_tracks_arbitrary_thesis():
    assumptions = decompose_thesis(SPIDER_THESIS)
    questions = generate_initial_questions(
        ResearchState(thesis=Thesis(text=SPIDER_THESIS), assumptions=assumptions)
    )

    assert len(assumptions) == 8
    assert len(questions) == 8
    assert "spider silk" in assumptions[0].text.lower()
    assert "graph memory" not in assumptions[0].text.lower()
    assert all(question.query for question in questions)


def test_openai_web_search_response_normalizes_sources():
    questions = generate_initial_questions(
        ResearchState(thesis=Thesis(text=SPIDER_THESIS), assumptions=decompose_thesis(SPIDER_THESIS))
    )
    payload = {
        "sources": [
            {
                "title": "NIJ Ballistic Resistance Standard",
                "url": "https://nij.ojp.gov/example-standard",
                "source_type": "standard",
                "published_year": 2023,
                "tags": ["ballistic armor", "testing"],
                "evidence_scope": "Defines standards-relevant test criteria.",
                "text": "NIJ ballistic armor standards define projectile threats and test conditions for body armor.",
            }
        ]
    }

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text=json.dumps(payload))

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)

    sources = build_web_corpus(
        SPIDER_THESIS,
        questions,
        model="test-search-model",
        max_sources=3,
        client=fake_client,
    )

    assert sources[0].id == "web_001"
    assert sources[0].source_type == "standard"
    assert sources[0].url == "https://nij.ojp.gov/example-standard"
    assert fake_responses.kwargs["tools"][0]["type"] == "web_search"
    assert fake_responses.kwargs["model"] == "test-search-model"


def test_research_loop_can_use_mock_live_web_corpus(monkeypatch):
    mock_sources = [
        Source(
            id="web_001",
            title="Spider silk mechanical properties review",
            source_type="review",
            url="https://example.org/spider-silk-review",
            published_year=2024,
            tags=["spider-silk", "mechanical-properties", "review"],
            evidence_scope="Material-property evidence, not a finished vest test.",
            text=(
                "Spider silk has high toughness and tensile strength, but translating fiber "
                "properties into ballistic armor is difficult and application-level vest tests are limited."
            ),
        ),
        Source(
            id="web_002",
            title="NIJ ballistic armor performance criteria",
            source_type="standard",
            url="https://nij.ojp.gov/example-standard",
            published_year=2023,
            tags=["ballistic", "armor", "standard"],
            evidence_scope="Defines application-level ballistic armor success criteria.",
            text=(
                "Ballistic-resistant body armor is evaluated against projectile threats, "
                "penetration, and backface deformation under controlled test conditions."
            ),
        ),
    ]

    def fake_build_web_corpus(thesis_text, questions, **kwargs):
        assert thesis_text == SPIDER_THESIS
        assert questions
        assert kwargs["max_sources"] == 4
        return mock_sources

    monkeypatch.setattr("thesisgraph.research_loop.build_web_corpus", fake_build_web_corpus)

    state = run_research_loop(
        SPIDER_THESIS,
        source_mode="web",
        allow_live_web_search=True,
        max_web_sources=4,
        observability_mode="off",
    )

    assert {source.id for source in state.sources} == {"web_001", "web_002"}
    assert state.evidence_items
    assert any(leap.id.startswith("leap_") for leap in state.invalid_leaps)
    assert state.generated_evals
    assert any(event.stage == "web_search" for event in state.trace_events)
