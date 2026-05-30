from __future__ import annotations

from thesisgraph.schemas import DecisiveTest, ResearchState


def propose_decisive_tests(state: ResearchState) -> list[DecisiveTest]:
    return [
        DecisiveTest(
            id="test_001",
            test=(
                "Run a blinded prospective materials-discovery challenge comparing a graph agent, "
                "a non-graph RAG agent, and a human expert baseline."
            ),
            would_resolve=["A3", "A4", "A6", "A7", "A8"],
            success_criteria=[
                "Independent experts judge hypotheses for novelty and testability before validation.",
                "Predictions are frozen before experimental or simulator outcomes are known.",
                "Graph-agent gains exceed strong non-graph baselines under matched budget.",
                "At least one prediction is prospectively validated.",
            ],
            why_decisive=(
                "It tests prospective discovery value rather than retrospective recall, benchmark QA, "
                "or plausible report generation."
            ),
        )
    ]

