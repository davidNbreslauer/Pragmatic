from __future__ import annotations

from thesisgraph.schemas import DecisiveTest, ResearchState


def propose_decisive_tests(state: ResearchState) -> list[DecisiveTest]:
    if not any("graph memory captures" in assumption.text.lower() for assumption in state.assumptions):
        return [
            DecisiveTest(
                id="test_001",
                test=(
                    "Find independent, application-level evidence that directly tests the thesis "
                    "against the relevant success criteria."
                ),
                would_resolve=[assumption.id for assumption in state.assumptions],
                success_criteria=[
                    "At least one credible source defines the application-level success criteria.",
                    "At least one independent source directly tests the claimed application or a close standards-relevant surrogate.",
                    "The evidence includes limitations or failure modes rather than only promotional support.",
                    "The result is compared against existing alternatives or incumbent practice.",
                ],
                why_decisive=(
                    "It distinguishes a practically supported conclusion from proxy-property evidence, "
                    "analogy, or unvalidated claims."
                ),
            )
        ]

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
