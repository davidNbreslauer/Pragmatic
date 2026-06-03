from __future__ import annotations

from pragmatic.schemas import DecisiveTest, ResearchState


def propose_decisive_tests(state: ResearchState) -> list[DecisiveTest]:
    if _is_spider_silk_thesis(state):
        return [
            DecisiveTest(
                id="test_001",
                test=(
                    "Run an NIJ Level IIIA and V50 ballistic test on a spider-silk-containing "
                    "panel at fixed areal density against an aramid control."
                ),
                would_resolve=[assumption.id for assumption in state.assumptions],
                success_criteria=[
                    "The spider-silk panel stops the specified Level IIIA rounds without perforation.",
                    "Back-face deformation remains within the NIJ limit.",
                    "V50 and trauma-depth results match or beat an aramid control at fixed areal density.",
                    "Panel construction, conditioning, and shot placement are independently documented.",
                ],
                why_decisive=(
                    "It tests ballistic armor performance directly instead of inferring vest utility "
                    "from quasi-static tensile toughness."
                ),
            )
        ]

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


def _is_spider_silk_thesis(state: ResearchState) -> bool:
    thesis = state.thesis.text.lower()
    return "spider" in thesis and "silk" in thesis and any(
        term in thesis for term in ["bullet", "vest", "armor", "ballistic"]
    )
