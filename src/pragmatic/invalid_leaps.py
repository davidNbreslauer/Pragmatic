from __future__ import annotations

from pragmatic.schemas import InvalidLeap, ResearchState


def detect_invalid_leaps(state: ResearchState) -> list[InvalidLeap]:
    leaps: list[InvalidLeap] = []

    benchmark_source_ids = sorted(
        {
            item.source_id
            for item in state.evidence_items
            if item.evidence_type in {"direct", "indirect", "proxy"}
            and _source_type(state, item.source_id) == "benchmark"
        }
    )
    if benchmark_source_ids:
        leaps.append(
            InvalidLeap(
                id="leap_benchmark_to_discovery",
                leap="Benchmark QA performance -> real scientific discovery.",
                why_invalid=(
                    "Benchmark performance must be bounded as proxy evidence. It does not establish "
                    "prospective discovery, novelty, experimental validation, or real-world acceleration."
                ),
                source_ids=benchmark_source_ids,
                affected_assumption_ids=["A5", "A6"],
                suggested_followup_question=(
                    "What evidence exists for prospective validation of AI-generated materials hypotheses?"
                ),
            )
        )

    hypothesis_source_ids = sorted(
        {
            item.source_id
            for item in state.evidence_items
            if "hypothes" in item.claim_supported.lower()
            and item.evidence_type in {"indirect", "proxy", "anecdotal"}
        }
    )
    if hypothesis_source_ids:
        leaps.append(
            InvalidLeap(
                id="leap_plausible_to_useful",
                leap="Plausible hypotheses -> useful novel hypotheses.",
                why_invalid=(
                    "A hypothesis can be plausible but obvious, untestable, false, or low-value."
                ),
                source_ids=hypothesis_source_ids,
                affected_assumption_ids=["A3", "A4"],
                suggested_followup_question=(
                    "Were generated hypotheses evaluated by blinded experts or validated experimentally?"
                ),
            )
        )

    graph_source_ids = sorted(
        {
            item.source_id
            for item in state.evidence_items
            if "graph" in item.claim_supported.lower()
            and item.evidence_type in {"indirect", "proxy", "anecdotal"}
        }
    )
    if graph_source_ids:
        leaps.append(
            InvalidLeap(
                id="leap_graph_to_mechanism",
                leap="Graph representation -> causal/mechanistic understanding.",
                why_invalid=(
                    "A graph can encode entities, citations, or correlations without encoding causal mechanism."
                ),
                source_ids=graph_source_ids,
                affected_assumption_ids=["A1"],
                suggested_followup_question=(
                    "What ablations show that graph structure improves mechanistic reasoning rather than retrieval alone?"
                ),
            )
        )

    company_source_ids = sorted(
        source.id for source in state.sources if source.source_type == "company_claim"
    )
    if company_source_ids:
        leaps.append(
            InvalidLeap(
                id="leap_claim_to_validated_outcome",
                leap="Company claim -> validated outcome.",
                why_invalid=(
                    "A company or product claim is anecdotal unless supported by independent validation."
                ),
                source_ids=company_source_ids,
                affected_assumption_ids=["A6", "A7"],
                suggested_followup_question=(
                    "Which independent prospective studies validate the claimed discovery acceleration?"
                ),
            )
        )

    severe_conflicts = [
        conflict
        for conflict in state.evidence_conflicts
        if conflict.severity == "high"
        and conflict.conflict_type in {"contradiction", "source_type_imbalance"}
    ]
    if severe_conflicts:
        leaps.append(
            InvalidLeap(
                id="leap_mixed_sources_to_stable_belief",
                leap="Mixed weak/conflicting sources -> stable discovery belief.",
                why_invalid=(
                    "Cross-source conflicts mean the system must preserve source-type boundaries "
                    "instead of smoothing weak, contradictory, or promotional evidence into support."
                ),
                source_ids=sorted(
                    {
                        source_id
                        for conflict in severe_conflicts
                        for source_id in conflict.source_ids
                    }
                ),
                affected_assumption_ids=sorted(
                    {
                        assumption_id
                        for conflict in severe_conflicts
                        for assumption_id in conflict.affected_assumption_ids
                    }
                ),
                suggested_followup_question=(
                    "Which independent sources directly validate the disputed discovery outcome?"
                ),
            )
        )

    if not _is_demo_ai_scientist_state(state):
        leaps.extend(_generic_invalid_leaps(state))

    return leaps


def _generic_invalid_leaps(state: ResearchState) -> list[InvalidLeap]:
    leaps: list[InvalidLeap] = []
    proxy_or_indirect = [
        item
        for item in state.evidence_items
        if item.evidence_type in {"proxy", "indirect", "anecdotal"}
    ]
    direct_items = [item for item in state.evidence_items if item.evidence_type == "direct"]
    contradictory_items = [
        item for item in state.evidence_items if item.evidence_type == "contradictory"
    ]

    if proxy_or_indirect:
        affected = sorted(
            {
                assumption_id
                for item in proxy_or_indirect
                for assumption_id in item.assumption_ids
            }
        )
        leaps.append(
            InvalidLeap(
                id="leap_proxy_to_application_ready",
                leap="Property/proxy evidence -> application-ready conclusion.",
                why_invalid=(
                    "Evidence about material properties, related mechanisms, or non-final demonstrations "
                    "does not by itself prove the claimed real-world application works."
                ),
                source_ids=sorted({item.source_id for item in proxy_or_indirect}),
                affected_assumption_ids=affected,
                suggested_followup_question=(
                    "Which sources test the final application or a standards-relevant surrogate directly?"
                ),
            )
        )

    standards_assumptions = [
        assumption.id
        for assumption in state.assumptions
        if any(
            token in assumption.text.lower()
            for token in ["standard", "safety", "validation", "independent", "testing"]
        )
    ]
    if standards_assumptions and not direct_items:
        leaps.append(
            InvalidLeap(
                id="leap_no_direct_validation_to_confidence",
                leap="No direct validation evidence -> confident practical answer.",
                why_invalid=(
                    "A practical recommendation should remain uncertain without independent, "
                    "application-level, or standards-relevant validation."
                ),
                source_ids=sorted({item.source_id for item in state.evidence_items}),
                affected_assumption_ids=standards_assumptions,
                suggested_followup_question=(
                    "What independent validation or standards test directly supports the claim?"
                ),
            )
        )

    if contradictory_items:
        leaps.append(
            InvalidLeap(
                id="leap_limitations_to_unqualified_yes",
                leap="Known limitations -> unqualified yes/no answer.",
                why_invalid=(
                    "Contradictory or limiting evidence must be carried into the final belief graph "
                    "instead of being averaged away."
                ),
                source_ids=sorted({item.source_id for item in contradictory_items}),
                affected_assumption_ids=sorted(
                    {
                        assumption_id
                        for item in contradictory_items
                        for assumption_id in item.assumption_ids
                    }
                ),
                suggested_followup_question=(
                    "Which limitations change the answer, and which decisive evidence would resolve them?"
                ),
            )
        )

    return leaps


def _source_type(state: ResearchState, source_id: str) -> str | None:
    for source in state.sources:
        if source.id == source_id:
            return source.source_type
    return None


def _is_demo_ai_scientist_state(state: ResearchState) -> bool:
    return any("graph memory captures" in assumption.text.lower() for assumption in state.assumptions)
