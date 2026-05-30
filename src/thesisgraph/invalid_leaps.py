from __future__ import annotations

from thesisgraph.schemas import InvalidLeap, ResearchState


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

    return leaps


def _source_type(state: ResearchState, source_id: str) -> str | None:
    for source in state.sources:
        if source.id == source_id:
            return source.source_type
    return None
