from __future__ import annotations

from thesisgraph.schemas import EvidenceConflict, EvidenceItem, Source


def detect_evidence_conflicts(
    sources: list[Source],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceConflict]:
    source_by_id = {source.id: source for source in sources}
    conflicts: list[EvidenceConflict] = []

    conflicts.extend(_prospective_validation_conflicts(source_by_id, evidence_items))
    conflicts.extend(_benchmark_cluster_conflicts(source_by_id, evidence_items))
    conflicts.extend(_weak_baseline_conflicts(source_by_id, evidence_items))
    if not _is_demo_source_set(sources):
        conflicts.extend(_generic_conflicts(source_by_id, evidence_items))
    return conflicts


def _prospective_validation_conflicts(
    source_by_id: dict[str, Source],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceConflict]:
    a6_items = [item for item in evidence_items if "A6" in item.assumption_ids]
    contradictory_items = [
        item for item in a6_items if item.evidence_type == "contradictory"
    ]
    weak_support_items = [
        item
        for item in a6_items
        if item.evidence_type in {"proxy", "anecdotal"}
    ]
    company_items = [
        item
        for item in a6_items
        if source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type == "company_claim"
    ]

    conflicts: list[EvidenceConflict] = []
    if contradictory_items and weak_support_items:
        all_items = contradictory_items + weak_support_items
        conflicts.append(
            EvidenceConflict(
                id="conflict_a6_sparse_prospective_validation",
                conflict_type="contradiction",
                severity="high",
                summary=(
                    "Prospective validation has weak proxy/anecdotal support while a review source "
                    "warns that prospective validation remains sparse."
                ),
                source_ids=sorted({item.source_id for item in all_items}),
                evidence_item_ids=sorted({item.id for item in all_items}),
                affected_assumption_ids=["A6"],
                suggested_action=(
                    "Require a frozen prediction with independent experimental or simulator confirmation."
                ),
            )
        )

    independent_direct_items = [
        item
        for item in a6_items
        if item.evidence_type == "direct"
        and source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type not in {"company_claim", "benchmark"}
    ]
    if company_items and not independent_direct_items:
        conflicts.append(
            EvidenceConflict(
                id="conflict_company_claim_without_independent_validation",
                conflict_type="source_type_imbalance",
                severity="high",
                summary=(
                    "Company claims are present, but no independent direct prospective-validation "
                    "source supports the discovery-acceleration claim."
                ),
                source_ids=sorted({item.source_id for item in company_items}),
                evidence_item_ids=sorted({item.id for item in company_items}),
                affected_assumption_ids=["A6", "A7"],
                suggested_action=(
                    "Separate product claims from independently validated outcomes in the belief graph."
                ),
            )
        )

    return conflicts


def _benchmark_cluster_conflicts(
    source_by_id: dict[str, Source],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceConflict]:
    benchmark_items = [
        item
        for item in evidence_items
        if source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type == "benchmark"
        and any(assumption_id in {"A5", "A6"} for assumption_id in item.assumption_ids)
    ]
    direct_a6_items = [
        item
        for item in evidence_items
        if "A6" in item.assumption_ids
        and item.evidence_type == "direct"
        and source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type != "benchmark"
    ]
    if not benchmark_items or direct_a6_items:
        return []

    return [
        EvidenceConflict(
            id="conflict_benchmark_cluster_without_outcomes",
            conflict_type="source_type_imbalance",
            severity="medium",
            summary=(
                "Benchmark evidence clusters around research skill measurement, but no direct "
                "prospective outcome evidence anchors the discovery claim."
            ),
            source_ids=sorted({item.source_id for item in benchmark_items}),
            evidence_item_ids=sorted({item.id for item in benchmark_items}),
            affected_assumption_ids=["A5", "A6"],
            suggested_action=(
                "Keep benchmark evidence as construct-validity support until outcome evidence is present."
            ),
        )
    ]


def _weak_baseline_conflicts(
    source_by_id: dict[str, Source],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceConflict]:
    weak_baseline_items = [
        item
        for item in evidence_items
        if "baseline is weak" in item.limitation.lower()
        and source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type == "case_study"
    ]
    if not weak_baseline_items:
        return []

    return [
        EvidenceConflict(
            id="conflict_weak_baseline_cluster",
            conflict_type="weak_source_cluster",
            severity="medium",
            summary=(
                "The available baseline evidence comes from selected examples with a weak comparison baseline."
            ),
            source_ids=sorted({item.source_id for item in weak_baseline_items}),
            evidence_item_ids=sorted({item.id for item in weak_baseline_items}),
            affected_assumption_ids=["A2", "A7"],
            suggested_action=(
                "Require matched-budget comparisons against strong non-graph RAG and expert-search baselines."
            ),
        )
    ]


def _generic_conflicts(
    source_by_id: dict[str, Source],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceConflict]:
    conflicts: list[EvidenceConflict] = []
    assumption_ids = sorted(
        {
            assumption_id
            for item in evidence_items
            for assumption_id in item.assumption_ids
        }
    )
    for assumption_id in assumption_ids:
        items = [item for item in evidence_items if assumption_id in item.assumption_ids]
        supportive = [
            item
            for item in items
            if item.evidence_type in {"direct", "indirect", "proxy", "anecdotal"}
        ]
        contradictory = [item for item in items if item.evidence_type == "contradictory"]
        if supportive and contradictory:
            all_items = supportive + contradictory
            conflicts.append(
                EvidenceConflict(
                    id=f"conflict_{assumption_id.lower()}_support_vs_limitation",
                    conflict_type="contradiction",
                    severity="high",
                    summary=(
                        "Supportive evidence and limiting or contradictory evidence both attach "
                        f"to {assumption_id}; the answer should preserve this uncertainty."
                    ),
                    source_ids=sorted({item.source_id for item in all_items}),
                    evidence_item_ids=sorted({item.id for item in all_items}),
                    affected_assumption_ids=[assumption_id],
                    suggested_action=(
                        "Separate what is directly tested from what remains a limitation or open question."
                    ),
                )
            )

        source_types = {
            source_by_id[item.source_id].source_type
            for item in supportive
            if source_by_id.get(item.source_id) is not None
        }
        has_direct_independent = any(
            item.evidence_type == "direct"
            and source_by_id.get(item.source_id) is not None
            and source_by_id[item.source_id].source_type
            in {"paper", "review", "standard", "government", "dataset"}
            for item in supportive
        )
        if source_types & {"company_claim", "blog_post", "news", "unknown"} and not has_direct_independent:
            conflicts.append(
                EvidenceConflict(
                    id=f"conflict_{assumption_id.lower()}_weak_source_mix",
                    conflict_type="source_type_imbalance",
                    severity="medium",
                    summary=(
                        f"{assumption_id} is supported mainly by weaker source types without "
                        "independent direct validation."
                    ),
                    source_ids=sorted({item.source_id for item in supportive}),
                    evidence_item_ids=sorted({item.id for item in supportive}),
                    affected_assumption_ids=[assumption_id],
                    suggested_action=(
                        "Require stronger independent or standards-relevant sources before increasing confidence."
                    ),
                )
            )
    return conflicts


def _is_demo_source_set(sources: list[Source]) -> bool:
    return any(source.id.startswith("source_") and "AI" in source.title for source in sources)
