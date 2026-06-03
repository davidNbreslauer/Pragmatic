from __future__ import annotations

from pragmatic.schemas import (
    DecisiveTest,
    EvidenceConflict,
    EvidenceItem,
    ResearchTask,
    Source,
    VerifierResult,
)


def build_verifier_tasks(
    decisive_tests: list[DecisiveTest],
    sources: list[Source],
    evidence_items: list[EvidenceItem],
    evidence_conflicts: list[EvidenceConflict],
) -> list[ResearchTask]:
    return [
        ResearchTask(
            id=f"task_verify_{index:03d}_{decisive_test.id}",
            task_type="verify_decisive_test",
            assumption_ids=decisive_test.would_resolve,
            sources=sorted(sources, key=lambda source: source.id),
            evidence_items=sorted(evidence_items, key=lambda item: item.id),
            evidence_conflicts=sorted(evidence_conflicts, key=lambda conflict: conflict.id),
            decisive_test=decisive_test,
            metadata={"decisive_test_id": decisive_test.id},
        )
        for index, decisive_test in enumerate(decisive_tests, start=1)
    ]


def run_mock_verifier(
    decisive_test: DecisiveTest,
    sources: list[Source],
    evidence_items: list[EvidenceItem],
    evidence_conflicts: list[EvidenceConflict],
) -> VerifierResult:
    return _run_generic_verifier(
        decisive_test,
        sources,
        evidence_items,
        evidence_conflicts,
    )


def _run_generic_verifier(
    decisive_test: DecisiveTest,
    sources: list[Source],
    evidence_items: list[EvidenceItem],
    evidence_conflicts: list[EvidenceConflict],
) -> VerifierResult:
    source_by_id = {source.id: source for source in sources}
    direct_independent_items = [
        item
        for item in evidence_items
        if item.evidence_type == "direct"
        and source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type
        in {"paper", "review", "standard", "government", "dataset", "case_study"}
    ]
    limiting_items = [item for item in evidence_items if item.evidence_type == "contradictory"]
    high_conflicts = [conflict for conflict in evidence_conflicts if conflict.severity == "high"]

    passed_criteria: list[str] = []
    failed_criteria: list[str] = []
    for criterion in decisive_test.success_criteria:
        lower = criterion.lower()
        if "success criteria" in lower:
            if any(source.source_type in {"standard", "government", "review"} for source in sources):
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)
        elif "directly tests" in lower:
            if direct_independent_items:
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)
        elif "limitations" in lower:
            if limiting_items or high_conflicts:
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)
        elif "compared against" in lower:
            if any("compar" in item.claim_supported.lower() or "alternative" in item.claim_supported.lower() for item in evidence_items):
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)
        else:
            failed_criteria.append(criterion)

    status = "pass" if direct_independent_items and not high_conflicts and len(failed_criteria) <= 1 else "fail"
    return VerifierResult(
        id=f"verifier_{decisive_test.id}",
        decisive_test_id=decisive_test.id,
        status=status,
        affected_assumption_ids=decisive_test.would_resolve,
        confidence_delta=0.10 if status == "pass" else -0.08,
        rationale=(
            "The generic verifier checked for independent direct evidence, standards-relevant "
            "criteria, limitation handling, and comparison against alternatives."
        ),
        passed_criteria=passed_criteria,
        failed_criteria=failed_criteria,
        evidence_item_ids=[item.id for item in direct_independent_items + limiting_items],
    )
