from __future__ import annotations

from pragmatic.schemas import (
    BeliefUpdate,
    EvidenceConflict,
    EvidenceItem,
    InvalidLeap,
    ResearchState,
    SupportLevel,
    VerifierResult,
)


EVIDENCE_WEIGHTS = {
    "direct": 0.55,
    "indirect": 0.35,
    "proxy": 0.14,
    "anecdotal": 0.08,
    "contradictory": -0.20,
    "irrelevant": 0.0,
    "not_relevant": 0.0,
}

LEAP_PENALTY = 0.12
CONFLICT_PENALTIES = {
    "low": 0.03,
    "medium": 0.08,
    "high": 0.14,
}


def update_beliefs(state: ResearchState) -> list[BeliefUpdate]:
    updates: list[BeliefUpdate] = []
    evidence_by_assumption = _evidence_by_assumption(state.evidence_items)
    leaps_by_assumption = _leaps_by_assumption(state.invalid_leaps)
    conflicts_by_assumption = _conflicts_by_assumption(state.evidence_conflicts)
    verifier_results_by_assumption = _verifier_results_by_assumption(state.verifier_results)

    for assumption in state.assumptions:
        items = evidence_by_assumption.get(assumption.id, [])
        leaps = leaps_by_assumption.get(assumption.id, [])
        conflicts = conflicts_by_assumption.get(assumption.id, [])
        verifier_results = verifier_results_by_assumption.get(assumption.id, [])
        score = sum(EVIDENCE_WEIGHTS[item.evidence_type] for item in items)
        score -= LEAP_PENALTY * len(leaps)
        score -= sum(CONFLICT_PENALTIES[conflict.severity] for conflict in conflicts)
        score += sum(result.confidence_delta for result in verifier_results)
        score = min(max(score, 0.0), 1.0)

        support = _support_from_score(score, items)
        rationale = _rationale(items, leaps, conflicts, verifier_results)
        updates.append(
            BeliefUpdate(
                assumption_id=assumption.id,
                previous_support=assumption.support_level,
                new_support=support,
                previous_confidence=assumption.confidence,
                new_confidence=round(score, 2),
                rationale=rationale,
            )
        )

    return updates


def apply_belief_updates(state: ResearchState, updates: list[BeliefUpdate]) -> None:
    updates_by_assumption_id = {update.assumption_id: update for update in updates}
    for assumption in state.assumptions:
        update = updates_by_assumption_id.get(assumption.id)
        if update is None:
            continue
        assumption.support_level = update.new_support
        assumption.confidence = update.new_confidence
        assumption.latest_update = update.rationale


def _evidence_by_assumption(items: list[EvidenceItem]) -> dict[str, list[EvidenceItem]]:
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in items:
        for assumption_id in item.assumption_ids:
            grouped.setdefault(assumption_id, []).append(item)
    return grouped


def _leaps_by_assumption(leaps: list[InvalidLeap]) -> dict[str, list[InvalidLeap]]:
    grouped: dict[str, list[InvalidLeap]] = {}
    for leap in leaps:
        for assumption_id in leap.affected_assumption_ids:
            grouped.setdefault(assumption_id, []).append(leap)
    return grouped


def _conflicts_by_assumption(
    conflicts: list[EvidenceConflict],
) -> dict[str, list[EvidenceConflict]]:
    grouped: dict[str, list[EvidenceConflict]] = {}
    for conflict in conflicts:
        for assumption_id in conflict.affected_assumption_ids:
            grouped.setdefault(assumption_id, []).append(conflict)
    return grouped


def _verifier_results_by_assumption(
    verifier_results: list[VerifierResult],
) -> dict[str, list[VerifierResult]]:
    grouped: dict[str, list[VerifierResult]] = {}
    for result in verifier_results:
        for assumption_id in result.affected_assumption_ids:
            grouped.setdefault(assumption_id, []).append(result)
    return grouped


def _support_from_score(score: float, items: list[EvidenceItem]) -> SupportLevel:
    if not items:
        return "unknown"
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "moderate"
    if score >= 0.22:
        return "weak"
    if any(item.evidence_type == "contradictory" for item in items) and score == 0.0:
        return "unsupported"
    return "unsupported"


def _rationale(
    items: list[EvidenceItem],
    leaps: list[InvalidLeap],
    conflicts: list[EvidenceConflict],
    verifier_results: list[VerifierResult],
) -> str:
    if not items:
        return "No evidence was retrieved for this assumption."
    evidence_counts: dict[str, int] = {}
    for item in items:
        evidence_counts[item.evidence_type] = evidence_counts.get(item.evidence_type, 0) + 1
    parts = [f"{count} {kind} evidence" for kind, count in sorted(evidence_counts.items())]
    if leaps:
        parts.append(f"{len(leaps)} invalid leap penalty")
    if conflicts:
        parts.append(f"{len(conflicts)} cross-source conflict penalty")
    if verifier_results:
        parts.append(f"{len(verifier_results)} verifier result adjustment")
    return "; ".join(parts) + "."
