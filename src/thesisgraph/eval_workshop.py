from __future__ import annotations

from thesisgraph.schemas import (
    EvalWorkshopLink,
    EvalWorkshopRecord,
    EvalWorkshopTaskSpan,
    GeneratedEval,
    InvalidLeap,
    ReplayComparison,
    ReplayOutcomeRecord,
    ResearchState,
)


def build_eval_workshop(
    state: ResearchState,
    *,
    replay_first_pass: ResearchState | None = None,
    replay_comparisons: list[ReplayComparison] | None = None,
    applied_eval_rules: list[str] | None = None,
) -> EvalWorkshopRecord:
    task_spans = build_task_spans(state)
    failure_eval_links = build_failure_eval_links(state)
    replay_outcomes = build_replay_outcomes(
        replay_first_pass=replay_first_pass,
        replay_pass=state,
        replay_comparisons=replay_comparisons or [],
        applied_eval_rules=applied_eval_rules or [],
    )

    summary = (
        f"Captured {len(task_spans)} task spans, {len(failure_eval_links)} failure links, "
        f"and {len(replay_outcomes)} replay outcomes."
    )
    return EvalWorkshopRecord(
        task_spans=task_spans,
        failure_eval_links=failure_eval_links,
        replay_outcomes=replay_outcomes,
        summary=summary,
    )


def build_task_spans(state: ResearchState) -> list[EvalWorkshopTaskSpan]:
    return [
        EvalWorkshopTaskSpan(
            id=f"span_{index:03d}",
            task_id=result.task_id,
            task_type=result.task_type,
            backend=result.backend,
            status=result.status,
            source_ids=result.source_ids,
            evidence_item_count=len(result.evidence_items),
            evidence_conflict_count=len(result.evidence_conflicts),
            verifier_result_count=len(result.verifier_results),
            error=result.error,
        )
        for index, result in enumerate(state.research_task_results, start=1)
    ]


def build_failure_eval_links(state: ResearchState) -> list[EvalWorkshopLink]:
    links: list[EvalWorkshopLink] = []
    links.extend(_invalid_leap_to_eval_links(state.invalid_leaps, state.generated_evals))
    links.extend(_conflict_to_invalid_leap_links(state))
    links.extend(_verifier_failure_to_eval_links(state))
    return links


def build_replay_outcomes(
    *,
    replay_first_pass: ResearchState | None,
    replay_pass: ResearchState,
    replay_comparisons: list[ReplayComparison],
    applied_eval_rules: list[str],
) -> list[ReplayOutcomeRecord]:
    if replay_first_pass is None or not replay_comparisons:
        return []

    eval_id_by_rule = {
        generated_eval.eval_rule: generated_eval.id
        for generated_eval in replay_first_pass.generated_evals
    }
    fallback_rule = applied_eval_rules[0] if applied_eval_rules else ""
    fallback_eval_id = eval_id_by_rule.get(fallback_rule)
    outcomes: list[ReplayOutcomeRecord] = []

    for index, comparison in enumerate(replay_comparisons, start=1):
        rule = fallback_rule or "No generated eval rule was applied."
        passed = comparison.after_confidence <= comparison.before_confidence
        outcomes.append(
            ReplayOutcomeRecord(
                id=f"replay_outcome_{index:03d}",
                assumption_id=comparison.assumption_id,
                generated_eval_id=fallback_eval_id,
                applied_eval_rule=rule,
                before_confidence=comparison.before_confidence,
                after_confidence=comparison.after_confidence,
                passed=passed,
                summary=(
                    f"{comparison.assumption_id}: {comparison.before_confidence} -> "
                    f"{comparison.after_confidence}; {comparison.change_summary}."
                ),
            )
        )
    return outcomes


def _invalid_leap_to_eval_links(
    invalid_leaps: list[InvalidLeap],
    generated_evals: list[GeneratedEval],
) -> list[EvalWorkshopLink]:
    links: list[EvalWorkshopLink] = []
    for index, (leap, generated_eval) in enumerate(
        zip(invalid_leaps, generated_evals, strict=False),
        start=1,
    ):
        links.append(
            EvalWorkshopLink(
                id=f"link_invalid_leap_eval_{index:03d}",
                link_type="invalid_leap_to_eval",
                source_id=leap.id,
                target_id=generated_eval.id,
                summary=f"{leap.leap} produced {generated_eval.id}.",
                affected_assumption_ids=leap.affected_assumption_ids,
                source_ids=leap.source_ids,
            )
        )
    return links


def _conflict_to_invalid_leap_links(state: ResearchState) -> list[EvalWorkshopLink]:
    links: list[EvalWorkshopLink] = []
    for index, conflict in enumerate(state.evidence_conflicts, start=1):
        leap = _best_invalid_leap_for_conflict(conflict.affected_assumption_ids, state.invalid_leaps)
        if leap is None:
            continue
        links.append(
            EvalWorkshopLink(
                id=f"link_conflict_leap_{index:03d}",
                link_type="evidence_conflict_to_invalid_leap",
                source_id=conflict.id,
                target_id=leap.id,
                summary=f"{conflict.id} contributed to {leap.id}.",
                affected_assumption_ids=conflict.affected_assumption_ids,
                source_ids=conflict.source_ids,
            )
        )
    return links


def _verifier_failure_to_eval_links(state: ResearchState) -> list[EvalWorkshopLink]:
    links: list[EvalWorkshopLink] = []
    for index, verifier_result in enumerate(state.verifier_results, start=1):
        if verifier_result.status != "fail":
            continue
        generated_eval = _best_eval_for_assumptions(
            verifier_result.affected_assumption_ids,
            state.invalid_leaps,
            state.generated_evals,
        )
        if generated_eval is None:
            continue
        links.append(
            EvalWorkshopLink(
                id=f"link_verifier_eval_{index:03d}",
                link_type="verifier_failure_to_eval",
                source_id=verifier_result.id,
                target_id=generated_eval.id,
                summary=f"{verifier_result.id} is guarded by {generated_eval.id}.",
                affected_assumption_ids=verifier_result.affected_assumption_ids,
                source_ids=[],
            )
        )
    return links


def _best_invalid_leap_for_conflict(
    assumption_ids: list[str],
    invalid_leaps: list[InvalidLeap],
) -> InvalidLeap | None:
    assumption_set = set(assumption_ids)
    mixed_leap = next(
        (leap for leap in invalid_leaps if leap.id == "leap_mixed_sources_to_stable_belief"),
        None,
    )
    if mixed_leap is not None and assumption_set.intersection(mixed_leap.affected_assumption_ids):
        return mixed_leap
    return next(
        (
            leap
            for leap in invalid_leaps
            if assumption_set.intersection(leap.affected_assumption_ids)
        ),
        None,
    )


def _best_eval_for_assumptions(
    assumption_ids: list[str],
    invalid_leaps: list[InvalidLeap],
    generated_evals: list[GeneratedEval],
) -> GeneratedEval | None:
    assumption_set = set(assumption_ids)
    leap_eval_pairs = list(zip(invalid_leaps, generated_evals, strict=False))
    mixed_pair = next(
        (
            (leap, generated_eval)
            for leap, generated_eval in leap_eval_pairs
            if leap.id == "leap_mixed_sources_to_stable_belief"
        ),
        None,
    )
    if mixed_pair is not None:
        return mixed_pair[1]
    match = next(
        (
            generated_eval
            for leap, generated_eval in leap_eval_pairs
            if assumption_set.intersection(leap.affected_assumption_ids)
        ),
        None,
    )
    return match
