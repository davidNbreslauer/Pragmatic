from __future__ import annotations

from thesisgraph.schemas import (
    AgentRunStep,
    EvalWorkshopLink,
    EvalWorkshopConnectionRow,
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
    failure_eval_links.extend(_replay_eval_to_outcome_links(replay_outcomes))
    connection_rows = build_connection_rows(
        task_spans=task_spans,
        failure_eval_links=failure_eval_links,
        replay_outcomes=replay_outcomes,
    )

    summary = (
        f"Captured {len(task_spans)} task spans, {len(failure_eval_links)} failure links, "
        f"{len(replay_outcomes)} replay outcomes, and {len(connection_rows)} connection rows."
    )
    return EvalWorkshopRecord(
        task_spans=task_spans,
        failure_eval_links=failure_eval_links,
        replay_outcomes=replay_outcomes,
        connection_rows=connection_rows,
        summary=summary,
    )


def build_task_spans(state: ResearchState) -> list[EvalWorkshopTaskSpan]:
    step_by_task_type = _agent_step_by_task_type(state.agent_run.steps if state.agent_run else [])
    return [
        EvalWorkshopTaskSpan(
            id=f"span_{index:03d}",
            task_id=result.task_id,
            task_type=result.task_type,
            backend=result.backend,
            status=result.status,
            agent_step_id=(
                step_by_task_type[result.task_type].id
                if result.task_type in step_by_task_type
                else None
            ),
            agent_name=(
                step_by_task_type[result.task_type].agent_name
                if result.task_type in step_by_task_type
                else None
            ),
            tool_name=(
                step_by_task_type[result.task_type].tool_name
                if result.task_type in step_by_task_type
                else None
            ),
            worker_status=result.metadata.get("worker_status"),
            duration_ms=_duration_ms(result.metadata.get("duration_ms")),
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


def build_connection_rows(
    *,
    task_spans: list[EvalWorkshopTaskSpan],
    failure_eval_links: list[EvalWorkshopLink],
    replay_outcomes: list[ReplayOutcomeRecord],
) -> list[EvalWorkshopConnectionRow]:
    rows: list[EvalWorkshopConnectionRow] = []
    for index, span in enumerate(task_spans, start=1):
        rows.append(
            EvalWorkshopConnectionRow(
                id=f"connection_task_{index:03d}",
                specialist=span.agent_name,
                tool=span.tool_name,
                task_id=span.task_id,
                backend=span.backend,
                worker_status=span.worker_status,
                status=span.status,
                summary=(
                    f"{span.task_type} ran via {span.backend} for "
                    f"{len(span.source_ids)} source(s)."
                ),
            )
        )
    for index, link in enumerate(failure_eval_links, start=1):
        rows.append(
            EvalWorkshopConnectionRow(
                id=f"connection_link_{index:03d}",
                failure_id=link.source_id,
                eval_id=link.target_id if link.target_id.startswith("eval_") else None,
                replay_id=(
                    link.target_id
                    if link.link_type == "replay_eval_to_outcome"
                    else None
                ),
                status=link.link_type,
                summary=link.summary,
            )
        )
    for index, outcome in enumerate(replay_outcomes, start=1):
        rows.append(
            EvalWorkshopConnectionRow(
                id=f"connection_replay_{index:03d}",
                eval_id=outcome.generated_eval_id,
                replay_id=outcome.id,
                status="passed" if outcome.passed else "failed",
                summary=outcome.summary,
            )
        )
    return rows


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
    leaps_by_id = {leap.id: leap for leap in invalid_leaps}
    ordered_pairs = [
        (leaps_by_id.get(generated_eval.source_failure_id), generated_eval)
        for generated_eval in generated_evals
        if generated_eval.source_failure_id in leaps_by_id
    ]
    if not ordered_pairs:
        ordered_pairs = list(zip(invalid_leaps, generated_evals, strict=False))
    for index, (leap, generated_eval) in enumerate(ordered_pairs, start=1):
        if leap is None:
            continue
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


def _replay_eval_to_outcome_links(
    replay_outcomes: list[ReplayOutcomeRecord],
) -> list[EvalWorkshopLink]:
    links: list[EvalWorkshopLink] = []
    for index, outcome in enumerate(replay_outcomes, start=1):
        if not outcome.generated_eval_id:
            continue
        links.append(
            EvalWorkshopLink(
                id=f"link_replay_eval_outcome_{index:03d}",
                link_type="replay_eval_to_outcome",
                source_id=outcome.generated_eval_id,
                target_id=outcome.id,
                summary=f"{outcome.generated_eval_id} replay produced {outcome.id}.",
                affected_assumption_ids=[outcome.assumption_id],
                source_ids=[],
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


def _agent_step_by_task_type(steps: list[AgentRunStep]) -> dict[str, AgentRunStep]:
    tool_by_task_type = {
        "parse_source": "execute_source_research_tasks_tool",
        "extract_evidence": "execute_source_research_tasks_tool",
        "cross_check": "cross_check_evidence_tool",
        "verify_decisive_test": "run_decisive_test_verifiers_tool",
    }
    steps_by_tool = {step.tool_name: step for step in steps}
    return {
        task_type: steps_by_tool[tool_name]
        for task_type, tool_name in tool_by_task_type.items()
        if tool_name in steps_by_tool
    }


def _duration_ms(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(value))
    except ValueError:
        return None
