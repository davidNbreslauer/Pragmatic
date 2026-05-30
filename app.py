from __future__ import annotations

import json
import os

import streamlit as st

from thesisgraph import (
    DEFAULT_THESIS,
    EvalSnapshotComparison,
    RegressionEvalSuiteResult,
    ResearchManager,
    ResearchState,
    compare_eval_snapshot_by_id,
    list_eval_snapshots,
    run_eval_suite,
    save_eval_snapshot,
)
from thesisgraph.agents import AgentsSDKCredentialsError, LiveAgentsSDKNotEnabled
from thesisgraph.persistence import compare_runs, list_runs, load_run, save_run
from thesisgraph.replay import BENCHMARK_SOURCE_IDS, run_replay_demo
from thesisgraph.schemas import ReplayResult, RunComparison


st.set_page_config(page_title="ThesisGraph", layout="wide")


def main() -> None:
    st.title("ThesisGraph")

    with st.sidebar:
        st.header("Run")
        thesis_text = st.text_area("Thesis", value=DEFAULT_THESIS, height=140)
        max_iterations = st.slider("Iterations", min_value=1, max_value=3, value=1)
        orchestration = st.selectbox(
            "Orchestration",
            options=["deterministic", "scripted_sdk", "live_sdk"],
            index=0,
        )
        live_sdk_enabled = st.checkbox("Enable live SDK calls", value=False)
        live_sdk_model = st.text_input("Live SDK model", value="")
        execution_backend = st.selectbox("Execution", options=["local", "modal"], index=0)
        observability_mode = st.selectbox(
            "Observability",
            options=["local", "raindrop", "off"],
            index=0,
        )
        replay_demo = st.checkbox("Replay demo", value=False)
        run_clicked = st.button("Run ThesisGraph", type="primary", width="stretch")

        st.header("History")
        saved_runs = list_runs()
        run_options = [""] + [run.run_id for run in saved_runs]
        run_labels = {"": "Select saved run"}
        run_labels.update({run.run_id: _run_label(run) for run in saved_runs})
        preferred_run_id = st.session_state.get("saved_run_id", "")
        selected_run_index = (
            run_options.index(preferred_run_id)
            if preferred_run_id in run_options
            else 0
        )
        selected_run_id = st.selectbox(
            "Saved runs",
            options=run_options,
            index=selected_run_index,
            format_func=lambda run_id: run_labels[run_id],
        )
        load_clicked = st.button(
            "Load Run",
            width="stretch",
            disabled=not selected_run_id,
        )
        compare_clicked = st.button(
            "Compare To Current",
            width="stretch",
            disabled=not selected_run_id,
        )

        st.header("Evaluation")
        eval_suite_clicked = st.button("Run Eval Suite", width="stretch")
        saved_eval_snapshots = list_eval_snapshots()
        eval_snapshot_options = [""] + [
            snapshot.snapshot_id for snapshot in saved_eval_snapshots
        ]
        eval_snapshot_labels = {"": "Select eval snapshot"}
        eval_snapshot_labels.update(
            {
                snapshot.snapshot_id: _eval_snapshot_label(snapshot)
                for snapshot in saved_eval_snapshots
            }
        )
        selected_eval_snapshot_id = st.selectbox(
            "Eval snapshots",
            options=eval_snapshot_options,
            format_func=lambda snapshot_id: eval_snapshot_labels[snapshot_id],
        )
        save_eval_snapshot_clicked = st.button("Save Passing Snapshot", width="stretch")
        compare_eval_snapshot_clicked = st.button(
            "Compare Snapshot",
            width="stretch",
            disabled=not selected_eval_snapshot_id,
        )

    if run_clicked or "research_state_json" not in st.session_state:
        if replay_demo:
            replay = run_replay_demo(
                thesis_text,
                max_iterations=max_iterations,
                execution_backend=execution_backend,
                observability_mode=observability_mode,
            )
            st.session_state.replay_result_json = replay.model_dump_json()
            st.session_state.research_state_json = replay.replay_pass.model_dump_json()
        else:
            manager = ResearchManager(model=live_sdk_model or None)
            try:
                if orchestration == "live_sdk":
                    if live_sdk_enabled and not os.getenv("OPENAI_API_KEY"):
                        st.warning("Live SDK mode requires OPENAI_API_KEY.")
                    with st.spinner("Running live OpenAI Agents SDK orchestration..."):
                        state = manager.run_live_sync(
                            thesis_text,
                            max_iterations=max_iterations,
                            execution_backend=execution_backend,
                            observability_mode=observability_mode,
                            allow_live_sdk=live_sdk_enabled,
                        )
                elif orchestration == "scripted_sdk":
                    state = manager.run_sdk_orchestrated(
                        thesis_text,
                        max_iterations=max_iterations,
                        execution_backend=execution_backend,
                        observability_mode=observability_mode,
                    )
                else:
                    state = manager.run_deterministic(
                        thesis_text,
                        max_iterations=max_iterations,
                        execution_backend=execution_backend,
                        observability_mode=observability_mode,
                    )
                st.session_state.research_state_json = state.model_dump_json()
                st.session_state.pop("replay_result_json", None)
            except (LiveAgentsSDKNotEnabled, AgentsSDKCredentialsError) as exc:
                st.error(str(exc))
                st.session_state.pop("replay_result_json", None)
                if "research_state_json" not in st.session_state:
                    fallback = manager.run_deterministic(
                        thesis_text,
                        max_iterations=max_iterations,
                        execution_backend=execution_backend,
                        observability_mode="off",
                    )
                    st.session_state.research_state_json = fallback.model_dump_json()
        st.session_state.pop("run_comparison_json", None)
        if run_clicked:
            st.session_state.pop("eval_suite_result_json", None)
            st.session_state.pop("eval_snapshot_comparison_json", None)
    elif not replay_demo:
        st.session_state.pop("replay_result_json", None)

    if load_clicked and selected_run_id:
        loaded_state = load_run(selected_run_id)
        st.session_state.research_state_json = loaded_state.model_dump_json()
        st.session_state.loaded_run_id = selected_run_id
        st.session_state.pop("replay_result_json", None)
        st.session_state.pop("run_comparison_json", None)

    if eval_suite_clicked:
        with st.spinner("Running regression gates..."):
            eval_suite = run_eval_suite(
                thesis_text,
                max_iterations=max_iterations,
            )
        st.session_state.eval_suite_result_json = eval_suite.model_dump_json()

    if save_eval_snapshot_clicked:
        with st.spinner("Saving known-good eval snapshot..."):
            summary = save_eval_snapshot(
                thesis_text,
                max_iterations=max_iterations,
            )
        st.session_state.eval_snapshot_notice = f"Saved eval snapshot {summary.snapshot_id}"
        st.rerun()

    if compare_eval_snapshot_clicked and selected_eval_snapshot_id:
        with st.spinner("Comparing current behavior against snapshot..."):
            comparison = compare_eval_snapshot_by_id(
                selected_eval_snapshot_id,
                thesis_text=thesis_text,
                max_iterations=max_iterations,
            )
        st.session_state.eval_snapshot_comparison_json = comparison.model_dump_json()

    state = ResearchState.model_validate_json(st.session_state.research_state_json)

    if compare_clicked and selected_run_id:
        baseline_state = load_run(selected_run_id)
        current_run_id = st.session_state.get("loaded_run_id", "current")
        comparison = compare_runs(
            baseline_state,
            state,
            baseline_run_id=selected_run_id,
            current_run_id=current_run_id,
        )
        st.session_state.run_comparison_json = comparison.model_dump_json()

    render_summary(state)
    render_history_controls(state)
    render_retrieval_scores(state)
    if "run_comparison_json" in st.session_state:
        comparison = RunComparison.model_validate_json(st.session_state.run_comparison_json)
        render_run_comparison(comparison)
    if "eval_suite_result_json" in st.session_state:
        eval_suite = RegressionEvalSuiteResult.model_validate_json(
            st.session_state.eval_suite_result_json
        )
        render_eval_suite_result(eval_suite)
    if "eval_snapshot_comparison_json" in st.session_state:
        eval_comparison = EvalSnapshotComparison.model_validate_json(
            st.session_state.eval_snapshot_comparison_json
        )
        render_eval_snapshot_comparison(eval_comparison)
    if "replay_result_json" in st.session_state:
        replay = ReplayResult.model_validate_json(st.session_state.replay_result_json)
        render_replay(replay)
    render_agent_run(state)
    render_eval_workshop(state)
    render_assumptions(state)
    render_evidence_conflicts(state)
    render_invalid_leaps(state)
    render_evidence(state)
    render_belief_updates(state)
    render_decisive_tests(state)
    render_verifier_results(state)
    render_generated_evals(state)
    render_observability(state)
    render_trace(state)


def render_summary(state: ResearchState) -> None:
    st.caption(state.thesis.text)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Assumptions", len(state.assumptions))
    col2.metric("Evidence Items", len(state.evidence_items))
    col3.metric("Evidence Conflicts", len(state.evidence_conflicts))
    col4.metric("Invalid Leaps", len(state.invalid_leaps))
    col5.metric("Verifier Results", len(state.verifier_results))
    col6.metric("Generated Evals", len(state.generated_evals))


def render_history_controls(state: ResearchState) -> None:
    st.subheader("Run History")
    col1, col2 = st.columns([1, 3])
    if col1.button("Save Current Run", type="secondary"):
        summary = save_run(state)
        st.session_state.saved_run_id = summary.run_id
        st.session_state.loaded_run_id = summary.run_id
        st.session_state.save_notice = f"Saved run {summary.run_id}"
        st.rerun()

    loaded_run_id = st.session_state.get("loaded_run_id")
    saved_run_id = st.session_state.get("saved_run_id")
    status_parts = []
    if loaded_run_id:
        status_parts.append(f"Loaded: {loaded_run_id}")
    if saved_run_id:
        status_parts.append(f"Last saved: {saved_run_id}")
    col2.caption(" | ".join(status_parts) if status_parts else "No run selected yet.")
    if "save_notice" in st.session_state:
        st.success(st.session_state.pop("save_notice"))
    if "eval_snapshot_notice" in st.session_state:
        st.success(st.session_state.pop("eval_snapshot_notice"))


def render_run_comparison(comparison: RunComparison) -> None:
    st.subheader("Belief Delta")
    st.caption(comparison.summary)
    st.dataframe(
        [
            {
                "Assumption": delta.assumption_id,
                "Previous": delta.previous_support,
                "Current": delta.current_support,
                "Previous confidence": delta.previous_confidence,
                "Current confidence": delta.current_confidence,
                "Delta": delta.delta,
                "Current rationale": delta.current_update or "",
            }
            for delta in comparison.deltas
        ],
        width="stretch",
        hide_index=True,
    )


def render_retrieval_scores(state: ResearchState) -> None:
    st.subheader("Retrieval Scores")
    if not state.retrieval_scores:
        st.info("No retrieval scores recorded.")
        return

    source_titles = {source.id: source.title for source in state.sources}
    st.dataframe(
        [
            {
                "Question": score.question_id,
                "Source": source_titles.get(score.source_id, score.source_id),
                "Score": score.score,
                "Matched terms": ", ".join(score.matched_terms),
                "Rationale": score.rationale,
            }
            for score in sorted(
                state.retrieval_scores,
                key=lambda item: (item.question_id, -item.score, item.source_id),
            )
        ],
        width="stretch",
        hide_index=True,
    )


def render_eval_suite_result(result: RegressionEvalSuiteResult) -> None:
    st.subheader("Regression Gates")
    st.caption(result.summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", result.status)
    col2.metric("Passed", result.passed)
    col3.metric("Failed", result.failed)

    st.dataframe(
        [
            {
                "Case": case_result.case.name,
                "Status": case_result.status,
                "Kind": case_result.case.kind,
                "Message": case_result.message,
                "Expected": case_result.case.expected_behavior,
            }
            for case_result in result.results
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Gate details"):
        st.code(json.dumps(result.model_dump(), indent=2), language="json")


def render_eval_snapshot_comparison(comparison: EvalSnapshotComparison) -> None:
    st.subheader("Eval Snapshot Comparison")
    st.caption(comparison.summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", comparison.status)
    col2.metric("Gate changes", len([delta for delta in comparison.case_deltas if delta.changed]))
    col3.metric(
        "Fixture changes",
        len([delta for delta in comparison.fixture_deltas if delta.status != "same"]),
    )

    st.dataframe(
        [
            {
                "Case": delta.case_name,
                "Kind": delta.kind,
                "Baseline": delta.baseline_status or "missing",
                "Current": delta.current_status or "missing",
                "Changed": delta.changed,
                "Regression": delta.regression,
            }
            for delta in comparison.case_deltas
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Generated eval fixture comparison"):
        st.dataframe(
            [
                {
                    "Eval": delta.eval_id,
                    "Status": delta.status,
                    "Baseline rule": delta.baseline_eval_rule or "",
                    "Current rule": delta.current_eval_rule or "",
                }
                for delta in comparison.fixture_deltas
            ],
            width="stretch",
            hide_index=True,
        )


def render_replay(replay: ReplayResult) -> None:
    st.subheader("Failure -> Eval -> Replay")
    st.caption(replay.summary)

    first_a6 = _assumption_by_id(replay.first_pass, "A6")
    replay_a6 = _assumption_by_id(replay.replay_pass, "A6")
    col1, col2, col3 = st.columns(3)
    col1.metric("First-pass A6", first_a6.support_level, first_a6.confidence)
    col2.metric("Replay A6", replay_a6.support_level, replay_a6.confidence)
    col3.metric("Applied eval rules", len(replay.applied_eval_rules))

    if replay.applied_eval_rules:
        with st.container(border=True):
            st.markdown("**Generated rule applied on replay**")
            for rule in replay.applied_eval_rules:
                st.write(rule)

    st.dataframe(
        [
            {
                "Assumption": comparison.assumption_id,
                "First-pass support": comparison.before_support,
                "Replay support": comparison.after_support,
                "First-pass confidence": comparison.before_confidence,
                "Replay confidence": comparison.after_confidence,
                "Change": comparison.change_summary,
                "Rationale": comparison.rationale,
            }
            for comparison in replay.comparisons
        ],
        width="stretch",
        hide_index=True,
    )

    st.dataframe(
        _benchmark_replay_rows(replay),
        width="stretch",
        hide_index=True,
    )


def render_agent_run(state: ResearchState) -> None:
    if state.agent_run is None:
        return

    st.subheader("Agent Orchestration")
    record = state.agent_run
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", record.mode)
    col2.metric("Status", record.status)
    col3.metric("Validated", "yes" if record.final_output_validated else "no")
    if record.message:
        st.caption(record.message)
    if record.steps:
        st.dataframe(
            [
                {
                    "Step": step.id,
                    "Tool": step.tool_name,
                    "Status": step.status,
                    "Summary": step.summary,
                }
                for step in record.steps
            ],
            width="stretch",
            hide_index=True,
        )


def render_eval_workshop(state: ResearchState) -> None:
    if state.eval_workshop is None:
        return

    workshop = state.eval_workshop
    st.subheader("Eval Workshop")
    st.caption(workshop.summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Task spans", len(workshop.task_spans))
    col2.metric("Failure links", len(workshop.failure_eval_links))
    col3.metric("Replay outcomes", len(workshop.replay_outcomes))

    if workshop.failure_eval_links:
        st.dataframe(
            [
                {
                    "ID": link.id,
                    "Type": link.link_type,
                    "Failure": link.source_id,
                    "Target": link.target_id,
                    "Assumptions": ", ".join(link.affected_assumption_ids),
                    "Summary": link.summary,
                }
                for link in workshop.failure_eval_links
            ],
            width="stretch",
            hide_index=True,
        )

    if workshop.replay_outcomes:
        st.dataframe(
            [
                {
                    "ID": outcome.id,
                    "Assumption": outcome.assumption_id,
                    "Eval": outcome.generated_eval_id or "",
                    "Before": outcome.before_confidence,
                    "After": outcome.after_confidence,
                    "Passed": outcome.passed,
                    "Summary": outcome.summary,
                }
                for outcome in workshop.replay_outcomes
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Task spans"):
        st.dataframe(
            [
                {
                    "Span": span.id,
                    "Task": span.task_id,
                    "Type": span.task_type,
                    "Backend": span.backend,
                    "Status": span.status,
                    "Sources": ", ".join(span.source_ids),
                    "Evidence": span.evidence_item_count,
                    "Conflicts": span.evidence_conflict_count,
                    "Verifiers": span.verifier_result_count,
                }
                for span in workshop.task_spans
            ],
            width="stretch",
            hide_index=True,
        )


def render_assumptions(state: ResearchState) -> None:
    st.subheader("Assumption Graph")
    st.dataframe(
        [
            {
                "ID": assumption.id,
                "Assumption": assumption.text,
                "Support": assumption.support_level,
                "Confidence": assumption.confidence,
                "Latest update": assumption.latest_update or "",
            }
            for assumption in state.assumptions
        ],
        width="stretch",
        hide_index=True,
    )


def render_evidence_conflicts(state: ResearchState) -> None:
    st.subheader("Evidence Conflicts")
    if not state.evidence_conflicts:
        st.success("No cross-source evidence conflicts detected.")
        return

    st.dataframe(
        [
            {
                "ID": conflict.id,
                "Type": conflict.conflict_type,
                "Severity": conflict.severity,
                "Assumptions": ", ".join(conflict.affected_assumption_ids),
                "Sources": ", ".join(conflict.source_ids),
                "Summary": conflict.summary,
                "Action": conflict.suggested_action,
            }
            for conflict in state.evidence_conflicts
        ],
        width="stretch",
        hide_index=True,
    )


def render_invalid_leaps(state: ResearchState) -> None:
    st.subheader("Invalid Inference Leaps")
    if not state.invalid_leaps:
        st.success("No invalid inference leaps detected.")
        return

    for leap in state.invalid_leaps:
        with st.container(border=True):
            st.markdown(f"**{leap.leap}**")
            st.write(leap.why_invalid)
            st.caption(f"Affected assumptions: {', '.join(leap.affected_assumption_ids)}")
            st.caption(f"Follow-up: {leap.suggested_followup_question}")


def render_evidence(state: ResearchState) -> None:
    st.subheader("Evidence")
    source_titles = {source.id: source.title for source in state.sources}
    st.dataframe(
        [
            {
                "ID": item.id,
                "Source": source_titles.get(item.source_id, item.source_id),
                "Type": item.evidence_type,
                "Assumptions": ", ".join(item.assumption_ids),
                "Claim": item.claim_supported,
                "Limitation": item.limitation,
                "Confidence": item.confidence,
            }
            for item in state.evidence_items
        ],
        width="stretch",
        hide_index=True,
    )


def render_belief_updates(state: ResearchState) -> None:
    st.subheader("Belief Updates")
    st.dataframe(
        [
            {
                "Assumption": update.assumption_id,
                "Previous": update.previous_support,
                "New": update.new_support,
                "Previous confidence": update.previous_confidence,
                "New confidence": update.new_confidence,
                "Rationale": update.rationale,
            }
            for update in state.belief_updates
        ],
        width="stretch",
        hide_index=True,
    )


def render_decisive_tests(state: ResearchState) -> None:
    st.subheader("Decisive Tests")
    for test in state.decisive_tests:
        with st.container(border=True):
            st.markdown(f"**{test.test}**")
            st.write(test.why_decisive)
            st.caption(f"Would resolve: {', '.join(test.would_resolve)}")
            st.markdown("Success criteria")
            for criterion in test.success_criteria:
                st.write(f"- {criterion}")


def render_verifier_results(state: ResearchState) -> None:
    st.subheader("Verifier Results")
    if not state.verifier_results:
        st.info("No verifier results yet.")
        return

    st.dataframe(
        [
            {
                "ID": result.id,
                "Decisive test": result.decisive_test_id,
                "Status": result.status,
                "Confidence delta": result.confidence_delta,
                "Assumptions": ", ".join(result.affected_assumption_ids),
                "Rationale": result.rationale,
                "Failed criteria": " | ".join(result.failed_criteria),
            }
            for result in state.verifier_results
        ],
        width="stretch",
        hide_index=True,
    )


def render_generated_evals(state: ResearchState) -> None:
    st.subheader("Generated Evals")
    for generated_eval in state.generated_evals:
        with st.container(border=True):
            st.markdown(f"**{generated_eval.id}**")
            st.write(generated_eval.failure_observed)
            st.caption(f"Root cause: {generated_eval.root_cause}")
            st.code(
                "\n".join(
                    [
                        f"Rule: {generated_eval.eval_rule}",
                        f"Expected: {generated_eval.expected_behavior}",
                    ]
                ),
                language="text",
            )


def render_observability(state: ResearchState) -> None:
    st.subheader("Observability")
    if state.observability is None:
        st.info("No observability record for this run.")
        return

    record = state.observability
    col1, col2, col3 = st.columns(3)
    col1.metric("Backend", record.backend)
    col2.metric("Status", record.status)
    col3.metric("Trace ID", record.trace_id)

    if record.trace_path:
        st.caption(f"Local trace artifact: {record.trace_path}")
    if record.message:
        st.caption(record.message)
    if record.eval_artifact_ids:
        st.caption(f"Eval artifacts: {', '.join(record.eval_artifact_ids)}")
    if record.workshop_artifact_ids:
        st.caption(f"Workshop artifacts: {len(record.workshop_artifact_ids)} recorded")


def render_trace(state: ResearchState) -> None:
    st.subheader("Trace")
    st.dataframe(
        [
            {
                "Step": event.id,
                "Stage": event.stage,
                "Message": event.message,
            }
            for event in state.trace_events
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("ResearchState JSON"):
        st.code(json.dumps(state.model_dump(), indent=2), language="json")


def _assumption_by_id(state: ResearchState, assumption_id: str):
    for assumption in state.assumptions:
        if assumption.id == assumption_id:
            return assumption
    raise ValueError(f"Assumption not found: {assumption_id}")


def _benchmark_replay_rows(replay: ReplayResult) -> list[dict[str, str | float]]:
    first_items = {
        item.source_id: item
        for item in replay.first_pass.evidence_items
        if item.source_id in BENCHMARK_SOURCE_IDS
    }
    replay_items = {
        item.source_id: item
        for item in replay.replay_pass.evidence_items
        if item.source_id in BENCHMARK_SOURCE_IDS
    }

    rows: list[dict[str, str | float]] = []
    for source_id in sorted(BENCHMARK_SOURCE_IDS):
        first_item = first_items.get(source_id)
        replay_item = replay_items.get(source_id)
        if first_item is None or replay_item is None:
            continue
        rows.append(
            {
                "Source": source_id,
                "First-pass type": first_item.evidence_type,
                "Replay type": replay_item.evidence_type,
                "First-pass confidence": first_item.confidence,
                "Replay confidence": replay_item.confidence,
                "Replay limitation": replay_item.limitation,
            }
        )
    return rows


def _run_label(run) -> str:
    thesis = run.thesis_text
    if len(thesis) > 48:
        thesis = thesis[:45] + "..."
    return f"{run.created_at[:19]} | {thesis} | {run.run_id}"


def _eval_snapshot_label(snapshot) -> str:
    thesis = snapshot.thesis_text
    if len(thesis) > 42:
        thesis = thesis[:39] + "..."
    return (
        f"{snapshot.created_at[:19]} | {snapshot.suite_status} "
        f"{snapshot.passed}/{snapshot.eval_case_count} | {thesis}"
    )


if __name__ == "__main__":
    main()
