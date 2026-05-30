from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pragmatic.replay import BENCHMARK_SOURCE_IDS, run_replay_demo
from pragmatic.research_loop import DEFAULT_THESIS, run_research_loop
from pragmatic.schemas import (
    GeneratedEval,
    GeneratedEvalFixture,
    RegressionEvalCase,
    RegressionEvalCaseResult,
    RegressionEvalSuiteResult,
    ReplayResult,
    ResearchState,
    Source,
)


COMPANY_CLAIM_SOURCE_TYPE = "company_claim"


def build_default_eval_cases() -> list[RegressionEvalCase]:
    return [
        RegressionEvalCase(
            id="eval_benchmark_proxy_boundary",
            name="Benchmark evidence remains proxy",
            kind="benchmark_proxy_boundary",
            description="Benchmark sources must not be treated as direct discovery evidence.",
            expected_behavior=(
                "Evidence from benchmark sources is classified as proxy and the benchmark leap is preserved."
            ),
        ),
        RegressionEvalCase(
            id="eval_a6_requires_direct_validation",
            name="A6 requires direct prospective validation",
            kind="a6_requires_direct_validation",
            description="Prospective validation should remain unsupported without independent direct evidence.",
            expected_behavior="A6 support is unsupported with low confidence when no direct validation exists.",
        ),
        RegressionEvalCase(
            id="eval_company_claim_anecdotal",
            name="Company claims remain anecdotal",
            kind="company_claim_anecdotal",
            description="Product/company claims cannot become validated outcomes without independent evidence.",
            expected_behavior="Company-claim evidence is anecdotal and generates a claim-to-validation leap.",
        ),
        RegressionEvalCase(
            id="eval_conflict_workshop_links",
            name="Conflict/failure links are present",
            kind="conflict_workshop_links",
            description="The eval workshop must link conflicts, invalid leaps, verifier failures, and evals.",
            expected_behavior="Workshop links include invalid-leap, conflict, and verifier failure edges.",
        ),
        RegressionEvalCase(
            id="eval_replay_confidence_not_increased",
            name="Replay does not increase overcredited confidence",
            kind="replay_confidence_not_increased",
            description="Replay after applying the generated eval should lower or preserve confidence.",
            expected_behavior="Replay outcomes pass and confidence does not increase after eval application.",
        ),
    ]


def run_eval_suite(
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
) -> RegressionEvalSuiteResult:
    state = run_research_loop(
        thesis_text,
        max_iterations=max_iterations,
        observability_mode="off",
    )
    replay = run_replay_demo(
        thesis_text,
        max_iterations=max_iterations,
        observability_mode="off",
    )
    results = evaluate_regression_cases(state, replay)
    return _suite_result(results)


def evaluate_regression_cases(
    state: ResearchState,
    replay: ReplayResult | None = None,
) -> list[RegressionEvalCaseResult]:
    cases = {case.kind: case for case in build_default_eval_cases()}
    return [
        check_benchmark_proxy_boundary(state, cases["benchmark_proxy_boundary"]),
        check_a6_requires_direct_validation(state, cases["a6_requires_direct_validation"]),
        check_company_claim_anecdotal(state, cases["company_claim_anecdotal"]),
        check_conflict_workshop_links(state, cases["conflict_workshop_links"]),
        check_replay_confidence_not_increased(replay, cases["replay_confidence_not_increased"]),
    ]


def check_benchmark_proxy_boundary(
    state: ResearchState,
    case: RegressionEvalCase | None = None,
) -> RegressionEvalCaseResult:
    resolved_case = case or build_default_eval_cases()[0]
    benchmark_items = [
        item for item in state.evidence_items if item.source_id in BENCHMARK_SOURCE_IDS
    ]
    non_proxy_items = [
        item for item in benchmark_items if item.evidence_type != "proxy"
    ]
    benchmark_leap_present = any(
        leap.id == "leap_benchmark_to_discovery" for leap in state.invalid_leaps
    )
    passed = bool(benchmark_items) and not non_proxy_items and benchmark_leap_present
    return _case_result(
        resolved_case,
        passed=passed,
        pass_message="Benchmark evidence is bounded as proxy evidence.",
        fail_message="Benchmark evidence was overcredited or the benchmark leap disappeared.",
        details={
            "benchmark_items": str(len(benchmark_items)),
            "non_proxy_items": ",".join(item.id for item in non_proxy_items),
            "benchmark_leap_present": str(benchmark_leap_present),
        },
    )


def check_a6_requires_direct_validation(
    state: ResearchState,
    case: RegressionEvalCase | None = None,
) -> RegressionEvalCaseResult:
    resolved_case = case or build_default_eval_cases()[1]
    source_by_id = {source.id: source for source in state.sources}
    direct_independent_items = [
        item
        for item in state.evidence_items
        if "A6" in item.assumption_ids
        and item.evidence_type == "direct"
        and _source_is_independent_direct(source_by_id.get(item.source_id))
    ]
    a6 = next((assumption for assumption in state.assumptions if assumption.id == "A6"), None)
    passed = (
        a6 is not None
        and (
            bool(direct_independent_items)
            or (a6.support_level == "unsupported" and a6.confidence <= 0.2)
        )
    )
    return _case_result(
        resolved_case,
        passed=passed,
        pass_message="A6 remains appropriately low without direct validation.",
        fail_message="A6 gained too much support without independent direct validation.",
        details={
            "direct_independent_items": ",".join(item.id for item in direct_independent_items),
            "a6_support": a6.support_level if a6 else "missing",
            "a6_confidence": str(a6.confidence if a6 else "missing"),
        },
    )


def check_company_claim_anecdotal(
    state: ResearchState,
    case: RegressionEvalCase | None = None,
) -> RegressionEvalCaseResult:
    resolved_case = case or build_default_eval_cases()[2]
    source_by_id = {source.id: source for source in state.sources}
    company_items = [
        item
        for item in state.evidence_items
        if source_by_id.get(item.source_id) is not None
        and source_by_id[item.source_id].source_type == COMPANY_CLAIM_SOURCE_TYPE
    ]
    non_anecdotal_items = [
        item for item in company_items if item.evidence_type != "anecdotal"
    ]
    company_leap_present = any(
        leap.id == "leap_claim_to_validated_outcome" for leap in state.invalid_leaps
    )
    passed = bool(company_items) and not non_anecdotal_items and company_leap_present
    return _case_result(
        resolved_case,
        passed=passed,
        pass_message="Company claims remain anecdotal and bounded.",
        fail_message="Company claims were overcredited or their invalid leap disappeared.",
        details={
            "company_items": str(len(company_items)),
            "non_anecdotal_items": ",".join(item.id for item in non_anecdotal_items),
            "company_leap_present": str(company_leap_present),
        },
    )


def check_conflict_workshop_links(
    state: ResearchState,
    case: RegressionEvalCase | None = None,
) -> RegressionEvalCaseResult:
    resolved_case = case or build_default_eval_cases()[3]
    link_types = set()
    if state.eval_workshop is not None:
        link_types = {link.link_type for link in state.eval_workshop.failure_eval_links}
    required_link_types = {
        "invalid_leap_to_eval",
        "evidence_conflict_to_invalid_leap",
        "verifier_failure_to_eval",
    }
    missing = sorted(required_link_types - link_types)
    passed = bool(state.evidence_conflicts) and state.eval_workshop is not None and not missing
    return _case_result(
        resolved_case,
        passed=passed,
        pass_message="Conflict and failure links are present in the eval workshop.",
        fail_message="Eval workshop is missing required failure links.",
        details={
            "evidence_conflicts": str(len(state.evidence_conflicts)),
            "link_types": ",".join(sorted(link_types)),
            "missing": ",".join(missing),
        },
    )


def check_replay_confidence_not_increased(
    replay: ReplayResult | None,
    case: RegressionEvalCase | None = None,
) -> RegressionEvalCaseResult:
    resolved_case = case or build_default_eval_cases()[4]
    if replay is None:
        return _case_result(
            resolved_case,
            passed=False,
            pass_message="Replay outcomes passed.",
            fail_message="Replay result is missing.",
            details={},
        )

    increased = [
        comparison.assumption_id
        for comparison in replay.comparisons
        if comparison.after_confidence > comparison.before_confidence
    ]
    replay_outcomes = (
        replay.eval_workshop.replay_outcomes
        if replay.eval_workshop is not None
        else []
    )
    failed_outcomes = [outcome.id for outcome in replay_outcomes if not outcome.passed]
    passed = bool(replay.comparisons) and not increased and not failed_outcomes
    return _case_result(
        resolved_case,
        passed=passed,
        pass_message="Replay lowers or preserves confidence after eval application.",
        fail_message="Replay increased confidence or produced a failing replay outcome.",
        details={
            "increased_assumptions": ",".join(increased),
            "failed_outcomes": ",".join(failed_outcomes),
            "replay_outcomes": str(len(replay_outcomes)),
        },
    )


def export_generated_eval_cases(
    state: ResearchState,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        generated_eval_to_fixture(generated_eval).model_dump()
        for generated_eval in state.generated_evals
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def generated_eval_to_fixture(generated_eval: GeneratedEval) -> GeneratedEvalFixture:
    return GeneratedEvalFixture(
        id=generated_eval.id,
        failure_observed=generated_eval.failure_observed,
        root_cause=generated_eval.root_cause,
        eval_rule=generated_eval.eval_rule,
        expected_behavior=generated_eval.expected_behavior,
    )


def _suite_result(
    results: list[RegressionEvalCaseResult],
) -> RegressionEvalSuiteResult:
    passed = sum(result.status == "pass" for result in results)
    failed = len(results) - passed
    status = "pass" if failed == 0 else "fail"
    return RegressionEvalSuiteResult(
        id=f"eval_suite_{uuid.uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        status=status,
        passed=passed,
        failed=failed,
        results=results,
        summary=f"{passed} passed, {failed} failed.",
    )


def _case_result(
    case: RegressionEvalCase,
    *,
    passed: bool,
    pass_message: str,
    fail_message: str,
    details: dict[str, str],
) -> RegressionEvalCaseResult:
    return RegressionEvalCaseResult(
        case=case,
        status="pass" if passed else "fail",
        message=pass_message if passed else fail_message,
        details=details,
    )


def _source_is_independent_direct(source: Source | None) -> bool:
    if source is None:
        return False
    return source.source_type not in {"benchmark", "company_claim"}

