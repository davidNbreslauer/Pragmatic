from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from thesisgraph.eval_suite import generated_eval_to_fixture, run_eval_suite
from thesisgraph.research_loop import DEFAULT_THESIS, run_research_loop
from thesisgraph.schemas import (
    EvalCaseDelta,
    EvalCorpusSnapshot,
    EvalSnapshotComparison,
    EvalSnapshotSummary,
    GeneratedEvalFixture,
    GeneratedEvalFixtureDelta,
    RegressionEvalCaseResult,
)


DEFAULT_EVAL_CORPUS_DIR = Path(".thesisgraph") / "eval_corpus"
INDEX_FILENAME = "index.json"
DEFAULT_BASELINE_SNAPSHOT_ID = "default_v1"
CANONICAL_BASELINE_CREATED_AT = "2026-05-30T00:00:00+00:00"


def create_eval_snapshot(
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
    snapshot_id: str | None = None,
) -> EvalCorpusSnapshot:
    created_at = datetime.now(UTC).isoformat()
    resolved_snapshot_id = snapshot_id or _make_snapshot_id(created_at, thesis_text)
    suite_result = run_eval_suite(
        thesis_text,
        max_iterations=max_iterations,
    )
    state = run_research_loop(
        thesis_text,
        max_iterations=max_iterations,
        observability_mode="off",
    )
    return EvalCorpusSnapshot(
        snapshot_id=resolved_snapshot_id,
        created_at=created_at,
        thesis_text=thesis_text,
        suite_result=suite_result,
        generated_eval_fixtures=[
            generated_eval_to_fixture(generated_eval)
            for generated_eval in state.generated_evals
        ],
    )


def save_eval_snapshot(
    thesis_text: str = DEFAULT_THESIS,
    *,
    corpus_dir: str | Path | None = None,
    max_iterations: int = 1,
    snapshot_id: str | None = None,
    require_pass: bool = True,
) -> EvalSnapshotSummary:
    directory = _corpus_dir(corpus_dir)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = create_eval_snapshot(
        thesis_text,
        max_iterations=max_iterations,
        snapshot_id=snapshot_id,
    )
    if require_pass and snapshot.suite_result.status != "pass":
        raise ValueError("Refusing to save a known-good eval snapshot because the suite failed.")

    path = directory / f"{snapshot.snapshot_id}.json"
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    summary = summarize_eval_snapshot(snapshot, path=path)
    _write_index(directory, _upsert_summary(_read_index(directory), summary))
    return summary


def create_canonical_eval_snapshot(
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
    snapshot_id: str = DEFAULT_BASELINE_SNAPSHOT_ID,
    created_at: str = CANONICAL_BASELINE_CREATED_AT,
) -> EvalCorpusSnapshot:
    snapshot = create_eval_snapshot(
        thesis_text,
        max_iterations=max_iterations,
        snapshot_id=snapshot_id,
    )
    snapshot.created_at = created_at
    snapshot.suite_result.id = f"eval_suite_{snapshot_id}"
    snapshot.suite_result.created_at = created_at
    return snapshot


def write_eval_baseline(
    output_path: str | Path,
    thesis_text: str = DEFAULT_THESIS,
    *,
    max_iterations: int = 1,
    snapshot_id: str = DEFAULT_BASELINE_SNAPSHOT_ID,
    require_pass: bool = True,
) -> Path:
    snapshot = create_canonical_eval_snapshot(
        thesis_text,
        max_iterations=max_iterations,
        snapshot_id=snapshot_id,
    )
    if require_pass and snapshot.suite_result.status != "pass":
        raise ValueError("Refusing to export a baseline because the eval suite failed.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_eval_baseline(path: str | Path) -> EvalCorpusSnapshot:
    return EvalCorpusSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def compare_eval_baseline(
    baseline_path: str | Path,
    current: EvalCorpusSnapshot | None = None,
    *,
    thesis_text: str = DEFAULT_THESIS,
    max_iterations: int = 1,
) -> EvalSnapshotComparison:
    baseline = load_eval_baseline(baseline_path)
    return compare_eval_snapshot(
        baseline,
        current,
        thesis_text=thesis_text,
        max_iterations=max_iterations,
    )


def load_eval_snapshot(
    snapshot_id: str,
    *,
    corpus_dir: str | Path | None = None,
) -> EvalCorpusSnapshot:
    directory = _corpus_dir(corpus_dir)
    path = directory / f"{snapshot_id}.json"
    return EvalCorpusSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def list_eval_snapshots(
    *,
    corpus_dir: str | Path | None = None,
) -> list[EvalSnapshotSummary]:
    directory = _corpus_dir(corpus_dir)
    summaries = _read_index(directory)
    existing = [summary for summary in summaries if Path(summary.path).exists()]
    return sorted(existing, key=lambda summary: summary.created_at, reverse=True)


def compare_eval_snapshot(
    baseline: EvalCorpusSnapshot,
    current: EvalCorpusSnapshot | None = None,
    *,
    thesis_text: str = DEFAULT_THESIS,
    max_iterations: int = 1,
) -> EvalSnapshotComparison:
    current_snapshot = current or create_eval_snapshot(
        thesis_text,
        max_iterations=max_iterations,
        snapshot_id="current",
    )
    case_deltas = _compare_cases(
        baseline.suite_result.results,
        current_snapshot.suite_result.results,
    )
    fixture_deltas = _compare_fixtures(
        baseline.generated_eval_fixtures,
        current_snapshot.generated_eval_fixtures,
    )
    regressions = [
        delta for delta in case_deltas if delta.regression
    ]
    changed_cases = [
        delta for delta in case_deltas if delta.changed
    ]
    changed_fixtures = [
        delta for delta in fixture_deltas if delta.status != "same"
    ]
    if regressions:
        status = "regression"
    elif changed_cases or changed_fixtures:
        status = "changed"
    else:
        status = "match"
    summary = _comparison_summary(
        status=status,
        changed_cases=len(changed_cases),
        regressions=len(regressions),
        changed_fixtures=len(changed_fixtures),
    )
    return EvalSnapshotComparison(
        baseline_snapshot_id=baseline.snapshot_id,
        current_snapshot_id=current_snapshot.snapshot_id,
        status=status,
        case_deltas=case_deltas,
        fixture_deltas=fixture_deltas,
        summary=summary,
    )


def compare_eval_snapshot_by_id(
    snapshot_id: str,
    *,
    corpus_dir: str | Path | None = None,
    thesis_text: str = DEFAULT_THESIS,
    max_iterations: int = 1,
) -> EvalSnapshotComparison:
    baseline = load_eval_snapshot(snapshot_id, corpus_dir=corpus_dir)
    return compare_eval_snapshot(
        baseline,
        thesis_text=thesis_text,
        max_iterations=max_iterations,
    )


def summarize_eval_snapshot(
    snapshot: EvalCorpusSnapshot,
    *,
    path: str | Path,
) -> EvalSnapshotSummary:
    return EvalSnapshotSummary(
        snapshot_id=snapshot.snapshot_id,
        created_at=snapshot.created_at,
        thesis_text=snapshot.thesis_text,
        path=str(path),
        suite_status=snapshot.suite_result.status,
        passed=snapshot.suite_result.passed,
        failed=snapshot.suite_result.failed,
        eval_case_count=len(snapshot.suite_result.results),
        generated_eval_count=len(snapshot.generated_eval_fixtures),
    )


def _compare_cases(
    baseline_results: list[RegressionEvalCaseResult],
    current_results: list[RegressionEvalCaseResult],
) -> list[EvalCaseDelta]:
    baseline_by_kind = {result.case.kind: result for result in baseline_results}
    current_by_kind = {result.case.kind: result for result in current_results}
    deltas: list[EvalCaseDelta] = []
    for kind in sorted(set(baseline_by_kind) | set(current_by_kind)):
        baseline = baseline_by_kind.get(kind)
        current = current_by_kind.get(kind)
        baseline_status = baseline.status if baseline else None
        current_status = current.status if current else None
        baseline_message = baseline.message if baseline else None
        current_message = current.message if current else None
        changed = (
            baseline_status != current_status
            or baseline_message != current_message
        )
        regression = baseline_status == "pass" and current_status != "pass"
        deltas.append(
            EvalCaseDelta(
                kind=kind,
                case_name=(current or baseline).case.name,
                baseline_status=baseline_status,
                current_status=current_status,
                changed=changed,
                regression=regression,
                baseline_message=baseline_message,
                current_message=current_message,
            )
        )
    return deltas


def _compare_fixtures(
    baseline_fixtures: list[GeneratedEvalFixture],
    current_fixtures: list[GeneratedEvalFixture],
) -> list[GeneratedEvalFixtureDelta]:
    baseline_by_id = {fixture.id: fixture for fixture in baseline_fixtures}
    current_by_id = {fixture.id: fixture for fixture in current_fixtures}
    deltas: list[GeneratedEvalFixtureDelta] = []
    for eval_id in sorted(set(baseline_by_id) | set(current_by_id)):
        baseline = baseline_by_id.get(eval_id)
        current = current_by_id.get(eval_id)
        if baseline is None:
            status = "new"
        elif current is None:
            status = "missing"
        elif baseline.model_dump() == current.model_dump():
            status = "same"
        else:
            status = "changed"
        deltas.append(
            GeneratedEvalFixtureDelta(
                eval_id=eval_id,
                status=status,
                baseline_eval_rule=baseline.eval_rule if baseline else None,
                current_eval_rule=current.eval_rule if current else None,
                baseline_expected_behavior=(
                    baseline.expected_behavior if baseline else None
                ),
                current_expected_behavior=current.expected_behavior if current else None,
            )
        )
    return deltas


def _comparison_summary(
    *,
    status: str,
    changed_cases: int,
    regressions: int,
    changed_fixtures: int,
) -> str:
    if status == "match":
        return "Current eval behavior matches the saved snapshot."
    if status == "regression":
        return (
            f"{regressions} regression(s), {changed_cases} changed gate(s), "
            f"and {changed_fixtures} changed generated eval fixture(s)."
        )
    return (
        f"{changed_cases} changed gate(s) and "
        f"{changed_fixtures} changed generated eval fixture(s)."
    )


def _corpus_dir(corpus_dir: str | Path | None) -> Path:
    return Path(corpus_dir) if corpus_dir is not None else DEFAULT_EVAL_CORPUS_DIR


def _make_snapshot_id(created_at: str, thesis_text: str) -> str:
    timestamp = (
        created_at.replace("+00:00", "Z")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )
    slug = _slugify(thesis_text)[:36] or "thesis"
    return f"eval_{timestamp}_{slug}_{uuid.uuid4().hex[:8]}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "thesis"


def _read_index(directory: Path) -> list[EvalSnapshotSummary]:
    path = directory / INDEX_FILENAME
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalSnapshotSummary.model_validate(item) for item in raw]


def _write_index(directory: Path, summaries: list[EvalSnapshotSummary]) -> None:
    path = directory / INDEX_FILENAME
    path.write_text(
        json.dumps([summary.model_dump() for summary in summaries], indent=2),
        encoding="utf-8",
    )


def _upsert_summary(
    summaries: list[EvalSnapshotSummary],
    summary: EvalSnapshotSummary,
) -> list[EvalSnapshotSummary]:
    retained = [item for item in summaries if item.snapshot_id != summary.snapshot_id]
    retained.append(summary)
    return sorted(retained, key=lambda item: item.created_at, reverse=True)
