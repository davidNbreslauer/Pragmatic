from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from thesisgraph.schemas import (
    BeliefDelta,
    ResearchState,
    RunComparison,
    RunSummary,
)


DEFAULT_RUN_DIR = Path(".thesisgraph") / "runs"
INDEX_FILENAME = "index.json"


def save_run(
    state: ResearchState,
    *,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RunSummary:
    directory = _run_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(UTC).isoformat()
    resolved_run_id = run_id or _make_run_id(created_at, state.thesis.text)
    path = directory / f"{resolved_run_id}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    summary = summarize_run(
        state,
        run_id=resolved_run_id,
        created_at=created_at,
        path=path,
    )
    _write_index(directory, _upsert_summary(_read_index(directory), summary))
    return summary


def load_run(run_id: str, *, run_dir: str | Path | None = None) -> ResearchState:
    directory = _run_dir(run_dir)
    path = directory / f"{run_id}.json"
    return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(*, run_dir: str | Path | None = None) -> list[RunSummary]:
    directory = _run_dir(run_dir)
    summaries = _read_index(directory)
    existing = []
    for summary in summaries:
        if Path(summary.path).exists():
            existing.append(summary)
    return sorted(existing, key=lambda summary: summary.created_at, reverse=True)


def compare_runs(
    baseline: ResearchState,
    current: ResearchState,
    *,
    baseline_run_id: str = "baseline",
    current_run_id: str = "current",
) -> RunComparison:
    baseline_assumptions = {assumption.id: assumption for assumption in baseline.assumptions}
    current_assumptions = {assumption.id: assumption for assumption in current.assumptions}
    deltas: list[BeliefDelta] = []

    for assumption_id in sorted(current_assumptions):
        current_assumption = current_assumptions[assumption_id]
        baseline_assumption = baseline_assumptions.get(assumption_id)
        if baseline_assumption is None:
            continue
        delta = round(current_assumption.confidence - baseline_assumption.confidence, 2)
        deltas.append(
            BeliefDelta(
                assumption_id=assumption_id,
                assumption_text=current_assumption.text,
                previous_support=baseline_assumption.support_level,
                current_support=current_assumption.support_level,
                previous_confidence=baseline_assumption.confidence,
                current_confidence=current_assumption.confidence,
                delta=delta,
                previous_update=baseline_assumption.latest_update,
                current_update=current_assumption.latest_update,
            )
        )

    changed = [delta for delta in deltas if delta.delta != 0]
    summary = (
        f"{len(changed)} of {len(deltas)} assumptions changed confidence "
        f"between {baseline_run_id} and {current_run_id}."
    )
    return RunComparison(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
        deltas=deltas,
        summary=summary,
    )


def summarize_run(
    state: ResearchState,
    *,
    run_id: str,
    created_at: str,
    path: str | Path,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        created_at=created_at,
        thesis_text=state.thesis.text,
        path=str(path),
        trace_id=state.observability.trace_id if state.observability else None,
        assumption_count=len(state.assumptions),
        evidence_item_count=len(state.evidence_items),
        evidence_conflict_count=len(state.evidence_conflicts),
        invalid_leap_count=len(state.invalid_leaps),
        verifier_result_count=len(state.verifier_results),
        generated_eval_count=len(state.generated_evals),
    )


def _run_dir(run_dir: str | Path | None) -> Path:
    return Path(run_dir) if run_dir is not None else DEFAULT_RUN_DIR


def _make_run_id(created_at: str, thesis_text: str) -> str:
    timestamp = (
        created_at.replace("+00:00", "Z")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )
    slug = _slugify(thesis_text)[:36] or "thesis"
    return f"run_{timestamp}_{slug}_{uuid.uuid4().hex[:8]}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "thesis"


def _read_index(directory: Path) -> list[RunSummary]:
    path = directory / INDEX_FILENAME
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RunSummary.model_validate(item) for item in raw]


def _write_index(directory: Path, summaries: list[RunSummary]) -> None:
    path = directory / INDEX_FILENAME
    path.write_text(
        json.dumps([summary.model_dump() for summary in summaries], indent=2),
        encoding="utf-8",
    )


def _upsert_summary(
    summaries: list[RunSummary],
    summary: RunSummary,
) -> list[RunSummary]:
    retained = [item for item in summaries if item.run_id != summary.run_id]
    retained.append(summary)
    return sorted(retained, key=lambda item: item.created_at, reverse=True)
