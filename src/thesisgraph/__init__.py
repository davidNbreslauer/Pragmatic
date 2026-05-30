"""ThesisGraph deterministic MVP scaffold."""

from thesisgraph.agents import ResearchManager
from thesisgraph.eval_corpus import (
    compare_eval_baseline,
    compare_eval_snapshot,
    compare_eval_snapshot_by_id,
    create_canonical_eval_snapshot,
    create_eval_snapshot,
    list_eval_snapshots,
    load_eval_baseline,
    load_eval_snapshot,
    save_eval_snapshot,
    write_eval_baseline,
)
from thesisgraph.eval_suite import (
    evaluate_regression_cases,
    export_generated_eval_cases,
    run_eval_suite,
)
from thesisgraph.persistence import compare_runs, list_runs, load_run, save_run
from thesisgraph.replay import run_replay_demo
from thesisgraph.research_loop import DEFAULT_THESIS, run_research_loop
from thesisgraph.schemas import (
    EvalSnapshotComparison,
    EvalSnapshotSummary,
    RegressionEvalSuiteResult,
    ReplayResult,
    ResearchState,
    RunComparison,
    RunSummary,
)

__all__ = [
    "DEFAULT_THESIS",
    "EvalSnapshotComparison",
    "EvalSnapshotSummary",
    "RegressionEvalSuiteResult",
    "ReplayResult",
    "ResearchManager",
    "ResearchState",
    "RunComparison",
    "RunSummary",
    "compare_eval_baseline",
    "compare_eval_snapshot",
    "compare_eval_snapshot_by_id",
    "create_canonical_eval_snapshot",
    "compare_runs",
    "create_eval_snapshot",
    "evaluate_regression_cases",
    "export_generated_eval_cases",
    "list_eval_snapshots",
    "list_runs",
    "load_eval_baseline",
    "load_eval_snapshot",
    "load_run",
    "run_eval_suite",
    "run_replay_demo",
    "run_research_loop",
    "save_eval_snapshot",
    "save_run",
    "write_eval_baseline",
]
