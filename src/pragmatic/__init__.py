"""Pragmatic deterministic MVP scaffold."""

from pragmatic.agents import ResearchManager
from pragmatic.demo import demo_scenario_by_id, demo_scenarios, run_demo_smoke
from pragmatic.doctor import run_integration_doctor
from pragmatic.eval_corpus import (
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
from pragmatic.eval_suite import (
    evaluate_regression_cases,
    export_generated_eval_cases,
    run_eval_suite,
)
from pragmatic.live_harness import run_live_harness, run_live_harness_sync
from pragmatic.persistence import compare_runs, list_runs, load_run, save_run
from pragmatic.replay import run_replay_demo
from pragmatic.research_loop import DEFAULT_THESIS, run_research_loop
from pragmatic.schemas import (
    EvalSnapshotComparison,
    EvalSnapshotSummary,
    IntegrationDoctorResult,
    DemoScenario,
    DemoSmokeResult,
    LiveRunResult,
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
    "IntegrationDoctorResult",
    "DemoScenario",
    "DemoSmokeResult",
    "LiveRunResult",
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
    "run_live_harness",
    "run_live_harness_sync",
    "run_eval_suite",
    "run_demo_smoke",
    "run_integration_doctor",
    "run_replay_demo",
    "run_research_loop",
    "save_eval_snapshot",
    "save_run",
    "write_eval_baseline",
    "demo_scenario_by_id",
    "demo_scenarios",
]
