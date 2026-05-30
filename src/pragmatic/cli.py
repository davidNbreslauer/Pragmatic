from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pragmatic.agents import (
    AgentsSDKCredentialsError,
    LiveAgentsSDKNotEnabled,
    ResearchManager,
)
from pragmatic.doctor import run_integration_doctor
from pragmatic.demo import demo_scenarios, run_demo_smoke
from pragmatic.eval_corpus import (
    compare_eval_baseline,
    compare_eval_snapshot_by_id,
    list_eval_snapshots,
    save_eval_snapshot,
    write_eval_baseline,
)
from pragmatic.eval_suite import export_generated_eval_cases, run_eval_suite
from pragmatic.live_harness import run_live_harness_sync
from pragmatic.research_loop import DEFAULT_THESIS, run_research_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pragmatic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser(
        "eval-suite",
        help="Run deterministic regression gates for the current research loop.",
    )
    eval_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    eval_parser.add_argument("--max-iterations", type=int, default=1)
    eval_parser.add_argument("--output")
    eval_parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit non-zero when any regression gate fails.",
    )

    live_parser = subparsers.add_parser(
        "run-live-sdk",
        help="Run Pragmatic through the live OpenAI Agents SDK path.",
    )
    live_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    live_parser.add_argument("--max-iterations", type=int, default=1)
    live_parser.add_argument("--model")
    live_parser.add_argument("--output")
    live_parser.add_argument(
        "--execution-backend",
        choices=["local", "modal"],
        default="local",
    )
    live_parser.add_argument(
        "--observability",
        choices=["local", "raindrop", "off"],
        default="off",
    )
    live_parser.add_argument(
        "--source-mode",
        choices=["prepared", "web"],
        default="prepared",
    )
    live_parser.add_argument(
        "--allow-live-web-search",
        action="store_true",
        help="Allow live web source acquisition when --source-mode web is selected.",
    )
    live_parser.add_argument("--web-search-model")
    live_parser.add_argument("--max-web-sources", type=int, default=8)
    live_parser.add_argument(
        "--allow-live-sdk",
        action="store_true",
        help="Required confirmation that this command may make a live OpenAI API call.",
    )

    harness_parser = subparsers.add_parser(
        "live-run-harness",
        help="Validate or run the live OpenAI Agents SDK path with guardrails.",
    )
    harness_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    harness_parser.add_argument("--max-iterations", type=int, default=1)
    harness_parser.add_argument("--model")
    harness_parser.add_argument("--output")
    harness_parser.add_argument("--output-dir")
    harness_parser.add_argument("--max-turns", type=int, default=4)
    harness_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    harness_parser.add_argument(
        "--execution-backend",
        choices=["local", "modal"],
        default="local",
    )
    harness_parser.add_argument(
        "--observability",
        choices=["local", "raindrop", "off"],
        default="local",
    )
    harness_parser.add_argument(
        "--source-mode",
        choices=["prepared", "web"],
        default="prepared",
    )
    harness_parser.add_argument(
        "--allow-live-web-search",
        action="store_true",
        help="Allow live web source acquisition when --source-mode web is selected.",
    )
    harness_parser.add_argument("--web-search-model")
    harness_parser.add_argument("--max-web-sources", type=int, default=8)
    harness_parser.add_argument(
        "--live",
        action="store_true",
        help="Actually make a live OpenAI Agents SDK call. Default is dry run.",
    )
    harness_parser.add_argument(
        "--allow-live-sdk",
        action="store_true",
        help="Required confirmation when --live is used.",
    )
    harness_parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="Do not write the harness result under .pragmatic/live_runs.",
    )
    harness_parser.add_argument(
        "--require-demo-proof",
        action="store_true",
        help="Fail unless the live result validates SDK output, requested Modal execution, and Workshop observability.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check OpenAI Agents SDK, Modal, and Raindrop Workshop integration readiness.",
    )
    doctor_parser.add_argument(
        "--run-openai-live",
        action="store_true",
        help="Run a live OpenAI API credential check in addition to SDK import checks.",
    )
    doctor_parser.add_argument(
        "--run-modal-remote",
        action="store_true",
        help="Run a tiny live Modal remote task instead of only checking local configuration.",
    )
    doctor_parser.add_argument("--output")
    doctor_parser.add_argument("--output-dir")
    doctor_parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="Exit non-zero when any integration is unavailable, skipped, or failed.",
    )

    scenarios_parser = subparsers.add_parser(
        "demo-scenarios",
        help="List curated hackathon demo scenarios.",
    )
    scenarios_parser.add_argument("--output")

    smoke_parser = subparsers.add_parser(
        "demo-smoke",
        help="Run the demo readiness harness and write replayable artifacts.",
    )
    smoke_parser.add_argument("--output-dir")
    smoke_parser.add_argument("--run-openai-live", action="store_true")
    smoke_parser.add_argument("--run-modal-remote", action="store_true")
    smoke_parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit non-zero if any demo smoke check fails.",
    )

    snapshot_parser = subparsers.add_parser(
        "save-eval-snapshot",
        help="Save the current passing regression suite as a known-good snapshot.",
    )
    snapshot_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    snapshot_parser.add_argument("--max-iterations", type=int, default=1)
    snapshot_parser.add_argument("--corpus-dir")
    snapshot_parser.add_argument("--snapshot-id")
    snapshot_parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Persist the snapshot even when the regression suite fails.",
    )

    compare_parser = subparsers.add_parser(
        "compare-eval-snapshot",
        help="Compare current regression behavior against a saved eval snapshot.",
    )
    compare_parser.add_argument("snapshot_id")
    compare_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    compare_parser.add_argument("--max-iterations", type=int, default=1)
    compare_parser.add_argument("--corpus-dir")
    compare_parser.add_argument("--output")
    compare_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero only when a passing baseline gate now fails.",
    )

    list_parser = subparsers.add_parser(
        "list-eval-snapshots",
        help="List saved eval-corpus snapshots.",
    )
    list_parser.add_argument("--corpus-dir")

    export_baseline_parser = subparsers.add_parser(
        "export-eval-baseline",
        help="Export a deterministic, repo-trackable eval baseline.",
    )
    export_baseline_parser.add_argument("output")
    export_baseline_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    export_baseline_parser.add_argument("--max-iterations", type=int, default=1)
    export_baseline_parser.add_argument("--snapshot-id", default="default_v1")
    export_baseline_parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Export the baseline even when the regression suite fails.",
    )

    check_baseline_parser = subparsers.add_parser(
        "check-eval-baseline",
        help="Compare current eval behavior against a committed baseline file.",
    )
    check_baseline_parser.add_argument("baseline")
    check_baseline_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    check_baseline_parser.add_argument("--max-iterations", type=int, default=1)
    check_baseline_parser.add_argument("--output")
    check_baseline_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when a passing baseline gate now fails.",
    )
    check_baseline_parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Exit non-zero on any gate or generated-eval fixture drift.",
    )

    export_parser = subparsers.add_parser(
        "export-generated-evals",
        help="Run the deterministic loop and export generated eval fixtures.",
    )
    export_parser.add_argument("output")
    export_parser.add_argument("--thesis", default=DEFAULT_THESIS)
    export_parser.add_argument("--max-iterations", type=int, default=1)

    args = parser.parse_args(argv)

    if args.command == "eval-suite":
        result = run_eval_suite(
            args.thesis,
            max_iterations=args.max_iterations,
        )
        payload = json.dumps(result.model_dump(), indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 1 if args.fail_on_fail and result.status == "fail" else 0

    if args.command == "run-live-sdk":
        manager = ResearchManager(model=args.model)
        try:
            state = manager.run_live_sync(
                args.thesis,
                max_iterations=args.max_iterations,
                execution_backend=args.execution_backend,
                observability_mode=args.observability,
                source_mode=args.source_mode,
                allow_live_web_search=args.allow_live_web_search,
                web_search_model=args.web_search_model,
                max_web_sources=args.max_web_sources,
                allow_live_sdk=args.allow_live_sdk,
            )
        except (LiveAgentsSDKNotEnabled, AgentsSDKCredentialsError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = state.model_dump_json(indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "live-run-harness":
        result = run_live_harness_sync(
            args.thesis,
            model=args.model,
            mode="live" if args.live else "dry_run",
            allow_live_sdk=args.allow_live_sdk,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            max_iterations=args.max_iterations,
            execution_backend=args.execution_backend,
            source_mode=args.source_mode,
            allow_live_web_search=args.allow_live_web_search,
            web_search_model=args.web_search_model,
            max_web_sources=args.max_web_sources,
            observability_mode=args.observability,
            output_dir=args.output_dir,
            write_artifact=not args.no_artifact,
            require_demo_proof=args.require_demo_proof,
        )
        payload = result.model_dump_json(indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if result.status in {"blocked", "failed", "timed_out"}:
            return 2
        return 0

    if args.command == "doctor":
        result = run_integration_doctor(
            run_openai_live=args.run_openai_live,
            run_modal_remote=args.run_modal_remote,
            output_dir=args.output_dir,
        )
        payload = result.model_dump_json(indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if result.status == "failed":
            return 2
        if args.fail_on_degraded and result.status == "degraded":
            return 1
        return 0

    if args.command == "demo-scenarios":
        payload = json.dumps([scenario.model_dump() for scenario in demo_scenarios()], indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "demo-smoke":
        result = run_demo_smoke(
            output_dir=args.output_dir,
            run_openai_live=args.run_openai_live,
            run_modal_remote=args.run_modal_remote,
        )
        print(result.model_dump_json(indent=2))
        return 1 if args.fail_on_fail and result.status == "fail" else 0

    if args.command == "save-eval-snapshot":
        summary = save_eval_snapshot(
            args.thesis,
            corpus_dir=args.corpus_dir,
            max_iterations=args.max_iterations,
            snapshot_id=args.snapshot_id,
            require_pass=not args.allow_fail,
        )
        print(json.dumps(summary.model_dump(), indent=2))
        return 0

    if args.command == "compare-eval-snapshot":
        comparison = compare_eval_snapshot_by_id(
            args.snapshot_id,
            corpus_dir=args.corpus_dir,
            thesis_text=args.thesis,
            max_iterations=args.max_iterations,
        )
        payload = json.dumps(comparison.model_dump(), indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 1 if args.fail_on_regression and comparison.status == "regression" else 0

    if args.command == "list-eval-snapshots":
        summaries = list_eval_snapshots(corpus_dir=args.corpus_dir)
        print(json.dumps([summary.model_dump() for summary in summaries], indent=2))
        return 0

    if args.command == "export-eval-baseline":
        path = write_eval_baseline(
            args.output,
            args.thesis,
            max_iterations=args.max_iterations,
            snapshot_id=args.snapshot_id,
            require_pass=not args.allow_fail,
        )
        print(path)
        return 0

    if args.command == "check-eval-baseline":
        comparison = compare_eval_baseline(
            args.baseline,
            thesis_text=args.thesis,
            max_iterations=args.max_iterations,
        )
        payload = json.dumps(comparison.model_dump(), indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if args.fail_on_change and comparison.status != "match":
            return 1
        if args.fail_on_regression and comparison.status == "regression":
            return 1
        return 0

    if args.command == "export-generated-evals":
        state = run_research_loop(
            args.thesis,
            max_iterations=args.max_iterations,
            observability_mode="off",
        )
        path = export_generated_eval_cases(state, args.output)
        print(path)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
