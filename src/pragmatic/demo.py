from __future__ import annotations

import json
from pathlib import Path

from pragmatic.doctor import run_integration_doctor
from pragmatic.corpus import SPIDER_SILK_CORPUS_PATH
from pragmatic.eval_suite import run_eval_suite
from pragmatic.replay import run_replay_demo
from pragmatic.research_loop import DEFAULT_THESIS, run_research_loop
from pragmatic.schemas import (
    DemoScenario,
    DemoSmokeCheck,
    DemoSmokeResult,
)


DEFAULT_DEMO_DIR = Path(".pragmatic") / "demo"


def demo_scenarios() -> list[DemoScenario]:
    return [
        DemoScenario(
            id="spider_silk_prepared",
            name="Spider Silk Prepared Corpus",
            thesis="Spider silk for bullet proof vests",
            orchestration="scripted_sdk",
            execution_backend="local",
            observability_backend="local",
            source_mode="prepared",
            corpus_path=str(SPIDER_SILK_CORPUS_PATH),
            allow_live_web_search=False,
            proves=[
                "Offline prepared-corpus source acquisition",
                "Tensile-toughness proxy boundary",
                "Invalid analogy to generated eval",
                "Standards-relevant ballistic decisive test",
            ],
            notes="Best first run for public playgrounds: no API calls, no Modal, and a clear proxy-evidence boundary.",
        ),
        DemoScenario(
            id="core_loop",
            name="Core Evidence Loop",
            thesis=DEFAULT_THESIS,
            orchestration="scripted_sdk",
            execution_backend="local",
            observability_backend="local",
            proves=[
                "OpenAI Agents SDK specialist-tool order",
                "Prepared corpus to typed evidence",
                "Invalid leap to generated eval",
            ],
            notes="Offline local run for understanding the full evidence loop.",
        ),
        DemoScenario(
            id="failure_replay",
            name="Failure To Eval Replay",
            thesis=DEFAULT_THESIS,
            orchestration="deterministic",
            execution_backend="local",
            observability_backend="local",
            replay_demo=True,
            proves=[
                "Proxy evidence downgrade",
                "Failure artifact to eval artifact",
                "Replay reduces over-confident belief",
            ],
            notes="Best closing run because it shows the Workshop story.",
        ),
        DemoScenario(
            id="modal_fanout",
            name="Modal Fan-Out",
            thesis=DEFAULT_THESIS,
            orchestration="scripted_sdk",
            execution_backend="modal",
            observability_backend="local",
            proves=[
                "Remote research execution layer",
                "Parse and extraction fan-out",
                "Cross-check/verifier task surfaces",
            ],
            notes="Use after the Integration Doctor shows Modal is live.",
        ),
        DemoScenario(
            id="live_guarded",
            name="Live SDK Guarded",
            thesis=DEFAULT_THESIS,
            orchestration="live_sdk",
            execution_backend="modal",
            observability_backend="local",
            live_dry_run=True,
            require_demo_proof=True,
            proves=[
                "Prepared-corpus/no-web-search guardrails",
                "Credential and timeout readiness",
                "Demo proof gate for live runs",
            ],
            notes="Starts dry-run by default; uncheck dry-run only when ready to spend a live call.",
        ),
        DemoScenario(
            id="live_full",
            name="Live Full System",
            thesis=DEFAULT_THESIS,
            orchestration="live_sdk",
            execution_backend="modal",
            observability_backend="local",
            source_mode="web",
            allow_live_web_search=True,
            live_dry_run=False,
            require_demo_proof=True,
            proves=[
                "Live OpenAI Agents SDK orchestration",
                "Live web evidence-source acquisition",
                "Modal remote research fan-out",
                "Raindrop Workshop artifact bundle",
                "Demo proof gate before accepting the run",
            ],
            notes="Full live run for configured demos: uses OpenAI API, live web search, and Modal.",
        ),
    ]


def demo_scenario_by_id(scenario_id: str) -> DemoScenario:
    scenarios = {scenario.id: scenario for scenario in demo_scenarios()}
    if scenario_id not in scenarios:
        raise KeyError(f"Unknown demo scenario: {scenario_id}")
    return scenarios[scenario_id]


def run_demo_smoke(
    *,
    output_dir: str | Path | None = None,
    run_openai_live: bool = False,
    run_modal_remote: bool = False,
) -> DemoSmokeResult:
    directory = Path(output_dir) if output_dir is not None else DEFAULT_DEMO_DIR
    directory.mkdir(parents=True, exist_ok=True)
    checks: list[DemoSmokeCheck] = []

    doctor = run_integration_doctor(
        run_openai_live=run_openai_live,
        run_modal_remote=run_modal_remote,
        output_dir=directory / "doctor",
    )
    doctor_path = directory / "integration_doctor.json"
    doctor_path.write_text(doctor.model_dump_json(indent=2), encoding="utf-8")
    checks.append(
        DemoSmokeCheck(
            name="integration_doctor",
            status="pass" if doctor.status != "failed" else "fail",
            message=doctor.summary,
            artifact_path=str(doctor_path),
        )
    )

    core_state = run_research_loop(
        DEFAULT_THESIS,
        execution_backend="local",
        observability_mode="local",
        observability_dir=directory / "core_trace",
    )
    core_path = directory / "core_state.json"
    core_path.write_text(core_state.model_dump_json(indent=2), encoding="utf-8")
    checks.append(
        DemoSmokeCheck(
            name="core_loop",
            status="pass" if core_state.generated_evals and core_state.invalid_leaps else "fail",
            message=(
                f"{len(core_state.invalid_leaps)} invalid leaps, "
                f"{len(core_state.generated_evals)} generated evals."
            ),
            artifact_path=str(core_path),
        )
    )

    replay = run_replay_demo(
        DEFAULT_THESIS,
        observability_mode="local",
        observability_dir=directory / "replay_trace",
    )
    replay_path = directory / "replay_result.json"
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    checks.append(
        DemoSmokeCheck(
            name="failure_replay",
            status="pass" if replay.eval_workshop and replay.eval_workshop.replay_outcomes else "fail",
            message=replay.summary,
            artifact_path=str(replay_path),
        )
    )

    eval_suite = run_eval_suite(DEFAULT_THESIS)
    eval_path = directory / "eval_suite.json"
    eval_path.write_text(eval_suite.model_dump_json(indent=2), encoding="utf-8")
    checks.append(
        DemoSmokeCheck(
            name="regression_gates",
            status=eval_suite.status,
            message=eval_suite.summary,
            artifact_path=str(eval_path),
        )
    )

    status = "pass" if all(check.status == "pass" for check in checks) else "fail"
    result = DemoSmokeResult(
        status=status,
        checks=checks,
        output_dir=str(directory),
        summary=f"{sum(check.status == 'pass' for check in checks)}/{len(checks)} demo smoke checks passed.",
    )
    summary_path = directory / "demo_smoke.json"
    summary_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return result
