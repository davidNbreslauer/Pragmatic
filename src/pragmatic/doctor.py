from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from pragmatic.corpus import load_corpus
from pragmatic.execution import execute_research_tasks
from pragmatic.research_loop import DEFAULT_THESIS, decompose_thesis, run_research_loop
from pragmatic.schemas import (
    IntegrationCheck,
    IntegrationDoctorResult,
    ResearchTask,
)


DEFAULT_DOCTOR_DIR = Path(".pragmatic") / "doctor"


def run_integration_doctor(
    *,
    run_openai_live: bool = False,
    run_modal_remote: bool = False,
    output_dir: str | Path | None = None,
) -> IntegrationDoctorResult:
    checks = [
        check_openai_agents_sdk(run_live=run_openai_live),
        check_modal(run_remote=run_modal_remote),
        check_raindrop_workshop(output_dir=output_dir),
    ]
    statuses = {check.status for check in checks}
    if "failed" in statuses:
        status = "failed"
    elif statuses & {"fallback", "unavailable", "skipped"}:
        status = "degraded"
    else:
        status = "ready"
    ready_count = sum(check.status in {"ready", "live"} for check in checks)
    return IntegrationDoctorResult(
        status=status,
        checks=checks,
        summary=f"{ready_count}/{len(checks)} integrations ready or live.",
    )


def check_openai_agents_sdk(*, run_live: bool = False) -> IntegrationCheck:
    started = time.perf_counter()
    credential_present = bool(os.getenv("OPENAI_API_KEY"))
    metadata = {"credential_present": str(credential_present).lower()}
    try:
        from agents import Agent  # noqa: F401
    except Exception as exc:
        return _check(
            "openai_agents_sdk",
            "unavailable",
            f"OpenAI Agents SDK import failed: {type(exc).__name__}: {exc}",
            started,
            metadata=metadata,
        )

    if not credential_present:
        return _check(
            "openai_agents_sdk",
            "unavailable",
            "OpenAI Agents SDK is installed, but OPENAI_API_KEY is missing.",
            started,
            metadata=metadata,
        )

    if not run_live:
        return _check(
            "openai_agents_sdk",
            "ready",
            "OpenAI Agents SDK is installed and credentials are present; live API check skipped.",
            started,
            metadata={**metadata, "live_check": "skipped"},
        )

    try:
        from openai import OpenAI

        models = OpenAI().models.list()
        first_model = models.data[0].id if models.data else ""
    except Exception as exc:
        return _check(
            "openai_agents_sdk",
            "failed",
            f"OpenAI API credential check failed: {type(exc).__name__}: {exc}",
            started,
            live=True,
            metadata={**metadata, "live_check": "models.list"},
        )

    return _check(
        "openai_agents_sdk",
        "live",
        "OpenAI Agents SDK is installed and the API key was accepted by a live API check.",
        started,
        live=True,
        metadata={**metadata, "live_check": "models.list", "first_model": first_model},
    )


def check_modal(*, run_remote: bool = False) -> IntegrationCheck:
    started = time.perf_counter()
    metadata = {"remote_check": str(run_remote).lower()}
    try:
        import modal

        metadata["client_version"] = getattr(modal, "__version__", "")
    except Exception as exc:
        return _check(
            "modal",
            "unavailable",
            f"Modal import failed: {type(exc).__name__}: {exc}",
            started,
            metadata=metadata,
        )

    profile_name = _modal_profile_name()
    if profile_name:
        metadata["profile"] = profile_name

    if not run_remote:
        return _check(
            "modal",
            "ready",
            "Modal is installed and configured; remote task check skipped.",
            started,
            metadata=metadata,
        )

    try:
        result = execute_research_tasks(
            [_modal_smoke_task()],
            backend="modal",
            fallback_to_local=False,
        )
    except Exception as exc:
        return _check(
            "modal",
            "failed",
            f"Modal remote smoke task failed: {type(exc).__name__}: {exc}",
            started,
            live=True,
            metadata=metadata,
        )

    task_result = result.results[0] if result.results else None
    if task_result is None or task_result.status != "succeeded":
        return _check(
            "modal",
            "failed",
            "Modal remote smoke task returned no successful task result.",
            started,
            live=True,
            metadata={
                **metadata,
                **result.metadata,
                "fallback_reason": result.fallback_reason or "",
                "task_status": task_result.status if task_result else "missing",
                "task_error": task_result.error if task_result else "",
            },
        )

    return _check(
        "modal",
        "live",
        "Modal executed a remote prepared-corpus smoke task successfully.",
        started,
        live=True,
        metadata={
            **metadata,
            **result.metadata,
            "task_id": task_result.task_id,
            "task_type": task_result.task_type,
            "source_ids": ",".join(task_result.source_ids),
        },
    )


def check_raindrop_workshop(
    *,
    output_dir: str | Path | None = None,
) -> IntegrationCheck:
    started = time.perf_counter()
    trace_dir = Path(output_dir) if output_dir is not None else DEFAULT_DOCTOR_DIR
    try:
        state = run_research_loop(
            DEFAULT_THESIS,
            max_iterations=1,
            execution_backend="local",
            observability_mode="local",
            observability_dir=trace_dir,
        )
    except Exception as exc:
        return _check(
            "raindrop_workshop",
            "failed",
            f"Raindrop Workshop local artifact check failed: {type(exc).__name__}: {exc}",
            started,
        )

    record = state.observability
    if record is None or record.workshop_path is None:
        return _check(
            "raindrop_workshop",
            "failed",
            "Raindrop Workshop local artifact check did not produce a workshop path.",
            started,
        )

    workshop_path = Path(record.workshop_path)
    if not workshop_path.exists():
        return _check(
            "raindrop_workshop",
            "failed",
            "Raindrop Workshop local artifact path was reported but not written.",
            started,
            artifact_path=str(workshop_path),
        )

    return _check(
        "raindrop_workshop",
        "live",
        "Raindrop Workshop bundle was generated locally for failure-to-eval inspection.",
        started,
        live=True,
        metadata={
            "trace_id": record.trace_id,
            "trace_path": record.trace_path or "",
            "eval_artifacts": str(len(record.eval_artifact_ids)),
            "failure_artifacts": str(len(record.failure_artifact_ids)),
            "workshop_artifacts": str(len(record.workshop_artifact_ids)),
        },
        artifact_path=str(workshop_path),
    )


def _modal_smoke_task() -> ResearchTask:
    source = load_corpus()[0]
    assumptions = decompose_thesis(DEFAULT_THESIS)
    return ResearchTask(
        id="doctor_modal_parse_001",
        task_type="parse_source",
        source=source,
        assumptions=assumptions[:1],
        metadata={"purpose": "integration_doctor"},
    )


def _modal_profile_name() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", "profile", "current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _check(
    layer,
    status,
    message: str,
    started: float,
    *,
    live: bool = False,
    metadata: dict[str, str] | None = None,
    artifact_path: str | None = None,
) -> IntegrationCheck:
    return IntegrationCheck(
        layer=layer,
        status=status,
        live=live,
        message=message,
        metadata=metadata or {},
        artifact_path=artifact_path,
        duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
    )
