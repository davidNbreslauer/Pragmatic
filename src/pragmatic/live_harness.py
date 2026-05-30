from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pragmatic.agents import (
    AgentsSDKCredentialsError,
    LiveAgentsSDKNotEnabled,
    ResearchManager,
)
from pragmatic.extractors import ExtractionMode
from pragmatic.raindrop_client import ObservabilityMode
from pragmatic.research_loop import DEFAULT_THESIS
from pragmatic.schemas import (
    ExecutionBackend,
    LiveRunGuardrails,
    LiveRunProof,
    LiveRunResult,
    LiveRunStatus,
    ResearchState,
    SourceAcquisitionMode,
)


DEFAULT_LIVE_RUN_DIR = Path(".pragmatic") / "live_runs"
ProgressCallback = Callable[[dict[str, Any]], None]


def run_live_harness_sync(
    thesis_text: str = DEFAULT_THESIS,
    *,
    model: str | None = None,
    mode: str = "dry_run",
    allow_live_sdk: bool = False,
    max_turns: int = 4,
    timeout_seconds: float = 60.0,
    max_iterations: int = 1,
    corpus_path: str | Path | None = None,
    execution_backend: ExecutionBackend = "local",
    extraction_mode: ExtractionMode = "local",
    source_mode: SourceAcquisitionMode = "prepared",
    allow_live_web_search: bool = False,
    web_search_model: str | None = None,
    max_web_sources: int = 8,
    observability_mode: ObservabilityMode = "local",
    output_dir: str | Path | None = None,
    write_artifact: bool = True,
    require_demo_proof: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LiveRunResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_live_harness(
                thesis_text,
                model=model,
                mode=mode,
                allow_live_sdk=allow_live_sdk,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
                corpus_path=corpus_path,
                execution_backend=execution_backend,
                extraction_mode=extraction_mode,
                source_mode=source_mode,
                allow_live_web_search=allow_live_web_search,
                web_search_model=web_search_model,
                max_web_sources=max_web_sources,
                observability_mode=observability_mode,
                output_dir=output_dir,
                write_artifact=write_artifact,
                require_demo_proof=require_demo_proof,
                progress_callback=progress_callback,
            )
        )
    raise RuntimeError("Use run_live_harness from an active event loop.")


async def run_live_harness(
    thesis_text: str = DEFAULT_THESIS,
    *,
    model: str | None = None,
    mode: str = "dry_run",
    allow_live_sdk: bool = False,
    max_turns: int = 4,
    timeout_seconds: float = 60.0,
    max_iterations: int = 1,
    corpus_path: str | Path | None = None,
    execution_backend: ExecutionBackend = "local",
    extraction_mode: ExtractionMode = "local",
    source_mode: SourceAcquisitionMode = "prepared",
    allow_live_web_search: bool = False,
    web_search_model: str | None = None,
    max_web_sources: int = 8,
    observability_mode: ObservabilityMode = "local",
    output_dir: str | Path | None = None,
    write_artifact: bool = True,
    require_demo_proof: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LiveRunResult:
    guardrails = LiveRunGuardrails(
        mode="live" if mode == "live" else "dry_run",
        allow_live_sdk=allow_live_sdk,
        source_mode=source_mode,
        prepared_corpus_only=source_mode == "prepared",
        allow_live_web_search=allow_live_web_search,
        web_search_model=web_search_model,
        max_web_sources=max_web_sources,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        max_iterations=max_iterations,
        execution_backend=execution_backend,
        observability_backend=observability_mode,
    )
    result_id = _live_run_id()
    created_at = _utc_now()
    credentials_available = bool(os.getenv("OPENAI_API_KEY"))

    if guardrails.mode == "dry_run":
        result = LiveRunResult(
            id=result_id,
            created_at=created_at,
            completed_at=_utc_now(),
            elapsed_seconds=0.0,
            mode=guardrails.mode,
            status="ready",
            thesis_text=thesis_text,
            model=model,
            guardrails=guardrails,
            credentials_available=credentials_available,
            message=(
                "Dry run validated the live SDK guardrails. No OpenAI API call was made."
            ),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "guardrails",
                    "status": "succeeded",
                    "message": "Dry run validated live SDK guardrails.",
                    "metadata": {"mode": guardrails.mode},
                }
            )
        return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)

    if not allow_live_sdk:
        result = _blocked_result(
            result_id,
            created_at,
            thesis_text,
            model,
            guardrails,
            credentials_available,
            LiveAgentsSDKNotEnabled(
                "Live OpenAI Agents SDK execution requires explicit opt-in with allow_live_sdk=True."
            ),
        )
        return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)

    if not credentials_available:
        result = _blocked_result(
            result_id,
            created_at,
            thesis_text,
            model,
            guardrails,
            credentials_available,
            AgentsSDKCredentialsError(
                "Live OpenAI Agents SDK execution requires OPENAI_API_KEY."
            ),
        )
        return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)

    manager = ResearchManager(model=model, max_turns=max_turns)
    started = time.monotonic()
    try:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "live_harness",
                    "status": "running",
                    "message": "Live run harness accepted guardrails and started the research manager.",
                    "metadata": {
                        "mode": guardrails.mode,
                        "execution_backend": execution_backend,
                        "source_mode": source_mode,
                    },
                }
            )
        state = await asyncio.wait_for(
            manager.run_live(
                thesis_text,
                max_iterations=max_iterations,
                corpus_path=corpus_path,
                execution_backend=execution_backend,
                extraction_mode=extraction_mode,
                source_mode=source_mode,
                allow_live_web_search=allow_live_web_search,
                web_search_model=web_search_model,
                max_web_sources=max_web_sources,
                observability_mode=observability_mode,
                allow_live_sdk=True,
                progress_callback=progress_callback,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        result = _failed_result(
            result_id,
            created_at,
            thesis_text,
            model,
            guardrails,
            credentials_available,
            status="timed_out",
            error=exc,
            elapsed_seconds=time.monotonic() - started,
            message=f"Live SDK run exceeded the {timeout_seconds:.1f}s timeout.",
        )
        return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)
    except Exception as exc:
        result = _failed_result(
            result_id,
            created_at,
            thesis_text,
            model,
            guardrails,
            credentials_available,
            status="failed",
            error=exc,
            elapsed_seconds=time.monotonic() - started,
            message=_error_message("Live SDK run failed", exc),
        )
        return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)

    proof = build_live_run_proof(state, guardrails)
    status: LiveRunStatus = "succeeded"
    message = "Live OpenAI Agents SDK run completed and returned a valid ResearchState."
    if require_demo_proof and not proof.demo_ready:
        status = "failed"
        message = f"Live SDK run completed, but demo proof was incomplete. {proof.summary}"
    result = LiveRunResult(
        id=result_id,
        created_at=created_at,
        completed_at=_utc_now(),
        elapsed_seconds=time.monotonic() - started,
        mode=guardrails.mode,
        status=status,
        thesis_text=thesis_text,
        model=model,
        guardrails=guardrails,
        credentials_available=credentials_available,
        state=state,
        trace_path=_trace_path(state),
        trace_id=_trace_id(state),
        proof=proof,
        message=message,
        error_type="DemoProofIncomplete" if status == "failed" else None,
    )
    return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)


def build_live_run_proof(
    state: ResearchState,
    guardrails: LiveRunGuardrails,
) -> LiveRunProof:
    modal_task_count = len(
        [result for result in state.research_task_results if result.backend == "modal"]
    )
    fallback_task_count = len(
        [result for result in state.research_task_results if result.backend != guardrails.execution_backend]
    )
    workshop_recorded = state.observability is not None and state.observability.status == "recorded"
    final_output_validated = bool(state.agent_run and state.agent_run.final_output_validated)
    source_policy_ok = guardrails.prepared_corpus_only or (
        guardrails.source_mode == "web" and guardrails.allow_live_web_search
    )
    demo_ready = (
        final_output_validated
        and source_policy_ok
        and bool(state.sources)
        and bool(state.evidence_items)
        and (guardrails.execution_backend != "modal" or modal_task_count > 0)
        and (guardrails.observability_backend == "off" or workshop_recorded)
        and bool(state.generated_evals)
    )
    summary = (
        f"validated={final_output_validated}; source_mode={guardrails.source_mode}; "
        f"sources={len(state.sources)}; evidence={len(state.evidence_items)}; "
        f"modal_tasks={modal_task_count}; "
        f"workshop_recorded={workshop_recorded}; evals={len(state.generated_evals)}."
    )
    return LiveRunProof(
        final_output_validated=final_output_validated,
        prepared_corpus_only=guardrails.prepared_corpus_only,
        modal_task_count=modal_task_count,
        remote_modal_task_count=modal_task_count,
        fallback_task_count=fallback_task_count,
        workshop_recorded=workshop_recorded,
        trace_id=_trace_id(state),
        trace_path=_trace_path(state),
        generated_eval_count=len(state.generated_evals),
        invalid_leap_count=len(state.invalid_leaps),
        replay_outcome_count=(
            len(state.eval_workshop.replay_outcomes)
            if state.eval_workshop is not None
            else 0
        ),
        demo_ready=demo_ready,
        summary=summary,
    )


def load_live_run_result(path: str | Path) -> LiveRunResult:
    artifact_path = Path(path)
    result = LiveRunResult.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    result.output_path = result.output_path or str(artifact_path)
    return result


def load_latest_live_run(
    output_dir: str | Path | None = None,
    *,
    require_state: bool = False,
) -> LiveRunResult | None:
    directory = Path(output_dir) if output_dir is not None else DEFAULT_LIVE_RUN_DIR
    if not directory.exists():
        return None

    candidates = sorted(
        [path for path in directory.glob("*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for path in candidates:
        try:
            result = load_live_run_result(path)
        except Exception:
            continue
        if require_state and result.state is None:
            continue
        return result
    return None


def _blocked_result(
    result_id: str,
    created_at: str,
    thesis_text: str,
    model: str | None,
    guardrails: LiveRunGuardrails,
    credentials_available: bool,
    error: Exception,
) -> LiveRunResult:
    return LiveRunResult(
        id=result_id,
        created_at=created_at,
        completed_at=_utc_now(),
        elapsed_seconds=0.0,
        mode=guardrails.mode,
        status="blocked",
        thesis_text=thesis_text,
        model=model,
        guardrails=guardrails,
        credentials_available=credentials_available,
        message=str(error),
        error_type=type(error).__name__,
    )


def _failed_result(
    result_id: str,
    created_at: str,
    thesis_text: str,
    model: str | None,
    guardrails: LiveRunGuardrails,
    credentials_available: bool,
    *,
    status: LiveRunStatus,
    error: Exception,
    elapsed_seconds: float,
    message: str,
) -> LiveRunResult:
    return LiveRunResult(
        id=result_id,
        created_at=created_at,
        completed_at=_utc_now(),
        elapsed_seconds=elapsed_seconds,
        mode=guardrails.mode,
        status=status,
        thesis_text=thesis_text,
        model=model,
        guardrails=guardrails,
        credentials_available=credentials_available,
        message=message,
        error_type=type(error).__name__,
    )


def _maybe_write_result(
    result: LiveRunResult,
    *,
    output_dir: str | Path | None,
    write_artifact: bool,
) -> LiveRunResult:
    if not write_artifact:
        return result

    directory = Path(output_dir) if output_dir is not None else DEFAULT_LIVE_RUN_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.id}.json"
    result.output_path = str(path)
    path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return result


def _trace_path(state: ResearchState) -> str | None:
    if state.observability is None:
        return None
    return state.observability.trace_path


def _trace_id(state: ResearchState) -> str | None:
    if state.observability is None:
        return None
    return state.observability.trace_id


def _live_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"live_{timestamp}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _error_message(prefix: str, error: Exception) -> str:
    detail = str(error)
    if not detail:
        return f"{prefix}: {type(error).__name__}."
    return f"{prefix}: {type(error).__name__}: {detail}"
