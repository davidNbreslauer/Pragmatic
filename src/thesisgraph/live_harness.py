from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from thesisgraph.agents import (
    AgentsSDKCredentialsError,
    LiveAgentsSDKNotEnabled,
    ResearchManager,
)
from thesisgraph.extractors import ExtractionMode
from thesisgraph.raindrop_client import ObservabilityMode
from thesisgraph.research_loop import DEFAULT_THESIS
from thesisgraph.schemas import (
    ExecutionBackend,
    LiveRunGuardrails,
    LiveRunResult,
    LiveRunStatus,
    ResearchState,
)


DEFAULT_LIVE_RUN_DIR = Path(".thesisgraph") / "live_runs"


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
    observability_mode: ObservabilityMode = "local",
    output_dir: str | Path | None = None,
    write_artifact: bool = True,
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
                observability_mode=observability_mode,
                output_dir=output_dir,
                write_artifact=write_artifact,
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
    observability_mode: ObservabilityMode = "local",
    output_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> LiveRunResult:
    guardrails = LiveRunGuardrails(
        mode="live" if mode == "live" else "dry_run",
        allow_live_sdk=allow_live_sdk,
        prepared_corpus_only=True,
        allow_live_web_search=False,
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
        state = await asyncio.wait_for(
            manager.run_live(
                thesis_text,
                max_iterations=max_iterations,
                corpus_path=corpus_path,
                execution_backend=execution_backend,
                extraction_mode=extraction_mode,
                observability_mode=observability_mode,
                allow_live_sdk=True,
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

    result = LiveRunResult(
        id=result_id,
        created_at=created_at,
        completed_at=_utc_now(),
        elapsed_seconds=time.monotonic() - started,
        mode=guardrails.mode,
        status="succeeded",
        thesis_text=thesis_text,
        model=model,
        guardrails=guardrails,
        credentials_available=credentials_available,
        state=state,
        trace_path=_trace_path(state),
        trace_id=_trace_id(state),
        message="Live OpenAI Agents SDK run completed and returned a valid ResearchState.",
    )
    return _maybe_write_result(result, output_dir=output_dir, write_artifact=write_artifact)


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
