from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pragmatic.extractors import extract_source_evidence
from pragmatic.schemas import (
    Assumption,
    ExecutionBackend,
    ResearchBatchResult,
    ResearchQuestion,
    ResearchTask,
    ResearchTaskResult,
    Source,
)


class ResearchExecutionUnavailable(RuntimeError):
    """Raised when a requested research execution backend is unavailable."""


def build_source_extraction_tasks(
    sources: list[Source],
    assumptions: list[Assumption],
    questions: list[ResearchQuestion],
) -> list[ResearchTask]:
    question_ids_by_source = {
        source.id: [question.id for question in questions]
        for source in sources
    }
    return [
        ResearchTask(
            id=f"task_extract_{index:03d}_{source.id}",
            task_type="extract_evidence",
            question_ids=question_ids_by_source[source.id],
            assumption_ids=[assumption.id for assumption in assumptions],
            source=source,
            assumptions=assumptions,
            metadata={
                "source_id": source.id,
                "source_type": source.source_type,
                "evidence_scope": source.evidence_scope or "",
                "tags": ",".join(source.tags),
            },
        )
        for index, source in enumerate(sorted(sources, key=lambda item: item.id), start=1)
    ]


def build_source_parse_tasks(
    sources: list[Source],
    questions: list[ResearchQuestion],
) -> list[ResearchTask]:
    question_ids_by_source = {
        source.id: [question.id for question in questions]
        for source in sources
    }
    return [
        ResearchTask(
            id=f"task_parse_{index:03d}_{source.id}",
            task_type="parse_source",
            question_ids=question_ids_by_source[source.id],
            source=source,
            metadata={
                "source_id": source.id,
                "source_type": source.source_type,
                "evidence_scope": source.evidence_scope or "",
                "tags": ",".join(source.tags),
            },
        )
        for index, source in enumerate(sorted(sources, key=lambda item: item.id), start=1)
    ]


def build_cross_check_task(
    sources: list[Source],
    evidence_items: list,
    assumptions: list[Assumption],
) -> ResearchTask:
    return ResearchTask(
        id="task_cross_check_001",
        task_type="cross_check",
        sources=sorted(sources, key=lambda source: source.id),
        assumptions=assumptions,
        evidence_items=sorted(evidence_items, key=lambda item: item.id),
        metadata={"evidence_item_count": str(len(evidence_items))},
    )


def execute_research_tasks(
    tasks: list[ResearchTask],
    *,
    backend: ExecutionBackend = "local",
    fallback_to_local: bool = True,
) -> ResearchBatchResult:
    if backend == "local":
        return LocalResearchExecutor().run_batch(tasks)

    try:
        return ModalResearchExecutor().run_batch(tasks)
    except Exception as exc:
        if not fallback_to_local:
            raise
        local_result = LocalResearchExecutor().run_batch(tasks)
        return ResearchBatchResult(
            backend="local",
            attempted_backend="modal",
            results=local_result.results,
            fallback_reason=f"{type(exc).__name__}: {exc}",
            metadata=local_result.metadata,
        )


@dataclass(frozen=True)
class LocalResearchExecutor:
    backend: ExecutionBackend = "local"

    def run_batch(self, tasks: list[ResearchTask]) -> ResearchBatchResult:
        if tasks:
            max_workers = min(len(tasks), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        lambda task: _run_local_batch_task(task, self.backend),
                        tasks,
                    )
                )
        else:
            results = []
        return ResearchBatchResult(
            backend=self.backend,
            attempted_backend=self.backend,
            results=results,
            metadata=_batch_metadata(tasks, results),
        )


@dataclass(frozen=True)
class ModalResearchExecutor:
    backend: ExecutionBackend = "modal"

    def run_batch(self, tasks: list[ResearchTask]) -> ResearchBatchResult:
        from pragmatic.modal_jobs import (
            MODAL_TASK_RETRIES,
            MODAL_TASK_TIMEOUT_SECONDS,
            run_research_tasks_with_modal,
        )

        results = _run_modal_batch_safely(tasks, run_research_tasks_with_modal)
        return ResearchBatchResult(
            backend=self.backend,
            attempted_backend=self.backend,
            results=results,
            metadata={
                **_batch_metadata(tasks, results),
                "remote_app": "pragmatic",
                "worker_timeout_seconds": str(MODAL_TASK_TIMEOUT_SECONDS),
                "worker_retries": str(MODAL_TASK_RETRIES),
            },
        )


def run_research_task_local(
    task: ResearchTask,
    *,
    backend: ExecutionBackend = "local",
) -> ResearchTaskResult:
    started = time.perf_counter()
    try:
        result = _run_research_task_inner(task, backend)
    except Exception as exc:
        result = ResearchTaskResult(
            task_id=task.id,
            task_type=task.task_type,
            backend=backend,
            status="failed",
            source_ids=_task_source_ids(task),
            error=f"{type(exc).__name__}: {exc}",
        )
    return _with_task_runtime_metadata(result, task, started)


def _run_local_batch_task(
    task: ResearchTask,
    backend: ExecutionBackend,
) -> ResearchTaskResult:
    started = time.perf_counter()
    try:
        return run_research_task_local(task, backend=backend)
    except Exception as exc:
        result = ResearchTaskResult(
            task_id=task.id,
            task_type=task.task_type,
            backend=backend,
            status="failed",
            source_ids=_task_source_ids(task),
            error=f"{type(exc).__name__}: {exc}",
            metadata={"worker_status": "failed"},
        )
        return _with_task_runtime_metadata(result, task, started)


def _run_modal_batch_safely(tasks, runner):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return runner(tasks)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(runner, tasks).result()


def _run_research_task_inner(
    task: ResearchTask,
    backend: ExecutionBackend,
) -> ResearchTaskResult:
    if task.task_type == "retrieve_source":
        return _retrieve_source(task, backend)
    if task.task_type == "parse_source":
        return _parse_source(task, backend)
    if task.task_type == "extract_evidence":
        return _extract_evidence(task, backend)
    if task.task_type == "cross_check":
        return _cross_check(task, backend)
    if task.task_type == "verify_decisive_test":
        return _verify_decisive_test(task, backend)
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="skipped",
        source_ids=_task_source_ids(task),
        metadata={"reason": "unsupported task type"},
    )


def _with_task_runtime_metadata(
    result: ResearchTaskResult,
    task: ResearchTask,
    started: float,
) -> ResearchTaskResult:
    metadata = {
        **result.metadata,
        "worker_status": "completed" if result.status == "succeeded" else result.status,
        "duration_ms": str(max(0, int((time.perf_counter() - started) * 1000))),
        "input_source_count": str(len(_task_source_ids(task))),
        "output_source_count": str(len(result.sources)),
        "evidence_item_count": str(len(result.evidence_items)),
        "evidence_conflict_count": str(len(result.evidence_conflicts)),
        "verifier_result_count": str(len(result.verifier_results)),
    }
    return result.model_copy(update={"metadata": metadata})


def _retrieve_source(task: ResearchTask, backend: ExecutionBackend) -> ResearchTaskResult:
    sources = [task.source] if task.source is not None else task.sources
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="succeeded",
        source_ids=[source.id for source in sources],
        sources=sources,
        metadata={"operation": "prepared corpus source returned"},
    )


def _parse_source(task: ResearchTask, backend: ExecutionBackend) -> ResearchTaskResult:
    sources = [task.source] if task.source is not None else task.sources
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="succeeded",
        source_ids=[source.id for source in sources],
        sources=sources,
        metadata={"operation": "prepared corpus source normalized"},
    )


def _extract_evidence(task: ResearchTask, backend: ExecutionBackend) -> ResearchTaskResult:
    if task.source is None:
        raise ValueError("extract_evidence tasks require a source.")
    evidence_items = extract_source_evidence(task.source, task.assumptions)
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="succeeded",
        source_ids=[task.source.id],
        sources=[task.source],
        evidence_items=evidence_items,
        metadata={"evidence_item_count": str(len(evidence_items))},
    )


def _cross_check(task: ResearchTask, backend: ExecutionBackend) -> ResearchTaskResult:
    from pragmatic.cross_checking import detect_evidence_conflicts

    conflicts = detect_evidence_conflicts(task.sources, task.evidence_items)
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="succeeded",
        source_ids=sorted({source.id for source in task.sources}),
        evidence_conflicts=conflicts,
        metadata={"conflict_count": str(len(conflicts))},
    )


def _verify_decisive_test(task: ResearchTask, backend: ExecutionBackend) -> ResearchTaskResult:
    from pragmatic.verifiers import run_mock_verifier

    if task.decisive_test is None:
        raise ValueError("verify_decisive_test tasks require a decisive test.")
    verifier_result = run_mock_verifier(
        task.decisive_test,
        task.sources,
        task.evidence_items,
        task.evidence_conflicts,
    )
    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="succeeded",
        source_ids=sorted({source.id for source in task.sources}),
        verifier_results=[verifier_result],
        metadata={"verifier_status": verifier_result.status},
    )


def _task_source_ids(task: ResearchTask) -> list[str]:
    if task.source is not None:
        return [task.source.id]
    return sorted({source.id for source in task.sources})


def _batch_metadata(
    tasks: list[ResearchTask],
    results: list[ResearchTaskResult],
) -> dict[str, str]:
    return {
        "task_count": str(len(tasks)),
        "succeeded": str(sum(result.status == "succeeded" for result in results)),
        "failed": str(sum(result.status == "failed" for result in results)),
        "skipped": str(sum(result.status == "skipped" for result in results)),
        "task_types": ",".join(sorted({task.task_type for task in tasks})),
    }
