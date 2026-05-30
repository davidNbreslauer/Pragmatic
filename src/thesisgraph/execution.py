from __future__ import annotations

from dataclasses import dataclass

from thesisgraph.extractors import extract_source_evidence
from thesisgraph.schemas import (
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
            metadata={"source_id": source.id},
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
        results = [run_research_task_local(task, backend=self.backend) for task in tasks]
        return ResearchBatchResult(
            backend=self.backend,
            attempted_backend=self.backend,
            results=results,
            metadata={
                "task_count": str(len(tasks)),
                "succeeded": str(sum(result.status == "succeeded" for result in results)),
            },
        )


@dataclass(frozen=True)
class ModalResearchExecutor:
    backend: ExecutionBackend = "modal"

    def run_batch(self, tasks: list[ResearchTask]) -> ResearchBatchResult:
        from thesisgraph.modal_jobs import run_research_tasks_with_modal

        results = run_research_tasks_with_modal(tasks)
        return ResearchBatchResult(
            backend=self.backend,
            attempted_backend=self.backend,
            results=results,
            metadata={
                "task_count": str(len(tasks)),
                "succeeded": str(sum(result.status == "succeeded" for result in results)),
            },
        )


def run_research_task_local(
    task: ResearchTask,
    *,
    backend: ExecutionBackend = "local",
) -> ResearchTaskResult:
    try:
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
    except Exception as exc:
        return ResearchTaskResult(
            task_id=task.id,
            task_type=task.task_type,
            backend=backend,
            status="failed",
            source_ids=_task_source_ids(task),
            error=f"{type(exc).__name__}: {exc}",
        )

    return ResearchTaskResult(
        task_id=task.id,
        task_type=task.task_type,
        backend=backend,
        status="skipped",
        source_ids=_task_source_ids(task),
        metadata={"reason": "unsupported task type"},
    )


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
    from thesisgraph.cross_checking import detect_evidence_conflicts

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
    from thesisgraph.verifiers import run_mock_verifier

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
