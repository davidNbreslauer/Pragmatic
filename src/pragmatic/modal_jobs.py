from __future__ import annotations

from typing import Any

from pragmatic.schemas import Assumption, EvidenceItem, ResearchTask, ResearchTaskResult, Source


MODAL_TASK_TIMEOUT_SECONDS = 300
MODAL_TASK_RETRIES = 1
MODAL_MIN_CONTAINERS = 1
MODAL_SCALEDOWN_WINDOW_SECONDS = 600

try:
    import modal
except ImportError:  # pragma: no cover - depends on optional local environment.
    modal = None


class ModalExtractionUnavailable(RuntimeError):
    """Raised when Modal extraction is requested without an available Modal runtime."""


if modal is not None:
    image = (
        modal.Image.debian_slim()
        .pip_install("pydantic>=2.7,<3")
        .add_local_python_source("pragmatic")
    )
    app = modal.App(name="pragmatic", image=image)

    @app.function(
        timeout=MODAL_TASK_TIMEOUT_SECONDS,
        retries=MODAL_TASK_RETRIES,
        min_containers=MODAL_MIN_CONTAINERS,
        scaledown_window=MODAL_SCALEDOWN_WINDOW_SECONDS,
    )
    def extract_source_job(source: dict[str, Any], assumptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from pragmatic.extractors import extract_source_evidence
        from pragmatic.schemas import Assumption, Source

        parsed_source = Source.model_validate(source)
        parsed_assumptions = [
            Assumption.model_validate(assumption)
            for assumption in assumptions
        ]
        return [
            evidence_item.model_dump()
            for evidence_item in extract_source_evidence(parsed_source, parsed_assumptions)
        ]

    @app.function(
        timeout=MODAL_TASK_TIMEOUT_SECONDS,
        retries=MODAL_TASK_RETRIES,
        min_containers=MODAL_MIN_CONTAINERS,
        scaledown_window=MODAL_SCALEDOWN_WINDOW_SECONDS,
    )
    def research_task_job(task: dict[str, Any]) -> dict[str, Any]:
        from pragmatic.execution import run_research_task_local
        from pragmatic.schemas import ResearchTask

        parsed_task = ResearchTask.model_validate(task)
        return run_research_task_local(parsed_task, backend="modal").model_dump()

else:
    app = None
    extract_source_job = None
    research_task_job = None


def make_extraction_payloads(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[dict[str, Any]]:
    assumption_payload = [assumption.model_dump() for assumption in assumptions]
    return [
        {
            "source": source.model_dump(),
            "assumptions": assumption_payload,
        }
        for source in sources
    ]


def extract_source_job_local(
    source: dict[str, Any],
    assumptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from pragmatic.extractors import extract_source_evidence

    parsed_source = Source.model_validate(source)
    parsed_assumptions = [
        Assumption.model_validate(assumption)
        for assumption in assumptions
    ]
    return [
        evidence_item.model_dump()
        for evidence_item in extract_source_evidence(parsed_source, parsed_assumptions)
    ]


def make_research_task_payloads(tasks: list[ResearchTask]) -> list[dict[str, Any]]:
    return [task.model_dump() for task in tasks]


def research_task_job_local(task: dict[str, Any]) -> dict[str, Any]:
    from pragmatic.execution import run_research_task_local

    parsed_task = ResearchTask.model_validate(task)
    return run_research_task_local(parsed_task, backend="modal").model_dump()


def run_research_tasks_with_modal(tasks: list[ResearchTask]) -> list[ResearchTaskResult]:
    if modal is None or app is None or research_task_job is None:
        raise ModalExtractionUnavailable(
            "Install and configure Modal to use execution_backend='modal'."
        )

    payloads = make_research_task_payloads(tasks)
    with app.run():
        raw_results = list(
            research_task_job.map(
                payloads,
                order_outputs=True,
                return_exceptions=True,
            )
        )

    return [
        _coerce_remote_task_result(task, result)
        for task, result in zip(tasks, raw_results, strict=True)
    ]


def prewarm_modal_functions() -> dict[str, str]:
    if modal is None or app is None or extract_source_job is None or research_task_job is None:
        raise ModalExtractionUnavailable("Install and configure Modal to prewarm Pragmatic jobs.")

    source = Source(
        id="prewarm_source_001",
        title="Prewarm source",
        url="https://example.test/prewarm",
        source_type="paper",
        text="Prewarm payload for Modal container readiness.",
    )
    assumption = Assumption(
        id="A0",
        text="Prewarm assumption.",
        why_it_matters="Keeps the Modal container warm before the demo run.",
        evidence_needed=["No evidence needed for prewarm."],
    )
    task = ResearchTask(
        id="prewarm_task_001",
        task_type="parse_source",
        source=source,
        metadata={"purpose": "modal_prewarm"},
    )

    with app.run():
        extract_source_job.remote(source.model_dump(), [assumption.model_dump()])
        research_task_job.remote(task.model_dump())

    return {
        "status": "succeeded",
        "extract_source_job": "warmed",
        "research_task_job": "warmed",
        "min_containers": str(MODAL_MIN_CONTAINERS),
        "scaledown_window": str(MODAL_SCALEDOWN_WINDOW_SECONDS),
    }


def extract_evidence_with_modal(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    if modal is None or app is None or extract_source_job is None:
        raise ModalExtractionUnavailable(
            "Install and configure Modal to use extraction_mode='modal'."
        )

    payloads = make_extraction_payloads(sources, assumptions)
    source_payloads = [payload["source"] for payload in payloads]
    assumption_payloads = [payload["assumptions"] for payload in payloads]

    with app.run():
        raw_batches = list(extract_source_job.map(source_payloads, assumption_payloads))

    raw_items = [
        item
        for batch in raw_batches
        for item in batch
    ]
    return [
        EvidenceItem.model_validate(item)
        for item in raw_items
    ]


def _coerce_remote_task_result(task: ResearchTask, result: Any) -> ResearchTaskResult:
    if isinstance(result, BaseException):
        return ResearchTaskResult(
            task_id=task.id,
            task_type=task.task_type,
            backend="modal",
            status="failed",
            source_ids=_task_source_ids(task),
            error=f"{type(result).__name__}: {result}",
            metadata={"worker_status": "exception"},
        )
    return ResearchTaskResult.model_validate(result)


def _task_source_ids(task: ResearchTask) -> list[str]:
    if task.source is not None:
        return [task.source.id]
    return sorted({source.id for source in task.sources})
