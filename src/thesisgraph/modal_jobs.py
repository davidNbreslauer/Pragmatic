from __future__ import annotations

from typing import Any

from thesisgraph.schemas import Assumption, EvidenceItem, ResearchTask, ResearchTaskResult, Source

try:
    import modal
except ImportError:  # pragma: no cover - depends on optional local environment.
    modal = None


class ModalExtractionUnavailable(RuntimeError):
    """Raised when Modal extraction is requested without an available Modal runtime."""


if modal is not None:
    image = modal.Image.debian_slim().pip_install("pydantic>=2.7,<3")
    app = modal.App(name="thesisgraph", image=image)

    @app.function()
    def extract_source_job(source: dict[str, Any], assumptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from thesisgraph.extractors import extract_source_evidence
        from thesisgraph.schemas import Assumption, Source

        parsed_source = Source.model_validate(source)
        parsed_assumptions = [
            Assumption.model_validate(assumption)
            for assumption in assumptions
        ]
        return [
            evidence_item.model_dump()
            for evidence_item in extract_source_evidence(parsed_source, parsed_assumptions)
        ]

    @app.function()
    def research_task_job(task: dict[str, Any]) -> dict[str, Any]:
        from thesisgraph.execution import run_research_task_local
        from thesisgraph.schemas import ResearchTask

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
    from thesisgraph.extractors import extract_source_evidence

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
    from thesisgraph.execution import run_research_task_local

    parsed_task = ResearchTask.model_validate(task)
    return run_research_task_local(parsed_task, backend="local").model_dump()


def run_research_tasks_with_modal(tasks: list[ResearchTask]) -> list[ResearchTaskResult]:
    if modal is None or app is None or research_task_job is None:
        raise ModalExtractionUnavailable(
            "Install and configure Modal to use execution_backend='modal'."
        )

    payloads = make_research_task_payloads(tasks)
    with app.run():
        raw_results = list(research_task_job.map(payloads))

    return [
        ResearchTaskResult.model_validate(result)
        for result in raw_results
    ]


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
