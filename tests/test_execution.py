from thesisgraph import DEFAULT_THESIS
from thesisgraph.corpus import load_corpus
from thesisgraph.execution import (
    LocalResearchExecutor,
    build_source_extraction_tasks,
    execute_research_tasks,
)
from thesisgraph.modal_jobs import make_research_task_payloads, research_task_job_local
from thesisgraph.research_loop import decompose_thesis, run_research_loop
from thesisgraph.schemas import EvidenceItem, ResearchTaskResult


def test_local_research_executor_runs_typed_source_tasks():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:2], assumptions, [])

    result = LocalResearchExecutor().run_batch(tasks)

    assert result.backend == "local"
    assert result.attempted_backend == "local"
    assert len(result.results) == 2
    assert all(isinstance(item, EvidenceItem) for item in result.results[0].evidence_items)
    assert tasks[0].metadata["source_type"] == "paper"
    assert tasks[0].metadata["evidence_scope"]


def test_modal_task_payload_preserves_general_task_shape():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])

    payloads = make_research_task_payloads(tasks)
    raw_result = research_task_job_local(payloads[0])

    assert payloads[0]["task_type"] == "extract_evidence"
    assert payloads[0]["source"]["id"] == "source_001"
    assert ResearchTaskResult.model_validate(raw_result).task_id == tasks[0].id


def test_modal_execution_backend_falls_back_to_local(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])

    def failing_modal_runner(modal_tasks):
        raise RuntimeError("modal unavailable in test")

    monkeypatch.setattr(
        "thesisgraph.modal_jobs.run_research_tasks_with_modal",
        failing_modal_runner,
    )

    result = execute_research_tasks(tasks, backend="modal", fallback_to_local=True)

    assert result.attempted_backend == "modal"
    assert result.backend == "local"
    assert result.results[0].evidence_items
    assert result.fallback_reason is not None


def test_research_loop_fans_out_one_extract_task_per_source():
    state = run_research_loop(
        DEFAULT_THESIS,
        execution_backend="local",
        observability_mode="off",
    )
    extract_results = [
        result for result in state.research_task_results if result.task_type == "extract_evidence"
    ]

    assert len(extract_results) == len(state.sources)
    assert [result.source_ids[0] for result in extract_results] == sorted(
        source.id for source in state.sources
    )
    assert any(event.stage == "execute" for event in state.trace_events)
