from pragmatic import DEFAULT_THESIS
from pragmatic.corpus import load_corpus
from pragmatic.execution import (
    LocalResearchExecutor,
    build_source_parse_tasks,
    build_source_extraction_tasks,
    execute_research_tasks,
    run_research_task_local,
)
from pragmatic.modal_jobs import (
    MODAL_TASK_RETRIES,
    MODAL_TASK_TIMEOUT_SECONDS,
    make_research_task_payloads,
    research_task_job_local,
    run_research_tasks_with_modal,
)
from pragmatic.research_loop import decompose_thesis, run_research_loop
from pragmatic.schemas import EvidenceItem, ResearchTaskResult


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


def test_source_parse_tasks_record_worker_metadata():
    sources = load_corpus()
    questions = []
    tasks = build_source_parse_tasks(sources[:1], questions)

    result = LocalResearchExecutor().run_batch(tasks)

    task_result = result.results[0]
    assert task_result.task_type == "parse_source"
    assert task_result.sources[0].id == "source_001"
    assert task_result.metadata["worker_status"] == "completed"
    assert "duration_ms" in task_result.metadata
    assert task_result.metadata["output_source_count"] == "1"


def test_modal_task_payload_preserves_general_task_shape():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])

    payloads = make_research_task_payloads(tasks)
    raw_result = research_task_job_local(payloads[0])

    assert payloads[0]["task_type"] == "extract_evidence"
    assert payloads[0]["source"]["id"] == "source_001"
    parsed_result = ResearchTaskResult.model_validate(raw_result)
    assert parsed_result.task_id == tasks[0].id
    assert parsed_result.backend == "modal"


def test_modal_remote_runner_maps_payloads_and_preserves_modal_backend(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])
    seen_payloads = []

    class FakeApp:
        def run(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeRemoteJob:
        def map(self, payloads, **kwargs):
            seen_payloads.extend(payloads)
            assert kwargs["order_outputs"] is True
            assert kwargs["return_exceptions"] is True
            return [
                run_research_task_local(tasks[0], backend="modal").model_dump()
            ]

    monkeypatch.setattr("pragmatic.modal_jobs.modal", object())
    monkeypatch.setattr("pragmatic.modal_jobs.app", FakeApp())
    monkeypatch.setattr("pragmatic.modal_jobs.research_task_job", FakeRemoteJob())

    results = run_research_tasks_with_modal(tasks)

    assert seen_payloads[0]["task_type"] == "extract_evidence"
    assert results[0].backend == "modal"
    assert results[0].status == "succeeded"
    assert results[0].evidence_items


def test_modal_remote_runner_converts_worker_exception_to_failed_result(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])

    class FakeApp:
        def run(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeRemoteJob:
        def map(self, payloads, **kwargs):
            del payloads, kwargs
            return [RuntimeError("worker boom")]

    monkeypatch.setattr("pragmatic.modal_jobs.modal", object())
    monkeypatch.setattr("pragmatic.modal_jobs.app", FakeApp())
    monkeypatch.setattr("pragmatic.modal_jobs.research_task_job", FakeRemoteJob())

    results = run_research_tasks_with_modal(tasks)

    assert results[0].backend == "modal"
    assert results[0].status == "failed"
    assert results[0].source_ids == ["source_001"]
    assert "worker boom" in (results[0].error or "")


def test_modal_execution_backend_falls_back_to_local(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:1], assumptions, [])

    def failing_modal_runner(modal_tasks):
        raise RuntimeError("modal unavailable in test")

    monkeypatch.setattr(
        "pragmatic.modal_jobs.run_research_tasks_with_modal",
        failing_modal_runner,
    )

    result = execute_research_tasks(tasks, backend="modal", fallback_to_local=True)

    assert result.attempted_backend == "modal"
    assert result.backend == "local"
    assert result.results[0].evidence_items
    assert result.fallback_reason is not None


def test_modal_execution_backend_records_remote_batch_metadata(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    tasks = build_source_extraction_tasks(sources[:2], assumptions, [])

    def fake_modal_runner(modal_tasks):
        return [
            run_research_task_local(task, backend="modal")
            for task in modal_tasks
        ]

    monkeypatch.setattr(
        "pragmatic.modal_jobs.run_research_tasks_with_modal",
        fake_modal_runner,
    )

    result = execute_research_tasks(tasks, backend="modal", fallback_to_local=False)

    assert result.attempted_backend == "modal"
    assert result.backend == "modal"
    assert result.metadata["task_count"] == "2"
    assert result.metadata["succeeded"] == "2"
    assert result.metadata["failed"] == "0"
    assert result.metadata["worker_timeout_seconds"] == str(MODAL_TASK_TIMEOUT_SECONDS)
    assert result.metadata["worker_retries"] == str(MODAL_TASK_RETRIES)


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


def test_research_loop_records_parse_and_cross_check_execution_tasks():
    state = run_research_loop(
        DEFAULT_THESIS,
        execution_backend="local",
        observability_mode="off",
    )
    task_types = [result.task_type for result in state.research_task_results]

    assert task_types.count("parse_source") == len(state.sources)
    assert task_types.count("extract_evidence") == len(state.sources)
    assert "cross_check" in task_types
    assert any(result.metadata.get("duration_ms") for result in state.research_task_results)
