from pragmatic import DEFAULT_THESIS
from pragmatic.corpus import load_corpus
from pragmatic.extractors import (
    ExtractionResult,
    extract_evidence_batch,
    extract_evidence_local,
)
from pragmatic.modal_jobs import make_extraction_payloads, extract_source_job_local
from pragmatic.research_loop import decompose_thesis, run_research_loop
from pragmatic.schemas import EvidenceItem


def test_local_extractor_returns_typed_evidence():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)

    evidence = extract_evidence_local(sources, assumptions)

    assert len(evidence) >= 8
    assert all(isinstance(item, EvidenceItem) for item in evidence)
    assert any(item.evidence_type == "proxy" for item in evidence)


def test_modal_payloads_preserve_source_and_assumptions():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)

    payloads = make_extraction_payloads(sources[:2], assumptions)

    assert len(payloads) == 2
    assert payloads[0]["source"]["id"] == "source_001"
    assert payloads[0]["assumptions"][0]["id"] == "A1"


def test_modal_job_local_handler_validates_payload_shape():
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    payload = make_extraction_payloads(sources[:1], assumptions)[0]

    raw_items = extract_source_job_local(payload["source"], payload["assumptions"])

    assert raw_items
    assert EvidenceItem.model_validate(raw_items[0]).source_id == "source_001"


def test_modal_extraction_mode_can_use_adapter_without_remote_call(monkeypatch):
    sources = load_corpus()
    assumptions = decompose_thesis(DEFAULT_THESIS)
    called = {"value": False}

    def fake_modal_extractor(modal_sources, modal_assumptions):
        called["value"] = True
        return extract_evidence_local(modal_sources, modal_assumptions)

    monkeypatch.setattr(
        "pragmatic.extractors.extract_evidence_with_modal",
        fake_modal_extractor,
    )

    result = extract_evidence_batch(
        sources,
        assumptions,
        mode="modal",
        fallback_to_local=False,
    )

    assert called["value"] is True
    assert isinstance(result, ExtractionResult)
    assert result.backend == "modal"
    assert result.items


def test_research_loop_records_modal_fallback_metadata(monkeypatch):
    def failing_modal_runner(tasks):
        del tasks
        raise RuntimeError("modal unavailable in test")

    monkeypatch.setattr(
        "pragmatic.modal_jobs.run_research_tasks_with_modal",
        failing_modal_runner,
    )

    state = run_research_loop(
        DEFAULT_THESIS,
        extraction_mode="modal",
        modal_fallback=True,
        observability_mode="off",
    )
    extract_events = [event for event in state.trace_events if event.stage == "extract"]

    assert extract_events
    assert extract_events[0].metadata["attempted_backend"] == "modal"
    assert extract_events[0].metadata["backend"] == "local"
    assert "fallback_reason" in extract_events[0].metadata
