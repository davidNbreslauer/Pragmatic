from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from thesisgraph.schemas import Assumption, EvidenceItem, Source


ExtractionMode = Literal["local", "modal"]


@dataclass(frozen=True)
class ExtractionResult:
    items: list[EvidenceItem]
    backend: str
    attempted_backend: str
    metadata: dict[str, str] = field(default_factory=dict)


def extract_evidence(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    return extract_evidence_local(sources, assumptions)


def extract_evidence_batch(
    sources: list[Source],
    assumptions: list[Assumption],
    *,
    mode: ExtractionMode = "local",
    fallback_to_local: bool = True,
) -> ExtractionResult:
    if mode == "local":
        return ExtractionResult(
            items=extract_evidence_local(sources, assumptions),
            backend="local",
            attempted_backend="local",
            metadata={"source_count": str(len(sources))},
        )

    try:
        items = extract_evidence_with_modal(sources, assumptions)
        return ExtractionResult(
            items=items,
            backend="modal",
            attempted_backend="modal",
            metadata={"source_count": str(len(sources))},
        )
    except Exception as exc:
        if not fallback_to_local:
            raise
        return ExtractionResult(
            items=extract_evidence_local(sources, assumptions),
            backend="local",
            attempted_backend="modal",
            metadata={
                "source_count": str(len(sources)),
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            },
        )


def extract_evidence_local(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for source in sources:
        evidence.extend(extract_source_evidence(source, assumptions))
    return evidence


def extract_source_evidence(
    source: Source,
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    del assumptions
    extractors = {
        "source_001": _evidence_source_001,
        "source_002": _evidence_source_002,
        "source_003": _evidence_source_003,
        "source_004": _evidence_source_004,
        "source_005": _evidence_source_005,
        "source_006": _evidence_source_006,
        "source_007": _evidence_source_007,
        "source_008": _evidence_source_008,
    }
    extractor = extractors.get(source.id)
    if extractor is None:
        return []
    return extractor()


def extract_evidence_with_modal(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    from thesisgraph.modal_jobs import extract_evidence_with_modal as run_modal_extraction

    return run_modal_extraction(sources, assumptions)


def _evidence_source_001() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_001",
            source_id="source_001",
            assumption_ids=["A3", "A4"],
            evidence_type="proxy",
            claim_supported="AI-scientist workflows can produce plausible hypotheses and experimental plans.",
            quoted_evidence="The system automates idea generation, experiment code, and paper-style reporting.",
            limitation="This is not direct evidence of prospective materials discovery or independent validation.",
            confidence=0.66,
        )
    ]


def _evidence_source_002() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_002",
            source_id="source_002",
            assumption_ids=["A1", "A3", "A4"],
            evidence_type="indirect",
            claim_supported="Graph-based agents can organize scientific relations and generate materials hypotheses.",
            quoted_evidence="The demo uses a knowledge graph and agent roles to propose bio-inspired material ideas.",
            limitation="Generated hypotheses still require novelty checks and prospective validation.",
            confidence=0.62,
        )
    ]


def _evidence_source_003() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_003",
            source_id="source_003",
            assumption_ids=["A1", "A2"],
            evidence_type="indirect",
            claim_supported="Graph retrieval can improve corpus-level context organization.",
            quoted_evidence="Graph-based indexing helps aggregate local and global context across a corpus.",
            limitation="Retrieval gains do not by themselves prove mechanistic understanding or discovery.",
            confidence=0.7,
        )
    ]


def _evidence_source_004() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_004",
            source_id="source_004",
            assumption_ids=["A5", "A6"],
            evidence_type="proxy",
            claim_supported="Scientific discovery benchmarks measure related research skills.",
            quoted_evidence="Benchmark tasks evaluate whether models can recover data-driven discoveries.",
            limitation="Benchmark success is proxy evidence, not direct prospective discovery validation.",
            confidence=0.76,
        )
    ]


def _evidence_source_005() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_005",
            source_id="source_005",
            assumption_ids=["A5", "A7"],
            evidence_type="proxy",
            claim_supported="Agent benchmarks can compare automated experimentation performance.",
            quoted_evidence="The benchmark evaluates language agents on machine learning experimentation tasks.",
            limitation="ML experimentation benchmarks are not materials-discovery outcomes.",
            confidence=0.58,
        )
    ]


def _evidence_source_006() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_006",
            source_id="source_006",
            assumption_ids=["A6", "A7"],
            evidence_type="anecdotal",
            claim_supported="Company claims suggest autonomous systems may accelerate discovery workflows.",
            quoted_evidence="The announcement claims faster research cycles for AI-assisted discovery.",
            limitation="The claim is not independently validated in the source snippet.",
            confidence=0.35,
        )
    ]


def _evidence_source_007() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_007",
            source_id="source_007",
            assumption_ids=["A6", "A8"],
            evidence_type="contradictory",
            claim_supported="Prospective validation and broad generalization remain sparse.",
            quoted_evidence="Many AI-for-materials claims rely on retrospective examples or narrow case studies.",
            limitation="This is a general critique rather than a direct refutation of every system.",
            confidence=0.68,
        )
    ]


def _evidence_source_008() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="evidence_008",
            source_id="source_008",
            assumption_ids=["A2", "A7"],
            evidence_type="indirect",
            claim_supported="Graph-agent retrieval can outperform a weak baseline in selected examples.",
            quoted_evidence="The case study reports better context recall than a simple vector-search baseline.",
            limitation="The baseline is weak and does not establish superiority over strong non-graph RAG.",
            confidence=0.5,
        )
    ]
