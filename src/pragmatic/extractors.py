from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pragmatic.schemas import Assumption, EvidenceItem, Source


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
    return _extract_generic_source_evidence(source, assumptions)


def _extract_generic_source_evidence(
    source: Source,
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    source_text = " ".join(
        [
            source.title,
            source.evidence_scope or "",
            " ".join(source.tags),
            source.text,
        ]
    )
    source_terms = _tokenize(source_text)
    scored_assumptions = sorted(
        [
            (
                _term_overlap_score(source_terms, _tokenize(_assumption_search_text(assumption))),
                assumption,
            )
            for assumption in assumptions
        ],
        key=lambda item: (-item[0], item[1].id),
    )
    selected = [
        (score, assumption)
        for score, assumption in scored_assumptions[:3]
        if score >= 0.02
    ]
    if not selected and assumptions:
        selected = [(0.01, assumptions[0])]

    evidence: list[EvidenceItem] = []
    for score, assumption in selected:
        best_sentence = _best_sentence(source.text, _tokenize(_assumption_search_text(assumption)))
        evidence_type = _classify_generic_evidence(source, assumption, best_sentence)
        confidence = _generic_confidence(score, evidence_type, source)
        evidence.append(
            EvidenceItem(
                id=_generic_evidence_id(source.id, assumption.id),
                source_id=source.id,
                assumption_ids=[assumption.id],
                evidence_type=evidence_type,
                claim_supported=_generic_claim(assumption, best_sentence),
                quoted_evidence=best_sentence,
                limitation=_generic_limitation(source, assumption, evidence_type),
                confidence=confidence,
            )
        )
    return evidence


def _assumption_search_text(assumption: Assumption) -> str:
    return " ".join([assumption.text, assumption.why_it_matters, *assumption.evidence_needed])


def _term_overlap_score(source_terms: set[str], assumption_terms: set[str]) -> float:
    if not source_terms or not assumption_terms:
        return 0.0
    return len(source_terms & assumption_terms) / len(assumption_terms)


def _classify_generic_evidence(
    source: Source,
    assumption: Assumption,
    sentence: str,
) -> str:
    text = " ".join([source.title, source.evidence_scope or "", source.text, sentence]).lower()
    assumption_text = assumption.text.lower()
    if any(
        phrase in text
        for phrase in [
            "no evidence",
            "not sufficient",
            "not enough",
            "cannot",
            "could not",
            "challenge",
            "limitation",
            "difficult",
            "barrier",
            "failed",
            "falls short",
        ]
    ):
        return "contradictory"
    if source.source_type == "company_claim":
        return "anecdotal"
    if source.source_type in {"standard", "government"}:
        return "direct"
    application_terms = {
        "application",
        "product",
        "prototype",
        "vest",
        "armor",
        "bullet",
        "ballistic",
        "standard",
        "testing",
        "validated",
        "independent",
    }
    has_application_terms = bool(application_terms & _tokenize(f"{text} {assumption_text}"))
    if source.source_type in {"paper", "review", "case_study"} and has_application_terms:
        return "direct"
    if source.source_type in {"paper", "review", "dataset", "case_study"}:
        return "indirect"
    if source.source_type in {"blog_post", "news", "unknown"}:
        return "proxy"
    return "indirect"


def _generic_confidence(score: float, evidence_type: str, source: Source) -> float:
    type_base = {
        "direct": 0.68,
        "indirect": 0.5,
        "proxy": 0.34,
        "anecdotal": 0.24,
        "contradictory": 0.56,
        "irrelevant": 0.0,
        "not_relevant": 0.0,
    }[evidence_type]
    source_bonus = 0.08 if source.source_type in {"paper", "review", "standard", "government"} else 0.0
    return round(min(0.9, max(0.12, type_base + source_bonus + min(score, 0.2))), 2)


def _generic_claim(assumption: Assumption, sentence: str) -> str:
    return (
        f"Evidence relevant to '{assumption.text}': "
        f"{sentence[:220]}"
    )


def _generic_limitation(source: Source, assumption: Assumption, evidence_type: str) -> str:
    if evidence_type == "direct":
        return source.evidence_scope or "Directly relevant, but still bounded by the source context."
    if evidence_type == "contradictory":
        return "This source introduces a limitation or negative constraint for the assumption."
    if source.source_type == "company_claim":
        return "Company or product claims need independent corroboration."
    if evidence_type == "proxy":
        return (
            "This is proxy evidence; it may not test the claimed application directly."
        )
    if "application-level" in assumption.text.lower():
        return "The source may measure related properties rather than the final application."
    return source.evidence_scope or "Indirect evidence; preserve uncertainty."


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return text[:260] or "No source text was available beyond metadata."
    scored = sorted(
        sentences,
        key=lambda sentence: (-len(_tokenize(sentence) & query_terms), len(sentence)),
    )
    return scored[0][:320]


def _generic_evidence_id(source_id: str, assumption_id: str) -> str:
    raw = f"evidence_{source_id}_{assumption_id}"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).lower()


def _tokenize(value: str) -> set[str]:
    stopwords = {
        "about",
        "against",
        "also",
        "and",
        "are",
        "can",
        "claim",
        "does",
        "for",
        "from",
        "has",
        "have",
        "into",
        "not",
        "that",
        "the",
        "this",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in stopwords
    }


def extract_evidence_with_modal(
    sources: list[Source],
    assumptions: list[Assumption],
) -> list[EvidenceItem]:
    from pragmatic.modal_jobs import extract_evidence_with_modal as run_modal_extraction

    return run_modal_extraction(sources, assumptions)
