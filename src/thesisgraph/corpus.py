from __future__ import annotations

import json
import re
from pathlib import Path

from thesisgraph.schemas import ResearchQuestion, RetrievalScore, Source


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "ai_scientist_sources.json"
STOPWORDS = {
    "a",
    "against",
    "and",
    "are",
    "as",
    "based",
    "be",
    "better",
    "can",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "over",
    "than",
    "the",
    "to",
    "with",
}


def load_corpus(path: str | Path | None = None) -> list[Source]:
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    with corpus_path.open("r", encoding="utf-8") as handle:
        raw_sources = json.load(handle)
    return [Source.model_validate(source) for source in raw_sources]


def score_corpus_for_questions(
    questions: list[ResearchQuestion],
    corpus: list[Source],
) -> list[RetrievalScore]:
    scores: list[RetrievalScore] = []
    for question in questions:
        query_terms = _tokenize(" ".join([question.question, question.query]))
        for source in corpus:
            scores.append(score_source_for_question(source, question, query_terms))
    return sorted(scores, key=lambda score: (score.question_id, -score.score, score.source_id))


def rank_sources_for_questions(
    questions: list[ResearchQuestion],
    corpus: list[Source],
    *,
    min_score: float = 0.0,
) -> tuple[list[Source], list[RetrievalScore]]:
    scores = score_corpus_for_questions(questions, corpus)
    best_score_by_source = {
        source.id: max(
            [score.score for score in scores if score.source_id == source.id],
            default=0.0,
        )
        for source in corpus
    }
    ranked_sources = [
        source
        for source in corpus
        if best_score_by_source[source.id] >= min_score
    ]
    ranked_sources.sort(key=lambda source: (-best_score_by_source[source.id], source.id))
    return ranked_sources, scores


def score_source_for_question(
    source: Source,
    question: ResearchQuestion,
    query_terms: set[str] | None = None,
) -> RetrievalScore:
    resolved_query_terms = query_terms or _tokenize(" ".join([question.question, question.query]))
    source_terms = _source_terms(source)
    matched_terms = sorted(resolved_query_terms & source_terms)
    coverage = (
        len(matched_terms) / len(resolved_query_terms)
        if resolved_query_terms
        else 0.0
    )
    tag_overlap = len(resolved_query_terms & _tokenize(" ".join(source.tags)))
    source_type_bonus = (
        0.05
        if source.source_type in {"paper", "benchmark", "review", "standard", "government"}
        else 0.0
    )
    score = min(1.0, coverage + (0.03 * tag_overlap) + source_type_bonus)
    rationale = _score_rationale(score, matched_terms, source)
    return RetrievalScore(
        id=f"retrieval_{question.id}_{source.id}",
        question_id=question.id,
        source_id=source.id,
        score=round(score, 3),
        matched_terms=matched_terms,
        rationale=rationale,
    )


def _source_terms(source: Source) -> set[str]:
    return _tokenize(
        " ".join(
            [
                source.title,
                source.source_type,
                source.citation or "",
                source.evidence_scope or "",
                " ".join(source.tags),
                source.text,
            ]
        )
    )


def _tokenize(value: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in STOPWORDS
    }
    return {_normalize_token(term) for term in terms}


def _normalize_token(term: str) -> str:
    aliases = {
        "benchmarks": "benchmark",
        "discoverybench": "benchmark",
        "discoveries": "discovery",
        "experimentation": "experiment",
        "experiments": "experiment",
        "generated": "generate",
        "generates": "generate",
        "generating": "generate",
        "graphs": "graph",
        "hypotheses": "hypothesis",
        "materials": "material",
        "predictions": "prediction",
        "retrieves": "retrieve",
        "retrieval": "retrieve",
        "validation": "validate",
        "validated": "validate",
    }
    return aliases.get(term, term)


def _score_rationale(score: float, matched_terms: list[str], source: Source) -> str:
    if not matched_terms:
        return f"No query terms matched {source.id}; retained only for full prepared-corpus coverage."
    terms = ", ".join(matched_terms[:6])
    return f"Matched {len(matched_terms)} terms ({terms}) with score {score:.3f}."
