from __future__ import annotations

import json
from pathlib import Path

from thesisgraph.schemas import Source


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "ai_scientist_sources.json"


def load_corpus(path: str | Path | None = None) -> list[Source]:
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    with corpus_path.open("r", encoding="utf-8") as handle:
        raw_sources = json.load(handle)
    return [Source.model_validate(source) for source in raw_sources]

