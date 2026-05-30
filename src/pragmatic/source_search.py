from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pragmatic.schemas import ResearchQuestion, Source


DEFAULT_WEB_SEARCH_MODEL = "gpt-5-mini"
ALLOWED_SOURCE_TYPES = {
    "paper",
    "benchmark",
    "blog_post",
    "company_claim",
    "case_study",
    "standard",
    "government",
    "news",
    "review",
    "dataset",
    "unknown",
}


class WebSearchUnavailable(RuntimeError):
    """Raised when live web source acquisition cannot run."""


def build_web_corpus(
    thesis_text: str,
    questions: list[ResearchQuestion],
    *,
    model: str | None = None,
    max_sources: int = 8,
    client: Any | None = None,
) -> list[Source]:
    """Search the web with OpenAI and normalize results into Source records."""

    if client is None and not os.getenv("OPENAI_API_KEY"):
        raise WebSearchUnavailable("OPENAI_API_KEY is required for live web source search.")

    resolved_client = client or _openai_client()
    resolved_model = model or os.getenv("PRAGMATIC_WEB_SEARCH_MODEL") or DEFAULT_WEB_SEARCH_MODEL
    response = _create_web_search_response(
        resolved_client,
        model=resolved_model,
        prompt=_build_source_search_prompt(thesis_text, questions, max_sources=max_sources),
    )
    response_text = _response_text(response)
    raw_sources = _parse_source_cards(response_text)
    if not raw_sources:
        raw_sources = _source_cards_from_response(response, response_text)
    sources = _normalize_source_cards(raw_sources, response_text=response_text, max_sources=max_sources)
    if not sources:
        raise WebSearchUnavailable("OpenAI web search returned no source cards.")
    return sources


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment guard.
        raise WebSearchUnavailable("Install the openai package to use live web search.") from exc
    return OpenAI()


def _create_web_search_response(client: Any, *, model: str, prompt: str) -> Any:
    kwargs = {
        "model": model,
        "input": prompt,
        "tools": [{"type": "web_search", "search_context_size": "medium"}],
        "max_output_tokens": 3500,
    }
    try:
        return client.responses.create(**kwargs, tool_choice="required")
    except TypeError:
        return client.responses.create(**kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "tool_choice" not in message and "required" not in message:
            raise
        return client.responses.create(**kwargs)


def _build_source_search_prompt(
    thesis_text: str,
    questions: list[ResearchQuestion],
    *,
    max_sources: int,
) -> str:
    question_lines = "\n".join(
        f"- {question.id}: {question.question} Query: {question.query}"
        for question in questions
    )
    return f"""
You are building an evidence corpus for Pragmatic.

Thesis:
{thesis_text}

Research questions:
{question_lines}

Use web search. Return only JSON with this shape:
{{
  "sources": [
    {{
      "title": "source title",
      "url": "https://...",
      "source_type": "paper|review|standard|government|dataset|case_study|company_claim|blog_post|news|unknown",
      "published_year": 2024,
      "tags": ["short", "lowercase", "tags"],
      "evidence_scope": "what this source can and cannot establish",
      "text": "dense evidence notes, including relevant quoted or near-quoted details, measured quantities, limitations, and how it bears on the thesis"
    }}
  ]
}}

Rules:
- Include at most {max_sources} sources.
- Prefer primary papers, standards, government or testing bodies, reviews, and credible technical sources.
- Include contradictory or limiting evidence when available.
- Do not invent URLs. Omit a source if you cannot attach a URL.
- Keep each text field under 900 characters but include enough detail for downstream evidence extraction.
""".strip()


def _response_text(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    texts: list[str] = []
    for item in _walk(_as_plain_data(response)):
        if not isinstance(item, dict):
            continue
        value = item.get("text") or item.get("content") or item.get("value")
        if isinstance(value, str):
            texts.append(value)
    return "\n".join(texts)


def _parse_source_cards(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    cleaned = _strip_code_fence(text.strip())
    payload = _load_json_payload(cleaned)
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        sources = payload.get("sources")
        if isinstance(sources, list):
            return [item for item in sources if isinstance(item, dict)]
    return []


def _source_cards_from_response(response: Any, response_text: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in _walk(_as_plain_data(response)):
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        cards.append(
            {
                "title": item.get("title") or url,
                "url": url,
                "source_type": "unknown",
                "tags": [],
                "evidence_scope": "Cited by OpenAI web search result.",
                "text": response_text[:900],
            }
        )
    return cards


def _normalize_source_cards(
    raw_sources: list[dict[str, Any]],
    *,
    response_text: str,
    max_sources: int,
) -> list[Source]:
    sources: list[Source] = []
    seen_urls: set[str] = set()
    for index, raw in enumerate(raw_sources, start=1):
        url = _clean_string(raw.get("url"))
        if not url or not url.startswith(("http://", "https://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        title = _clean_string(raw.get("title")) or url
        text = _clean_string(raw.get("text")) or _fallback_source_text(raw, response_text)
        fetched = _fetch_source_page(url) if _needs_page_fetch(title, text, url) else {}
        if fetched.get("title") and (not title or title == url or title.startswith("http")):
            title = fetched["title"]
        if fetched.get("text") and _needs_page_fetch(title, text, url):
            text = fetched["text"]
        source_type = _normalize_source_type(raw.get("source_type"), {**raw, "title": title, "text": text})
        try:
            source = Source(
                id=f"web_{len(sources) + 1:03d}",
                title=title[:240],
                source_type=source_type,
                url=url,
                citation=_clean_string(raw.get("citation")) or title[:120],
                published_year=_published_year(raw.get("published_year")),
                tags=_normalize_tags(raw.get("tags"), title, text),
                evidence_scope=(
                    _clean_string(raw.get("evidence_scope"))
                    or "Live web search source normalized for Pragmatic."
                )[:300],
                text=text[:1400],
            )
        except ValueError:
            continue
        sources.append(source)
        if len(sources) >= max_sources:
            break
    return sources


def _fallback_source_text(raw: dict[str, Any], response_text: str) -> str:
    fields = [
        raw.get("summary"),
        raw.get("relevant_findings"),
        raw.get("quoted_evidence"),
        raw.get("limitations"),
        response_text,
    ]
    return " ".join(_clean_string(field) for field in fields if _clean_string(field))[:1200]


def _needs_page_fetch(title: str, text: str, url: str) -> bool:
    stripped = text.strip()
    return (
        not stripped
        or len(stripped) < 120
        or stripped.startswith(("{", "["))
        or title == url
        or title.startswith("http")
    )


def _fetch_source_page(url: str) -> dict[str, str]:
    try:
        request = Request(
            url,
            headers={
                "User-Agent": "Pragmatic/0.1 evidence-source-fetcher",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        with urlopen(request, timeout=8) as response:
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return {}
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(600_000).decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return {}

    title = _html_title(raw)
    text = _html_text(raw)
    return {
        "title": title,
        "text": text[:5000],
    }


def _html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return ""
    return _clean_string(unescape(_strip_tags(match.group(1))))


def _html_text(html: str) -> str:
    value = re.sub(r"(?is)<(script|style|noscript|svg|math).*?</\1>", " ", html)
    value = re.sub(r"(?is)<(nav|footer|header|form).*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", ". ", value)
    value = re.sub(r"(?i)</(p|div|section|article|li|h[1-6])>", ". ", value)
    value = _strip_tags(value)
    value = unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def _normalize_source_type(value: Any, raw: dict[str, Any]) -> str:
    text = f"{value or ''} {raw.get('title') or ''} {raw.get('url') or ''}".lower()
    aliases = {
        "journal": "paper",
        "academic": "paper",
        "preprint": "paper",
        "article": "paper",
        "review": "review",
        "standard": "standard",
        "nij": "standard",
        "astm": "standard",
        "pmc": "paper",
        "pubmed": "paper",
        "ncbi": "paper",
        "government": "government",
        ".gov": "government",
        "dataset": "dataset",
        "database": "dataset",
        "case": "case_study",
        "company": "company_claim",
        "manufacturer": "company_claim",
        "blog": "blog_post",
        "news": "news",
    }
    for needle, source_type in aliases.items():
        if needle in text:
            return source_type
    if isinstance(value, str) and value in ALLOWED_SOURCE_TYPES:
        return value
    return "unknown"


def _published_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    if isinstance(value, str):
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def _normalize_tags(value: Any, title: str, text: str) -> list[str]:
    tags: list[str] = []
    if isinstance(value, list):
        tags.extend(_slug(tag) for tag in value if _clean_string(tag))
    elif isinstance(value, str):
        tags.extend(_slug(part) for part in re.split(r"[,;]", value) if part.strip())
    inferred = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", f"{title} {text}".lower())
        if token not in {"the", "and", "with", "from", "that", "this", "source"}
    ]
    tags.extend(inferred[:8])
    deduped: list[str] = []
    for tag in tags:
        if tag and tag not in deduped:
            deduped.append(tag)
    return deduped[:10]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_json_payload(text: str) -> Any | None:
    candidates = [text]
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start: object_end + 1])
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(text[array_start: array_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _as_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _as_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return re.sub(r"\s+", " ", str(value)).strip()


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", _clean_string(value).lower()).strip("-")[:48]
