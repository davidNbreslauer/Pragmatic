from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from pragmatic import ResearchManager
from pragmatic.corpus import AI_SCIENTIST_CORPUS_PATH, SPIDER_SILK_CORPUS_PATH
from pragmatic.live_harness import run_live_harness_sync
from pragmatic.replay import run_replay_demo
from pragmatic.schemas import LiveRunResult, ResearchState


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_THESIS = "Spider silk for bullet proof vests"
MAX_STORED_EVENTS_PER_JOB = 2000
RUNS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()
LOG = logging.getLogger("pragmatic.realtime")

# Realtime cockpit event taxonomy:
# - coarse events keep the stable {stage,status,message,metadata,index} envelope.
# - rich events add top-level kind plus metadata payloads:
#   reasoning.delta, tool.call, tool.output, fanout.spawn, fanout.task,
#   node.add, edge.add, node.confidence, counter.


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_job(
    thesis_text: str,
    config: dict[str, Any],
    *,
    event_loop: asyncio.AbstractEventLoop | None = None,
    event_signal: asyncio.Event | None = None,
) -> str:
    job_id = f"run_{uuid.uuid4().hex[:12]}"
    with RUN_LOCK:
        RUNS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "thesis_text": thesis_text,
            "config": config,
            "events": [],
            "next_event_index": 1,
            "event_loop": event_loop,
            "event_signal": event_signal,
            "result": None,
            "error": None,
        }
    return job_id


def _signal_job_unlocked(job: dict[str, Any]) -> None:
    loop = job.get("event_loop")
    signal = job.get("event_signal")
    if loop is not None and signal is not None:
        loop.call_soon_threadsafe(signal.set)


def _append_event(job_id: str, event: dict[str, Any]) -> None:
    normalized = {
        "id": f"event_{uuid.uuid4().hex[:10]}",
        "created_at": _now(),
        "stage": event.get("stage", "run"),
        "status": event.get("status", "running"),
        "message": event.get("message", ""),
        "kind": event.get("kind") or (event.get("metadata") or {}).get("kind"),
        "metadata": event.get("metadata") or {},
    }
    with RUN_LOCK:
        job = RUNS[job_id]
        if job["status"] in {"succeeded", "failed"}:
            return
        normalized["index"] = job["next_event_index"]
        job["next_event_index"] += 1
        job["events"].append(normalized)
        if len(job["events"]) > MAX_STORED_EVENTS_PER_JOB:
            job["events"] = job["events"][-MAX_STORED_EVENTS_PER_JOB:]
        job["updated_at"] = normalized["created_at"]
        _signal_job_unlocked(job)


def _set_job_status(
    job_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with RUN_LOCK:
        job = RUNS[job_id]
        job["status"] = status
        job["updated_at"] = _now()
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        _signal_job_unlocked(job)


async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


async def start_run(request: Request) -> JSONResponse:
    payload = await request.json()
    thesis_text = str(payload.get("thesis_text") or DEFAULT_THESIS).strip() or DEFAULT_THESIS
    config = _normalize_config(payload.get("config") or {})
    job_id = _new_job(
        thesis_text,
        config,
        event_loop=asyncio.get_running_loop(),
        event_signal=asyncio.Event(),
    )
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, thesis_text, config),
        name=f"pragmatic-{job_id}",
        daemon=True,
    )
    thread.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})


async def get_run(request: Request) -> JSONResponse:
    job_id = request.path_params["job_id"]
    with RUN_LOCK:
        job = RUNS.get(job_id)
        if job is None:
            return JSONResponse({"error": "run not found"}, status_code=404)
        return JSONResponse(_public_job(job))


async def run_events(request: Request) -> StreamingResponse:
    job_id = request.path_params["job_id"]

    async def event_stream():
        last_sent_index = 0
        while True:
            with RUN_LOCK:
                job = RUNS.get(job_id)
                if job is None:
                    yield _sse("error", {"error": "run not found"})
                    return
                event_signal = job.get("event_signal")
            if event_signal is not None:
                event_signal.clear()

            with RUN_LOCK:
                job = RUNS.get(job_id)
                if job is None:
                    yield _sse("error", {"error": "run not found"})
                    return
                events = list(job["events"])
                status = job["status"]
                result = job["result"]
                error = job["error"]

            new_events = [event for event in events if event["index"] > last_sent_index]
            for event in new_events:
                yield _sse("progress", event)
                last_sent_index = event["index"]

            if status in {"succeeded", "failed"}:
                yield _sse(
                    "done",
                    {
                        "status": status,
                        "result": result,
                        "error": error,
                    },
                )
                return
            if await request.is_disconnected():
                return
            if event_signal is None:
                await asyncio.sleep(0.25)
                continue
            try:
                await asyncio.wait_for(event_signal.wait(), timeout=15)
            except TimeoutError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_job(job_id: str, thesis_text: str, config: dict[str, Any]) -> None:
    _set_job_status(job_id, "running")
    _append_event(
        job_id,
        {
            "stage": "input",
            "status": "created",
            "message": "Question received.",
            "metadata": {"thesis_text": thesis_text},
        },
    )
    started = time.monotonic()
    try:
        state: ResearchState
        live_result: LiveRunResult | None = None
        manager = ResearchManager(model=config["model"] or None)

        if config["replay_demo"]:
            _append_event(
                job_id,
                {
                    "stage": "replay",
                    "status": "running",
                    "message": "Running failure-to-eval replay.",
                    "metadata": {},
                },
            )
            replay = run_replay_demo(
                thesis_text,
                max_iterations=config["max_iterations"],
                corpus_path=config["corpus_path"] or None,
                execution_backend=config["execution_backend"],
                observability_mode=config["observability_mode"],
            )
            state = replay.replay_pass
        elif config["orchestration"] == "live_sdk":
            live_result = run_live_harness_sync(
                thesis_text,
                model=config["model"] or None,
                mode="dry_run" if config["live_dry_run"] else "live",
                allow_live_sdk=config["live_sdk_enabled"],
                max_turns=config["max_turns"],
                timeout_seconds=config["timeout_seconds"],
                max_iterations=config["max_iterations"],
                execution_backend=config["execution_backend"],
                source_mode=config["source_mode"],
                corpus_path=config["corpus_path"] or None,
                allow_live_web_search=config["allow_live_web_search"],
                web_search_model=config["web_search_model"] or config["model"] or None,
                max_web_sources=config["max_web_sources"],
                observability_mode=config["observability_mode"],
                require_demo_proof=config["require_demo_proof"],
                progress_callback=lambda event: _append_event(job_id, event),
            )
            if live_result.state is not None:
                state = live_result.state
            else:
                _append_event(
                    job_id,
                    {
                        "stage": "fallback",
                        "status": "running",
                        "message": "Live run did not return a state; building an inspectable fallback graph.",
                        "metadata": {"live_status": live_result.status},
                    },
                )
                fallback_source_mode = "prepared" if live_result.mode == "dry_run" else config["source_mode"]
                state = manager.run_deterministic(
                    thesis_text,
                    max_iterations=config["max_iterations"],
                    execution_backend=config["execution_backend"],
                    corpus_path=config["corpus_path"] or None,
                    source_mode=fallback_source_mode,
                    allow_live_web_search=(
                        config["allow_live_web_search"] and fallback_source_mode == "web"
                    ),
                    web_search_model=config["web_search_model"] or config["model"] or None,
                    max_web_sources=config["max_web_sources"],
                    observability_mode="off",
                )
        elif config["orchestration"] == "scripted_sdk":
            _append_event(
                job_id,
                {
                    "stage": "scripted_sdk",
                    "status": "running",
                    "message": "Running SDK-scripted specialist tools.",
                    "metadata": {},
                },
            )
            state = manager.run_sdk_orchestrated(
                thesis_text,
                max_iterations=config["max_iterations"],
                execution_backend=config["execution_backend"],
                corpus_path=config["corpus_path"] or None,
                source_mode=config["source_mode"],
                allow_live_web_search=config["allow_live_web_search"],
                web_search_model=config["web_search_model"] or config["model"] or None,
                max_web_sources=config["max_web_sources"],
                observability_mode=config["observability_mode"],
            )
        else:
            _append_event(
                job_id,
                {
                    "stage": "deterministic",
                    "status": "running",
                    "message": "Running deterministic research loop.",
                    "metadata": {},
                },
            )
            state = manager.run_deterministic(
                thesis_text,
                max_iterations=config["max_iterations"],
                execution_backend=config["execution_backend"],
                corpus_path=config["corpus_path"] or None,
                source_mode=config["source_mode"],
                allow_live_web_search=config["allow_live_web_search"],
                web_search_model=config["web_search_model"] or None,
                max_web_sources=config["max_web_sources"],
                observability_mode=config["observability_mode"],
            )

        _append_state_cockpit_events(job_id, state)
        summary = _summarize_state(state)
        _maybe_polish_bottom_line(summary, config)
        result = {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "summary": summary,
            "state": state.model_dump(mode="json"),
            "live_run": live_result.model_dump(mode="json") if live_result is not None else None,
        }
        _append_event(
            job_id,
            {
                "stage": "answer",
                "status": "succeeded",
                "message": "Answer ready.",
                "metadata": summary["counts"],
            },
        )
        _set_job_status(job_id, "succeeded", result=result)
    except Exception as exc:
        _append_event(
            job_id,
            {
                "stage": "error",
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
                "metadata": {},
            },
        )
        _set_job_status(job_id, "failed", error=f"{type(exc).__name__}: {exc}")


def _append_state_cockpit_events(job_id: str, state: ResearchState) -> None:
    """Emit a final graph snapshot for deterministic/scripted/recovered runs."""

    seen_nodes: set[str] = set()

    def add_node(node_id: str, node_kind: str, label: str, confidence: float | None = None) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        metadata: dict[str, Any] = {
            "id": node_id,
            "node_kind": node_kind,
            "label": label[:140],
        }
        if confidence is not None:
            metadata["confidence"] = confidence
        _append_event(
            job_id,
            {
                "stage": "graph.snapshot",
                "status": "created",
                "message": f"Added {node_kind} node.",
                "kind": "node.add",
                "metadata": metadata,
            },
        )

    def add_edge(from_id: str, to_id: str, relation: str) -> None:
        _append_event(
            job_id,
            {
                "stage": "graph.snapshot",
                "status": "created",
                "message": f"Linked {from_id} to {to_id}.",
                "kind": "edge.add",
                "metadata": {"from": from_id, "to": to_id, "relation": relation},
            },
        )

    for assumption in state.assumptions:
        add_node(assumption.id, "assumption", assumption.text, assumption.confidence)
    for source in state.sources:
        add_node(source.id, "source", source.title)
    for item in state.evidence_items:
        add_node(item.id, "evidence", item.claim_supported, item.confidence)
        add_edge(item.source_id, item.id, "contradicts" if item.evidence_type == "contradictory" else "supports")
        for assumption_id in item.assumption_ids:
            relation = "proxy_only" if item.evidence_type in {"proxy", "indirect"} else "supports"
            if item.evidence_type == "contradictory":
                relation = "contradicts"
            add_edge(item.id, assumption_id, relation)
    for update in state.belief_updates:
        _append_event(
            job_id,
            {
                "stage": "graph.snapshot",
                "status": "updated",
                "message": f"Updated confidence for {update.assumption_id}.",
                "kind": "node.confidence",
                "metadata": {
                    "id": update.assumption_id,
                    "from": update.previous_confidence,
                    "to": update.new_confidence,
                    "reason": update.rationale,
                },
            },
        )
    for test in state.decisive_tests:
        add_node(test.id, "test", test.test)
        for assumption_id in test.would_resolve:
            add_edge(assumption_id, test.id, "tests")
    for generated_eval in state.generated_evals:
        add_node(generated_eval.id, "eval", generated_eval.eval_rule)
        if generated_eval.source_failure_id:
            add_edge(generated_eval.source_failure_id, generated_eval.id, "becomes_eval")
    _append_event(
        job_id,
        {
            "stage": "graph.snapshot",
            "status": "updated",
            "message": "Updated research counters.",
            "kind": "counter",
            "metadata": {
                "sources": len(state.sources),
                "evidence": len(state.evidence_items),
                "leaps": len(state.invalid_leaps),
                "conflicts": len(state.evidence_conflicts),
                "tests": len(state.decisive_tests),
            },
        },
    )


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    corpus_choice = str(raw.get("corpus_choice") or "auto")
    corpus_path = ""
    if corpus_choice == "spider_silk":
        corpus_path = str(SPIDER_SILK_CORPUS_PATH)
    elif corpus_choice == "ai_scientist":
        corpus_path = str(AI_SCIENTIST_CORPUS_PATH)
    return {
        "orchestration": str(raw.get("orchestration") or "live_sdk"),
        "execution_backend": str(raw.get("execution_backend") or "modal"),
        "observability_mode": str(raw.get("observability_mode") or "local"),
        "source_mode": str(raw.get("source_mode") or "web"),
        "corpus_choice": corpus_choice,
        "corpus_path": corpus_path,
        "allow_live_web_search": bool(raw.get("allow_live_web_search", True)),
        "live_sdk_enabled": bool(raw.get("live_sdk_enabled", True)),
        "live_dry_run": bool(raw.get("live_dry_run", False)),
        "require_demo_proof": bool(raw.get("require_demo_proof", True)),
        "replay_demo": bool(raw.get("replay_demo", False)),
        "model": str(raw.get("model") or DEFAULT_MODEL),
        "web_search_model": str(raw.get("web_search_model") or DEFAULT_MODEL),
        "max_iterations": int(raw.get("max_iterations") or 1),
        "max_turns": int(raw.get("max_turns") or 3),
        "timeout_seconds": float(raw.get("timeout_seconds") or 300),
        "max_web_sources": int(raw.get("max_web_sources") or 8),
    }


def _summarize_state(state: ResearchState) -> dict[str, Any]:
    direct_count = len([item for item in state.evidence_items if item.evidence_type == "direct"])
    limiting_count = len(
        [item for item in state.evidence_items if item.evidence_type == "contradictory"]
    )
    unresolved = [
        assumption
        for assumption in state.assumptions
        if assumption.support_level in {"unknown", "unsupported", "contradicted", "weak"}
    ]
    if state.invalid_leaps or limiting_count or unresolved:
        verdict = "Not proven yet"
        headline = "The run found evidence, but not enough for a confident yes."
    elif direct_count:
        verdict = "Supported with limits"
        headline = "The evidence supports the claim, with stated limits."
    else:
        verdict = "Promising but indirect"
        headline = "The evidence is relevant, but mostly indirect."

    source_titles = {source.id: source.title for source in state.sources}
    evidence = [
        {
            "type": item.evidence_type,
            "confidence": item.confidence,
            "source": source_titles.get(item.source_id, item.source_id),
            "claim": item.claim_supported,
            "limitation": item.limitation,
        }
        for item in state.evidence_items[:8]
    ]
    gaps = [
        {
            "kind": "Invalid leap",
            "summary": leap.leap,
            "detail": leap.why_invalid,
            "next": leap.suggested_followup_question,
        }
        for leap in state.invalid_leaps[:4]
    ]
    gaps.extend(
        {
            "kind": assumption.support_level,
            "summary": assumption.text,
            "detail": assumption.latest_update or assumption.why_it_matters,
            "next": assumption.why_it_matters,
        }
        for assumption in unresolved[: max(0, 6 - len(gaps))]
    )
    decisive = state.decisive_tests[0].model_dump(mode="json") if state.decisive_tests else None
    mean_confidence = (
        sum(assumption.confidence for assumption in state.assumptions) / len(state.assumptions)
        if state.assumptions
        else 0.0
    )
    return {
        "verdict": verdict,
        "headline": headline,
        "mean_confidence": round(mean_confidence, 3),
        "counts": {
            "assumptions": len(state.assumptions),
            "sources": len(state.sources),
            "evidence": len(state.evidence_items),
            "invalid_leaps": len(state.invalid_leaps),
            "generated_evals": len(state.generated_evals),
            "modal_tasks": len(
                [result for result in state.research_task_results if result.backend == "modal"]
            ),
        },
        "bottom_line": _build_bottom_line(state),
        "evidence": evidence,
        "gaps": gaps,
        "decisive_test": decisive,
        "trace_id": state.observability.trace_id if state.observability is not None else None,
        "trace_path": state.observability.trace_path if state.observability is not None else None,
        "workshop_path": (
            state.observability.workshop_path if state.observability is not None else None
        ),
    }


def _build_bottom_line(state: ResearchState) -> dict[str, Any]:
    direct_items = [item for item in state.evidence_items if item.evidence_type == "direct"]
    limiting_items = [item for item in state.evidence_items if item.evidence_type == "contradictory"]
    unresolved = [
        assumption
        for assumption in state.assumptions
        if assumption.support_level in {"unknown", "unsupported", "contradicted", "weak"}
    ]
    if state.invalid_leaps or limiting_items or unresolved:
        verdict = "Not proven yet"
    elif direct_items:
        verdict = "Supported with limits"
    else:
        verdict = "Promising but indirect"

    mean_confidence = (
        sum(assumption.confidence for assumption in state.assumptions) / len(state.assumptions)
        if state.assumptions
        else 0.0
    )
    if mean_confidence < 0.34:
        confidence_label = "Low"
        confidence_band = "low"
    elif mean_confidence <= 0.66:
        confidence_label = "Moderate"
        confidence_band = "mid"
    else:
        confidence_label = "High"
        confidence_band = "high"

    source_titles = {source.id: source.title for source in state.sources}
    counts = {
        "sources": len(state.sources),
        "evidence": len(state.evidence_items),
        "invalid_leaps": len(state.invalid_leaps),
        "generated_evals": len(state.generated_evals),
        "modal_tasks": len([result for result in state.research_task_results if result.backend == "modal"]),
    }
    one_liner = (
        f"{state.thesis.text}: {verdict.lower()} — "
        f"{len(direct_items)} direct evidence item(s), "
        f"{len(limiting_items)} limiting item(s), "
        f"{len(state.invalid_leaps)} unsupported leap(s)."
    )

    because: list[str] = []
    strongest_direct = max(direct_items, key=lambda item: item.confidence, default=None)
    strongest_support = max(
        [item for item in state.evidence_items if item.evidence_type != "contradictory"],
        key=lambda item: item.confidence,
        default=None,
    )
    if strongest_direct is not None:
        because.append(
            f"{strongest_direct.claim_supported} ({source_titles.get(strongest_direct.source_id, strongest_direct.source_id)})"
        )
    elif strongest_support is not None:
        because.append(
            f"{strongest_support.claim_supported} ({source_titles.get(strongest_support.source_id, strongest_support.source_id)})"
        )
    top_limiter = max(limiting_items, key=lambda item: item.confidence, default=None)
    if top_limiter is not None:
        because.append(f"Limiter: {top_limiter.claim_supported}")
    elif state.invalid_leaps:
        because.append(f"Limiter: {state.invalid_leaps[0].leap}")
    if not because:
        because.append("The run found relevant evidence but no decisive direct proof.")

    if state.invalid_leaps:
        leap = state.invalid_leaps[0]
        biggest_risk = {"label": "Unsupported leap", "text": f"{leap.leap}: {leap.why_invalid}"}
    else:
        weak_assumption = min(unresolved, key=lambda assumption: assumption.confidence, default=None)
        biggest_risk = {
            "label": weak_assumption.support_level if weak_assumption is not None else "Residual uncertainty",
            "text": (
                f"{weak_assumption.text}: {weak_assumption.latest_update or weak_assumption.why_it_matters}"
                if weak_assumption is not None
                else "No major caveat surfaced beyond normal evidence limits."
            ),
        }

    if state.decisive_tests:
        test = state.decisive_tests[0]
        criterion = test.success_criteria[0] if test.success_criteria else test.why_decisive
        decisive_next_test = f"{test.test} Success means: {criterion}"
    else:
        decisive_next_test = "No decisive next test was generated."

    stat_line = (
        f"{counts['sources']} sources · {counts['evidence']} evidence · "
        f"{counts['invalid_leaps']} leaps · {counts['generated_evals']} evals · "
        f"{counts['modal_tasks']} Modal tasks"
    )
    return {
        "verdict": verdict,
        "confidence": round(mean_confidence, 3),
        "confidence_label": confidence_label,
        "confidence_band": confidence_band,
        "one_liner": one_liner,
        "one_liner_source": "deterministic",
        "because": because[:3],
        "biggest_risk": biggest_risk,
        "decisive_next_test": decisive_next_test,
        "stat_line": stat_line,
        "counts": counts,
    }


def _maybe_polish_bottom_line(summary: dict[str, Any], config: dict[str, Any]) -> None:
    if config.get("orchestration") not in {"live_sdk", "scripted_sdk"}:
        return
    if not os.getenv("OPENAI_API_KEY"):
        return
    bottom_line = summary.get("bottom_line")
    if not isinstance(bottom_line, dict):
        return
    try:
        from openai import OpenAI

        risk = bottom_line.get("biggest_risk") or {}
        because = "; ".join(str(item) for item in bottom_line.get("because", [])[:3])
        prompt = (
            "Bottom line this research result in one decisive plain-English sentence. "
            "Maximum 30 words. Do not add facts. Avoid hype.\n\n"
            f"Verdict: {bottom_line.get('verdict')}\n"
            f"Confidence: {bottom_line.get('confidence_label')} {bottom_line.get('confidence')}\n"
            f"Evidence: {because}\n"
            f"Risk: {risk.get('label', '')}: {risk.get('text', '')}\n"
            f"Draft: {bottom_line.get('one_liner')}"
        )
        response = OpenAI(timeout=5.0).responses.create(
            model=str(config.get("model") or DEFAULT_MODEL),
            input=prompt,
            max_output_tokens=80,
        )
        polished = str(getattr(response, "output_text", "") or "").strip().strip('"')
        if polished:
            bottom_line["one_liner"] = " ".join(polished.split())
            bottom_line["one_liner_source"] = "model"
    except Exception as exc:
        LOG.info("Bottom-line model polish skipped: %s: %s", type(exc).__name__, exc)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "thesis_text": job["thesis_text"],
        "config": job["config"],
        "events": job["events"],
        "result": job["result"],
        "error": job["error"],
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _json_for_script(value: Any) -> str:
    return html.escape(json.dumps(value), quote=False)


def _maybe_start_modal_prewarm() -> None:
    if os.getenv("PRAGMATIC_PREWARM_MODAL", "").lower() not in {"1", "true", "yes"}:
        return

    def worker() -> None:
        try:
            from pragmatic.modal_jobs import prewarm_modal_functions

            result = prewarm_modal_functions()
            LOG.info("Modal prewarm finished: %s", result)
        except Exception as exc:  # pragma: no cover - depends on live Modal availability.
            LOG.warning("Modal prewarm skipped: %s: %s", type(exc).__name__, exc)

    threading.Thread(target=worker, name="pragmatic-modal-prewarm", daemon=True).start()


@asynccontextmanager
async def _lifespan(app: Starlette):
    del app
    _maybe_start_modal_prewarm()
    yield


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pragmatic AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --canvas: #070A0F;
      --canvas-2: #0B1018;
      --panel: rgba(255,255,255,0.04);
      --panel-strong: rgba(255,255,255,0.07);
      --line: rgba(255,255,255,0.08);
      --line-strong: rgba(255,255,255,0.16);
      --text: #E8EDF4;
      --text-muted: #8A95A6;
      --text-faint: #5A6473;
      --accent: #5EE6C9;
      --accent-2: #6AA8FF;
      --accent-low: #FF5C7A;
      --accent-mid: #F4C152;
      --accent-warn: #FF5C7A;
      --shadow: 0 20px 60px -20px rgba(0,0,0,0.7);
      --radius-panel: 16px;
      --radius-control: 10px;
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
      --font-ui: "Space Grotesk", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    html { background: var(--canvas); }
    body {
      margin: 0;
      overflow: hidden;
      background:
        radial-gradient(circle at 50% -10%, rgba(94,230,201,.14), transparent 34%),
        radial-gradient(circle at 78% 18%, rgba(106,168,255,.10), transparent 30%),
        linear-gradient(145deg, var(--canvas-2) 0%, var(--canvas) 55%, #04060A 100%);
      color: var(--text);
      font-family: var(--font-ui);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .025;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='.7'/%3E%3C/svg%3E");
      mix-blend-mode: screen;
    }
    main {
      height: 100vh;
      max-width: 1440px;
      margin: 0 auto;
      padding: 12px 18px 14px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 12px;
      overflow: hidden;
    }
    .topbar {
      min-height: 56px;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-panel);
      box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.06);
      backdrop-filter: blur(18px) saturate(140%);
      animation: riseIn .7s var(--ease) both;
    }
    .header-grid { display: grid; gap: 12px; }
    .identity-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .brand-block { min-width: 240px; }
    .brand {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      color: var(--text);
      font-size: 15px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .brand::before {
      content: "";
      width: 13px;
      height: 13px;
      border-radius: 999px;
      background: radial-gradient(circle, #fff 0 12%, var(--accent) 34%, rgba(94,230,201,.1) 72%);
      box-shadow: 0 0 18px rgba(94,230,201,.9), 0 0 42px rgba(94,230,201,.24);
    }
    .tagline { color: var(--text-muted); font-size: 12px; margin-top: 3px; }
    h2, h3, label, .phase, .badge, .chip, .metric .label, .counter .label {
      text-transform: uppercase;
      letter-spacing: .12em;
    }
    h2 { margin: 0; font-size: 11px; color: var(--text); font-weight: 700; }
    h3 { margin: 0 0 8px; font-size: 11px; color: var(--text-muted); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-panel);
      box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.06);
      backdrop-filter: blur(18px) saturate(140%);
    }
    .ask { position: relative; overflow: hidden; }
    .ask::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(94,230,201,.75), rgba(106,168,255,.4), transparent);
    }
    label { display: block; font-weight: 700; margin-bottom: 5px; font-size: 10px; color: var(--text-faint); }
    .question-row { display: flex; align-items: flex-end; gap: 12px; }
    .question-field { flex: 1 1 auto; min-width: 0; }
    .question-actions {
      display: flex;
      flex: 0 0 auto;
      gap: 8px;
      align-items: flex-end;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .button-group { display: flex; gap: 8px; align-items: center; }
    .run-indicator {
      display: none;
      align-items: center;
      min-height: 36px;
      border: 1px solid rgba(94,230,201,.18);
      border-radius: 999px;
      padding: 0 10px;
      background: rgba(94,230,201,.06);
      color: var(--text-muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 0 18px rgba(94,230,201,.08);
    }
    .run-indicator.visible { display: inline-flex; }
    textarea {
      display: block;
      width: 100%;
      height: 52px;
      min-height: 52px;
      max-height: 120px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: var(--radius-control);
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      background: rgba(255,255,255,.045);
      color: var(--text);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
      outline: none;
      transition: border-color .2s ease, box-shadow .2s ease, background .2s ease;
    }
    textarea:focus, select:focus, input:focus {
      border-color: rgba(94,230,201,.65);
      box-shadow: 0 0 0 3px rgba(94,230,201,.11), 0 0 24px rgba(94,230,201,.12);
    }
    button {
      border: 1px solid var(--line);
      border-radius: var(--radius-control);
      padding: 9px 12px;
      height: 36px;
      background: rgba(255,255,255,.045);
      color: var(--text);
      font-family: var(--font-ui);
      font-weight: 700;
      cursor: pointer;
      transition: transform .16s var(--ease), border-color .2s ease, box-shadow .2s ease, background .2s ease;
    }
    button:hover { border-color: rgba(94,230,201,.35); background: rgba(255,255,255,.075); }
    button:active { transform: translateY(1px) scale(.985); }
    button.primary {
      min-width: 92px;
      border-color: rgba(94,230,201,.7);
      color: #04100D;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: inset 0 1px 0 rgba(255,255,255,.45), 0 0 28px rgba(94,230,201,.22);
    }
    button.primary:hover { box-shadow: inset 0 1px 0 rgba(255,255,255,.5), 0 0 38px rgba(94,230,201,.34); }
    .answer-close { min-width: 0; padding: 5px 9px; font-size: 13px; }
    button:disabled { opacity: .55; cursor: wait; }
    select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-control);
      padding: 7px;
      background: rgba(255,255,255,.05);
      color: var(--text);
      font: inherit;
      font-size: 12px;
      outline: none;
    }
    details { margin-top: 7px; color: var(--text-muted); font-size: 12px; }
    details .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-end; }
    .chip, .badge {
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0 10px;
      background: rgba(255,255,255,.045);
      color: var(--text-muted);
      font-size: 10px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    }
    .chip { white-space: nowrap; }
    #modeBadge { justify-content: center; max-width: 340px; text-align: right; }
    .chip::before {
      content: "";
      display: inline-block;
      width: 6px;
      height: 6px;
      margin-right: 7px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 12px rgba(94,230,201,.8);
      vertical-align: 1px;
    }
    .phase-ribbon {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0;
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,.025);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
      animation: riseIn .7s .08s var(--ease) both;
    }
    .phase {
      position: relative;
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--text-faint);
      padding: 8px;
      min-height: 30px;
      font-size: 10px;
      font-weight: 700;
      text-align: center;
      transition: color .28s ease, background .28s ease;
    }
    .phase:last-child { border-right: 0; }
    .phase::after {
      content: "";
      position: absolute;
      inset: auto 12px 5px;
      height: 2px;
      border-radius: 999px;
      background: transparent;
      transform: scaleX(0);
      transform-origin: left;
    }
    .phase.active { color: var(--text); background: rgba(94,230,201,.055); }
    .phase.active::after { background: var(--accent); animation: sweep .6s var(--ease) both, pulseLine 1.5s ease-in-out infinite; }
    .phase.done { color: rgba(94,230,201,.95); background: rgba(94,230,201,.085); }
    .phase.done::after { background: var(--accent); transform: scaleX(1); box-shadow: 0 0 14px rgba(94,230,201,.45); }
    .cockpit {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(250px, .78fr) minmax(390px, 1.28fr) minmax(270px, .82fr);
      gap: 12px;
    }
    .pane {
      min-height: 0;
      padding: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: riseIn .7s var(--ease) both;
    }
    .pane:nth-child(1) { animation-delay: .12s; }
    .pane:nth-child(2) { animation-delay: .18s; }
    .pane:nth-child(3) { animation-delay: .24s; }
    .pane-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; flex: 0 0 auto; }
    .thinking {
      flex: 1;
      min-height: 0;
      overflow: hidden;
      border: 1px solid rgba(94,230,201,.10);
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(3,8,12,.95), rgba(5,10,15,.88));
      color: var(--text-muted);
      padding: 10px;
      mask-image: linear-gradient(to bottom, transparent 0, #000 14px, #000 100%);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.025), inset 0 0 40px rgba(94,230,201,.025);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .thought-stream {
      flex: 1;
      min-height: 0;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 7px;
      padding: 4px 2px 2px;
    }
    .thought-card {
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr) 8px;
      gap: 8px;
      align-items: center;
      border: 1px solid rgba(255,255,255,.075);
      border-radius: 10px;
      padding: 8px 9px;
      min-height: 42px;
      background: rgba(255,255,255,.035);
      color: var(--text-muted);
      font: 12px/1.25 var(--font-ui);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
      animation: tileIn .28s var(--ease) both;
      transition: border-color .18s ease, background .18s ease, box-shadow .18s ease;
    }
    .thought-card:hover, .thought-card.active {
      border-color: rgba(94,230,201,.32);
      background: rgba(94,230,201,.065);
      box-shadow: 0 0 24px rgba(94,230,201,.08), inset 0 1px 0 rgba(255,255,255,.05);
    }
    .thought-card .icon {
      width: 22px;
      height: 22px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(94,230,201,.18);
      border-radius: 999px;
      color: var(--accent);
      background: rgba(94,230,201,.07);
      font: 13px var(--font-mono);
    }
    .thought-card .label {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      white-space: normal;
      color: var(--text);
      font-weight: 650;
      line-height: 1.2;
    }
    .thought-card .sub {
      margin-top: 2px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text-faint);
      font: 10px var(--font-mono);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 12px rgba(94,230,201,.7);
    }
    .thought-card.failed .status-dot { background: var(--accent-low); box-shadow: 0 0 12px rgba(255,92,122,.7); }
    .thought-card.updated .status-dot { background: var(--accent-mid); box-shadow: 0 0 12px rgba(244,193,82,.58); }
    .raw-reasoning {
      flex: 0 0 auto;
      margin-top: 0;
      border-top: 1px solid rgba(255,255,255,.06);
      padding-top: 7px;
      color: var(--text-faint);
    }
    .raw-reasoning summary {
      cursor: pointer;
      color: var(--text-muted);
      font: 10px var(--font-ui);
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .raw-reasoning-body {
      max-height: 120px;
      margin-top: 8px;
      overflow: auto;
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 10px;
      padding: 9px;
      background: rgba(0,0,0,.24);
      color: var(--text-muted);
      font: 12px/1.5 var(--font-mono);
      white-space: pre-wrap;
    }
    .tool-chip {
      display: inline-block;
      margin: 2px 4px 2px 0;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid rgba(94,230,201,.18);
      background: rgba(94,230,201,.07);
      color: var(--text);
      font: 11px var(--font-ui);
      box-shadow: 0 0 16px rgba(94,230,201,.09);
    }
    .tool-chip::before { content: "▸"; color: var(--accent); margin-right: 5px; }
    .cursor { display: inline-block; width: 8px; height: 15px; background: var(--accent); box-shadow: 0 0 18px rgba(94,230,201,.75); animation: blink 1s steps(2) infinite; vertical-align: -2px; }
    @keyframes blink { 50% { opacity: 0; } }
    .graph-wrap {
      position: relative;
      flex: 1;
      min-height: 0;
      border: 1px solid rgba(255,255,255,.07);
      border-radius: 14px;
      overflow: hidden;
      background:
        radial-gradient(circle at 50% 48%, rgba(94,230,201,.08), transparent 36%),
        radial-gradient(circle at 50% 50%, transparent 0 58%, rgba(0,0,0,.35) 100%),
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px),
        rgba(4,8,12,.72);
      background-size: auto, auto, 32px 32px, 32px 32px, auto;
    }
    #beliefGraph { width: 100%; height: 100%; display: block; }
    .link { fill: none; stroke: rgba(94,230,201,.42); stroke-opacity: .38; stroke-width: 1.3px; transition: opacity .22s ease, stroke .25s ease, d .4s var(--ease); }
    .link.supports { stroke: var(--accent); }
    .link.contradicts { stroke: var(--accent-low); stroke-dasharray: 7 5; stroke-opacity: .62; }
    .link.proxy_only { stroke: var(--accent-mid); stroke-dasharray: 1 6; stroke-linecap: round; stroke-opacity: .72; }
    .link.tests, .link.becomes_eval { stroke: var(--accent-2); }
    .link.dim { opacity: .08; }
    .link.focus { opacity: .95; stroke-width: 2px; }
    .node { opacity: 0; cursor: pointer; animation: nodeIn .35s var(--ease) forwards; transition: opacity .22s ease, transform .4s var(--ease); }
    .node .shape { stroke: rgba(255,255,255,.82); stroke-width: 1.4px; transition: fill .35s ease, filter .35s ease, stroke .25s ease, r .25s ease; }
    .node text { font-size: 10px; font-weight: 650; pointer-events: none; fill: rgba(232,237,244,.84); paint-order: stroke; stroke: rgba(7,10,15,.88); stroke-width: 3px; }
    .node .badge { fill: rgba(4,8,12,.88); stroke: rgba(255,255,255,.15); stroke-width: 1px; }
    .node .badge-text { font: 9px var(--font-mono); fill: var(--text); stroke: none; }
    .node.dim { opacity: .18; }
    .node.focus .shape, .node.highlight .shape { stroke: var(--accent); stroke-width: 3px; filter: drop-shadow(0 0 8px var(--accent)) drop-shadow(0 0 24px var(--accent)) !important; }
    .node.pulse .shape { animation: nodePulse .72s var(--ease); stroke: var(--accent-warn); }
    @keyframes nodePulse { 0% { stroke-width: 9px; filter: drop-shadow(0 0 4px var(--accent-warn)) drop-shadow(0 0 30px var(--accent-warn)); } 100% { stroke-width: 1.4px; } }
    .graph-detail {
      position: absolute;
      left: 12px;
      bottom: 12px;
      width: min(290px, calc(100% - 24px));
      max-height: min(42%, 260px);
      overflow: auto;
      display: none;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 12px;
      padding: 10px;
      background: rgba(8,13,20,.82);
      backdrop-filter: blur(18px) saturate(140%);
      box-shadow: var(--shadow), 0 0 34px rgba(94,230,201,.08);
      color: var(--text-muted);
      font-size: 11px;
      z-index: 3;
    }
    .graph-detail.visible { display: block; animation: tileIn .24s var(--ease) both; }
    .graph-detail h3 { margin: 0 0 6px; color: var(--text); font-size: 13px; line-height: 1.25; }
    .graph-detail .meta { color: var(--text-faint); font: 10px var(--font-mono); margin-bottom: 8px; }
    .graph-detail .claim { border-top: 1px solid rgba(255,255,255,.08); padding-top: 7px; margin-top: 7px; line-height: 1.35; }
    .graph-detail .pill { display: inline-flex; margin: 2px 4px 2px 0; padding: 2px 6px; border-radius: 999px; border: 1px solid rgba(255,255,255,.12); color: var(--text); font: 10px var(--font-mono); }
    .toast {
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      display: none;
      border: 1px solid rgba(255,92,122,.28);
      background: rgba(28,9,16,.82);
      color: var(--text);
      border-radius: 12px;
      padding: 11px 13px;
      font-weight: 700;
      box-shadow: 0 12px 40px rgba(255,92,122,.16), var(--shadow);
      backdrop-filter: blur(18px) saturate(140%);
      transform: translateY(12px);
      opacity: 0;
    }
    .toast.visible { display: block; animation: toastIn .6s var(--ease) forwards; }
    .workers { flex: 0 0 34%; min-height: 100px; overflow: auto; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; align-content: start; }
    .worker {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
      min-height: 58px;
      background: rgba(255,255,255,.04);
      color: var(--text-muted);
      font-size: 11px;
      animation: tileIn .42s var(--ease) both;
      position: relative;
      overflow: hidden;
    }
    .worker.running { border-color: rgba(94,230,201,.24); background: rgba(94,230,201,.055); }
    .worker.running::after { content: ""; position: absolute; inset: 0; transform: translateX(-100%); background: linear-gradient(90deg, transparent, rgba(94,230,201,.08), transparent); animation: shimmer 1.4s ease-in-out infinite; }
    .worker.done { border-color: rgba(94,230,201,.35); background: rgba(94,230,201,.08); color: var(--text); box-shadow: 0 0 24px rgba(94,230,201,.08); }
    .worker.done strong::before { content: "✓"; color: var(--accent); margin-right: 6px; }
    .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid rgba(94,230,201,.18); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; margin-right: 5px; box-shadow: 0 0 14px rgba(94,230,201,.28); }
    @keyframes spin { to { transform: rotate(360deg); } }
    .counters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin-top: 8px; flex: 0 0 auto; }
    .counter { border: 1px solid var(--line); border-radius: 12px; padding: 8px; background: rgba(255,255,255,.035); transition: border-color .28s ease, box-shadow .28s ease; }
    .counter .label { font-size: 9px; color: var(--text-faint); }
    .counter .value { font-variant-numeric: tabular-nums; font-size: 22px; font-weight: 700; margin-top: 1px; color: var(--text); font-family: var(--font-mono); }
    .counter.bump { border-color: rgba(94,230,201,.5); box-shadow: 0 0 24px rgba(94,230,201,.14); }
    .event-log { flex: 1; min-height: 0; overflow: auto; display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
    .event { border: 1px solid var(--line); border-radius: 10px; padding: 7px; background: rgba(255,255,255,.03); font-size: 11px; color: var(--text-muted); animation: tileIn .28s var(--ease) both; }
    .event strong { display: block; font-size: 11px; color: var(--text); }
    .event span { color: var(--text-faint); }
    .answer {
      position: fixed;
      left: max(18px, calc((100vw - 1440px) / 2 + 18px));
      right: max(18px, calc((100vw - 1440px) / 2 + 18px));
      bottom: 14px;
      z-index: 20;
      max-height: min(58vh, 520px);
      padding: 14px;
      display: none;
      overflow: auto;
      background: rgba(9,14,22,.88);
      border-color: rgba(255,255,255,.14);
      box-shadow: 0 30px 90px rgba(0,0,0,.66), 0 0 80px rgba(94,230,201,.08);
      transform: translateY(24px) scale(.985);
      opacity: 0;
      transition: transform .6s var(--ease), opacity .45s ease;
    }
    .answer.visible { display: block; transform: translateY(0) scale(1); opacity: 1; }
    .answer-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .bottom-line { margin-top: 12px; padding: 14px; border: 1px solid var(--line); border-radius: 14px; background: linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.035)); box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 0 44px rgba(94,230,201,.06); }
    .bottom-top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .verdict-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 8px 12px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; animation: payoffPulse .72s var(--ease) both; }
    .verdict-pill.low { color: #19070B; background: var(--accent-low); box-shadow: 0 0 28px rgba(255,92,122,.26); }
    .verdict-pill.mid { color: #181000; background: var(--accent-mid); box-shadow: 0 0 28px rgba(244,193,82,.22); }
    .verdict-pill.high { color: #04100D; background: var(--accent); box-shadow: 0 0 28px rgba(94,230,201,.26); }
    .confidence-meter { display: inline-flex; align-items: center; gap: 8px; color: var(--text-muted); font: 12px var(--font-mono); }
    .confidence-meter::before { content: ""; width: 86px; height: 6px; border-radius: 999px; background: linear-gradient(90deg, var(--accent-low), var(--accent-mid), var(--accent)); box-shadow: 0 0 18px rgba(94,230,201,.12); }
    .ai-tag { border: 1px solid rgba(106,168,255,.22); border-radius: 999px; padding: 4px 7px; color: var(--accent-2); font-size: 10px; text-transform: uppercase; letter-spacing: .12em; background: rgba(106,168,255,.07); }
    .bottom-one-liner { margin: 12px 0; font-size: clamp(21px, 2.6vw, 34px); line-height: 1.08; color: var(--text); font-weight: 700; }
    .bottom-grid { display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 8px; }
    .bottom-row { border: 1px solid var(--line); border-radius: 12px; padding: 10px; background: rgba(255,255,255,.035); min-height: 88px; }
    .bottom-row .label { color: var(--text-faint); font-size: 10px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 6px; }
    .bottom-row ul { margin: 0; padding-left: 16px; color: var(--text-muted); }
    .bottom-row li { margin: 0 0 4px; }
    .bottom-risk, .bottom-test { color: var(--text-muted); line-height: 1.35; }
    .stat-line { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; color: var(--text-faint); font: 11px var(--font-mono); }
    .stat-line span { border: 1px solid var(--line); border-radius: 999px; padding: 4px 7px; background: rgba(255,255,255,.03); }
    .headline { font-size: clamp(22px, 3vw, 34px); line-height: 1.08; font-weight: 700; margin: 8px 0 10px; color: var(--text); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 12px; padding: 9px; background: rgba(255,255,255,.04); }
    .metric .label { color: var(--text-faint); font-size: 10px; }
    .metric .value { font-size: 18px; font-weight: 700; margin-top: 2px; color: var(--accent); font-variant-numeric: tabular-nums; }
    .tables { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--line); padding: 6px; text-align: left; vertical-align: top; color: var(--text-muted); }
    th { color: var(--text-faint); font-size: 10px; text-transform: uppercase; letter-spacing: .12em; }
    ::selection { background: rgba(94,230,201,.22); }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border: 3px solid transparent; background-clip: padding-box; border-radius: 999px; }
    ::-webkit-scrollbar-track { background: transparent; }
    @keyframes riseIn { from { opacity: 0; transform: translateY(14px) scale(.985); } to { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes tileIn { from { opacity: 0; transform: translateY(8px) scale(.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes nodeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes sweep { from { transform: scaleX(0); } to { transform: scaleX(1); } }
    @keyframes pulseLine { 0%, 100% { box-shadow: 0 0 8px rgba(94,230,201,.25); } 50% { box-shadow: 0 0 22px rgba(94,230,201,.75); } }
    @keyframes toastIn { to { opacity: 1; transform: translateY(0); } }
    @keyframes shimmer { to { transform: translateX(100%); } }
    @keyframes payoffPulse { 0% { opacity: 0; transform: scale(.88); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
    @media (max-width: 1050px) {
      main { overflow: auto; }
      body { overflow: auto; }
      .identity-row, .question-row { align-items: stretch; flex-direction: column; }
      .cockpit, .tables, .bottom-grid { grid-template-columns: 1fr; }
      .pane { min-height: 360px; }
      details .grid, .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chips, .question-actions { align-items: flex-start; justify-content: flex-start; }
    }
    @media (max-width: 650px) {
      details .grid, .metrics, .phase-ribbon { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <main>
    <section class="topbar">
      <section class="ask header-grid">
        <div class="identity-row">
          <div class="brand-block">
            <div class="brand">Pragmatic AI: Do diligence.</div>
            <div class="tagline">Sourced evidence, confidence updates, next decisive test.</div>
          </div>
          <div class="chips">
            <span class="chip">OpenAI Agents SDK</span>
            <span class="chip">Modal</span>
            <span class="chip">Raindrop</span>
          </div>
        </div>
        <div class="question-row">
          <div class="question-field">
            <label for="question">Question</label>
            <textarea id="question" rows="2">__DEFAULT_THESIS__</textarea>
          </div>
          <div class="question-actions">
            <div class="button-group">
              <button class="primary" id="ask">Ask Pragmatic</button>
              <button id="stop" disabled>Stop</button>
              <span class="run-indicator" id="runIndicator"><span class="spinner"></span>Running</span>
            </div>
            <span class="badge" id="modeBadge">live_sdk / modal / live web / local Workshop</span>
          </div>
        </div>
        <details>
          <summary>Run controls</summary>
          <div class="grid">
            <label>Orchestration<select id="orchestration"><option value="live_sdk">live_sdk</option><option value="scripted_sdk">scripted_sdk</option><option value="deterministic">deterministic</option></select></label>
            <label>Execution<select id="execution_backend"><option value="modal">modal</option><option value="local">local</option></select></label>
            <label>Sources<select id="source_mode"><option value="web">live web</option><option value="prepared">prepared</option></select></label>
            <label>Corpus<select id="corpus_choice"><option value="auto">auto</option><option value="spider_silk">Spider silk (prepared)</option><option value="ai_scientist">AI scientist (prepared)</option></select></label>
            <label>Model<input id="model" value="__DEFAULT_MODEL__" /></label>
            <label>Timeout seconds<input id="timeout_seconds" type="number" value="300" min="5" max="600" /></label>
            <label>Max turns<input id="max_turns" type="number" value="3" min="1" max="20" /></label>
            <label>Max web sources<input id="max_web_sources" type="number" value="8" min="1" max="20" /></label>
            <label>Full proof<select id="require_demo_proof"><option value="true">required</option><option value="false">not required</option></select></label>
          </div>
        </details>
      </section>
    </section>
    <section class="phase-ribbon" id="phases">
      <div class="phase" data-phase="decompose">Decompose</div>
      <div class="phase" data-phase="retrieve">Retrieve</div>
      <div class="phase" data-phase="extract">Extract</div>
      <div class="phase" data-phase="check">Cross-check</div>
      <div class="phase" data-phase="update">Update</div>
      <div class="phase" data-phase="test">Test</div>
    </section>
    <section class="cockpit">
      <aside class="panel pane">
        <div class="pane-head"><h2>Thinking</h2><span class="badge" id="reasoningStatus">waiting</span></div>
        <div class="thinking" id="thinking">
          <div class="thought-stream" id="thoughtStream"></div>
          <details class="raw-reasoning" id="rawReasoningWrap">
            <summary>Raw reasoning</summary>
            <div class="raw-reasoning-body" id="rawReasoning"><span class="cursor"></span></div>
          </details>
        </div>
      </aside>
      <section class="panel pane">
        <div class="pane-head"><h2>Belief Graph</h2><span class="badge" id="graphStatus">0 nodes</span></div>
        <div class="graph-wrap">
          <svg id="beliefGraph"></svg>
          <div class="graph-detail" id="graphDetail"></div>
          <div class="toast" id="toast"></div>
        </div>
      </section>
      <aside class="panel pane">
        <div class="pane-head"><h2>Fan-out</h2><span class="badge" id="workerStatus">idle</span></div>
        <div class="workers" id="workers"></div>
        <div class="counters" id="counters"></div>
        <div class="event-log" id="log"></div>
      </aside>
    </section>
    <section class="answer panel" id="answer">
      <div class="answer-head">
        <div class="brand">Best Current Answer</div>
        <button class="answer-close" id="answerClose" type="button">Close</button>
      </div>
      <section class="bottom-line" id="bottomLine"></section>
      <div class="headline" id="headline"></div>
      <p id="answerText"></p>
      <div class="metrics" id="metrics"></div>
      <div class="tables">
        <div><h3>Evidence</h3><table id="evidence"></table></div>
        <div><h3>Limits and Next Checks</h3><table id="gaps"></table></div>
      </div>
      <div id="artifact"></div>
    </section>
  </main>
  <script>
    const ask = document.getElementById("ask");
    const stop = document.getElementById("stop");
    const runIndicator = document.getElementById("runIndicator");
    const thinking = document.getElementById("thinking");
    const thoughtStream = document.getElementById("thoughtStream");
    const rawReasoning = document.getElementById("rawReasoning");
    const log = document.getElementById("log");
    const workers = document.getElementById("workers");
    const toast = document.getElementById("toast");
    const graphDetail = document.getElementById("graphDetail");
    let source = null;
    let pendingText = "";
    let textFrame = null;
    let graphNodes = [];
    let graphLinks = [];
    let nodeById = new Map();
    let expandedNodes = new Set();
    let focusedNodeId = null;
    let workerById = new Map();
    let latestCounters = {sources: 0, evidence: 0, leaps: 0, conflicts: 0, tests: 0};
    let thinkingLines = 0;
    const svg = document.getElementById("beliefGraph");
    const linkLayer = svgEl("g", {});
    const nodeLayer = svgEl("g", {});
    svg.appendChild(linkLayer);
    svg.appendChild(nodeLayer);
    svg.addEventListener("click", event => {
      if (event.target === svg) clearGraphFocus();
    });
    const simulation = {alpha: () => simulation, restart: () => { layoutGraph(); return simulation; }};
    new ResizeObserver(resizeGraph).observe(document.querySelector(".graph-wrap"));
    resizeGraph();
    reset();

    function resizeGraph() {
      const box = document.querySelector(".graph-wrap").getBoundingClientRect();
      svg.setAttribute("viewBox", `0 0 ${box.width || 720} ${box.height || 590}`);
      layoutGraph();
    }
    function reset() {
      pendingText = "";
      thinkingLines = 0;
      thoughtStream.innerHTML = "";
      rawReasoning.innerHTML = '<span class="cursor"></span>';
      log.innerHTML = "";
      workers.innerHTML = "";
      workerById.clear();
      graphNodes = [];
      graphLinks = [];
      nodeById = new Map();
      expandedNodes = new Set();
      focusedNodeId = null;
      graphDetail.classList.remove("visible");
      graphDetail.innerHTML = "";
      latestCounters = {sources: 0, evidence: 0, leaps: 0, conflicts: 0, tests: 0};
      updateGraph();
      renderCounters();
      document.querySelectorAll(".phase").forEach(el => el.className = "phase");
      document.getElementById("answer").classList.remove("visible");
      document.getElementById("graphStatus").textContent = "0 nodes";
      document.getElementById("workerStatus").textContent = "idle";
      document.getElementById("reasoningStatus").textContent = "waiting";
    }
    function onProgress(event) {
      addLog(event);
      markPhase(event.stage);
      const kind = event.kind || event.metadata?.kind || "";
      if (kind === "reasoning.delta") appendReasoning(event.metadata?.text || "");
      else if (kind === "tool.call") addToolChip(event.metadata?.name || event.stage, event);
      else if (kind === "tool.output") addToolChip(`${event.metadata?.name || "tool"} done`, event);
      else if (kind === "fanout.spawn") spawnWorkers(Number(event.metadata?.tasks || 0), event.metadata?.backend || "local");
      else if (kind === "fanout.task") settleWorker(event.metadata?.task_id, event.metadata?.task_status || event.status);
      else if (kind === "node.add") addGraphNode(event.metadata);
      else if (kind === "edge.add") addGraphEdge(event.metadata);
      else if (kind === "node.confidence") updateConfidence(event.metadata);
      else if (kind === "counter") updateCounters(event.metadata || {});
      if (!kind && event.stage?.startsWith("tool.")) addToolChip(event.stage.replace("tool.", ""), event);
      addThoughtCard(event);
    }
    function appendReasoning(text) {
      if (!text) return;
      const display = formatReasoningChunk(text);
      if (!display) return;
      appendRawReasoning(display);
      document.getElementById("reasoningStatus").textContent = "streaming";
    }
    function appendRawReasoning(text) {
      rawReasoning.insertBefore(document.createTextNode(text), rawReasoning.querySelector(".cursor"));
      thinkingLines += Math.max(1, String(text).split("\\n").length - 1);
      trimThinking();
      rawReasoning.scrollTop = rawReasoning.scrollHeight;
    }
    function trimThinking() {
      while (thinkingLines > 240 && rawReasoning.firstChild && !rawReasoning.firstChild.classList?.contains("cursor")) {
        const first = rawReasoning.firstChild;
        thinkingLines -= Math.max(1, (first.textContent || "").split("\\n").length - 1);
        first.remove();
      }
    }
    function formatReasoningChunk(text) {
      let clean = String(text)
        .replace(/\\\\n/g, " ")
        .replace(/\\\\r/g, " ")
        .replace(/\\\\t/g, " ")
        .replace(/\\\\"/g, '"')
        .replace(/\\s+/g, " ")
        .trim();
      if (!clean) return "";
      if (isJsonish(clean)) return `\\n↳ ${summarizeArgs(clean)}\\n`;
      return clean.endsWith(".") || clean.endsWith(":") ? `${clean} ` : `${clean} `;
    }
    function isJsonish(text) {
      const start = text[0];
      const jsonMarks = (text.match(/[{}":,\\[\\]]/g) || []).length;
      return start === "{" || start === "[" || jsonMarks / Math.max(text.length, 1) > .18;
    }
    function summarizeArgs(text) {
      try {
        const parsed = JSON.parse(text);
        const flat = flattenArgs(parsed);
        const parts = Object.entries(flat)
          .filter(([, value]) => value !== undefined && value !== null && String(value).trim())
          .slice(0, 3)
          .map(([key, value]) => `${key}=${quoteShort(value, key.includes("query") || key.includes("text"))}`);
        return parts.length ? `args: ${parts.join(", ")}` : "{...}";
      } catch {
        const id = text.match(/"?(id|task_id|source_id)"?\\s*:\\s*"?([^",}]+)/i)?.[2];
        const query = text.match(/"?(query|text|claim)"?\\s*:\\s*"([^"]+)/i)?.[2];
        const summary = [id ? `id=${id}` : "", query ? `query=${quoteShort(query, true)}` : ""].filter(Boolean).join(", ");
        return summary ? `args: ${summary}` : "{...}";
      }
    }
    function flattenArgs(value, prefix = "", out = {}) {
      if (Array.isArray(value)) {
        out[prefix || "items"] = `${value.length} items`;
        return out;
      }
      if (value && typeof value === "object") {
        Object.entries(value).forEach(([key, child]) => {
          const next = prefix ? `${prefix}.${key}` : key;
          if (child && typeof child === "object" && !Array.isArray(child)) flattenArgs(child, next, out);
          else out[key] = child;
        });
        return out;
      }
      out[prefix || "value"] = value;
      return out;
    }
    function quoteShort(value, quoted = false) {
      let text = String(value).replace(/\\s+/g, " ").trim();
      if (text.length > 80) text = `${text.slice(0, 77)}...`;
      return quoted ? `"${text}"` : text;
    }
    function addToolChip(name, event = null) {
      void name;
      void event;
      document.getElementById("reasoningStatus").textContent = "tools active";
    }
    function addThoughtCard(event) {
      const card = describeThought(event);
      if (!card) return;
      const div = document.createElement("div");
      div.className = `thought-card ${card.statusClass || ""}`;
      if (card.nodeId) div.dataset.nodeId = card.nodeId;
      div.innerHTML = `
        <span class="icon">${escapeHtml(card.icon)}</span>
        <div>
          <div class="label">${escapeHtml(card.label)}</div>
          <div class="sub">${escapeHtml(card.sub || "")}</div>
        </div>
        <span class="status-dot"></span>
      `;
      if (card.nodeId) {
        div.addEventListener("mouseenter", () => setNodeHighlight(card.nodeId, true));
        div.addEventListener("mouseleave", () => setNodeHighlight(card.nodeId, false));
      }
      thoughtStream.appendChild(div);
      while (thoughtStream.children.length > 40) thoughtStream.firstElementChild?.remove();
      thoughtStream.scrollTop = thoughtStream.scrollHeight;
      document.getElementById("reasoningStatus").textContent = "tools active";
      if (card.nodeId) flashNode(card.nodeId);
    }
    function describeThought(event) {
      const kind = event.kind || event.metadata?.kind || "";
      const meta = event.metadata || {};
      if (kind === "reasoning.delta" || kind === "edge.add") return null;
      if (kind === "tool.call") {
        return {
          icon: "▸",
          label: toolLabel(meta.name || event.stage, "call"),
          sub: meta.args_preview || event.stage || "",
          nodeId: nodeIdFromEvent(event),
        };
      }
      if (kind === "tool.output") {
        return {
          icon: "✓",
          label: toolLabel(meta.name || event.stage, "done"),
          sub: meta.summary || event.message || "",
          nodeId: nodeIdFromEvent(event),
        };
      }
      if (kind === "node.add") {
        return {
          icon: "+",
          label: `${titleCase(meta.node_kind || "node")}: ${thoughtLabel(meta.label || meta.id)}`,
          sub: meta.id || "",
          nodeId: meta.id,
        };
      }
      if (kind === "node.confidence") {
        const from = formatConfidence(meta.from);
        const to = formatConfidence(meta.to);
        const down = Number(meta.to) < Number(meta.from);
        return {
          icon: down ? "↓" : "↑",
          label: `${meta.id || "Node"} confidence ${from} → ${to}`,
          sub: down ? "Self-correction applied" : "Support increased",
          nodeId: meta.id,
          statusClass: "updated",
        };
      }
      if (kind === "fanout.spawn") {
        return {icon: "⇉", label: `${meta.tasks || 0} ${meta.backend || "local"} workers dispatched`, sub: event.stage || ""};
      }
      if (kind === "fanout.task") {
        return {
          icon: meta.task_status === "failed" || event.status === "failed" ? "!" : "✓",
          label: `${meta.task_id || "Worker"} ${meta.task_status || event.status}`,
          sub: meta.task_type || event.stage || "",
          statusClass: meta.task_status === "failed" || event.status === "failed" ? "failed" : "",
        };
      }
      if (kind === "counter") return {icon: "#", label: "Research counters updated", sub: counterSummary(meta)};
      if (event.stage === "input") return {icon: "?", label: "Question received", sub: event.metadata?.thesis_text || ""};
      if (event.stage === "answer") return {icon: "✓", label: "Answer ready", sub: event.message || ""};
      if (event.stage === "fallback") return {icon: "↳", label: "Recovered inspectable graph", sub: event.message || ""};
      if (event.stage === "error" || event.status === "failed") {
        return {icon: "!", label: event.message || "Run failed", sub: event.stage || "", statusClass: "failed"};
      }
      if (event.stage === "live_harness" || event.stage === "live_sdk") {
        return {icon: "•", label: event.message || event.stage, sub: event.status || ""};
      }
      return null;
    }
    function toolLabel(name, mode) {
      const cleaned = String(name || "tool").replace(/^tool\\./, "");
      const labels = {
        run_deterministic_research_loop_tool: "Running canonical research loop",
        retrieve_sources_tool: "Retrieving sources",
        score_retrieval_tool: "Scoring source relevance",
        execute_source_research_tasks_tool: "Dispatching research workers",
        extract_evidence_tool: "Extracting evidence",
        cross_check_evidence_tool: "Cross-checking evidence",
        detect_invalid_leaps_tool: "Finding unsafe inference leaps",
        update_beliefs_tool: "Updating belief graph",
        propose_decisive_tests_tool: "Drafting decisive tests",
        run_decisive_test_verifiers_tool: "Running verifier tests",
        generate_evals_from_failures_tool: "Writing evals from failures",
        build_eval_workshop_tool: "Assembling eval workshop",
        record_observability_tool: "Recording Raindrop trace",
      };
      const label = labels[cleaned] || titleCase(cleaned.replace(/_tool$/, "").replace(/_/g, " "));
      return mode === "done" ? `${label} completed` : label;
    }
    function nodeIdFromEvent(event) {
      const meta = event.metadata || {};
      const direct = meta.id || meta.node_id || meta.assumption_id || meta.evidence_id || meta.source_id;
      if (direct) return String(direct);
      const text = [meta.args_preview, meta.summary, event.message].filter(Boolean).join(" ");
      return text.match(/\\b(A\\d+|evidence_[\\w-]+|source_[\\w-]+|test_[\\w-]+|eval_[\\w-]+)\\b/)?.[1] || "";
    }
    function flashNode(id) {
      if (!id) return;
      const group = nodeLayer.querySelector(`[data-id="${cssEscape(id)}"]`);
      if (!group) return;
      group.classList.remove("pulse");
      group.getBoundingClientRect();
      group.classList.add("pulse");
      setTimeout(() => group.classList.remove("pulse"), 820);
    }
    function setNodeHighlight(id, enabled) {
      const group = nodeLayer.querySelector(`[data-id="${cssEscape(id)}"]`);
      if (group) group.classList.toggle("highlight", enabled);
    }
    function formatConfidence(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(2) : "?";
    }
    function counterSummary(meta) {
      return Object.entries(meta || {}).map(([key, value]) => `${key}: ${value}`).join(" · ");
    }
    function titleCase(value) {
      return String(value || "").replace(/\\b\\w/g, char => char.toUpperCase());
    }
    function addGraphNode(meta) {
      if (!meta?.id || nodeById.has(meta.id)) return;
      const node = {
        id: meta.id,
        kind: meta.node_kind || "unknown",
        label: meta.label || meta.id,
        confidence: Number(meta.confidence ?? .5),
        previousConfidence: Number(meta.confidence ?? .5),
      };
      nodeById.set(node.id, node);
      graphNodes.push(node);
      bumpCounter(counterForNode(node.kind));
      updateGraph();
      updateGraphStatus();
    }
    function addGraphEdge(meta) {
      if (!meta?.from || !meta?.to) return;
      if (!nodeById.has(meta.from)) addGraphNode({id: meta.from, node_kind: "unknown", label: meta.from});
      if (!nodeById.has(meta.to)) addGraphNode({id: meta.to, node_kind: "unknown", label: meta.to});
      const id = `${meta.from}->${meta.to}:${meta.relation || "relates"}`;
      const pair = graphLinks.find(link => link.source === meta.from && link.target === meta.to);
      if (pair) {
        if (pair.relation !== (meta.relation || "relates")) {
          pair.relation = meta.relation || "relates";
          flashNode(pair.target);
        }
        updateGraph();
        return;
      }
      if (graphLinks.some(link => link.id === id)) return;
      graphLinks.push({id, source: meta.from, target: meta.to, relation: meta.relation || "relates"});
      if (meta.relation === "contradicts") bumpCounter("conflicts");
      if (meta.relation === "proxy_only") bumpCounter("leaps");
      updateGraph();
    }
    function updateConfidence(meta) {
      const id = meta?.id;
      if (!id) return;
      if (!nodeById.has(id)) addGraphNode({id, node_kind: "assumption", label: id, confidence: meta.from || 0});
      const node = nodeById.get(id);
      node.previousConfidence = Number(meta.from ?? node.confidence);
      node.confidence = Number(meta.to ?? node.confidence);
      updateGraph();
      flashNode(id);
      if (Number(meta.to) < Number(meta.from)) showToast("⚠ Reclassified: benchmark wins ≠ prospective validation");
    }
    function updateGraph() {
      layoutGraph();
    }
    function layoutGraph() {
      const box = document.querySelector(".graph-wrap").getBoundingClientRect();
      const width = box.width || 720;
      const height = box.height || 590;
      const centerX = width / 2;
      const centerY = height / 2;
      const visible = buildHierarchyLayout(width, height, centerX, centerY);
      renderLinks(visible.links);
      renderNodes(visible.nodes);
      applyGraphFocus();
      updateGraphStatus();
    }
    function buildHierarchyLayout(width, height, centerX, centerY) {
      const thesisText = document.getElementById("question")?.value || "Thesis";
      const assumptions = graphNodes
        .filter(node => node.kind === "assumption")
        .sort((a, b) => a.id.localeCompare(b.id, undefined, {numeric: true}));
      const evidenceByAssumption = evidenceGroupsByAssumption();
      const assumptionRadius = Math.max(118, Math.min(width, height) * .29);
      const evidenceRadius = Math.max(52, Math.min(width, height) * .13);
      const visibleNodes = [
        {
          id: "thesis",
          kind: "thesis",
          label: thesisText,
          confidence: meanConfidence(assumptions),
          x: centerX,
          y: centerY,
          relationCounts: countAllEvidence(evidenceByAssumption),
          visible: true,
        },
      ];
      const visibleLinks = [];
      assumptions.forEach((node, index) => {
        const count = Math.max(assumptions.length, 1);
        const angle = -Math.PI / 2 + (Math.PI * 2 * index) / count;
        node.angle = angle;
        node.x = centerX + Math.cos(angle) * assumptionRadius;
        node.y = centerY + Math.sin(angle) * assumptionRadius * .82;
        node.relationCounts = relationCountsFor(node.id, evidenceByAssumption);
        node.visible = true;
        visibleNodes.push(node);
        visibleLinks.push({id: `thesis->${node.id}`, source: "thesis", target: node.id, relation: "supports", synthetic: true});
        if (!expandedNodes.has(node.id)) return;
        const leaves = evidenceByAssumption.get(node.id) || [];
        leaves.forEach((entry, leafIndex) => {
          const spread = Math.min(Math.PI * .62, Math.max(Math.PI * .18, leaves.length * .13));
          const leafAngle = angle + (leaves.length === 1 ? 0 : -spread / 2 + (spread * leafIndex) / (leaves.length - 1));
          const leaf = entry.node;
          leaf.angle = leafAngle;
          leaf.x = node.x + Math.cos(leafAngle) * evidenceRadius;
          leaf.y = node.y + Math.sin(leafAngle) * evidenceRadius * .78;
          leaf.visible = true;
          visibleNodes.push(leaf);
          visibleLinks.push({id: `${leaf.id}->${node.id}:${entry.relation}`, source: leaf.id, target: node.id, relation: entry.relation});
        });
      });
      return {nodes: visibleNodes, links: visibleLinks};
    }
    function evidenceGroupsByAssumption() {
      const groups = new Map();
      graphLinks.forEach(link => {
        const source = nodeById.get(link.source);
        const target = nodeById.get(link.target);
        if (!source || !target) return;
        if (source.kind === "evidence" && target.kind === "assumption") {
          if (!groups.has(target.id)) groups.set(target.id, []);
          groups.get(target.id).push({node: source, relation: link.relation || "supports"});
        }
      });
      groups.forEach(entries => entries.sort((a, b) => relationRank(a.relation) - relationRank(b.relation) || a.node.id.localeCompare(b.node.id)));
      return groups;
    }
    function relationRank(relation) {
      return {contradicts: 0, proxy_only: 1, supports: 2}[relation] ?? 3;
    }
    function relationCountsFor(id, evidenceByAssumption = evidenceGroupsByAssumption()) {
      const counts = {supports: 0, contradicts: 0, proxy_only: 0};
      (evidenceByAssumption.get(id) || []).forEach(entry => {
        counts[entry.relation] = (counts[entry.relation] || 0) + 1;
      });
      return counts;
    }
    function countAllEvidence(evidenceByAssumption) {
      const counts = {supports: 0, contradicts: 0, proxy_only: 0};
      evidenceByAssumption.forEach(entries => entries.forEach(entry => {
        counts[entry.relation] = (counts[entry.relation] || 0) + 1;
      }));
      return counts;
    }
    function meanConfidence(nodes) {
      if (!nodes.length) return .5;
      return nodes.reduce((sum, node) => sum + Number(node.confidence || 0), 0) / nodes.length;
    }
    function renderLinks(visibleLinks) {
      const visibleIds = new Set(visibleLinks.map(link => link.id));
      [...linkLayer.querySelectorAll(".link")].forEach(path => {
        if (!visibleIds.has(path.dataset.id)) path.remove();
      });
      visibleLinks.forEach(link => {
        const sourceNode = visibleNodeById(link.source);
        const targetNode = visibleNodeById(link.target);
        if (!sourceNode || !targetNode) return;
        let path = linkLayer.querySelector(`[data-id="${cssEscape(link.id)}"]`);
        if (!path) {
          path = svgEl("path", {"data-id": link.id});
          linkLayer.appendChild(path);
        }
        path.setAttribute("class", `link ${link.relation || ""}`);
        path.dataset.source = link.source;
        path.dataset.target = link.target;
        path.setAttribute("d", curvedPath(sourceNode, targetNode));
      });
    }
    function renderNodes(visibleNodes) {
      const visibleIds = new Set(visibleNodes.map(node => node.id));
      const pulsing = new Set([...nodeLayer.querySelectorAll(".pulse")].map(el => el.dataset.id));
      [...nodeLayer.querySelectorAll(".node")].forEach(group => {
        if (!visibleIds.has(group.dataset.id)) group.remove();
      });
      visibleNodes.forEach(node => {
        let group = nodeLayer.querySelector(`[data-id="${cssEscape(node.id)}"]`);
        if (!group) {
          group = svgEl("g", {"data-id": node.id});
          group.addEventListener("mouseenter", () => focusGraphNode(node.id, false));
          group.addEventListener("mouseleave", () => {
            if (!focusedNodeId) clearGraphFocus();
          });
          group.addEventListener("click", event => {
            event.stopPropagation();
            if (node.kind === "assumption") {
              expandedNodes.has(node.id) ? expandedNodes.delete(node.id) : expandedNodes.add(node.id);
            }
            focusGraphNode(node.id, true);
            updateGraph();
          });
          nodeLayer.appendChild(group);
        }
        group.setAttribute("class", `node ${pulsing.has(node.id) ? "pulse" : ""}`);
        group.setAttribute("transform", `translate(${node.x},${node.y})`);
        group.innerHTML = "";
        const glow = colorFor(node);
        const r = radiusFor(node);
        if (node.kind === "evidence") {
          group.appendChild(svgEl("rect", {
            class: "shape",
            x: -r * .72,
            y: -r * .72,
            width: r * 1.44,
            height: r * 1.44,
            transform: "rotate(45)",
            rx: 2,
            fill: glow,
            style: `filter: drop-shadow(0 0 6px ${glow}) drop-shadow(0 0 16px ${glow}55);`,
          }));
        } else {
          group.appendChild(svgEl("circle", {
            class: "shape",
            r,
            fill: node.kind === "thesis" ? "rgba(4,8,12,.92)" : glow,
            style: `filter: drop-shadow(0 0 7px ${glow}) drop-shadow(0 0 ${node.kind === "thesis" ? 28 : 20}px ${glow}55);`,
          }));
          if (node.kind === "thesis") {
            group.appendChild(svgEl("circle", {r: r + 5, fill: "none", stroke: glow, "stroke-width": 1.8, opacity: .8}));
          }
        }
        const title = svgEl("title", {});
        title.textContent = node.label || node.id;
        group.appendChild(title);
        if (node.kind === "assumption") renderEvidenceBadge(group, node, r);
        const labelOffset = labelOffsetFor(node);
        const text = svgEl("text", {x: labelOffset.x, y: labelOffset.y, "text-anchor": labelOffset.anchor});
        text.textContent = shortLabel(node.label);
        group.appendChild(text);
      });
    }
    function visibleNodeById(id) {
      if (id === "thesis") {
        const box = document.querySelector(".graph-wrap").getBoundingClientRect();
        return {id: "thesis", x: (box.width || 720) / 2, y: (box.height || 590) / 2};
      }
      return nodeById.get(id);
    }
    function curvedPath(source, target) {
      const mx = (source.x + target.x) / 2;
      const my = (source.y + target.y) / 2;
      const cx = mx + (target.y - source.y) * .08;
      const cy = my - (target.x - source.x) * .08;
      return `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${target.x.toFixed(1)} ${target.y.toFixed(1)}`;
    }
    function renderEvidenceBadge(group, node, r) {
      const counts = node.relationCounts || relationCountsFor(node.id);
      const up = counts.supports || 0;
      const down = (counts.contradicts || 0) + (counts.proxy_only || 0);
      if (!up && !down) return;
      const text = `▲${up} ▼${down}`;
      const width = Math.max(34, text.length * 6 + 10);
      group.appendChild(svgEl("rect", {class: "badge", x: -width / 2, y: -r - 21, width, height: 16, rx: 8}));
      const badgeText = svgEl("text", {class: "badge-text", y: -r - 9, "text-anchor": "middle"});
      badgeText.textContent = text;
      group.appendChild(badgeText);
    }
    function focusGraphNode(id, persist) {
      focusedNodeId = persist ? id : focusedNodeId;
      showGraphDetail(id);
      applyGraphFocus(id);
    }
    function clearGraphFocus() {
      focusedNodeId = null;
      graphDetail.classList.remove("visible");
      applyGraphFocus();
    }
    function applyGraphFocus(hoverId = null) {
      const id = hoverId || focusedNodeId;
      const neighbors = id ? directNeighbors(id) : new Set();
      nodeLayer.querySelectorAll(".node").forEach(group => {
        const active = !id || group.dataset.id === id || neighbors.has(group.dataset.id);
        group.classList.toggle("dim", !active);
        group.classList.toggle("focus", group.dataset.id === id);
      });
      linkLayer.querySelectorAll(".link").forEach(path => {
        const active = !id || path.dataset.source === id || path.dataset.target === id;
        path.classList.toggle("dim", !active);
        path.classList.toggle("focus", active && !!id);
      });
    }
    function directNeighbors(id) {
      const neighbors = new Set();
      if (id === "thesis") {
        graphNodes.filter(node => node.kind === "assumption").forEach(node => neighbors.add(node.id));
        return neighbors;
      }
      graphLinks.forEach(link => {
        if (link.source === id) neighbors.add(link.target);
        if (link.target === id) neighbors.add(link.source);
      });
      if (nodeById.get(id)?.kind === "assumption") neighbors.add("thesis");
      return neighbors;
    }
    function showGraphDetail(id) {
      const node = id === "thesis" ? {
        id: "thesis",
        kind: "thesis",
        label: document.getElementById("question")?.value || "Thesis",
        confidence: meanConfidence(graphNodes.filter(n => n.kind === "assumption")),
      } : nodeById.get(id);
      if (!node) return;
      const evidence = node.kind === "assumption" ? evidenceGroupsByAssumption().get(node.id) || [] : [];
      const counts = node.kind === "assumption" ? relationCountsFor(node.id) : null;
      graphDetail.innerHTML = `
        <h3>${escapeHtml(node.label || node.id)}</h3>
        <div class="meta">${escapeHtml(titleCase(node.kind))} · confidence ${formatConfidence(node.confidence)}</div>
        ${counts ? `<div><span class="pill">supports ${counts.supports || 0}</span><span class="pill">contradicts ${counts.contradicts || 0}</span><span class="pill">proxy ${counts.proxy_only || 0}</span></div>` : ""}
        ${evidence.slice(0, 7).map(entry => `<div class="claim"><span class="pill">${escapeHtml(entry.relation)}</span>${escapeHtml(entry.node.label || entry.node.id)}</div>`).join("")}
      `;
      graphDetail.classList.add("visible");
    }
    function radiusFor(d) {
      if (d.kind === "thesis") return 21;
      if (d.kind === "assumption") {
        const counts = d.relationCounts || relationCountsFor(d.id);
        const evidenceCount = (counts.supports || 0) + (counts.contradicts || 0) + (counts.proxy_only || 0);
        return Math.min(24, 13 + evidenceCount * 1.1);
      }
      return {evidence: 7, source: 9, test: 10, eval: 10, unknown: 8}[d.kind] || 8;
    }
    function colorFor(d) {
      if (d.confidence >= .66) return "#5EE6C9";
      if (d.confidence >= .34) return "#F4C152";
      return "#FF5C7A";
    }
    function labelOffsetFor(node) {
      if (node.kind === "thesis") return {x: 0, y: 38, anchor: "middle"};
      const angle = node.angle ?? 0;
      const outward = Math.cos(angle) >= 0;
      const x = Math.cos(angle) * (radiusFor(node) + 8);
      const y = Math.sin(angle) * (radiusFor(node) + 8) + 4;
      return {x, y, anchor: outward ? "start" : "end"};
    }
    function shortLabel(value) {
      const text = String(value || "");
      return text.length > 26 ? `${text.slice(0, 25)}…` : text;
    }
    function thoughtLabel(value) {
      const text = String(value || "").replace(/\\s+/g, " ").trim();
      return text.length > 88 ? `${text.slice(0, 87)}…` : text;
    }
    function updateGraphStatus() {
      const assumptionCount = graphNodes.filter(node => node.kind === "assumption").length;
      const evidenceCount = graphNodes.filter(node => node.kind === "evidence").length;
      document.getElementById("graphStatus").textContent = `${assumptionCount} assumptions · ${evidenceCount} evidence`;
    }
    function svgEl(name, attrs) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }
    function spawnWorkers(count, backend) {
      document.getElementById("workerStatus").textContent = `${count} ${backend} tasks`;
      for (let i = 0; i < count; i++) {
        const id = `worker-${Date.now()}-${i}`;
        const div = document.createElement("div");
        div.className = "worker running";
        div.dataset.id = id;
        div.dataset.task = id;
        div.style.animationDelay = `${i * 40}ms`;
        div.innerHTML = `<strong><span class="spinner"></span>${backend} worker</strong><div>task ${i + 1}</div>`;
        workers.appendChild(div);
        workerById.set(id, div);
      }
    }
    function settleWorker(taskId, status) {
      let div = taskId ? workers.querySelector(`[data-task="${cssEscape(taskId)}"]`) : null;
      if (!div) div = workers.querySelector(".worker.running");
      if (!div) return;
      if (taskId) div.dataset.task = taskId;
      div.className = status === "failed" ? "worker" : "worker done";
      div.innerHTML = `<strong>${status === "failed" ? "failed" : "done"}</strong><div>${escapeHtml(taskId || "task")}</div>`;
      bumpCounter("tests", 0);
    }
    function updateCounters(meta) {
      const previous = {...latestCounters};
      latestCounters = {...latestCounters, ...meta};
      renderCounters(previous);
    }
    function bumpCounter(key, amount = 1) {
      if (!key) return;
      latestCounters[key] = Math.max(Number(latestCounters[key] || 0), Number(latestCounters[key] || 0) + amount);
      renderCounters();
    }
    function counterForNode(kind) {
      if (kind === "source") return "sources";
      if (kind === "evidence") return "evidence";
      if (kind === "test" || kind === "eval") return "tests";
      return null;
    }
    function renderCounters(previous = {}) {
      document.getElementById("counters").innerHTML = [
        ["Sources", latestCounters.sources],
        ["Evidence", latestCounters.evidence],
        ["Leaps", latestCounters.leaps],
        ["Conflicts", latestCounters.conflicts],
        ["Tests", latestCounters.tests],
        ["Nodes", graphNodes.length],
      ].map(([label, value]) => {
        const key = label.toLowerCase();
        const oldValue = label === "Nodes" ? previous.nodes : previous[key];
        const bumped = oldValue !== undefined && Number(value || 0) !== Number(oldValue || 0);
        return `<div class="counter ${bumped ? "bump" : ""}"><div class="label">${label}</div><div class="value">${value || 0}</div></div>`;
      }).join("");
    }
    function markPhase(stage) {
      const map = [
        ["decompose", ["decompose"]],
        ["retrieve", ["retrieve", "score"]],
        ["extract", ["execute_source", "extract", "fanout"]],
        ["check", ["cross_check", "invalid"]],
        ["update", ["belief", "confidence"]],
        ["test", ["decisive", "verifier", "eval", "workshop", "observability", "answer"]],
      ];
      const key = map.find(([, needles]) => needles.some(n => String(stage).includes(n)))?.[0];
      if (!key) return;
      const phases = [...document.querySelectorAll(".phase")];
      const index = phases.findIndex(el => el.dataset.phase === key);
      phases.forEach((el, i) => {
        if (i < index) el.className = "phase done";
        if (i === index) el.className = "phase active";
      });
    }
    function addLog(event) {
      const div = document.createElement("div");
      div.className = "event";
      div.innerHTML = `<strong>${escapeHtml(event.message || event.stage)}</strong><span>${event.index || ""} | ${escapeHtml(event.kind || event.stage)} | ${escapeHtml(event.status)}</span>`;
      log.prepend(div);
    }
    function renderAnswer(result) {
      const summary = result.summary;
      document.querySelectorAll(".phase").forEach(el => el.className = "phase done");
      renderBottomLine(summary.bottom_line);
      document.getElementById("headline").textContent = summary.headline;
      document.getElementById("answerText").textContent =
        summary.verdict === "Not proven yet"
          ? "Pragmatic is showing the useful boundary: what evidence exists, where the inference would be unsafe, and what would settle it."
          : "Pragmatic found enough evidence to state a provisional answer while preserving limits.";
      document.getElementById("metrics").innerHTML = [
        ["Verdict", summary.verdict],
        ["Confidence", summary.mean_confidence],
        ["Sources", summary.counts.sources],
        ["Evidence", summary.counts.evidence],
        ["Modal tasks", summary.counts.modal_tasks]
      ].map(([label, value]) => `<div class="metric"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div></div>`).join("");
      document.getElementById("evidence").innerHTML = tableHtml(["Type", "Source", "Claim"], summary.evidence.map(row => [row.type, row.source, row.claim]));
      document.getElementById("gaps").innerHTML = tableHtml(["Kind", "Issue", "Next"], summary.gaps.map(row => [row.kind, row.summary, row.next]));
      document.getElementById("artifact").innerHTML = [
        summary.trace_id ? `<p><strong>Trace:</strong> ${escapeHtml(summary.trace_id)}</p>` : "",
        summary.trace_path ? `<p><strong>Local trace:</strong> ${escapeHtml(summary.trace_path)}</p>` : "",
        summary.workshop_path ? `<p><strong>Workshop:</strong> ${escapeHtml(summary.workshop_path)}</p>` : ""
      ].join("");
      document.getElementById("answer").classList.add("visible");
    }
    function renderBottomLine(bottomLine) {
      if (!bottomLine) {
        document.getElementById("bottomLine").innerHTML = "";
        return;
      }
      const risk = bottomLine.biggest_risk || {};
      const stats = String(bottomLine.stat_line || "")
        .split(" · ")
        .filter(Boolean)
        .map(item => `<span>${escapeHtml(item)}</span>`)
        .join("");
      const because = (bottomLine.because || [])
        .map(item => `<li>${escapeHtml(item)}</li>`)
        .join("");
      const band = bottomLine.confidence_band || "mid";
      const sourceTag = bottomLine.one_liner_source === "model" ? '<span class="ai-tag">AI summary</span>' : "";
      document.getElementById("bottomLine").innerHTML = `
        <div class="bottom-top">
          <span class="verdict-pill ${escapeHtml(band)}">${escapeHtml(bottomLine.verdict || "Verdict")}</span>
          <span class="confidence-meter">${escapeHtml(bottomLine.confidence_label || "Confidence")} ${escapeHtml(String(bottomLine.confidence ?? ""))}</span>
          ${sourceTag}
        </div>
        <div class="bottom-one-liner">${escapeHtml(bottomLine.one_liner || "")}</div>
        <div class="bottom-grid">
          <div class="bottom-row"><div class="label">Because</div><ul>${because}</ul></div>
          <div class="bottom-row"><div class="label">Biggest risk</div><div class="bottom-risk"><strong>${escapeHtml(risk.label || "")}</strong><br>${escapeHtml(risk.text || "")}</div></div>
          <div class="bottom-row"><div class="label">Decisive next test</div><div class="bottom-test">${escapeHtml(bottomLine.decisive_next_test || "")}</div></div>
        </div>
        <div class="stat-line">${stats}</div>
      `;
    }
    function tableHtml(headers, rows) {
      if (!rows.length) return "<tbody><tr><td>No rows yet.</td></tr></tbody>";
      return `<thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(String(cell || ""))}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }
    function showToast(message) {
      toast.textContent = message;
      toast.classList.add("visible");
      setTimeout(() => toast.classList.remove("visible"), 4200);
    }
    function cssEscape(value) {
      return String(value || "").replace(/["\\\\]/g, "\\\\$&");
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }
    async function startRun() {
      reset();
      ask.disabled = true;
      stop.disabled = false;
      runIndicator.classList.add("visible");
      document.getElementById("modeBadge").textContent =
        `${document.getElementById("orchestration").value} / ${document.getElementById("execution_backend").value} / ${document.getElementById("source_mode").value}`;
      const payload = {
        thesis_text: document.getElementById("question").value,
        config: {
          orchestration: document.getElementById("orchestration").value,
          execution_backend: document.getElementById("execution_backend").value,
          source_mode: document.getElementById("source_mode").value,
          corpus_choice: document.getElementById("corpus_choice").value,
          model: document.getElementById("model").value,
          web_search_model: document.getElementById("model").value,
          timeout_seconds: Number(document.getElementById("timeout_seconds").value),
          max_turns: Number(document.getElementById("max_turns").value),
          max_web_sources: Number(document.getElementById("max_web_sources").value),
          require_demo_proof: document.getElementById("require_demo_proof").value === "true",
          live_sdk_enabled: true,
          live_dry_run: false,
          allow_live_web_search: document.getElementById("source_mode").value === "web",
          observability_mode: "local"
        }
      };
      const response = await fetch("/api/runs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const data = await response.json();
      source = new EventSource(`/api/runs/${data.job_id}/events`);
      source.addEventListener("progress", message => onProgress(JSON.parse(message.data)));
      source.addEventListener("done", message => {
        const done = JSON.parse(message.data);
        ask.disabled = false;
        stop.disabled = true;
        runIndicator.classList.remove("visible");
        source.close();
        document.getElementById("reasoningStatus").textContent = "complete";
        if (done.status === "succeeded") renderAnswer(done.result);
        else onProgress({stage: "error", status: "failed", message: done.error || "Run failed", metadata: {}});
      });
    }
    ask.addEventListener("click", startRun);
    document.getElementById("answerClose").addEventListener("click", () => document.getElementById("answer").classList.remove("visible"));
    stop.addEventListener("click", () => {
      if (source) source.close();
      ask.disabled = false;
      stop.disabled = true;
      runIndicator.classList.remove("visible");
      onProgress({stage: "input", status: "stopped", message: "Stopped watching. The server job may still finish.", metadata: {}});
    });
  </script>
</body>
</html>
""".replace("__DEFAULT_THESIS__", html.escape(DEFAULT_THESIS)).replace("__DEFAULT_MODEL__", DEFAULT_MODEL)


app = Starlette(
    debug=False,
    routes=[
        Route("/", index, methods=["GET"]),
        Route("/api/runs", start_run, methods=["POST"]),
        Route("/api/runs/{job_id}", get_run, methods=["GET"]),
        Route("/api/runs/{job_id}/events", run_events, methods=["GET"]),
    ],
    lifespan=_lifespan,
)
