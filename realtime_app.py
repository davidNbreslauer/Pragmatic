from __future__ import annotations

import asyncio
import html
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from pragmatic import ResearchManager
from pragmatic.live_harness import run_live_harness_sync
from pragmatic.replay import run_replay_demo
from pragmatic.schemas import LiveRunResult, ResearchState


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_THESIS = "Spider silk for bullet proof vests"
MAX_STORED_EVENTS_PER_JOB = 2000
RUNS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()

# Realtime cockpit event taxonomy:
# - coarse events keep the stable {stage,status,message,metadata,index} envelope.
# - rich events add top-level kind plus metadata payloads:
#   reasoning.delta, tool.call, tool.output, fanout.spawn, fanout.task,
#   node.add, edge.add, node.confidence, counter.


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_job(thesis_text: str, config: dict[str, Any]) -> str:
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
            "result": None,
            "error": None,
        }
    return job_id


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
        normalized["index"] = len(job["events"]) + 1
        job["events"].append(normalized)
        if len(job["events"]) > MAX_STORED_EVENTS_PER_JOB:
            job["events"] = job["events"][-MAX_STORED_EVENTS_PER_JOB:]
        job["updated_at"] = normalized["created_at"]


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


async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


async def start_run(request: Request) -> JSONResponse:
    payload = await request.json()
    thesis_text = str(payload.get("thesis_text") or DEFAULT_THESIS).strip() or DEFAULT_THESIS
    config = _normalize_config(payload.get("config") or {})
    job_id = _new_job(thesis_text, config)
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
        sent_count = 0
        while True:
            with RUN_LOCK:
                job = RUNS.get(job_id)
                if job is None:
                    yield _sse("error", {"error": "run not found"})
                    return
                events = list(job["events"])
                status = job["status"]
                result = job["result"]
                error = job["error"]

            for event in events[sent_count:]:
                yield _sse("progress", event)
            sent_count = len(events)

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
            await asyncio.sleep(0.25)

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
                source_mode=config["source_mode"],
                allow_live_web_search=config["allow_live_web_search"],
                web_search_model=config["web_search_model"] or None,
                max_web_sources=config["max_web_sources"],
                observability_mode=config["observability_mode"],
            )

        _append_state_cockpit_events(job_id, state)
        summary = _summarize_state(state)
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
    return {
        "orchestration": str(raw.get("orchestration") or "live_sdk"),
        "execution_backend": str(raw.get("execution_backend") or "modal"),
        "observability_mode": str(raw.get("observability_mode") or "local"),
        "source_mode": str(raw.get("source_mode") or "web"),
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
        "evidence": evidence,
        "gaps": gaps,
        "decisive_test": decisive,
        "trace_id": state.observability.trace_id if state.observability is not None else None,
        "trace_path": state.observability.trace_path if state.observability is not None else None,
        "workshop_path": (
            state.observability.workshop_path if state.observability is not None else None
        ),
    }


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


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pragmatic AI</title>
  <style>
    :root {
      --bg: #f6f7f5;
      --panel: #ffffff;
      --ink: #18212b;
      --muted: #667085;
      --line: #d9dee5;
      --accent: #d84f45;
      --blue: #4267c8;
      --green: #2f7d47;
      --amber: #b97716;
      --red: #c2415b;
      --teal: #168a7a;
      --violet: #6d56bf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body { overflow: hidden; }
    main {
      height: 100vh;
      max-width: 1440px;
      margin: 0 auto;
      padding: 12px 18px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 10px;
      overflow: hidden;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) auto;
      gap: 12px;
      align-items: start;
      min-height: 56px;
    }
    .brand { font-size: 15px; font-weight: 850; color: #253041; letter-spacing: 0; }
    .tagline { color: var(--muted); font-size: 12px; margin-top: 2px; }
    h2 { margin: 0; font-size: 15px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 14px; }
    .ask, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .ask { padding: 10px; }
    label { display: block; font-weight: 740; margin-bottom: 5px; font-size: 12px; color: #344054; }
    .question-row { display: grid; grid-template-columns: minmax(280px, 1fr) auto; gap: 10px; align-items: end; }
    .question-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    textarea {
      width: 100%;
      height: 52px;
      min-height: 52px;
      max-height: 120px;
      resize: vertical;
      border: 1px solid #e1e5ea;
      border-radius: 7px;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      background: #f2f4f7;
      color: var(--ink);
    }
    button {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 12px;
      background: #fff;
      color: var(--ink);
      font-weight: 760;
      cursor: pointer;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); min-width: 92px; }
    .answer-close { min-width: 0; padding: 5px 9px; font-size: 13px; }
    button:disabled { opacity: .55; cursor: wait; }
    select, input { width: 100%; border: 1px solid var(--line); border-radius: 7px; padding: 7px; background: #fff; font: inherit; font-size: 12px; }
    details { margin-top: 7px; color: var(--muted); font-size: 12px; }
    details .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; align-items: center; }
    .chip, .badge {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 5px 7px;
      background: #fff;
      color: #344054;
      font-size: 11px;
    }
    .phase-ribbon { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px; margin: 0; }
    .phase {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--muted);
      padding: 7px 8px;
      min-height: 30px;
      font-size: 11px;
      font-weight: 800;
      text-align: center;
      transition: background .25s ease, color .25s ease, border-color .25s ease;
    }
    .phase.active { background: #fff5f2; color: #9f3a32; border-color: #eda39d; }
    .phase.done { background: #edf7ef; color: #27673b; border-color: #a7d9b4; }
    .cockpit {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(250px, .78fr) minmax(390px, 1.28fr) minmax(270px, .82fr);
      gap: 10px;
    }
    .pane { min-height: 0; padding: 12px; overflow: hidden; display: flex; flex-direction: column; }
    .pane-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; flex: 0 0 auto; }
    .thinking {
      flex: 1;
      min-height: 0;
      overflow: auto;
      border: 1px solid #e6e9ee;
      border-radius: 8px;
      background: #101820;
      color: #d9f3ee;
      padding: 12px;
      font: 11.5px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      mask-image: linear-gradient(to bottom, transparent 0, #000 14px, #000 100%);
    }
    .tool-chip {
      display: inline-block;
      margin: 2px 4px 2px 0;
      padding: 3px 6px;
      border-radius: 6px;
      background: #22324a;
      color: #cde3ff;
      font: 11px ui-sans-serif, system-ui, sans-serif;
    }
    .cursor { display: inline-block; width: 7px; height: 14px; background: #d9f3ee; animation: blink 1s steps(2) infinite; vertical-align: -2px; }
    @keyframes blink { 50% { opacity: 0; } }
    .graph-wrap { position: relative; flex: 1; min-height: 0; border: 1px solid #e1e5ea; border-radius: 8px; overflow: hidden; background: #fbfbf8; }
    #beliefGraph { width: 100%; height: 100%; display: block; }
    .link { stroke: #98a2b3; stroke-opacity: .62; stroke-width: 1.8px; }
    .link.contradicts { stroke: var(--red); stroke-dasharray: 6 4; }
    .link.proxy_only { stroke: var(--amber); stroke-dasharray: 4 4; }
    .link.tests, .link.becomes_eval { stroke: var(--violet); }
    .node circle { stroke: #fff; stroke-width: 2.4px; filter: drop-shadow(0 3px 7px rgba(17, 24, 39, .18)); }
    .node text { font-size: 11px; font-weight: 750; pointer-events: none; fill: #253041; }
    .node.pulse circle { animation: nodePulse .7s ease-out; }
    @keyframes nodePulse { 0% { stroke-width: 8px; } 100% { stroke-width: 2.4px; } }
    .toast {
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      display: none;
      border: 1px solid #f1b8b1;
      background: #fff6f4;
      color: #9f3a32;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 760;
    }
    .toast.visible { display: block; }
    .workers { flex: 0 0 34%; min-height: 100px; overflow: auto; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; align-content: start; }
    .worker {
      border: 1px solid #dfe4eb;
      border-radius: 8px;
      padding: 8px;
      min-height: 58px;
      background: #fbfcfd;
      font-size: 11px;
    }
    .worker.running { border-color: #96d4cc; background: #edf8f6; }
    .worker.done { border-color: #a8d8b5; background: #eef8f0; }
    .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #b3d9d5; border-top-color: var(--teal); border-radius: 50%; animation: spin .8s linear infinite; margin-right: 5px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .counters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin-top: 8px; flex: 0 0 auto; }
    .counter { border: 1px solid #e1e5ea; border-radius: 8px; padding: 8px; background: #fbfbf9; }
    .counter .label { font-size: 10.5px; color: var(--muted); }
    .counter .value { font-size: 18px; font-weight: 820; margin-top: 1px; }
    .event-log { flex: 1; min-height: 0; overflow: auto; display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
    .event { border: 1px solid #e5e7eb; border-radius: 7px; padding: 7px; background: #fbfbfb; font-size: 11px; }
    .event strong { display: block; font-size: 11px; }
    .event span { color: var(--muted); }
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
      box-shadow: 0 20px 60px rgba(15, 23, 42, .24);
      transform: translateY(18px);
      opacity: 0;
      transition: transform .22s ease, opacity .22s ease;
    }
    .answer.visible { display: block; transform: translateY(0); opacity: 1; }
    .answer-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .headline { font-size: 20px; line-height: 1.25; font-weight: 780; margin: 6px 0 8px; }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 9px; background: #fbfbf9; }
    .metric .label { color: var(--muted); font-size: 11px; }
    .metric .value { font-size: 18px; font-weight: 780; margin-top: 2px; }
    .tables { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #eaecf0; padding: 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 11px; }
    @media (max-width: 1050px) {
      main { overflow: auto; }
      body { overflow: auto; }
      .topbar, .question-row, .cockpit, .tables { grid-template-columns: 1fr; }
      .pane { min-height: 360px; }
      details .grid, .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chips, .question-actions { justify-content: flex-start; }
    }
    @media (max-width: 650px) {
      details .grid, .metrics, .phase-ribbon { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="topbar">
      <section class="ask">
        <div class="question-row">
          <div>
            <div class="brand">Pragmatic AI</div>
            <div class="tagline">Do diligence: sourced evidence, confidence updates, next decisive test.</div>
            <label for="question">Question</label>
            <textarea id="question" rows="2">__DEFAULT_THESIS__</textarea>
          </div>
          <div class="question-actions">
            <button class="primary" id="ask">Ask</button>
            <button id="stop" disabled>Stop</button>
            <span class="badge" id="modeBadge">live_sdk / modal / live web / local Workshop</span>
          </div>
        </div>
        <details>
          <summary>Run controls</summary>
          <div class="grid">
            <label>Orchestration<select id="orchestration"><option value="live_sdk">live_sdk</option><option value="scripted_sdk">scripted_sdk</option><option value="deterministic">deterministic</option></select></label>
            <label>Execution<select id="execution_backend"><option value="modal">modal</option><option value="local">local</option></select></label>
            <label>Sources<select id="source_mode"><option value="web">live web</option><option value="prepared">prepared</option></select></label>
            <label>Model<input id="model" value="__DEFAULT_MODEL__" /></label>
            <label>Timeout seconds<input id="timeout_seconds" type="number" value="300" min="5" max="600" /></label>
            <label>Max turns<input id="max_turns" type="number" value="3" min="1" max="20" /></label>
            <label>Max web sources<input id="max_web_sources" type="number" value="8" min="1" max="20" /></label>
            <label>Full proof<select id="require_demo_proof"><option value="true">required</option><option value="false">not required</option></select></label>
          </div>
        </details>
      </section>
      <div class="chips">
        <span class="chip">OpenAI Agents SDK</span>
        <span class="chip">Modal fan-out</span>
        <span class="chip">Raindrop Workshop</span>
      </div>
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
        <div class="thinking" id="thinking"><span class="cursor"></span></div>
      </aside>
      <section class="panel pane">
        <div class="pane-head"><h2>Belief Graph</h2><span class="badge" id="graphStatus">0 nodes</span></div>
        <div class="graph-wrap">
          <svg id="beliefGraph"></svg>
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
    const thinking = document.getElementById("thinking");
    const log = document.getElementById("log");
    const workers = document.getElementById("workers");
    const toast = document.getElementById("toast");
    let source = null;
    let pendingText = "";
    let textTimer = null;
    let graphNodes = [];
    let graphLinks = [];
    let nodeById = new Map();
    let workerById = new Map();
    let latestCounters = {sources: 0, evidence: 0, leaps: 0, conflicts: 0, tests: 0};
    let thinkingLines = 0;
    const svg = document.getElementById("beliefGraph");
    const linkLayer = svgEl("g", {});
    const nodeLayer = svgEl("g", {});
    svg.appendChild(linkLayer);
    svg.appendChild(nodeLayer);
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
      thinking.innerHTML = '<span class="cursor"></span>';
      log.innerHTML = "";
      workers.innerHTML = "";
      workerById.clear();
      graphNodes = [];
      graphLinks = [];
      nodeById = new Map();
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
      else if (kind === "tool.call") addToolChip(event.metadata?.name || event.stage);
      else if (kind === "tool.output") addToolChip(`${event.metadata?.name || "tool"} done`);
      else if (kind === "fanout.spawn") spawnWorkers(Number(event.metadata?.tasks || 0), event.metadata?.backend || "local");
      else if (kind === "fanout.task") settleWorker(event.metadata?.task_id, event.metadata?.task_status || event.status);
      else if (kind === "node.add") addGraphNode(event.metadata);
      else if (kind === "edge.add") addGraphEdge(event.metadata);
      else if (kind === "node.confidence") updateConfidence(event.metadata);
      else if (kind === "counter") updateCounters(event.metadata || {});
      if (!kind && event.stage?.startsWith("tool.")) addToolChip(event.stage.replace("tool.", ""));
    }
    function appendReasoning(text) {
      if (!text) return;
      const display = formatReasoningChunk(text);
      if (!display) return;
      pendingText += display;
      document.getElementById("reasoningStatus").textContent = "streaming";
      if (!textTimer) textTimer = setInterval(releaseThinkingText, 70);
    }
    function releaseThinkingText() {
      if (!pendingText) {
        clearInterval(textTimer);
        textTimer = null;
        return;
      }
      const chunk = pendingText.slice(0, 18);
      pendingText = pendingText.slice(18);
      appendThinkingNode(document.createTextNode(chunk), chunk);
      thinking.scrollTop = thinking.scrollHeight;
    }
    function appendThinkingNode(node, text = "") {
      thinking.insertBefore(node, thinking.querySelector(".cursor"));
      thinkingLines += Math.max(1, String(text).split("\\n").length - 1);
      trimThinking();
    }
    function trimThinking() {
      while (thinkingLines > 400 && thinking.firstChild && !thinking.firstChild.classList?.contains("cursor")) {
        const first = thinking.firstChild;
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
    function addToolChip(name) {
      const chip = document.createElement("span");
      chip.className = "tool-chip";
      chip.textContent = name;
      appendThinkingNode(chip, "\\n");
      appendThinkingNode(document.createTextNode(" "), " ");
      thinking.scrollTop = thinking.scrollHeight;
      document.getElementById("reasoningStatus").textContent = "tools active";
    }
    function addGraphNode(meta) {
      if (!meta?.id || nodeById.has(meta.id)) return;
      const node = {id: meta.id, kind: meta.node_kind || "unknown", label: meta.label || meta.id, confidence: Number(meta.confidence ?? .5)};
      nodeById.set(node.id, node);
      graphNodes.push(node);
      bumpCounter(counterForNode(node.kind));
      updateGraph();
      document.getElementById("graphStatus").textContent = `${graphNodes.length} nodes`;
    }
    function addGraphEdge(meta) {
      if (!meta?.from || !meta?.to) return;
      if (!nodeById.has(meta.from)) addGraphNode({id: meta.from, node_kind: "unknown", label: meta.from});
      if (!nodeById.has(meta.to)) addGraphNode({id: meta.to, node_kind: "unknown", label: meta.to});
      const id = `${meta.from}->${meta.to}:${meta.relation || "relates"}`;
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
      node.confidence = Number(meta.to ?? node.confidence);
      updateGraph();
      const group = nodeLayer.querySelector(`[data-id="${cssEscape(id)}"]`);
      if (group) group.classList.add("pulse");
      setTimeout(() => nodeLayer.querySelectorAll("g.node").forEach(el => el.classList.remove("pulse")), 800);
      if (Number(meta.to) < Number(meta.from)) showToast(`Reclassified: ${meta.reason || "confidence dropped after replay"}`);
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
      const rings = {assumption: Math.min(width, height) * .27, evidence: Math.min(width, height) * .16, source: Math.min(width, height) * .36, question: Math.min(width, height) * .08, test: Math.min(width, height) * .31, eval: Math.min(width, height) * .38, unknown: Math.min(width, height) * .08};
      const groups = {};
      graphNodes.forEach(node => {
        groups[node.kind] = groups[node.kind] || [];
        groups[node.kind].push(node);
      });
      Object.entries(groups).forEach(([kind, nodes]) => positionGroup(kind, nodes, rings[kind] || Math.min(width, height) * .2, centerX, centerY));
      relaxGraph(width, height, centerX, centerY);
      renderLinks();
      renderNodes();
    }
    function positionGroup(kind, nodes, radius, centerX, centerY) {
      if (kind === "question") {
        nodes.forEach((node, index) => {
          const spread = (index - (nodes.length - 1) / 2) * 42;
          node.x = centerX + spread;
          node.y = centerY - radius;
        });
        return;
      }
      nodes.forEach((node, index) => {
        const angle = (Math.PI * 2 * index / Math.max(nodes.length, 1)) + kindOffset(kind);
        node.x = centerX + Math.cos(angle) * radius;
        node.y = centerY + Math.sin(angle) * radius * .7;
      });
    }
    function relaxGraph(width, height, centerX, centerY) {
      const margin = 24;
      for (let pass = 0; pass < 60; pass++) {
        graphNodes.forEach(node => {
          node.x += (centerX - node.x) * .005;
          node.y += (centerY - node.y) * .005;
        });
        graphLinks.forEach(link => {
          const a = nodeById.get(link.source);
          const b = nodeById.get(link.target);
          if (!a || !b) return;
          const dx = b.x - a.x || .1;
          const dy = b.y - a.y || .1;
          const dist = Math.hypot(dx, dy);
          const target = (a.kind === "question" || b.kind === "question") ? 50 : 120;
          const shift = (dist - target) * .035;
          const ux = dx / dist;
          const uy = dy / dist;
          a.x += ux * shift;
          a.y += uy * shift;
          b.x -= ux * shift;
          b.y -= uy * shift;
        });
        for (let i = 0; i < graphNodes.length; i++) {
          for (let j = i + 1; j < graphNodes.length; j++) {
            const a = graphNodes[i];
            const b = graphNodes[j];
            const dx = b.x - a.x || .1;
            const dy = b.y - a.y || .1;
            const dist = Math.hypot(dx, dy);
            const min = radiusFor(a) + radiusFor(b) + 18;
            if (dist < min) {
              const push = (min - dist) / 2;
              const ux = dx / dist;
              const uy = dy / dist;
              a.x -= ux * push;
              a.y -= uy * push;
              b.x += ux * push;
              b.y += uy * push;
            }
          }
        }
        graphNodes.forEach(node => {
          node.x = Math.max(margin, Math.min(width - margin, node.x));
          node.y = Math.max(margin, Math.min(height - margin - 18, node.y));
        });
      }
    }
    function kindOffset(kind) {
      return {assumption: -.2, evidence: .5, source: 1.1, question: -1.55, test: 2.1, eval: 2.8, unknown: 0}[kind] || 0;
    }
    function renderLinks() {
      linkLayer.innerHTML = "";
      graphLinks.forEach(link => {
        const sourceNode = typeof link.source === "string" ? nodeById.get(link.source) : link.source;
        const targetNode = typeof link.target === "string" ? nodeById.get(link.target) : link.target;
        if (!sourceNode || !targetNode) return;
        const line = svgEl("line", {
          class: `link ${link.relation || ""}`,
          x1: sourceNode.x,
          y1: sourceNode.y,
          x2: targetNode.x,
          y2: targetNode.y,
        });
        linkLayer.appendChild(line);
      });
    }
    function renderNodes() {
      const pulsing = new Set([...nodeLayer.querySelectorAll(".pulse")].map(el => el.dataset.id));
      nodeLayer.innerHTML = "";
      graphNodes.forEach(node => {
        const group = svgEl("g", {class: `node ${pulsing.has(node.id) ? "pulse" : ""}`, "data-id": node.id, transform: `translate(${node.x},${node.y})`});
        group.appendChild(svgEl("circle", {r: radiusFor(node), fill: colorFor(node)}));
        const title = svgEl("title", {});
        title.textContent = node.label;
        group.appendChild(title);
        const text = svgEl("text", {y: radiusFor(node) + 12, "text-anchor": "middle"});
        text.textContent = shortLabel(node.label);
        group.appendChild(text);
        nodeLayer.appendChild(group);
      });
    }
    function radiusFor(d) {
      return {assumption: 14, question: 9, evidence: 11, source: 10, test: 12, eval: 12, unknown: 9}[d.kind] || 10;
    }
    function colorFor(d) {
      if (d.kind === "source") return "#60744d";
      if (d.kind === "evidence") return "#b06b20";
      if (d.kind === "test") return "#6d56bf";
      if (d.kind === "eval") return "#7b55c7";
      if (d.confidence >= .66) return "#2f7d47";
      if (d.confidence >= .34) return "#b97716";
      return "#c2415b";
    }
    function shortLabel(value) {
      const text = String(value || "");
      return text.length > 22 ? `${text.slice(0, 21)}…` : text;
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
      latestCounters = {...latestCounters, ...meta};
      renderCounters();
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
    function renderCounters() {
      document.getElementById("counters").innerHTML = [
        ["Sources", latestCounters.sources],
        ["Evidence", latestCounters.evidence],
        ["Leaps", latestCounters.leaps],
        ["Conflicts", latestCounters.conflicts],
        ["Tests", latestCounters.tests],
        ["Nodes", graphNodes.length],
      ].map(([label, value]) => `<div class="counter"><div class="label">${label}</div><div class="value">${value || 0}</div></div>`).join("");
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
      document.getElementById("modeBadge").textContent =
        `${document.getElementById("orchestration").value} / ${document.getElementById("execution_backend").value} / ${document.getElementById("source_mode").value}`;
      const payload = {
        thesis_text: document.getElementById("question").value,
        config: {
          orchestration: document.getElementById("orchestration").value,
          execution_backend: document.getElementById("execution_backend").value,
          source_mode: document.getElementById("source_mode").value,
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
)
