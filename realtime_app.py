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
RUNS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()


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
        "metadata": event.get("metadata") or {},
    }
    with RUN_LOCK:
        job = RUNS[job_id]
        normalized["index"] = len(job["events"]) + 1
        job["events"].append(normalized)
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
        "max_turns": int(raw.get("max_turns") or 12),
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


INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pragmatic</title>
  <style>
    :root {{
      --bg: #f8f9f7;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #667085;
      --line: #d7dce2;
      --accent: #ef5550;
      --agent: #496ddb;
      --source: #60744d;
      --worker: #168a7a;
      --evidence: #b06b20;
      --failure: #c2415b;
      --eval: #7b55c7;
      --belief: #2f7d47;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 30px 18px 60px; }}
    .brand {{ font-size: 12px; font-weight: 800; color: #4b5563; text-transform: uppercase; }}
    h1 {{ margin: 18px 0 8px; font-size: clamp(36px, 6vw, 62px); line-height: 1; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 17px; max-width: 760px; }}
    .ask {{
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    label {{ display: block; font-weight: 700; margin-bottom: 8px; }}
    textarea {{
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid #e1e5ea;
      border-radius: 7px;
      padding: 14px;
      font: inherit;
      background: #f2f4f7;
      color: var(--ink);
    }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }}
    button {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px 18px;
      background: #fff;
      color: var(--ink);
      font-weight: 760;
      cursor: pointer;
    }}
    button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); min-width: 160px; }}
    button:disabled {{ opacity: 0.55; cursor: wait; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; }}
    .chip {{ border: 1px solid var(--line); border-radius: 7px; padding: 9px 11px; background: #fff; color: #344054; font-size: 13px; }}
    details {{ margin-top: 12px; color: var(--muted); }}
    details .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    select, input {{ width: 100%; border: 1px solid var(--line); border-radius: 7px; padding: 9px; background: #fff; font: inherit; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(300px, 0.8fr); gap: 16px; margin-top: 28px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    h2 {{ margin: 0 0 14px; font-size: 28px; letter-spacing: 0; }}
    .graph-shell {{
      height: 620px;
      background: linear-gradient(180deg, #fbfbf8 0%, #f0f3ef 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      position: relative;
    }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .node {{ opacity: 0; transform-origin: center; transition: opacity .22s ease, transform .22s ease; }}
    .node.visible {{ opacity: 1; }}
    .node.running .ring {{ animation: pulse 680ms ease-in-out infinite alternate; }}
    @keyframes pulse {{ from {{ stroke-width: 2; opacity: .35; }} to {{ stroke-width: 7; opacity: .85; }} }}
    .edge {{ stroke: rgba(74, 85, 104, .28); stroke-width: 2; fill: none; }}
    .edge.hot {{ stroke: var(--accent); stroke-width: 3; stroke-dasharray: 10 8; animation: dash 880ms linear infinite; }}
    @keyframes dash {{ to {{ stroke-dashoffset: -36; }} }}
    .log {{ height: 620px; overflow: auto; display: flex; flex-direction: column; gap: 8px; }}
    .event {{ border: 1px solid #e5e7eb; border-radius: 7px; padding: 10px; background: #fbfbfb; }}
    .event strong {{ display: block; font-size: 13px; }}
    .event span {{ color: var(--muted); font-size: 12px; }}
    .answer {{ margin-top: 18px; display: none; }}
    .answer.visible {{ display: block; }}
    .headline {{ font-size: 24px; line-height: 1.25; font-weight: 780; margin: 6px 0 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfbf9; }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 23px; font-weight: 780; margin-top: 3px; }}
    .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #eaecf0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; }}
    .error {{ color: #b42318; font-weight: 760; }}
    @media (max-width: 850px) {{
      .layout, .tables {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chips {{ margin-left: 0; }}
      details .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="brand">Pragmatic</div>
    <h1>Ask a technical question.</h1>
    <p class="sub">Watch the real research job run: OpenAI Agents SDK orchestration, Modal fan-out, evidence extraction, invalid-leap detection, belief update, and Workshop artifacts.</p>

    <section class="ask">
      <label for="question">Question</label>
      <textarea id="question">{html.escape(DEFAULT_THESIS)}</textarea>
      <div class="actions">
        <button class="primary" id="ask">Ask Pragmatic</button>
        <button id="stop" disabled>Stop Watching</button>
        <div class="chips">
          <span class="chip">live_sdk</span>
          <span class="chip">modal</span>
          <span class="chip">live web</span>
          <span class="chip">Raindrop Workshop local</span>
        </div>
      </div>
      <details>
        <summary>Run controls</summary>
        <div class="grid">
          <label>Orchestration<select id="orchestration"><option value="live_sdk">live_sdk</option><option value="scripted_sdk">scripted_sdk</option><option value="deterministic">deterministic</option></select></label>
          <label>Execution<select id="execution_backend"><option value="modal">modal</option><option value="local">local</option></select></label>
          <label>Sources<select id="source_mode"><option value="web">live web</option><option value="prepared">prepared</option></select></label>
          <label>Model<input id="model" value="{DEFAULT_MODEL}" /></label>
          <label>Timeout seconds<input id="timeout_seconds" type="number" value="300" min="5" max="600" /></label>
          <label>Max turns<input id="max_turns" type="number" value="12" min="1" max="20" /></label>
          <label>Max web sources<input id="max_web_sources" type="number" value="8" min="1" max="20" /></label>
          <label>Full proof<select id="require_demo_proof"><option value="true">required</option><option value="false">not required</option></select></label>
        </div>
      </details>
    </section>

    <section class="layout">
      <div>
        <h2>Thinking Graph</h2>
        <div class="graph-shell"><svg id="graph" viewBox="0 0 900 620"></svg></div>
      </div>
      <aside>
        <h2>Live Trace</h2>
        <div class="panel log" id="log"></div>
      </aside>
    </section>

    <section class="answer panel" id="answer">
      <div class="brand">Best Current Answer</div>
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
    const graph = document.getElementById("graph");
    const log = document.getElementById("log");
    const ask = document.getElementById("ask");
    const stop = document.getElementById("stop");
    let source = null;
    let lastNode = null;
    const nodes = new Map();
    const positions = {{
      input: [80, 300],
      live_harness: [210, 170],
      live_sdk: [360, 170],
      "tool.decompose_thesis_tool": [500, 90],
      "tool.plan_questions_tool": [620, 90],
      "tool.retrieve_sources_tool": [760, 150],
      "tool.score_retrieval_tool": [760, 260],
      "tool.execute_source_research_tasks_tool": [620, 360],
      "tool.extract_evidence_tool": [500, 460],
      "tool.cross_check_evidence_tool": [360, 460],
      "tool.detect_invalid_leaps_tool": [220, 450],
      "tool.update_beliefs_tool": [220, 320],
      "tool.propose_decisive_tests_tool": [360, 320],
      "tool.run_decisive_test_verifiers_tool": [500, 300],
      "tool.generate_evals_from_failures_tool": [620, 470],
      "tool.build_eval_workshop_tool": [760, 460],
      "tool.record_observability_tool": [760, 350],
      fallback: [210, 520],
      answer: [820, 300],
      error: [820, 520],
      deterministic: [360, 520],
      scripted_sdk: [360, 520],
      replay: [360, 520]
    }};
    const labels = {{
      input: "Question",
      live_harness: "Harness",
      live_sdk: "Agents SDK",
      "tool.decompose_thesis_tool": "Assumptions",
      "tool.plan_questions_tool": "Questions",
      "tool.retrieve_sources_tool": "Sources",
      "tool.score_retrieval_tool": "Retrieval",
      "tool.execute_source_research_tasks_tool": "Modal Fan-out",
      "tool.extract_evidence_tool": "Evidence",
      "tool.cross_check_evidence_tool": "Cross-check",
      "tool.detect_invalid_leaps_tool": "Invalid leaps",
      "tool.update_beliefs_tool": "Belief graph",
      "tool.propose_decisive_tests_tool": "Decisive tests",
      "tool.run_decisive_test_verifiers_tool": "Verifiers",
      "tool.generate_evals_from_failures_tool": "Failure evals",
      "tool.build_eval_workshop_tool": "Workshop",
      "tool.record_observability_tool": "Trace",
      fallback: "Fallback graph",
      answer: "Answer",
      error: "Error",
      deterministic: "Loop",
      scripted_sdk: "Scripted SDK",
      replay: "Replay"
    }};
    const colors = {{
      input: "#5a6673",
      live_harness: "#496ddb",
      live_sdk: "#496ddb",
      source: "#60744d",
      worker: "#168a7a",
      evidence: "#b06b20",
      failure: "#c2415b",
      eval: "#7b55c7",
      belief: "#2f7d47",
      answer: "#2f7d47",
      error: "#c2415b"
    }};
    const stageKinds = {{
      "tool.retrieve_sources_tool": "source",
      "tool.execute_source_research_tasks_tool": "worker",
      "tool.extract_evidence_tool": "evidence",
      "tool.cross_check_evidence_tool": "evidence",
      "tool.detect_invalid_leaps_tool": "failure",
      "tool.update_beliefs_tool": "belief",
      "tool.generate_evals_from_failures_tool": "eval",
      "tool.build_eval_workshop_tool": "eval",
      "tool.record_observability_tool": "eval"
    }};

    function reset() {{
      graph.innerHTML = "";
      log.innerHTML = "";
      nodes.clear();
      lastNode = null;
      document.getElementById("answer").classList.remove("visible");
    }}
    function ensureNode(stage, event) {{
      const id = positions[stage] ? stage : "scripted_sdk";
      if (nodes.has(id)) return nodes.get(id);
      const [x, y] = positions[id];
      const kind = stageKinds[id] || id;
      const color = colors[kind] || colors[id] || "#496ddb";
      const group = svgEl("g", {{ class: "node visible", "data-id": id }});
      group.appendChild(svgEl("circle", {{ cx: x, cy: y, r: 30, fill: "#fff", stroke: "#d0d5dd" }}));
      group.appendChild(svgEl("circle", {{ cx: x, cy: y, r: 18, fill: color, opacity: ".16" }}));
      group.appendChild(svgEl("circle", {{ cx: x, cy: y, r: 9, fill: color }}));
      group.appendChild(svgEl("circle", {{ cx: x, cy: y, r: 23, fill: "none", stroke: color, class: "ring" }}));
      group.appendChild(textEl(x, y + 46, labels[id] || stage.replace("tool.", ""), "middle", "13px", "700"));
      graph.appendChild(group);
      const node = {{ id, x, y, el: group }};
      nodes.set(id, node);
      return node;
    }}
    function svgEl(name, attrs) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}
    function textEl(x, y, value, anchor, size, weight) {{
      const t = svgEl("text", {{ x, y, "text-anchor": anchor, "font-size": size, "font-weight": weight || "500", fill: "#17212b" }});
      t.textContent = value;
      return t;
    }}
    function drawEdge(from, to, hot) {{
      const path = svgEl("path", {{
        d: `M ${{from.x}} ${{from.y}} Q ${{(from.x + to.x) / 2}} ${{(from.y + to.y) / 2 - 42}} ${{to.x}} ${{to.y}}`,
        class: hot ? "edge hot" : "edge"
      }});
      graph.insertBefore(path, graph.firstChild);
      setTimeout(() => path.classList.remove("hot"), 1600);
    }}
    function setRunning(node, status) {{
      nodes.forEach((n) => n.el.classList.remove("running"));
      if (status === "running") node.el.classList.add("running");
    }}
    function onProgress(event) {{
      addLog(event);
      const node = ensureNode(event.stage, event);
      if (lastNode && lastNode.id !== node.id) drawEdge(lastNode, node, true);
      lastNode = node;
      setRunning(node, event.status);
    }}
    function addLog(event) {{
      const div = document.createElement("div");
      div.className = "event";
      div.innerHTML = `<strong>${{escapeHtml(event.message || event.stage)}}</strong><span>${{event.index || ""}} | ${{escapeHtml(event.stage)}} | ${{escapeHtml(event.status)}}</span>`;
      log.prepend(div);
    }}
    function renderAnswer(result) {{
      const summary = result.summary;
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
      ].map(([label, value]) => `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join("");
      document.getElementById("evidence").innerHTML = tableHtml(["Type", "Source", "Claim"], summary.evidence.map(row => [row.type, row.source, row.claim]));
      document.getElementById("gaps").innerHTML = tableHtml(["Kind", "Issue", "Next"], summary.gaps.map(row => [row.kind, row.summary, row.next]));
      document.getElementById("artifact").innerHTML = [
        summary.trace_id ? `<p><strong>Trace:</strong> ${{escapeHtml(summary.trace_id)}}</p>` : "",
        summary.trace_path ? `<p><strong>Local trace:</strong> ${{escapeHtml(summary.trace_path)}}</p>` : "",
        summary.workshop_path ? `<p><strong>Workshop:</strong> ${{escapeHtml(summary.workshop_path)}}</p>` : ""
      ].join("");
      document.getElementById("answer").classList.add("visible");
      onProgress({{ stage: "answer", status: "succeeded", message: "Answer rendered.", metadata: {{}} }});
    }}
    function tableHtml(headers, rows) {{
      if (!rows.length) return "<tbody><tr><td>No rows yet.</td></tr></tbody>";
      return `<thead><tr>${{headers.map(h => `<th>${{escapeHtml(h)}}</th>`).join("")}}</tr></thead><tbody>${{rows.map(row => `<tr>${{row.map(cell => `<td>${{escapeHtml(String(cell || ""))}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
    }}
    function escapeHtml(value) {{
      return value.replace(/[&<>"']/g, c => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[c]));
    }}
    async function startRun() {{
      reset();
      ask.disabled = true;
      stop.disabled = false;
      const payload = {{
        thesis_text: document.getElementById("question").value,
        config: {{
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
        }}
      }};
      const response = await fetch("/api/runs", {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify(payload) }});
      const data = await response.json();
      source = new EventSource(`/api/runs/${{data.job_id}}/events`);
      source.addEventListener("progress", (message) => onProgress(JSON.parse(message.data)));
      source.addEventListener("done", (message) => {{
        const done = JSON.parse(message.data);
        ask.disabled = false;
        stop.disabled = true;
        source.close();
        if (done.status === "succeeded") renderAnswer(done.result);
        else onProgress({{ stage: "error", status: "failed", message: done.error || "Run failed", metadata: {{}} }});
      }});
    }}
    ask.addEventListener("click", startRun);
    stop.addEventListener("click", () => {{
      if (source) source.close();
      ask.disabled = false;
      stop.disabled = true;
      onProgress({{ stage: "input", status: "stopped", message: "Stopped watching. The server job may still finish.", metadata: {{}} }});
    }});
    reset();
  </script>
</body>
</html>
"""


app = Starlette(
    debug=False,
    routes=[
        Route("/", index, methods=["GET"]),
        Route("/api/runs", start_run, methods=["POST"]),
        Route("/api/runs/{job_id}", get_run, methods=["GET"]),
        Route("/api/runs/{job_id}/events", run_events, methods=["GET"]),
    ],
)
