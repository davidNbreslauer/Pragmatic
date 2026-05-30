from __future__ import annotations

import json
from collections import Counter
from typing import Any

from thesisgraph.schemas import ResearchState


def build_orchestration_flow_snapshot(
    state: ResearchState,
    *,
    scenario_name: str,
    current_run_source: str,
) -> dict[str, Any]:
    agent_names = _agent_names(state)
    task_type_counts = Counter(result.task_type for result in state.research_task_results)
    remote_tasks = [
        result for result in state.research_task_results if result.backend == "modal"
    ]
    failed_tasks = [
        result for result in state.research_task_results if result.status == "failed"
    ]
    worker_labels = sorted(
        {
            result.metadata.get("worker_status", "")
            for result in state.research_task_results
            if result.metadata.get("worker_status")
        }
    )

    pipeline_nodes = [
        {
            "id": "thesis",
            "label": "Input thesis",
            "detail": _shorten(state.thesis.text, 24),
            "count": 1,
            "kind": "input",
        },
        {
            "id": "agents",
            "label": "Agents",
            "detail": _shorten(", ".join(agent_names[:4]), 24),
            "count": len(agent_names),
            "kind": "agent",
        },
        {
            "id": "questions",
            "label": "Assumptions",
            "detail": f"{len(state.research_questions)} research questions",
            "count": len(state.assumptions),
            "kind": "question",
        },
        {
            "id": "sources",
            "label": "Prepared corpus",
            "detail": f"{len(state.sources)} retrieved sources",
            "count": len(state.sources),
            "kind": "source",
        },
        {
            "id": "workers",
            "label": "Fan-out workers",
            "detail": _worker_detail(remote_tasks, state.research_task_results),
            "count": len(state.research_task_results),
            "kind": "worker",
        },
        {
            "id": "evidence",
            "label": "Typed evidence",
            "detail": f"{len(state.evidence_conflicts)} conflicts",
            "count": len(state.evidence_items),
            "kind": "evidence",
        },
        {
            "id": "failures",
            "label": "Invalid leaps",
            "detail": f"{len(failed_tasks)} failed worker tasks",
            "count": len(state.invalid_leaps),
            "kind": "failure",
        },
        {
            "id": "evals",
            "label": "Failure evals",
            "detail": "Workshop linked" if state.eval_workshop is not None else "No workshop",
            "count": len(state.generated_evals),
            "kind": "eval",
        },
        {
            "id": "belief",
            "label": "Belief graph",
            "detail": f"{len(state.belief_updates)} updates",
            "count": len(state.belief_updates),
            "kind": "belief",
        },
    ]

    return {
        "scenario": scenario_name,
        "run_source": current_run_source,
        "mode": state.agent_run.mode if state.agent_run is not None else "deterministic",
        "status": state.agent_run.status if state.agent_run is not None else "succeeded",
        "final_validated": (
            state.agent_run.final_output_validated
            if state.agent_run is not None
            else True
        ),
        "agent_names": agent_names,
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "worker_labels": worker_labels,
        "counts": {
            "agents": len(agent_names),
            "steps": len(state.agent_run.steps) if state.agent_run is not None else 0,
            "tasks": len(state.research_task_results),
            "modal_tasks": len(remote_tasks),
            "sources": len(state.sources),
            "evidence": len(state.evidence_items),
            "conflicts": len(state.evidence_conflicts),
            "invalid_leaps": len(state.invalid_leaps),
            "generated_evals": len(state.generated_evals),
            "belief_updates": len(state.belief_updates),
            "workshop_rows": (
                len(state.eval_workshop.connection_rows)
                if state.eval_workshop is not None
                else 0
            ),
        },
        "pipeline_nodes": pipeline_nodes,
        "pipeline_edges": _pipeline_edges(pipeline_nodes),
        "data_nodes": _data_nodes(state),
        "data_edges": _data_edges(state),
        "flow_events": _flow_events(state),
    }


def render_orchestration_flow_html(snapshot: dict[str, Any]) -> str:
    payload = (
        json.dumps(snapshot, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{
    color-scheme: light;
    --bg: #f7f8f6;
    --ink: #1f2933;
    --muted: #67717c;
    --line: rgba(60, 72, 82, 0.28);
    --agent: #496ddb;
    --worker: #168a7a;
    --evidence: #b06b20;
    --failure: #c2415b;
    --eval: #7b55c7;
    --belief: #2f7d47;
    --panel: rgba(255, 255, 255, 0.88);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: transparent;
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .flow-shell {{
    height: 650px;
    border: 1px solid rgba(80, 91, 104, 0.18);
    border-radius: 8px;
    background:
      radial-gradient(circle at 18% 14%, rgba(73, 109, 219, 0.12), transparent 28%),
      radial-gradient(circle at 82% 20%, rgba(22, 138, 122, 0.12), transparent 28%),
      linear-gradient(180deg, #fbfbf8 0%, #f1f3ef 100%);
    overflow: hidden;
    position: relative;
  }}
  .flow-header {{
    position: absolute;
    left: 18px;
    right: 18px;
    top: 14px;
    display: flex;
    align-items: start;
    justify-content: space-between;
    gap: 18px;
    z-index: 5;
  }}
  .title {{
    font-size: 17px;
    font-weight: 720;
    line-height: 1.2;
  }}
  .subtitle {{
    margin-top: 4px;
    font-size: 12px;
    color: var(--muted);
    max-width: 660px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .badges {{
    display: flex;
    flex-wrap: wrap;
    justify-content: end;
    gap: 6px;
    max-width: 430px;
  }}
  .badge {{
    padding: 5px 8px;
    border-radius: 7px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(55, 65, 81, 0.14);
    font-size: 11px;
    color: #2f3a45;
  }}
  .badge strong {{
    font-weight: 760;
    color: var(--ink);
  }}
  .legend {{
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 14px;
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 8px;
    z-index: 5;
  }}
  .legend-item {{
    min-width: 0;
    padding: 8px 9px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(55, 65, 81, 0.13);
  }}
  .legend-label {{
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
  }}
  .legend-value {{
    margin-top: 2px;
    font-size: 17px;
    font-weight: 760;
  }}
  svg {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  .edge {{
    fill: none;
    stroke: var(--line);
    stroke-width: 1.7;
    opacity: 0;
    transition: opacity 180ms ease, stroke-width 180ms ease;
  }}
  .edge.visible {{
    opacity: 0.42;
  }}
  .edge.hot {{
    stroke-width: 2.4;
    stroke: rgba(73, 109, 219, 0.44);
  }}
  .node-card {{
    filter: drop-shadow(0 8px 14px rgba(31, 41, 51, 0.08));
    opacity: 0;
    transition: opacity 220ms ease;
  }}
  .node-card.visible {{
    opacity: 1;
  }}
  .data-node-wrap {{
    opacity: 0;
    transition: opacity 220ms ease;
  }}
  .data-node-wrap.visible {{
    opacity: 1;
  }}
  .node-rect {{
    fill: var(--panel);
    stroke: rgba(55, 65, 81, 0.2);
    stroke-width: 1;
    rx: 8;
  }}
  .node-ring {{
    fill: none;
    stroke-width: 2;
    opacity: 0.85;
  }}
  .node-label {{
    font-size: 12px;
    font-weight: 720;
    fill: var(--ink);
  }}
  .node-detail {{
    font-size: 9.5px;
    fill: var(--muted);
  }}
  .node-count {{
    font-size: 18px;
    font-weight: 800;
    fill: var(--ink);
  }}
  .data-label {{
    font-size: 9px;
    font-weight: 680;
    fill: #28323d;
  }}
  .data-node {{
    stroke: rgba(255, 255, 255, 0.86);
    stroke-width: 1.6;
    filter: drop-shadow(0 4px 8px rgba(31, 41, 51, 0.16));
  }}
  .pulse {{
    opacity: 0.92;
    filter: drop-shadow(0 0 7px currentColor);
  }}
  .event-edge {{
    stroke-dasharray: 4 6;
  }}
  .event-card {{
    position: absolute;
    left: 18px;
    right: 18px;
    top: 78px;
    display: grid;
    grid-template-columns: 120px minmax(0, 1fr) 120px;
    gap: 8px;
    z-index: 5;
  }}
  .event-pill,
  .event-main {{
    min-width: 0;
    padding: 7px 9px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(55, 65, 81, 0.13);
  }}
  .event-kicker {{
    color: var(--muted);
    font-size: 9px;
    text-transform: uppercase;
  }}
  .event-value {{
    margin-top: 2px;
    color: var(--ink);
    font-size: 12px;
    font-weight: 760;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .event-summary {{
    margin-top: 2px;
    color: var(--muted);
    font-size: 10.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .active-node .node-ring,
  .active-node .data-node {{
    stroke: #111827;
    stroke-width: 2.6;
    animation: real-pulse 760ms ease-in-out infinite alternate;
  }}
  @keyframes real-pulse {{
    from {{
      filter: drop-shadow(0 0 0 rgba(17, 24, 39, 0));
      opacity: 0.74;
    }}
    to {{
      filter: drop-shadow(0 0 10px rgba(17, 24, 39, 0.35));
      opacity: 1;
    }}
  }}
  .flow-caption {{
    position: absolute;
    top: 376px;
    left: 22px;
    font-size: 11px;
    color: var(--muted);
    z-index: 4;
  }}
  .phase-caption {{
    position: absolute;
    top: 148px;
    left: 22px;
    font-size: 11px;
    color: var(--muted);
    z-index: 4;
  }}
  @media (max-width: 760px) {{
    .flow-shell {{ height: 720px; }}
    .flow-header {{ flex-direction: column; }}
    .badges {{ display: none; }}
    .event-card {{ grid-template-columns: 86px minmax(0, 1fr) 86px; }}
    .legend {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  }}
</style>
</head>
<body>
<div class="flow-shell">
  <div class="flow-header">
    <div>
      <div class="title">Orchestration Flow</div>
      <div class="subtitle" id="flow-subtitle"></div>
    </div>
    <div class="badges" id="flow-badges"></div>
  </div>
  <div class="event-card">
    <div class="event-pill">
      <div class="event-kicker">Replay</div>
      <div class="event-value" id="event-progress">0 / 0</div>
    </div>
    <div class="event-main">
      <div class="event-kicker" id="event-kind">Real run event</div>
      <div class="event-value" id="event-title">Waiting for current ResearchState</div>
      <div class="event-summary" id="event-summary">The moving dot follows actual artifacts from this run.</div>
    </div>
    <div class="event-pill">
      <div class="event-kicker">Artifact</div>
      <div class="event-value" id="event-artifact">none</div>
    </div>
  </div>
  <div class="phase-caption">Real event replay from the current ResearchState</div>
  <div class="flow-caption">Evidence graph: assumptions, sources, failures, evals, and belief updates</div>
  <svg viewBox="0 0 860 650" preserveAspectRatio="xMidYMid meet" id="flow-svg"></svg>
  <div class="legend" id="flow-legend"></div>
</div>
<script type="application/json" id="flow-data">{payload}</script>
<script>
(function() {{
  const data = JSON.parse(document.getElementById("flow-data").textContent);
  const svg = document.getElementById("flow-svg");
  const NS = "http://www.w3.org/2000/svg";
  const colors = {{
    input: "#5a6673",
    agent: "#496ddb",
    question: "#7b55c7",
    source: "#60744d",
    worker: "#168a7a",
    evidence: "#b06b20",
    failure: "#c2415b",
    eval: "#7b55c7",
    belief: "#2f7d47"
  }};
  const pipelineLayout = [
    [50, 132], [198, 132], [346, 132], [494, 132], [642, 132],
    [642, 254], [494, 254], [346, 254], [198, 254]
  ];

  function el(name, attrs = {{}}, text = null) {{
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) {{
      node.setAttribute(key, value);
    }}
    if (text !== null) node.textContent = text;
    return node;
  }}

  function pathBetween(a, b, bend = 0) {{
    const x1 = a.x + 100;
    const y1 = a.y + 34;
    const x2 = b.x;
    const y2 = b.y + 34;
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2 + bend;
    return `M ${{x1}} ${{y1}} Q ${{mx}} ${{my}} ${{x2}} ${{y2}}`;
  }}

  function addText(parent, x, y, text, attrs = {{}}) {{
    parent.appendChild(el("text", {{ x, y, ...attrs }}, text || ""));
  }}

  document.getElementById("flow-subtitle").textContent =
    `${{data.scenario}} | ${{data.run_source || "Current run"}}`;
  document.getElementById("flow-badges").innerHTML = [
    ["Mode", data.mode],
    ["Status", data.status],
    ["Validated", data.final_validated ? "yes" : "no"],
    ["Agents", data.counts.agents],
    ["Modal tasks", data.counts.modal_tasks]
  ].map(([label, value]) => `<span class="badge">${{label}} <strong>${{value}}</strong></span>`).join("");

  const legend = [
    ["Agents", data.counts.agents],
    ["Tool steps", data.counts.steps],
    ["Fan-out tasks", data.counts.tasks],
    ["Evidence", data.counts.evidence],
    ["Failures", data.counts.invalid_leaps],
    ["Generated evals", data.counts.generated_evals]
  ];
  document.getElementById("flow-legend").innerHTML = legend.map(([label, value]) =>
    `<div class="legend-item"><div class="legend-label">${{label}}</div><div class="legend-value">${{value}}</div></div>`
  ).join("");

  const pathByPair = new Map();
  const nodeElements = new Map();
  const allNodes = new Map();

  function pairKey(from, to) {{
    return `${{from}}->${{to}}`;
  }}

  function registerPath(from, to, path) {{
    pathByPair.set(pairKey(from, to), path);
  }}

  function showElement(node) {{
    if (node) node.classList.add("visible");
  }}

  function hideElement(node) {{
    if (node) node.classList.remove("visible", "active-node");
  }}

  function markActive(...ids) {{
    nodeElements.forEach((node) => node.classList.remove("active-node"));
    ids.forEach((id) => nodeElements.get(id)?.classList.add("active-node"));
  }}

  function centerOf(node) {{
    if (node.visual === "pipeline") {{
      return {{ x: node.x + 52, y: node.y + 35 }};
    }}
    return {{ x: node.x, y: node.y }};
  }}

  function eventPathD(fromNode, toNode) {{
    const from = centerOf(fromNode);
    const to = centerOf(toNode);
    return `M ${{from.x}} ${{from.y}} Q ${{(from.x + to.x) / 2}} ${{(from.y + to.y) / 2 - 26}} ${{to.x}} ${{to.y}}`;
  }}

  function ensurePath(event) {{
    const existing = pathByPair.get(pairKey(event.from, event.to));
    if (existing) return existing;
    const from = allNodes.get(event.from);
    const to = allNodes.get(event.to);
    if (!from || !to) return null;
    const path = el("path", {{
      d: eventPathD(from, to),
      class: "edge event-edge"
    }});
    svg.insertBefore(path, svg.firstChild);
    registerPath(event.from, event.to, path);
    return path;
  }}

  const pipelineNodes = data.pipeline_nodes.map((node, index) => {{
    const [x, y] = pipelineLayout[index] || [70 + index * 115, 132];
    return {{ ...node, x, y, visual: "pipeline" }};
  }});
  const nodeById = Object.fromEntries(pipelineNodes.map((node) => [node.id, node]));
  pipelineNodes.forEach((node) => allNodes.set(node.id, node));

  data.pipeline_edges.forEach((edge, index) => {{
    const from = nodeById[edge.from];
    const to = nodeById[edge.to];
    if (!from || !to) return;
    const id = `pipe-edge-${{index}}`;
    const bend = edge.to === "belief" ? 58 : 0;
    const path = el("path", {{
      id,
      d: pathBetween(from, to, bend),
      class: `edge ${{edge.hot ? "hot" : ""}}`
    }});
    svg.appendChild(path);
    registerPath(edge.from, edge.to, path);
  }});

  pipelineNodes.forEach((node) => {{
    const group = el("g", {{ class: "node-card" }});
    group.appendChild(el("rect", {{ x: node.x, y: node.y, width: 105, height: 70, class: "node-rect" }}));
    group.appendChild(el("circle", {{
      cx: node.x + 18,
      cy: node.y + 19,
      r: 9,
      fill: colors[node.kind] || "#5a6673",
      opacity: "0.92"
    }}));
    group.appendChild(el("circle", {{
      cx: node.x + 18,
      cy: node.y + 19,
      r: 13,
      stroke: colors[node.kind] || "#5a6673",
      class: "node-ring"
    }}));
    addText(group, node.x + 33, node.y + 23, node.label, {{ class: "node-label" }});
    addText(group, node.x + 14, node.y + 51, String(node.count), {{ class: "node-count" }});
    addText(group, node.x + 42, node.y + 50, node.detail || "", {{ class: "node-detail" }});
    svg.appendChild(group);
    nodeElements.set(node.id, group);
  }});

  const dataArea = {{ x: 48, y: 382, width: 760, height: 150 }};
  const center = {{ x: 430, y: 456 }};
  const dataNodes = layoutDataNodes(data.data_nodes, center, dataArea).map((node) => ({{
    ...node,
    visual: "data"
  }}));
  const dataNodeById = Object.fromEntries(dataNodes.map((node) => [node.id, node]));
  dataNodes.forEach((node) => allNodes.set(node.id, node));

  data.data_edges.forEach((edge, index) => {{
    const from = dataNodeById[edge.from];
    const to = dataNodeById[edge.to];
    if (!from || !to) return;
    const pathId = `data-edge-${{index}}`;
    const path = el("path", {{
      id: pathId,
      d: `M ${{from.x}} ${{from.y}} Q ${{(from.x + to.x) / 2}} ${{(from.y + to.y) / 2 - 24}} ${{to.x}} ${{to.y}}`,
      class: "edge"
    }});
    svg.appendChild(path);
    registerPath(edge.from, edge.to, path);
  }});

  dataNodes.forEach((node) => {{
    const radius = Math.max(8, Math.min(19, 8 + (node.weight || 1) * 1.7));
    const group = el("g", {{ class: "data-node-wrap" }});
    group.appendChild(el("circle", {{
      cx: node.x,
      cy: node.y,
      r: radius,
      fill: colors[node.kind] || "#5a6673",
      class: "data-node",
      opacity: node.dim ? "0.56" : "0.94"
    }}));
    addText(group, node.x + radius + 4, node.y + 3, node.label, {{ class: "data-label" }});
    svg.appendChild(group);
    nodeElements.set(node.id, group);
  }});

  const eventPulse = el("circle", {{
    r: 5.4,
    fill: "#111827",
    class: "pulse",
    style: "color:#111827;opacity:0"
  }});
  svg.appendChild(eventPulse);

  function resetReplay() {{
    nodeElements.forEach((node) => hideElement(node));
    pathByPair.forEach((path) => path.classList.remove("visible"));
    eventPulse.style.opacity = "0";
    markActive();
    showElement(nodeElements.get("thesis"));
  }}

  function describeEvent(event, index, total) {{
    document.getElementById("event-progress").textContent = `${{index + 1}} / ${{total}}`;
    document.getElementById("event-kind").textContent =
      `${{event.action || "event"}} | ${{event.kind || "state"}}`;
    document.getElementById("event-title").textContent = event.title || "ResearchState event";
    document.getElementById("event-summary").textContent = event.summary || "";
    document.getElementById("event-artifact").textContent = event.artifact_id || event.to || "none";
  }}

  function animatePulseAlong(path, duration, done) {{
    const totalLength = Math.max(1, path.getTotalLength());
    const startedAt = performance.now();
    eventPulse.style.opacity = "0.92";
    function frame(now) {{
      const progress = Math.min(1, (now - startedAt) / duration);
      const point = path.getPointAtLength(totalLength * progress);
      eventPulse.setAttribute("cx", point.x);
      eventPulse.setAttribute("cy", point.y);
      if (progress < 1) {{
        requestAnimationFrame(frame);
      }} else {{
        eventPulse.style.opacity = "0";
        done();
      }}
    }}
    requestAnimationFrame(frame);
  }}

  function playEvent(index = 0) {{
    const replayEvents = (data.flow_events || []).filter((event) => event.from && event.to);
    if (!replayEvents.length) {{
      nodeElements.forEach((node) => showElement(node));
      pathByPair.forEach((path) => path.classList.add("visible"));
      document.getElementById("event-progress").textContent = "0 / 0";
      document.getElementById("event-title").textContent = "No lifecycle events recorded";
      document.getElementById("event-summary").textContent = "Showing the latest ResearchState graph.";
      return;
    }}
    if (index === 0) resetReplay();
    const current = replayEvents[index];
    const path = ensurePath(current);
    const fromNode = nodeElements.get(current.from);
    const toNode = nodeElements.get(current.to);
    describeEvent(current, index, replayEvents.length);

    if (!path || !fromNode || !toNode) {{
      setTimeout(() => playEvent((index + 1) % replayEvents.length), 180);
      return;
    }}

    showElement(fromNode);
    showElement(toNode);
    markActive(current.from, current.to);

    if (current.action === "destroy") {{
      path.classList.add("visible");
      animatePulseAlong(path, 420, () => {{
        path.classList.remove("visible");
        hideElement(toNode);
        markActive();
        const next = (index + 1) % replayEvents.length;
        setTimeout(() => playEvent(next), next === 0 ? 1000 : 120);
      }});
      return;
    }}

    path.classList.add("visible");
    const duration = Math.max(360, Math.min(2600, Number(current.duration_ms) || 720));
    animatePulseAlong(path, duration, () => {{
      markActive();
      const next = (index + 1) % replayEvents.length;
      setTimeout(() => playEvent(next), next === 0 ? 1100 : 170);
    }});
  }}

  playEvent(0);

  function layoutDataNodes(nodes, center, area) {{
    const assumptions = nodes.filter((node) => node.kind === "belief");
    const sources = nodes.filter((node) => node.kind === "source" || node.kind === "evidence");
    const failures = nodes.filter((node) => node.kind === "failure" || node.kind === "eval");
    const others = nodes.filter((node) => !assumptions.includes(node) && !sources.includes(node) && !failures.includes(node));
    const placed = [];
    assumptions.forEach((node, index) => {{
      const angle = (Math.PI * 2 * index) / Math.max(1, assumptions.length) - Math.PI / 2;
      placed.push({{
        ...node,
        x: center.x + Math.cos(angle) * 130,
        y: center.y + Math.sin(angle) * 62
      }});
    }});
    sources.forEach((node, index) => {{
      const col = index % 4;
      const row = Math.floor(index / 4);
      placed.push({{ ...node, x: area.x + col * 58, y: area.y + 28 + row * 45 }});
    }});
    failures.forEach((node, index) => {{
      const col = index % 4;
      const row = Math.floor(index / 4);
      placed.push({{ ...node, x: area.x + area.width - col * 58, y: area.y + 28 + row * 45 }});
    }});
    others.forEach((node, index) => {{
      placed.push({{ ...node, x: center.x - 25 + index * 50, y: center.y + 6 }});
    }});
    return placed;
  }}
}})();
</script>
</body>
</html>
"""


def _agent_names(state: ResearchState) -> list[str]:
    names = []
    if state.agent_run is not None:
        names.append(state.agent_run.agent_name)
        names.extend(step.agent_name for step in state.agent_run.steps if step.agent_name)
    if not names:
        names.append("Deterministic Loop")
    return sorted(set(names))


def _pipeline_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for left, right in zip(nodes, nodes[1:]):
        edges.append(
            {
                "from": left["id"],
                "to": right["id"],
                "weight": max(1, min(5, right["count"] or 1)),
                "hot": right["count"] > 0,
            }
        )
    return edges


def _data_nodes(state: ResearchState) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for assumption in state.assumptions[:10]:
        nodes.append(
            {
                "id": assumption.id,
                "label": assumption.id,
                "kind": "belief",
                "weight": max(1, round(assumption.confidence * 8)),
                "status": assumption.support_level,
            }
        )
    for index, source in enumerate(state.sources[:6], start=1):
        nodes.append(
            {
                "id": source.id,
                "label": f"S{index}",
                "kind": "source",
                "weight": 2,
            }
        )
    for index, item in enumerate(state.evidence_items[:8], start=1):
        nodes.append(
            {
                "id": item.id,
                "label": f"E{index}",
                "kind": "evidence",
                "weight": max(1, round(item.confidence * 6)),
            }
        )
    for index, leap in enumerate(state.invalid_leaps[:6], start=1):
        nodes.append(
            {
                "id": leap.id,
                "label": f"L{index}",
                "kind": "failure",
                "weight": 3,
            }
        )
    for index, generated_eval in enumerate(state.generated_evals[:6], start=1):
        nodes.append(
            {
                "id": generated_eval.id,
                "label": f"EV{index}",
                "kind": "eval",
                "weight": 3,
            }
        )
    return nodes


def _data_edges(state: ResearchState) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    source_ids = {source.id for source in state.sources[:6]}
    assumption_ids = {assumption.id for assumption in state.assumptions[:10]}
    evidence_ids = {item.id for item in state.evidence_items[:8]}
    leap_ids = {leap.id for leap in state.invalid_leaps[:6]}
    eval_ids = {generated_eval.id for generated_eval in state.generated_evals[:6]}

    for item in state.evidence_items[:8]:
        if item.source_id in source_ids:
            edges.append(
                {
                    "from": item.source_id,
                    "to": item.id,
                    "kind": "source_evidence",
                    "weight": max(1, round(item.confidence * 4)),
                }
            )
        for assumption_id in item.assumption_ids:
            if assumption_id in assumption_ids and item.id in evidence_ids:
                edges.append(
                    {
                        "from": item.id,
                        "to": assumption_id,
                        "kind": "evidence_belief",
                        "weight": max(1, round(item.confidence * 4)),
                    }
                )

    for leap in state.invalid_leaps[:6]:
        for assumption_id in leap.affected_assumption_ids:
            if assumption_id in assumption_ids and leap.id in leap_ids:
                edges.append(
                    {
                        "from": assumption_id,
                        "to": leap.id,
                        "kind": "belief_failure",
                        "weight": 2,
                    }
                )

    first_leap_id = next(iter(leap_ids), None)
    for generated_eval in state.generated_evals[:6]:
        if generated_eval.id not in eval_ids:
            continue
        failure_id = generated_eval.source_failure_id or first_leap_id
        if failure_id in leap_ids:
            edges.append(
                {
                    "from": failure_id,
                    "to": generated_eval.id,
                    "kind": "failure_eval",
                    "weight": 4,
                }
            )
    return edges


def _flow_events(state: ResearchState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add_event(
        source: str,
        target: str,
        kind: str,
        title: str,
        summary: str,
        artifact_id: str,
        *,
        action: str = "create",
        duration_ms: int | None = None,
    ) -> None:
        if not source or not target:
            return
        events.append(
            {
                "seq": len(events) + 1,
                "from": source,
                "to": target,
                "kind": kind,
                "title": title,
                "summary": _shorten(summary, 150),
                "artifact_id": artifact_id,
                "action": action,
                "duration_ms": duration_ms or _event_duration(kind),
            }
        )

    add_event(
        "thesis",
        "agents",
        "input",
        "Thesis received",
        state.thesis.text,
        "thesis",
    )

    if state.agent_run is not None:
        previous = "agents"
        for step in state.agent_run.steps:
            target = _tool_target(step.tool_name)
            add_event(
                previous,
                target,
                "agent_step",
                step.agent_name or state.agent_run.agent_name,
                step.summary or step.tool_name,
                step.id,
                action="run",
            )
            previous = target

    for result in state.research_task_results:
        duration_ms = _duration_ms(result.metadata)
        add_event(
            "sources",
            "workers",
            "research_task",
            f"{result.task_type} task",
            f"{result.backend} worker {result.status}; sources={len(result.source_ids)}",
            result.task_id,
            action="run",
            duration_ms=duration_ms,
        )
        add_event(
            "workers",
            _task_target(result.task_type),
            "research_result",
            f"{result.task_type} result",
            _task_result_summary(result),
            result.task_id,
            action="create",
            duration_ms=duration_ms,
        )

    visible_source_ids = {source.id for source in state.sources[:6]}
    visible_assumption_ids = {assumption.id for assumption in state.assumptions[:10]}
    visible_evidence_ids = {item.id for item in state.evidence_items[:8]}
    visible_leap_ids = {leap.id for leap in state.invalid_leaps[:6]}
    visible_eval_ids = {generated_eval.id for generated_eval in state.generated_evals[:6]}

    for source in state.sources[:6]:
        add_event(
            "sources",
            source.id,
            "source",
            f"Source loaded: {source.id}",
            source.title,
            source.id,
            action="create",
        )

    for item in state.evidence_items[:8]:
        if item.source_id in visible_source_ids:
            add_event(
                item.source_id,
                item.id,
                "evidence_item",
                f"Evidence extracted: {item.id}",
                item.claim_supported,
                item.id,
                action="create",
            )
        for assumption_id in item.assumption_ids:
            if assumption_id in visible_assumption_ids and item.id in visible_evidence_ids:
                add_event(
                    item.id,
                    assumption_id,
                    "evidence_link",
                    f"Evidence linked to {assumption_id}",
                    item.limitation,
                    item.id,
                    action="link",
                )

    for leap in state.invalid_leaps[:6]:
        for assumption_id in leap.affected_assumption_ids:
            if assumption_id in visible_assumption_ids and leap.id in visible_leap_ids:
                add_event(
                    assumption_id,
                    leap.id,
                    "invalid_leap",
                    f"Invalid leap detected: {leap.id}",
                    leap.why_invalid,
                    leap.id,
                    action="create",
                )

    first_leap_id = next(iter(visible_leap_ids), None)
    for generated_eval in state.generated_evals[:6]:
        if generated_eval.id not in visible_eval_ids:
            continue
        failure_id = generated_eval.source_failure_id or first_leap_id
        if failure_id in visible_leap_ids:
            add_event(
                failure_id,
                generated_eval.id,
                "generated_eval",
                f"Eval generated: {generated_eval.id}",
                generated_eval.eval_rule,
                generated_eval.id,
                action="create",
            )

    for update in state.belief_updates:
        if update.assumption_id not in visible_assumption_ids:
            continue
        source = _belief_update_source(update.assumption_id, state)
        if source:
            add_event(
                source,
                update.assumption_id,
                "belief_update",
                f"Belief updated: {update.assumption_id}",
                update.rationale,
                update.assumption_id,
                action="update",
            )

    return events


def _tool_target(tool_name: str) -> str:
    if "decompose" in tool_name or "plan" in tool_name:
        return "questions"
    if "retrieve" in tool_name or "score" in tool_name:
        return "sources"
    if "execute" in tool_name or "extract" in tool_name or "verifier" in tool_name:
        return "workers"
    if "cross_check" in tool_name:
        return "evidence"
    if "invalid" in tool_name:
        return "failures"
    if "eval" in tool_name or "workshop" in tool_name or "observability" in tool_name:
        return "evals"
    if "belief" in tool_name:
        return "belief"
    return "agents"


def _task_target(task_type: str) -> str:
    if task_type == "parse_source":
        return "sources"
    if task_type in {"extract_evidence", "cross_check"}:
        return "evidence"
    if task_type == "verify_decisive_test":
        return "belief"
    return "workers"


def _task_result_summary(result: Any) -> str:
    return (
        f"{result.status}; evidence={len(result.evidence_items)}; "
        f"conflicts={len(result.evidence_conflicts)}; "
        f"verifiers={len(result.verifier_results)}"
    )


def _duration_ms(metadata: dict[str, str]) -> int | None:
    raw = metadata.get("duration_ms")
    if raw is None:
        return None
    try:
        return max(350, min(2400, int(float(raw))))
    except ValueError:
        return None


def _event_duration(kind: str) -> int:
    if kind in {"research_task", "research_result"}:
        return 950
    if kind in {"evidence_item", "generated_eval", "belief_update"}:
        return 760
    return 620


def _belief_update_source(assumption_id: str, state: ResearchState) -> str | None:
    for generated_eval in state.generated_evals[:6]:
        failure_id = generated_eval.source_failure_id
        if not failure_id:
            continue
        for leap in state.invalid_leaps[:6]:
            if leap.id == failure_id and assumption_id in leap.affected_assumption_ids:
                return generated_eval.id
    for leap in state.invalid_leaps[:6]:
        if assumption_id in leap.affected_assumption_ids:
            return leap.id
    for item in state.evidence_items[:8]:
        if assumption_id in item.assumption_ids:
            return item.id
    return None


def _worker_detail(remote_tasks: list[Any], all_tasks: list[Any]) -> str:
    if remote_tasks:
        return f"{len(remote_tasks)} Modal remote tasks"
    if all_tasks:
        return f"{len(all_tasks)} local tasks"
    return "No worker tasks yet"


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
