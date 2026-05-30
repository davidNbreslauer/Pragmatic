# Pragmatic

Pragmatic is an autoresearch state machine for testing whether a technical thesis is supported by evidence. The first slice is deterministic: it turns a thesis into assumptions, retrieves from a prepared corpus, extracts typed evidence, detects invalid inference leaps, updates beliefs, and generates evals from failures.

See [PRAGMATIC_PRD.md](PRAGMATIC_PRD.md) for the product requirements.

## Milestone 1

This scaffold intentionally does not include live web search, Modal, Raindrop Workshop, or the OpenAI Agents SDK runtime yet. Those become required layers in later milestones after the local research-state loop is testable.

## Milestone 2

The OpenAI Agents SDK boundary is now represented in `src/pragmatic/agents.py`. It exposes a `ResearchManager` facade and SDK `@function_tool` wrappers around the deterministic business functions. Tests use the offline deterministic path; live SDK execution requires an API key and should be added to app/CLI surfaces later.

## Milestone 3

The minimal Streamlit UI is available in `app.py`. It runs the deterministic `ResearchManager` path and renders the resulting `ResearchState`.

Run the Streamlit app:

```bash
PYTHONPATH=src streamlit run app.py
```

## Milestone 4

Evidence extraction received the first local/Modal adapter boundary. `src/pragmatic/extractors.py` owns deterministic extraction, and `src/pragmatic/modal_jobs.py` defines the original Modal extraction payload shape. This milestone is now subsumed by the broader execution backend introduced in Milestone 7.

## Milestone 5

Observability now has a local/Raindrop adapter boundary. `src/pragmatic/raindrop_client.py` writes Raindrop-compatible local trace artifacts by default and can send traces through the Raindrop Python SDK when `RAINDROP_WRITE_KEY` is configured. The Streamlit app defaults to local observability and shows the trace ID, backend, status, and generated eval artifact IDs.

## Milestone 6

The replay demo is now available through `src/pragmatic/replay.py`, the OpenAI Agents SDK tool boundary, and the Streamlit `Replay demo` toggle. It simulates a first pass that over-credits benchmark results as direct discovery evidence, generates an eval from that failure, then replays the same thesis with the stricter proxy-evidence boundary so the before/after belief update is visible.

## Milestone 7

Modal is now modeled as a general research execution backend rather than an extraction-only switch. `src/pragmatic/execution.py` defines typed research tasks, local execution, Modal-shaped execution, and fallback behavior.

## Milestone 8

Prepared-corpus processing now fans out into one typed extraction task per source. The loop merges task results deterministically and records task-level trace events in `ResearchState`.

## Milestone 9

Cross-source evidence checking now produces typed `EvidenceConflict` artifacts. These conflicts are shown in the UI, feed invalid-leap detection, and apply confidence penalties during belief updates.

## Milestone 10

Decisive tests now have a deterministic verifier harness. The loop executes verifier tasks, records typed `VerifierResult` artifacts, and applies verifier confidence adjustments to the final belief graph.

## Milestone 11

OpenAI Agents SDK orchestration now has a schema-constrained agent and a testable SDK-scripted path. `ResearchManager.run_sdk_orchestrated()` invokes the canonical research-loop tool through the SDK function-tool boundary, validates the output as `ResearchState`, and records an `AgentRunRecord`. `ResearchManager.run_live()` is ready to use `Runner.run()` for a live SDK run when API credentials are configured.

## Milestone 12

Raindrop/local observability now records an eval workshop, not just a run summary. Trace payloads include task spans, evidence conflicts, verifier results, failure-to-eval links, and replay outcomes. The Streamlit app shows an `Eval Workshop` section so the failure -> eval -> replay chain is visible without opening raw JSON.

## Milestone 13

Runs can now be saved and reloaded locally. `src/pragmatic/persistence.py` stores complete `ResearchState` payloads under `.pragmatic/runs`, maintains an index, and compares belief confidence deltas between runs. The Streamlit app adds run-history controls for saving the current run, loading a saved run, and comparing a saved baseline to the current belief graph.

## Milestone 14

The deterministic loop now has regression gates. `src/pragmatic/eval_suite.py` runs fixture-style checks that protect the benchmark-proxy boundary, the prospective-validation support threshold, company-claim evidence classification, eval-workshop failure links, and replay confidence behavior. The Streamlit app adds a `Run Eval Suite` button, and the CLI can run the suite or export generated eval fixtures:

```bash
PYTHONPATH=src python -m pragmatic eval-suite --fail-on-fail
PYTHONPATH=src python -m pragmatic export-generated-evals .pragmatic/generated_evals.json
```

## Milestone 15

Regression gates can now be frozen into a local eval corpus. `src/pragmatic/eval_corpus.py` saves versioned known-good snapshots under `.pragmatic/eval_corpus`, stores an index, and compares current behavior against a saved baseline. Snapshot comparison checks both gate status changes and generated eval fixture drift. The Streamlit app can save a passing snapshot and compare against it from the Evaluation sidebar.

```bash
PYTHONPATH=src python -m pragmatic save-eval-snapshot
PYTHONPATH=src python -m pragmatic list-eval-snapshots
PYTHONPATH=src python -m pragmatic compare-eval-snapshot <snapshot-id> --fail-on-regression
```

## Milestone 16

The eval corpus now has a committed known-good baseline and a CI-ready regression gate. `eval_baselines/default_v1.json` is generated through the canonical baseline exporter, and `check-eval-baseline` compares the current deterministic loop against that tracked baseline. CI runs tests, bytecode compilation, and the baseline check.

```bash
PYTHONPATH=src python -m pragmatic export-eval-baseline eval_baselines/default_v1.json
PYTHONPATH=src python -m pragmatic check-eval-baseline eval_baselines/default_v1.json --fail-on-regression
PYTHONPATH=src python -m pragmatic check-eval-baseline eval_baselines/default_v1.json --fail-on-change
```

## Milestone 17

Prepared-corpus ingestion now carries structured source metadata and deterministic retrieval scoring. `Source` records include published year, tags, and evidence scope, while `RetrievalScore` records capture source/question matches, matched terms, scores, and rationales. The research loop ranks prepared sources by score, records the score matrix in `ResearchState`, and the Streamlit app shows a `Retrieval Scores` table.

## Milestone 18

Live OpenAI Agents SDK orchestration is now available as an explicit opt-in path. `ResearchManager.run_live_sync()` and `run-live-sdk` CLI execution require both `OPENAI_API_KEY` and an explicit `allow_live_sdk` confirmation, then validate the final output as `ResearchState` and record a `live_sdk` `AgentRunRecord`. The Streamlit app exposes `live_sdk` mode behind an `Enable live SDK calls` checkbox.

```bash
PYTHONPATH=src python -m pragmatic run-live-sdk --allow-live-sdk --observability off --output .pragmatic/live_state.json
```

## Milestone 19

Live SDK execution now has a guardrailed harness. `live-run-harness` defaults to dry run, records whether credentials are available without exposing them, enforces prepared-corpus-only/no-web-search policy, caps max turns and timeout, and writes a replayable JSON artifact under `.pragmatic/live_runs`. The Streamlit `live_sdk` mode now shows the same harness status before any real API call.

```bash
PYTHONPATH=src python -m pragmatic live-run-harness
PYTHONPATH=src python -m pragmatic live-run-harness --live --allow-live-sdk --max-turns 4 --timeout-seconds 60 --observability local
```

## Milestone 20

Modal execution now ships the local `pragmatic` package into the remote worker image, applies worker timeout/retry guardrails, and maps typed `ResearchTask` payloads through Modal with ordered parallel fan-out. Individual remote worker exceptions are converted into typed failed `ResearchTaskResult` records, while full Modal unavailability still falls back to the local executor when requested. Batch metadata now records task counts, success/failure counts, task types, Modal app name, timeout, and retry settings.

## Milestone 21

Raindrop Workshop observability now writes a durable workshop bundle alongside the chronological trace. Local and Raindrop modes share the same failure/eval/replay payload shape, including failure artifacts, generated eval artifacts linked to source failures, replay artifacts, and a Raindrop event plan. The Streamlit observability section now surfaces the workshop bundle path and failure artifact IDs.

## Milestone 22

OpenAI Agents SDK scripted orchestration now runs through visible specialist tool steps instead of invoking the canonical loop as one opaque tool call. `ResearchManager.run_sdk_orchestrated()` records specialist steps for assumption decomposition, question planning, retrieval, retrieval scoring, backend execution fan-out, cross-checking, invalid-leap detection, belief updates, decisive-test writing, verifier execution, eval writing, workshop assembly, and observability recording. The live SDK prompt now instructs the same specialist-tool order while preserving the prepared-corpus-only/no-live-web-search guardrail.

## Milestone 23

Live-demo integration readiness now has a first-class doctor. `run_integration_doctor()` checks OpenAI Agents SDK import/credential readiness, Modal installation/profile readiness with an optional live remote smoke task, and local Raindrop Workshop bundle generation without requiring a hosted write key. The CLI and Streamlit app expose the same checks so the demo can show which layers are live versus skipped or unavailable.

```bash
PYTHONPATH=src python -m pragmatic doctor
PYTHONPATH=src python -m pragmatic doctor --run-openai-live --run-modal-remote
```

## Milestone 24

Modal-backed execution is now visible as a broader research fan-out layer. Prepared corpus sources pass through typed `parse_source` tasks before `extract_evidence` tasks, and the same execution backend also runs cross-checking and verifier tasks. Each `ResearchTaskResult` records worker status, duration, source counts, and output counts. The Streamlit app shows a `Research Execution Tasks` table so a demo viewer can see which work ran locally or through Modal.

## Milestone 25

Raindrop Workshop artifacts now include a connection layer that makes the demo trace readable: specialist SDK steps, Modal/local task spans, failure artifacts, generated evals, and replay outcomes are linked through stable IDs. Generated evals carry their source failure ID, task spans include agent/tool and worker metadata, the workshop bundle includes specialist/task artifacts plus connection rows, and the Streamlit Eval Workshop panel renders the chain directly.

## Milestone 26

The live SDK harness now produces a demo-readiness proof. Successful live runs include a `LiveRunProof` showing schema validation, prepared-corpus guardrails, Modal task counts, Workshop observability, trace paths, generated eval counts, invalid leap counts, and replay outcome counts. `--require-demo-proof` can fail a live run when the requested Modal/Workshop proof is missing, and the Streamlit live harness panel shows the same proof metrics.

```bash
PYTHONPATH=src python -m pragmatic live-run-harness --live --allow-live-sdk --execution-backend modal --observability local --require-demo-proof
```

## Milestone 27

The Streamlit app now opens as a hackathon cockpit. Curated demo scenarios set the thesis, orchestration mode, execution backend, observability backend, replay toggle, and live proof defaults. The first visible panels prioritize the Demo Cockpit, integration status, live proof, agent orchestration, research execution tasks, Eval Workshop, replay, invalid leaps, and observability before lower-level tables.

## Milestone 28

The repo now includes a curated scenario pack in `src/pragmatic/demo.py`: Core Evidence Loop, Modal Fan-Out, Failure To Eval Replay, and Live SDK Guarded. The same scenarios are exposed through the app and the CLI.

```bash
PYTHONPATH=src python -m pragmatic demo-scenarios
```

## Milestone 29

Demo reliability now has a one-command smoke harness. It runs the integration doctor, core loop, replay demo, and regression gates, then writes replayable artifacts under `.pragmatic/demo`.

```bash
PYTHONPATH=src python -m pragmatic demo-smoke --fail-on-fail
```

## Milestone 30

The hackathon demo script is tracked in `docs/hackathon_demo_script.md`, including setup commands, a three-minute run order, and backup artifact paths for live-service failures.

## Milestone 31

Arbitrary-thesis research now has a real-source path. Non-demo theses are decomposed into generic evidence assumptions, the OpenAI web-search adapter can normalize live search results into `Source` records, generic extraction works for arbitrary source IDs, and the skeptic/verifier layers generate failure/eval artifacts for proxy-to-application leaps. The Streamlit app exposes `Evidence search`, `Allow live web search`, web-search model, and max-source controls, and the Sources panel lets the run output be inspected before evidence, belief updates, and Raindrop Workshop traces.

```bash
PYTHONPATH=src python -m pragmatic live-run-harness \
  --live \
  --allow-live-sdk \
  --source-mode web \
  --allow-live-web-search \
  --thesis "Can spider silk make a bullet proof vest?" \
  --execution-backend modal \
  --observability local \
  --require-demo-proof
```

## Milestone 32

The user-facing demo now has a realtime web app in `realtime_app.py`. It keeps the same live OpenAI Agents SDK, Modal, live web evidence, and local Raindrop Workshop execution path, but streams actual progress events to the browser with a live graph while the run is in flight.

```bash
PYTHONPATH=src python -m uvicorn realtime_app:app --host 0.0.0.0 --port 8501
```

Run tests:

```bash
PYTHONPATH=src pytest
```

## Milestone 33

The realtime app is now a research cockpit instead of a fixed pipeline diagram. The SSE envelope remains backward-compatible, but events may include a top-level `kind` plus structured metadata for richer animation:

- `reasoning.delta`: streamed model text for the Thinking pane.
- `tool.call` / `tool.output`: inline tool chips in the transcript.
- `fanout.spawn` / `fanout.task`: Modal/local worker cards that spawn and resolve.
- `node.add` / `edge.add`: belief graph growth.
- `node.confidence`: confidence recoloring and self-correction pulses.
- `counter`: cumulative sources, evidence, leaps, conflicts, and tests.

The cockpit has three zones: Thinking transcript, Belief Graph, and Fan-out/Counters. Live SDK runs use `Runner.run_streamed`; deterministic, scripted, replay, and recovered-timeout paths also emit graph snapshots so the demo remains inspectable when live services are slow.
