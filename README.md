# ThesisGraph

ThesisGraph is an autoresearch state machine for testing whether a technical thesis is supported by evidence. The first slice is deterministic: it turns a thesis into assumptions, retrieves from a prepared corpus, extracts typed evidence, detects invalid inference leaps, updates beliefs, and generates evals from failures.

See [THESISGRAPH_PRD.md](THESISGRAPH_PRD.md) for the product requirements.

## Milestone 1

This scaffold intentionally does not include live web search, Modal, Raindrop Workshop, or the OpenAI Agents SDK runtime yet. Those become required layers in later milestones after the local research-state loop is testable.

## Milestone 2

The OpenAI Agents SDK boundary is now represented in `src/thesisgraph/agents.py`. It exposes a `ResearchManager` facade and SDK `@function_tool` wrappers around the deterministic business functions. Tests use the offline deterministic path; live SDK execution requires an API key and should be added to app/CLI surfaces later.

## Milestone 3

The minimal Streamlit UI is available in `app.py`. It runs the deterministic `ResearchManager` path and renders the resulting `ResearchState`.

Run the app:

```bash
PYTHONPATH=src streamlit run app.py
```

## Milestone 4

Evidence extraction received the first local/Modal adapter boundary. `src/thesisgraph/extractors.py` owns deterministic extraction, and `src/thesisgraph/modal_jobs.py` defines the original Modal extraction payload shape. This milestone is now subsumed by the broader execution backend introduced in Milestone 7.

## Milestone 5

Observability now has a local/Raindrop adapter boundary. `src/thesisgraph/raindrop_client.py` writes Raindrop-compatible local trace artifacts by default and can send traces through the Raindrop Python SDK when `RAINDROP_WRITE_KEY` is configured. The Streamlit app defaults to local observability and shows the trace ID, backend, status, and generated eval artifact IDs.

## Milestone 6

The replay demo is now available through `src/thesisgraph/replay.py`, the OpenAI Agents SDK tool boundary, and the Streamlit `Replay demo` toggle. It simulates a first pass that over-credits benchmark results as direct discovery evidence, generates an eval from that failure, then replays the same thesis with the stricter proxy-evidence boundary so the before/after belief update is visible.

## Milestone 7

Modal is now modeled as a general research execution backend rather than an extraction-only switch. `src/thesisgraph/execution.py` defines typed research tasks, local execution, Modal-shaped execution, and fallback behavior.

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

Runs can now be saved and reloaded locally. `src/thesisgraph/persistence.py` stores complete `ResearchState` payloads under `.thesisgraph/runs`, maintains an index, and compares belief confidence deltas between runs. The Streamlit app adds run-history controls for saving the current run, loading a saved run, and comparing a saved baseline to the current belief graph.

## Milestone 14

The deterministic loop now has regression gates. `src/thesisgraph/eval_suite.py` runs fixture-style checks that protect the benchmark-proxy boundary, the prospective-validation support threshold, company-claim evidence classification, eval-workshop failure links, and replay confidence behavior. The Streamlit app adds a `Run Eval Suite` button, and the CLI can run the suite or export generated eval fixtures:

```bash
PYTHONPATH=src python -m thesisgraph eval-suite --fail-on-fail
PYTHONPATH=src python -m thesisgraph export-generated-evals .thesisgraph/generated_evals.json
```

## Milestone 15

Regression gates can now be frozen into a local eval corpus. `src/thesisgraph/eval_corpus.py` saves versioned known-good snapshots under `.thesisgraph/eval_corpus`, stores an index, and compares current behavior against a saved baseline. Snapshot comparison checks both gate status changes and generated eval fixture drift. The Streamlit app can save a passing snapshot and compare against it from the Evaluation sidebar.

```bash
PYTHONPATH=src python -m thesisgraph save-eval-snapshot
PYTHONPATH=src python -m thesisgraph list-eval-snapshots
PYTHONPATH=src python -m thesisgraph compare-eval-snapshot <snapshot-id> --fail-on-regression
```

Run tests:

```bash
PYTHONPATH=src pytest
```
