# Playground Guide

This is the fastest way to try Pragmatic without API keys, Modal, or hosted observability.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version  # should be 3.11 or newer
python -m pip install -e ".[dev]"
```

## 2. Run The Offline Smoke Check

```bash
python -m pragmatic demo-smoke --fail-on-fail
```

Expected result: `4/4 demo smoke checks passed.`

This writes replayable local artifacts under `.pragmatic/demo`, which is intentionally ignored by Git.

## 3. Open The Realtime Cockpit

```bash
python -m uvicorn realtime_app:app --host 127.0.0.1 --port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501).

The default run is:

- question: `Spider silk for bullet proof vests`
- orchestration: `scripted_sdk`
- execution: `local`
- sources: `prepared`
- observability: local Workshop bundle

That means the first click should not make OpenAI API calls, run Modal, or perform live web search.

## 4. What To Look For

In the spider-silk prepared run, Pragmatic should:

- separate material-property evidence from ballistic-vest performance evidence
- mark tensile toughness as proxy evidence, not proof of a finished vest
- surface contradictory or limiting evidence
- generate an eval from the invalid analogy
- propose a standards-relevant NIJ Level IIIA / V50 panel comparison against an aramid control

The point is not that the answer is final. The point is that the uncertainty is inspectable.

## 5. Try Live Paths Later

Live OpenAI API and Modal paths are available after the offline run is working.

```bash
export OPENAI_API_KEY="..."
python -m pragmatic live-run-harness --live --allow-live-sdk --source-mode prepared --observability local
```

For live web search, add:

```bash
--source-mode web --allow-live-web-search
```

For Modal, use:

```bash
--execution-backend modal
```
