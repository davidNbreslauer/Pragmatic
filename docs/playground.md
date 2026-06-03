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
- `Use Modal`: off
- `Use Raindrop`: off

That means the first click should not make OpenAI API calls, run Modal, or perform live web search.

The offline prepared corpus is the spider-silk demo corpus. If you type an unrelated topic while `Sources` is still `prepared` and `Corpus` is still `auto`, Pragmatic will block the run instead of silently reusing spider-silk evidence.

## 4. What To Look For

In the spider-silk prepared run, Pragmatic should:

- separate material-property evidence from ballistic-vest performance evidence
- mark tensile toughness as proxy evidence, not proof of a finished vest
- surface contradictory or limiting evidence
- generate an eval from the invalid analogy
- propose a standards-relevant NIJ Level IIIA / V50 panel comparison against an aramid control

The point is not that the answer is final. The point is that the uncertainty is inspectable.

## 5. Try Arbitrary Topics With Live Web Search

Meaningful arbitrary-topic runs need live source acquisition:

```bash
export OPENAI_API_KEY="..."
python -m pragmatic run \
  --thesis "Room-temperature superconductors are ready for commercial grid storage" \
  --source-mode web \
  --allow-live-web-search \
  --max-web-sources 5 \
  --output .pragmatic/live_web_state.json
```

In the realtime cockpit, switch `Sources` to `live web` before asking an unrelated question.

## 6. Try Heavier Live Paths Later

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
