# Pragmatic Hackathon Demo Script

## Setup

```bash
.venv/bin/python -m pragmatic doctor --run-openai-live --run-modal-remote
.venv/bin/python -m pragmatic demo-smoke --run-modal-remote --fail-on-fail
.venv/bin/python -m streamlit run app.py --server.port 8501
```

## Three Minute Run

1. Open with the `Core Evidence Loop` scenario.
   - Show the Demo Cockpit.
   - Point to assumptions, typed evidence, invalid leaps, and generated evals.

2. Switch to `Modal Fan-Out`.
   - Run the Integration Doctor with remote Modal enabled.
   - Run Pragmatic with execution set to `modal`.
   - Show `Research Execution Tasks` and the Modal task count.

3. Show the Raindrop Workshop story.
   - In `Eval Workshop`, show connection rows.
   - Follow one row from task/span to failure to generated eval.
   - Show the local Workshop bundle path in Observability.

4. Switch to `Failure To Eval Replay`.
   - Run with Replay enabled.
   - Show A5/A6 confidence before and after replay.
   - Show the generated benchmark-proxy rule.

5. Close with `Live SDK Guarded`.
   - Start in dry-run.
   - If live services are healthy, uncheck dry-run and keep `Require demo proof` enabled.
   - Show whether the live proof passed or why it failed.

## Backup

If any live service flakes, use the latest artifacts under `.pragmatic/demo` and `.pragmatic/doctor`.
