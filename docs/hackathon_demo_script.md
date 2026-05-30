# Pragmatic Hackathon Demo Script

## Setup

```bash
.venv/bin/python -m pragmatic doctor --run-openai-live --run-modal-remote
.venv/bin/python -m pragmatic demo-smoke --run-modal-remote --fail-on-fail
.venv/bin/python -m uvicorn realtime_app:app --host 0.0.0.0 --port 8501
```

## Three Minute Run

1. Ask a real question.
   - Use the main box, for example `Spider silk for bullet proof vests`.
   - Keep the default live SDK, Modal, live web, and Workshop settings.
   - Let the Thinking Graph stream live run events while the job is still in flight.

2. Narrate the live graph.
   - Point out source acquisition, Modal fan-out, typed evidence, invalid leaps, belief update, and Workshop recording as nodes appear.
   - Use the Live Trace panel as the plain-English event log.
   - Do not wait silently during the live SDK call; the graph is the demo.

3. Read the answer.
   - Show the verdict, confidence, source/evidence counts, and Modal task count.
   - Review the evidence table and limits/next-checks table.
   - Show the local trace and Workshop artifact paths.

4. Use controls only if needed.
   - For reliability, switch from live SDK to scripted SDK while keeping live web and Modal.
   - For backup, use prepared sources or the latest artifacts under `.pragmatic/demo` and `.pragmatic/doctor`.

## Backup

If any live service flakes, use the latest artifacts under `.pragmatic/demo` and `.pragmatic/doctor`.
