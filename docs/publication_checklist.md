# Public GitHub Checklist

Use this before switching the repository from private to public or announcing it.

## Already Covered

- Public-facing README explains the product, quickstart, architecture, live integration guardrails, and evidence boundaries.
- Apache-2.0 license is tracked in `LICENSE` and declared in package metadata.
- CI runs tests, bytecode compilation, and the committed eval-baseline check.
- Local generated directories are ignored: `.pragmatic/`, `.thesisgraph/`, `.playwright-mcp/`, `.pytest_cache/`, `.venv/`, and Python caches.
- Optional live credentials are represented by environment variables only.
- `.env.example` documents optional knobs without containing secrets.
- Live model paths use public OpenAI API credentials through `OPENAI_API_KEY`; the repo does not depend on the Codex CLI or a Codex benchmark harness.
- Search ranking is documented as lightweight deterministic term-overlap scoring, not a deep-research or authoritative evidence-ranking system.
- The realtime cockpit defaults to offline prepared-source/local execution so first-time visitors can play without API keys or Modal.

## Check Before Publishing

- Confirm that generated local artifacts under ignored directories are not force-added.
- Keep `artifacts/linkedin/` untracked unless those social-post assets are intentionally part of the repo.
- Run through `docs/playground.md` from a clean clone or fresh virtual environment.
- Re-run the public smoke path:

```bash
python -m pip install -e ".[dev]"
python -m pragmatic demo-smoke --fail-on-fail
python -m pytest
python -m pragmatic check-eval-baseline eval_baselines/default_v1.json --fail-on-regression
```

- If using live services for a public demo, run the doctor first:

```bash
python -m pragmatic doctor
```

- Confirm a live API-backed run only after setting `OPENAI_API_KEY`:

```bash
python -m pragmatic live-run-harness --live --allow-live-sdk --source-mode prepared --observability local
```

## Claim Boundaries To Preserve

- Prepared source packs are demo and regression fixtures, not scientific validation.
- Retrieval scores are transparent relevance heuristics, not evidence strength or source authority scores.
- Retrieval support, proxy benchmark evidence, cross-domain transfer, and prospective validation are different claim surfaces.
- The spider-silk demo should keep the tensile-toughness-to-ballistic-vest leap visibly marked as proxy evidence until a standards-relevant ballistic test exists.
