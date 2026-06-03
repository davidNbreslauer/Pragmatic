import json

from pragmatic import DEFAULT_THESIS
from pragmatic.cli import main
from pragmatic.eval_suite import (
    check_limiting_evidence_preserved,
    check_proxy_application_boundary,
    export_generated_eval_cases,
    run_eval_suite,
)
from pragmatic.replay import simulate_overcredited_first_pass
from pragmatic.research_loop import run_research_loop


def test_default_eval_suite_passes():
    result = run_eval_suite(DEFAULT_THESIS)

    assert result.status == "pass"
    assert result.passed == 5
    assert result.failed == 0


def test_proxy_gate_fails_on_overcredited_first_pass():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    overcredited = simulate_overcredited_first_pass(state)

    result = check_proxy_application_boundary(overcredited)

    assert result.status == "fail"
    assert "overcredited" in result.message


def test_limiting_evidence_gate_fails_when_limitations_are_removed():
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")
    without_limitations = state.model_copy(deep=True)
    for item in without_limitations.evidence_items:
        if item.evidence_type == "contradictory":
            item.evidence_type = "direct"

    result = check_limiting_evidence_preserved(without_limitations)

    assert result.status == "fail"
    assert result.details["limiting_items"] == "0"


def test_export_generated_eval_cases(tmp_path):
    state = run_research_loop(DEFAULT_THESIS, observability_mode="off")

    path = export_generated_eval_cases(state, tmp_path / "generated_evals.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload
    assert payload[0]["eval_rule"]
    assert payload[0]["expected_behavior"]


def test_cli_eval_suite_writes_output(tmp_path):
    output_path = tmp_path / "suite.json"

    exit_code = main(["eval-suite", "--output", str(output_path), "--fail-on-fail"])
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "pass"
    assert len(payload["results"]) == 5


def test_cli_exports_generated_eval_fixtures(tmp_path):
    output_path = tmp_path / "fixtures.json"

    exit_code = main(["export-generated-evals", str(output_path)])
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload
    assert payload[0]["failure_observed"]
