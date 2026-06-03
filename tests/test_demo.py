import json

from pragmatic.cli import main
from pragmatic.demo import demo_scenario_by_id, demo_scenarios, run_demo_smoke


def test_demo_scenarios_cover_core_modal_replay_and_live():
    scenarios = demo_scenarios()
    scenario_ids = {scenario.id for scenario in scenarios}

    assert scenarios[0].id == "spider_silk_prepared"
    assert {
        "spider_silk_prepared",
        "live_full",
        "core_loop",
        "modal_fanout",
        "failure_replay",
        "live_guarded",
    }.issubset(
        scenario_ids
    )
    live_full = demo_scenario_by_id("live_full")
    assert live_full.orchestration == "live_sdk"
    assert live_full.execution_backend == "modal"
    assert live_full.live_dry_run is False
    assert live_full.require_demo_proof is True
    assert demo_scenario_by_id("modal_fanout").execution_backend == "modal"
    assert demo_scenario_by_id("failure_replay").replay_demo is True


def test_demo_smoke_writes_replayable_artifacts(tmp_path):
    result = run_demo_smoke(output_dir=tmp_path)

    assert result.status == "pass"
    assert (tmp_path / "demo_smoke.json").exists()
    assert {check.name for check in result.checks} == {
        "integration_doctor",
        "core_loop",
        "failure_replay",
        "regression_gates",
    }
    assert all(check.artifact_path for check in result.checks)


def test_cli_demo_scenarios_and_smoke(tmp_path, capsys):
    scenarios_path = tmp_path / "scenarios.json"
    exit_code = main(["demo-scenarios", "--output", str(scenarios_path)])

    assert exit_code == 0
    payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
    assert payload[0]["id"] == "spider_silk_prepared"

    exit_code = main(["demo-smoke", "--output-dir", str(tmp_path / "smoke"), "--fail-on-fail"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "pass"' in captured.out
