import json

from thesisgraph import DEFAULT_THESIS
from thesisgraph.cli import main
from thesisgraph.eval_corpus import (
    compare_eval_baseline,
    compare_eval_snapshot,
    create_eval_snapshot,
    load_eval_baseline,
    list_eval_snapshots,
    load_eval_snapshot,
    save_eval_snapshot,
    write_eval_baseline,
)


def test_save_list_load_eval_snapshot_round_trips(tmp_path):
    summary = save_eval_snapshot(
        DEFAULT_THESIS,
        corpus_dir=tmp_path,
        snapshot_id="eval_known_good",
    )
    loaded = load_eval_snapshot("eval_known_good", corpus_dir=tmp_path)
    snapshots = list_eval_snapshots(corpus_dir=tmp_path)

    assert summary.snapshot_id == "eval_known_good"
    assert summary.suite_status == "pass"
    assert summary.generated_eval_count == len(loaded.generated_eval_fixtures)
    assert loaded.suite_result.status == "pass"
    assert snapshots[0].snapshot_id == "eval_known_good"


def test_compare_eval_snapshot_matches_identical_current_snapshot():
    baseline = create_eval_snapshot(DEFAULT_THESIS, snapshot_id="baseline")
    current = baseline.model_copy(deep=True)
    current.snapshot_id = "current"

    comparison = compare_eval_snapshot(baseline, current)

    assert comparison.status == "match"
    assert all(not delta.changed for delta in comparison.case_deltas)
    assert all(delta.status == "same" for delta in comparison.fixture_deltas)


def test_compare_eval_snapshot_detects_passing_gate_regression():
    baseline = create_eval_snapshot(DEFAULT_THESIS, snapshot_id="baseline")
    current = baseline.model_copy(deep=True)
    current.snapshot_id = "current"
    current.suite_result.results[0].status = "fail"
    current.suite_result.results[0].message = "Benchmark gate failed in fixture."

    comparison = compare_eval_snapshot(baseline, current)

    assert comparison.status == "regression"
    assert any(delta.regression for delta in comparison.case_deltas)


def test_compare_eval_snapshot_detects_generated_eval_fixture_change():
    baseline = create_eval_snapshot(DEFAULT_THESIS, snapshot_id="baseline")
    current = baseline.model_copy(deep=True)
    current.snapshot_id = "current"
    current.generated_eval_fixtures[0].eval_rule = "Changed eval rule."

    comparison = compare_eval_snapshot(baseline, current)

    assert comparison.status == "changed"
    assert comparison.fixture_deltas[0].status == "changed"


def test_write_eval_baseline_uses_stable_metadata(tmp_path):
    baseline_path = write_eval_baseline(tmp_path / "default_v1.json")

    baseline = load_eval_baseline(baseline_path)

    assert baseline.snapshot_id == "default_v1"
    assert baseline.created_at == "2026-05-30T00:00:00+00:00"
    assert baseline.suite_result.id == "eval_suite_default_v1"
    assert baseline.suite_result.created_at == "2026-05-30T00:00:00+00:00"
    assert baseline.suite_result.status == "pass"


def test_compare_eval_baseline_matches_current_behavior(tmp_path):
    baseline_path = write_eval_baseline(tmp_path / "default_v1.json")

    comparison = compare_eval_baseline(baseline_path)

    assert comparison.status == "match"
    assert comparison.summary == "Current eval behavior matches the saved snapshot."


def test_cli_saves_lists_and_compares_eval_snapshot(tmp_path):
    save_code = main(
        [
            "save-eval-snapshot",
            "--corpus-dir",
            str(tmp_path),
            "--snapshot-id",
            "eval_cli_good",
        ]
    )
    list_code = main(["list-eval-snapshots", "--corpus-dir", str(tmp_path)])
    output_path = tmp_path / "comparison.json"
    compare_code = main(
        [
            "compare-eval-snapshot",
            "eval_cli_good",
            "--corpus-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--fail-on-regression",
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert save_code == 0
    assert list_code == 0
    assert compare_code == 0
    assert payload["status"] == "match"


def test_cli_exports_and_checks_eval_baseline(tmp_path):
    baseline_path = tmp_path / "default_v1.json"

    export_code = main(["export-eval-baseline", str(baseline_path)])
    check_code = main(
        [
            "check-eval-baseline",
            str(baseline_path),
            "--fail-on-regression",
        ]
    )

    assert export_code == 0
    assert check_code == 0


def test_cli_check_eval_baseline_can_fail_on_fixture_change(tmp_path):
    baseline_path = tmp_path / "default_v1.json"
    write_eval_baseline(baseline_path)
    baseline = load_eval_baseline(baseline_path)
    baseline.generated_eval_fixtures[0].eval_rule = "Changed expected rule."
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")

    exit_code = main(
        [
            "check-eval-baseline",
            str(baseline_path),
            "--fail-on-change",
        ]
    )

    assert exit_code == 1
