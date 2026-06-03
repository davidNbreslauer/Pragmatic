from pragmatic import DEFAULT_THESIS
from pragmatic.replay import run_replay_demo


def test_replay_demo_treats_proxy_evidence_more_strictly_after_eval():
    replay = run_replay_demo(DEFAULT_THESIS, observability_mode="off")

    overcredited_types = {
        item.source_id: item.evidence_type
        for item in replay.first_pass.evidence_items
        if "First-pass failure" in item.limitation
    }
    replay_proxy_types = {
        item.source_id: item.evidence_type
        for item in replay.replay_pass.evidence_items
        if item.source_id in overcredited_types
    }

    assert overcredited_types
    assert all(evidence_type == "direct" for evidence_type in overcredited_types.values())
    assert any(evidence_type in {"proxy", "indirect", "anecdotal"} for evidence_type in replay_proxy_types.values())
    assert any("directly measures" in rule.lower() for rule in replay.applied_eval_rules)


def test_replay_demo_downgrades_prospective_validation_confidence():
    replay = run_replay_demo(DEFAULT_THESIS, observability_mode="off")
    first_assumptions = {assumption.id: assumption for assumption in replay.first_pass.assumptions}
    replay_assumptions = {assumption.id: assumption for assumption in replay.replay_pass.assumptions}

    assert replay_assumptions["A5"].confidence < first_assumptions["A5"].confidence
    assert replay_assumptions["A6"].support_level == "unsupported"
    assert any(comparison.assumption_id == "A5" for comparison in replay.comparisons)


def test_replay_result_serializes_to_json():
    replay = run_replay_demo(DEFAULT_THESIS, observability_mode="off")

    serialized = replay.model_dump_json()

    assert "first_pass" in serialized
    assert "replay_pass" in serialized
    assert "First pass over-credited" in serialized


def test_replay_demo_records_workshop_replay_outcomes():
    replay = run_replay_demo(DEFAULT_THESIS, observability_mode="off")

    assert replay.eval_workshop is not None
    assert replay.replay_pass.eval_workshop is not None
    assert replay.eval_workshop.replay_outcomes
    assert replay.eval_workshop.connection_rows
    assert any(link.link_type == "replay_eval_to_outcome" for link in replay.eval_workshop.failure_eval_links)
    assert all(outcome.passed for outcome in replay.eval_workshop.replay_outcomes)
    assert replay.replay_pass.observability is not None
    assert replay.replay_pass.observability.backend == "off"
