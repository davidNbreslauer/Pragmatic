from pragmatic.doctor import run_integration_doctor
from pragmatic.schemas import ResearchBatchResult, ResearchTaskResult


def test_integration_doctor_reports_ready_without_live_checks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pragmatic.doctor._modal_profile_name", lambda: "test-profile")

    result = run_integration_doctor(run_openai_live=False, run_modal_remote=False)

    checks = {check.layer: check for check in result.checks}
    assert result.status == "ready"
    assert checks["openai_agents_sdk"].status == "ready"
    assert checks["modal"].status == "ready"
    assert checks["raindrop_workshop"].status == "live"
    assert checks["raindrop_workshop"].artifact_path


def test_integration_doctor_can_report_live_modal_smoke(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pragmatic.doctor._modal_profile_name", lambda: "test-profile")

    def fake_execute_research_tasks(tasks, *, backend, fallback_to_local):
        assert backend == "modal"
        assert fallback_to_local is False
        task = tasks[0]
        return ResearchBatchResult(
            backend="modal",
            attempted_backend="modal",
            results=[
                ResearchTaskResult(
                    task_id=task.id,
                    task_type=task.task_type,
                    backend="modal",
                    status="succeeded",
                    source_ids=["source_001"],
                    metadata={"operation": "test"},
                )
            ],
            metadata={"task_count": "1", "succeeded": "1"},
        )

    monkeypatch.setattr("pragmatic.doctor.execute_research_tasks", fake_execute_research_tasks)

    result = run_integration_doctor(run_modal_remote=True)

    modal_check = next(check for check in result.checks if check.layer == "modal")
    assert modal_check.status == "live"
    assert modal_check.live is True
    assert modal_check.metadata["task_type"] == "parse_source"


def test_integration_doctor_is_degraded_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pragmatic.doctor._modal_profile_name", lambda: "test-profile")

    result = run_integration_doctor()

    openai_check = next(check for check in result.checks if check.layer == "openai_agents_sdk")
    assert result.status == "degraded"
    assert openai_check.status == "unavailable"
