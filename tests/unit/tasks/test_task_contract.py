from datetime import UTC, datetime

from memopilot.tasks.background import BackgroundTask


def test_background_task_is_serializable_without_operational_job_fields() -> None:
    task = BackgroundTask(
        task_id="execution-1",
        kind="schedule.run",
        priority=1,
        session_key="feishu:chat-1",
        payload={"message": "提醒"},
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )

    assert task.payload_json == '{"message": "提醒"}'
    assert not hasattr(task, "run_id")
