from __future__ import annotations

from pathlib import Path

from memopilot.app.service import AppService
from memopilot.bootstrap import build_app, build_worker
from memopilot.config import MemoPilotSettings
from memopilot.worker.service import WorkerService


async def test_phase3_builds_separate_app_and_worker_without_starting_scheduler(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path / "workspace",
        chat_api_key="chat-secret",
        feishu_app_id="cli-app",
        feishu_app_secret="feishu-secret",
        _env_file=None,
    )

    app = build_app(settings)
    worker = build_worker(settings)

    assert isinstance(app.service, AppService)
    assert isinstance(worker.service, WorkerService)
    assert settings.operational_database.exists()
    assert not hasattr(app, "scheduler")
    assert not hasattr(worker, "scheduler")
    await app.close()
    await worker.close()
