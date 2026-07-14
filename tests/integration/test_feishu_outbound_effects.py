from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from memopilot.channels.contracts import SendReceipt
from memopilot.delivery.effects import EffectRepository, EffectTransition
from memopilot.delivery.feishu import FinalResponseDispatcher
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import InboundCommand, OperationalRepository

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Provider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        return ModelResponse(content="完整回复", tool_calls=(), finish_reason="stop")


def _claimed(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "你好", "chat_id": "chat-1"},
            received_at=NOW,
        )
    )
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    return repository, EffectRepository(database), lease, claim


@pytest.mark.asyncio
async def test_job_succeeds_only_after_feishu_confirms_message(tmp_path: Path) -> None:
    repository, effects, lease, claim = _claimed(tmp_path)
    transport = AsyncMock()
    transport.send.return_value = SendReceipt(message_id="om_confirmed")
    dispatcher = FinalResponseDispatcher(repository, effects, transport, clock=lambda: NOW)
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=dispatcher,
        clock=lambda: NOW,
    )

    result = await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key=claim.session_key, content="你好"),
        now=NOW,
    )

    assert result.reply == "完整回复"
    assert repository.get_job(claim.job_id).state == "succeeded"  # type: ignore[union-attr]
    effect = effects.for_run(claim.run_id)
    assert effect is not None
    assert effect.state == "confirmed"
    assert effect.message_id == "om_confirmed"
    transport.send.assert_awaited_once_with(
        "chat-1",
        "完整回复",
        provider_uuid=effect.provider_uuid,
    )


@pytest.mark.asyncio
async def test_response_loss_marks_effect_and_job_needs_review_without_blind_retry(
    tmp_path: Path,
) -> None:
    repository, effects, lease, claim = _claimed(tmp_path)
    transport = AsyncMock()
    transport.send.side_effect = httpx.ReadTimeout("response lost")
    dispatcher = FinalResponseDispatcher(repository, effects, transport, clock=lambda: NOW)
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=dispatcher,
        clock=lambda: NOW,
    )

    await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key=claim.session_key, content="你好"),
        now=NOW,
    )

    assert repository.get_job(claim.job_id).state == "needs_review"  # type: ignore[union-attr]
    assert effects.for_run(claim.run_id).state == "unknown"  # type: ignore[union-attr]
    assert transport.send.await_count == 1


@pytest.mark.asyncio
async def test_explicit_reconciliation_retry_reuses_uuid_and_resolves_job(
    tmp_path: Path,
) -> None:
    repository, effects, lease, claim = _claimed(tmp_path)
    current = [NOW]
    transport = AsyncMock()
    transport.send.side_effect = [
        httpx.ReadTimeout("response lost"),
        SendReceipt(message_id="om_retried"),
    ]
    dispatcher = FinalResponseDispatcher(
        repository,
        effects,
        transport,
        clock=lambda: current[0],
    )
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=dispatcher,
        clock=lambda: current[0],
    )
    await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key=claim.session_key, content="你好"),
        now=NOW,
    )
    effect = effects.for_run(claim.run_id)
    assert effect is not None
    first_uuid = transport.send.await_args_list[0].kwargs["provider_uuid"]

    current[0] = NOW + timedelta(minutes=30)
    result = await dispatcher.retry_unknown(effect.operation_id, lease=lease)

    assert result.outcome == "confirmed"
    assert transport.send.await_count == 2
    assert transport.send.await_args_list[1].kwargs["provider_uuid"] == first_uuid
    assert effects.get(effect.operation_id).message_id == "om_retried"  # type: ignore[union-attr]
    assert repository.get_job(claim.job_id).state == "succeeded"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_confirmed_effect_closes_job_after_reconciliation_process_crash(
    tmp_path: Path,
) -> None:
    repository, effects, lease, claim = _claimed(tmp_path)
    current = [NOW]
    transport = AsyncMock()
    transport.send.side_effect = httpx.ReadTimeout("response lost")
    dispatcher = FinalResponseDispatcher(
        repository,
        effects,
        transport,
        clock=lambda: current[0],
    )
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=dispatcher,
        clock=lambda: current[0],
    )
    await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key=claim.session_key, content="你好"),
        now=NOW,
    )
    effect = effects.for_run(claim.run_id)
    assert effect is not None
    current[0] = NOW + timedelta(minutes=10)
    assert (
        effects.begin_reconciliation(effect.operation_id, lease=lease, now=current[0])
        is EffectTransition.SEND
    )
    effects.mark_confirmed(
        effect.operation_id,
        lease=lease,
        message_id="om_already_sent",
        now=current[0],
    )
    transport.reset_mock()

    result = await dispatcher.retry_unknown(effect.operation_id, lease=lease)

    assert result.outcome == "confirmed"
    transport.send.assert_not_awaited()
    assert repository.get_job(claim.job_id).state == "succeeded"  # type: ignore[union-attr]
