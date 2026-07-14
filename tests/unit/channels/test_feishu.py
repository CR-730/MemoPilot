from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memopilot.channels.base import AttachmentStore, SessionIdentityIndex
from memopilot.channels.contracts import InterruptAcknowledgement, MessageBus
from memopilot.channels.feishu import FeishuChannel
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import OperationalRepository


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def _event(
    *,
    event_id: str = "event-1",
    message_id: str = "message-1",
    sender_ids: dict[str, str] | None = None,
    chat_type: str = "p2p",
    message_type: str = "text",
    content: dict[str, object] | str | None = None,
) -> dict[str, object]:
    return {
        "schema": "2.0",
        "header": {"event_id": event_id, "event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_id": sender_ids or {"open_id": "ou_user"}},
            "message": {
                "message_id": message_id,
                "chat_id": "chat-1",
                "chat_type": chat_type,
                "message_type": message_type,
                "content": json.dumps(content or {"text": "你好"}, ensure_ascii=False)
                if not isinstance(content, str)
                else content,
            },
        },
    }


def _channel(tmp_path: Path, **overrides: object) -> tuple[FeishuChannel, MessageBus]:
    bus = MessageBus()
    repository = _repository(tmp_path)
    values: dict[str, object] = {
        "app_id": "cli_test",
        "app_secret": "secret",
        "bus": bus,
        "identity_index": SessionIdentityIndex(repository, channel="feishu"),
        "attachment_store": AttachmentStore(tmp_path / "uploads"),
        "allow_from": ("u_allowed",),
    }
    values.update(overrides)
    return FeishuChannel(**values), bus  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_private_text_is_normalized_and_any_sender_id_can_match_allowlist(
    tmp_path: Path,
) -> None:
    channel, bus = _channel(tmp_path)

    result = await channel.handle_event(
        _event(sender_ids={"open_id": "ou_user", "user_id": "u_allowed", "union_id": "on_user"})
    )
    message = await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    assert result == {"ok": True}
    assert message.channel == "feishu"
    assert message.sender == "ou_user"
    assert message.session_key == "feishu:chat-1"
    assert message.content == "你好"
    assert message.metadata == {
        "message_id": "message-1",
        "event_id": "event-1",
        "chat_type": "p2p",
        "message_type": "text",
        "open_id": "ou_user",
        "user_id": "u_allowed",
        "union_id": "on_user",
    }


@pytest.mark.asyncio
async def test_duplicate_unauthorized_and_non_private_events_are_not_published(
    tmp_path: Path,
) -> None:
    channel, bus = _channel(tmp_path)
    allowed = _event(sender_ids={"open_id": "u_allowed"})

    assert await channel.handle_event(allowed) == {"ok": True}
    assert await channel.handle_event(allowed) == {"ok": True, "deduped": True}
    assert (
        await channel.handle_event(
            _event(event_id="event-2", message_id="message-2", sender_ids={"open_id": "other"})
        )
    ) == {"ok": True, "ignored": "unauthorized"}
    assert (
        await channel.handle_event(
            _event(
                event_id="event-3",
                message_id="message-3",
                sender_ids={"open_id": "u_allowed"},
                chat_type="group",
            )
        )
    ) == {"ok": True, "ignored": "non_private"}

    assert bus.inbound_size == 1


@pytest.mark.asyncio
async def test_failed_persistence_releases_process_dedupe_for_sdk_redelivery(
    tmp_path: Path,
) -> None:
    channel, bus = _channel(tmp_path)
    attempts = 0
    persisted: list[object] = []

    async def flaky_persistence(message: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("database unavailable")
        persisted.append(message)

    bus.subscribe_inbound(flaky_persistence)  # type: ignore[arg-type]
    payload = _event(sender_ids={"open_id": "u_allowed"})

    with pytest.raises(RuntimeError, match="database unavailable"):
        await channel.handle_event(payload)
    result = await channel.handle_event(payload)

    assert result == {"ok": True}
    assert attempts == 2
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_stop_uses_interrupt_controller_and_does_not_publish_turn(tmp_path: Path) -> None:
    interrupt = SimpleNamespace(
        request_interrupt=AsyncMock(
            return_value=InterruptAcknowledgement(
                message="已请求中断",
                provider_uuid="4fddab9d-f30c-5c31-850c-849ebc958f9a",
            )
        )
    )
    channel, bus = _channel(tmp_path, interrupt_controller=interrupt)
    channel.send = AsyncMock(return_value=SimpleNamespace(message_id="om_stop"))  # type: ignore[method-assign]

    result = await channel.handle_event(
        _event(sender_ids={"open_id": "u_allowed"}, content={"text": "/stop"})
    )

    assert result == {"ok": True}
    assert bus.inbound_size == 0
    interrupt.request_interrupt.assert_awaited_once()
    channel.send.assert_awaited_once_with(
        "chat-1",
        "已请求中断",
        provider_uuid="4fddab9d-f30c-5c31-850c-849ebc958f9a",
    )


@pytest.mark.asyncio
async def test_image_is_downloaded_to_workspace_uploads(tmp_path: Path) -> None:
    channel, bus = _channel(tmp_path)
    channel._download_message_resource = AsyncMock(  # type: ignore[method-assign]
        return_value=(b"\x89PNG\r\n\x1a\nimage", "image/png")
    )

    await channel.handle_event(
        _event(
            sender_ids={"open_id": "u_allowed"},
            message_type="image",
            content={"image_key": "img-key"},
        )
    )
    message = await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    assert message.content == "[图片]"
    assert len(message.media) == 1
    image = Path(message.media[0])
    assert image.parent == tmp_path / "uploads"
    assert image.suffix == ".png"


@pytest.mark.asyncio
async def test_ws_callback_from_thread_is_scheduled_on_main_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel, bus = _channel(tmp_path)
    monkeypatch.setattr(
        "memopilot.channels.feishu._marshal_ws_event",
        lambda data: data,
    )
    channel._main_loop = asyncio.get_running_loop()
    payload = _event(sender_ids={"open_id": "u_allowed"})

    thread = threading.Thread(target=channel._on_ws_message, args=(payload,))
    thread.start()
    await asyncio.to_thread(thread.join, 1)
    message = await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    assert thread.is_alive() is False
    assert message.content == "你好"


@pytest.mark.asyncio
async def test_ws_callback_does_not_return_before_inbound_persistence_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel, bus = _channel(tmp_path)
    monkeypatch.setattr("memopilot.channels.feishu._marshal_ws_event", lambda data: data)
    channel._main_loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def persist(message: object) -> None:
        entered.set()
        await release.wait()

    bus.subscribe_inbound(persist)  # type: ignore[arg-type]
    thread = threading.Thread(
        target=channel._on_ws_message,
        args=(_event(sender_ids={"open_id": "u_allowed"}),),
    )
    thread.start()
    await asyncio.wait_for(entered.wait(), timeout=1)

    assert thread.is_alive() is True
    release.set()
    await asyncio.to_thread(thread.join, 1)
    assert thread.is_alive() is False


@pytest.mark.asyncio
async def test_stop_during_ws_initialization_prevents_late_connection(tmp_path: Path) -> None:
    channel, _ = _channel(tmp_path)
    initializing = threading.Event()
    connected = threading.Event()

    def delayed_start() -> None:
        initializing.set()
        threading.Event().wait(0.05)
        stop_requested = getattr(channel, "_ws_stop_requested", threading.Event())
        if not stop_requested.is_set():
            connected.set()

    channel._run_ws_client = delayed_start  # type: ignore[method-assign]
    await channel.start()
    await asyncio.to_thread(initializing.wait, 1)
    await channel.stop()
    await asyncio.sleep(0.06)

    assert connected.is_set() is False
    assert channel._ws_thread is None or channel._ws_thread.is_alive() is False


@pytest.mark.asyncio
async def test_ws_disconnect_obeys_stop_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel, _ = _channel(tmp_path)

    async def never_disconnect() -> None:
        await asyncio.Event().wait()

    channel._ws_client = SimpleNamespace(_disconnect=never_disconnect)
    monkeypatch.setattr(channel, "_get_ws_loop", asyncio.get_running_loop)

    with pytest.raises(TimeoutError):
        await channel._disconnect_ws_client_once(timeout_seconds=0.01)


@pytest.mark.asyncio
async def test_text_send_uses_caller_supplied_uuid_and_returns_message_id(tmp_path: Path) -> None:
    channel, _ = _channel(tmp_path)
    channel._get_tenant_access_token = AsyncMock(return_value="token")  # type: ignore[method-assign]
    channel._api_request = AsyncMock(return_value={"message_id": "om_1"})  # type: ignore[method-assign]

    receipt = await channel.send(
        "chat-1",
        "回复",
        provider_uuid="4fddab9d-f30c-5c31-850c-849ebc958f9a",
    )

    assert receipt.message_id == "om_1"
    body = channel._api_request.await_args.args[2]
    assert body["uuid"] == "4fddab9d-f30c-5c31-850c-849ebc958f9a"
    assert json.loads(body["content"]) == {"text": "回复"}


@pytest.mark.asyncio
async def test_image_and_file_send_keep_local_channel_behavior(tmp_path: Path) -> None:
    channel, _ = _channel(tmp_path)
    image = tmp_path / "picture.png"
    image.write_bytes(b"image")
    document = tmp_path / "report.pdf"
    document.write_bytes(b"document")
    channel._get_tenant_access_token = AsyncMock(return_value="token")  # type: ignore[method-assign]
    channel._upload_image = AsyncMock(return_value="img-key")  # type: ignore[method-assign]
    channel._upload_file = AsyncMock(return_value="file-key")  # type: ignore[method-assign]
    channel._api_request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[{"message_id": "om-image"}, {"message_id": "om-file"}]
    )

    image_receipt = await channel.send_image(
        "chat-1",
        str(image),
        provider_uuid="image-uuid",
    )
    file_receipt = await channel.send_file(
        "chat-1",
        str(document),
        name="简历.pdf",
        provider_uuid="file-uuid",
    )

    assert (image_receipt.message_id, file_receipt.message_id) == ("om-image", "om-file")
    image_body = channel._api_request.await_args_list[0].args[2]
    file_body = channel._api_request.await_args_list[1].args[2]
    assert image_body["uuid"] == "image-uuid"
    assert json.loads(image_body["content"]) == {"image_key": "img-key"}
    assert file_body["uuid"] == "file-uuid"
    assert json.loads(file_body["content"]) == {
        "file_key": "file-key",
        "file_name": "简历.pdf",
    }
