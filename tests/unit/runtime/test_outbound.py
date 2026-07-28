from __future__ import annotations

from memopilot.runtime.common_tools.message_push import MessagePushTool
from memopilot.runtime.outbound import OutboundDispatch, PushToolOutboundPort


async def test_push_tool_outbound_port_matches_prototype_contract() -> None:
    sent: list[tuple[str, str]] = []
    push = MessagePushTool()

    async def send(chat_id: str, message: str) -> None:
        sent.append((chat_id, message))

    push.register_channel("feishu", text=send)
    outbound = PushToolOutboundPort(push)

    result = await outbound.dispatch(
        OutboundDispatch(
            channel="feishu",
            chat_id="chat-1",
            content="提醒内容",
        )
    )

    assert result is True
    assert sent == [("chat-1", "提醒内容")]


async def test_push_tool_outbound_port_reports_send_failure() -> None:
    push = MessagePushTool()

    async def fail(chat_id: str, message: str) -> None:
        del chat_id, message
        raise RuntimeError("network down")

    push.register_channel("feishu", text=fail)

    result = await PushToolOutboundPort(push).dispatch(
        OutboundDispatch(channel="feishu", chat_id="chat-1", content="提醒内容")
    )

    assert result is False


async def test_push_tool_outbound_port_forwards_stable_provider_uuid() -> None:
    captured: list[str | None] = []
    push = MessagePushTool()

    async def send(
        chat_id: str,
        message: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        del chat_id, message
        captured.append(provider_uuid)

    push.register_channel("feishu", text=send)

    assert await PushToolOutboundPort(push).dispatch(
        OutboundDispatch(
            channel="feishu",
            chat_id="chat-1",
            content="回复",
            metadata={"provider_uuid": "reply-uuid"},
        )
    )
    assert captured == ["reply-uuid"]


async def test_media_provider_uuids_are_stable_on_replay_and_distinct_per_item() -> None:
    captured: list[tuple[str, str | None]] = []
    push = MessagePushTool()

    async def send_image(
        chat_id: str,
        image: str,
        *,
        provider_uuid: str | None,
    ) -> None:
        del chat_id, image
        captured.append(("image", provider_uuid))

    async def send_file(
        chat_id: str,
        file: str,
        name: str | None,
        *,
        provider_uuid: str | None,
    ) -> None:
        del chat_id, file, name
        captured.append(("file", provider_uuid))

    async def send_text(
        chat_id: str,
        message: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        del chat_id, message, provider_uuid

    push.register_channel(
        "feishu",
        text=send_text,
        image=send_image,
        file=send_file,
    )
    outbound = PushToolOutboundPort(push)
    dispatch = OutboundDispatch(
        channel="feishu",
        chat_id="chat-1",
        content="回复",
        metadata={"provider_uuid": "reply-uuid"},
        media=["meme.png", "report.pdf"],
    )

    assert await outbound.dispatch(dispatch)
    assert await outbound.dispatch(dispatch)

    first, replay = captured[:2], captured[2:]
    assert first == replay
    assert first[0][1] is not None
    assert first[1][1] is not None
    assert first[0][1] != first[1][1]

    captured.clear()
    fallback = OutboundDispatch(
        channel="feishu",
        chat_id="chat-1",
        content="回复",
        media=["meme.png", "report.pdf"],
    )
    assert await outbound.dispatch(fallback)
    assert await outbound.dispatch(fallback)
    assert captured[:2] == captured[2:]
