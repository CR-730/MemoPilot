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
