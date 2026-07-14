"""消息渠道与飞书适配。"""

from memopilot.channels.contracts import (
    InboundMessage,
    InterruptAcknowledgement,
    MessageBus,
    SendReceipt,
)

__all__ = ["InboundMessage", "InterruptAcknowledgement", "MessageBus", "SendReceipt"]
