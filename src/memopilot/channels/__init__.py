"""消息渠道与飞书适配。"""

from memopilot.channels.contracts import (
    InboundHandler,
    InboundMessage,
    InterruptAcknowledgement,
    SendReceipt,
)

__all__ = ["InboundHandler", "InboundMessage", "InterruptAcknowledgement", "SendReceipt"]
