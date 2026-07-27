"""Agent 与 Channel 之间的进程内消息总线。"""

from memopilot.bus.events import InboundMessage, OutboundMessage
from memopilot.bus.queue import MessageBus

__all__ = ["InboundMessage", "MessageBus", "OutboundMessage"]
