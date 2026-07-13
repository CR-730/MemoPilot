"""持久化状态机的代码枚举。"""

from enum import StrEnum


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"


class OutboxState(StrEnum):
    PENDING = "pending"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    DEAD = "dead"


class EffectState(StrEnum):
    PENDING = "pending"
    SENDING = "sending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    NEEDS_REVIEW = "needs_review"


__all__ = ["EffectState", "JobState", "OutboxState", "RunState"]
