from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime

import pytest

from memopilot.memory.service import MemoryService
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease

NOW = datetime(2026, 7, 30, tzinfo=UTC)
LEASE = SessionLease("s", "o", 1, "k", "v")


class _Repo:
    def __init__(self) -> None:
        self.fences = []

    def assert_current_fence(self, lease):  # type: ignore[no-untyped-def]
        self.fences.append(lease)

    def fenced_write(self, lease):  # type: ignore[no-untyped-def]
        assert lease is LEASE
        return nullcontext()


class _Async:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result

    async def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((args, kwargs))
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "payload", "target"),
    [
        ("memory.vectorize", {"consolidation_id": "c1"}, "vector"),
        ("memory.optimize", {}, "optimizer"),
        ("memory.post_response", {"turn_id": "t", "protected_ids": ["p"]}, "post"),
    ],
)
async def test_memory_service_routes_task_with_fence(kind, payload, target) -> None:  # type: ignore[no-untyped-def]
    repo, consolidation, vector, optimizer, post = _Repo(), _Async(), _Async(), _Async(), _Async()
    service = MemoryService(consolidation, vector, optimizer, repo, post_response=post)  # type: ignore[arg-type]
    await service.execute_task(AgentTask("id", kind, 3, "s", payload, NOW), lease=LEASE, now=NOW)
    assert {"vector": vector, "optimizer": optimizer, "post": post}[target].calls
    assert repo.fences[0] is LEASE


@pytest.mark.asyncio
async def test_consolidate_then_vectorize_and_reinforce_keep_contract() -> None:
    repo, vector = _Repo(), _Async()
    consolidation = _Async(type("Result", (), {"consolidation_id": "c1"})())
    class Store:
        calls: list[tuple[tuple[str, ...], str]] = []

        def reinforce_items_once(self, ids, usage_ref):  # type: ignore[no-untyped-def]
            self.calls.append((ids, usage_ref))

    store = Store()
    vector.store = store
    service = MemoryService(consolidation, vector, _Async(), repo)  # type: ignore[arg-type]
    await service.execute_task(
        AgentTask("a", "memory.consolidate", 3, "s", {}, NOW), lease=LEASE, now=NOW
    )
    await service.execute_task(
        AgentTask("b", "memory.reinforce", 3, "s", {"usage_ref": "u", "item_ids": ["x"]}, NOW),
        lease=LEASE,
        now=NOW,
    )
    assert vector.calls[0][0] == ("c1",)
    assert store.calls == [(('x',), "u")]


@pytest.mark.asyncio
async def test_unknown_memory_kind_is_rejected() -> None:
    service = MemoryService(_Async(), _Async(), _Async(), _Repo())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="不支持"):
        await service.execute_task(
            AgentTask("x", "memory.nope", 3, "s", {}, NOW), lease=LEASE, now=NOW
        )
