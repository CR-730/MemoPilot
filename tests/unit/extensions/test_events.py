from __future__ import annotations

import asyncio

import pytest

from memopilot.extensions.events import EventBus, EventHandler


async def test_ordered_event_handlers_transform_payload_and_isolate_observers() -> None:
    observed: list[dict[str, object]] = []

    async def transform(payload: dict[str, object]) -> dict[str, object]:
        return {**payload, "value": 2}

    async def broken_observer(payload: dict[str, object]) -> None:
        del payload
        raise RuntimeError("observer failed")

    async def observer(payload: dict[str, object]) -> None:
        observed.append(payload)

    bus = EventBus(
        (
            EventHandler("transform", "before_turn", transform),
            EventHandler("broken", "before_turn", broken_observer, observer=True),
            EventHandler("observe", "before_turn", observer, observer=True),
        )
    )

    result = await bus.emit("before_turn", {"value": 1})
    await bus.observe("before_turn", result)

    assert result == {"value": 2}
    assert observed == [{"value": 2}]
    assert bus.diagnostics[0].handler_id == "broken"


async def test_sync_and_async_gates_are_ordered_and_subscription_is_revocable() -> None:
    bus = EventBus()

    def first(event: dict[str, object]) -> dict[str, object]:
        return {**event, "order": ["first"]}

    async def second(event: dict[str, object]) -> dict[str, object]:
        return {**event, "order": [*event["order"], "second"]}  # type: ignore[misc]

    first_subscription = bus.on("turn", first, priority=10)
    bus.on("turn", second, priority=20)

    assert await bus.emit("turn", {}) == {"order": ["first", "second"]}

    first_subscription.unsubscribe()
    assert await bus.emit("turn", {"order": []}) == {"order": ["second"]}


async def test_fanout_is_concurrent_and_background_queue_can_be_drained() -> None:
    bus = EventBus()
    both_started = asyncio.Event()
    started = 0
    observed: list[str] = []

    async def observer(event: dict[str, object]) -> None:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        observed.append(str(event["value"]))

    bus.on("turn", observer, observer=True, handler_id="one")
    bus.on("turn", observer, observer=True, handler_id="two")

    bus.enqueue("turn", {"value": "done"})
    await bus.drain()
    await bus.aclose()

    assert observed == ["done", "done"]


async def test_aclose_rejects_enqueue_after_close_starts_and_drains_accepted_events() -> None:
    class PausingDrainEventBus(EventBus):
        def __init__(self) -> None:
            super().__init__()
            self.drain_finished = asyncio.Event()
            self.continue_close = asyncio.Event()

        async def drain(self) -> None:
            await super().drain()
            self.drain_finished.set()
            await self.continue_close.wait()

    observed: list[str] = []
    bus = PausingDrainEventBus()
    bus.on(
        "turn",
        lambda event: observed.append(str(event["value"])),
        observer=True,
    )

    bus.enqueue("turn", {"value": "accepted-before-close"})
    close_task = asyncio.create_task(bus.aclose())
    await asyncio.wait_for(bus.drain_finished.wait(), timeout=0.2)

    with pytest.raises(RuntimeError, match="EventBus"):
        bus.enqueue("turn", {"value": "must-be-rejected"})

    bus.continue_close.set()
    await asyncio.wait_for(close_task, timeout=0.2)
    await bus.aclose()

    assert observed == ["accepted-before-close"]
