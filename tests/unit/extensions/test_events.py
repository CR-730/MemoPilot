from __future__ import annotations

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

    assert result == {"value": 2}
    assert observed == [{"value": 2}]
    assert bus.diagnostics[0].handler_id == "broken"

