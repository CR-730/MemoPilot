from __future__ import annotations

from pathlib import Path

from memopilot.extensions.hooks import ToolExecutionRequest
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolExecutor, ToolObservation, ToolRegistry

PLUGIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "memopilot"
    / "builtin_plugins"
    / "tool_loop_guard"
)


async def test_third_identical_tool_batch_is_denied_but_changed_batch_runs() -> None:
    handler_calls: list[str] = []

    def make_handler(tool_name: str):
        async def handler(value: int) -> str:
            handler_calls.append(f"{tool_name}:{value}")
            return "ok"

        return handler

    tool_names = ("alpha", "beta", "task_output", "task_stop")
    registry = ToolRegistry(
        Tool(
            tool_name,
            tool_name,
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            make_handler(tool_name),
        )
        for tool_name in tool_names
    )
    manager = PluginManager([PLUGIN_DIR], tool_registry=registry)
    await manager.load_all()
    executor = ToolExecutor(registry, manager.tool_hooks)

    call_sequence = 0

    async def execute_batch(
        session_key: str,
        calls: tuple[tuple[str, int], ...],
    ) -> list[ToolObservation]:
        nonlocal call_sequence
        batch = tuple(
            {
                "call_id": f"c{call_sequence + index}",
                "tool_name": tool_name,
                "arguments": {"value": value},
            }
            for index, (tool_name, value) in enumerate(calls)
        )
        call_sequence += len(batch)
        observations: list[ToolObservation] = []
        for index, call in enumerate(batch):
            tool_name = str(call["tool_name"])
            arguments = dict(call["arguments"])
            observations.append(
                await executor.execute(
                    FunctionCall(str(call["call_id"]), tool_name, arguments),
                    request=ToolExecutionRequest(
                        call_id=str(call["call_id"]),
                        tool_name=tool_name,
                        arguments=arguments,
                        source="passive",
                        session_key=session_key,
                        channel="feishu",
                        chat_id="chat-1",
                        tool_batch=batch,
                        tool_batch_index=index,
                    ),
                )
            )
        return observations

    single = (("alpha", 1),)
    assert (await execute_batch("single", single))[0].ok
    assert (await execute_batch("single", single))[0].ok
    assert (await execute_batch("single", single))[0].status == "denied"
    assert (await execute_batch("single", (("alpha", 2),)))[0].ok

    repeated = (("alpha", 1), ("beta", 2))
    assert all(item.ok for item in await execute_batch("batch", repeated))
    assert all(item.ok for item in await execute_batch("batch", repeated))
    before_third_batch = len(handler_calls)
    third_batch = await execute_batch("batch", repeated)
    assert handler_calls[before_third_batch:] == []
    assert [item.status for item in third_batch] == ["denied", "denied"]
    assert all(
        item.ok
        for item in await execute_batch(
            "batch",
            (("beta", 2), ("alpha", 1)),
        )
    )

    with_excluded = (
        ("task_output", 0),
        ("alpha", 1),
        ("task_stop", 0),
    )
    await execute_batch("excluded", with_excluded)
    await execute_batch("excluded", with_excluded)
    excluded_third = await execute_batch("excluded", with_excluded)
    assert [item.status for item in excluded_third] == [
        "success",
        "denied",
        "success",
    ]
