from __future__ import annotations

import asyncio
import json

from memopilot.runtime.tool_search import (
    ToolDiscoveryState,
    ToolSearchTool,
)
from memopilot.runtime.tools import Tool, ToolRegistry


async def _noop(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


def _tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        handler=_noop,
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        _tool("recall_memory", "检索长期记忆"),
        always_on=True,
        search_hint="回忆 用户偏好",
    )
    registry.register(
        _tool("weather_forecast", "查询未来天气"),
        search_hint="气温 下雨 预报",
    )
    registry.register(
        _tool("mcp_calendar_create", "创建日历事件"),
        risk="external-side-effect",
        source_type="mcp",
        source_name="calendar",
    )
    return registry


def test_registry_indexes_metadata_and_filters_schemas_in_registration_order() -> None:
    registry = _registry()

    assert registry.get_always_on_names() == {"recall_memory"}
    assert registry.get_registered_order(
        {"mcp_calendar_create", "recall_memory"}
    ) == ["recall_memory", "mcp_calendar_create"]
    assert [
        item["function"]["name"]
        for item in registry.schemas({"mcp_calendar_create", "recall_memory"})
    ] == ["recall_memory", "mcp_calendar_create"]
    assert registry.search("下雨")[0]["name"] == "weather_forecast"
    assert registry.search("创建日历", allowed_risk=["read-only"]) == []


def test_registry_dynamic_unregister_removes_search_document() -> None:
    registry = _registry()
    assert registry.search("天气")

    registry.unregister("weather_forecast")

    assert registry.search("天气") == []
    assert "weather_forecast" not in registry.tool_names


async def test_tool_search_supports_keyword_and_select_unlocks() -> None:
    registry = _registry()
    search = ToolSearchTool(registry)
    search.set_excluded_names({"recall_memory"})

    keyword = json.loads(await search.execute(query="下雨"))
    search.set_excluded_names({"recall_memory"})
    selected = json.loads(
        await search.execute(query="select:mcp_calendar_create,recall_memory")
    )

    assert keyword["unlocked"] == ["weather_forecast"]
    assert selected["unlocked"] == ["mcp_calendar_create"]
    assert selected["already_loaded"] == ["recall_memory"]


async def test_tool_search_returns_structured_empty_result() -> None:
    result = json.loads(await ToolSearchTool(_registry()).execute(query="  "))

    assert result["matched"] == []
    assert result["unlocked"] == []
    assert result["already_loaded"] == []
    assert result["tip"]


def test_discovery_state_is_session_scoped_bounded_lru() -> None:
    state = ToolDiscoveryState(capacity=2)

    state.update("s1", ["a", "b"], always_on=set())
    state.update("s1", ["a", "c", "tool_search"], always_on={"always"})
    state.update("s2", ["z", "always"], always_on={"always"})

    assert state.get_preloaded_ordered("s1") == ["a", "c"]
    assert state.get_preloaded_ordered("s2") == ["z"]
    assert state.unlock_names_from_result('{"unlocked":["x","x","y"]}') == [
        "x",
        "y",
    ]
    assert state.unlock_names_from_result("not-json") == []
    assert state.unlock_names_from_result("[]") == []


async def test_tool_search_visible_context_is_isolated_between_concurrent_turns() -> None:
    search = ToolSearchTool(_registry())
    first_ready = asyncio.Event()
    release_first = asyncio.Event()

    async def first_turn() -> dict[str, object]:
        search.set_excluded_names({"weather_forecast"})
        first_ready.set()
        await release_first.wait()
        return json.loads(await search.execute(query="select:weather_forecast"))

    first = asyncio.create_task(first_turn())
    await first_ready.wait()
    search.set_excluded_names({"recall_memory"})
    second = json.loads(await search.execute(query="select:recall_memory"))
    release_first.set()

    assert (await first)["already_loaded"] == ["weather_forecast"]
    assert second["already_loaded"] == ["recall_memory"]
