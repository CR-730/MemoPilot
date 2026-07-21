"""Agent 动态管理本地 stdio MCP Server 的标准工具。"""

from __future__ import annotations

from memopilot.extensions.mcp_registry import McpServerRegistry
from memopilot.runtime.tools import Tool, ToolRegistry


def build_mcp_management_tools(registry: McpServerRegistry) -> tuple[Tool, ...]:
    async def add(
        name: str,
        command: list[str],
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        return await registry.add(name, command, args=args, env=env, cwd=cwd)

    async def remove(name: str) -> str:
        return await registry.remove(name)

    async def list_servers() -> tuple[dict[str, object], ...]:
        return registry.list_servers()

    return (
        Tool(
            name="mcp_add",
            description=(
                "连接并注册本地 stdio MCP Server。所有 env 值必须写成 ${ENV_NAME}，"
                "连接成功后远程工具立即可用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "唯一短名称"},
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "args": {"type": "array", "items": {"type": "string"}},
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "cwd": {"type": "string"},
                },
                "required": ["name", "command"],
                "additionalProperties": False,
            },
            handler=add,
            source="builtin:mcp",
        ),
        Tool(
            name="mcp_remove",
            description="注销 MCP Server、关闭连接并移除它注册的全部工具。",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=remove,
            source="builtin:mcp",
        ),
        Tool(
            name="mcp_list",
            description="列出 MCP Server、连接状态、工具和脱敏配置。",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=list_servers,
            source="builtin:mcp",
        ),
    )


def register_mcp_management_tools(
    tools: ToolRegistry,
    registry: McpServerRegistry,
) -> tuple[Tool, ...]:
    """按原型风险等级注册 MCP 管理工具。"""
    add, remove, list_servers = build_mcp_management_tools(registry)
    tools.register(add, risk="external-side-effect")
    tools.register(remove, risk="write")
    tools.register(list_servers, risk="read-only")
    return add, remove, list_servers


__all__ = ["build_mcp_management_tools", "register_mcp_management_tools"]
