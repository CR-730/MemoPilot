from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

mcp = FastMCP("memopilot-test")
active_calls = 0
max_active_calls = 0
call_counts: dict[str, int] = {}


@mcp.tool()
async def echo(text: str) -> dict[str, str]:
    return {"text": text}


@mcp.tool()
async def serialized(delay: float = 0.05) -> dict[str, int]:
    global active_calls, max_active_calls
    active_calls += 1
    max_active_calls = max(max_active_calls, active_calls)
    try:
        await asyncio.sleep(delay)
        return {"max_active": max_active_calls}
    finally:
        active_calls -= 1


@mcp.tool()
def fail() -> CallToolResult:
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text="remote rejected")],
    )


@mcp.tool()
async def wait(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "done"


@mcp.tool()
async def counted(label: str) -> int:
    call_counts[label] = call_counts.get(label, 0) + 1
    return call_counts[label]


@mcp.tool()
def get_count(label: str) -> int:
    return call_counts.get(label, 0)


if __name__ == "__main__":
    mcp.run(transport="stdio")
