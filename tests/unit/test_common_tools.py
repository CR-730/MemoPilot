from __future__ import annotations

import os
from pathlib import Path

import pytest

from memopilot.config import MemoPilotSettings
from memopilot.persistence.migrations import migrate_all_databases
from memopilot.runtime.common_tools import register_common_tools
from memopilot.runtime.common_tools.shell import ShellTool
from memopilot.runtime.tools import ToolRegistry
from memopilot.tasks.operational import OperationalRepository

COMMON_TOOL_NAMES = {
    "shell",
    "task_output",
    "task_stop",
    "web_search",
    "web_fetch",
    "read_file",
    "list_dir",
    "fetch_messages",
    "search_messages",
    "message_push",
    "write_file",
    "edit_file",
}


def _repository(tmp_path: Path) -> tuple[MemoPilotSettings, OperationalRepository]:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    return settings, OperationalRepository(settings.operational_database)


async def test_common_toolset_contains_old_public_tools(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()

    register_common_tools(
        registry,
        workspace=settings.workspace,
        repository=repository,
    )

    assert COMMON_TOOL_NAMES <= set(registry.tool_names)


@pytest.mark.asyncio
async def test_file_tools_preserve_read_write_edit_list_contract(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()
    register_common_tools(
        registry,
        workspace=settings.workspace,
        repository=repository,
    )

    write_result = await registry.get_tool("write_file").handler(
        path="notes.txt",
        content="before\n",
    )
    assert "已写入" in str(write_result)

    edit_result = await registry.get_tool("edit_file").handler(
        path="notes.txt",
        old_text="before",
        new_text="after",
    )
    assert "已成功编辑" in str(edit_result)

    read_result = await registry.get_tool("read_file").handler(path="notes.txt")
    assert "after" in str(read_result)

    list_result = await registry.get_tool("list_dir").handler(path=".")
    assert "notes.txt" in str(list_result)


@pytest.mark.asyncio
async def test_shell_rejects_bash_only_commands_on_windows() -> None:
    if os.name != "nt":
        pytest.skip("仅验证 Windows 下的 cmd/bash 语法边界")

    tool = ShellTool()
    arithmetic = await tool.execute(
        command="echo $((37 * 24))",
        description="计算表达式",
    )
    python3 = await tool.execute(
        command='python3 -c "print(37*24)"',
        description="执行 Python 计算",
    )
    python = await tool.execute(
        command='python -c "print(37*24)"',
        description="执行 Python 计算",
    )

    assert "Windows" in arithmetic
    assert "bash" in arithmetic.lower()
    assert "python3" in python3
    assert "python" in python3
    assert '"output": "888' in python
