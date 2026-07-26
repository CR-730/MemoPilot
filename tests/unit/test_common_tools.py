from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from memopilot.config import MemoPilotSettings
from memopilot.persistence.migrations import migrate_all_databases
from memopilot.runtime.common_tools import register_common_tools
from memopilot.runtime.common_tools.http import HttpRequester
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


class _UnusedHttpRequester:
    async def get(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("本测试不应访问网络")


def _register(
    registry: ToolRegistry,
    settings: MemoPilotSettings,
    repository: OperationalRepository,
) -> None:
    register_common_tools(
        registry,
        workspace=settings.workspace,
        repository=repository,
        http_requester=cast(HttpRequester, _UnusedHttpRequester()),
    )


def _repository(tmp_path: Path) -> tuple[MemoPilotSettings, OperationalRepository]:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    return settings, OperationalRepository(settings.operational_database)


async def test_common_toolset_contains_old_public_tools(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()

    _register(registry, settings, repository)

    assert COMMON_TOOL_NAMES <= set(registry.tool_names)


@pytest.mark.asyncio
async def test_file_tools_preserve_read_write_edit_list_contract(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()
    _register(registry, settings, repository)

    notes_path = tmp_path / "notes.txt"
    write_result = await registry.get_tool("write_file").handler(
        path=str(notes_path),
        content="before\n",
    )
    assert "已写入" in str(write_result)

    edit_result = await registry.get_tool("edit_file").handler(
        path=str(notes_path),
        old_text="before",
        new_text="after",
    )
    assert "已成功编辑" in str(edit_result)

    read_result = await registry.get_tool("read_file").handler(path=str(notes_path))
    assert "after" in str(read_result)

    list_result = await registry.get_tool("list_dir").handler(path=str(tmp_path))
    assert "notes.txt" in str(list_result)


@pytest.mark.asyncio
async def test_registered_shell_keeps_prototype_execution_contract(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()
    _register(registry, settings, repository)

    result = await registry.get_tool("shell").handler(
        command='python -c "print(37*24)"',
        description="计算表达式",
    )

    payload = json.loads(str(result))
    assert payload["exit_code"] == 0
    assert payload["output"].strip() == "888"


def test_shell_schema_does_not_claim_the_host_always_uses_bash(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()
    _register(registry, settings, repository)

    shell = registry.get_tool("shell")

    assert shell is not None
    assert "bash 中执行" not in shell.description
    assert "bash 命令" not in json.dumps(shell.parameters, ensure_ascii=False)


def test_common_tool_registration_keeps_prototype_search_hints(tmp_path: Path) -> None:
    settings, repository = _repository(tmp_path)
    registry = ToolRegistry()
    _register(registry, settings, repository)

    expected = {
        "shell": "终端 脚本 bash 命令",
        "task_output": "后台任务输出 task_output 进程日志",
        "task_stop": "停止后台任务 task_stop 杀进程",
        "web_search": "谷歌 Bing 查资料",
        "web_fetch": "读取网址 浏览网页",
        "list_dir": "ls 查看目录",
        "fetch_messages": "消息回溯 按ID查对话原文 source_ref",
        "search_messages": "你之前说 聊过什么 历史对话",
    }
    assert {
        name: registry.get_document(name).search_hint  # type: ignore[union-attr]
        for name in expected
    } == expected
