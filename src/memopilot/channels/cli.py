"""从原型直接迁移的纯文本 CLI 客户端。"""

from __future__ import annotations

import asyncio
import json
import sys

DEFAULT_SOCKET = "127.0.0.1:8765"
_EXIT_CMDS = {"exit", "quit", "q"}


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int] | None:
    if endpoint.count(":") != 1:
        return None
    host, port = endpoint.rsplit(":", 1)
    if not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


class CLIClient:
    """原型 infra/channels/cli.py 的最小适配版。"""

    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        self._socket_path = socket_path

    async def run(self) -> None:
        try:
            endpoint = _parse_tcp_endpoint(self._socket_path)
            if endpoint is None:
                raise OSError("MemoPilot 当前仅支持 Windows 本地 TCP CLI")
            reader, writer = await asyncio.open_connection(*endpoint)
        except (ConnectionRefusedError, OSError):
            print(f"无法连接到 MemoPilot（{self._socket_path}），请先启动主进程：uv run main.py")
            return

        _print_banner()
        receive_task = asyncio.create_task(self._receive(reader))
        try:
            while True:
                text = await _read_line()
                stripped = text.strip()
                if stripped.lower() in _EXIT_CMDS:
                    break
                if not stripped:
                    continue
                writer.write(
                    (json.dumps({"content": stripped}, ensure_ascii=False) + "\n").encode()
                )
                await writer.drain()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            receive_task.cancel()
            writer.close()
            await writer.wait_closed()
            print("\n再见")

    @staticmethod
    async def _receive(reader: asyncio.StreamReader) -> None:
        while True:
            line = await reader.readline()
            if not line:
                print("\n连接已断开")
                return
            data = json.loads(line)
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                tool_chain = metadata.get("tool_chain")
                if isinstance(tool_chain, list):
                    for group in tool_chain:
                        if not isinstance(group, dict):
                            continue
                        calls = group.get("calls")
                        if not isinstance(calls, list):
                            continue
                        names = [
                            str(item.get("name") or "unknown")
                            for item in calls
                            if isinstance(item, dict)
                        ]
                        if names:
                            print(f"\n工具调用: {'、'.join(names)}")
            print(f"\n{data['content']}\n> ", end="", flush=True)


def _print_banner() -> None:
    print("memopilot Agent CLI  |  输入 exit 退出\n")


async def _read_line() -> str:
    loop = asyncio.get_running_loop()
    sys.stdout.write("> ")
    sys.stdout.flush()
    return await loop.run_in_executor(None, sys.stdin.readline)


__all__ = ["CLIClient", "DEFAULT_SOCKET"]
