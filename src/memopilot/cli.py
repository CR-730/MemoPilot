"""MemoPilot 单进程异步入口。"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, cast

from memopilot.bootstrap import build_app, build_scheduler, build_worker
from memopilot.channels.cli_tui import run_tui_async
from memopilot.config import load_settings
from memopilot.redis_runtime import RedisRuntime

logger = logging.getLogger(__name__)


def main() -> None:
    _configure_logging()
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.info("收到停止信号，进程已退出")


def _configure_logging() -> None:
    """保留运行状态日志，隐藏 HTTP SDK 的握手细节，避免污染终端 CLI。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def _run(args: argparse.Namespace) -> None:
    settings = load_settings(args.config, workspace=args.workspace)
    if args.command == "run":
        shutdown_event = asyncio.Event()
        shutdown_bridge = _WindowsConsoleShutdownBridge(
            asyncio.get_running_loop(),
            shutdown_event,
        )
        shutdown_bridge.install()
        try:
            await run_all(
                settings,
                interactive=False,
                shutdown_event=shutdown_event,
            )
        finally:
            shutdown_bridge.mark_shutdown_complete()
            shutdown_bridge.uninstall()
        return


async def run_all(
    settings: Any,
    *,
    interactive: bool = False,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """在同一个 asyncio 事件循环中运行全部常驻服务。"""
    redis_runtime: RedisRuntime | None = None
    app_bundle: Any = None
    scheduler_bundle: Any = None
    worker_bundle: Any = None
    service_tasks: list[asyncio.Task[Any]] = []
    cli_task: asyncio.Task[Any] | None = None
    shutdown_task: asyncio.Task[Any] | None = None
    try:
        redis_runtime = await RedisRuntime.ensure(settings.redis_url)
        if redis_runtime.managed:
            logger.info("本地 Redis 未运行，已由 MemoPilot 自动启动")
        else:
            logger.info("检测到可用 Redis，直接复用现有服务")
        app_bundle = build_app(settings)
        scheduler_bundle = build_scheduler(settings)
        if hasattr(app_bundle, "console"):
            console = app_bundle.console
            worker_bundle = await build_worker(settings, console_transport=console)
        else:
            console = None
            worker_bundle = await build_worker(settings)
        await worker_bundle.start_extensions()
        for diagnostic in worker_bundle.mcp_diagnostics:
            logger.warning("MCP Server 不可用，核心 Runtime 继续启动: %s", diagnostic)
        if not settings.feishu_allow_from:
            logger.warning("飞书 allowlist 为空：当前允许所有私聊用户访问")
        if console is not None:
            await console.start()
        await app_bundle.service.start()
        logger.info("MemoPilot 已启动：Gateway、Scheduler、Runner 运行于同一 asyncio 事件循环")
        service_tasks = [
            asyncio.create_task(
                scheduler_bundle.service.run_forever(),
                name="memopilot-scheduler",
            ),
            asyncio.create_task(
                worker_bundle.service.run_forever(),
                name="memopilot-worker",
            ),
        ]
        wait_tasks: set[asyncio.Task[Any]] = set(service_tasks)
        if interactive:
            cli_task = asyncio.create_task(
                run_tui_async(),
                name="memopilot-tui",
            )
            wait_tasks.add(cli_task)
        if shutdown_event is not None:
            shutdown_task = asyncio.create_task(
                shutdown_event.wait(),
                name="memopilot-shutdown",
            )
            wait_tasks.add(shutdown_task)
        if interactive or shutdown_event is not None:
            done, _ = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed_services = [task for task in service_tasks if task in done]
            for task in completed_services:
                await task
                raise RuntimeError(f"常驻服务意外退出: {task.get_name()}")
            if cli_task is not None and cli_task in done:
                await cli_task
        else:
            await asyncio.gather(*service_tasks)
    finally:
        tasks_to_close = [*service_tasks]
        if cli_task is not None:
            tasks_to_close.append(cli_task)
        if shutdown_task is not None:
            tasks_to_close.append(shutdown_task)
        for task in tasks_to_close:
            task.cancel()
        if tasks_to_close:
            await asyncio.gather(*tasks_to_close, return_exceptions=True)
        if worker_bundle is not None:
            await worker_bundle.close()
        if scheduler_bundle is not None:
            await scheduler_bundle.close()
        if app_bundle is not None:
            await app_bundle.close()
        if redis_runtime is not None:
            await redis_runtime.close()


class _WindowsConsoleShutdownBridge:
    """把 Windows 终端关闭事件转成 asyncio 停机信号，并等待清理完成。"""

    _CTRL_CLOSE_EVENT = 2
    _CTRL_LOGOFF_EVENT = 5
    _CTRL_SHUTDOWN_EVENT = 6

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        shutdown_event: asyncio.Event,
    ) -> None:
        self._loop = loop
        self._shutdown_event = shutdown_event
        self._shutdown_complete = threading.Event()
        self._callback: Any | None = None
        self._kernel32: Any | None = None

    def install(self) -> None:
        if os.name != "nt":
            return
        callback_factory = cast(Any, ctypes.WINFUNCTYPE)
        callback_type = callback_factory(ctypes.c_bool, ctypes.c_uint)

        def handle_console_event(event_type: int) -> bool:
            if event_type not in {
                self._CTRL_CLOSE_EVENT,
                self._CTRL_LOGOFF_EVENT,
                self._CTRL_SHUTDOWN_EVENT,
            }:
                return False
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
            self._shutdown_complete.wait(4.5)
            return True

        callback = callback_type(handle_console_event)
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCtrlHandler.argtypes = [callback_type, ctypes.c_bool]
        kernel32.SetConsoleCtrlHandler.restype = ctypes.c_bool
        if not kernel32.SetConsoleCtrlHandler(callback, True):
            logger.warning("未能注册 Windows 终端关闭处理器")
            return
        self._callback = callback
        self._kernel32 = kernel32

    def mark_shutdown_complete(self) -> None:
        self._shutdown_complete.set()

    def uninstall(self) -> None:
        if self._kernel32 is None or self._callback is None:
            return
        self._kernel32.SetConsoleCtrlHandler(self._callback, False)
        self._callback = None
        self._kernel32 = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="在一个 asyncio 事件循环中启动全部服务")
    _add_config_arguments(run)
    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))


if __name__ == "__main__":
    main()


__all__ = ["main", "run_all"]
