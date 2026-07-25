"""MemoPilot 单进程异步入口。"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from memopilot.bootstrap import build_app, build_scheduler, build_worker
from memopilot.config import load_settings
from memopilot.redis_runtime import RedisRuntime

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.info("收到停止信号，进程已退出")


async def _run(args: argparse.Namespace) -> None:
    settings = load_settings(args.config, workspace=args.workspace)
    if args.command == "run":
        await run_all(settings)
        return


async def run_all(settings: Any) -> None:
    """在同一个 asyncio 事件循环中运行全部常驻服务。"""
    redis_runtime: RedisRuntime | None = None
    app_bundle: Any = None
    scheduler_bundle: Any = None
    worker_bundle: Any = None
    service_tasks: list[asyncio.Task[None]] = []
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
        await asyncio.gather(*service_tasks)
    finally:
        for task in service_tasks:
            task.cancel()
        if service_tasks:
            await asyncio.gather(*service_tasks, return_exceptions=True)
        if worker_bundle is not None:
            await worker_bundle.close()
        if scheduler_bundle is not None:
            await scheduler_bundle.close()
        if app_bundle is not None:
            await app_bundle.close()
        if redis_runtime is not None:
            await redis_runtime.close()


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
