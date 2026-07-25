"""MemoPilot 单进程异步入口与 Effect 人工核对命令。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from memopilot.bootstrap import build_app, build_effects, build_scheduler, build_worker
from memopilot.config import load_settings
from memopilot.delivery.effects import EffectRecord
from memopilot.delivery.reconciliation import EffectReconciliationService

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
    effect_bundle = build_effects(settings)
    try:
        result = await execute_effect_action(
            effect_bundle.service,
            action=args.effect_action,
            operation_id=getattr(args, "operation_id", None),
            message_id=getattr(args, "message_id", None),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await effect_bundle.close()


async def run_all(settings: Any) -> None:
    """在同一个 asyncio 事件循环中运行全部常驻服务。"""
    app_bundle: Any = None
    scheduler_bundle: Any = None
    worker_bundle: Any = None
    service_tasks: list[asyncio.Task[None]] = []
    try:
        app_bundle = build_app(settings)
        scheduler_bundle = build_scheduler(settings)
        worker_bundle = await build_worker(settings)
        await worker_bundle.start_extensions()
        for diagnostic in worker_bundle.mcp_diagnostics:
            logger.warning("MCP Server 不可用，核心 Runtime 继续启动: %s", diagnostic)
        if not settings.feishu_allow_from:
            logger.warning("飞书 allowlist 为空：当前允许所有私聊用户访问")
        await app_bundle.service.start()
        logger.info("MemoPilot 已启动：App、Scheduler、Worker 运行于同一 asyncio 事件循环")
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


async def execute_effect_action(
    service: EffectReconciliationService,
    *,
    action: str,
    operation_id: str | None = None,
    message_id: str | None = None,
) -> Any:
    if action == "list":
        return [_effect_dict(effect) for effect in service.list_reviewable()]
    if operation_id is None:
        raise ValueError(f"{action} 必须提供 operation_id")
    if action == "show":
        return _effect_dict(service.show(operation_id))
    if action == "confirm":
        if not message_id:
            raise ValueError("confirm 必须提供 --message-id")
        return _effect_dict(await service.confirm(operation_id, message_id=message_id))
    if action == "fail":
        return _effect_dict(await service.fail(operation_id))
    if action == "retry":
        result = await service.retry(operation_id)
        return {"outcome": str(result.outcome), "effect": _effect_dict(result.effect)}
    raise ValueError(f"未知 Effect 操作: {action}")


def _effect_dict(effect: EffectRecord | Any) -> dict[str, Any]:
    return {
        "operation_id": effect.operation_id,
        "run_id": getattr(effect, "run_id", None),
        "session_key": getattr(effect, "session_key", None),
        "state": effect.state,
        "message_id": getattr(effect, "message_id", None),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="在一个 asyncio 事件循环中启动全部服务")
    _add_config_arguments(run)
    effects = subparsers.add_parser("effects")
    _add_config_arguments(effects)
    effect_commands = effects.add_subparsers(dest="effect_action", required=True)
    effect_commands.add_parser("list")
    show = effect_commands.add_parser("show")
    show.add_argument("operation_id")
    confirm = effect_commands.add_parser("confirm")
    confirm.add_argument("operation_id")
    confirm.add_argument("--message-id", required=True)
    fail = effect_commands.add_parser("fail")
    fail.add_argument("operation_id")
    retry = effect_commands.add_parser("retry")
    retry.add_argument("operation_id")
    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))


if __name__ == "__main__":
    main()


__all__ = ["execute_effect_action", "main", "run_all"]
