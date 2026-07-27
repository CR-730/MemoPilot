"""由 workspace 配置直接驱动的主动信息源 MCP Gateway。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.extensions.mcp import McpCallResult, McpInvocationError

if TYPE_CHECKING:
    from memopilot.proactive.store import ProactiveRepository

ProactiveChannel = Literal["alert", "context", "content"]
logger = logging.getLogger(__name__)


class McpCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult: ...


@dataclass(frozen=True, slots=True)
class ProactiveSourceConfig:
    source_id: str
    server: str
    channel: ProactiveChannel
    get_tool: str
    ack_tool: str = ""
    poll_tool: str = ""
    fetch_timeout_seconds: float = 30.0
    ack_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.source_id or not self.server or not self.get_tool:
            raise ValueError("Proactive Source 的 id、server 和 get_tool 不能为空")
        if self.channel not in {"alert", "context", "content"}:
            raise ValueError("Proactive Source channel 必须是 alert/context/content")
        if self.channel != "context" and not self.ack_tool:
            raise ValueError("Alert/Content Source 必须配置 ack_tool")
        if self.fetch_timeout_seconds <= 0 or self.ack_timeout_seconds <= 0:
            raise ValueError("Proactive Source 超时必须大于 0")


@dataclass(frozen=True, slots=True)
class ProactiveEvent:
    source_id: str
    event_id: str
    kind: str
    occurred_at: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProactiveFetchResult:
    events: tuple[ProactiveEvent, ...]


@dataclass(frozen=True, slots=True)
class CollectionReport:
    succeeded: tuple[str, ...]
    failed: dict[str, str]
    inserted_events: int


@dataclass(frozen=True, slots=True)
class AckReplayReport:
    acknowledged: int
    failed: int


def load_proactive_sources(path: Path) -> tuple[ProactiveSourceConfig, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"读取 proactive_sources.json 失败: {exc}") from exc
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else None
    if not isinstance(raw_sources, list):
        raise ValueError("proactive_sources.json 的 sources 必须是数组")
    sources = tuple(_parse_source(raw) for raw in raw_sources)
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("proactive_sources.json 存在重复 source id")
    return tuple(source for source in sources if source is not None)


class ProactiveSourceGateway:
    """读取 Source 配置，拉取事件并重放待确认 ACK。"""

    def __init__(
        self,
        repository: ProactiveRepository,
        *,
        config_path: Path,
        caller_for_server: Callable[[str], McpCaller],
    ) -> None:
        self._repository = repository
        self._config_path = config_path
        self._caller_for_server = caller_for_server

    async def collect(self, *, session_key: str, fetched_at: datetime) -> CollectionReport:
        configs = load_proactive_sources(self._config_path)

        async def fetch_one(
            config: ProactiveSourceConfig,
        ) -> tuple[str, int | None, str | None]:
            try:
                result = await self._fetch(config, fetched_at=fetched_at)
                inserted = self._repository.commit_fetch(
                    session_key=session_key,
                    source_id=config.source_id,
                    result=result,
                    fetched_at=fetched_at,
                )
                return config.source_id, inserted, None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return config.source_id, None, f"{type(exc).__name__}: {exc}"

        results = await asyncio.gather(*(fetch_one(config) for config in configs))
        succeeded = tuple(source_id for source_id, inserted, _ in results if inserted is not None)
        failed = {source_id: error for source_id, _, error in results if error is not None}
        for source_id, error in failed.items():
            logger.warning("Proactive Source 拉取失败 %s: %s", source_id, error)
        return CollectionReport(
            succeeded=succeeded,
            failed=failed,
            inserted_events=sum(inserted or 0 for _, inserted, _ in results),
        )

    async def replay_pending_acknowledgements(self, *, now: datetime) -> AckReplayReport:
        configs = {source.source_id: source for source in load_proactive_sources(self._config_path)}
        acknowledged = 0
        failed = 0
        for pending in self._repository.list_pending_acknowledgements(now):
            try:
                config = configs[pending.source_id]
                await self._ack(
                    config,
                    pending.source_event_id,
                    ttl_hours=pending.ttl_hours,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed += 1
                self._repository.mark_ack_failed(
                    pending.acknowledgement_id,
                    failed_at=now,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                acknowledged += 1
                self._repository.mark_acknowledged(
                    pending.acknowledgement_id,
                    acknowledged_at=now,
                )
        return AckReplayReport(acknowledged, failed)

    async def _fetch(
        self,
        config: ProactiveSourceConfig,
        *,
        fetched_at: datetime,
    ) -> ProactiveFetchResult:
        caller = self._caller_for_server(config.server)
        try:
            async with asyncio.timeout(config.fetch_timeout_seconds):
                if config.poll_tool:
                    poll_result = await caller.call_tool(config.poll_tool, {})
                    if not poll_result.ok:
                        raise McpInvocationError(
                            "proactive_source_error",
                            poll_result.error_message or "主动信息源轮询失败",
                        )
                result = await caller.call_tool(
                    config.get_tool,
                    {},
                )
        except TimeoutError as exc:
            raise McpInvocationError("proactive_source_timeout", "主动信息源拉取超时") from exc
        return _parse_fetch_result(config, result, fetched_at=fetched_at)

    async def _ack(
        self,
        config: ProactiveSourceConfig,
        event_id: str,
        *,
        ttl_hours: int,
    ) -> None:
        if not config.ack_tool:
            return
        caller = self._caller_for_server(config.server)
        arguments: dict[str, object] = {
            "event_ids": [event_id],
        }
        if config.channel == "content":
            arguments["ttl_hours"] = ttl_hours
        try:
            async with asyncio.timeout(config.ack_timeout_seconds):
                result = await caller.call_tool(
                    config.ack_tool,
                    arguments,
                )
        except TimeoutError as exc:
            raise McpInvocationError("proactive_ack_timeout", "主动信息源 ACK 超时") from exc
        payload = _decode_tool_result(result)
        acknowledged = payload.get("acknowledged") if isinstance(payload, dict) else None
        failed = payload.get("failed", []) if isinstance(payload, dict) else []
        if (
            not isinstance(acknowledged, list)
            or event_id not in {str(item) for item in acknowledged}
            or (isinstance(failed, list) and event_id in {str(item) for item in failed})
        ):
            raise McpInvocationError(
                "proactive_ack_error",
                result.error_message or "主动信息源 ACK 未确认成功",
            )


def stable_ack_operation_id(source_id: str, event_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:proactive-ack:{source_id}:{event_id}"))


def _parse_source(raw: object) -> ProactiveSourceConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("proactive_sources.json 的每个 source 必须是对象")
    server = str(raw.get("server") or raw.get("server_id") or "").strip()
    channel = str(raw.get("channel") or "").strip()
    source_id = str(raw.get("id") or raw.get("source_id") or "").strip()
    if not source_id and server and channel:
        # 兼容旧原型：同一 MCP Server 可以按 alert/content/context
        # 配置多条来源，因此不能只用 server 作为唯一键。
        source_id = f"{server}:{channel}"
    return ProactiveSourceConfig(
        source_id=source_id,
        server=server,
        channel=channel,  # type: ignore[arg-type]
        get_tool=str(raw.get("get_tool") or raw.get("fetch_tool") or ""),
        ack_tool=str(raw.get("ack_tool") or ""),
        poll_tool=str(raw.get("poll_tool") or ""),
        fetch_timeout_seconds=float(raw.get("fetch_timeout_seconds", 30)),
        ack_timeout_seconds=float(raw.get("ack_timeout_seconds", 30)),
    )


def _parse_fetch_result(
    config: ProactiveSourceConfig,
    result: McpCallResult,
    *,
    fetched_at: datetime,
) -> ProactiveFetchResult:
    data = _decode_tool_result(result)
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        raw_events: object = data["events"]
    elif config.channel == "context" and isinstance(data, dict):
        raw_events = [data]
    else:
        raw_events = data
    if not isinstance(raw_events, list):
        raise McpInvocationError(
            "proactive_source_contract",
            "主动信息源必须返回 JSON 事件数组",
        )
    events: list[ProactiveEvent] = []
    seen: set[str] = set()
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        event_id = str(raw.get("event_id") or raw.get("id") or "").strip()
        kind = str(raw.get("kind") or config.channel)
        if kind != config.channel:
            continue
        if not event_id:
            canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            event_id = "source_" + hashlib.sha1(canonical.encode()).hexdigest()[:16]
        if event_id in seen:
            continue
        occurred_at = str(
            raw.get("occurred_at")
            or raw.get("published_at")
            or raw.get("updated_at")
            or raw.get("last_seen")
            or raw.get("timestamp")
            or fetched_at.isoformat()
        )
        if not _valid_time(occurred_at):
            raise McpInvocationError("proactive_source_contract", "event 时间字段无效")
        nested_payload = raw.get("payload")
        payload = dict(nested_payload) if isinstance(nested_payload, dict) else dict(raw)
        payload.pop("event_id", None)
        payload.pop("id", None)
        payload.pop("kind", None)
        payload.setdefault("source", config.server)
        seen.add(event_id)
        events.append(ProactiveEvent(config.source_id, event_id, kind, occurred_at, payload))
    return ProactiveFetchResult(tuple(events))


def _decode_tool_result(result: McpCallResult) -> object:
    if not result.ok:
        raise McpInvocationError(
            "proactive_source_error",
            result.error_message or "主动信息源调用失败",
        )
    if result.structured is not None:
        value: object = result.structured
        if set(result.structured) == {"result"}:
            value = result.structured["result"]
        if not isinstance(value, str):
            return value
        return _decode_json_text(value)
    for block in result.content:
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            return _decode_json_text(text)
    raise McpInvocationError("proactive_source_error", "主动信息源缺少 JSON 文本结果")


def _decode_json_text(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise McpInvocationError(
            "proactive_source_contract", "主动信息源返回的文本不是合法 JSON"
        ) from exc


def _valid_time(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


__all__ = [
    "AckReplayReport",
    "CollectionReport",
    "McpCaller",
    "ProactiveSourceConfig",
    "ProactiveSourceGateway",
    "ProactiveEvent",
    "ProactiveFetchResult",
    "load_proactive_sources",
    "stable_ack_operation_id",
]
