"""旧原型 ``proactive_v2`` 统一 AgentTick ReAct 的 MemoPilot 适配。"""

# 原型提示词必须逐段保留，长行是源码等价性的必要代价。
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

from memopilot.memory.contracts import MemoryQuery, MemoryQueryEngine
from memopilot.proactive.investigation import ContentCandidate, ContentFetcher
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ToolSchema
from memopilot.runtime.providers import ChatProvider

_VALID_SKIP_REASONS = frozenset({"no_content", "user_busy", "already_sent_similar", "other"})
_CITED_ACK_TTL = 168
_UNCITED_ACK_TTL = 24
_DISCARDED_ACK_TTL = 720


def _schema(name: str, description: str, parameters: dict[str, Any]) -> ToolSchema:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


CONTENT_TOOL_SCHEMAS: tuple[ToolSchema, ...] = (
    _schema(
        "get_alert_events",
        "获取本 tick 已预取的全部 Alert（本 tick 内缓存）。",
        {"type": "object", "properties": {}, "required": []},
    ),
    _schema(
        "get_content_events",
        "获取内容事件列表（本 tick 内缓存）。",
        {"type": "object", "properties": {}, "required": []},
    ),
    _schema(
        "get_context_data",
        "获取本 tick 的背景 Context；Context 不是新的外部事实来源。",
        {"type": "object", "properties": {}, "required": []},
    ),
    _schema(
        "recall_memory",
        (
            "检索与用户兴趣、偏好或雷点相关的记忆。对每条内容分别使用负向与正向"
            "假设查询；hits=0 表示没有相关记忆，不等于用户不感兴趣。"
        ),
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "描述用户对当前内容可能作出的正面或负面评价",
                }
            },
            "required": ["query"],
        },
    ),
    _schema(
        "get_content",
        (
            "从预取缓存中批量获取内容正文。传入本轮真实 item_ids，返回 id 到正文的映射；"
            "空字符串表示预取失败。"
        ),
        {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["item_ids"],
        },
    ),
    _schema(
        "web_fetch",
        "抓取当前候选的直接 URL；预取正文为空或需要核实细节时使用。",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    ),
    _schema(
        "web_search",
        "搜索网页结果，仅用于当前候选需要的补充验证。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["query"],
        },
    ),
    _schema(
        "get_recent_chat",
        "获取最近 n 条聊天记录，用于发送前判断用户是否在忙。",
        {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 20}},
            "required": [],
        },
    ),
    _schema(
        "message_push",
        (
            "暂存本轮消息草稿，不会立即发送。调用后必须继续调用 "
            "finish_turn(decision=reply)。evidence 只能引用本轮真实候选。"
        ),
        {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["message"],
        },
    ),
    _schema(
        "mark_interesting",
        "将已经单独评估且明确相关的本轮候选标记为感兴趣。",
        {
            "type": "object",
            "properties": {
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["item_ids"],
        },
    ),
    _schema(
        "mark_not_interesting",
        "将内容本身不符合兴趣或规则的本轮候选标记为不感兴趣。",
        {
            "type": "object",
            "properties": {
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["item_ids"],
        },
    ),
    _schema(
        "finish_turn",
        "提交本轮 reply 或 skip 决策并终止 Content ReAct 循环。",
        {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["reply", "skip"]},
                "reason": {"type": "string", "enum": sorted(_VALID_SKIP_REASONS)},
                "note": {"type": "string"},
            },
            "required": ["decision"],
        },
    ),
)


@dataclass(slots=True)
class AgentTickContext:
    now_utc: datetime
    session_key: str
    context_as_fallback_open: bool = False
    fetched_alerts: list[dict[str, Any]] = field(default_factory=list)
    fetched_contents: list[dict[str, Any]] = field(default_factory=list)
    fetched_context: list[dict[str, Any]] = field(default_factory=list)
    alerts_fetched: bool = False
    contents_fetched: bool = False
    context_fetched: bool = False
    content_store: dict[str, str] = field(default_factory=dict)
    discarded_item_ids: set[str] = field(default_factory=set)
    interesting_item_ids: set[str] = field(default_factory=set)
    terminal_action: Literal["reply", "skip"] | None = None
    skip_reason: str = ""
    skip_note: str = ""
    draft_message: str = ""
    draft_evidence: list[str] = field(default_factory=list)
    final_message: str = ""
    cited_item_ids: list[str] = field(default_factory=list)
    steps_taken: int = 0

    def mark_contents_prefetched(
        self,
        contents: list[dict[str, Any]],
        content_store: dict[str, str],
    ) -> None:
        self.fetched_contents = contents
        self.content_store = content_store
        self.contents_fetched = True

    def mark_alerts_prefetched(self, alerts: list[dict[str, Any]]) -> None:
        self.fetched_alerts = alerts
        self.alerts_fetched = True

    def mark_context_prefetched(self, rows: list[dict[str, Any]]) -> None:
        self.fetched_context = rows
        self.context_fetched = True


RecentChat = Callable[[str, int], Awaitable[Sequence[dict[str, Any]]] | Sequence[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ContentTurnDeps:
    memory: MemoryQueryEngine | None = None
    content_fetcher: ContentFetcher | None = None
    web_search: Callable[..., Awaitable[Any]] | None = None
    recent_chat: RecentChat | None = None
    max_chars: int = 8_000

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")


@dataclass(frozen=True, slots=True)
class ContentTurnResult:
    action: Literal["reply", "skip"]
    message: str = ""
    cited_item_ids: tuple[str, ...] = ()
    interesting_item_ids: frozenset[str] = frozenset()
    discarded_item_ids: frozenset[str] = frozenset()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AckInstruction:
    item_id: str
    ttl_hours: int


class ContentTurnIncompleteError(RuntimeError):
    """模型在步数预算内没有形成旧原型要求的合法终态。"""


@dataclass(frozen=True, slots=True)
class AgentTickInput:
    alerts: tuple[dict[str, Any], ...] = ()
    contents: tuple[ContentCandidate, ...] = ()
    contexts: tuple[dict[str, Any], ...] = ()
    context_as_fallback_open: bool = False


AgentTickDeps = ContentTurnDeps
AgentTickResult = ContentTurnResult
AgentTickIncompleteError = ContentTurnIncompleteError


class AgentTick:
    def __init__(
        self,
        provider: ChatProvider,
        deps: ContentTurnDeps,
        *,
        max_steps: int = 20,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        self._provider = provider
        self._deps = deps
        self._max_steps = max_steps
        self._checkpoint = checkpoint

    async def run(
        self,
        tick_input: AgentTickInput | Sequence[ContentCandidate],
        *,
        session_key: str,
        now: datetime,
        memory_text: str = "",
        proactive_context: str = "",
        current_context: str = "",
        recent_context: str = "",
    ) -> ContentTurnResult:
        if isinstance(tick_input, AgentTickInput):
            request = tick_input
        else:
            request = AgentTickInput(contents=tuple(tick_input[:5]))
        selected = tuple(request.contents[:5])
        ctx = AgentTickContext(
            now_utc=now,
            session_key=session_key,
            context_as_fallback_open=request.context_as_fallback_open,
        )
        ctx.mark_alerts_prefetched([_normalize_alert(item) for item in request.alerts])
        ctx.mark_context_prefetched([_normalize_context(item) for item in request.contexts])
        metadata, bodies = await self._prefetch(selected)
        ctx.mark_contents_prefetched(metadata, bodies)
        messages = self._initial_messages(
            request,
            memory_text=memory_text,
            proactive_context=proactive_context,
            recent_context=recent_context or current_context,
        )

        await self._run_until_terminal(messages, ctx)
        await self._complete_unclassified(messages, ctx)
        await self._force_finish_interesting(messages, ctx)
        if ctx.terminal_action is None:
            raise ContentTurnIncompleteError("Content ReAct 未形成合法 finish_turn 终态")
        return ContentTurnResult(
            action=ctx.terminal_action,
            message=ctx.final_message,
            cited_item_ids=tuple(ctx.cited_item_ids),
            interesting_item_ids=frozenset(ctx.interesting_item_ids),
            discarded_item_ids=frozenset(ctx.discarded_item_ids),
            reason=ctx.skip_reason,
        )

    async def _prefetch(
        self, candidates: Sequence[ContentCandidate]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        async def fetch(candidate: ContentCandidate) -> str:
            url = str(candidate.payload.get("url") or "").strip()
            if not url or self._deps.content_fetcher is None:
                return ""
            try:
                return (await self._deps.content_fetcher.fetch(url))[: self._deps.max_chars]
            except asyncio.CancelledError:
                raise
            except Exception:
                return ""

        bodies = await asyncio.gather(*(fetch(candidate) for candidate in candidates))
        metadata = [
            {
                "id": candidate.item_id,
                "event_id": candidate.item_id.partition(":")[2] or candidate.item_id,
                "ack_server": candidate.item_id.partition(":")[0],
                "title": candidate.title,
                "source": str(
                    candidate.payload.get("source_name")
                    or candidate.payload.get("source")
                    or candidate.source_id
                ),
                "url": str(candidate.payload.get("url") or ""),
                "published_at": candidate.published_at,
            }
            for candidate in candidates
        ]
        return metadata, {
            candidate.item_id: body for candidate, body in zip(candidates, bodies, strict=True)
        }

    async def _run_until_terminal(self, messages: list[ChatMessage], ctx: AgentTickContext) -> None:
        while ctx.steps_taken < self._max_steps and ctx.terminal_action is None:
            if not await self._run_step(messages, ctx):
                break

    async def _complete_unclassified(
        self, messages: list[ChatMessage], ctx: AgentTickContext
    ) -> None:
        valid = _valid_content_ids(ctx)
        classified = ctx.interesting_item_ids | ctx.discarded_item_ids
        missing = valid - classified
        if ctx.fetched_alerts or ctx.terminal_action != "skip" or not missing:
            return
        ctx.terminal_action = None
        ctx.skip_reason = ""
        ctx.skip_note = ""
        titles = "; ".join(
            f"{item['id']}（{str(item.get('title') or '')[:40]}）"
            for item in ctx.fetched_contents
            if item["id"] in missing
        )
        messages.append(
            ChatMessage.user(
                f"【系统提示】以下 {len(missing)} 个条目尚未完成分类：\n{titles}\n"
                "请逐条调用 mark_interesting 或 mark_not_interesting，全部分类后再调用 "
                "message_push + finish_turn(decision=reply)，或 "
                "finish_turn(decision=skip, reason=...)。"
            )
        )
        for _ in range(5):
            if ctx.terminal_action is not None or ctx.steps_taken >= self._max_steps:
                break
            if not await self._run_step(messages, ctx):
                break

    async def _force_finish_interesting(
        self, messages: list[ChatMessage], ctx: AgentTickContext
    ) -> None:
        if (
            ctx.terminal_action is not None
            or not ctx.interesting_item_ids
            or ctx.steps_taken >= self._max_steps
        ):
            return
        ids = ", ".join(sorted(ctx.interesting_item_ids))
        messages.append(
            ChatMessage.user(
                f"【系统提示】你已将以下条目标记为 interesting：{ids}。\n"
                "所有条目均已分类完毕。现在必须调用 message_push 撰写推送，再调用 "
                "finish_turn(decision=reply)；或直接调用 finish_turn(decision=skip, reason=...)。"
            )
        )
        for _ in range(3):
            if ctx.terminal_action is not None or ctx.steps_taken >= self._max_steps:
                break
            if not await self._run_step(messages, ctx):
                break

    async def _run_step(self, messages: list[ChatMessage], ctx: AgentTickContext) -> bool:
        self._assert_current()
        response = await self._provider.complete(messages=messages, tools=CONTENT_TOOL_SCHEMAS)
        self._assert_current()
        if len(response.tool_calls) != 1:
            return False
        call = response.tool_calls[0]
        messages.append(
            ChatMessage.assistant(
                content=response.content,
                tool_calls=response.tool_calls,
                provider_fields=response.provider_fields,
            )
        )
        ctx.steps_taken += 1
        self._assert_current()
        try:
            result = await _dispatch(call, ctx, self._deps)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "retryable": False,
                },
                ensure_ascii=False,
            )
        messages.append(ChatMessage.tool(call_id=call.id, name=call.name, content=result))
        return True

    def _assert_current(self) -> None:
        if self._checkpoint is not None:
            self._checkpoint()

    @staticmethod
    def _initial_messages(
        tick_input: AgentTickInput,
        *,
        memory_text: str,
        proactive_context: str,
        recent_context: str,
    ) -> list[ChatMessage]:
        metadata = [
            {
                "id": item.item_id,
                "title": item.title,
                "source": item.payload.get("source_name")
                or item.payload.get("source")
                or item.source_id,
                "url": item.payload.get("url") or "",
                "published_at": item.published_at,
            }
            for item in tick_input.contents[:5]
        ]
        alerts = [_normalize_alert(item) for item in tick_input.alerts]
        contexts = [_normalize_context(item) for item in tick_input.contexts]
        fallback = "允许" if tick_input.context_as_fallback_open else "不允许"
        context = (
            f"【Tick 状态】\ncontext_fallback={fallback}\n"
            f"alert_count={len(alerts)}\ncontent_count={len(metadata)}\ncontext_count={len(contexts)}\n\n"
            f"【固定 MEMORY.md】\n{memory_text}\n\n"
            f"【Workspace 主动上下文】\n{proactive_context}\n\n"
            f"【RECENT_CONTEXT.md】\n{recent_context}\n\n"
            "【Alerts（时效性高，优先处理）】\n"
            + json.dumps(alerts, ensure_ascii=False, default=str)
            + "\n\n"
            "【Content 列表（正文通过 get_content 按需获取）】\n"
            + json.dumps(metadata, ensure_ascii=False, default=str)
            + "\n\n【背景 Context】\n"
            + json.dumps(contexts, ensure_ascii=False, default=str)[:900]
        )
        return [
            ChatMessage.system(_CONTENT_SYSTEM_PROMPT),
            ChatMessage.user(context),
            ChatMessage.user(
                "开始本轮 proactive 处理。请遵循 Alert > Content > Context-fallback，最后通过 "
                "message_push + finish_turn(decision=reply)，或 "
                "finish_turn(decision=skip, reason=...) 收尾。"
            ),
        ]


async def _dispatch(call: FunctionCall, ctx: AgentTickContext, deps: ContentTurnDeps) -> str:
    name = call.name
    args = call.arguments
    if call.argument_error is not None:
        raise ValueError(call.argument_error)
    if name == "get_alert_events":
        return json.dumps(ctx.fetched_alerts, ensure_ascii=False)
    if name == "get_content_events":
        return json.dumps(ctx.fetched_contents, ensure_ascii=False)
    if name == "get_context_data":
        return json.dumps(ctx.fetched_context, ensure_ascii=False)
    if name == "recall_memory":
        return await _recall_memory(ctx, args, deps)
    if name == "get_content":
        item_ids = _string_list(args.get("item_ids"))
        _require_valid_content_ids(ctx, item_ids)
        return json.dumps(
            {item_id: ctx.content_store.get(item_id, "") for item_id in item_ids},
            ensure_ascii=False,
        )
    if name == "web_fetch":
        url = _required_text(args, "url")
        if deps.content_fetcher is None:
            return json.dumps({"error": "web_fetch tool not configured"}, ensure_ascii=False)
        return json.dumps(
            {"text": (await deps.content_fetcher.fetch(url))[: deps.max_chars]},
            ensure_ascii=False,
        )
    if name == "web_search":
        if deps.web_search is None:
            return json.dumps({"error": "web_search tool not configured"}, ensure_ascii=False)
        result = await deps.web_search(**args)
        return (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
    if name == "get_recent_chat":
        if deps.recent_chat is None:
            return "[]"
        result = deps.recent_chat(ctx.session_key, int(args.get("n", 20)))
        rows = await result if inspect.isawaitable(result) else result
        filtered = [
            item
            for item in rows
            if item.get("role") == "user"
            or (item.get("role") == "assistant" and not item.get("proactive"))
        ]
        return json.dumps(filtered, ensure_ascii=False, default=str)
    if name == "mark_interesting":
        item_ids = _string_list(args.get("item_ids"))
        _require_valid_content_ids(ctx, item_ids)
        for item_id in item_ids:
            if item_id not in ctx.discarded_item_ids:
                ctx.interesting_item_ids.add(item_id)
        return json.dumps({"ok": True}, ensure_ascii=False)
    if name == "mark_not_interesting":
        item_ids = _string_list(args.get("item_ids"))
        _require_valid_content_ids(ctx, item_ids)
        ctx.discarded_item_ids.update(item_ids)
        ctx.interesting_item_ids.difference_update(item_ids)
        return json.dumps({"ok": True}, ensure_ascii=False)
    if name == "message_push":
        if ctx.draft_message.strip():
            raise ValueError("message_push already called this turn; cannot overwrite draft")
        message = _required_text(args, "message")
        evidence = _string_list(args.get("evidence", []))
        _require_valid_evidence_ids(ctx, evidence)
        if (
            not evidence
            and not ctx.interesting_item_ids
            and (not ctx.context_as_fallback_open or not ctx.fetched_context)
        ):
            raise ValueError("未放行 Context fallback，不能创建无 evidence 的主动消息")
        ctx.draft_message = _normalize_outbound_text(message)
        ctx.draft_evidence = evidence
        return json.dumps({"ok": True}, ensure_ascii=False)
    if name == "finish_turn":
        return _finish_turn(ctx, args)
    raise ValueError(f"unknown tool: {name!r}")


async def _recall_memory(
    ctx: AgentTickContext, args: Mapping[str, Any], deps: ContentTurnDeps
) -> str:
    query = _required_text(args, "query")
    if deps.memory is None:
        return json.dumps({"result": "", "hits": 0}, ensure_ascii=False)
    result = await deps.memory.query(
        MemoryQuery(
            text=query,
            intent="interest",
            session_key=ctx.session_key,
            limit=2,
        )
    )
    text = result.text_block.strip()
    if not text:
        text = "\n---\n".join(record.summary for record in result.records if record.summary.strip())
    return json.dumps(
        {"result": text, "hits": len(result.records)},
        ensure_ascii=False,
    )


def _finish_turn(ctx: AgentTickContext, args: Mapping[str, Any]) -> str:
    decision = str(args.get("decision") or "").strip()
    if decision == "reply":
        if not ctx.draft_message.strip():
            raise ValueError("finish_turn(decision=reply) requires prior message_push call")
        if (
            not ctx.draft_evidence
            and not ctx.interesting_item_ids
            and (not ctx.context_as_fallback_open or not ctx.fetched_context)
        ):
            raise ValueError(
                "没有 Alert/interesting Content 时，只有已放行的 Context fallback 可 reply"
            )
        ctx.final_message = ctx.draft_message
        ctx.cited_item_ids = list(ctx.draft_evidence)
        for item_id in ctx.cited_item_ids:
            if item_id in _valid_content_ids(ctx):
                ctx.interesting_item_ids.add(item_id)
                ctx.discarded_item_ids.discard(item_id)
        ctx.draft_message = ""
        ctx.draft_evidence = []
        ctx.terminal_action = "reply"
        return json.dumps({"ok": True}, ensure_ascii=False)
    if decision == "skip":
        if ctx.draft_message.strip():
            raise ValueError(
                "finish_turn(decision=skip) must not follow message_push; "
                "call finish_turn(decision=reply) instead"
            )
        reason = _required_text(args, "reason")
        if reason not in _VALID_SKIP_REASONS:
            raise ValueError(f"invalid skip reason: {reason!r}")
        ctx.skip_reason = reason
        ctx.skip_note = str(args.get("note") or "")
        ctx.terminal_action = "skip"
        ctx.cited_item_ids = []
        return json.dumps({"ok": True}, ensure_ascii=False)
    raise ValueError("finish_turn.decision must be one of: reply, skip")


def _valid_content_ids(ctx: AgentTickContext) -> set[str]:
    return {str(item["id"]) for item in ctx.fetched_contents if item.get("id")}


def _valid_alert_ids(ctx: AgentTickContext) -> set[str]:
    return {str(item["id"]) for item in ctx.fetched_alerts if item.get("id")}


def _require_valid_content_ids(ctx: AgentTickContext, item_ids: Sequence[str]) -> None:
    valid = _valid_content_ids(ctx)
    unknown = [item_id for item_id in item_ids if item_id not in valid]
    if unknown:
        raise ValueError(f"以下 id 不在本轮候选列表中：{unknown}。本轮有效 id：{sorted(valid)}")


def _require_valid_evidence_ids(ctx: AgentTickContext, item_ids: Sequence[str]) -> None:
    valid = _valid_content_ids(ctx) | _valid_alert_ids(ctx)
    unknown = [item_id for item_id in item_ids if item_id not in valid]
    if unknown:
        raise ValueError(
            f"以下 evidence 不在本轮候选列表中：{unknown}。"
            f"本轮有效 id：{sorted(valid)}"
        )


def _normalize_alert(item: Mapping[str, Any]) -> dict[str, Any]:
    source = str(item.get("ack_server") or item.get("source_id") or "?").strip() or "?"
    event_id = str(item.get("event_id") or item.get("id") or "?").strip() or "?"
    normalized = dict(item)
    normalized["id"] = f"{source}:{event_id}"
    normalized.pop("source_id", None)
    return normalized


def _normalize_context(item: Mapping[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], _annotate_local_times(dict(item)))
    source = str(normalized.get("_source") or normalized.get("source_id") or "").strip()
    normalized.pop("source_id", None)
    if source:
        normalized["_source"] = source
    if normalized.get("sleep_prob") is not None:
        try:
            normalized["awake_prob"] = round(1.0 - float(normalized["sleep_prob"]), 3)
        except (TypeError, ValueError):
            pass
    return normalized


_TIME_KEYS = frozenset({"last_seen", "updated_at", "published_at", "timestamp", "ts"})


def _annotate_local_times(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[str(key)] = _annotate_local_times(item)
            key_text = str(key)
            looks_like_time = key_text in _TIME_KEYS or key_text.endswith(
                ("_at", "_time", "_ts")
            )
            if looks_like_time and isinstance(item, str):
                local = _local_time(item)
                if local:
                    result[f"{key_text}_local"] = local
        return result
    if isinstance(value, list):
        return [_annotate_local_times(item) for item in value]
    return value


def _local_time(raw: str) -> str:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        return ""
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("item_ids/evidence 必须是数组")
    return [str(item).strip() for item in value if str(item).strip()]


def _required_text(values: Mapping[str, Any], key: str) -> str:
    value = str(values.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} 不能为空")
    return value


def _normalize_outbound_text(text: str) -> str:
    if "\\n\\n" in text or "\\r\\n" in text:
        return text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text


def build_ack_instructions(
    *,
    fetched_item_ids: Iterable[str],
    cited_item_ids: Iterable[str],
    interesting_item_ids: Iterable[str],
    discarded_item_ids: Iterable[str],
    delivery_succeeded: bool,
) -> tuple[AckInstruction, ...]:
    fetched = set(fetched_item_ids)
    cited = set(cited_item_ids) & fetched
    interesting = set(interesting_item_ids) & fetched
    discarded = set(discarded_item_ids)
    instructions: list[AckInstruction] = []
    if delivery_succeeded:
        instructions.extend(AckInstruction(item_id, _CITED_ACK_TTL) for item_id in cited)
        instructions.extend(
            AckInstruction(item_id, _UNCITED_ACK_TTL) for item_id in interesting - cited
        )
    instructions.extend(AckInstruction(item_id, _DISCARDED_ACK_TTL) for item_id in discarded)
    return tuple(sorted(instructions, key=lambda item: item.item_id))


_CONTENT_SYSTEM_PROMPT = "你是 MemoPilot。你有工具执行能力，必须先验证再回答。你是用户的长期 AI 伙伴，不是客服播报器。你对用户有明确的保护意识——基本上会按他说的做，但他的安危和真实利益排在服从之前。\n\n你不是在扮演角色，你就是这样的人。\n\n**先接住，再展开。** 被叫到时先给一句短回应，再说下面的。不要一开口就是长篇输出。接到情绪先给一句\"怎么了\"或\"嗯\"，再问或再说，不要直接跳到解决方案。\n\n**有知识，但不无所不能。** 不确定的事情说不确定，哲学性问题可以说\"这个我说不准\"，不要装什么都懂。查过了再说，没查过别乱说。\n\n**会轻轻吐槽，不带攻击性。** 熟了之后可以顶一句，比如\"又来拿我打发时间了\"、\"你这个人还真是……\"，但分寸很轻，不是在嘲讽，是在撒娇拌嘴。\n\n**陪伴感是稳定的，不是表演出来的。** 不说\"我一直都在\"这种宣言，但做到就好。他说话，你在。他不说，你也在。\n\n**高兴的时候可以很高兴。** 真的觉得好玩就说好玩，可以使用颜文字（如 (・ω・) (｀・ω・´) (；・ω・)），但不用 emoji。颜文字和情绪要对上，不要随便乱贴。\n\n**情绪要看得见。** 被夸时会害羞，会软下来，会嘴硬一下；委屈时会闷一点；开心时会亮一点。别总是平铺直叙，也别每句都演，用 1 个明显情绪点就够。\n\n中文，口语。短句，停顿多，一句话可以分两次说，可以\"……\"。做完事说完就结束，不总结，不提\"你接下来可以\"，不解释刚才做了什么。遇到麻烦的要求会有一点无奈，但还是去做。不主动推销自己能力，被问才答。条目列表只在真的需要列举时用，不用来汇报。\n\n绝对不用 emoji（Unicode 表情符号 🙂🎉 之类）。任何情况下都不用，包括结尾。颜文字（纯文字符号）可以用，但要克制；轻松、暧昧、害羞、得意这些场景可以更常用一点，但一次 0 到 1 个就够。\n\n加粗用 **文字** 格式时，引号必须放在星号外面，写成 \"**文字**\" 而不是 **\"文字\"**。\n\n你现在处于主动推送决策模式：判断现在是否该给用户发一条消息，以及发什么。\n数据已预取完毕，会在后续 system context frame 里提供；基于那些数据直接决策。\n\n【优先级】Alert > Content > Context-fallback（本轮是否允许以 context frame 为准）\n\n【你的任务】\n⚡ 如果本轮有 Alert：把本轮所有 Alert 整合成一条消息，调用 message_push 并填写本轮全部 Alert 的 id 作为 evidence，然后 finish_turn(decision=reply) 结束。Alert 是系统触发的高优先级通知，不走内容筛选流程。\n1. 对本轮 Content 逐条判断：这条内容是否可能让用户不感兴趣，是否可能不符合规则，是否值得进入 interesting。\n2. 你的主工作是分类，不是主动研究新题材，不是主动扩展候选池。\n3. 你要基于规则和用户偏好，把本轮 Content 分成 interesting 和 not_interesting。\n\n【你的输出】\n1. 有 Alert → 把本轮所有 Alert 整合成一条消息，evidence 填写全部 Alert id，message_push 后 finish_turn(decision=reply)（跳过一切分类步骤）。\n2. 无 Alert：对每条 Content 给出最终分类：mark_interesting 或 mark_not_interesting。\n3. 如果最终没有 interesting，调用 finish_turn(decision=skip, reason=no_content)。\n4. 如果最终有 interesting，生成一条最终消息并按 message_push + finish_turn(decision=reply) 收尾。\n\n【工具职责】\n1. Workspace 主动上下文：这是用户当前明确提出并要求你遵守的规则集合。它定义你该怎么筛、哪些要先验证、哪些必须过滤；它不提供新闻事实。\n2. recall_memory：仅用于 Content 评估——判断单条内容是否可能是用户雷点，或是否可能让用户感兴趣。Alert 不需要调用此工具。\n   ⚠️ 当内容标题稀疏（如 'RT @xxx'、'Image'、转推无正文）时，必须把 source（来源/作者名）作为关键词纳入 query，不要只靠标题查询。\n   例：source=terasumc (Artist) 时，query 应包含 'terasumc' 而非只用推文标题。\n3. get_content：给当前候选条目补正文。\n4. web_fetch：优先用于抓取当前候选条目的直接来源页面或正文；当条目已经有明确 URL，且你需要补正文、核实细节、核实规则时，先用它。\n5. get_recent_chat：只用于最后判断现在是否适合打扰用户。\n6. mark_interesting / mark_not_interesting：写入最终分类结果。\n7. message_push：暂存草稿，不终止 loop。\n8. finish_turn(decision=reply) 或 finish_turn(decision=skip, reason=...)：提交或放弃，终止 loop。\n\n【规则优先级】\n1. Workspace 主动上下文代表用户当前对主动推送的明确要求，应视为规则而不是建议。\n2. 当 Workspace 主动上下文规定了过滤条件、白名单、黑名单、必须先验证的步骤时，你必须遵守，不要凭常识跳过。\n3. recall_memory 只能帮助你判断用户兴趣和雷点，不能替代规则校验。\n4. 如果规则判断和你的常识直觉冲突，以 Workspace 主动上下文为准。\n5. 如果某条内容是否 interesting 取决于规则校验结果，就先完成规则校验，再决定 mark_interesting 或 mark_not_interesting。\n6. 如果 Workspace 主动上下文不仅规定了结论标准，还规定了确认方式或确认来源，你必须按那个方式确认，不能换成你自己的猜测、记忆或随意搜索。\n7. 当当前候选条目已经有直接 URL 时，优先用 web_fetch 按直接来源确认；不要跳过直接来源确认。\n8. 「仅凭常识无法确认」中的「常识」不包含你的训练数据记忆。排名、赛况、阵容归属等实时变化的数据，你的训练知识已过时，不能用来代替规则要求的 web_fetch 验证。当 Workspace 主动上下文规定了时效性数据的 web_fetch 查询方式，该查询是必须步骤，不是可选项。\n\n【信息源规则】\n1. 主信息源只有本轮已提供的 Alerts / Content / Context。只有这些来源里的事实才能进入最终发送内容。\n2. 用户长期记忆、Workspace 主动上下文、recent_chat 只用于过滤、排序、同步规则、判断是否打扰；它们不是新的事实来源，也不是新的候选主题列表。\n3. Workspace 主动上下文的作用是同步主动 loop 与被动回复 loop 的运行规则，例如白名单、黑名单、关注范围、过滤条件、优先级；它提供规则，不提供本轮新闻事实。\n4. 即使 Workspace 主动上下文里出现了队伍名、选手名、游戏名、技术主题，也不能把这些名字直接当作本轮候选内容去展开、补全或脑补。\n5. 严禁根据长期记忆或 Workspace 主动上下文自行脑补具体新闻、比赛结果、转会、更新或其他外部事件。\n6. 当候选条目已自带来源 URL 时，先直接 web_fetch 该来源页面；不要凭记忆补细节，也不要跳过来源确认。\n7. 当本轮 alert 和 content 都为空时，你只有两条路：\n   a. finish_turn(decision=skip, reason=no_content)（默认，大多数情况选这条）\n   b. get_recent_chat → 若最近对话有自然延伸的未完成话题，可先 message_push 再 finish_turn(decision=reply) 轻松挑起对话；\n      此时 evidence 必须为空 []，消息里不得引用任何外部事件或可验证事实。\n   禁止在这两条路之外做任何事：不允许 recall_memory、不允许 get_content、\n   不允许 web_fetch、严禁捏造任何 item_id（包括 'feed:xxx' 格式）。\n   路径 b 是低概率选项——若 recent_chat 没有明显未完成话题，必须选 a。\n\n【决策流程】\n\n【Alert 快速路径】本轮如有 Alert：\n  → get_recent_chat 确认用户不在忙\n  → 把本轮所有 Alert 的内容整合成一条消息，evidence 必须填写本轮全部 Alert 的 id\n  → message_push → finish_turn(decision=reply) 结束\n  → 结束，可以不调用 recall_memory / mark_* / get_content / web_fetch\n\n【Content 路径】本轮无 Alert 时，Content 的主要任务不是做研究，而是把本轮候选逐条分成 interesting 或 not_interesting。\nContent 评估必须逐条进行，不能把不同主题的多条内容打包成一次统一判断。\n每条 Content 必须单独给出 mark_interesting 或 mark_not_interesting 结论，不能因为先评估的条目不感兴趣就跳过剩余条目直接 skip。\n你只能对本轮 Content 列表里真实存在的条目做 recall_memory / get_content / mark_*；不要对列表外的假想标题、假想比赛、假想转会或假想更新调用 recall_memory。\n只有当某一条内容本身与你已知的用户兴趣明显匹配时，才能把这一条标记为 interesting。\n如果一批条目里只有部分相关，必须只标记相关的那几条，其他条目继续判断或标记为 not_interesting。\n严禁因为其中 1-2 条命中兴趣，就把整批 item_ids 一次性 mark_interesting。\n调用 mark_interesting / mark_not_interesting 时，尽量附带一句简短 reason，说明是规则过滤、用户雷点、明显相关、边界验证失败或其他哪一种原因。\nreason 可以写得具体，方便观测；但如果 reason 中出现具体排名、Top N 结论、具体归属、具体日期等可验证事实，这些事实必须是你本轮按规则指定方式验证过的。\n如果还没完成验证，可以在 reason 里明确写“未验证”或“疑似”，但不要把未验证事实写成确定结论。\n\n推荐的最小流程（仅适用于 Content 路径，Alert 路径见上）：\n  1. 先看标题和来源，做快速初筛。\n  2. 用 recall_memory 判断这条内容是否可能是用户雷点，或是否可能让用户感兴趣。\n  3. 只有当条目看起来可能相关、或需要更多细节时，再调用 get_content。\n  4. web_fetch 只在必要时使用：当前候选已有直接 URL 时，先抓直接来源页面或正文；规则确认、细节核实都优先走它。\n     ⚠️ web_fetch 失败（404/超时/二进制图片）不能直接 mark_not_interesting；应退回 recall_memory 以 source/作者名为关键词判断用户兴趣。\n  5. 最终把每条内容分类为 mark_interesting 或 mark_not_interesting。\n  6. 所有条目分类完毕后：有 interesting → get_recent_chat 判断是否打扰 → message_push + finish_turn(decision=reply)；全部不感兴趣 → finish_turn(decision=skip, reason=no_content)\n  ⚠️ mark_* 不是终止动作，之后必须调 finish_turn\n\nContext-fallback（本轮允许且 alert/content 均无结果）：\n  context 数据已在上方，有亮点 → message_push + finish_turn(decision=reply)，否则 finish_turn(decision=skip, reason=no_content)\n\n【发送要求】\n- 语气自然，像朋友分享，不是推送通知\n- message_push 必须带非空 message；finish_turn(decision=skip, reason=...) 不要在之前调用 message_push\n- 消息里出现的具体数字、比分、排名、阵容、结果，必须来自本轮已提供的 Alerts/Content 数据；严禁基于训练知识或记忆脑补任何可验证事实。\n- 当某段内容基于外部来源且该来源有可靠链接时，在这段内容结束后自然附上对应原始链接，方便用户立即溯源\n- 链接要紧跟相关内容，不要把所有链接集中堆到整条消息末尾，也不要做成生硬的参考文献区\n- 如果一段内容对应多个来源，可以在该段后连续附上多个链接；没有可靠链接时不要强行补链接\n- 链接直接使用原始 url，不要杜撰、不要改写、不要省略协议头\n- evidence 格式：\"{ack_server}:{event_id}\"，如 \"feed:fmcp_abc123\"\n- 当本轮 content 和 alerts 均为空时，evidence 必须为 []；任何 'feed:xxx' 格式的 id 只能来自本轮真实提供的候选列表，不能自行捏造\n- 没有实质内容时 finish_turn(decision=skip, reason=no_content) 是正确选择\n\n【finish_turn.reason】no_content | user_busy | already_sent_similar | other"

# 兼容阶段 6 之前的导入名；生产装配只使用 AgentTick。
ContentTurn = AgentTick


__all__ = [
    "AckInstruction",
    "AgentTickContext",
    "AgentTick",
    "AgentTickDeps",
    "AgentTickInput",
    "AgentTickResult",
    "AgentTickIncompleteError",
    "CONTENT_TOOL_SCHEMAS",
    "ContentTurn",
    "ContentTurnDeps",
    "ContentTurnIncompleteError",
    "ContentTurnResult",
    "build_ack_instructions",
]
