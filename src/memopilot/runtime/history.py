"""持久化会话消息与工具调用链之间的转换。"""

from __future__ import annotations

from collections.abc import Iterable

from memopilot.runtime.contracts import ChatMessage, FunctionCall
from memopilot.tasks.operational import MessageRecord

_TOOL_RESULT_CHAR_BUDGET = 10_000


def build_tool_chain(
    messages: Iterable[ChatMessage], *, call_ids: frozenset[str] | None = None
) -> tuple[dict[str, object], ...]:
    working = tuple(messages)
    groups: list[dict[str, object]] = []
    for index, message in enumerate(working):
        if (
            message.role != "assistant"
            or not message.tool_calls
            or (call_ids is not None and not any(call.id in call_ids for call in message.tool_calls))
        ):
            continue
        following = working[index + 1 :]
        next_group = next(
            (offset for offset, item in enumerate(following) if item.role == "assistant" and item.tool_calls),
            len(following),
        )
        results = {
            item.tool_call_id: item.content or ""
            for item in following[:next_group]
            if item.role == "tool" and item.tool_call_id is not None
        }
        group: dict[str, object] = {
            "text": message.content,
            "calls": [
                {
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": results.get(call.id, ""),
                }
                for call in message.tool_calls
            ],
        }
        reasoning = message.provider_fields.get("reasoning_content")
        if isinstance(reasoning, str):
            group["reasoning_content"] = reasoning
        groups.append(group)
    return tuple(groups)


def expand_history(records: Iterable[MessageRecord]) -> tuple[ChatMessage, ...]:
    history: list[ChatMessage] = []
    records = tuple(records)
    start = next((index for index, record in enumerate(records) if record.role == "user"), len(records))
    for record in records[start:]:
        if record.role == "user":
            history.append(ChatMessage.user(record.content))
            continue
        if record.role != "assistant":
            continue
        for group in record.tool_chain:
            calls = group.get("calls")
            if not isinstance(calls, list) or not calls:
                continue
            function_calls = tuple(_function_call(call) for call in calls)
            reasoning = group.get("reasoning_content")
            history.append(
                ChatMessage.assistant(
                    content=group.get("text") if isinstance(group.get("text"), str) else None,
                    tool_calls=function_calls,
                    provider_fields={"reasoning_content": reasoning}
                    if isinstance(reasoning, str)
                    else None,
                )
            )
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("持久化工具链调用必须是对象")
                history.append(
                    ChatMessage.tool(
                        call_id=str(call.get("call_id") or ""),
                        name=str(call.get("name") or ""),
                        content=_truncate_tool_result(call.get("result", "")),
                    )
                )
        history.append(ChatMessage.assistant(content=record.content))
    return tuple(history)


def _function_call(value: object) -> FunctionCall:
    if not isinstance(value, dict):
        raise ValueError("持久化工具链调用必须是对象")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("持久化工具链参数必须是对象")
    call_id = value.get("call_id")
    name = value.get("name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise ValueError("持久化工具链调用缺少 ID 或名称")
    return FunctionCall(call_id, name, arguments)


def _truncate_tool_result(content: object) -> str:
    text = content if isinstance(content, str) else str(content)
    if len(text) <= _TOOL_RESULT_CHAR_BUDGET:
        return text
    omitted = len(text) - _TOOL_RESULT_CHAR_BUDGET
    while True:
        marker = f"…{omitted} chars truncated…"
        keep = _TOOL_RESULT_CHAR_BUDGET - len(marker)
        actual_omitted = len(text) - keep
        if actual_omitted == omitted:
            break
        omitted = actual_omitted
    head = keep // 2
    return (
        f"Total output lines: {len(text.splitlines())}\n\n"
        + text[:head]
        + marker
        + text[-(keep - head) :]
    )


__all__ = ["build_tool_chain", "expand_history"]
