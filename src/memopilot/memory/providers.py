"""记忆模块与通用 Chat/Embedding Provider 的适配。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from openai import AsyncOpenAI

from memopilot.memory.prompts import (
    CONSOLIDATION_SYSTEM,
    CONSOLIDATION_USER,
    HYPOTHESIS_SYSTEM,
    IMPLICIT_LONG_TERM_SYSTEM,
    IMPLICIT_LONG_TERM_USER,
    MEMORY_OPTIMIZER_SYSTEM,
    MEMORY_OPTIMIZER_USER,
    RECENT_CONTEXT_SYSTEM,
    RECENT_CONTEXT_USER,
    SELF_OPTIMIZER_SYSTEM,
    SELF_OPTIMIZER_USER,
)
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.providers import ChatProvider


async def _complete_memory_task(
    provider: ChatProvider,
    *,
    messages: tuple[ChatMessage, ...],
    tools: tuple[ToolSchema, ...],
    max_output_tokens: int,
) -> ModelResponse:
    complete_task = getattr(provider, "complete_task", None)
    if callable(complete_task):
        response: ModelResponse = await complete_task(
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            thinking_enabled=False,
        )
        return response
    return await provider.complete(messages=messages, tools=tools)


class MarkdownProfileReader(Protocol):
    def read(self, name: str) -> str: ...


def _recent_history_entries(history: str, *, limit: int) -> list[str]:
    if not history.strip() or limit <= 0:
        return []
    entries = [value.strip() for value in re.split(r"\n\s*\n+", history) if value.strip()]
    return entries[-limit:]


class ChatConsolidationExtractor:
    def __init__(
        self,
        provider: ChatProvider,
        profile: MarkdownProfileReader | None = None,
    ) -> None:
        self.provider = provider
        self.profile = profile

    async def extract(self, conversation: str) -> dict[str, object]:
        current_memory = self.profile.read("MEMORY.md").strip() if self.profile else ""
        history = self.profile.read("HISTORY.md").strip() if self.profile else ""
        recent_history = "\n".join(
            f"- {entry}" for entry in _recent_history_entries(history, limit=3)
        )
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.system(CONSOLIDATION_SYSTEM),
                ChatMessage.user(
                    CONSOLIDATION_USER.format(
                        conversation=conversation,
                        current_memory=current_memory or "（空）",
                        recent_history=recent_history or "（空）",
                    )
                ),
            ),
            tools=(),
            max_output_tokens=1024,
        )
        content = (response.content or "").strip()
        if not content:
            raise ValueError("Consolidation 模型返回空响应")
        return _validate_consolidation_output(_load_json_object(content))


class ChatImplicitMemoryExtractor:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def extract(self, conversation: str) -> list[dict[str, object]]:
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.system(IMPLICIT_LONG_TERM_SYSTEM),
                ChatMessage.user(
                    IMPLICIT_LONG_TERM_USER.format(
                        conversation=conversation,
                        existing_profile="（空）",
                    )
                ),
            ),
            tools=(),
            max_output_tokens=600,
        )
        content = (response.content or "").strip()
        if not content:
            raise ValueError("长期记忆提取模型返回空响应")
        payload = _load_json_object(content)
        memories: list[dict[str, object]] = []
        for kind in ("profile", "preference", "procedure"):
            values = payload.get(kind, [])
            if not isinstance(values, list):
                raise ValueError(f"长期记忆 {kind} 必须是数组")
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError(f"长期记忆 {kind} 条目必须是对象")
                memories.append({"kind": kind, **value})
        normalized = _validate_consolidation_output(
            {"artifacts": {}, "memories": memories}
        )["memories"]
        assert isinstance(normalized, list)
        return normalized


class ChatRecentContextCompressor:
    _FIELDS = (
        "active_topics",
        "user_preferences",
        "follow_ups",
        "avoidances",
        "ongoing_threads",
    )

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def compress(
        self,
        *,
        old_context: str,
        conversation: str,
        recent_turns: str,
        compression_until: str,
    ) -> str:
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.system(RECENT_CONTEXT_SYSTEM),
                ChatMessage.user(
                    RECENT_CONTEXT_USER.format(
                        old_context=old_context or "（空）",
                        conversation=conversation or "（空）",
                        recent_turns=recent_turns or "（空）",
                    )
                ),
            ),
            tools=(),
            max_output_tokens=512,
        )
        content = (response.content or "").strip()
        if not content:
            return _replace_recent_turns(old_context, recent_turns)
        try:
            payload = _load_json_object(content)
            values = {
                field: _string_list(payload.get(field), limit=3) for field in self._FIELDS
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return _replace_recent_turns(old_context, recent_turns)
        return _render_recent_context(
            values,
            compression_until=compression_until,
            recent_turns=recent_turns,
        )


class ChatPostResponseModel:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def extract_invalidation_topics(self, user_message: str) -> list[str]:
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.user(
                    "判断用户消息是否在明确声明 agent 某个现有行为/流程有误，且希望废弃它。\n\n"
                    f"用户消息：{user_message}\n\n"
                    "【必须同时满足才触发】\n"
                    "1. 用户表达了明确的否定/纠错/废弃意图——句子里有“错了/不对/不要再/"
                    "忘掉/废弃/过时/改掉”等否定词\n"
                    "2. 否定的对象是 agent 的某个操作行为（不是用户自己的事，不是第三方信息）\n\n"
                    "【以下情况绝对不触发，返回 []】\n"
                    "✗ 用户在询问/确认 agent 的流程\n"
                    "✗ 用户在描述/回顾自己的操作\n"
                    "✗ 用户提问句、疑问句（即使涉及 agent 行为）\n"
                    "✗ 含“也许/可能/猜测”等不确定措辞且无明确废弃指令\n\n"
                    "若触发，提取受影响的行为主题。返回 JSON 数组，大多数消息应返回 []。"
                ),
            ),
            tools=(),
            max_output_tokens=96,
        )
        return _json_string_list(response.content)

    async def select_invalidated_ids(
        self,
        topic: str,
        candidates: list[dict[str, object]],
    ) -> list[str]:
        block = "\n".join(
            f"- id={item.get('item_id')} | {item.get('summary')}" for item in candidates
        )
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.user(
                    f"用户明确表示 agent 关于“{topic}”的现有行为/流程有误，需要废弃。\n"
                    "以下是数据库中与该主题相关的现有规则，判断哪些应被标记为废弃：\n\n"
                    f"{block}\n\n"
                    f"- 若条目确实描述了“{topic}”相关的 agent 操作流程/行为，输出其 id\n"
                    "- 若条目无关，不输出；若无关联条目，返回 []\n"
                    "只返回 JSON 数组，如 [\"abc123\"] 或 []"
                ),
            ),
            tools=(),
            max_output_tokens=96,
        )
        return _json_string_list(response.content)


class ChatProcedureTagger:
    def __init__(
        self,
        provider: ChatProvider,
        *,
        allowed_tools: set[str] | None = None,
        allowed_skills: set[str] | None = None,
    ) -> None:
        self.provider = provider
        self.allowed_tools = (
            None if allowed_tools is None else {value.casefold() for value in allowed_tools}
        )
        self.allowed_skills = (
            None if allowed_skills is None else {value.casefold() for value in allowed_skills}
        )

    async def tag(
        self,
        summary: str,
        *,
        tool_requirement: str | None,
        steps: list[str],
    ) -> dict[str, object] | None:
        response = await _complete_memory_task(
            self.provider,
            messages=(
                ChatMessage.system(
                    "为 Agent 流程记忆生成触发标签。只返回 JSON 对象："
                    '{"scope":"tool_triggered|global","tools":[],"skills":[],"keywords":[]}。'
                    "keywords 应使用用户自然语言中可能出现的具体中文/英文任务词，避免宽泛词。"
                ),
                ChatMessage.user(
                    f"流程：{summary}\n要求工具：{tool_requirement or '无'}\n"
                    f"步骤：{json.dumps(steps, ensure_ascii=False)}"
                ),
            ),
            tools=(),
            max_output_tokens=128,
        )
        payload = _load_json_object((response.content or "").strip())
        tools = _string_list(payload.get("tools"), limit=4)
        skills = _string_list(payload.get("skills"), limit=4)
        if tool_requirement:
            tools.insert(0, tool_requirement)
        tools = list(dict.fromkeys(value.casefold() for value in tools))
        skills = list(dict.fromkeys(value.casefold() for value in skills))
        if self.allowed_tools is not None:
            tools = [value for value in tools if value in self.allowed_tools]
        if self.allowed_skills is not None:
            skills = [value for value in skills if value in self.allowed_skills]
        keywords = _string_list(payload.get("keywords"), limit=8)
        scope = str(payload.get("scope") or "tool_triggered")
        if scope not in {"tool_triggered", "global"}:
            scope = "tool_triggered"
        if scope == "global" and (tools or skills or keywords):
            scope = "tool_triggered"
        return {
            "scope": scope,
            "tools": tools,
            "skills": skills,
            "keywords": keywords,
        }

class ChatHypothesisProvider:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def generate(self, query: str, *, style: str) -> str:
        focus = "可能发生过的具体事件" if style == "event" else "一般事实或稳定偏好"
        response = await self.provider.complete(
            messages=(
                ChatMessage.system(HYPOTHESIS_SYSTEM),
                ChatMessage.user(f"原问题：{query}\n改写重点：{focus}"),
            ),
            tools=(),
        )
        return (response.content or "").strip()


class ChatOptimizerModel:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]:
        memory_response = await self.provider.complete(
            messages=(
                ChatMessage.system(MEMORY_OPTIMIZER_SYSTEM),
                ChatMessage.user(
                    MEMORY_OPTIMIZER_USER.format(
                        today=datetime.now().strftime("%Y-%m-%d"),
                        memory=memory or "（空）",
                        pending=pending or "（无新内容）",
                    )
                ),
            ),
            tools=(),
        )
        self_response = await self.provider.complete(
            messages=(
                ChatMessage.system(SELF_OPTIMIZER_SYSTEM),
                ChatMessage.user(
                    SELF_OPTIMIZER_USER.format(
                        self_text=self_text or "（空）",
                        pending=pending or "（无新内容）",
                    )
                ),
            ),
            tools=(),
        )
        memory_output = (memory_response.content or "").strip()
        self_output = (self_response.content or "").strip()
        if not memory_output or not self_output:
            raise ValueError("记忆优化模型返回空响应")
        _validate_markdown_contract(
            memory_output,
            name="MEMORY.md",
            title="# 用户长期记忆",
            required_sections=(
                "## 用户事实",
                "## 用户偏好",
                "## 用户明确要求长期记住的关键内容",
            ),
            optional_sections=("## 助手操作上下文",),
        )
        _validate_markdown_contract(
            self_output,
            name="SELF.md",
            title="# MemoPilot 的自我认知",
            required_sections=(
                "## 人格与形象",
                "## 我对当前用户的理解",
                "## 我们关系的定义",
            ),
        )
        return memory_output, self_output


class OpenAIEmbeddingProvider:
    """兼容 OpenAI Embeddings API 的向量模型适配器。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        if not response.data:
            raise ValueError("Embedding Provider 返回空向量")
        return [float(value) for value in response.data[0].embedding]


def _load_json_object(text: str) -> dict[str, object]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        loaded: Any = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型响应不包含 JSON 对象") from None
        loaded = json.loads(value[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("模型响应必须是 JSON 对象")
    return {str(key): item for key, item in loaded.items()}


def _validate_consolidation_output(output: dict[str, object]) -> dict[str, object]:
    if "history_entries" in output:
        raw_entries = output.get("history_entries")
        if not isinstance(raw_entries, list):
            raise ValueError("Consolidation history_entries 必须是数组")
        history_entries: list[dict[str, object]] = []
        for index, raw in enumerate(raw_entries):
            if isinstance(raw, str):
                summary = raw.strip()
                weight = 0
            elif isinstance(raw, dict):
                summary = str(raw.get("summary") or "").strip()
                weight = raw.get("emotional_weight", 0)
            else:
                raise ValueError(f"Consolidation history_entries[{index}] 必须是对象")
            if not summary:
                raise ValueError(f"Consolidation history_entries[{index}].summary 不能为空")
            if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= 10:
                raise ValueError(
                    f"Consolidation history_entries[{index}].emotional_weight 无效"
                )
            history_entries.append({"summary": summary, "emotional_weight": weight})
        output["history_entries"] = history_entries
        pending_items = output.get("pending_items", [])
        if not isinstance(pending_items, list):
            raise ValueError("Consolidation pending_items 必须是数组")
        pending = "\n".join(str(item).strip() for item in pending_items if str(item).strip())
        _validate_pending_artifact(pending)
        output["pending_items"] = pending.splitlines()
    artifacts = output.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Consolidation artifacts 必须是对象")
    allowed_artifacts = {"HISTORY.md", "PENDING.md"}
    for name, content in artifacts.items():
        if str(name) not in allowed_artifacts or not isinstance(content, str):
            raise ValueError(f"Consolidation artifact 无效: {name}")
        if str(name) == "PENDING.md":
            _validate_pending_artifact(content)
    has_memories = "memories" in output
    memories = output.get("memories", [])
    if not isinstance(memories, list):
        raise ValueError("Consolidation memories 必须是数组")
    allowed_kinds = {"event", "profile", "preference", "procedure"}
    normalized: list[dict[str, object]] = []
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict):
            raise ValueError(f"Consolidation memories[{index}] 必须是对象")
        kind = memory.get("kind")
        memory_summary = memory.get("summary")
        if kind not in allowed_kinds:
            raise ValueError(f"Consolidation memories[{index}].kind 无效: {kind}")
        if not isinstance(memory_summary, str) or not memory_summary.strip():
            raise ValueError(f"Consolidation memories[{index}].summary 不能为空")
        weight = memory.get("emotional_weight", 0)
        if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= 10:
            raise ValueError(f"Consolidation memories[{index}].emotional_weight 无效")
        item = {str(key): value for key, value in memory.items()}
        item["emotional_weight"] = weight
        happened_at = item.get("happened_at")
        if happened_at is not None and not isinstance(happened_at, str):
            raise ValueError(f"Consolidation memories[{index}].happened_at 无效")
        if isinstance(happened_at, str):
            try:
                parsed_time = datetime.fromisoformat(happened_at.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    f"Consolidation memories[{index}].happened_at 不是有效 ISO-8601"
                ) from None
            if parsed_time.tzinfo is None:
                parsed_time = parsed_time.replace(tzinfo=UTC)
            item["happened_at"] = parsed_time.astimezone(UTC).isoformat()
        extra: dict[str, object] = {}
        if kind == "profile":
            category = str(item.pop("category", "personal_fact") or "personal_fact")
            if category not in {"personal_fact", "purchase", "decision", "status"}:
                raise ValueError(f"Consolidation memories[{index}].category 无效")
            extra["category"] = category
        if kind == "procedure":
            requirement = str(item.pop("tool_requirement", "") or "").strip()
            steps = _string_list(item.pop("steps", []), limit=12)
            schema = item.pop("rule_schema", {})
            if not requirement and not steps:
                item["kind"] = "preference"
            else:
                if not isinstance(schema, dict):
                    raise ValueError(f"Consolidation memories[{index}].rule_schema 无效")
                extra = {
                    "tool_requirement": requirement or None,
                    "steps": steps,
                    "rule_schema": {
                        field: _string_list(schema.get(field), limit=20)
                        for field in ("required_tools", "forbidden_tools", "mentioned_tools")
                    },
                }
        item["extra"] = extra
        normalized.append(item)
    if has_memories:
        output["memories"] = normalized
    return output


def _string_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _json_string_list(content: str | None) -> list[str]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _render_recent_context(
    values: dict[str, list[str]],
    *,
    compression_until: str,
    recent_turns: str,
) -> str:
    labels = (
        ("active_topics", "最近持续关注"),
        ("user_preferences", "最近明确偏好"),
        ("follow_ups", "最近待延续话题"),
        ("avoidances", "最近避免事项"),
    )
    compression = [f"until: {compression_until or 'none'}"]
    for key, label in labels:
        items = values.get(key, [])
        compression.append(f"- {label}：{'；'.join(items)}" if items else f"- {label}：none")
    ongoing = values.get("ongoing_threads", [])
    ongoing_lines = [f"- {item}" for item in ongoing] or ["- none"]
    return (
        "# Recent Context\n\n## Compression\n"
        + "\n".join(compression)
        + "\n\n## Ongoing Threads\n"
        + "\n".join(ongoing_lines)
        + "\n\n## Recent Turns\n<!-- a-preview = assistant reply preview only -->\n"
        + (recent_turns.strip() or "- none")
        + "\n"
    )


def _replace_recent_turns(old_context: str, recent_turns: str) -> str:
    marker = "\n## Recent Turns\n"
    block = (
        "## Recent Turns\n<!-- a-preview = assistant reply preview only -->\n"
        + (recent_turns.strip() or "- none")
        + "\n"
    )
    text = old_context.strip()
    if marker in text:
        return text.split(marker, 1)[0].rstrip() + "\n\n" + block
    if text:
        return text + "\n\n" + block
    return _render_recent_context({}, compression_until="", recent_turns=recent_turns)


def _validate_pending_artifact(content: str) -> None:
    allowed_tags = {
        "identity",
        "preference",
        "key_info",
        "health_long_term",
        "requested_memory",
        "correction",
        "agent_context",
    }
    for line_number, line in enumerate(content.splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        match = re.fullmatch(r"- \[([a-z_]+)]\s+(.+)", value)
        if match is None or match.group(1) not in allowed_tags:
            raise ValueError(f"PENDING.md 第 {line_number} 行标签或格式无效")


def _validate_markdown_contract(
    text: str,
    *,
    name: str,
    title: str,
    required_sections: tuple[str, ...],
    optional_sections: tuple[str, ...] = (),
) -> None:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    if not lines or lines[0] != title:
        raise ValueError(f"{name} 标题无效，必须是 {title}")
    headings = [line for line in lines if re.fullmatch(r"## .+", line)]
    allowed = (*required_sections, *optional_sections)
    if any(heading not in allowed for heading in headings):
        raise ValueError(f"{name} 包含不允许的 section")
    if tuple(headings[: len(required_sections)]) != required_sections:
        raise ValueError(f"{name} 缺少必需 section 或顺序错误")
    if len(headings) != len(set(headings)):
        raise ValueError(f"{name} section 不得重复")
    if headings[len(required_sections) :] not in ([], list(optional_sections)):
        raise ValueError(f"{name} 可选 section 顺序错误")
    for line in lines[1:]:
        if line and not line.startswith(("## ", "- ")):
            raise ValueError(f"{name} 只能包含标题、section 和 bullet")


__all__ = [
    "ChatConsolidationExtractor",
    "ChatImplicitMemoryExtractor",
    "ChatHypothesisProvider",
    "ChatOptimizerModel",
    "ChatPostResponseModel",
    "ChatRecentContextCompressor",
    "OpenAIEmbeddingProvider",
]
