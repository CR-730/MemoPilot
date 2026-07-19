from __future__ import annotations

from collections.abc import Sequence

from memopilot.memory.contracts import MemoryQueryResult, MemoryRecord
from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.phases import LifecyclePhase
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import Tool, ToolRegistry


class _Provider(ChatProvider):
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        return self.responses.pop(0)


async def _echo(*, text: str) -> str:
    return text


def _tools() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=_echo,
            )
        ]
    )


async def test_runtime_traces_outer_phases_around_each_react_step() -> None:
    provider = _Provider(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    FunctionCall(id="c1", name="echo", arguments={"text": "hi"}),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="完成", tool_calls=(), finish_reason="stop"),
        ]
    )

    result = await AgentRuntime(provider, _tools()).run(
        TurnInput(session_key="fake:1", content="echo hi")
    )

    assert result.reply == "完成"
    assert [entry.phase for entry in result.phase_trace] == [
        LifecyclePhase.BEFORE_TURN,
        LifecyclePhase.BEFORE_REASONING,
        LifecyclePhase.PROMPT_RENDER,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.AFTER_REASONING,
        LifecyclePhase.AFTER_TURN,
    ]
    assert [entry.iteration for entry in result.phase_trace[3:7]] == [1, 1, 2, 2]
    assert [event.step_type for event in result.trace if event.step_type == "tool"] == [
        "tool"
    ]


async def test_runtime_prompt_render_preserves_system_history_and_user_order() -> None:
    provider = _CapturingProvider()
    runtime = AgentRuntime(provider, ToolRegistry())

    await runtime.run(
        TurnInput(
            session_key="fake:1",
            content="new",
            system_prompt="system",
            history=(ChatMessage.user("old"), ChatMessage.assistant(content="answer")),
        )
    )

    assert [message.role for message in provider.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert provider.messages[-1].content == "new"


async def test_runtime_records_symmetric_step_phases_when_provider_fails() -> None:
    result = await AgentRuntime(_FailingProvider(), ToolRegistry()).run(
        TurnInput(session_key="fake:1", content="hello")
    )

    step_phases = [
        entry.phase
        for entry in result.phase_trace
        if entry.phase in {LifecyclePhase.BEFORE_STEP, LifecyclePhase.AFTER_STEP}
    ]
    assert step_phases == [LifecyclePhase.BEFORE_STEP, LifecyclePhase.AFTER_STEP]
    model_events = [event for event in result.trace if event.step_type == "model"]
    assert model_events[0].state == "failed"


class _CapturingProvider(ChatProvider):
    def __init__(self) -> None:
        self.messages: tuple[ChatMessage, ...] = ()

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        self.messages = tuple(messages)
        return ModelResponse(content="ok", tool_calls=(), finish_reason="stop")


class _FailingProvider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        raise TimeoutError("provider timeout")


class _MemoryEngine:
    def __init__(self) -> None:
        self.requests = []

    async def query(self, request):
        self.requests.append(request)
        return MemoryQueryResult(
            text_block="## 用户偏好与流程\n- [p1] 先读文档",
            records=(MemoryRecord("p1", "procedure", "先读文档", 0.8, injected=True),),
            trace={"candidate_ids": ["p1"], "injected_ids": ["p1"]},
        )


class _MemoryProfile:
    def read(self, name: str) -> str:
        return {
            "MEMORY.md": "# 长期记忆\n- 用户是 AI 工程师",
            "SELF.md": "# MemoPilot\n- 保持务实",
            "RECENT_CONTEXT.md": "- 当前正在重构项目",
        }[name]


async def test_runtime_prerecall_uses_raw_context_query_and_injects_system_memory() -> None:
    provider = _CapturingProvider()
    memory = _MemoryEngine()
    runtime = AgentRuntime(
        provider,
        ToolRegistry(),
        memory_engine=memory,  # type: ignore[arg-type]
        memory_profile=_MemoryProfile(),
    )

    result = await runtime.run(TurnInput(session_key="feishu:chat-1", content="原始问题"))

    assert [(request.text, request.intent) for request in memory.requests] == [
        ("原始问题", "context")
    ]
    assert provider.messages[0].role == "system"
    assert "先读文档" in (provider.messages[0].content or "")
    assert "用户是 AI 工程师" in (provider.messages[0].content or "")
    assert "当前正在重构项目" in (provider.messages[0].content or "")
    assert any(
        entry.module_slot == "before_reasoning.memory_prerecall"
        for entry in result.phase_trace
    )
    recall_events = [event for event in result.trace if event.step_type == "memory_recall"]
    assert recall_events[0].observation == {
        "candidate_ids": ["p1"],
        "injected_ids": ["p1"],
    }


async def test_runtime_does_not_duplicate_recent_turns_from_markdown() -> None:
    class _RecentProfile:
        def read(self, name: str) -> str:
            return {
                "MEMORY.md": "",
                "SELF.md": "",
                "RECENT_CONTEXT.md": (
                    "# Recent Context\n\n## Compression\n- 当前关注阶段四\n\n"
                    "## Recent Turns\n[user] 已经位于短期消息窗口"
                ),
            }[name]

    provider = _CapturingProvider()
    await AgentRuntime(
        provider,
        ToolRegistry(),
        memory_profile=_RecentProfile(),
    ).run(TurnInput(session_key="feishu:chat-1", content="继续"))

    prompt = provider.messages[0].content or ""
    assert "当前关注阶段四" in prompt
    assert "已经位于短期消息窗口" not in prompt
    assert "## Recent Turns" not in prompt


async def test_runtime_strips_memory_citation_protocol_and_exposes_used_ids() -> None:
    provider = _Provider(
        [
            ModelResponse(
                content="我记得你偏好简洁。\n§cited:[p1,m2]§",
                tool_calls=(),
                finish_reason="stop",
            )
        ]
    )

    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        memory_engine=_MemoryEngine(),  # type: ignore[arg-type]
    ).run(
        TurnInput(session_key="feishu:chat-1", content="我喜欢什么风格？")
    )

    assert result.reply == "我记得你偏好简洁。"
    assert result.cited_memory_ids == ("p1",)
    assert "§cited" not in (result.messages[-1].content or "")


async def test_memory_budget_preserves_retrieval_and_recent_context_before_long_memory() -> None:
    class _LongProfile:
        def read(self, name: str) -> str:
            return {
                "MEMORY.md": "长期事实" * 1000,
                "SELF.md": "保持务实",
                "RECENT_CONTEXT.md": "当前正在处理阶段四验收",
            }[name]

    provider = _CapturingProvider()
    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        memory_engine=_MemoryEngine(),  # type: ignore[arg-type]
        memory_profile=_LongProfile(),
        memory_markdown_max_chars=500,
    ).run(TurnInput(session_key="feishu:chat-1", content="发送邮件"))

    prompt = provider.messages[0].content or ""
    assert "先读文档" in prompt
    assert "当前正在处理阶段四验收" in prompt
    recall = next(event for event in result.trace if event.step_type == "memory_recall")
    assert recall.observation["injected_ids"] == ["p1"]


async def test_candidate_not_injected_cannot_be_cited_or_reinforced() -> None:
    class _CandidateOnlyMemory:
        async def query(self, request):
            return MemoryQueryResult(
                text_block="",
                records=(MemoryRecord("p2", "preference", "未注入", 0.4),),
                trace={"candidate_ids": ["p2"], "injected_ids": []},
            )

    provider = _Provider(
        [ModelResponse(content="回答\n§cited:[p2]§", tool_calls=(), finish_reason="stop")]
    )
    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        memory_engine=_CandidateOnlyMemory(),  # type: ignore[arg-type]
    ).run(TurnInput(session_key="feishu:chat-1", content="问题"))

    assert result.reply == "回答"
    assert result.cited_memory_ids == ()
