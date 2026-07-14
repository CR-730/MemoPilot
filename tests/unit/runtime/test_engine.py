from __future__ import annotations

from collections.abc import Sequence

from memopilot.memory.contracts import MemoryQueryResult
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
        return MemoryQueryResult(text_block="## 用户偏好与流程\n- [p1] 先读文档")


class _MemoryProfile:
    def read(self, name: str) -> str:
        return {
            "MEMORY.md": "# 长期记忆\n- 用户是 AI 工程师",
            "SELF.md": "# MemoPilot\n- 保持务实",
            "CONTEXT.md": "- 当前正在重构项目",
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
