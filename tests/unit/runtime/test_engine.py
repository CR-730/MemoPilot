from __future__ import annotations

from collections.abc import Sequence

from memopilot.extensions.events import EventBus
from memopilot.extensions.hooks import HookContext, HookOutcome, ToolHook
from memopilot.extensions.prompts import (
    PromptBlock,
    PromptRenderContext,
    PromptSectionRender,
)
from memopilot.extensions.skills import SkillCatalog, SkillDefinition
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
from memopilot.runtime.tool_search import ToolSearchTool
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
        LifecyclePhase.BEFORE_TURN,
        LifecyclePhase.BEFORE_TURN,
        LifecyclePhase.BEFORE_REASONING,
        LifecyclePhase.BEFORE_REASONING,
        LifecyclePhase.BEFORE_REASONING,
        LifecyclePhase.PROMPT_RENDER,
        LifecyclePhase.PROMPT_RENDER,
        LifecyclePhase.PROMPT_RENDER,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.AFTER_REASONING,
        LifecyclePhase.AFTER_REASONING,
        LifecyclePhase.AFTER_REASONING,
        LifecyclePhase.AFTER_TURN,
        LifecyclePhase.AFTER_TURN,
    ]
    assert [entry.iteration for entry in result.phase_trace[9:19]] == [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    ]
    assert [event.step_type for event in result.trace if event.step_type == "tool"] == [
        "tool"
    ]


async def test_runtime_keeps_successful_deferred_tool_in_session_lru() -> None:
    visible_by_step: list[frozenset[str] | None] = []

    class CaptureVisibleTools:
        phase = LifecyclePhase.BEFORE_STEP
        slot = "capture.visible_tools"
        requires = ("before_step.emit", "step:ctx")
        produces = ()

        async def run(self, context):
            visible_by_step.append(context.slots["step:ctx"].visible_tool_names)
            return {}

    class DiscoveryProvider(ChatProvider):
        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    content=None,
                    tool_calls=(
                        FunctionCall(
                            id="search",
                            name="tool_search",
                            arguments={"query": "select:echo"},
                        ),
                    ),
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    content=None,
                    tool_calls=(
                        FunctionCall(
                            id="echo", name="echo", arguments={"text": "hi"}
                        ),
                    ),
                    finish_reason="tool_calls",
                ),
                ModelResponse(content="done", tool_calls=(), finish_reason="stop"),
                ModelResponse(content="again", tool_calls=(), finish_reason="stop"),
                ModelResponse(content="other", tool_calls=(), finish_reason="stop"),
            ]
            self.calls: list[tuple[tuple[ChatMessage, ...], tuple[ToolSchema, ...]]] = []

        async def complete(self, *, messages, tools):
            self.calls.append((tuple(messages), tuple(tools)))
            return self.responses.pop(0)

    registry = _tools()
    search = ToolSearchTool(registry)
    registry.register(
        Tool(
            "tool_search",
            "search tools",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            search.execute,
        ),
        always_on=True,
    )
    provider = DiscoveryProvider()
    runtime = AgentRuntime(
        provider,
        registry,
        tool_search_enabled=True,
        modules=(CaptureVisibleTools(),),
    )

    await runtime.run(TurnInput("feishu:user-a", "first"))
    await runtime.run(TurnInput("feishu:user-a", "second"))
    await runtime.run(TurnInput("feishu:user-b", "third"))

    def names(index: int) -> list[str]:
        return [item["function"]["name"] for item in provider.calls[index][1]]
    assert names(0) == ["tool_search"]
    assert names(1) == ["tool_search", "echo"]
    assert names(3) == ["tool_search", "echo"]
    assert names(4) == ["tool_search"]
    assert visible_by_step[:3] == [
        frozenset({"tool_search"}),
        frozenset({"tool_search", "echo"}),
        frozenset({"tool_search", "echo"}),
    ]
    first_system = provider.calls[0][0][0].content or ""
    assert "echo" in first_system
    assert "search tools" not in first_system


async def test_before_turn_abort_returns_auditable_result_without_provider_call() -> None:
    provider = _Provider([ModelResponse(content="must not run", tool_calls=())])
    bus = EventBus()

    async def abort(payload):
        payload["abort"] = True
        payload["abort_reply"] = "turn stopped"
        return payload

    bus.on("before_turn", abort)

    result = await AgentRuntime(provider, _tools(), event_bus=bus).run(
        TurnInput(session_key="fake:1", content="question")
    )

    assert len(provider.responses) == 1
    assert result.reply == "turn stopped"
    assert result.react.exit_reason == "before_turn_abort"
    assert result.react.iterations == 0
    assert result.react.tool_chain == ()
    assert result.messages[-1] == ChatMessage.assistant(content="turn stopped")
    assert {entry.phase for entry in result.phase_trace} == {LifecyclePhase.BEFORE_TURN}


async def test_before_reasoning_abort_returns_without_prompt_or_provider_call() -> None:
    provider = _Provider([ModelResponse(content="must not run", tool_calls=())])
    bus = EventBus()

    async def abort(payload):
        payload["abort"] = True
        payload["abort_reply"] = "reasoning stopped"
        return payload

    bus.on("before_reasoning", abort)

    result = await AgentRuntime(provider, _tools(), event_bus=bus).run(
        TurnInput(session_key="fake:1", content="question")
    )

    assert len(provider.responses) == 1
    assert result.reply == "reasoning stopped"
    assert result.react.exit_reason == "before_reasoning_abort"
    assert result.react.iterations == 0
    assert result.react.tool_chain == ()
    assert result.messages[-1] == ChatMessage.assistant(content="reasoning stopped")
    assert LifecyclePhase.PROMPT_RENDER not in {
        entry.phase for entry in result.phase_trace
    }


async def test_after_step_early_stop_keeps_tool_result_and_skips_next_provider_call() -> None:
    provider = _Provider(
        [
            ModelResponse(
                content="working",
                tool_calls=(FunctionCall("c1", "echo", {"text": "done"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="must not run", tool_calls=()),
        ]
    )

    class StopAfterTool:
        phase = LifecyclePhase.AFTER_STEP
        slot = "context_pressure.stop"
        requires = ("after_step.copy_input", "step:ctx")
        produces = ()

        async def run(self, context):
            step = context.slots["step:ctx"]
            step.early_stop = True
            step.early_stop_reason = "context_pressure"
            return {}

    result = await AgentRuntime(provider, _tools(), modules=(StopAfterTool(),)).run(
        TurnInput(session_key="fake:1", content="question")
    )

    assert len(provider.responses) == 1
    assert result.reply == "working"
    assert result.react.exit_reason == "context_pressure"
    assert [record.observation.result for record in result.react.tool_chain] == ["done"]
    assert any(message.role == "tool" for message in result.messages)


async def test_runtime_audit_keeps_denied_status_and_structured_hook_details() -> None:
    class _DenyHook(ToolHook):
        async def run(self, context: HookContext) -> HookOutcome:
            return HookOutcome(
                decision="deny",
                reason="策略拒绝",
                extra_message="已记录策略拒绝",
            )

    provider = _Provider(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    FunctionCall(id="c1", name="echo", arguments={"text": "hi"}),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="已解释拒绝", tool_calls=(), finish_reason="stop"),
        ]
    )
    tools = _tools()
    tools.register_hook(_DenyHook("policy", event="pre_tool_use"))

    result = await AgentRuntime(provider, tools).run(
        TurnInput(session_key="fake:1", content="echo hi")
    )

    audit = next(event for event in result.trace if event.step_type == "tool")
    assert audit.state == "denied"
    assert audit.observation["status"] == "denied"
    assert audit.observation["extra_messages"] == ("已记录策略拒绝",)
    assert audit.observation["hook_trace"] == [
        {
            "hook_name": "policy",
            "event": "pre_tool_use",
            "matched": True,
            "decision": "deny",
            "reason": "策略拒绝",
            "extra_message": "已记录策略拒绝",
        }
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
    assert step_phases == [
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.BEFORE_STEP,
        LifecyclePhase.AFTER_STEP,
        LifecyclePhase.AFTER_STEP,
    ]
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


async def test_runtime_renders_scoped_plugin_prompt_blocks() -> None:
    provider = _CapturingProvider()
    runtime = AgentRuntime(
        provider,
        ToolRegistry(),
        prompt_blocks=(
            PromptBlock("plugin.prompt", "插件提示", scopes=("passive",)),
            PromptBlock("background.only", "后台提示", scopes=("background",)),
        ),
    )

    await runtime.run(
        TurnInput(
            session_key="feishu:chat-1",
            content="你好",
            system_prompt="核心提示",
        )
    )

    assert provider.messages[0].content == "核心提示\n\n插件提示"


async def test_runtime_injects_catalog_always_and_explicitly_mentioned_skills() -> None:
    provider = _CapturingProvider()
    catalog = SkillCatalog(
        (
            SkillDefinition(
                name="review",
                description="审查",
                background_allowed=False,
                required_tools=(),
                content="先运行测试，再检查差异。",
                source="workspace",
                path=None,
            ),
            SkillDefinition(
                name="policy",
                description="固定规则",
                background_allowed=False,
                required_tools=(),
                content="每轮核验事实。",
                source="builtin",
                path=None,
                always=True,
            ),
        )
    )
    runtime = AgentRuntime(provider, ToolRegistry(), skills=catalog)

    await runtime.run(TurnInput(session_key="feishu:1", content="请用 $review"))

    prompt = provider.messages[0].content or ""
    assert "# Skills Catalog" in prompt
    assert "review | 审查 | workspace | available" in prompt
    assert "policy | 固定规则 | builtin | available" in prompt
    assert "# Skill: review" in prompt
    assert "先运行测试" in prompt
    assert "# Skill: policy" in prompt
    assert "每轮核验事实" in prompt
    assert "read_file" not in prompt


async def test_runtime_skill_selection_does_not_leak_between_users() -> None:
    catalog = SkillCatalog(
        (
            SkillDefinition(
                name="review",
                description="审查",
                background_allowed=False,
                required_tools=(),
                content="只在命中时加载的审查正文。",
                source="workspace",
                path=None,
            ),
        )
    )
    first_provider = _CapturingProvider()
    runtime = AgentRuntime(first_provider, ToolRegistry(), skills=catalog)

    await runtime.run(TurnInput(session_key="feishu:user-a", content="请使用 $review"))
    first_prompt = first_provider.messages[0].content or ""

    second_provider = _CapturingProvider()
    runtime._provider = second_provider
    await runtime.run(TurnInput(session_key="feishu:user-b", content="普通问题"))
    second_prompt = second_provider.messages[0].content or ""

    assert "# Skill: review" in first_prompt
    assert "# Skill: review" not in second_prompt
    assert "只在命中时加载的审查正文" not in second_prompt
    assert "review | 审查 | workspace | available" in second_prompt


async def test_runtime_prioritizes_complete_active_skill_over_long_catalog() -> None:
    active_body = "ACTIVE-BEGIN\n" + ("必须完整保留。" * 16) + "\nACTIVE-END"
    skills = [
        SkillDefinition(
            name="review",
            description="审查",
            background_allowed=False,
            required_tools=(),
            content=active_body,
            source="workspace",
            path=None,
        )
    ]
    skills.extend(
        SkillDefinition(
            name=f"catalog-{index}",
            description="很长的目录说明" * 8,
            background_allowed=False,
            required_tools=(),
            content="不应激活",
            source="builtin",
            path=None,
        )
        for index in range(20)
    )
    provider = _CapturingProvider()
    runtime = AgentRuntime(
        provider,
        ToolRegistry(),
        skills=SkillCatalog(tuple(skills)),
        prompt_max_chars=520,
    )

    await runtime.run(
        TurnInput(
            session_key="feishu:user-a",
            content="请使用 $review",
            system_prompt="CORE-PROMPT",
        )
    )

    prompt = provider.messages[0].content or ""
    assert len(prompt) <= 520
    assert "CORE-PROMPT" in prompt
    assert active_body in prompt
    assert "# Skills Catalog" in prompt
    assert "catalog-19" not in prompt


async def test_runtime_plain_skill_word_does_not_activate_but_dollar_name_does() -> None:
    catalog = SkillCatalog(
        (
            SkillDefinition(
                name="review",
                description="审查",
                background_allowed=False,
                required_tools=(),
                content="REVIEW-BODY",
                source="workspace",
                path=None,
            ),
        )
    )
    plain_provider = _CapturingProvider()
    runtime = AgentRuntime(plain_provider, ToolRegistry(), skills=catalog)

    await runtime.run(TurnInput("feishu:plain", "please review this"))
    assert "REVIEW-BODY" not in (plain_provider.messages[0].content or "")

    explicit_provider = _CapturingProvider()
    runtime._provider = explicit_provider
    await runtime.run(TurnInput("feishu:explicit", "please use $review"))
    assert "REVIEW-BODY" in (explicit_provider.messages[0].content or "")


async def test_runtime_omits_oversized_active_skill_atomically_with_diagnostic() -> None:
    catalog = SkillCatalog(
        (
            SkillDefinition(
                name="huge",
                description="超长技能",
                background_allowed=False,
                required_tools=(),
                content="BEGIN-SECRET\n" + ("x" * 1000) + "\nEND-SECRET",
                source="workspace",
                path=None,
            ),
        )
    )
    provider = _CapturingProvider()
    runtime = AgentRuntime(
        provider,
        ToolRegistry(),
        skills=catalog,
        prompt_max_chars=240,
    )

    await runtime.run(
        TurnInput("feishu:1", "请使用 $huge", system_prompt="CORE-PROMPT")
    )

    prompt = provider.messages[0].content or ""
    assert "CORE-PROMPT" in prompt
    assert "BEGIN-SECRET" not in prompt
    assert "END-SECRET" not in prompt
    assert "Skill 正文因超出 Prompt 预算未注入: huge" in prompt


async def test_runtime_runs_dynamic_prompt_module_every_turn_without_cross_user_cache() -> None:
    class _DynamicPromptModule:
        phase = LifecyclePhase.PROMPT_RENDER
        slot = "dynamic.prompt"
        requires = ("prompt_render.emit", "prompt:ctx")
        produces = ()

        async def run(self, context):
            prompt = context.slots["prompt:ctx"]
            assert isinstance(prompt, PromptRenderContext)
            prompt.system_sections_top.append(
                PromptSectionRender("identity", "固定身份", is_static=True)
            )
            prompt.system_sections_bottom.append(
                PromptSectionRender(
                    "turn_context",
                    f"本轮用户={prompt.session_key};问题={prompt.content}",
                    is_static=False,
                )
            )
            return {}

    first_provider = _CapturingProvider()
    runtime = AgentRuntime(
        first_provider,
        ToolRegistry(),
        modules=(_DynamicPromptModule(),),
    )
    await runtime.run(TurnInput("feishu:user-a", "甲的问题", system_prompt="核心"))
    first_prompt = first_provider.messages[0].content or ""

    second_provider = _CapturingProvider()
    runtime._provider = second_provider
    await runtime.run(TurnInput("feishu:user-b", "乙的问题", system_prompt="核心"))
    second_prompt = second_provider.messages[0].content or ""

    assert first_prompt == "固定身份\n\n核心\n\n本轮用户=feishu:user-a;问题=甲的问题"
    assert second_prompt == "固定身份\n\n核心\n\n本轮用户=feishu:user-b;问题=乙的问题"
    assert "user-a" not in second_prompt


async def test_lifecycle_gates_rewrite_real_provider_input_and_final_reply() -> None:
    provider = _CapturingProvider()
    bus = EventBus()

    async def rewrite_before_turn(payload):
        payload["content"] = "Gate 改写后的问题"
        return payload

    async def rewrite_after_reasoning(payload):
        payload["reply"] = "Gate 改写后的回复"
        return payload

    bus.on("before_turn", rewrite_before_turn)
    bus.on("after_reasoning", rewrite_after_reasoning)
    runtime = AgentRuntime(provider, ToolRegistry(), event_bus=bus)

    result = await runtime.run(TurnInput("feishu:1", "原始问题"))

    assert provider.messages[-1].content == "Gate 改写后的问题"
    assert result.reply == "Gate 改写后的回复"
